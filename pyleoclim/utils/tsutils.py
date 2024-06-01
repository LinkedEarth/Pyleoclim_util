#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to manipulate timeseries - useful for preprocessing prior to analysis
"""

__all__ = [
    'simple_stats',
    'bin',
    'interp',
    'gkernel',
    'standardize',
    'ts2segments',
    'annualize',
    'gaussianize',
    'detrend',
    'detect_outliers_DBSCAN',
    'detect_outliers_kmeans',
    'remove_outliers',
    'phaseran',
    'phaseran2'
]

import numpy as np
from numpy import pi
import pandas as pd
import warnings
import copy
from scipy import special
from scipy import signal
from scipy import interpolate
from scipy.interpolate import splrep, splev
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
#import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as sms

import math
from .filter import savitzky_golay

from .tsbase import (
    clean_ts,
    dropna,
)


from .emd_utils import (
    extr,
    boundary_conditions)

class EmpiricalModeDecomposition(object):
    """The EMD class. This class has been adapted from the pyhht package (https://github.com/jaidevd/pyhht/tree/dev) and adapated to work with scipy 0.12.0. See the copyright notice in the emd_utils module."""

    def __init__(self, x, t=None, threshold_1=0.05, threshold_2=0.5,
                 alpha=0.05, ndirs=4, fixe=0, maxiter=2000, fixe_h=0, n_imfs=0,
                 nbsym=2, bivariate_mode='bbox_center'):
        """Empirical mode decomposition.

        Parameters
        ----------
        x : array-like, shape (n_samples,)
            The signal on which to perform EMD
        t : array-like, shape (n_samples,), optional
            The timestamps of the signal.
        threshold_1 : float, optional
            Threshold for the stopping criterion, corresponding to
            :math:`\\theta_{1}` in [3]. Defaults to 0.05.
        threshold_2 : float, optional
            Threshold for the stopping criterion, corresponding to
            :math:`\\theta_{2}` in [3]. Defaults to 0.5.
        alpha : float, optional
            Tolerance for the stopping criterion, corresponding to
            :math:`\\alpha` in [3]. Defaults to 0.05.
        ndirs : int, optional
            Number of directions in which interpolants for envelopes are
            computed for bivariate EMD. Defaults to 4. This is ignored if the
            signal is real valued.
        fixe : int, optional
            Number of sifting iterations to perform for each IMF. By default,
            the stopping criterion mentioned in [1] is used. If set to a
            positive integer, each mode is either the result of exactly
            `fixe` number of sifting iterations, or until a pure IMF is
            found, whichever is sooner.
        maxiter : int, optional
            Upper limit of the number of sifting iterations for each mode.
            Defaults to 2000.
        n_imfs : int, optional
            Number of IMFs to extract. By default, this is ignored and
            decomposition is continued until a monotonic trend is left in the
            residue.
        nbsym : int, optional
            Number of extrema to use to mirror the signals on each side of
            their boundaries.
        bivariate_mode : str, optional
            The algorithm to be used for bivariate EMD as described in [4].
            Can be one of 'centroid' or 'bbox_center'. This is ignored if the
            signal is real valued.

        Attributes
        ----------
        is_bivariate : bool
            Whether the decomposer performs bivariate EMD. This is
            automatically determined by the input value. This is True if at
            least one non-zero imaginary component is found in the signal.
        nbits : list
            List of number of sifting iterations it took to extract each IMF.

        References
        ----------

        .. [1] Huang H. et al. 1998 'The empirical mode decomposition and the \
                Hilbert spectrum for nonlinear and non-stationary time series \
                analysis.' \
                Procedings of the Royal Society 454, 903-995

        .. [2] Zhao J., Huang D. 2001 'Mirror extending and circular spline \
                function for empirical mode decomposition method'. \
                Journal of Zhejiang University (Science) V.2, No.3, 247-252

        .. [3] Gabriel Rilling, Patrick Flandrin, Paulo Gonçalves, June 2003: \
                'On Empirical Mode Decomposition and its Algorithms',\
                IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing \
                NSIP-03

        .. [4] Gabriel Rilling, Patrick Flandrin, Paulo Gonçalves, \
                Jonathan M. Lilly. Bivariate Empirical Mode Decomposition. \
                10 pages, 3 figures. Submitted to Signal Processing Letters, \
                IEEE. Matlab/C codes and additional .. 2007. <ensl-00137611>
        
        .. [5] https://github.com/jaidevd/pyhht/tree/dev


        """

        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.alpha = alpha
        self.maxiter = maxiter
        self.fixe_h = fixe_h
        self.ndirs = ndirs
        self.nbit = 0
        self.Nbit = 0
        self.n_imfs = n_imfs
        self.k = 1
        # self.mask = mask
        self.nbsym = nbsym
        self.nbit = 0
        self.NbIt = 0

        if x.ndim > 1:
            if 1 not in x.shape:
                raise ValueError("x must have only one row or one column.")
        if x.shape[0] > 1:
            x = x.ravel()
        if np.any(np.isinf(x)):
            raise ValueError("All elements of x must be finite.")
        self.x = x
        self.ner = self.nzr = len(self.x)
        self.residue = self.x.copy()

        if t is None:
            self.t = np.arange(max(x.shape))
        else:
            if t.shape != self.x.shape:
                raise ValueError("t must have the same dimensions as x.")
            if t.ndim > 1:
                if 1 not in t.shape:
                    raise ValueError("t must have only one column or one row.")
            if not np.all(np.isreal(t)):
                raise TypeError("t must be a real vector.")
            if t.shape[0] > 1:
                t = t.ravel()
            self.t = t

        if fixe:
            self.maxiter = fixe
            if self.fixe_h:
                raise TypeError("Cannot use both fixe and fixe_h modes")
        self.fixe = fixe

        self.is_bivariate = np.any(np.iscomplex(self.x))
        if self.is_bivariate:
            self.bivariate_mode = bivariate_mode

        self.imf = []
        self.nbits = []

    def io(self):
        r"""Compute the index of orthoginality, as defined by:

        .. math::

            \sum_{i,j=1,i\neq j}^{N}\frac{\|C_{i}\overline{C_{j}}\|}{\|x\|^2}

        Where :math:`C_{i}` is the :math:`i` th IMF.

        Returns
        -------
        float
            Index of orthogonality. Lower values are better.

        """
        imf = np.array(self.imf)
        dp = np.dot(imf, np.conj(imf).T)
        mask = np.logical_not(np.eye(len(self.imf)))
        s = np.abs(dp[mask]).sum()
        return s / (2 * np.sum(self.x ** 2))

    def stop_EMD(self):
        """Check if there are enough extrema (3) to continue sifting.

        Returns
        -------
        bool
            Whether to stop further cubic spline interpolation for lack of
            local extrema.

        """
        if self.is_bivariate:
            stop = False
            for k in range(self.ndirs):
                phi = k * pi / self.ndirs
                indmin, indmax, _ = extr(
                    np.real(np.exp(1j * phi) * self.residue))
                if len(indmin) + len(indmax) < 3:
                    stop = True
                    break
        else:
            indmin, indmax, _ = extr(self.residue)
            ner = len(indmin) + len(indmax)
            stop = ner < 3
        return stop

    def mean_and_amplitude(self, m):
        """ Compute the mean of the envelopes and the mode amplitudes.

        Parameters
        ----------
        m : array-like, shape (n_samples,)
            The input array or an itermediate value of the sifting process.

        Returns
        -------
        tuple
            A tuple containing the mean of the envelopes, the number of
            extrema, the number of zero crosssing and the estimate of the
            amplitude of themode.
        """
        
        if self.is_bivariate:
            if self.bivariate_mode == 'centroid':
                nem = []
                nzm = []
                envmin = np.zeros((self.ndirs, len(self.t)))
                envmax = np.zeros((self.ndirs, len(self.t)))
                for k in range(self.ndirs):
                    phi = k * pi / self.ndirs
                    y = np.real(np.exp(-1j * phi) * m)
                    indmin, indmax, indzer = extr(y)
                    nem.append(len(indmin) + len(indmax))
                    nzm.append(len(indzer))
                    if self.nbsym:
                        tmin, tmax, zmin, zmax = boundary_conditions(
                            y, self.t, m, self.nbsym)
                    else:
                        tmin = np.r_[self.t[0], self.t[indmin], self.t[-1]]
                        tmax = np.r_[self.t[0], self.t[indmax], self.t[-1]]
                        zmin, zmax = m[tmin], m[tmax]

                    f = splrep(tmin, zmin)
                    spl = splev(self.t, f)
                    envmin[k, :] = spl

                    f = splrep(tmax, zmax)
                    spl = splev(self.t, f)
                    envmax[k, :] = spl

                envmoy = np.mean((envmin + envmax) / 2, axis=0)
                amp = np.mean(abs(envmax - envmin), axis=0) / 2

            elif self.bivariate_mode == 'bbox_center':
                nem = []
                nzm = []
                envmin = np.zeros((self.ndirs, len(self.t)), dtype=complex)
                envmax = np.zeros((self.ndirs, len(self.t)), dtype=complex)
                for k in range(self.ndirs):
                    phi = k * pi / self.ndirs
                    y = np.real(np.exp(-1j * phi) * m)
                    indmin, indmax, indzer = extr(y)
                    nem.append(len(indmin) + len(indmax))
                    nzm.append(len(indzer))
                    if self.nbsym:
                        tmin, tmax, zmin, zmax = boundary_conditions(
                            y, self.t, m, self.nbsym)
                    else:
                        tmin = np.r_[self.t[0], self.t[indmin], self.t[-1]]
                        tmax = np.r_[self.t[0], self.t[indmax], self.t[-1]]
                        zmin, zmax = m[tmin], m[tmax]
                    f = splrep(tmin, zmin)
                    spl = splev(self.t, f)
                    envmin[k, ] = np.exp(1j * phi) * spl

                    f = splrep(tmax, zmax)
                    spl = splev(self.t, f)
                    envmax[k, ] = np.exp(1j * phi) * spl

                envmoy = np.mean((envmin + envmax), axis=0)
                amp = np.mean(abs(envmax - envmin), axis=0) / 2

        else:
            indmin, indmax, indzer = extr(m)
            nem = len(indmin) + len(indmax)
            nzm = len(indzer)
            if self.nbsym:
                tmin, tmax, mmin, mmax = boundary_conditions(m, self.t, m,
                                                             self.nbsym)
            else:
                tmin = np.r_[self.t[0], self.t[indmin], self.t[-1]]
                tmax = np.r_[self.t[0], self.t[indmax], self.t[-1]]
                mmin, mmax = m[tmin], m[tmax]

            f = splrep(tmin, mmin)
            envmin = splev(self.t, f)

            f = splrep(tmax, mmax)
            envmax = splev(self.t, f)

            envmoy = (envmin + envmax) / 2
            amp = np.abs(envmax - envmin) / 2.0
        if self.is_bivariate:
            nem = np.array(nem)
            nzm = np.array(nzm)

        return envmoy, nem, nzm, amp

    def stop_sifting(self, m):
        """Evaluate the stopping criteria for the current mode.

        Parameters
        ----------
        m : array-like, shape (n_samples,)
            The current mode.

        Returns
        -------
        bool
            Whether to stop sifting. If this evaluates to true, the current
            mode is interpreted as an IMF.

        """
        # FIXME: This method needs a better name.
        if self.fixe:
            (moyenne, _, _, _), stop_sift = self.mean_and_amplitude(m), 0  # NOQA
        elif self.fixe_h:
            stop_count = 0
            try:
                moyenne, nem, nzm = self.mean_and_amplitude(m)[:3]

                if np.all(abs(nzm - nem) > 1):
                    stop = 0
                    stop_count = 0
                else:
                    stop_count += 1
                    stop = (stop_count == self.fixe_h)
            except:
                moyenne = np.zeros((len(m)))
                stop = 1
            stop_sift = stop
        else:
            try:
                envmoy, nem, nzm, amp = self.mean_and_amplitude(m)
            except TypeError as err:
                if err.args[0] == "m > k must hold":
                    return 1, np.zeros((len(m)))
            except ValueError as err:
                if err.args[0] == "Not enough extrema.":
                    return 1, np.zeros((len(m)))
            sx = np.abs(envmoy) / amp
            stop = not(((np.mean(sx > self.threshold_1) > self.alpha) or
                        np.any(sx > self.threshold_2)) and np.all(nem > 2))
            if not self.is_bivariate:
                stop = stop and not(np.abs(nzm - nem) > 1)
            stop_sift = stop
            moyenne = envmoy
        return stop_sift, moyenne

    def keep_decomposing(self):
        """Check whether to continue the sifting operation."""
        return not(self.stop_EMD()) and \
            (self.k < self.n_imfs + 1 or self.n_imfs == 0)  # and \
# not(np.any(self.mask))

    def decompose(self):
        """Decompose the input signal into IMFs.

        This function does all the heavy lifting required for sifting, and
        should ideally be the only public method of this class.

        Returns
        -------
        imfs : array-like, shape (n_imfs, n_samples)
            A matrix containing one IMF per row.

        Examples
        --------

        >>> from pyhht.visualization import plot_imfs
        >>> import numpy as np
        >>> t = np.linspace(0, 1, 1000)
        >>> modes = np.sin(2 * pi * 5 * t) + np.sin(2 * pi * 10 * t)
        >>> x = modes + t
        >>> decomposer = EMD(x)
        >>> imfs = decomposer.decompose()
        """
        while self.keep_decomposing():

            # current mode
            m = self.residue

            # computing mean and stopping criterion
            stop_sift, moyenne = self.stop_sifting(m)

            # in case current mode is small enough to cause spurious extrema
            if np.max(np.abs(m)) < (1e-10) * np.max(np.abs(self.x)):
                if not stop_sift:
                    warnings.warn(
                        "EMD Warning: Amplitude too small, stopping.")
                else:
                    print("Force stopping EMD: amplitude too small.")
                return

            # SIFTING LOOP:
            while not(stop_sift) and (self.nbit < self.maxiter):
                # The following should be controlled by a verbosity parameter.
                # if (not(self.is_bivariate) and
                #     (self.nbit > self.maxiter / 5) and
                #     self.nbit % np.floor(self.maxiter / 10) == 0 and
                #     not(self.fixe) and self.nbit > 100):
                #     print("Mode " + str(self.k) +
                #           ", Iteration " + str(self.nbit))
                #     im, iM, _ = extr(m)
                #     print(str(np.sum(m[im] > 0)) + " minima > 0; " +
                #           str(np.sum(m[im] < 0)) + " maxima < 0.")

                # Sifting
                m = m - moyenne

                # Computing mean and stopping criterion
                stop_sift, moyenne = self.stop_sifting(m)

                self.nbit += 1
                self.NbIt += 1

                # This following warning depends on verbosity and needs better
                # handling
                # if not self.fixe and self.nbit > 100(self.nbit ==
                # (self.maxiter - 1)) and not(self.fixe) and (self.nbit > 100):
                #     warnings.warn("Emd:warning, Forced stop of sifting - " +
                #                   "Maximum iteration limit reached.")

            self.imf.append(m)

            self.nbits.append(self.nbit)
            self.nbit = 0
            self.k += 1

            self.residue = self.residue - m
            self.ort = self.io()

        if np.any(self.residue):
            self.imf.append(self.residue)
        return np.array(self.imf)



def simple_stats(y, axis=None):
    """ Computes simple statistics

    Computes the mean, median, min, max, standard deviation, and interquartile range of a numpy array y, ignoring NaNs.

    Parameters
    ----------
    y: array
        A Numpy array
    axis : int, tuple of ints
        Axis or Axes along which the means
        are computed, the default is to compute the mean of the flattened
        array. If a tuple of ints, performed over multiple axes

    Returns
    -------
    mean :  float
        mean of y, ignoring NaNs
    median : float
        median of y, ignoring NaNs
    min_ : float
        mininum value in y, ignoring NaNs
    max_ : float
        max value in y, ignoring NaNs
    std : float
        standard deviation of y, ignoring NaNs
    IQR : float
        Interquartile range of y along specified axis, ignoring NaNs
        
    """
    # make sure that y is an array
    y = np.array(y, dtype='float64')

    # Perform the various calculations
    mean = np.nanmean(y, axis=axis)
    std = np.nanstd(y, axis=axis)
    median = np.nanmedian(y, axis=axis)
    min_ = np.nanmin(y, axis=axis)
    max_ = np.nanmax(y, axis=axis)
    IQR = np.nanpercentile(y, 75, axis=axis) - np.nanpercentile(y, 25, axis=axis)

    return mean, median, min_, max_, std, IQR


def bin(x, y, bin_size=None, start=None, stop=None, step_style=None, evenly_spaced = False, statistic = 'mean', bin_edges=None, time_axis=None,no_nans = True):
    """ Bin the values

    The behavior of bins, as defined either by start, stop and step or by the bins argument, is to have all bins
    except the last one be half open. That is if bins are defined as bins = [1,2,3,4], bins will be [1,2), [2,3), [3,4].
    This is the default behaviour of scipy.stats.binned_statistic (upon which this function is built).

    Parameters
    ----------
    x : array
        The x-axis series.

    y : array
        The y-axis series.

    bin_size : float
        The size of the bins. Default is the maximum resolution if no_nans is True.

    start : float
        Where/when to start binning. Default is the minimum.

    stop : float
        When/where to stop binning. Default is the maximum.

    step_style : str; {'min','mean','median','max'}
        Step style to use when determining the size of the interval between points. Default is None.

    evenly_spaced : {True,False}
        Makes the series evenly-spaced. This option is ignored if bin_size is set to float.
        This option is being deprecated, no_nans should be used instead.

    statistic : str
        Statistic to calculate and return in values. Default is 'mean'.
        See scipy.stats.binned_statistic for other options.

    bin_edges : np.ndarray
        The edge of bins to use for binning. 
        E.g. if bins = [1,2,3,4], bins will be [1,2), [2,3), [3,4].
        See scipy.stats.binned_statistic for details.
        Start, stop, bin_size, step_style, and time_axis will be ignored if this is passed.
    
    time_axis : np.ndarray
        The time axis to use for binning. If passed, bin_edges will be set as the midpoints between times.
        The first time will be used as the left most edge, the last time will be used as the right most edge.
        Start, stop, bin_size, and step_style will be ignored if this is passed.

    no_nans : bool; {True,False}
        Sets the step_style to max, ensuring that the resulting series contains no empty values (nans).
        Default is True.

    Returns
    -------
    binned_values : array
        The binned values
    bins : array
        The bins (centered on the median, i.e., the 100-200 bin is 150)
    n : array
        Number of data points in each bin
    error : array
        The standard error on the mean in each bin

    Notes
    -----

    `start`, `stop`, `bin_size`, and `step_style` are interpreted as defining the `bin_edges` for this function.
    This differs from the `interp` interpretation, which uses these to define the time axis over which interpolation is applied.
    For `bin`, the time axis will be specified as the midpoints between `bin_edges`, unless `time_axis` is explicitly passed.

    See also
    --------

    pyleoclim.utils.tsutils.gkernel : Coarsen time resolution using a Gaussian kernel

    pyleoclim.utils.tsutils.interp : Interpolate y onto a new x-axis

    `scipy.stats.binned_statistic <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html>`_ : Scipy function around which this function is written

    Examples
    --------

    There are several ways to specify the way binning is conducted via this function. Within these there is a hierarchy which we demonstrate below.

    Top priority is given to `bin_edges` if it is not None. All other arguments will be ignored (except for x and y).
    The resulting time axis will be comprised of the midpoints between bin edges.

    .. jupyter-execute::

        import numpy as np
        import pyleoclim as pyleo

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xb,yb = pyleo.utils.tsutils.bin(x,y,bin_edges=[1,4,8,12,16,20])
        xb

    Next, priority will go to `time_axis` if it is passed. In this case, bin edges will be taken as the midpoints between time axis points.
    The first and last time point will be used as the left most and right most bin edges.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xb,yb = pyleo.utils.tsutils.bin(x,y,time_axis=[1,4,8,12,16,20])
        xb
    
    If `time_axis` is None, `bin_size` will be considered, overriding `step_style if it is passed. `start` and `stop` will be generated using defaults if not passed.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xb,yb = pyleo.utils.tsutils.bin(x,y,bin_size=2)
        xb
    
    If both `time_axis` and `step` are None but `step_style` is specified, the step will be generated using the prescribed `step_style`.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xb,yb = pyleo.utils.tsutils.bin(x,y,step_style='max')
        xb

    If none of these are specified, the mean spacing will be used.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xb,yb = pyleo.utils.tsutils.bin(x,y)
        xb

    """

    if evenly_spaced:
        no_nans=True
        warnings.warn('`evenly_spaced` is being deprecated. Please switch to using the option `no_nans` (behaviour is identical).',DeprecationWarning,stacklevel=2)

    # Make sure x and y are numpy arrays
    x = np.array(x, dtype='float64')
    y = np.array(y, dtype='float64')
    
    # Set the bin edges
    if bin_edges is not None:
        if start is not None or stop is not None or bin_size is not None or step_style is not None or time_axis is not None:
            warnings.warn('Bins have been passed with other bin relevant arguments {start,stop,bin_size,step_style,time_axis}. Bin_edges take priority and will be used.',stacklevel=2)
        time_axis = (bin_edges[1:] + bin_edges[:-1])/2
    # A bit of wonk is required to get the proper bin edges from the time axis
    elif time_axis is not None:
        if start is not None or stop is not None or bin_size is not None or step_style is not None:
            warnings.warn('The time axis has been passed with other time axis relevant arguments {start,stop,bin_size,step_style}. Time_axis takes priority and will be used.',stacklevel=2)
        bin_edges = np.zeros(len(time_axis)+1)
        bin_edges[0] = time_axis[0]
        bin_edges[-1] = time_axis[-1]
        bin_edges[1:-1] = (time_axis[1:]+time_axis[:-1])/2
    else:
        bin_edges = make_even_axis(x=x,start=start,stop=stop,step=bin_size,step_style=step_style,no_nans=no_nans)
        time_axis = (bin_edges[1:]+bin_edges[:-1])/2

    # Perform the calculation
    binned_values = stats.binned_statistic(x=x,values=y,bins=bin_edges,statistic=statistic).statistic
    n = stats.binned_statistic(x=x,values=y,bins=bin_edges,statistic='count').statistic
    error = stats.binned_statistic(x=x,values=y,bins=bin_edges,statistic='std').statistic

    #Returned bins should be at the midpoint of the bin edges
    res_dict = {
        'bins': time_axis,
        'binned_values': binned_values,
        'n': n,
        'error': error,
    }

    if no_nans:
        check = np.isnan(binned_values).any()
        if check:
            warnings.warn('no_nans is set to True but nans are present in the series. It has likely been overridden by other parameters. See tsutils.bin() documentation for details on parameter hierarchy',stacklevel=2)

    return  res_dict


def gkernel(t,y, h = None, step=None,start=None,stop=None, step_style = None, evenly_spaced=False, bin_edges=None, time_axis=None,no_nans=True):
    '''Coarsen time resolution using a Gaussian kernel

    The behavior of bins, as defined either by start, stop and step (or step_style) or by the bins argument, is to have all bins
    except the last one be half open. That is if bins are defined as bins = [1,2,3,4], bins will be [1,2), [2,3), [3,4].
    This is the default behaviour of our binning functionality (upon which this function is based).

    Parameters
    ----------
    t  : 1d array
        the original time axis
    
    y  : 1d array
        values on the original time axis
        
    h  : float 
        kernel e-folding scale. Default value is None, in which case the median time step will be used.
        If the median time step results in a series with nan values, the maximum time step will be used.
        Note that if this variable is too small, this method may return nan values in parts of the series.
    
    step : float
        The interpolation step. Default is max spacing between consecutive points.

    start : float
        where/when to start the interpolation. Default is min(t).
        
    stop : float
        where/when to stop the interpolation. Default is max(t).
   
    step_style : str
            step style to be applied from 'increments' [default = 'max']

    evenly_spaced : {True,False}
        Makes the series evenly-spaced. This option is ignored if bins are passed.
        This option is being deprecated, no_nans should be used instead. 

    bin_edges : array
        The right hand edge of bins to use for binning.
        E.g. if bins = [1,2,3,4], bins will be [1,2), [2,3), [3,4].
        Same behavior as scipy.stats.binned_statistic
        Start, stop, step, and step_style will be ignored if this is passed.

    time_axis : np.ndarray
        The time axis to use for binning. If passed, bin_edges will be set as the midpoints between times.
        The first time will be used as the left most edge, the last time will be used as the right most edge.
        Start, stop, bin_size, and step_style will be ignored if this is passed.

    no_nans : bool; {True,False}
        Sets the step_style to max, ensuring that the resulting series contains no empty values (nans).
        Default is True.

    Returns
    -------
    tc : 1d array
        the coarse-grained time axis
        
    yc:  1d array
        The coarse-grained time series

    Notes
    -----

    `start`, `stop`, `step`, and `step_style` are interpreted as defining the `bin_edges` for this function.
    This differs from the `interp` interpretation, which uses these to define the time axis over which interpolation is applied.
    For `gkernel`, the time axis will be specified as the midpoints between `bin_edges`, unless `time_axis` is explicitly passed.

    References
    ----------

    Rehfeld, K., Marwan, N., Heitzig, J., and Kurths, J.: Comparison of correlation analysis
    techniques for irregularly sampled time series, Nonlin. Processes Geophys.,
    18, 389–404, doi:10.5194/npg-18-389-2011, 2011.

    See also
    --------

    pyleoclim.utils.tsutils.increments : Establishes the increments of a numerical array
    
    pyleoclim.utils.tsutils.make_even_axis : Create an even time axis

    pyleoclim.utils.tsutils.bin : Bin the values

    pyleoclim.utils.tsutils.interp : Interpolate y onto a new x-axis

    Examples
    --------

    There are several ways to specify the way coarsening is done via this function. Within these there is a hierarchy which we demonstrate below.

    Top priority is given to `bin_edges` if it is not None. All other arguments will be ignored (except for x and y).
    The resulting time axis will be comprised of the midpoints between bin edges.

    .. jupyter-execute::

        import numpy as np
        import pyleoclim as pyleo

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(x))
        xc,yc = pyleo.utils.tsutils.gkernel(x,y,bin_edges=[1,4,8,12,16,20])
        xc

    Next, priority will go to `time_axis` if it is passed. In this case, bin edges will be taken as the midpoints between time axis points.
    The first and last time point will be used as the left most and right most bin edges.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(x))
        xc,yc = pyleo.utils.tsutils.gkernel(x,y,time_axis=[1,4,8,12,16,20])
        xc
    
    If `time_axis` is None, `step` will be considered, overriding `step_style` if it is passed. `start` and `stop` will be generated using defaults if not passed.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(x))
        xc,yc = pyleo.utils.tsutils.gkernel(x,y,step=2)
        xc
    
    If both `time_axis` and `step` are None but `step_style` is specified, the step will be generated using the prescribed `step_style`.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(x))
        xc,yc = pyleo.utils.tsutils.gkernel(x,y,step_style='max')
        xc

    If none of these are specified, the mean spacing will be used.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(x))
        xc,yc = pyleo.utils.tsutils.gkernel(x,y)
        xc

    '''

    if len(t) != len(y):
        raise ValueError('y and t must have the same length')
    
    if evenly_spaced:
        no_nans=True
        warnings.warn('`evenly_spaced` is being deprecated. Please switch to using the option `no_nans` (behaviour is identical).',DeprecationWarning,stacklevel=2)

        # Make sure x and y are numpy arrays
    t = np.array(t, dtype='float64')
    y = np.array(y, dtype='float64')
    
    # Set the bin edges
    if bin_edges is not None:
        bin_edges = np.array(bin_edges)
        if start is not None or stop is not None or step is not None or step_style is not None or time_axis is not None:
            warnings.warn('Bins have been passed with other axis relevant arguments {start,stop,step,step_style,time_axis}. Bin_edges take priority and will be used.',stacklevel=2)
        time_axis = (bin_edges[1:] + bin_edges[:-1])/2
    # A bit of wonk is required to get the proper bin edges from the time axis
    elif time_axis is not None:
        time_axis = np.array(time_axis)
        if start is not None or stop is not None or step is not None or step_style is not None:
            warnings.warn('The time axis has been passed with other axis relevant arguments {start,stop,step,step_style}. Time_axis takes priority and will be used.',stacklevel=2)
        bin_edges = np.zeros(len(time_axis)+1)
        bin_edges[0] = time_axis[0]
        bin_edges[-1] = time_axis[-1]
        bin_edges[1:-1] = (time_axis[1:]+time_axis[:-1])/2
    else:
        bin_edges = make_even_axis(x=t,start=start,stop=stop,step=step,step_style=step_style,no_nans=no_nans)
        time_axis = (bin_edges[1:]+bin_edges[:-1])/2

    kernel = lambda x, s : 1.0/(s*np.sqrt(2*np.pi))*np.exp(-0.5*(x/s)**2)  # define kernel function

    if h is None:
        h = np.median(t)

    yc    = np.zeros((len(time_axis)))
    yc[:] = np.nan

    for i in range(len(bin_edges)-1):
        if i < len(bin_edges)-1:
            xslice = t[(t>=bin_edges[i])&(t<bin_edges[i+1])]
            yslice = y[(t>=bin_edges[i])&(t<bin_edges[i+1])]
        else:
            xslice = t[(t>=bin_edges[i])&(t<=bin_edges[i+1])]
            yslice = y[(t>=bin_edges[i])&(t<=bin_edges[i+1])]

        if len(xslice)>0:
            d      = xslice-time_axis[i]
            weight = kernel(d,h)
            yc[i]  = sum(weight*yslice)/sum(weight) # normalize by the sum of weights
        else:
            yc[i] = np.nan

    if no_nans:
        check = np.isnan(yc).any()
        if check:
            warnings.warn('no_nans is set to True but nans are present in the series. It may have been overridden by other parameters. See tsutils.gkernel() documentation for details on parameter hierarchy, and check that your h parameter is large enough.',stacklevel=2)

    return time_axis, yc


def increments(x,step_style='median'):
    '''Establishes the increments of a numerical array: start, stop, and representative step.

    Parameters
    ----------
    x : array

    step_style : str
        Method to obtain a representative step if x is not evenly spaced.
        Valid entries: 'median' [default], 'mean', 'mode' or 'max'
        The mode is the most frequent entry in a dataset, and may be a good choice if the timeseries
        is nearly equally spaced but for a few gaps. 
        
        Max is a conservative choice, appropriate for binning methods and Gaussian kernel coarse-graining

    Returns
    -------
    start : float
        min(x)
    stop : float
        max(x)
    step : float
        The representative spacing between consecutive values, computed as above

    See also
    --------

    pyleoclim.utils.tsutils.bin : Bin the values

    pyleoclim.utils.tsutils.gkernel : Coarsen time resolution using a Gaussian kernel

    '''

    start = np.nanmin(x)
    stop = np.nanmax(x)

    delta = np.diff(x)
    if step_style == 'mean':
        step = delta.mean()
    elif step_style == 'max':
        step = delta.max()
    elif step_style == 'mode':
        step = stats.mode(delta)[0][0]
    else:
        step = np.median(delta)

    return start, stop, step


def interp(x,y, interp_type='linear', step=None, start=None, stop=None, step_style=None, time_axis=None,**kwargs):
    """ Interpolate y onto a new x-axis

    Largely a wrapper for `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.

    Parameters
    ----------
    x : array
       The x-axis

    y : array
       The y-axis

    interp_type : str
        Options include: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'
        where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order; 
        'previous' and 'next' simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. 
        Default is 'linear'.

    step : float
        The interpolation step. Default is mean spacing between consecutive points.
        Step_style will be ignored if this is passed.

    start : float
        Where/when to start the interpolation. Default is the minimum.

    stop : float
        Where/when to stop the interpolation. Default is the maximum.

    step_style : str; {'min','mean','median','max'}
        Step style to use when determining the size of the interval between points. Default is None.

    time_axis : np.ndarray
        Time axis onto which the series will be interpolated.
        Start, stop, step, and step_style will be ignored if this is passed

    kwargs :  kwargs
        Aguments specific to interpolate.interp1D.
        If getting an error about extrapolation, you can use the arguments `bound_errors=False` and `fill_value="extrapolate"` to allow for extrapolation. 

    Returns
    -------
    xi : array
        The interpolated x-axis
    yi : array
        The interpolated y values

    Notes
    -----

    `start`, `stop`, `step` and `step_styl`e pertain to the creation of the time axis over which interpolation will be conducted.
    This differs from the way that the functions `bin` and `gkernel` interpret these arguments, which is as defining
    the `bin_edges` parameter used in those functions.

    See Also
    --------

    pyleoclim.utils.tsutils.increments : Establishes the increments of a numerical array

    pyleoclim.utils.tsutils.make_even_axis : Makes an evenly spaced time axis

    pyleoclim.utils.tsutils.bin : Bin the values

    pyleoclim.utils.tsutils.gkernel : Coarsen time resolution using a Gaussian kernel

    Examples
    --------

    There are several ways to specifiy a time axis for interpolation. Within these there is a hierarchy which we demonstrate below.

    Top priority will always go to `time_axis` if it is passed. All other arguments will be overwritten (except for x,y, and interp_type).

    .. jupyter-execute::

        import numpy as np
        import pyleoclim as pyleo

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(x))
        xi,yi = pyleo.utils.tsutils.interp(x,y,time_axis=[1,4,8,12,16])
        xi
    
    If `time_axis` is None, `step` will be considered, overriding `step_style if it is passed. `start` and `stop` will be generated using defaults if not passed.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(x))
        xi,yi = pyleo.utils.tsutils.interp(x,y,step=2)
        xi
    
    If both `time_axis` and `step` are None but `step_style` is specified, the step will be generated using the prescribed `step_style`.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(x))
        xi,yi = pyleo.utils.tsutils.interp(x,y,step_style='max')
        xi

    If none of these are specified, the mean spacing will be used.

    .. jupyter-execute::

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(x))
        xi,yi = pyleo.utils.tsutils.interp(x,y)
        xi

    """

    #Make sure x and y are numpy arrays
    x = np.array(x,dtype='float64')
    y = np.array(y,dtype='float64')

    #Drop nans if present before interpolating
    if np.isnan(y).any():
        y,x = dropna(y,x)

    # get the evenly spaced time axis if one is not passed.
    if time_axis is not None:
        if start is not None or stop is not None or step is not None or step_style is not None:
            warnings.warn('A time axis has been passed with other time axis relevant arguments {start,stop,step,step_style}. The passed time axis takes priority and will be used.',stacklevel=3)
        pass
    else:
        time_axis = make_even_axis(x=x,start=start,stop=stop,step=step,step_style=step_style)

    #Make sure the data is increasing
    data = pd.DataFrame({"x-axis": x, "y-axis": y}).sort_values('x-axis')
    time_axis = np.sort(time_axis)

    # Add arguments
    yi = interpolate.interp1d(data['x-axis'],data['y-axis'],kind=interp_type,**kwargs)(time_axis)

    return time_axis, yi


def standardize(x, scale=1, axis=0, ddof=0, eps=1e-3):
    """Centers and normalizes a time series. Constant or nearly constant time series not rescaled.

    Parameters
    ----------
    x : array
        vector of (real) numbers as a time series, NaNs allowed
    scale : real
        A scale factor used to scale a record to a match a given variance
    axis : int or None
        axis along which to operate, if None, compute over the whole array
    ddof : int
        degress of freedom correction in the calculation of the standard deviation
    eps : real
        a threshold to determine if the standard deviation is too close to zero

    Returns
    -------
    z : array
       The standardized time series (z-score), Z = (X - E[X])/std(X)*scale, NaNs allowed
    mu : real
        The mean of the original time series, E[X]
    sig : real
         The standard deviation of the original time series, std[X]

    References
    ----------

    Tapio Schneider's MATLAB code: https://github.com/tapios/RegEM/blob/master/standardize.m

    The zscore function in SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html

    See also
    --------

    pyleoclim.utils.tsutils.preprocess : pre-processes a times series using standardization and detrending.

    """
    x = np.asanyarray(x)
    assert x.ndim <= 2, 'The time series x should be a vector or 2-D array!'

    mu = np.nanmean(x, axis=axis)  # the mean of the original time series
    sig = np.nanstd(x, axis=axis, ddof=ddof)  # the standard deviation of the original time series

    mu2 = np.asarray(np.copy(mu))  # the mean used in the calculation of zscore
    sig2 = np.asarray(np.copy(sig) / scale)  # the standard deviation used in the calculation of zscore

    if np.any(np.abs(sig) < eps):  # check if x contains (nearly) constant time series
        warnings.warn('Constant or nearly constant time series not rescaled.',stacklevel=2)
        where_const = np.abs(sig) < eps  # find out where we have (nearly) constant time series

        # if a vector is (nearly) constant, keep it the same as original, i.e., substract by 0 and divide by 1.
        mu2[where_const] = 0
        sig2[where_const] = 1

    if axis and mu.ndim < x.ndim:
        z = (x - np.expand_dims(mu2, axis=axis)) / np.expand_dims(sig2, axis=axis)
    else:
        z = (x - mu2) / sig2

    return z, mu, sig

def center(y, axis=0):
    """ Centers array y (i.e. removes the sample mean) 

    Parameters
    ----------
    y : array
        Vector of (real) numbers as a time series, NaNs allowed
    axis : int or None
        Axis along which to operate, if None, compute over the whole array
        
    Returns
    -------
    yc : array
       The centered time series, yc = (y - ybar), NaNs allowed
    ybar : real
        The sampled mean of the original time series, y 

    References
    ----------

    Tapio Schneider's MATLAB code: https://github.com/tapios/RegEM/blob/master/center.m

    """
    y = np.asanyarray(y)
    assert y.ndim <= 2, 'The time series y should be a vector or 2-D array!'

    ybar = np.nanmean(y, axis=axis)  # the mean of the original time series

    if axis and ybar.ndim < y.ndim:
        yc = y - np.expand_dims(ybar, axis=axis) 
    else:
        yc = y - ybar

    return yc, ybar


def ts2segments(ys, ts, factor=10):
    ''' Chop a time series into several segments based on gap detection.

    The rule of gap detection is very simple:
        we define the intervals between time points as dts, then if dts[i] is larger than factor * dts[i-1],
        we think that the change of dts (or the gradient) is too large, and we regard it as a breaking point
        and chop the time series into two segments here

    Parameters
    ----------
    ys : array
        A time series, NaNs allowed
    ts : array
        The time points
    factor : float
        The factor that adjusts the threshold for gap detection

    Returns
    -------
    seg_ys : list
        A list of several segments with potentially different lengths
    seg_ts : list
        A list of the time axis of the several segments
    n_segs : int
        The number of segments
    '''

    ys, ts = clean_ts(ys, ts)

    nt = np.size(ts)
    dts = np.diff(ts)

    seg_ys, seg_ts = [], []  # store the segments with lists

    n_segs = 1
    i_start = 0
    for i in range(1, nt-1):
        if np.abs(dts[i]) > factor*np.abs(dts[i-1]):
            i_end = i + 1
            seg_ys.append(ys[i_start:i_end])
            seg_ts.append(ts[i_start:i_end])
            i_start = np.copy(i_end)
            n_segs += 1

    seg_ys.append(ys[i_start:nt])
    seg_ts.append(ts[i_start:nt])

    return seg_ys, seg_ts, n_segs



def annualize(ys, ts):
    ''' Annualize a time series whose time resolution is finer than 1 year

    Parameters
    ----------
    ys : array
        A time series, NaNs allowed
    ts : array
        The time axis of the time series, NaNs allowed

    Returns
    -------
    ys_ann : array
            the annualized time series
    year_int : array
              The time axis of the annualized time series

    '''
    ys = np.asarray(ys, dtype=float)
    ts = np.asarray(ts, dtype=float)
    assert ys.size == ts.size, 'The size of time axis and data value should be equal!'

    year_int = list(set(np.floor(ts)))
    year_int = np.sort(list(map(int, year_int)))
    n_year = len(year_int)
    year_int_pad = list(year_int)
    year_int_pad.append(np.max(year_int)+1)
    ys_ann = np.zeros(n_year)

    for i in range(n_year):
        t_start = year_int_pad[i]
        t_end = year_int_pad[i+1]
        t_range = (ts >= t_start) & (ts < t_end)
        ys_ann[i] = np.average(ys[t_range], axis=0)

    return ys_ann, year_int

def gaussianize(ys):
    """ Maps a 1D array to a Gaussian distribution using the inverse Rosenblatt transform
    
    The resulting array is mapped to a standard normal distribution, and therefore
    has zero mean and unit standard deviation. Using `gaussianize()` obviates the 
    need for `standardize()`. 
    
    Parameters
    ----------
    ys : 1D Array
        e.g. a timeseries

    Returns
    -------
    yg : 1D Array
        Gaussianized values of ys.

    References
    ----------
    
    van Albada, S., and P. Robinson (2007), Transformation of arbitrary 
        distributions to the normal distribution with application to EEG 
        test-retest reliability, Journal of Neuroscience Methods, 161(2), 
        205 - 211, doi:10.1016/j.jneumeth.2006.11.004.   

    See also
    --------

    pyleoclim.utils.tsutils.standardize : Centers and normalizes a time series

    """
    # Count only elements with data.

    n = ys[~np.isnan(ys)].shape[0]

    # Create a blank copy of the array.
    yg = copy.deepcopy(ys)
    yg[:] = np.NAN

    nz = np.logical_not(np.isnan(ys))
    index = np.argsort(ys[nz])
    rank = np.argsort(index)
    CDF = 1.*(rank+1)/(1.*n) - 1./(2*n)
    yg[nz] = np.sqrt(2)*special.erfinv(2*CDF - 1)

    return yg


def detrend(y, x=None, method="emd", n=1, preserve_mean = False, sg_kwargs=None):
    """Detrend a timeseries according to four methods

    Detrending methods include: "linear", "constant", using a low-pass Savitzky-Golay filter, and Empirical Mode Decomposition (default).
    Linear and constant methods use `scipy.signal.detrend <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html>`_.,
    EMD uses `pyhht.emd.EMD <https://pyhht.readthedocs.io/en/stable/apiref/pyhht.html>`_.

    Parameters
    ----------
    y : array

       The series to be detrended.

    x : array

       Abscissa for array y. Necessary for use with the Savitzky-Golay 
       method, since the series should be evenly spaced.

    method : str

        The type of detrending:

        - "linear": the result of a linear least-squares fit to y is subtracted from y.
        - "constant": only the mean of data is subtracted.
        - "savitzky-golay", y is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        - "emd" (default): Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series

    n : int

        Works only if `method == 'emd'`. The number of smoothest modes to remove.

    preserve_mean : boolean

        flag to indicate whether the mean of the series should be preserved despite the detrending

    sg_kwargs : dict

        The parameters for the Savitzky-Golay filters.

    Returns
    -------
    ys : array
        The detrended version of y.
        
    trend : array
        The removed trend. Only non-empty for EMD and Savitzy-Golay methods, since SciPy detrending does not retain the trends

    See also
    --------

    pyleoclim.utils.filter.savitzky_golay : Filtering using Savitzy-Golay

    pyleoclim.utils.tsutils.preprocess : pre-processes a times series using standardization and detrending.

    """
    y = np.array(y)
    mu = y.mean()

    if x is not None:
        x = np.array(x)

    if method == "linear":
        ys = signal.detrend(y,type='linear')
        trend = y - ys
    elif method == 'constant':
        ys = signal.detrend(y,type='constant')
        trend = y - ys 
    elif method == "savitzky-golay":
        # Check that the timeseries is uneven and interpolate if needed
        if x is None:
            raise ValueError("An independent variable is needed for the Savitzky-Golay filter method")
        # Check whether the timeseries is unvenly-spaced and interpolate if needed
        if len(np.unique(np.diff(x)))>1:
            warnings.warn("Timeseries is not evenly-spaced, interpolating...")
            x_interp, y_interp = interp(x,y,bounds_error=False,fill_value='extrapolate')
        else:
            x_interp = x
            y_interp = y
        sg_kwargs = {} if sg_kwargs is None else sg_kwargs.copy()
        # Now filter
        y_filt = savitzky_golay(y_interp,**sg_kwargs)
        # Put it all back on the original x axis
        trend = np.interp(x,x_interp,y_filt)
        ys = y - trend
    elif method == "emd":
        imfs = EmpiricalModeDecomposition(y).decompose()
        if np.shape(imfs)[0] == 1:
            trend = np.zeros(np.size(y))
        else:
            trend = np.sum(imfs[-n:], axis=0)  # remove the n smoothest modes

        ys = y - trend
    else:
        raise KeyError('Unknown method. Use one of linear, constant, savitzky-golay, emd (case-sensitive)')
    
    if preserve_mean:
        ys = ys - ys.mean() + mu

    return ys, trend

def calculate_distances(ys, n_neighbors=None, NN_kwargs=None):
    """
    
    Uses the scikit-learn unsupervised learner for implementing neighbor searches and calculate the distance between a point and its nearest neighbors to estimate epsilon for DBSCAN. 
    

    Parameters
    ----------
    ys : numpy.array
        the y-values for the timeseries
    n_neighbors : int, optional
        Number of neighbors to use by default for kneighbors queries. The default is None.
    NN_kwargs : dict, optional
        Other arguments for sklearn.neighbors.NearestNeighbors. The default is None.
        See: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors

    Returns
    -------
    min_eps : int
        Minimum value for epsilon.
    max_eps : int
        Maximum value for epsilon.

    """
    
    ys=standardize(ys)[0]
    ys=np.array(ys)
    
    if n_neighbors is None:
        # Lowest number of nearest neighbors
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(ys.reshape(-1, 1))
        distances, indices = nbrs.kneighbors(ys.reshape(-1, 1))
        min_eps = np.min(distances)
        if min_eps<=0:
            min_eps=0.01
    
        # Highest number of nearest neighbors
        neigh = NearestNeighbors(n_neighbors=len(ys)-1)
        nbrs = neigh.fit(ys.reshape(-1, 1))
        distances, indices = nbrs.kneighbors(ys.reshape(-1, 1))
        max_eps = np.max(distances)
    
    else:
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(ys.reshape(-1, 1))
        distances, indices = nbrs.kneighbors(ys.reshape(-1, 1))
        min_eps = np.min(distances)
        max_eps = np.max(distances)
    
    return min_eps, max_eps

def detect_outliers_DBSCAN(ys, nbr_clusters = None, eps=None, min_samples=None, n_neighbors=None, metric='euclidean', NN_kwargs= None, DBSCAN_kwargs=None):
    """
    Uses the unsupervised learning DBSCAN algorithm to identify outliers in timeseries data. 
    The algorithm uses the silhouette score calculated over a range of epsilon and minimum sample values to determine the best clustering. In this case, we take the largest silhouette score (as close to 1 as possible). 
    
    The DBSCAN implementation used here is from scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    
    The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar. For additional details, see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

    Parameters
    ----------
    ys : numpy.array
        The y-values for the timeseries data.
    nbr_clusters : int, optional
        Number of clusters. Note that the DBSCAN algorithm calculates the number of clusters automatically. This paramater affects the optimization over the silhouette score. The default is None.
    eps : float or list, optional
        epsilon. The default is None, which allows the algorithm to optimize for the best value of eps, using the silhouette score as the optimization criterion. The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
    min_samples : int or list, optional
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.. The default is None and optimized using the silhouette score
    n_neighbors : int, optional
        Number of neighbors to use by default for kneighbors queries, which can be used to calculate a range of plausible eps values. The default is None.
    metric : str, optional
        The metric to use when calculating distance between instances in a feature array. The default is 'euclidean'. See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html for alternative values. 
    NN_kwargs : dict, optional
        Other arguments for sklearn.neighbors.NearestNeighbors. The default is None.
        See: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors

    DBSCAN_kwargs : dict, optional
        Other arguments for sklearn.cluster.DBSCAN. The default is None.
        See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html


    Returns
    -------
    indices : list
        list of indices that are considered outliers.
    res : pandas.DataFrame
        Results of the clustering analysis. Contains information about eps value, min_samples value, number of clusters, the silhouette score, the indices of the outliers for each combination, and the cluster assignment for each point. 

    """
    
    
    NN_kwargs = {} if NN_kwargs is None else NN_kwargs.copy()
    DBSCAN_kwargs = {} if DBSCAN_kwargs is None else DBSCAN_kwargs.copy()
    
    ys=standardize(ys)[0] # standardization is key for the alogrithm to work.
    ys=np.array(ys)
    
    if eps and n_neighbors:
        print('Since eps is passed, ignoring the n_neighbors for distance calculation')
    
    if eps is None:
        min_eps,max_eps = calculate_distances(ys, n_neighbors=n_neighbors, NN_kwargs=NN_kwargs)       
        eps_list = np.linspace(min_eps,max_eps,50)
    elif type(eps) is list:
        eps_list=eps
    else:
        print("You have tried to pass a float or integer, coercing to a list")
        eps_list=list(eps)
    
    if min_samples is None:
        min_samples_list = np.linspace(2,len(ys)/4,50,dtype='int')
    elif type(min_samples) is list:
        min_samples_list = min_samples
    else:
        print("You have tried to pass a float or integer, coercing to a list")
        min_samples_list=list(min_samples)
    
    print("Optimizing for the best number of clusters, this may take a few minutes")
    
    
    nbr_clusters=[]
    sil_score =[]
    eps_matrix=[]
    min_sample_matrix=[]
    idx_out = []
    clusters = []

    for eps_item in eps_list:
        for min_samples_item in min_samples_list:
            eps_matrix.append(eps_item)
            min_sample_matrix.append(min_samples_item)
            m = DBSCAN(eps=eps_item, min_samples=min_samples_item,**DBSCAN_kwargs)
            m.fit(ys.reshape(-1,1))
            nbr_clusters.append(len(np.unique(m.labels_))-1)
            try:
                sil_score.append(silhouette_score(ys.reshape(-1,1), m.labels_))
            except:
                sil_score.append(np.nan)
            idx_out.append(np.where(m.labels_==-1)[0])
            clusters.append(m.labels_)
            
    res = pd.DataFrame({'eps':eps_matrix,'min_samples':min_sample_matrix,'number of clusters':nbr_clusters,'silhouette score':sil_score,'outlier indices':idx_out,'clusters':clusters})
    
    if nbr_clusters is None: 
        res_sil = res.loc[res['silhouette score']==np.max(res['silhouette score'])]
    else:
        try: 
            res_cl = res.loc[res['number of clusters']==nbr_clusters]
            res_sil = res_cl.loc[res_cl['silhouette score']==np.max(res_cl['silhouette score'])]
        except:
            print("No valid solutions for the number of clusters, returning from silhouette score")
            res_sil = res.loc[res['silhouette score']==np.max(res['silhouette score'])]
    
    unique_idx = list(res_sil['outlier indices'].iloc[0])
    
    if res_sil.shape[0]>1:
        for idx,row in res_sil.iterrows():
            for item in row['outlier indices']:
                if item not in unique_idx:
                    unique_idx.append(item)
            
    indices = np.array(unique_idx)
    
    return indices, res

def detect_outliers_kmeans(ys, nbr_clusters = None, max_cluster = 10, threshold=3, LOF=False, n_frac=0.9, contamination='auto', kmeans_kwargs=None):
    """
    Outlier detection using the unsupervised alogrithm kmeans. The algorithm runs through various number of clusters and optimizes based on the silhouette score.
    
    KMeans implementation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    
    The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar. For additional details, see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

    Outliers are identified based on their distance from the clusters. This can be done in two ways: (1) by using a threshold that corresponds to the Euclidean distance from the centroid and (2) using the Local Outlier Function (https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html)

    Parameters
    ----------
    ys : numpy.array
        The y-values for the timeseries data
    nbr_clusters : int or list, optional
        A user number of clusters to considered. The default is None.
    max_cluster : int, optional
        The maximum number of clusters to consider in the optimization based on the Silhouette Score. The default is 10.
    threshold : int, optional
        The algorithm uses the euclidean distance for each point in the cluster to identify the outliers. This parameter sets the threshold on the euclidean distance to define an outlier. The default is 3.
    LOF : bool, optional
        By default, detect_outliers_kmeans uses euclidean distance for outlier detection. Set LOF to True to use LocalOutlierFactor for outlier detection.
    n_frac : float, optional
        The percentage of the time series length (the length, representing number of points) to be used to set the n_neighbors parameter for the LOF function in scikit-learn. 
        We recommend using at least 50% (n_frac=0.5) of the timeseries. You cannot use 100% (n_frac!=1)
    contamination : ('auto', float), optional
        Same as LOF parameter from scikit-learn. We recommend using the default mode of auto. See: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html for details.
    kmeans_kwargs : dict, optional
        Other parameters for the kmeans function. See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for details. The default is None.

    Returns
    -------
    indices : list
        list of indices that are considered outliers.
    res : pandas.DataFrame
        Results of the clustering analysis. Contains information about number of clusters, the silhouette score, the indices of the outliers for each combination, and the cluster assignment for each point. 


    """
    
    
    kmeans_kwargs = {} if kmeans_kwargs is None else kmeans_kwargs.copy()
    
    ys=standardize(ys)[0] # standardization is key for the alogrithm to work.
    ys=np.array(ys)
    
    # run with either one cluster number of several
    if nbr_clusters is not None:
        if type(nbr_clusters) == list:
            range_n_clusters = nbr_clusters
        else:
            range_n_clusters = [nbr_clusters]
    else:
        range_n_clusters = np.arange(2,max_cluster+1,1,dtype='int')
    silhouette_avg = []
    idx_out=[]
    clusters = []
    
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(ys.reshape(-1, 1), **kmeans_kwargs)
        silhouette_avg.append(silhouette_score(ys.reshape(-1, 1), kmeans.labels_))
        center=kmeans.cluster_centers_[kmeans.labels_,0]
        if LOF:
            model = LocalOutlierFactor(n_neighbors=int(ys.size*n_frac), contamination=contamination)
            pred = model.fit_predict(ys.reshape(-1,1))
            idx_out.append(np.where(pred==-1))
        else:
            distance=np.sqrt((ys-center)**2)
            idx_out.append(np.argwhere(distance>threshold).reshape(1,-1)[0])
        clusters.append(kmeans.labels_)
    
    res = pd.DataFrame({'number of clusters':range_n_clusters, 'silhouette score':silhouette_avg,'outlier indices':idx_out,'clusters':clusters})
    res_sil = res.loc[res['silhouette score']==np.max(res['silhouette score'])]

    unique_idx = list(res_sil['outlier indices'].iloc[0])
    
    if res_sil.shape[0]>1:
        for idx,row in res_sil.iterrows():
            for item in row['outlier indices']:
                if item not in unique_idx:
                    unique_idx.append(item)
            
    indices = np.array(unique_idx)
    
    return indices, res

def remove_outliers(ts,ys,indices):
    """
    Remove the outliers from timeseries data

    Parameters
    ----------
    ts : numpy.array
        The time axis for the timeseries data.
    ys : numpy.array
        The y-values for the timeseries data.
    indices : numpy.array
        The indices of the outliers to be removed.

    Returns
    -------
    ys : numpy.array
        The time axis for the timeseries data after outliers removal
    ts : numpy.array
        The y-values for the timeseries data after outliers removal

    """
    ys = np.delete(ys,indices)
    ts = np.delete(ts,indices)

    return ys,ts

def eff_sample_size(y, detrend_flag=False):
    '''Effective Sample Size of timeseries y

    Parameters
    ----------
    y : float 
       1d array 
       
    detrend_flag : boolean
        if True (default), detrends y before estimation.         

    Returns
    -------
    neff : float
        The effective sample size
    
    References
    ----------

    Thiébaux HJ and Zwiers FW, 1984: The interpretation and estimation of
    effective sample sizes. Journal of Climate and Applied Meteorology 23: 800–811.

    '''
    if len(y) < 100:
        fft = False
    else:
        fft = True
        
    if detrend_flag:
        yd = detrend(y)
    else:
        yd = y
    
    n     = len(y)
    nl    = math.floor(max(np.sqrt(n),10))     # rule of thumb for choosing number of lags
    rho   = sms.acf(yd,adjusted=True,fft=fft,nlags=nl) # compute autocorrelation function         
    kvec  = np.arange(nl)
    fac   = (1-kvec/nl)*rho[1:]
    neff  = n/(1+2*np.sum(fac))   # Thiébaux & Zwiers 84, Eq 2.1
    
    return neff

# alias
std = standardize
gauss = gaussianize

def preprocess(ys, ts, detrend=False, sg_kwargs=None,
               gaussianize=False, standardize=True):
    ''' Return the processed time series using detrend and standardization.

    Parameters
    ----------
    ys : array

        a time series

    ts : array

        The time axis for the timeseries. Necessary for use with
        the Savitzky-Golay filters method since the series should be evenly spaced.

    detrend : string

        'none'/False/None - no detrending will be applied;
        'emd' - the last mode is assumed to be the trend and removed from the series
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filter and the resulting filtered series is subtracted.

    sg_kwargs : dict

        The parameters for the Savitzky-Golay filter.

    gaussianize : bool

        If True, gaussianizes the timeseries
        
    standardize : bool

        If True, standardizes the timeseries

    Returns
    -------
    res : array
        the processed time series

    See also
    --------

    pyleoclim.utils.tsutils.detrend : Detrend a timeseries according to four methods

    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay method

    pyleoclim.utils.tsutils.standardize : Centers and normalizes a given time series

    pyleoclim.utils.tsutils.gaussianize : Quantile maps a matrix to a Gaussian distribution

    '''

    if detrend == 'none' or detrend is False or detrend is None:
        ys_d = ys
    else:
        ys_d = detrend(ys, ts, method=detrend, sg_kwargs=sg_kwargs)

    if standardize:
        res, _, _ = std(ys_d)
    else:
        res = ys_d

    if gaussianize:
        res = gauss(res)

    return res

def make_even_axis(x=None,start=None,stop=None,step=None,step_style=None,no_nans=False):
    """Create a uniform time axis for binning/interpolating
    
    Parameters
    ----------
    x : np.ndarray
        Uneven time axis upon which to base the uniform time axis.
    
    start : float
        Where to start the axis. Default is the first value of the passed time axis.
    
    stop : float
        Where to stop the axis. Default is the last of value of the passed time axis.
    
    step : float
        The step size to use for the axis. Must be greater than 0.
        
    step_style : str; {}
        Step style to use when defining the step size. Will be overridden by `step` if it is passed.
    
    no_nans : bool: {True,False}
        Whether or not to allow nans. When True, will set step style to 'max'.
        Will be overridden by `step_style` or `step` if they are passed. Default is False.

    -------

    time_axis : np.ndarray
        An evenly spaced time axis.

    """
    
    if start is None:
        if x is None:
            raise ValueError('If x is not passed then start, stop and step must be passed')
        else:
            start = x[0]

    if stop is None:
        if x is None:
            raise ValueError('If x is not passed then start, stop and step must be passed')
        else:
            stop = x[-1]
    
    if step is not None:
        pass
    elif step_style is not None:
        if x is None:
            raise ValueError('If x is not passed then start, stop and step must be passed')
        else:
            _, _, step = increments(np.asarray(x), step_style = step_style)
    elif no_nans:
        if x is None:
            raise ValueError('If x is not passed then start, stop and step must be passed')
        else:
            _, _, step = increments(np.asarray(x), step_style = 'max')
    else:
        if x is None:
            raise ValueError('If x is not passed then start, stop and step must be passed')
        else:
            _, _, step = increments(np.asarray(x), step_style = 'mean')
    
    new_axis = np.arange(start,stop+step,step)

    #Make sure that values in time_axis don't exceed the stop value
    if step > 0:
        if max(new_axis) > stop:
            time_axis = np.array([t for t in new_axis if t <= stop])
        else:
            time_axis = new_axis
    elif step < 0:
        if min(new_axis) < stop:
            time_axis = np.array([t for t in new_axis if t >= stop])
        else:
            time_axis = new_axis
    else:
        raise ValueError('Step must be nonzero')

    return time_axis


def phaseran(recblk, nsurr):
    ''' Simultaneous phase randomization of a set of time series
    
    It creates blocks of surrogate data with the same second order properties as the original
    time series dataset by transforming the original data into the frequency domain, randomizing the
    phases simultaneoulsy across the time series and converting the data back into the time domain. 
    
    Written by Carlos Gias for MATLAB

    http://www.mathworks.nl/matlabcentral/fileexchange/32621-phase-randomization/content/phaseran.m

    Parameters
    ----------
    recblk : numpy array
        2D array , Row: time sample. Column: recording.
        An odd number of time samples (height) is expected.
        If that is not the case, recblock is reduced by 1 sample before the surrogate data is created.
        The class must be double and it must be nonsparse.
    
    nsurr : int
        is the number of image block surrogates that you want to generate.

    Returns
    -------
    surrblk : numpy array
        3D multidimensional array image block with the surrogate datasey along the third dimension

    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    pyleoclim.utils.correlation.fdf : Determine significance based on the false discovery rate

    References
    ----------

    - Prichard, D., Theiler, J. Generating Surrogate Data for Time Series with Several Simultaneously Measured Variables (1994) Physical Review Letters, Vol 73, Number 7
    
    - Carlos Gias (2020). Phase randomization, MATLAB Central File Exchange
    '''
    # Get parameters
    nfrms = recblk.shape[0]

    if nfrms % 2 == 0:
        nfrms = nfrms-1
        recblk = recblk[0:nfrms]

    len_ser = int((nfrms-1)/2)
    interv1 = np.arange(1, len_ser+1)
    interv2 = np.arange(len_ser+1, nfrms)

    # Fourier transform of the original dataset
    fft_recblk = np.fft.fft(recblk)

    surrblk = np.zeros((nfrms, nsurr))

    #  for k in tqdm(np.arange(nsurr)):
    for k in np.arange(nsurr):
        ph_rnd = np.random.rand(len_ser)

        # Create the random phases for all the time series
        ph_interv1 = np.exp(2*np.pi*1j*ph_rnd)
        ph_interv2 = np.conj(np.flipud(ph_interv1))

        # Randomize all the time series simultaneously
        fft_recblk_surr = np.copy(fft_recblk)
        fft_recblk_surr[interv1] = fft_recblk[interv1] * ph_interv1
        fft_recblk_surr[interv2] = fft_recblk[interv2] * ph_interv2

        # Inverse transform
        surrblk[:, k] = np.real(np.fft.ifft(fft_recblk_surr))

    return surrblk


def phaseran2(y, nsurr):
    '''
    Phase randomization of a time series y, of even or odd length. 
    
    Closely follows this strategy: https://stackoverflow.com/q/39543002

    Parameters
    ----------
    y : array, length nt
        Signal to be scrambled
        
    nsurr : int
        is the number of image block surrogates that you want to generate.

    Returns
    -------
    ysurr: array nt x nsurr
        Array of y surrogates

    '''
    from scipy.fft import fft, ifft
    nt = len(y); n2 = nt //2 
    ys = fft(y)
    pow_ys = np.abs(ys) ** 2.
    phase_ys = np.angle(ys)
    ysurr = np.zeros((nt, nsurr))
    
    for i in range(nsurr):
        phase_ysr = np.empty_like(phase_ys)
        if nt % 2 == 0: # deal with even and odd-length arrays
            phase_ysr_lh = np.random.rand(n2-1)
            phase_ysr_rh = -phase_ysr_lh[::-1]
            phase_ysr = np.concatenate((np.array((phase_ys[0],)), phase_ysr_lh,
                                        np.array((phase_ys[n2],)),
                                        phase_ysr_rh))
        else:
            phase_ysr_lh = np.random.rand(n2)
            phase_ysr_rh = -phase_ysr_lh[::-1]
            phase_ysr = np.concatenate((np.array((phase_ys[0],)), phase_ysr_lh, phase_ysr_rh))
        # put it back together
        ysrp = np.sqrt(pow_ys) * np.exp(2*np.pi*1j*phase_ysr) 
        ysurr[:,i] = ifft(ysrp).real
        
            
    return ysurr
    


