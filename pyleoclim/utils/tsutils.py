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
    'remove_outliers'
]

import numpy as np
import pandas as pd
import warnings
import copy
from scipy import special
from scipy import signal
from scipy import interpolate
from scipy import stats
from pyhht import EMD
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as sms

import math
from .filter import savitzky_golay

from .tsbase import (
    clean_ts,
    dropna,
    is_evenly_spaced
)


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

        Examples
    --------

    There are several ways to specify the way binning is conducted via this function. Within these there is a hierarchy which we demonstrate below.

    Top priority is given to `bin_edges` if it is not None. All other arguments will be ignored (except for x and y).
    The resulting time axis will be comprised of the midpoints between bin edges.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xb,yb = pyleo.utils.tsutils.bin(x,y,bin_edges=[1,4,8,12,16,20])
        xb

    Next, priority will go to `time_axis` if it is passed. In this case, bin edges will be taken as the midpoints between time axis points.
    The first and last time point will be used as the left most and right most bin edges.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xb,yb = pyleo.utils.tsutils.bin(x,y,time_axis=[1,4,8,12,16,20])
        xb
    
    If `time_axis` is None, `bin_size` will be considered, overriding `step_style if it is passed. `start` and `stop` will be generated using defaults if not passed.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xb,yb = pyleo.utils.tsutils.bin(x,y,bin_size=2)
        xb
    
    If both `time_axis` and `step` are None but `step_style` is specified, the step will be generated using the prescribed `step_style`.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xb,yb = pyleo.utils.tsutils.bin(x,y,step_style='max')
        xb

    If none of these are specified, the mean spacing will be used.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xb,yb = pyleo.utils.tsutils.bin(x,y)
        xb

    """

    if evenly_spaced:
        no_nans=True
        warnings.warn('`evenly_spaced` is being deprecated. Please switch to using the option `no_nans` (behaviour is identical).',DeprecationWarning,stacklevel=3)

    # Make sure x and y are numpy arrays
    x = np.array(x, dtype='float64')
    y = np.array(y, dtype='float64')
    
    # Set the bin edges
    if bin_edges is not None:
        if start is not None or stop is not None or bin_size is not None or step_style is not None or time_axis is not None:
            warnings.warn('Bins have been passed with other bin relevant arguments {start,stop,bin_size,step_style,time_axis}. Bin_edges take priority and will be used.',stacklevel=3)
        time_axis = (bin_edges[1:] + bin_edges[:-1])/2
    # A bit of wonk is required to get the proper bin edges from the time axis
    elif time_axis is not None:
        if start is not None or stop is not None or bin_size is not None or step_style is not None:
            warnings.warn('The time axis has been passed with other time axis relevant arguments {start,stop,bin_size,step_style}. Time_axis takes priority and will be used.',stacklevel=3)
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

    if no_nans is True:
        _,ts = dropna(binned_values,time_axis)
        check = is_evenly_spaced(ts)
        if not check:
            warnings.warn('no_nans is set to True but has been overridden by other parameters. This has resulted in nans being present in the returned series',stacklevel=3)

    return  res_dict


def gkernel(t,y, h = 3.0, step=None,start=None,stop=None, step_style = None, evenly_spaced=False, bin_edges=None, time_axis=None,no_nans=True):
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
        kernel e-folding scale
    
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

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xc,yc = pyleo.utils.tsutils.gkernel(x,y,bin_edges=[1,4,8,12,16,20])
        xc

    Next, priority will go to `time_axis` if it is passed. In this case, bin edges will be taken as the midpoints between time axis points.
    The first and last time point will be used as the left most and right most bin edges.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xc,yc = pyleo.utils.tsutils.gkernel(x,y,time_axis=[1,4,8,12,16,20])
        xc
    
    If `time_axis` is None, `step` will be considered, overriding `step_style` if it is passed. `start` and `stop` will be generated using defaults if not passed.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xc,yc = pyleo.utils.tsutils.gkernel(x,y,step=2)
        xc
    
    If both `time_axis` and `step` are None but `step_style` is specified, the step will be generated using the prescribed `step_style`.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xc,yc = pyleo.utils.tsutils.gkernel(x,y,step_style='max')
        xc

    If none of these are specified, the mean spacing will be used.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xc,yc = pyleo.utils.tsutils.gkernel(x,y)
        xc

    '''

    if len(t) != len(y):
        raise ValueError('y and t must have the same length')
    
    if evenly_spaced:
        no_nans=True
        warnings.warn('`evenly_spaced` is being deprecated. Please switch to using the option `no_nans` (behaviour is identical).',DeprecationWarning,stacklevel=3)

        # Make sure x and y are numpy arrays
    t = np.array(t, dtype='float64')
    y = np.array(y, dtype='float64')
    
    # Set the bin edges
    if bin_edges is not None:
        if start is not None or stop is not None or step is not None or step_style is not None or time_axis is not None:
            warnings.warn('Bins have been passed with other axis relevant arguments {start,stop,step,step_style,time_axis}. Bin_edges take priority and will be used.',stacklevel=3)
        time_axis = (bin_edges[1:] + bin_edges[:-1])/2
    # A bit of wonk is required to get the proper bin edges from the time axis
    elif time_axis is not None:
        if start is not None or stop is not None or step is not None or step_style is not None:
            warnings.warn('The time axis has been passed with other axis relevant arguments {start,stop,step,step_style}. Time_axis takes priority and will be used.',stacklevel=3)
        bin_edges = np.zeros(len(time_axis)+1)
        bin_edges[0] = time_axis[0]
        bin_edges[-1] = time_axis[-1]
        bin_edges[1:-1] = (time_axis[1:]+time_axis[:-1])/2
    else:
        bin_edges = make_even_axis(x=t,start=start,stop=stop,step=step,step_style=step_style,no_nans=no_nans)
        time_axis = (bin_edges[1:]+bin_edges[:-1])/2

    kernel = lambda x, s : 1.0/(s*np.sqrt(2*np.pi))*np.exp(-0.5*(x/s)**2)  # define kernel function

    yc    = np.zeros((len(time_axis)))
    yc[:] = np.nan

    for i in range(len(bin_edges)-1):
        if i < len(bin_edges-1):
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

        if no_nans is True:
            _,ts = dropna(yc,time_axis)
            check = is_evenly_spaced(ts)
            if not check:
                warnings.warn('no_nans is set to True but has been overridden by other parameters. This has resulted in nans being present in the returned series',stacklevel=3)

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

    Largely a wrapper for [scipy.interpolate.interp1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)

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

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xi,yi = pyleo.utils.tsutils.interp(x,y,time_axis=[1,4,8,12,16])
        xi
    
    If `time_axis` is None, `step` will be considered, overriding `step_style if it is passed. `start` and `stop` will be generated using defaults if not passed.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xi,yi = pyleo.utils.tsutils.interp(x,y,step=2)
        xi
    
    If both `time_axis` and `step` are None but `step_style` is specified, the step will be generated using the prescribed `step_style`.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
        xi,yi = pyleo.utils.tsutils.interp(x,y,step_style='max')
        xi

    If none of these are specified, the mean spacing will be used.

    .. ipython:: python
        :okwarning:
        :okexcept:

        x = np.array([1,2,3,5,8,12,20])
        y = np.ones(len(t))
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


def detrend(y, x=None, method="emd", n=1, sg_kwargs=None):
    """Detrend a timeseries according to four methods

    Detrending methods include: "linear", "constant", using a low-pass Savitzky-Golay filter, and Empirical Mode Decomposition (default).
    Linear and constant methods use [scipy.signal.detrend](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html),
    EMD uses [pyhht.emd.EMD](https://pyhht.readthedocs.io/en/stable/apiref/pyhht.html)

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
        imfs = EMD(y).decompose()
        if np.shape(imfs)[0] == 1:
            trend = np.zeros(np.size(y))
        else:
            trend = np.sum(imfs[-n:], axis=0)  # remove the n smoothest modes

        ys = y - trend
    else:
        raise KeyError('Not a valid detrending method')

    return ys, trend

def calculate_distances(ys, n_neighbors=None, NN_kwargs=None):
    """
    
    Uses the scikit-learn unsupervised learner for implementing neighbor searches and calculate the distance between a point and its nearest neighbors to estimate epsilon for DBSCAN. 
    

    Parameters
    ----------
    ys : tnumpy.array
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

def detect_outliers_kmeans(ys, nbr_clusters = None, max_cluster = 10, threshold=3, kmeans_kwargs=None):
    """
    Outlier detection using the unsupervised alogrithm kmeans. The algorithm runs through various number of clusters and optimizes based on the silhouette score.
    
    KMeans implementation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    
    The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar. For additional details, see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

    Parameters
    ----------
    ys : numpy.array
        The y-values for the timeseries data
    nbr_clusters : int or list, optional
        A user number of clusters to considered. The default is None.
    max_cluster : int, optional
        The maximum number of clusters to consider in the optimization based on the Silhouette Score. The default is 10.
    threshold : int, optional
        The algorithm uses the suclidean distance for each point in the cluster to identify the outliers. This parameter sets the threshold on the euclidean distance to define an outlier. The default is 3.
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

    pyleoclim.utils.tsutils.gaussianize_1d : Quantile maps a matrix to a Gaussian distribution

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

