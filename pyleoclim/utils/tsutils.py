#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:43:14 2020

@author: deborahkhider, fzhu, jeg

Utilities to manipulate timeseries
"""

__all__ = [
    'simple_stats',
    'bin',
    'interp',
    'gkernel',
    'grid_properties',
    'standardize',
    'ts2segments',
    'annualize',
    'gaussianize',
    'gaussianize_single',
    'detrend',
    'detect_outliers',
    'remove_outliers',
    'eff_sample_size'
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
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
import statsmodels.tsa.stattools as sms

import math
from sys import exit
from .plotting import plot_scatter_xy,plot_xy,savefig,showfig
from .filter import savitzky_golay

from .tsbase import (
    clean_ts
)

def simple_stats(y, axis=None):
    """ Computes simple statistics

    Computes the mean, median, min, max, standard deviation, and interquartile
    range of a numpy array y, ignoring NaNs.

    Parameters
    ----------

    y: array
        A Numpy array
    axis : int, tuple of ints
        Optional. Axis or Axes along which the means
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


def bin(x, y, bin_size=None, start=None, stop=None, evenly_spaced = True):
    """ Bin the values

    Parameters
    ----------

    x : array
        The x-axis series.
    y : array
        The y-axis series.
    bin_size : float
        The size of the bins. Default is the mean resolution if evenly_spaced is not True
    start : float
        Where/when to start binning. Default is the minimum
    stop : float
        When/where to stop binning. Default is the maximum
    evenly_spaced : {True,False}
        Makes the series evenly-spaced. This option is ignored if bin_size is set to float

    Returns
    -------

    binned_values : array
        The binned values
    bins : array
        The bins (centered on the median, i.e., the 100-200 bin is 150)
    n : array
        number of data points in each bin
    error : array
        the standard error on the mean in each bin

    """

    # Make sure x and y are numpy arrays
    x = np.array(x, dtype='float64')
    y = np.array(y, dtype='float64')
    
    if bin_size is not None and evenly_spaced == True:
        warnings.warn('The bin_size has been set, the series may not be evenly_spaced')

    # Get the bin_size if not available
    if bin_size is None:
        if evenly_spaced == True:
            bin_size = np.nanmax(np.diff(x))
        else:
            bin_size = np.nanmean(np.diff(x))

    # Get the start/stop if not given
    if start is None:
        start = np.nanmin(x)
    if stop is None:
        stop = np.nanmax(x)

    # Set the bin medians
    bins = np.arange(start+bin_size/2, stop + bin_size/2, bin_size)

    # Perform the calculation
    binned_values = []
    n = []
    error = []
    for val in np.nditer(bins):
        idx = [idx for idx, c in enumerate(x) if c >= (val-bin_size/2) and c < (val+bin_size/2)]
        if y[idx].size == 0:
            binned_values.append(np.nan)
            n.append(np.nan)
            error.append(np.nan)
        else:
            binned_values.append(np.nanmean(y[idx]))
            n.append(y[idx].size)
            error.append(np.nanstd(y[idx]))

    res_dict = {
        'bins': bins,
        'binned_values': binned_values,
        'n': n,
        'error': error,
    }

    return  res_dict


def gkernel(t,y, h = 3.0, step=None,start=None,stop=None, step_style = 'max'):
    '''
    Coarsen time resolution using a Gaussian kernel

    Parameters
    ----------
    t  : 1d array
        the original time axis
    
    y  : 1d array
        values on the original time axis
    
    step : float
        The interpolation step. Default is max spacing between consecutive points.
   
    step_style : 'string'
            step style to be applied from `grid_properties` [default = 'max']
    
    start : float
        where/when to start the interpolation. Default is min(t).
        
    stop : float
        where/when to stop the interpolation. Default is max(t).
    
    h  : scalar;  kernel e-folding scale

    Returns
    -------
    tc : 1d array
        the coarse-grained time axis
        
    yc:  1d array
        The coarse-grained time series

    References
    ----------

    Rehfeld, K., Marwan, N., Heitzig, J., and Kurths, J.: Comparison of correlation analysis
    techniques for irregularly sampled time series, Nonlin. Processes Geophys.,
    18, 389–404, https://doi.org/10.5194/npg-18-389-2011, 2011.
    '''

    if len(t) != len(y):
        raise ValueError('y and t must have the same length')
        
    # get the interpolation step if not provided
    if step is None:
        _, _, step = grid_properties(np.asarray(t), step_style = step_style)
        # Get the start and end point if not given
    if start is None:
        start = np.nanmin(np.asarray(t))
    if stop is None:
        stop = np.nanmax(np.asarray(t))
    
    # Get the uniform time axis.
    tc = np.arange(start,stop,step)
        

    kernel = lambda x, s : 1.0/(s*np.sqrt(2*np.pi))*np.exp(-0.5*(x/s)**2)  # define kernel function

    yc    = np.zeros((len(tc)))
    yc[:] = np.nan

    for i in range(len(tc)-1):
        xslice = t[(t>=tc[i])&(t<tc[i+1])]
        yslice = y[(t>=tc[i])&(t<tc[i+1])]

        if len(xslice)>0:
            d      = xslice-tc[i]
            weight = kernel(d,h)
            yc[i]  = sum(weight*yslice)/sum(weight)
        else:
            yc[i] = np.nan

    return tc, yc


def grid_properties(x,step_style='median'):
    ''' Establishes the grid properties of a numerical array:
        start, stop, and representative step.

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


def interp(x,y, interp_type='linear', step=None,start=None,stop=None, step_style= 'mean',**kwargs):
    """ Interpolate y onto a new x-axis

    Parameters
    ----------

    x : array
       The x-axis
    y : array
       The y-axis
    interp_type : {‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’}
        where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. Default is ‘linear’.
    step : float
            The interpolation step. Default is mean spacing between consecutive points.
    start : float
           where/when to start the interpolation. Default is min..
    stop : float
         where/when to stop the interpolation. Default is max.
    kwargs :  kwargs
        Aguments specific to interpolate.interp1D. See scipy for details https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
        If getting an error about extrapolation, you can use the arguments `bound_errors=False` and `fill_value="extrapolate"` to allow for extrapolation. 
    Returns
    -------

    xi : array
        The interpolated x-axis
    yi : array
        The interpolated y values
    """

        #Make sure x and y are numpy arrays
    x = np.array(x,dtype='float64')
    y = np.array(y,dtype='float64')

    # get the interpolation step if not available
    if step is None:
        _, _, step = grid_properties(np.asarray(x), step_style = step_style)

        # Get the start and end point if not given
    if start is None:
        start = np.nanmin(np.asarray(x))
    if stop is None:
        stop = np.nanmax(np.asarray(x))

    # Get the interpolated x-axis.
    xi = np.arange(start,stop,step)

    #Make sure the data is increasing
    data = pd.DataFrame({"x-axis": x, "y-axis": y}).sort_values('x-axis')

    # Add arguments

    yi = interpolate.interp1d(data['x-axis'],data['y-axis'],kind=interp_type,**kwargs)(xi)

    return xi, yi


def on_common_axis(x1, y1, x2, y2, method = 'interpolation', step=None, start=None, stop=None):
    """Places two timeseries on a common axis

    Note this function assumes that the time representation and units are the same (e.g., BP vs CE)

    Parameters
    ----------
    x1 : array
        x-axis values of the first timeseries
    y1 : array
        y-axis values of the first timeseries
    x2 : array
        x-axis values of the second timeseries
    y2 : array
        y-axis values of the second timeseries
    method : str
        Which method to use to get the timeseries on the same x axis.
        Valid entries: 'interpolation' (default, linear interpolation),
        'bin', 'None'. 'None' only cuts the timeseries to the common
        period but does not attempt to generate a common time axis
    step : float
        The interpolation step. Default is mean resolution
        of lowest resolution series
    start : float
        where/when to start. Default is the maximum of the minima of
        the two timeseries
    stop : float
        Where/when to stop. Default is the minimum of the maxima of
        the two timeseries

    Returns
    -------

    xi1, xi2 : array
        The interpolated x-axis
    interp_values1, interp_values2 : array
        the interpolated y-values
    """
    # make sure that x1, y1, x2, y2 are numpy arrays
    x1 = np.array(x1, dtype='float64')
    y1 = np.array(y1, dtype='float64')
    x2 = np.array(x2, dtype='float64')
    y2 = np.array(y2, dtype='float64')

    # Find the mean/max x-axis is not provided
    if start is None:
        start = np.nanmax([np.nanmin(x1), np.nanmin(x2)])
    if stop is None:
        stop = np.nanmin([np.nanmax(x1), np.nanmax(x2)])

    # Get the interp_step
    if step is None:
        step = np.nanmin([np.nanmean(np.diff(x1)), np.nanmean(np.diff(x2))])

    if method == 'interpolation':
    # perform the interpolation
        xi1, interp_values1 = interp(x1, y1, step=step, start=start,
                                stop=stop)
        xi2, interp_values2 = interp(x2, y2, step=step, start=start,
                                stop=stop)
    elif method == 'bin':
        xi1, interp_values1, _ , _ = bin(x1, y1, bin_size=step, start=start,
                                stop=stop)
        xi2, interp_values2, _ , _ = bin(x2, y2, bin_size=step, start=start,
                                stop=stop)
    elif method == None:
        min_idx1 = np.where(x1>=start)[0][0]
        min_idx2 = np.where(x2>=start)[0][0]
        max_idx1 = np.where(x1<=stop)[0][-1]
        max_idx2 = np.where(x2<=stop)[0][-1]

        xi1 = x1[min_idx1:max_idx1+1]
        xi2 = x2[min_idx2:max_idx2+1]
        interp_values1 = y1[min_idx1:max_idx1+1]
        interp_values2 = y2[min_idx2:max_idx2+1]

    else:
        raise KeyError('Not a valid interpolation method')

    return xi1, xi2, interp_values1, interp_values2


def standardize(x, scale=1, axis=0, ddof=0, eps=1e-3):
    """ Centers and normalizes a given time series. Constant or nearly constant time series not rescaled.

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

    1. Tapio Schneider's MATLAB code: https://github.com/tapios/RegEM/blob/master/standardize.m
    2. The zscore function in SciPy: https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py

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
        warnings.warn('Constant or nearly constant time series not rescaled.')
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
        vector of (real) numbers as a time series, NaNs allowed
    axis : int or None
        axis along which to operate, if None, compute over the whole array
        
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
        the factor that adjusts the threshold for gap detection

    Returns
    -------

    seg_ys : list
        a list of several segments with potentially different lengths
    seg_ts : list
        a list of the time axis of the several segments
    n_segs : int
        the number of segments
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


def gaussianize(X):
    """ Transforms a (proxy) timeseries to a Gaussian distribution.

    Originator: Michael Erb, Univ. of Southern California - April 2017

    Parameters
    ----------

    X : array
        Values for the timeseries.

    Returns
    -------

    Xn : array
        Gaussianized timseries

    """

    # Give every record at least one dimensions, or else the code will crash.
    X = np.atleast_1d(X)

    # Make a blank copy of the array, retaining the data type of the original data variable.
    Xn = copy.deepcopy(X)
    Xn[:] = np.NAN

    if len(X.shape) == 1:
        Xn = gaussianize_single(X)
    else:
        for i in range(X.shape[1]):
            Xn[:, i] = gaussianize_single(X[:, i])

    return Xn


def gaussianize_single(X_single):
    """ Transforms a single (proxy) timeseries to Gaussian distribution.

    Originator: Michael Erb, Univ. of Southern California - April 2017

    Parameters
    ----------

    X_single : 1D Array
        A single timeseries

    Returns
    -------

    Xn_single : Gaussianized values for a single timeseries.

    """
    # Count only elements with data.

    n = X_single[~np.isnan(X_single)].shape[0]

    # Create a blank copy of the array.
    Xn_single = copy.deepcopy(X_single)
    Xn_single[:] = np.NAN

    nz = np.logical_not(np.isnan(X_single))
    index = np.argsort(X_single[nz])
    rank = np.argsort(index)
    CDF = 1.*(rank+1)/(1.*n) - 1./(2*n)
    Xn_single[nz] = np.sqrt(2)*special.erfinv(2*CDF - 1)

    return Xn_single


def detrend(y, x=None, method="emd", n=1, sg_kwargs=None):
    """Detrend a timeseries according to four methods

    Detrending methods include, "linear", "constant", using a low-pass
        Savitzky-Golay filter, and using Empirical Mode Decomposition (default).

    Parameters
    ----------

    y : array
       The series to be detrended.
    x : array
       The time axis for the timeseries. Necessary for use with
       the Savitzky-Golay filters method since the series should be evenly spaced.
    method : str
        The type of detrending:
        - "linear": the result of a linear least-squares fit to y is subtracted from y.
        - "constant": only the mean of data is subtracted.
        - "savitzky-golay", y is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        - "emd" (default): Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
    n : int
        Works only if `method == 'emd'`. The number of smoothest modes to remove.
    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.

    Returns
    -------

    ys : array
        The detrended timeseries.

    See also
    --------

    pylecolim.utils.filter.savitzky_golay : Filtering using Savitzy-Golay

    pylecolim.utils.tsutils.preprocess : pre-processes a times series using standardization and detrending.

    """
    y = np.array(y)

    if x is not None:
        x = np.array(x)

    if method == "linear":
        ys = signal.detrend(y,type='linear')
    elif method == 'constant':
        ys = signal.detrend(y,type='constant')
    elif method == "savitzky-golay":
        # Check that the timeseries is uneven and interpolate if needed
        if x is None:
            raise ValueError("A time axis is needed for use with the Savitzky-Golay filter method")
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
        y_filt_x = np.interp(x,x_interp,y_filt)
        ys = y-y_filt_x
    elif method == "emd":
        imfs = EMD(y).decompose()
        if np.shape(imfs)[0] == 1:
            trend = np.zeros(np.size(y))
        else:
            # trend = imfs[-1]
            trend = np.sum(imfs[-n:], axis=0)  # remove the n smoothest modes

        ys = y - trend
    else:
        raise KeyError('Not a valid detrending method')

    return ys


def distance_neighbors(signal):
    '''Finds Distance of each point in the timeseries from its 4 nearest neighbors

       Parameters
       ----------

       signal : array
           The timeseries

       Returns
       -------
       distances : array
           Distance of each point from its nearest neighbors in decreasing order
    '''
    nn = NearestNeighbors(n_neighbors=4) # 4 nearest neighbors
    nbrs =nn.fit(signal.reshape(-1,1))
    distances, indices = nbrs.kneighbors(signal.reshape(-1,1))
    distances = sorted(distances[:,-1],reverse=True)
    return distances

def find_knee(distances):
    '''Finds knee point automatically in a given array sorted in decreasing order

       Parameters
       ----------

       distances : array
                  Distance of each point in the timeseries from it's nearest neighbors in decreasing order

       Returns
       -------

      knee : float
            knee point in the array
    '''
    nPoints = len(distances)
    allCoord = np.vstack((range(nPoints), distances)).T
    np.array([range(nPoints), distances])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    # scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    scalarProduct = np.sum(vecFromFirst * np.tile(lineVecNorm, (nPoints, 1)), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    knee = distances[idxOfBestPoint]
    return knee


def detect_outliers(ts, ys,auto=True, plot_knee=True,plot_outliers=True,
                    plot_outliers_kwargs=None,plot_knee_kwargs=None,
                    figsize=[10,4],saveknee_settings=None,
                    saveoutliers_settings=None,mute=False):
    ''' Function to detect outliers in the given timeseries

       for more details, see: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html


       Parameters
       ----------

       ts : array
            time axis of time series
       ys : array
            y values of time series
       plot : boolean
             true by default, plots the outliers using a scatter plot
       auto : boolean
             true by default, if false the user manually selects the knee point
       mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)
       plot_kwargs : dict
            keyword arguments for ax.plot()

       Returns
       -------

       outliers : array
                   a list of values consisting of outlier indices

       See also
       --------

       pylecolim.utils.tsutils.distance_neighbors : Finds Distance of each point in the timeseries from its 4 nearest neighbors

       pylecolim.utils.tsustils.find_knee : Finds knee point automatically in a given array sorted in decreasing order

       pylecolim.utils.tsutils.remove_outliers : Removes outliers from a timeseries

       '''
    #Take care of arguments for the knee plot
    saveknee_settings = {} if saveknee_settings is None else saveknee_settings.copy()

    try:
        minpts = math.log(len(ys))
        distances = distance_neighbors(ys)
        flag = all(v < 0.0001 for v in distances)

        knee_point = find_knee(distances)
        mark = distances.index(knee_point)
        index = [i for i in range(len(distances))]

        if auto == True:
            db = DBSCAN(eps=knee_point, min_samples=minpts)
            clusters = db.fit(ys.reshape(-1, 1))
            cluster_labels = clusters.labels_
            outliers = np.where(cluster_labels == -1)
            if plot_knee==True:
                fig1, ax1 = plt.subplots(figsize=figsize)
                if flag == True:
                    knee_point = 0.1
                ax1.annotate("knee={}".format(knee_point), (mark, knee_point),
                        arrowprops=dict(facecolor='black', shrink=0.05))
                plot_xy(index, distances,xlabel='Indices',ylabel='Distances',plot_kwargs=plot_knee_kwargs,ax=ax1)


        elif auto == False:
            plot_xy(index, distances, xlabel='Indices', ylabel='Distances',plot_kwargs=plot_knee_kwargs)
            eps = float(input('Enter the value for knee point'))
            if plot_knee==True:
                fig1,ax1 = plt.subplots(figsize=figsize)
                ax1.annotate("knee={}".format(eps), (mark, knee_point),
                        arrowprops=dict(facecolor='black', shrink=0.05))
                plot_xy(index, distances, xlabel='Indices', ylabel='Distances',plot_kwargs=plot_knee_kwargs,ax=ax1)

            db = DBSCAN(eps=eps, min_samples=minpts)
            clusters = db.fit(ys.reshape(-1, 1))
            cluster_labels = clusters.labels_
            outliers = np.where(cluster_labels == -1)

        if 'fig1' in locals():
            if 'path' in saveknee_settings:
                savefig(fig1, settings=saveknee_settings)
            # else:
            #     if not mute:
            #         showfig(fig1)

        if plot_outliers==True:
            x2 = ts[outliers]
            y2 = ys[outliers]
            plot_scatter_xy(ts,ys,x2,y2,figsize=figsize,xlabel='time',ylabel='value',savefig_settings=saveoutliers_settings,plot_kwargs=plot_outliers_kwargs, mute=mute)

        return outliers

    except ValueError:
        choice = input('Switch to Auto Mode(y/n)?')
        choice = choice.lower()
        if choice == 'y':
            a = detect_outliers(ts, ys, auto=True)
            return a
        else:
            exit(1)

def remove_outliers(ts,ys,outlier_points):
    ''' Removes outliers from a timeseries

    Parameters
    ----------

    ts : array
         x axis of timeseries
    ys : array
        y axis of timeseries
   outlier_points : array
                   indices of outlier points

    Returns
    -------
    ys : array
        y axis of timeseries
    ts : array
          x axis of timeseries

    See also
    --------

    pylecolim.utils.tsutils.detect_outliers : Function to detect outliers in the given timeseries
    '''

    ys = np.delete(ys,outlier_points)
    ts = np.delete(ts,outlier_points)

    return ys,ts

def eff_sample_size(y, detrend_flag=False):
    '''
    Effective Sample Size of timeseries y

    Parameters
    ----------
    y : float 
       1d array 
       
    detrend : boolean
        if True (default), detrends y before estimation.         

    Returns
    -------
    neff : float
        The effective sample size
        
        
    Reference
    ---------
    Thiébaux HJ, Zwiers FW. 1984. The interpretation and estimation of
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
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
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

    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay

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

