#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:43:14 2020

@author: deborahkhider, fzhu

Utilities to manipulate timeseries
"""

__all__ = [
    'simple_stats',
    'bin_values',
    'interp',
    'on_common_axis',
    'standardize',
    'ts2segments',
    'clean_ts',
    'annualize',
    'gaussianize',
    'gaussianize_single',
    'detrend',
    'detect_outliers',
    'is_evenly_spaced',
    'remove_outliers'
]

import numpy as np
import pandas as pd
from scipy import interpolate
import warnings
import copy
from scipy import special
from scipy import signal
from pyhht import EMD
from sklearn.cluster import DBSCAN
#from matplotlib import cm
import matplotlib.pyplot as plt

import numpy.matlib
from sklearn.neighbors import NearestNeighbors
import math
from sys import exit
from .plotting import plot_scatter_xy,plot_xy,savefig,showfig
from .filter import savitzky_golay



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


def bin_values(x, y, bin_size=None, start=None, end=None):
    """ Bin the values

    Parameters
    ----------

    x : array
        The x-axis series.
    y : array
        The y-axis series.
    bin_size : float
        The size of the bins. Default is the average resolution
    start : float
        Where/when to start binning. Default is the minimum
    end : float
        When/where to stop binning. Default is the maximum

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

    # Get the bin_size if not available
    if bin_size is None:
        bin_size = np.nanmean(np.diff(x))

    # Get the start/end if not given
    if start is None:
        start = np.nanmin(x)
    if end is None:
        end = np.nanmax(x)

    # Set the bin medians
    bins = np.arange(start+bin_size/2, end + bin_size/2, bin_size)

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


def interp(x,y, interp_type='linear', interp_step=None,start=None,end=None, **args):
    """ Interpolation onto a new x-axis

    Parameters
    ----------

    x : array
       The x-axis
    y : array
       The y-axis
    interp_type : {‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’}
        where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. Default is ‘linear’.
    interp_step : float
            The interpolation step. Default is mean resolution.
    start : float
           where/when to start the interpolation. Default is min..
    end : float
         where/when to stop the interpolation. Default is max.
    args :  args
        Aguments specific to interpolate.interp1D. See scipy for details https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

    Returns
    -------

    xi : array
        The interpolated x-axis
    interp_values : array
        The interpolated values
    """

        #Make sure x and y are numpy arrays
    x = np.array(x,dtype='float64')
    y = np.array(y,dtype='float64')

        # get the interpolation step if not available
    if interp_step is None:
        interp_step = np.nanmean(np.diff(x))

        # Get the start and end point if not given
    if start is None:
        start = np.nanmin(np.asarray(x))
    if end is None:
        end = np.nanmax(np.asarray(x))

    # Get the interpolated x-axis.
    xi = np.arange(start,end,interp_step)

    #Make sure the data is increasing
    data = pd.DataFrame({"x-axis": x, "y-axis": y}).sort_values('x-axis')

    # Add arguments

    interp_values = interpolate.interp1d(data['x-axis'],data['y-axis'],kind=interp_type,**args)(xi)

    return xi, interp_values


def on_common_axis(x1, y1, x2, y2, method = 'interpolation', step=None, start=None, end=None):
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
    end : float
        Where/when to end. Default is the minimum of the maxima of
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
    if end is None:
        end = np.nanmin([np.nanmax(x1), np.nanmax(x2)])

    # Get the interp_step
    if step is None:
        step = np.nanmin([np.nanmean(np.diff(x1)), np.nanmean(np.diff(x2))])

    if method == 'interpolation':
    # perform the interpolation
        xi1, interp_values1 = interp(x1, y1, interp_step=step, start=start,
                                end=end)
        xi2, interp_values2 = interp(x2, y2, interp_step=step, start=start,
                                end=end)
    elif method == 'bin':
        xi1, interp_values1, n, error = bin_values(x1, y1, bin_size=step, start=start,
                                end=end)
        xi2, interp_values2, n, error = bin_values(x2, y2, bin_size=step, start=start,
                                end=end)
    elif method == None:
        min_idx1 = np.where(x1>=start)[0][0]
        min_idx2 = np.where(x2>=start)[0][0]
        max_idx1 = np.where(x1<=end)[0][-1]
        max_idx2 = np.where(x2<=end)[0][-1]

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

    1. Tapio Schneider's MATLAB code: http://www.clidyn.ethz.ch/imputation/standardize.m
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


def clean_ts(ys, ts):
    ''' Delete the NaNs in the time series and sort it with time axis ascending

    Parameters
    ----------
    ys : array
        A time series, NaNs allowed
    ts : array
        The time axis of the time series, NaNs allowed

    Returns
    -------
    ys : array
        The time series without nans
    ts : array
        The time axis of the time series without nans

    '''
    ys = np.asarray(ys, dtype=np.float)
    ts = np.asarray(ts, dtype=np.float)
    assert ys.size == ts.size, 'The size of time axis and data value should be equal!'

    ys_tmp = np.copy(ys)
    ys = ys[~np.isnan(ys_tmp)]
    ts = ts[~np.isnan(ys_tmp)]
    ts_tmp = np.copy(ts)
    ys = ys[~np.isnan(ts_tmp)]
    ts = ts[~np.isnan(ts_tmp)]

    # sort the time series so that the time axis will be ascending
    sort_ind = np.argsort(ts)
    ys = ys[sort_ind]
    ts = ts[sort_ind]

    return ys, ts


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


def detrend(y, x = None, method = "emd", sg_kwargs = None):
    """Detrend a timeseries according to four methods

    Detrending methods include, "linear", "constant", using a low-pass
        Savitzky-Golay filter, and using eigen mode decomposition (default).

    Parameters
    ----------

    y : array
       The series to be detrended.
    x : array
       The time axis for the timeseries. Necessary for use with
       the Savitzky-Golay filters method since the series should be evenly spaced.
    method : str
        The type of detrending:
        - linear: the result of a linear least-squares fit to y is subtracted from y.
        - constant: only the mean of data is subtrated.
        - "savitzky-golay", y is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        - "emd" (default): Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
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
            trend = imfs[-1]
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
    nn = NearestNeighbors(4) # 4 nearest neighbors
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
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
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
            else:
                if not mute:
                    showfig(fig1)

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

def is_evenly_spaced(ts):
    ''' Check if a time axis is evenly spaced.

    Parameters
    ----------

    ts : array
        the time axis of a time series

    Returns
    -------

    check : bool
        True - evenly spaced; False - unevenly spaced.

    '''
    if ts is None:
        check = True
    else:
        dts = np.diff(ts)
        dt_mean = np.mean(dts)
        if any(dt == dt_mean for dt in np.diff(ts)):
            check = True
        else:
            check = False

    return check


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
