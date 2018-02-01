#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:08:39 2017

@author: deborahkhider

Basic manipulation of timeseries for the pyleoclim module
"""

import numpy as np
import pandas as pd
import warnings
import copy
from scipy import special
import sys
from scipy import signal

from pyleoclim import Spectral


def bin(x, y, bin_size="", start="", end=""):
    """ Bin the values

    Args:
        x (array): the x-axis series.
        y (array): the y-axis series.
        bin_size (float): The size of the bins. Default is the average resolution
        start (float): Where/when to start binning. Default is the minimum
        end (float): When/where to stop binning. Defulat is the maximum

    Returns:
        binned_values - the binned output \n
        bins - the bins (centered on the median, i.e., the 100-200 bin is 150) \n
        n - number of data points in each bin \n
        error -  the standard error on the mean in each bin

    """

    # Make sure x and y are numpy arrays
    x = np.array(x, dtype='float64')
    y = np.array(y, dtype='float64')

    # Get the bin_size if not available
    if not bin_size:
        bin_size = np.nanmean(np.diff(x))

    # Get the start/end if not given
    if type(start) is str:
        start = np.nanmin(x)
        if type(end) is str:
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

    return bins, binned_values, n, error

def interp(x,y,interp_step="",start="",end=""):
    """ Linear interpolation onto a new x-axis

    Args:
        x (array): the x-axis
        y (array): the y-axis
        interp_step (float): the interpolation step. Default is mean resolution.
        start (float): where/when to start the interpolation. Default is min..
        end (float): where/when to stop the interpolation. Default is max.

    Returns:
        xi - the interpolated x-axis \n
        interp_values - the interpolated values
        """

    #Make sure x and y are numpy arrays
    x = np.array(x,dtype='float64')
    y = np.array(y,dtype='float64')

    # get the interpolation step if not available
    if not interp_step:
        interp_step = np.nanmean(np.diff(x))

    # Get the start and end point if not given
    if type(start) is str:
        start = np.nanmin(np.asarray(x))
    if type(end) is str:
        end = np.nanmax(np.asarray(x))

    # Get the interpolated x-axis.
    xi = np.arange(start,end,interp_step)

    #Make sure the data is increasing
    data = pd.DataFrame({"x-axis": x, "y-axis": y}).sort_values('x-axis')

    interp_values = np.interp(xi,data['x-axis'],data['y-axis'])

    return xi, interp_values


def onCommonAxis(x1, y1, x2, y2, interp_step="", start="", end=""):
    """Places two timeseries on a common axis

    Args:
        x1 (array): x-axis values of the first timeseries
        y1 (array): y-axis values of the first timeseries
        x2 (array): x-axis values of the second timeseries
        y2 (array): y-axis values of the second timeseries
        interp_step (float): The interpolation step. Default is mean resolution
        of lowest resolution series
        start (float): where/when to start. Default is the maximum of the minima of
        the two timeseries
        end (float): Where/when to end. Default is the minimum of the maxima of
        the two timeseries

    Returns:
        xi -  the interpolated x-axis \n
        interp_values1 -  the interpolated y-values for the first timeseries
        interp_values2 - the intespolated y-values for the second timeseries
        """

    # make sure that x1, y1, x2, y2 are numpy arrays
    x1 = np.array(x1, dtype='float64')
    y1 = np.array(y1, dtype='float64')
    x2 = np.array(x2, dtype='float64')
    y2 = np.array(y2, dtype='float64')

    # Find the mean/max x-axis is not provided
    if type(start) is str:
        start = np.nanmax([np.nanmin(x1), np.nanmin(x2)])
        if type(end) is str:
            end = np.nanmin([np.nanmax(x1), np.nanmax(x2)])

    # Get the interp_step
    if not interp_step:
        interp_step = np.nanmin([np.nanmean(np.diff(x1)), np.nanmean(np.diff(x2))])

    # perform the interpolation
    xi, interp_values1 = interp(x1, y1, interp_step=interp_step, start=start,
                                end=end)
    xi, interp_values2 = interp(x2, y2, interp_step=interp_step, start=start,
                                end=end)

    return xi, interp_values1, interp_values2


def standardize(x, scale=1, axis=0, ddof=0, eps=1e-3):
    """ Centers and normalizes a given time series. Constant or nearly constant time series not rescaled.

    Args:
        x (array): vector of (real) numbers as a time series, NaNs allowed
        scale (real): a scale factor used to scale a record to a match a given variance
        axis (int or None): axis along which to operate, if None, compute over the whole array
        ddof (int): degress of freedom correction in the calculation of the standard deviation
        eps (real): a threshold to determine if the standard deviation is too close to zero

    Returns:
        z (array): the standardized time series (z-score), Z = (X - E[X])/std(X)*scale, NaNs allowed
        mu (real): the mean of the original time series, E[X]
        sig (real): the standard deviation of the original time series, std[X]

    References:
        1. Tapio Schneider's MATLAB code: http://www.clidyn.ethz.ch/imputation/standardize.m
        2. The zscore function in SciPy: https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py

    @author: fzhu
    """
    x = np.asanyarray(x)
    assert x.ndim <= 2, 'The time series x should be a vector or 2-D array!'

    mu = np.nanmean(x, axis=axis)  # the mean of the original time series
    sig = np.nanstd(x, axis=axis, ddof=ddof)  # the std of the original time series

    mu2 = np.asarray(np.copy(mu))  # the mean used in the calculation of zscore
    sig2 = np.asarray(np.copy(sig) / scale)  # the std used in the calculation of zscore

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

    Args:
        ys (array): a time series, NaNs allowed
        ts (array): the time points
        factor (float): the factor that adjusts the threshold for gap detection

    Returns:
        seg_ys (list): a list of several segments with potentially different lengths
        seg_ts (list): a list of the time axis of the several segments
        n_segs (int): the number of segments

    @author: fzhu
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

    Args:
        ys (array): a time series, NaNs allowed
        ts (array): the time axis of the time series, NaNs allowed

    Returns:
        ys (array): the time series without nans
        ts (array): the time axis of the time series without nans

    '''
    # delete NaNs if there is any
    ys = np.asarray(ys, dtype=np.float)
    ts = np.asarray(ts, dtype=np.float)
    assert(ys.size == ts.size, 'The size of time axis and data value should be equal!')

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

    Args:
        ys (array): a time series, NaNs allowed
        ts (array): the time axis of the time series, NaNs allowed

    Returns:
        ys_ann (array): the annualized time series
        year_int (array): the time axis of the annualized time series

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
    """ Transforms a (proxy) timeseries to Gaussian distribution.

    Originator: Michael Erb, Univ. of Southern California - April 2017
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


def detrend(y, x = None, method = "linear", params = ["default",4,0,1]):
    """Detrend a timeseries according to three methods

    Detrending methods include, "linear" (default), "constant", and using a low-pass
        Savitzky-Golay filters.

    Args:
        y (array): The series to be detrended.
        x (array): The time axis for the timeseries. Necessary for use with
            the Savitzky-Golay filters method since the series should be evenly spaced.
        method (str): The type of detrending. If linear (default), the result of
            a linear least-squares fit to y is subtracted from y. If constant,
            only the mean of data is subtrated. If "savitzy-golay", y is filtered
            using the Savitzky-Golay filters and the resulting filtered series
            is subtracted from y.
        params (list): The paramters for the Savitzky-Golay filters. The first parameter
            corresponds to the window size (default it set to half of the data)
            while the second parameter correspond to the order of the filter
            (default is 4). The third parameter is the order of the derivative
            (the default is zero, which means only smoothing.)

    Returns:
        ys (array) - the detrended timeseries.
    """
    option = ["linear", "constant", "savitzy-golay"]
    if method not in option:
        sys.exit("The selected method is not currently supported")

    if method == "linear":
        ys = signal.detrend(y,type='linear')
    if method == 'constant':
        ys = signal.detrend(y,type='constant')
    else:
        # Check that the timeseries is uneven and interpolate if needed
        if x is None:
            sys.exit("A time axis is needed for use with the Savitzky-Golay filters method")
        # Check whether the timeseries is unvenly-spaced and interpolate if needed
        if len(np.unique(np.diff(x)))>1:
            warnings.warn("Timeseries is not evenly-spaced, interpolating...")
            interp_step = np.nanmean(np.diff(x))
            start = np.nanmin(x)
            end = np.nanmax(x)
            x_interp, y_interp = interp(x,y,interp_step=interp_step,\
                                             start=start,end=end)
        else:
            x_interp = x
            y_interp = y
        if params[0] == "default":
            l = len(y) # Use the length of the timeseries for the window side
            l = np.ceil(l)//2*2+1 # Make sure this is an odd number
            l = int(l) # Make sure that the type is int
            o = int(params[1]) # Make sure the order is type int
            d = int(params[2])
            e = int(params[3])
        else:
            #Assume the users know what s/he is doing and just force to type int
            l = int(params[0])
            o = int(params[1])
            d = int(params[2])
            e = int(params[3])
        # Now filter
        y_filt = Spectral.Filter.savitzky_golay(y_interp,l,o,d,e)
        # Put it all back on the original x axis
        y_filt_x = np.interp(x,x_interp,y_filt)
        ys = y-y_filt_x

    return ys
