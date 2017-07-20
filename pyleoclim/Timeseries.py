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
        end (float): where/when to stop the interpolation. Defaul is max.

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

    mu2 = np.copy(mu)  # the mean used in the calculation of zscore
    sig2 = np.copy(sig) / scale  # the std used in the calculation of zscore

    if np.any(np.abs(sig) < eps):  # check if x contains (nearly) constant time series
        warnings.warn('Constant or nearly constant time series not rescaled.')
        where_const = np.where(np.abs(sig) < eps)  # find out where we have (nearly) constant time series

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
    # delete the NaNs if there is any
    ys_tmp = np.copy(ys)
    ys = ys[~np.isnan(ys_tmp)]
    ts = ts[~np.isnan(ys_tmp)]
    ts_tmp = np.copy(ts)
    ys = ys[~np.isnan(ts_tmp)]
    ts = ts[~np.isnan(ts_tmp)]

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
