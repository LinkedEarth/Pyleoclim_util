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


def binvalues(x, y, bin_size=None, start=None, end=None):
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

    return bins, binned_values, n, error

def interp(x,y,interp_step=None,start=None,end=None):
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

    interp_values = np.interp(xi,data['x-axis'],data['y-axis'])

    return xi, interp_values


def onCommonAxis(x1, y1, x2, y2, method = 'interpolation', step=None, start=None, end=None):
    """Places two timeseries on a common axis

    Args:
        x1 (array): x-axis values of the first timeseries
        y1 (array): y-axis values of the first timeseries
        x2 (array): x-axis values of the second timeseries
        y2 (array): y-axis values of the second timeseries
        method (str): Which method to use to get the timeseries on the same x axis.
            Valid entries: 'interpolation' (default), 'bin'
        step (float): The interpolation step. Default is mean resolution
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
    
    #make sure the method is correct
    method_list = ['interpolation', 'bin']
    assert method in method_list, 'Invalid method.'
    
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
        xi, interp_values1 = interp(x1, y1, interp_step=step, start=start,
                                end=end)
        xi, interp_values2 = interp(x2, y2, interp_step=step, start=start,
                                end=end)
    elif method == 'bin':
        xi, interp_values1 = binvalues(x1, y1, bin_size=step, start=start,
                                end=end)
        xi, interp_values2 = binvalues(x2, y2, bin_size=step, start=start,
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

#def liang_causality(xx1, xx2, p=1):
#    """
#    Estimate T21, the Liang information transfer from series x2 to series x1
#    dt is taken to be 1.
#
#    Args:
#        x1, x2 (array) - vectors of (real) numbers with identical length, no NaNs allowed
#        p (int >=1 ) -  time advance in performing Euler forward differencing, e.g., 1, 2. Unless the series are generated \
#                        with a highly chaotic deterministic system, p=1 should be used.
#
#    Returns:
#        T21 - info flow from X2 to X1	(Note: Not X1 -> X2!)
#        err90(float) - standard error at 90% significance level
#        err95(float) - standard error at 95% significance level
#        err99(float) - standard error at 99% significance level
#
#    References:
#        - Liang, X.S. (2013) The Liang-Kleeman Information Flow: Theory and
#            Applications. Entropy, 15, 327-360, doi:10.3390/e15010327
#        - Liang, X.S. (2014) Unraveling the cause-efect relation between timeseries.
#            Physical review, E 90, 052150
#        - Liang, X.S. (2015) Normalizing the causality between time series.
#            Physical review, E 92, 022126
#        - Liang, X.S. (2016) Information flow and causality as rigorous notions ab initio.
#            Physical review, E 94, 052201
#    """
#    dt = 1
#
#    x1 = xx1[:len(xx1)-p, ]
#    dx1 = (xx1[p:,] - x1) / (p * dt)
#
#    x2 = xx2[:len(xx2)-p, ]
#    dx2 = (xx2[p:, ] - x2) / (p * dt)
#
#    N = len(xx1) - p
#
#    C = np.cov(x1, x2)
#
#    x1_normalized = x1 - np.mean(x1)
#    x2_normalized = x2 - np.mean(x2)
#    dx1_normalized = dx1 - np.mean(dx1)
#    dx2_normalized = dx2 - np.mean(dx2)
#
#    dC = np.matrix([[np.sum(np.multiply(x1_normalized, dx1_normalized)), np.sum(np.multiply(x1_normalized, dx2_normalized))],\
#        [np.sum(np.multiply(x2_normalized, dx1_normalized)), np.sum(np.multiply(x2_normalized, dx2_normalized))]]) / (N - 1)
#
#    C_infty = C
#    det_C = np.linalg.det(C)
#
#
#    a11 = C[1,1] * dC[0,0] - C[0,1] * dC[1,0]
#    a12 = -C[0,1] * dC[0,0] + C[0,0] * dC[1,0]
#
#    a11 = a11 / det_C
#    a12 = a12 / det_C
#
#    f1 = np.mean(dx1) - a11 * np.mean(x1) - a12 * np.mean(x2)
#    R1 = dx1 - (f1 + a11*x1 + a12*x2)
#    Q1 = np.sum(np.multiply(R1,R1))
#    b1 = np.sqrt(Q1 * dt / N)
#
#    #print(f1, R1,Q1,b1)
#
#    # covariance matrix of the estimation of (f1, a11, a12, b1)
#    NI = np.matrix(np.zeros((4, 4)))
#    NI[0,0] = N * dt / b1/b1
#    NI[1,1] = dt/b1/b1 * np.sum(np.multiply(x1, x1))
#    NI[2,2] = dt/b1/b1 * np.sum(np.multiply(x2, x2))
#    NI[3,3] = 3*dt/np.power(b1, 4) * np.sum(np.multiply(R1, R1)) - N/b1/b1
#    NI[0,1] = dt/b1/b1 * np.sum(x1)
#    NI[0,2] = dt/b1/b1 * np.sum(x2)
#    NI[0,3] = 2*dt/np.power(b1,3) * np.sum(R1)
#    NI[1,2] = dt/b1/b1 * np.sum(np.multiply(x1, x2))
#    NI[1,3] = 2*dt/np.power(b1, 3) * np.sum(np.multiply(R1, x1))
#    NI[2,3] = 2*dt/np.power(b1, 3) * np.sum(np.multiply(R1, x2))
#
#    NI[1,0] = NI[0,1]
#    NI[2,0] = NI[0,2]
#    NI[2,1] = NI[1,2]
#    NI[3,0] = NI[0,3]
#    NI[3,1] = NI[1,3]
#    NI[3,2] = NI[2,3]
#
#    invNI = np.linalg.inv(NI)
#    var_a12 = invNI[2,2]
#
#    T21 = C_infty[0,1]/C_infty[0,0] * (-C[1,0]*dC[0,0] + C[0,0]*dC[1,0]) / det_C
#    var_T21 = np.power((C_infty[0,1]/C_infty[0,0]),2) * var_a12
#
## From the standard normal distribution table,
## significance level alpha=95%, z=1.96
##		           99%, z=2.56
##			   90%, z=1.65
#    z99 = 2.56
#    z95 = 1.96
#    z90 = 1.65
#    err90 = np.sqrt(var_T21) * z90
#    err95 = np.sqrt(var_T21) * z95
#    err99 = np.sqrt(var_T21) * z99
#    return T21, err90, err95, err99
