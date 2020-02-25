#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:15:52 2020

@author: deborahkhider

Handles filtering
"""

import numpy as np
from math import factorial
import statsmodels.api as sm
from scipy import signal

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """ Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Args
    ----

    y : array
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering. Must be less then `window_size` - 1.
    deriv : int
        the order of the derivative to compute (default = 0 means only smoothing)

    Returns
    -------

    ys : array
        ndarray of shape (N), the smoothed signal (or it's n-th derivative).

    Reference
    ---------

    - A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
    - Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
    - SciPy Cookbook: shttps://github.com/scipy/scipy-cookbook/blob/master/ipython/SavitzkyGolay.ipynb
    """
    if type(window_size) is not int:
        raise TypeError("window_size should be of type int")
    if type(order) is not int:
        raise TypeError("order should be of type int")
    # Check window size and order
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size-1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], y, mode='valid')

def ts_pad(ys,ts,method = 'reflect', params=(1,0,0), reflect_type = 'odd',padFrac=0.1):
    """ Pad a timeseries based on timeseries model predictions

    Args
    ----

    x : numpy array
        Evenly-spaced timeseries
    t : numpy array
        Time axis
    method : string
        The method to use to pad the series
        - ARIMA: uses a fitted ARIMA model
        - reflect (default): Reflects the time series
    params : tuple ARIMA model order parameters (p,d,q), Default corresponds to an AR(1) model
    reflect_type : string
         {‘even’, ‘odd’}, optional
         Used in ‘reflect’, and ‘symmetric’. The ‘even’ style is the default with an unaltered reflection around the edge value.
         For the ‘odd’ style, the extented part of the array is created by subtracting the reflected values from two times the edge value.
         For more details, see np.lib.pad()
    padFrac : float
        padding fraction (scalar) such that padLength = padFrac*length(series)

    Returns
    -------

    yp : array
        padded timeseries
    tp : array
        augmented axis

    Author
    ------

    Julien Emile-Geay, Deborah Khider
    """
    padLength =  np.round(len(ts)*padFrac).astype(np.int64)

    if not (np.std(np.diff(ts)) == 0):
        raise ValueError("ts needs to be composed of even increments")
    else:
        dt = np.diff(ts)[0] # computp time interval

    if method == 'ARIMA':
        # fit ARIMA model
        fwd_mod = sm.tsa.ARIMA(ys,params).fit()  # model with time going forward
        bwd_mod = sm.tsa.ARIMA(np.flip(ys,0),params).fit()  # model with time going backwards

        # predict forward & backward
        fwd_pred  = fwd_mod.forecast(padLength); yf = fwd_pred[0]
        bwd_pred  = bwd_mod.forecast(padLength); yb = np.flip(bwd_pred[0],0)

        # define extra time axes
        tf = np.linspace(max(ts)+dt, max(ts)+padLength*dt,padLength)
        tb = np.linspace(min(ts)-padLength*dt, min(ts)-1, padLength)

        # extend time series
        tp = np.arange(ts[0]-padLength*dt,ts[-1]+padLength*dt+1,dt)
        yp = np.empty(len(tp))
        yp[np.isin(tp,ts)] =ys
        yp[np.isin(tp,tb)]=yb
        yp[np.isin(tp,tf)]=yf

    elif method == 'reflect':
        yp = np.pad(ys,(padLength,padLength),mode='reflect',reflect_type=reflect_type)
        tp = np.arange(ts[0]-padLength,ts[-1]+padLength+1,1)

    else:
        raise ValueError('Not a valid argument. Enter "ARIMA" or "reflect"')

    return yp, tp


def butterworth(ys,fc,fs=1,filter_order=3,pad='reflect',
                reflect_type='odd',params=(2,1,2),padFrac=0.1):
    '''Applies a Butterworth filter with frequency fc, with padding

    Args
    ----

    ys : numpy array
        Timeseries
    fc : float or list
        cutoff frequency. If scalar, it is interpreted as a low-frequency cutoff (lowpass)
        If fc is a list,  it is interpreted as a frequency band (f1, f2), with f1 < f2 (bandpass)
    fs : float
        sampling frequency
    filter_order : int
        order n of Butterworth filter
    pad : string
        Indicates if padding is needed.
        - 'reflect': Reflects the timeseries
        - 'ARIMA': Uses an ARIMA model for the padding
        - None: No padding.
    params : tuple
        model parameters for ARIMA model (if pad = True)
    padFrac : float
        fraction of the series to be padded

    Returns
    -------

    yf : array
        filtered array

    Author
    ------

    Julien Emile-Geay
    '''
    nyq = 0.5 * fs

    if isinstance(fc, list) and len(fc) == 2:
        fl = fc[0] / nyq
        fh = fc[1] / nyq
        b, a = signal.butter(filter_order, [fl, fh], btype='bandpass')
    else:
        fl = fc / nyq
        b, a = signal.butter(filter_order, fl , btype='lowpass')

    ts = np.arange(len(ys)) # define time axis

    if pad=='ARIMA':
        yp,tp = ts_pad(ys,ts,method = 'ARIMA', params=params, padFrac=padFrac)
    elif pad=='reflect':
        yp,tp = ts_pad(ys,ts,method = 'reflect', reflect_type=reflect_type, padFrac=padFrac)
    elif pad is None:
        yp = ys; tp = ts
    else:
        raise ValueError('Not a valid argument. Enter "ARIMA", "reflect" or None')

    ypf = signal.filtfilt(b, a, yp)
    yf  = ypf[np.isin(tp,ts)]

    return yf