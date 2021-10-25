#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:15:52 2020

@author: deborahkhider

Handles filtering
"""

__all__ = [
    'butterworth',
    'savitzky_golay',
    'firwin',
    'lanczos'
]

import numpy as np
import statsmodels.api as sm
from scipy import signal

from .tsbase import (
    is_evenly_spaced
)

# ----
# Main functions
# ----

def savitzky_golay(ys, window_length=None, polyorder=2, deriv=0, delta=1,
                   axis=-1, mode='mirror', cval=0):
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
    
    Uses the implementation from scipy.signal: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

    Parameters
    ----------

    ys : array
        the values of the signal to be filtered
        
    window_length : int
        The length of the filter window. Must be a positive integer. 
            If mode is 'interp', window_length must be less than or equal to the size of ys. 
            Default is the size of ys.
    
    polyorder : int
        The order of the polynomial used to fit the samples. polyorder Must be less than window_length. 
            Default is 2
    
    deriv : int
        The order of the derivative to compute. 
            This must be a nonnegative integer. 
            The default is 0, which means to filter the data without differentiating
    
    delta : float
        The spacing of the samples to which the filter will be applied.
            This is only used if deriv>0.
            Default is 1.0
    
    axis : int
        The axis of the array ys along which the filter will be applied. Default is -1
    
    mode : str
        Must be ‘mirror’ (the default), ‘constant’, ‘nearest’, ‘wrap’ or ‘interp’. This determines the type of extension to use for the padded signal to which the filter is applied. When mode is ‘constant’, the padding value is given by cval. See the Notes for more details on ‘mirror’, ‘constant’, ‘wrap’, and ‘nearest’. When the ‘interp’ mode is selected, no extension is used. Instead, a degree polyorder polynomial is fit to the last window_length values of the edges, and this polynomial is used to evaluate the last window_length // 2 output values.
    
    cval : scalar
        Value to fill past the edges of the input if mode is ‘constant’. Default is 0.0.
    
    Returns
    -------

    yf : array
        ndarray of shape (N), the smoothed signal (or it's n-th derivative).

    References
    ----------

    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
        
    Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
        
    SciPy Cookbook: shttps://github.com/scipy/scipy-cookbook/blob/master/ipython/SavitzkyGolay.ipynb
    
    Notes
    -----
    
    Details on the mode option:
        
       - ‘mirror’: Repeats the values at the edges in reverse order. The value closest to the edge is not included.
       - ‘nearest’: The extension contains the nearest input value.
       - ‘constant’: The extension contains the value given by the cval argument.
       - ‘wrap’: The extension contains the values from the other end of the array. 
    """
    
    if window_length==None:
        window_length=int(np.ceil(len(ys))//2*2-1)
    elif type(window_length) is not int:
        raise TypeError("window_length should be of type int")
    
    if type(polyorder) is not int:
        raise TypeError("polyorder should be of type int")
    
    if window_length % 2 != 1 or window_length < 1:
        raise TypeError("window_length size must be a positive odd number")
    if window_length < polyorder + 2:
        raise TypeError("window_length is too small for the polynomial's order")
    
    yf=signal.savgol_filter(ys,window_length=window_length,
                            polyorder=polyorder,
                            deriv=deriv,
                            delta=delta,
                            axis=axis,
                            mode=mode,
                            cval=cval)

    return yf

def ts_pad(ys,ts,method = 'reflect', params=(1,0,0), reflect_type = 'odd',padFrac=0.1):
    """ Pad a timeseries based on timeseries model predictions

    Parameters
    ----------

    ys : numpy array
        Evenly-spaced timeseries
    ts : numpy array
        Time axis
    method : string
        The method to use to pad the series
        - ARIMA: uses a fitted ARIMA model
        - reflect (default): Reflects the time series around either end.
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
        augmented time axis
    """
    padLength =  np.round(len(ts)*padFrac).astype(np.int64)

    if is_evenly_spaced(ts)==False:
        raise ValueError("ts needs to be composed of even increments")
    else:
        dt = np.diff(ts)[0] # computp time interval
    
    #time axis
    tp = np.arange(ts[0]-padLength*dt,ts[-1]+padLength*dt+dt,dt)
    
    if method == 'ARIMA':
        # fit ARIMA model
        fwd_mod = sm.tsa.ARIMA(ys,params).fit()  # model with time going forward
        bwd_mod = sm.tsa.ARIMA(np.flip(ys,0),params).fit()  # model with time going backwards

        # predict forward & backward
        fwd_pred  = fwd_mod.forecast(padLength); yf = fwd_pred[0]
        bwd_pred  = bwd_mod.forecast(padLength); yb = np.flip(bwd_pred[0],0)

        # extend time series
        yp = np.empty(len(tp))
        yp[0:padLength]=yb
        yp[padLength:len(ts)+padLength]=ys
        yp[len(ts)+padLength:]=yf

    elif method == 'reflect':
        yp = np.pad(ys,(padLength,padLength),mode='reflect',reflect_type=reflect_type)

    else:
        raise ValueError('Not a valid argument. Enter "ARIMA" or "reflect"')

    return yp, tp


def butterworth(ys,fc,fs=1,filter_order=3,pad='reflect',
                reflect_type='odd',params=(1,0,0),padFrac=0.1):
    '''Applies a Butterworth filter with frequency fc, with padding
       Supports both lowpass and band-pass filtering.  

    Parameters
    ----------

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
        model parameters for ARIMA model (if pad = 'ARIMA')
    padFrac : float
        fraction of the series to be padded

    Returns
    -------

    yf : array
        filtered array
    
    See also
    --------
    
    pyleoclim.utils.filter.ts_pad : Pad a timeseries based on timeseries model predictions
    
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
        yp, tp = ts_pad(ys,ts,method = 'ARIMA', params=params, padFrac=padFrac)
    elif pad=='reflect':
        yp, tp = ts_pad(ys,ts,method = 'reflect', reflect_type=reflect_type, padFrac=padFrac)
    elif pad is None:
        yp = ys
        tp = ts
    else:
        raise ValueError('Not a valid argument. Enter "ARIMA", "reflect" or None')

    ypf = signal.filtfilt(b, a, yp)
    yf  = ypf[np.isin(tp,ts)]

    return yf

def lanczos(ys,fc,fs=1,pad='reflect',
                reflect_type='odd',params=(1,0,0),padFrac=0.1):
    '''Applies a Lanczos (lowpass) filter with frequency fc, with optional padding

    Parameters
    ----------

    ys : numpy array
        Timeseries
    fc : float
        cutoff frequency. 
    fs : float
        sampling frequency
    
    pad : string
        Indicates if padding is needed.
        - 'reflect': Reflects the timeseries
        - 'ARIMA': Uses an ARIMA model for the padding
        - None: No padding.
    params : tuple
        model parameters for ARIMA model (if pad = 'ARIMA'). May require fiddling.
    padFrac : float
        fraction of the series to be padded

    Returns
    -------

    yf : array
        filtered array
        
    References
    ----------
    Filter design from http://scitools.org.uk/iris/docs/v1.2/examples/graphics/SOI_filtering.html
    
    See also
    --------
    
    pyleoclim.utils.filter.ts_pad : Pad a timeseries based on timeseries model predictions
    
    '''
    ts = np.arange(len(ys)) # define "time" axis

    if pad=='ARIMA':
        yp, tp = ts_pad(ys,ts,method = 'ARIMA', params=params, padFrac=padFrac)
    elif pad=='reflect':
        yp, tp = ts_pad(ys,ts,method = 'reflect', reflect_type=reflect_type, padFrac=padFrac)
    elif pad is None:
        yp = ys
        tp = ts
    else:
        raise ValueError("Not a valid argument. Enter 'ARIMA', 'reflect' or None")

    window = max(51,len(yp)//4)  # arbitrary?

    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * fc / fs
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * fc / fs * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    wgts = w[1:-1]

    ypf = np.convolve(yp,wgts, 'same')
    yf  = ypf[np.isin(tp,ts)]

    return yf    


def firwin(ys, fc, numtaps=None, fs=1, pad='reflect', window='hamming', reflect_type='odd', params=(1,0,0), padFrac=0.1, **kwargs):
    '''Applies a Finite Impulse Response filter design with window method and frequency fc, with padding

    Parameters
    ----------

    ys : numpy array
        Timeseries
    fc : float or list
        cutoff frequency. If scalar, it is interpreted as a low-frequency cutoff (lowpass)
        If fc is a list,  it is interpreted as a frequency band (f1, f2), with f1 < f2 (bandpass)
    numptaps : int
        Length of the filter (number of coefficients, i.e. the filter order + 1). numtaps must be odd if a passband includes the Nyquist frequency.
        If None, will use the largest number that is smaller than 1/3 of the the data length.
    fs : float
        sampling frequency
    window : str or tuple of string and parameter values, optional
        Desired window to use. See scipy.signal.get_window for a list of windows and required parameters.
    pad : string
        Indicates if padding is needed.
        - 'reflect': Reflects the timeseries
        - 'ARIMA': Uses an ARIMA model for the padding
        - None: No padding.
    params : tuple
        model parameters for ARIMA model (if pad = True)
    padFrac : float
        fraction of the series to be padded
    kwargs : dict
        a dictionary of keyword arguments for scipy.signal.firwin

    Returns
    -------

    yf : array
        filtered array
    
    See also
    --------
    
    scipy.signal.firwin : FIR filter design using the window method
    
    '''
    # taps = signal.firwin(numtaps, fc, window=window, fs=fs, **kwargs)
    nyq = 0.5 * fs
    if np.isscalar(fc):
        pass_zero = 'lowpass'
    elif len(fc) == 2:
        pass_zero = 'bandpass'
    else:
        raise ValueError('Wrong input fc')

    if numtaps is None:
        # use the largest number of taps that the default padding method in scipy.signal.filtfilt allows
        numtaps = int(np.size(ys)//3)

    taps = signal.firwin(numtaps, fc/nyq, window=window, pass_zero=pass_zero, **kwargs)

    ts = np.arange(len(ys)) # define time axis

    if pad=='ARIMA':
        yp, tp = ts_pad(ys,ts,method = 'ARIMA', params=params, padFrac=padFrac)
    elif pad=='reflect':
        yp, tp = ts_pad(ys,ts,method = 'reflect', reflect_type=reflect_type, padFrac=padFrac)
    elif pad is None:
        yp = ys
        tp = ts
    else:
        raise ValueError('Not a valid argument. Enter "ARIMA", "reflect" or None')

    ypf = signal.filtfilt(taps, 1, yp)
    yf  = ypf[np.isin(tp,ts)]

    return yf

