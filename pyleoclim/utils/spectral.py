#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:23:29 2020

@author: deborahkhider

Sectral analysis functions
"""

import numpy as np
from scipy import signal
import nitime.algorithms as nialg

from .wavelet import preprocess, is_evenly_spaced, make_freq_vector
from .tsutils import clean_ts, interp, bin_values

#-----------
#Wrapper
#-----------

#---------
#Main functions
#---------


def welch(ys, ts, ana_args={}, prep_args={}, interp_method='interp', interp_args={}):
    '''
    Args
    ----

    ys : array
        a time series
    ts  array
        time axis of the time series
    ana_args : dict
        the arguments for spectral analysis with periodogram, including
        - window (str): Desired window to use. See get_window for a list of windows and required parameters. If window is an array it will be used directly as the window. Defaults to None; equivalent to ‘boxcar’.
        - nfft (int): length of the FFT used. If None the length of x will be used.
        - return_onesided (bool): If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Note that for complex data, a two-sided spectrum is always returned.
        - nperseg (int): Length of each segment. Defaults to None, but if window is str or tuple, is set to 256, and if window is array_like, is set to the length of the window.
        - noverlap (int): Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
        - scaling (str, {'density', 'spectrum'}): Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz if x is measured in V and computing the power spectrum (‘spectrum’) where Pxx has units of V**2 if x is measured in V. Defaults to ‘density’
        - axis (int):     Axis along which the periodogram is computed; the default is over the last axis (i.e. axis=-1).
        - average : { ‘mean’, ‘median’ }, optional
        see https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.signal.welch.html for details
    interp_method : string
        {'interp', 'bin'}): perform interpolation or binning
    interp_args : dict
        the arguments for the interpolation or binning methods, for the details, check interp() and binvalues()
    prep_args : dict
        the arguments for preprocess, including
        - detrend (str): 'none' - the original time series is assumed to have no trend;
                         'linear' - a linear least-squares fit to `ys` is subtracted;
                         'constant' - the mean of `ys` is subtracted
                         'savitzy-golay' - ys is filtered using the Savitzky-Golay
                             filters and the resulting filtered series is subtracted from y.
                         'hht' - detrending with Hilbert-Huang Transform
        - params (list): The paramters for the Savitzky-Golay filters. The first parameter
                         corresponds to the window size (default it set to half of the data)
                         while the second parameter correspond to the order of the filter
                         (default is 4). The third parameter is the order of the derivative
                         (the default is zero, which means only smoothing.)
        - gaussianize (bool): If True, gaussianizes the timeseries
        - standardize (bool): If True, standardizes the timeseries

    Returns
    -------
    res_dict : dict
        the result dictionary, including
        - freq (array): the frequency vector
        - psd (array): the spectral density vector

    '''
    #make default nperseg len(ts)//3
    if not ana_args or not ana_args.get('nperseg'):
        ana_args['nperseg']=len(ts)

    # preprocessing
    
    ys, ts = clean_ts(ys, ts)
    ys = preprocess(ys, ts, **prep_args)

    # if data is not evenly spaced, interpolate
    if not is_evenly_spaced(ts):
        interp_func = {
            'interp': interp,
            'bin': bin_values,
        }
        ts, ys = interp_func[interp_method](ts, ys, **interp_args)

    # calculate sampling frequency fs
    dt = np.median(np.diff(ts))
    fs = 1 / dt

    # spectral analysis with scipy welch
    freq, psd = signal.welch(ys, fs, **ana_args)

    # fix zero frequency point
    if freq[0] == 0:
        psd[0] = np.nan

    # output result
    res_dict = {
        'freq': np.asarray(freq),
        'psd' : np.asarray(psd),
    }

    return res_dict


def mtm(ys, ts, NW=2.5, ana_args={}, prep_args={}, interp_method='interp', interp_args={}):
    #  ''' Call MTM from the package [spectrum](https://github.com/cokelaer/spectrum)
    ''' Call MTM from the package [nitime](http://nipy.org)

    Args
    ----

    ys : array
        a time series
    ts : array
        time axis of the time series
    ana_args : dict
        the arguments for spectral analysis with periodogram, including
        - window (str): Desired window to use. See get_window for a list of windows and required parameters. If window is an array it will be used directly as the window. Defaults to None; equivalent to ‘boxcar’.
        - nfft (int): length of the FFT used. If None the length of x will be used.
        - return_onesided (bool): If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Note that for complex data, a two-sided spectrum is always returned.
        - scaling (str, {'density', 'spectrum'}): Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz if x is measured in V and computing the power spectrum (‘spectrum’) where Pxx has units of V**2 if x is measured in V. Defaults to ‘density’
        see https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.periodogram.html for the details
    interp_method : string
        {'interp', 'bin'}): perform interpolation or binning
    interp_args :dict
        the arguments for the interpolation or binning methods, for the details, check interp() and binvalues()
    prep_args : dict
        the arguments for preprocess, including
        - detrend (str): 'none' - the original time series is assumed to have no trend;
                         'linear' - a linear least-squares fit to `ys` is subtracted;
                         'constant' - the mean of `ys` is subtracted
                         'savitzy-golay' - ys is filtered using the Savitzky-Golay
                             filters and the resulting filtered series is subtracted from y.
                         'hht' - detrending with Hilbert-Huang Transform
        - params (list): The paramters for the Savitzky-Golay filters. The first parameter
                         corresponds to the window size (default it set to half of the data)
                         while the second parameter correspond to the order of the filter
                         (default is 4). The third parameter is the order of the derivative
                         (the default is zero, which means only smoothing.)
        - gaussianize (bool): If True, gaussianizes the timeseries
        - standardize (bool): If True, standardizes the timeseries

    Returns
    -------

    res_dict : dict
        the result dictionary, including
        - freq (array): the frequency vector
        - psd (array): the spectral density vector


    '''
    # preprocessing
    
    ys, ts = clean_ts(ys, ts)
    ys = preprocess(ys, ts, **prep_args)

    # interpolate if not evenly-spaced
    if not is_evenly_spaced(ts):
        interp_func = {
            'interp': interp,
            'bin': bin_values,
        }
        ts, ys = interp_func[interp_method](ts, ys, **interp_args)

    # calculate sampling frequency
    dt = np.median(np.diff(ts))
    fs = 1 / dt

    # spectral analysis
    #  res = spectrum.MultiTapering(ys, sampling=fs, NW=NW, **ana_args)
    #  freq = res.frequencies()
    #  psd = res.psd
    freq, psd, nu = nialg.multi_taper_psd(ys, Fs=fs, NW=NW, **ana_args)  # call nitime func

    # fix the zero frequency point
    if freq[0] == 0:
        psd[0] = np.nan

    # output result
    res_dict = {
        'freq': np.asarray(freq),
        'psd': np.asarray(psd),
    }

    return res_dict


def lomb_scargle(ys, ts, freq=None, make_freq_method='nfft', prep_args={}, ana_args={}):
    """ Return the computed periodogram using lomb-scargle algorithm
    Lombscargle algorithm

    Args
    ----

    ys : array
        a time series
    ts : array
        time axis of the time series
    freq : array
        vector of frequency
    make_freq_method : string
        Method to be used to make the time series. Default is nfft
    prep_args : dict
                the arguments for preprocess, including
                - detrend (str): 'none' - the original time series is assumed to have no trend;
                                 'linear' - a linear least-squares fit to `ys` is subtracted;
                                 'constant' - the mean of `ys` is subtracted
                                 'savitzy-golay' - ys is filtered using the Savitzky-Golay
                                     filters and the resulting filtered series is subtracted from y.
                                 'hht' - detrending with Hilbert-Huang Transform
                - params (list): The paramters for the Savitzky-Golay filters. The first parameter
                                 corresponds to the window size (default it set to half of the data)
                                 while the second parameter correspond to the order of the filter
                                 (default is 4). The third parameter is the order of the derivative
                                 (the default is zero, which means only smoothing.)
                - gaussianize (bool): If True, gaussianizes the timeseries
                - standardize (bool): If True, standardizes the timeseries
    ana_args : dict
        Extra argumemnts which may be needed such as
        - precenter (bool): Pre-center amplitudes by subtracting the mean
        - normalize (bool): Compute normalized periodogram

    Returns
    -------

    res_dict : dict
        the result dictionary, including
        - freq (array): the frequency vector
        - psd (array): the spectral density vector
    """
    ys, ts = clean_ts(ys, ts)
    ys = preprocess(ys, ts, **prep_args)

    if freq is None:
        freq = make_freq_vector(ts, method=make_freq_method)

    freq_angular = 2 * np.pi * freq

    # fix the zero frequency point
    if freq[0] == 0:
        freq_copy = freq[1:]
        freq_angular = 2 * np.pi * freq_copy

    psd = signal.lombscargle(ts, ys, freq_angular, **ana_args)

    if freq[0] == 0:
        psd = np.insert(psd, 0, np.nan)

    # output result
    res_dict = {
        'freq': np.asarray(freq),
        'psd': np.asarray(psd),
    }

    return res_dict


def periodogram(ys, ts, ana_args={}, prep_args={}, interp_method='interp', interp_args={}):
    ''' Call periodogram from scipy

    Args
    ----

    ys : array
        a time series
    ts : array
        time axis of the time series
    ana_args : dict
        the arguments for spectral analysis with periodogram, including
        - window (str): Desired window to use. See get_window for a list of windows and required parameters. If window is an array it will be used directly as the window. Defaults to None; equivalent to ‘boxcar’.
        - nfft (int): length of the FFT used. If None the length of x will be used.
        - return_onesided (bool): If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Note that for complex data, a two-sided spectrum is always returned.
        - scaling (str, {'density', 'spectrum'}): Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz if x is measured in V and computing the power spectrum (‘spectrum’) where Pxx has units of V**2 if x is measured in V. Defaults to ‘density’
        see https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.periodogram.html for the details
    interp_method : string
        {'interp', 'bin'}): perform interpolation or binning
    interp_args : dict
        the arguments for the interpolation or binning methods, for the details, check interp() and binvalues()
    prep_args : dict)
        the arguments for preprocess, including
        - detrend (str): 'none' - the original time series is assumed to have no trend;
                         'linear' - a linear least-squares fit to `ys` is subtracted;
                         'constant' - the mean of `ys` is subtracted
                         'savitzy-golay' - ys is filtered using the Savitzky-Golay
                             filters and the resulting filtered series is subtracted from y.
                         'hht' - detrending with Hilbert-Huang Transform
        - params (list): The paramters for the Savitzky-Golay filters. The first parameter
                         corresponds to the window size (default it set to half of the data)
                         while the second parameter correspond to the order of the filter
                         (default is 4). The third parameter is the order of the derivative
                         (the default is zero, which means only smoothing.)
        - gaussianize (bool): If True, gaussianizes the timeseries
        - standardize (bool): If True, standardizes the timeseries

    Returns
    -------

    res_dict : dict
        the result dictionary, including
        - freq (array): the frequency vector
        - psd (array): the spectral density vector


    '''
    # preprocessing
    
    ys, ts = clean_ts(ys, ts)
    ys = preprocess(ys, ts, **prep_args)

    # interpolate if not evenly-spaced
    if not is_evenly_spaced(ts):
        interp_func = {
            'interp': interp,
            'bin': bin_values,
        }
        ts, ys = interp_func[interp_method](ts, ys, **interp_args)

    # calculate sampling frequency
    dt = np.median(np.diff(ts))
    fs = 1 / dt

    # spectral analysis
    freq, psd = signal.periodogram(ys, fs, **ana_args)

    # fix the zero frequency point
    if freq[0] == 0:
        psd[0] = np.nan

    # output result
    res_dict = {
        'freq': np.asarray(freq),
        'psd': np.asarray(psd),
    }

    return res_dict