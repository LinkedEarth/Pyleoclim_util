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
import collections

__all__ = [
    'wwz_psd',
    'mtm',
]

from .wavelet import (
    preprocess,
    is_evenly_spaced,
    make_freq_vector,
    prepare_wwz,
    wwz,
    wwa2psd,
)
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


def wwz_psd(ys, ts, freq=None, tau=None, c=1e-3, nproc=8, nMC=200,
            detrend=False, params=["default", 4, 0, 1], gaussianize=False,
            standardize=True, Neff=3, anti_alias=False, avgs=2,
            method='default'):
    ''' Return the psd of a timeseries directly using wwz method.

    Args
    ----

    ys : array
        a time series, NaNs will be deleted automatically
    ts : array
        the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points, namely the time shift for wavelet analysis
    c : float
        the decay constant, the default value 1e-3 is good for most of the cases
    nproc : int
        the number of processes for multiprocessing
    nMC : int
        the number of Monte-Carlo simulations
    detrend : str
        None - the original time series is assumed to have no trend;
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay
               filters and the resulting filtered series is subtracted from y.
    params : list
        The paramters for the Savitzky-Golay filters. The first parameter
        corresponds to the window size (default it set to half of the data)
        while the second parameter correspond to the order of the filter
        (default is 4). The third parameter is the order of the derivative
        (the default is zero, which means only smoothing.)
    gaussianize : bool
        If True, gaussianizes the timeseries
    standardize : bool
        If True, standardizes the timeseries
    method : string
        'Foster' - the original WWZ method;
        'Kirchner' - the method Kirchner adapted from Foster;
        'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py
        'default' - the Numba version of the Kirchner algorithm will be called
    Neff : int
        effective number of points
    anti_alias : bool): If True, uses anti-aliasing
    avgs : int
        flag for whether spectrum is derived from instantaneous point measurements (avgs<>1)
        OR from measurements averaged over each sampling interval (avgs==1)

    Returns
    -------

    psd : array
        power spectral density
    freq : array
        vector of frequency
    psd_ar1_q95 : array
        the 95% quantile of the psds of AR1 processes
    psd_ar1 : array
        the psds of AR1 processes

    '''
    ys_cut, ts_cut, freq, tau = prepare_wwz(ys, ts, freq=freq, tau=tau)

    # get wwa but AR1_q is not needed here so set nMC=0
    #  wwa, _, _, coi, freq, _, Neffs, _ = wwz(ys_cut, ts_cut, freq=freq, tau=tau, c=c, nproc=nproc, nMC=0,
    res_wwz = wwz(ys_cut, ts_cut, freq=freq, tau=tau, c=c, nproc=nproc, nMC=0,
              detrend=detrend, params=params,
              gaussianize=gaussianize, standardize=standardize, method=method)

    psd = wwa2psd(res_wwz.amplitude, ts_cut, res_wwz.Neffs, freq=res_wwz.freq, Neff=Neff, anti_alias=anti_alias, avgs=avgs)
    #  psd[1/freqs > np.max(coi)] = np.nan  # cut off the unreliable part out of the coi
    #  psd = psd[1/freqs <= np.max(coi)] # cut off the unreliable part out of the coi
    #  freqs = freqs[1/freqs <= np.max(coi)]

    # Monte-Carlo simulations of AR1 process
    nf = np.size(freq)

    psd_ar1 = np.ndarray(shape=(nMC, nf))

    if nMC >= 1:
        #  tauest = wa.tau_estimation(ys_cut, ts_cut, detrend=detrend)

        for i in tqdm(range(nMC), desc='Monte-Carlo simulations'):
            #  r = wa.ar1_model(ts_cut, tauest)
            r = ar1_sim(ys_cut, np.size(ts_cut), 1, ts=ts_cut)
            res_red = wwz(r, ts_cut, freq=freq, tau=tau, c=c, nproc=nproc, nMC=0,
                                                                     detrend=detrend, params=params,
                                                                     gaussianize=gaussianize, standardize=standardize,
                                                                     method=method)
            psd_ar1[i, :] = wa.wwa2psd(res_red.wwa, ts_cut, res_red.Neffs,
                                       freq=res_red.freq, Neff=Neff, anti_alias=anti_alias, avgs=avgs)
            #  psd_ar1[i, 1/freqs_red > np.max(coi_red)] = np.nan  # cut off the unreliable part out of the coi
            #  psd_ar1 = psd_ar1[1/freqs_red <= np.max(coi_red)] # cut off the unreliable part out of the coi

        psd_ar1_q95 = mquantiles(psd_ar1, 0.95, axis=0)[0]

    else:
        psd_ar1_q95 = None

    Results = collections.namedtuple('Results', ['psd', 'freq', 'psd_ar1_q95', 'psd_ar1'])
    res = Results(psd=psd, freq=freq, psd_ar1_q95=psd_ar1_q95, psd_ar1=psd_ar1)

    return res
