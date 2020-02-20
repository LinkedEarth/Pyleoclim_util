#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:23:06 2017

@author: deborahkhider, fengzhu

Spectral module for pyleoclim
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy import interpolate
from scipy import optimize
from scipy import signal
from scipy.stats.mstats import mquantiles
from scipy.stats import pearsonr
import scipy.fftpack as fft

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib import gridspec

from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

import warnings

import collections

from math import factorial

#  import spectrum
import nitime.algorithms as nialg

import numba as nb
from numba.errors import NumbaPerformanceWarning

from sklearn import preprocessing
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

'''
Core functions below, focusing on algorithms
'''

class SpectralAnalysis(object):
    def welch(self, ys, ts, ana_args={}, prep_args={}, interp_method='interp', interp_args={}):
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
        wa = WaveletAnalysis()
        ys, ts = Timeseries.clean_ts(ys, ts)
        ys = wa.preprocess(ys, ts, **prep_args)

        # if data is not evenly spaced, interpolate
        if not wa.is_evenly_spaced(ts):
            interp_func = {
                'interp': Timeseries.interp,
                'bin': Timeseries.binvalues,
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


    def mtm(self, ys, ts, NW=2.5, ana_args={}, prep_args={}, interp_method='interp', interp_args={}):
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
        wa = WaveletAnalysis()
        ys, ts = Timeseries.clean_ts(ys, ts)
        ys = wa.preprocess(ys, ts, **prep_args)

        # interpolate if not evenly-spaced
        if not wa.is_evenly_spaced(ts):
            interp_func = {
                'interp': Timeseries.interp,
                'bin': Timeseries.binvalues,
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


    def lomb_scargle(self, ys, ts, freq=None, make_freq_method='nfft', prep_args={}, ana_args={}):
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
        ys, ts = Timeseries.clean_ts(ys, ts)
        wa = WaveletAnalysis()
        ys = wa.preprocess(ys, ts, **prep_args)

        if freq is None:
            freq = wa.make_freq_vector(ts, method=make_freq_method)

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


    def periodogram(self, ys, ts, ana_args={}, prep_args={}, interp_method='interp', interp_args={}):
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
        wa = WaveletAnalysis()
        ys, ts = Timeseries.clean_ts(ys, ts)
        ys = wa.preprocess(ys, ts, **prep_args)

        # interpolate if not evenly-spaced
        if not wa.is_evenly_spaced(ts):
            interp_func = {
                'interp': Timeseries.interp,
                'bin': Timeseries.binvalues,
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


class WaveletAnalysis(object):
    '''Performing wavelet analysis @author: fzhu
    '''
    def is_evenly_spaced(self, ts):
        ''' Check if a time axis is evenly spaced.

        Args
        ----

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

    def ar1_fit_evenly(self, ys, ts, detrend=False, params=["default", 4, 0, 1], gaussianize=False):
        ''' Returns the lag-1 autocorrelation from ar1 fit.

        Args
        ----

        ys : array
            vector of (float) numbers as a time series
        ts : array
            The time axis for the timeseries. Necessary for use with the Savitzky-Golay filters method since the series should be evenly spaced.
        detrend : string
            'linear' - a linear least-squares fit to `ys` is subtracted;
            'constant' - the mean of `ys` is subtracted
            'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
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

        Returns
        -------

        g : float
            lag-1 autocorrelation coefficient

        '''
        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize)
        ar1_mod = sm.tsa.AR(pd_ys, missing='drop').fit(maxlag=1)
        g = ar1_mod.params[1]

        return g

    def preprocess(self, ys, ts, detrend=False, params=["default", 4, 0, 1], gaussianize=False, standardize=True):
        ''' Return the processed time series using detrend and standardization.

        Args
        ----

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

        Returns
        -------

        res : array
            the processed time series

        '''

        if detrend == 'none' or detrend is False or detrend is None:
            ys_d = ys
        else:
            ys_d = detrend(ys, ts, method=detrend, params=params)

        if standardize:
            res, _, _ = Timeseries.standardize(ys_d)
        else:
            res = ys_d

        if gaussianize:
            res = Timeseries.gaussianize(res)

        return res

    def tau_estimation(self, ys, ts, detrend=False, params=["default", 4, 0, 1], gaussianize=False, standardize=True):
        ''' Return the estimated persistence of a givenevenly/unevenly spaced time series.

        Args
        ----

        ys : array
            a time series
        ts : array
            time axis of the time series
        detrend : string
            'linear' - a linear least-squares fit to `ys` is subtracted;
            'constant' - the mean of `ys` is subtracted
            'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
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

        Returns
        -------

        tau_est : float
            the estimated persistence

        References
        ----------

        Mudelsee, M. TAUEST: A Computer Program for Estimating Persistence in Unevenly Spaced Weather/Climate Time Series.
            Comput. Geosci. 28, 69–72 (2002).

        '''
        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)
        dt = np.diff(ts)
        #  assert dt > 0, "The time points should be increasing!"

        def ar1_fun(a):
            return np.sum((pd_ys[1:] - pd_ys[:-1]*a**dt)**2)

        a_est = optimize.minimize_scalar(ar1_fun, bounds=[0, 1], method='bounded').x
        #  a_est = optimize.minimize_scalar(ar1_fun, method='brent').x

        tau_est = -1 / np.log(a_est)

        return tau_est

    def assertPositiveInt(self, *args):
        ''' Assert that the args are all positive integers.
        '''
        for arg in args:
            assert isinstance(arg, int) and arg >= 1

    def ar1_model(self, ts, tau, n=None):
        ''' Return a time series with the AR1 process

        Args
        ----

        ts : array
            time axis of the time series
        tau : float
            the averaged persistence
        n : int
            the length of the AR1 process

        Returns
        -------

        r : array
            the AR1 time series

        References
        ----------

        Schulz, M. & Mudelsee, M. REDFIT: estimating red-noise spectra directly from unevenly spaced
            paleoclimatic time series. Computers & Geosciences 28, 421–426 (2002).

        '''
        if n is None:
            n = np.size(ts)
        else:
            self.assertPositiveInt(n)

        r = np.zeros(n)

        r[0] = 1
        for i in range(1, n):
            scaled_dt = (ts[i] - ts[i-1]) / tau
            rho = np.exp(-scaled_dt)
            err = np.random.normal(0, np.sqrt(1 - rho**2), 1)
            r[i] = r[i-1]*rho + err

        return r

    def wwz_basic(self, ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend=False, params=['default', 4, 0, 1],
                  gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA).

        Original method from Foster. Not multiprocessing.

        Args
        ----

        ys : array
            a time series
        ts : array
            time axis of the time series
        freq : array
            vector of frequency
        tau : array
            the evenly-spaced time points, namely the time shift for wavelet analysis
        c : float
            the decay constant
        Neff : int
            the threshold of the number of effective degree of freedom
        nproc :int
            fake argument, just for convenience
        detrend : string
            None - the original time series is assumed to have no trend;
            'linear' - a linear least-squares fit to `ys` is subtracted;
            'constant' - the mean of `ys` is subtracted
            'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
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

        Returns
        -------

        wwa : array
            the weighted wavelet amplitude
        phase : array
            the weighted wavelet phase
        Neffs : array
            the matrix of effective number of points in the time-scale coordinates
        coeff : array
            the wavelet transform coefficients (a0, a1, a2)

        References
        ----------

        Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
        Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
            Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        assert nproc == 1, "wwz_basic() only supports nproc=1"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nf = np.size(freq)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freq)

        Neffs = np.ndarray(shape=(nt, nf))
        ywave_1 = np.ndarray(shape=(nt, nf))
        ywave_2 = np.ndarray(shape=(nt, nf))
        ywave_3 = np.ndarray(shape=(nt, nf))

        S = np.zeros(shape=(3, 3))

        for k in range(nf):
            for j in range(nt):
                dz = omega[k] * (ts - tau[j])
                weights = np.exp(-c*dz**2)

                sum_w = np.sum(weights)
                Neffs[j, k] = sum_w**2 / np.sum(weights**2)  # local number of effective dof

                if Neffs[j, k] <= Neff:
                    ywave_1[j, k] = np.nan  # the coefficients cannot be estimated reliably when Neff_loc <= Neff
                    ywave_2[j, k] = np.nan
                    ywave_3[j, k] = np.nan
                else:
                    phi2 = np.cos(dz)
                    phi3 = np.sin(dz)

                    S[0, 0] = 1
                    S[1, 1] = np.sum(weights*phi2*phi2) / sum_w
                    S[2, 2] = np.sum(weights*phi3*phi3) / sum_w
                    S[1, 0] = S[0, 1] = np.sum(weights*phi2) / sum_w
                    S[2, 0] = S[0, 2] = np.sum(weights*phi3) / sum_w
                    S[2, 1] = S[1, 2] = np.sum(weights*phi2*phi3) / sum_w

                    S_inv = np.linalg.pinv(S)

                    weighted_phi1 = np.sum(weights*pd_ys) / sum_w
                    weighted_phi2 = np.sum(weights*phi2*pd_ys) / sum_w
                    weighted_phi3 = np.sum(weights*phi3*pd_ys) / sum_w

                    ywave_1[j, k] = S_inv[0, 0]*weighted_phi1 + S_inv[0, 1]*weighted_phi2 + S_inv[0, 2]*weighted_phi3
                    ywave_2[j, k] = S_inv[1, 0]*weighted_phi1 + S_inv[1, 1]*weighted_phi2 + S_inv[1, 2]*weighted_phi3
                    ywave_3[j, k] = S_inv[2, 0]*weighted_phi1 + S_inv[2, 1]*weighted_phi2 + S_inv[2, 2]*weighted_phi3

        wwa = np.sqrt(ywave_2**2 + ywave_3**2)
        phase = np.arctan2(ywave_3, ywave_2)
        #  coeff = ywave_2 + ywave_3*1j
        coeff = (ywave_1, ywave_2, ywave_3)

        return wwa, phase, Neffs, coeff

    def wwz_nproc(self, ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, nproc=8,  detrend=False, params=['default', 4, 0, 1],
                  gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA).

        Original method from Foster. Supports multiprocessing.

        Args
        ----

        ys : array
            a time series
        ts : array
            time axis of the time series
        freq : array
            vector of frequency
        tau : array
            the evenly-spaced time points, namely the time shift for wavelet analysis
        c : float
            the decay constant
        Neff : int
            the threshold of the number of effective degree of freedom
        nproc : int
            the number of processes for multiprocessing
        detrend : string
            None - the original time series is assumed to have no trend;
            'linear' - a linear least-squares fit to `ys` is subtracted;
            'constant' - the mean of `ys` is subtracted
            'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
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

        Returns
        -------

        wwa : array
            the weighted wavelet amplitude
        phase : array
            the weighted wavelet phase
        Neffs : array
            the matrix of effective number of points in the time-scale coordinates
        coeff : array
            the wavelet transform coefficients (a0, a1, a2)

        '''
        assert nproc >= 2, "wwz_nproc() should use nproc >= 2, if want serial run, please use wwz_basic()"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nf = np.size(freq)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freq)

        Neffs = np.ndarray(shape=(nt, nf))
        ywave_1 = np.ndarray(shape=(nt, nf))
        ywave_2 = np.ndarray(shape=(nt, nf))
        ywave_3 = np.ndarray(shape=(nt, nf))

        def wwa_1g(tau, omega):
            dz = omega * (ts - tau)
            weights = np.exp(-c*dz**2)

            sum_w = np.sum(weights)
            Neff_loc = sum_w**2 / np.sum(weights**2)

            S = np.zeros(shape=(3, 3))

            if Neff_loc <= Neff:
                ywave_2_1g = np.nan
                ywave_3_1g = np.nan
            else:
                phi2 = np.cos(dz)
                phi3 = np.sin(dz)

                S[0, 0] = 1
                S[1, 1] = np.sum(weights*phi2*phi2) / sum_w
                S[2, 2] = np.sum(weights*phi3*phi3) / sum_w
                S[1, 0] = S[0, 1] = np.sum(weights*phi2) / sum_w
                S[2, 0] = S[0, 2] = np.sum(weights*phi3) / sum_w
                S[2, 1] = S[1, 2] = np.sum(weights*phi2*phi3) / sum_w

                S_inv = np.linalg.pinv(S)

                weighted_phi1 = np.sum(weights*pd_ys) / sum_w
                weighted_phi2 = np.sum(weights*phi2*pd_ys) / sum_w
                weighted_phi3 = np.sum(weights*phi3*pd_ys) / sum_w

                ywave_1_1g = S_inv[0, 0]*weighted_phi1 + S_inv[0, 1]*weighted_phi2 + S_inv[0, 2]*weighted_phi3
                ywave_2_1g = S_inv[1, 0]*weighted_phi1 + S_inv[1, 1]*weighted_phi2 + S_inv[1, 2]*weighted_phi3
                ywave_3_1g = S_inv[2, 0]*weighted_phi1 + S_inv[2, 1]*weighted_phi2 + S_inv[2, 2]*weighted_phi3

            return Neff_loc, ywave_1_1g, ywave_2_1g, ywave_3_1g

        tf_mesh = np.meshgrid(tau, omega)
        list_of_grids = list(zip(*(grid.flat for grid in tf_mesh)))
        tau_grids, omega_grids = zip(*list_of_grids)

        with Pool(nproc) as pool:
            res = pool.map(wwa_1g, tau_grids, omega_grids)
            res_array = np.asarray(res)
            Neffs = res_array[:, 0].reshape((np.size(omega), np.size(tau))).T
            ywave_1 = res_array[:, 1].reshape((np.size(omega), np.size(tau))).T
            ywave_2 = res_array[:, 2].reshape((np.size(omega), np.size(tau))).T
            ywave_3 = res_array[:, 3].reshape((np.size(omega), np.size(tau))).T

        wwa = np.sqrt(ywave_2**2 + ywave_3**2)
        phase = np.arctan2(ywave_3, ywave_2)
        #  coeff = ywave_2 + ywave_3*1j
        coeff = (ywave_1, ywave_2, ywave_3)

        return wwa, phase, Neffs, coeff

    def kirchner_basic(self, ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend=False, params=["default", 4, 0, 1],
                       gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Method modified by Kirchner. No multiprocessing.

        Args
        ----

        ys : array
            a time series
        ts : array
            time axis of the time series
        freq : array
            vector of frequency
        tau : array
            the evenly-spaced time points, namely the time shift for wavelet analysis
        c : float
            the decay constant
        Neff : int
            the threshold of the number of effective degree of freedom
        nproc : int
            fake argument for convenience, for parameter consistency between functions, does not need to be specified
        detrend : string
            None - the original time series is assumed to have no trend;
            'linear' - a linear least-squares fit to `ys` is subtracted;
            'constant' - the mean of `ys` is subtracted
            'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
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

        Returns
        -------

        wwa : array
            the weighted wavelet amplitude
        phase : array
            the weighted wavelet phase
        Neffs : array
            the matrix of effective number of points in the time-scale coordinates
        coeff : array
            the wavelet transform coefficients (a0, a1, a2)

        References
        ----------

        Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
        Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
        Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        assert nproc == 1, "wwz_basic() only supports nproc=1"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nts = np.size(ts)
        nf = np.size(freq)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freq)

        Neffs = np.ndarray(shape=(nt, nf))
        a0 = np.ndarray(shape=(nt, nf))
        a1 = np.ndarray(shape=(nt, nf))
        a2 = np.ndarray(shape=(nt, nf))

        for k in range(nf):
            for j in range(nt):
                dz = omega[k] * (ts - tau[j])
                weights = np.exp(-c*dz**2)

                sum_w = np.sum(weights)
                Neffs[j, k] = sum_w**2 / np.sum(weights**2)  # local number of effective dof

                if Neffs[j, k] <= Neff:
                    a0[j, k] = np.nan  # the coefficients cannot be estimated reliably when Neff_loc <= Neff
                    a1[j, k] = np.nan
                    a2[j, k] = np.nan
                else:
                    def w_prod(xs, ys):
                        return np.sum(weights*xs*ys) / sum_w

                    sin_basis = np.sin(omega[k]*ts)
                    cos_basis = np.cos(omega[k]*ts)
                    one_v = np.ones(nts)

                    sin_one = w_prod(sin_basis, one_v)
                    cos_one = w_prod(cos_basis, one_v)
                    sin_cos = w_prod(sin_basis, cos_basis)
                    sin_sin = w_prod(sin_basis, sin_basis)
                    cos_cos = w_prod(cos_basis, cos_basis)

                    numerator = 2 * (sin_cos - sin_one * cos_one)
                    denominator = (cos_cos - cos_one**2) - (sin_sin - sin_one**2)
                    time_shift = np.arctan2(numerator, denominator) / (2*omega[k])  # Eq. (S5)

                    sin_shift = np.sin(omega[k]*(ts - time_shift))
                    cos_shift = np.cos(omega[k]*(ts - time_shift))
                    sin_tau_center = np.sin(omega[k]*(time_shift - tau[j]))
                    cos_tau_center = np.cos(omega[k]*(time_shift - tau[j]))

                    ys_cos_shift = w_prod(pd_ys, cos_shift)
                    ys_sin_shift = w_prod(pd_ys, sin_shift)
                    ys_one = w_prod(pd_ys, one_v)
                    cos_shift_one = w_prod(cos_shift, one_v)
                    sin_shift_one = w_prod(sin_shift, one_v)

                    A = 2*(ys_cos_shift-ys_one*cos_shift_one)
                    B = 2*(ys_sin_shift-ys_one*sin_shift_one)

                    a0[j, k] = ys_one
                    a1[j, k] = cos_tau_center*A - sin_tau_center*B  # Eq. (S6)
                    a2[j, k] = sin_tau_center*A + cos_tau_center*B  # Eq. (S7)

        wwa = np.sqrt(a1**2 + a2**2)
        phase = np.arctan2(a2, a1)
        #  coeff = a1 + a2*1j
        coeff = (a0, a1, a2)

        return wwa, phase, Neffs, coeff
    def kirchner_nproc(self, ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend=False, params=['default', 4, 0, 1],
                       gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Method modified by kirchner. Supports multiprocessing.

        Args
        ----

        ys : array
            a time series
        ts : array
            time axis of the time series
        freq : array
            vector of frequency
        tau : array
            the evenly-spaced time points, namely the time shift for wavelet analysis
        c : float
            the decay constant
        Neff : int
            the threshold of the number of effective degree of freedom
        nproc : int
            the number of processes for multiprocessing
        detrend : string
            None - the original time series is assumed to have no trend;
            'linear' - a linear least-squares fit to `ys` is subtracted;
            'constant' - the mean of `ys` is subtracted
            'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
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

        Returns
        -------

        wwa (array): the weighted wavelet amplitude
        phase (array): the weighted wavelet phase
        Neffs (array): the matrix of effective number of points in the time-scale coordinates
        coeff (array): the wavelet transform coefficients (a0, a1, a2)

        '''
        assert nproc >= 2, "wwz_nproc() should use nproc >= 2, if want serial run, please use wwz_basic()"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nts = np.size(ts)
        nf = np.size(freq)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freq)

        Neffs = np.ndarray(shape=(nt, nf))
        a0 = np.ndarray(shape=(nt, nf))
        a1 = np.ndarray(shape=(nt, nf))
        a2 = np.ndarray(shape=(nt, nf))

        def wwa_1g(tau, omega):
            dz = omega * (ts - tau)
            weights = np.exp(-c*dz**2)

            sum_w = np.sum(weights)
            Neff_loc = sum_w**2 / np.sum(weights**2)

            if Neff_loc <= Neff:
                a0_1g = np.nan  # the coefficients cannot be estimated reliably when Neff_loc <= Neff
                a1_1g = np.nan  # the coefficients cannot be estimated reliably when Neff_loc <= Neff
                a2_1g = np.nan
            else:
                def w_prod(xs, ys):
                    return np.sum(weights*xs*ys) / sum_w

                sin_basis = np.sin(omega*ts)
                cos_basis = np.cos(omega*ts)
                one_v = np.ones(nts)

                sin_one = w_prod(sin_basis, one_v)
                cos_one = w_prod(cos_basis, one_v)
                sin_cos = w_prod(sin_basis, cos_basis)
                sin_sin = w_prod(sin_basis, sin_basis)
                cos_cos = w_prod(cos_basis, cos_basis)

                numerator = 2*(sin_cos - sin_one*cos_one)
                denominator = (cos_cos - cos_one**2) - (sin_sin - sin_one**2)
                time_shift = np.arctan2(numerator, denominator) / (2*omega)  # Eq. (S5)

                sin_shift = np.sin(omega*(ts - time_shift))
                cos_shift = np.cos(omega*(ts - time_shift))
                sin_tau_center = np.sin(omega*(time_shift - tau))
                cos_tau_center = np.cos(omega*(time_shift - tau))

                ys_cos_shift = w_prod(pd_ys, cos_shift)
                ys_sin_shift = w_prod(pd_ys, sin_shift)
                ys_one = w_prod(pd_ys, one_v)
                cos_shift_one = w_prod(cos_shift, one_v)
                sin_shift_one = w_prod(sin_shift, one_v)

                A = 2*(ys_cos_shift - ys_one*cos_shift_one)
                B = 2*(ys_sin_shift - ys_one*sin_shift_one)

                a0_1g = ys_one
                a1_1g = cos_tau_center*A - sin_tau_center*B  # Eq. (S6)
                a2_1g = sin_tau_center*A + cos_tau_center*B  # Eq. (S7)

            return Neff_loc, a0_1g, a1_1g, a2_1g

        tf_mesh = np.meshgrid(tau, omega)
        list_of_grids = list(zip(*(grid.flat for grid in tf_mesh)))
        tau_grids, omega_grids = zip(*list_of_grids)

        with Pool(nproc) as pool:
            res = pool.map(wwa_1g, tau_grids, omega_grids)
            res_array = np.asarray(res)
            Neffs = res_array[:, 0].reshape((np.size(omega), np.size(tau))).T
            a0 = res_array[:, 1].reshape((np.size(omega), np.size(tau))).T
            a1 = res_array[:, 2].reshape((np.size(omega), np.size(tau))).T
            a2 = res_array[:, 3].reshape((np.size(omega), np.size(tau))).T

        wwa = np.sqrt(a1**2 + a2**2)
        phase = np.arctan2(a2, a1)
        #  coeff = a1 + a2*1j
        coeff = (a0, a1, a2)

        return wwa, phase, Neffs, coeff

    def kirchner_numba(self, ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, detrend=False, params=["default", 4, 0, 1],
                       gaussianize=False, standardize=True, nproc=1):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Using numba.

        Args
        ----

        ys : array
            a time series
        ts : array
            time axis of the time series
        freq : array
            vector of frequency
        tau : array
            the evenly-spaced time points, namely the time shift for wavelet analysis
        c : float
            the decay constant
        Neff : int
            the threshold of the number of effective degree of freedom
        nproc : int
            fake argument, just for convenience
        detrend : string
            None - the original time series is assumed to have no trend;
            'linear' - a linear least-squares fit to `ys` is subtracted;
            'constant' - the mean of `ys` is subtracted
            'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
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

        Returns
        -------

        wwa : array
            the weighted wavelet amplitude
        phase : array
            the weighted wavelet phase
        Neffs : array
            the matrix of effective number of points in the time-scale coordinates
        coeff : array
            the wavelet transform coefficients (a0, a1, a2)

        References
        ----------

        Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
        Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
            Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        self.assertPositiveInt(Neff)
        nt = np.size(tau)
        nts = np.size(ts)
        nf = np.size(freq)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freq)

        Neffs = np.ndarray(shape=(nt, nf))
        a0 = np.ndarray(shape=(nt, nf))
        a1 = np.ndarray(shape=(nt, nf))
        a2 = np.ndarray(shape=(nt, nf))

        @nb.jit(nopython=True, parallel=True, fastmath=True)
        def loop_over(nf, nt, Neffs, a0, a1, a2):
            def wwa_1g(tau, omega):
                dz = omega * (ts - tau)
                weights = np.exp(-c*dz**2)

                sum_w = np.sum(weights)
                Neff_loc = sum_w**2 / np.sum(weights**2)

                if Neff_loc <= Neff:
                    a0_1g = np.nan  # the coefficients cannot be estimated reliably when Neff_loc <= Neff
                    a1_1g = np.nan  # the coefficients cannot be estimated reliably when Neff_loc <= Neff
                    a2_1g = np.nan
                else:
                    def w_prod(xs, ys):
                        return np.sum(weights*xs*ys) / sum_w

                    sin_basis = np.sin(omega*ts)
                    cos_basis = np.cos(omega*ts)
                    one_v = np.ones(nts)

                    sin_one = w_prod(sin_basis, one_v)
                    cos_one = w_prod(cos_basis, one_v)
                    sin_cos = w_prod(sin_basis, cos_basis)
                    sin_sin = w_prod(sin_basis, sin_basis)
                    cos_cos = w_prod(cos_basis, cos_basis)

                    numerator = 2*(sin_cos - sin_one*cos_one)
                    denominator = (cos_cos - cos_one**2) - (sin_sin - sin_one**2)
                    time_shift = np.arctan2(numerator, denominator) / (2*omega)  # Eq. (S5)

                    sin_shift = np.sin(omega*(ts - time_shift))
                    cos_shift = np.cos(omega*(ts - time_shift))
                    sin_tau_center = np.sin(omega*(time_shift - tau))
                    cos_tau_center = np.cos(omega*(time_shift - tau))

                    ys_cos_shift = w_prod(pd_ys, cos_shift)
                    ys_sin_shift = w_prod(pd_ys, sin_shift)
                    ys_one = w_prod(pd_ys, one_v)
                    cos_shift_one = w_prod(cos_shift, one_v)
                    sin_shift_one = w_prod(sin_shift, one_v)

                    A = 2*(ys_cos_shift - ys_one*cos_shift_one)
                    B = 2*(ys_sin_shift - ys_one*sin_shift_one)

                    a0_1g = ys_one
                    a1_1g = cos_tau_center*A - sin_tau_center*B  # Eq. (S6)
                    a2_1g = sin_tau_center*A + cos_tau_center*B  # Eq. (S7)

                return Neff_loc, a0_1g, a1_1g, a2_1g

            for k in nb.prange(nf):
                for j in nb.prange(nt):
                    Neffs[j, k], a0[j, k], a1[j, k], a2[j, k] = wwa_1g(tau[j], omega[k])

            return Neffs, a0, a1, a2

        Neffs, a0, a1, a2 = loop_over(nf, nt, Neffs, a0, a1, a2)

        wwa = np.sqrt(a1**2 + a2**2)
        phase = np.arctan2(a2, a1)
        #  coeff = a1 + a2*1j
        coeff = (a0, a1, a2)

        return wwa, phase, Neffs, coeff

    def kirchner_f2py(self, ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend=False, params=['default', 4, 0, 1],
                      gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Fastest method. Calls Fortran libraries.

        Args
        ----

        ys : array
            a time series
        ts : array
            time axis of the time series
        freq : array
            vector of frequency
        tau : array
            the evenly-spaced time points, namely the time shift for wavelet analysis
        c : float
            the decay constant
        Neff : int
            the threshold of the number of effective degree of freedom
        nproc : int
            fake argument, just for convenience
        detrend : string
            None - the original time series is assumed to have no trend;
            'linear' - a linear least-squares fit to `ys` is subtracted;
            'constant' - the mean of `ys` is subtracted
            'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
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

        Returns
        -------

        wwa : array
            the weighted wavelet amplitude
        phase : array
            the weighted wavelet phase
        Neffs : array
            the matrix of effective number of points in the time-scale coordinates
        coeff : array
            the wavelet transform coefficients (a0, a1, a2)

        '''
        from . import f2py_wwz as f2py
        self.assertPositiveInt(Neff, nproc)

        nt = np.size(tau)
        nts = np.size(ts)
        nf = np.size(freq)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freq)

        Neffs, a0, a1, a2 = f2py.f2py_wwz.wwa(tau, omega, c, Neff, ts, pd_ys, nproc, nts, nt, nf)

        undef = -99999.
        a0[a0 == undef] = np.nan
        a1[a1 == undef] = np.nan
        a2[a2 == undef] = np.nan
        wwa = np.sqrt(a1**2 + a2**2)
        phase = np.arctan2(a2, a1)

        #  coeff = a1 + a2*1j
        coeff = (a0, a1, a2)

        return wwa, phase, Neffs, coeff

    def make_coi(self, tau, Neff=3):
        ''' Return the cone of influence.

        Args
        ----

        tau : array
            the evenly-spaced time points, namely the time shift for wavelet analysis
        Neff : int
            the threshold of the number of effective samples

        Returns
        -------

            coi : array
                cone of influence

        References
        ----------

        wave_signif() in http://paos.colorado.edu/research/wavelets/wave_python/waveletFunctions.py

        '''
        assert isinstance(Neff, int) and Neff >= 1
        nt = np.size(tau)

        fourier_factor = 4*np.pi / (Neff+np.sqrt(2+Neff**2))
        coi_const = fourier_factor / np.sqrt(2)

        dt = np.median(np.diff(tau))
        nt_half = (nt+1)//2 - 1

        A = np.append(0.00001, np.arange(nt_half)+1)
        B = A[::-1]

        if nt % 2 == 0:
            C = np.append(A, B)
        else:
            C = np.append(A, B[1:])

        coi = coi_const * dt * C

        return coi

    def make_omega(self, ts, freq):
        ''' Return the angular frequency based on the time axis and given frequency vector

        Args
        ----

        ys : array
            a time series
        ts : array
            time axis of the time series
        freq : array
            vector of frequency

        Returns
        -------


        omega : array
            the angular frequency vector

        '''
        # for the frequency band larger than f_Nyquist, the wwa will be marked as NaNs
        f_Nyquist = 0.5 / np.median(np.diff(ts))
        freq_with_nan = np.copy(freq)
        freq_with_nan[freq > f_Nyquist] = np.nan
        omega = 2*np.pi*freq_with_nan

        return omega

    def wwa2psd(self, wwa, ts, Neffs, freq=None, Neff=3, anti_alias=False, avgs=2):
        """ Return the power spectral density (PSD) using the weighted wavelet amplitude (WWA).

        Args
        ----

        wwa : array
            the weighted wavelet amplitude.
        ts : array
            the time points, should be pre-truncated so that the span is exactly what is used for wwz
        Neffs : array
            the matrix of effective number of points in the time-scale coordinates obtained from wwz from wwz
        freq : array
            vector of frequency from wwz
        Neff : int
            the threshold of the number of effective samples
        anti_alias : bool
            whether to apply anti-alias filter
        avgs : int
            flag for whether spectrum is derived from instantaneous point measurements (avgs<>1) OR from measurements averaged over each sampling interval (avgs==1)

        Returns
        -------

        psd : array
            power spectral density

        References
        ----------

        Kirchner's C code for weighted psd calculation

        """
        af = AliasFilter()

        # weighted psd calculation start
        power = wwa**2 * 0.5 * (np.max(ts)-np.min(ts))/np.size(ts) * Neffs

        Neff_diff = Neffs - Neff
        Neff_diff[Neff_diff < 0] = 0

        sum_power = np.nansum(power * Neff_diff, axis=0)
        sum_eff = np.nansum(Neff_diff, axis=0)

        psd = sum_power / sum_eff
        # weighted psd calculation end

        if anti_alias:
            assert freq is not None, "freq is required for alias filter!"
            dt = np.median(np.diff(ts))
            f_sampling = 1/dt
            psd_copy = psd[1:]
            freq_copy = freq[1:]
            alpha, filtered_pwr, model_pwer, aliased_pwr = af.alias_filter(
                freq_copy, psd_copy, f_sampling, f_sampling*1e3, np.min(freq), avgs)

            psd[1:] = np.copy(filtered_pwr)

        return psd

    def freq_vector_lomb_scargle(self, ts, nf=None, ofac=4, hifac=1):
        ''' Return the frequency vector based on the Lomb-Scargle algorithm.

        Args
        ----

        ts : array
            time axis of the time series
        ofac : float
            Oversampling rate that influences the resolution of the frequency axis,
                     when equals to 1, it means no oversamling (should be >= 1).
                     The default value 4 is usaually a good value.
        hifac : float
            fhi/fnyq (should be >= 1), where fhi is the highest frequency that
            can be analyzed by the Lomb-Scargle algorithm and fnyq is the Nyquist frequency.

        Returns
        -------

        freq : array
            the frequency vector

        References
        ----------

        Trauth, M. H. MATLAB® Recipes for Earth Sciences. (Springer, 2015). pp 181.

        '''
        assert ofac >= 1 and hifac <= 1, "`ofac` should be >= 1, and `hifac` should be <= 1"

        dt = np.median(np.diff(ts))
        flo = (1/(2*dt)) / (np.size(ts)*ofac)
        fhi = hifac / (2*dt)

        if nf is None:
            df = flo
            nf = (fhi - flo) / df + 1

        freq = np.linspace(flo, fhi, nf)

        return freq

    def freq_vector_welch(self, ts):
        ''' Return the frequency vector based on the Welch's method.

        Args
        ----

        ts : array
            time axis of the time series

        Returns
        -------

        freq : array
            the frequency vector

        References
        ----------

        https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/Spectral.py

        '''
        nt = np.size(ts)
        dt = np.median(np.diff(ts))
        fs = 1 / dt
        if nt % 2 == 0:
            n_freq = nt//2 + 1
        else:
            n_freq = (nt+1) // 2

        freq = np.arange(n_freq) * fs / nt

        return freq

    def freq_vector_nfft(self, ts):
        ''' Return the frequency vector based on NFFT

        Args
        ----

        ts : array
            time axis of the time series

        Returns
        -------

        freq : array
            the frequency vector

        '''
        nt = np.size(ts)
        dt = np.median(np.diff(ts))
        fs = 1 / dt
        n_freq = nt//2 + 1

        freq = np.linspace(0, fs/2, n_freq)

        return freq

    def make_freq_vector(self, ts, method = 'nfft', **kwargs):
        ''' Make frequency vector- Selector function.

        This function selects among various methods to obtain the frequency
        vector.

        Args
        ----

        ts : array): time axis of the time series
        method : string
            The method to use. Options are 'nfft' (default), 'Lomb-Scargle', 'Welch'
        kwargs : dict, optional
                For Lomb_Scargle, additional parameters may be passed:
                - nf (int): number of frequency points
                - ofac (float): Oversampling rate that influences the resolution of the frequency axis,
                     when equals to 1, it means no oversamling (should be >= 1).
                     The default value 4 is usaually a good value.
                - hifac (float): fhi/fnyq (should be >= 1), where fhi is the highest frequency that
                      can be analyzed by the Lomb-Scargle algorithm and fnyq is the Nyquist frequency.

        Returns
        -------

        freq : array
            the frequency vector

        '''

        if method == 'Lomb-Scargle':
            freq = self.freq_vector_lomb_scargle(ts,**kwargs)
        elif method == 'Welch':
            freq = self.freq_vector_welch(ts)
        else:
            freq = self.freq_vector_nfft(ts)
        #  freq = freq[1:]  # discard the first element 0

        return freq

    def beta_estimation(self, psd, freq, fmin=None, fmax=None):
        ''' Estimate the power slope of a 1/f^beta process.

        Args
        ----

        psd : array
            the power spectral density
        freq : array
            the frequency vector
        fmin : float
            the min of frequency range for beta estimation
        fmax : float
            the max of frequency range for beta estimation

        Returns
        -------

        beta : float
            the estimated slope
        f_binned : array
            binned frequency vector
        psd_binned : array
            binned power spectral density
        Y_reg : array
            prediction based on linear regression

        '''
        # drop the PSD at frequency zero
        if freq[0] == 0:
            psd = psd[1:]
            freq = freq[1:]

        if fmin is None or fmin == 0:
            fmin = np.min(freq)

        if fmax is None:
            fmax = np.max(freq)

        Results = collections.namedtuple('Results', ['beta', 'f_binned', 'psd_binned', 'Y_reg', 'std_err'])
        if np.max(freq) < fmax or np.min(freq) > fmin:
            print(fmin, fmax)
            print(np.min(freq), np.max(freq))
            print('WRONG')
            res = Results(beta=np.nan, f_binned=np.nan, psd_binned=np.nan, Y_reg=np.nan, std_err=np.nan)
            return res

        # frequency binning start
        fminindx = np.where(freq >= fmin)[0][0]
        fmaxindx = np.where(freq <= fmax)[0][-1]

        if fminindx >= fmaxindx:
            res = Results(beta=np.nan, f_binned=np.nan, psd_binned=np.nan, Y_reg=np.nan, std_err=np.nan)
            return res

        logf = np.log(freq)
        logf_step = logf[fminindx+1] - logf[fminindx]
        logf_start = logf[fminindx]
        logf_end = logf[fmaxindx]
        logf_binedges = np.arange(logf_start, logf_end+logf_step, logf_step)

        n_intervals = np.size(logf_binedges)-1
        logpsd_binned = np.empty(n_intervals)
        logf_binned = np.empty(n_intervals)

        logpsd = np.log(psd)

        for i in range(n_intervals):
            lb = logf_binedges[i]
            ub = logf_binedges[i+1]
            q = np.where((logf > lb) & (logf <= ub))

            logpsd_binned[i] = np.nanmean(logpsd[q])
            logf_binned[i] = (ub + lb) / 2

        f_binned = np.exp(logf_binned)
        psd_binned = np.exp(logpsd_binned)
        # frequency binning end

        # linear regression below
        Y = np.log10(psd_binned)
        X = np.log10(f_binned)
        X_ex = sm.add_constant(X)

        model = sm.OLS(Y, X_ex)
        results = model.fit()

        if np.size(results.params) < 2:
            beta = np.nan
            Y_reg = np.nan
            std_err = np.nan
        else:
            beta = -results.params[1]  # the slope we want
            Y_reg = 10**model.predict(results.params)  # prediction based on linear regression
            std_err = results.bse[1]

        res = Results(beta=beta, f_binned=f_binned, psd_binned=psd_binned, Y_reg=Y_reg, std_err=std_err)

        return res

    def beta2HurstIndex(self, beta):
        ''' Translate psd slope to Hurst index

        Args
        ----

        beta : float
            the estimated slope of a power spectral density curve

        Returns
        -------

        H : float
            Hurst index, should be in (0, 1)

        References
        ----------

        Equation 2 in http://www.bearcave.com/misl/misl_tech/wavelets/hurst/

        '''
        H = (beta-1)/2

        return H

    def psd_ar(self, var_noise, freq, ar_params, f_sampling):
        ''' Return the theoretical power spectral density (PSD) of an autoregressive model

        Args
        ----

        var_noise : float
            the variance of the noise of the AR process
        freq : array
            vector of frequency
        ar_params : array
            autoregressive coefficients, not including zero-lag
        f_sampling : float
            sampling frequency

        Returns
        -------

        psd : array
            power spectral density

        '''
        p = np.size(ar_params)

        tmp = np.ndarray(shape=(p, np.size(freq)), dtype=complex)
        for k in range(p):
            tmp[k, :] = np.exp(-1j*2*np.pi*(k+1)*freq/f_sampling)

        psd = var_noise / np.absolute(1-np.sum(ar_params*tmp, axis=0))**2

        return psd

    def fBMsim(self, N=128, H=0.25):
        '''Simple method to generate fractional Brownian Motion

        Args
        ----

        N : int
            the length of the simulated time series
        H : float
            Hurst index, should be in (0, 1). The relationship between H and the scaling exponent beta is
            H = (beta-1) / 2

        Returns
        -------

        xfBm : array
            the simulated fractional Brownian Motion time series

        References
        ----------

        1. http://cours-physique.lps.ens.fr/index.php/TD11_Correlated_Noise_2011
        2. https://www.wikiwand.com/en/Fractional_Brownian_motion

        @authors: jeg, fzhu
        '''
        assert isinstance(N, int) and N >= 1
        assert H > 0 and H < 1, "H should be in (0, 1)!"

        HH = 2 * H

        ns = N-1  # number of steps
        covariance = np.ones((ns, ns))

        for i in range(ns):
            for j in range(i, ns):
                x = np.abs(i-j)
                covariance[i, j] = covariance[j, i] = (np.abs(x-1)**HH + (x+1)**HH - 2*x**HH) / 2.

        w, v = np.linalg.eig(covariance)

        A = np.zeros((ns, ns))
        for i in range(ns):
            for j in range(i, ns):
                A[i, j] = A[j, i] = np.sum(np.sqrt(w) * v[i, :] * v[j, :])

        xi = np.random.randn((ns))
        eta = np.dot(A, xi)

        xfBm = np.zeros(N)
        xfBm[0] = 0
        for i in range(1, N):
            xfBm[i] = xfBm[i-1] + eta[i-1]

        return xfBm

    def psd_fBM(self, freq, ts, H):
        ''' Return the theoretical psd of a fBM

        Args
        ----

        freq : array
            vector of frequency
        ts : array
            the time axis of the time series
        H : float
            Hurst index, should be in (0, 1)

        Returns
        --------

        psd : array
            power spectral density

        References
        ----------

        Flandrin, P. On the spectrum of fractional Brownian motions.
            IEEE Transactions on Information Theory 35, 197–199 (1989).

        '''
        nf = np.size(freq)
        psd = np.ndarray(shape=(nf))
        T = np.max(ts) - np.min(ts)

        omega = 2 * np.pi * freq

        for k in range(nf):
            tmp = 2 * omega[k] * T
            psd[k] = (1 - 2**(1 - 2*H)*np.sin(tmp)/tmp) / np.abs(omega[k])**(1 + 2*H)

        return psd

    def get_wwz_func(self, nproc, method):
        ''' Return the wwz function to use.

        Args
        ----

        nproc : int
            the number of processes for multiprocessing
        method : string
            'Foster' - the original WWZ method;
            'Kirchner' - the method Kirchner adapted from Foster;
            'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py (default)

        Returns
        -------

        wwz_func : function
            the wwz function to use

        '''
        wa = WaveletAnalysis()
        wa.assertPositiveInt(nproc)

        if method == 'Foster':
            if nproc == 1:
                wwz_func = wa.wwz_basic
            else:
                wwz_func = wa.wwz_nproc

        elif method == 'Kirchner':
            if nproc == 1:
                wwz_func = wa.kirchner_basic
            else:
                wwz_func = wa.kirchner_nproc
        elif method == 'Kirchner_f2py':
            wwz_func = wa.kirchner_f2py
        else:
            # default method; Kirchner's algorithm with Numba support for acceleration
            wwz_func = wa.kirchner_numba

        return wwz_func

    def prepare_wwz(self, ys, ts, freq=None, tau=None, len_bd=0, bc_mode='reflect', reflect_type='odd', **kwargs):
        ''' Return the truncated time series with NaNs deleted and estimate frequency vector and tau

        Args
        ----

        ys : array
            a time series, NaNs will be deleted automatically
        ts : array
            the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
        freq : array
            vector of frequency. If None, use the nfft method.If using Lomb-Scargle, additional parameters
            may be set. See make_freq_vector
        tau : array
            the evenly-spaced time points, namely the time shift for wavelet analysis
            if the boundaries of tau are not exactly on two of the time axis points, then tau will be adjusted to be so
        len_bd : int
            the number of the ghost grids want to create on each boundary
        bc_mode : string
            {'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect' , 'symmetric', 'wrap'}
            For more details, see np.lib.pad()
        reflect_type : string
             {‘even’, ‘odd’}, optional
             Used in ‘reflect’, and ‘symmetric’. The ‘even’ style is the default with an unaltered reflection around the edge value.
             For the ‘odd’ style, the extented part of the array is created by subtracting the reflected values from two times the edge value.
             For more details, see np.lib.pad()

        Returns
        -------

        ys_cut : array
            the truncated time series with NaNs deleted
        ts_cut : array
            the truncated time axis of the original time series with NaNs deleted
        freq : array
            vector of frequency
        tau : array
            the evenly-spaced time points, namely the time shift for wavelet analysis

        '''
        ys, ts = Timeseries.clean_ts(ys, ts)

        if tau is None:
            med_res = np.size(ts) // np.median(np.diff(ts))
            tau = np.linspace(np.min(ts), np.max(ts), np.max([np.size(ts)//10, 50, med_res]))

        elif np.isnan(tau).any():
            warnings.warn("The input tau contains some NaNs." +
                          "It will be regenerated using the boundarys of the time axis of the time series with NaNs deleted," +
                          "with the length of the size of the input tau.")
            tau = np.linspace(np.min(ts), np.max(ts), np.size(tau))

        elif np.min(tau) < np.min(ts) and np.max(tau) > np.max(ts):
            warnings.warn("tau should be within the time span of the time series." +
                          "Note that sometimes if the leading points of the time series are NaNs," +
                          "they will be deleted and cause np.min(tau) < np.min(ts)." +
                          "A new tau with the same size of the input tau will be generated.")
            tau = np.linspace(np.min(ts), np.max(ts), np.size(tau))

        elif np.min(tau) not in ts or np.max(tau) not in ts:
            warnings.warn("The boundaries of tau are not exactly on two of the time axis points," +
                          "and it will be adjusted to be so.")
            tau_lb = np.min(ts[ts > np.min(tau)])
            tau_ub = np.max(ts[ts < np.max(tau)])
            tau = np.linspace(tau_lb, tau_ub, np.size(tau))

        # boundary condition
        if len_bd > 0:
            dt = np.median(np.diff(ts))
            dtau = np.median(np.diff(tau))
            len_bd_tau = len_bd*dt//dtau

            if bc_mode in ['reflect', 'symmetric']:
                ys = np.lib.pad(ys, (len_bd, len_bd), bc_mode, reflect_type=reflect_type)
            else:
                ys = np.lib.pad(ys, (len_bd, len_bd), bc_mode)

            ts_left_bd = np.linspace(ts[0]-dt*len_bd, ts[0]-dt, len_bd)
            ts_right_bd = np.linspace(ts[-1]+dt, ts[-1]+dt*len_bd, len_bd)
            ts = np.concatenate((ts_left_bd, ts, ts_right_bd))

            warnings.warn("The tau will be regenerated to fit the boundary condition.")
            tau_left_bd = np.linspace(tau[0]-dtau*len_bd_tau, tau[0]-dtau, len_bd_tau)
            tau_right_bd = np.linspace(tau[-1]+dtau, tau[-1]+dtau*len_bd_tau, len_bd_tau)
            tau = np.concatenate((tau_left_bd, tau, tau_right_bd))

        # truncate the time series when the range of tau is smaller than that of the time series
        ts_cut = ts[(np.min(tau) <= ts) & (ts <= np.max(tau))]
        ys_cut = ys[(np.min(tau) <= ts) & (ts <= np.max(tau))]

        if freq is None:
            freq = self.make_freq_vector(ts_cut, method='nfft')

        # remove 0 in freq vector
        freq = freq[freq != 0]

        return ys_cut, ts_cut, freq, tau

    def cross_wt(self, coeff1, coeff2):
        ''' Return the cross wavelet transform.

        Args
        ----

        coeff1 : array
            the first of two sets of wavelet transform coefficients **in the form of a1 + a2*1j**
        coeff2 : array
            the second of two sets of wavelet transform coefficients **in the form of a1 + a2*1j**
        freq : array
            vector of frequency
        tau : array'
            the evenly-spaced time points, namely the time shift for wavelet analysis

        Returns
        -------

        xw_amplitude : array
            the cross wavelet amplitude
        xw_phase : array
            the cross wavelet phase

        References
        ----------

        1.Grinsted, A., Moore, J. C. & Jevrejeva, S. Application of the cross wavelet transform and
            wavelet coherence to geophysical time series. Nonlin. Processes Geophys. 11, 561–566 (2004).

        '''
        xwt = coeff1 * np.conj(coeff2)
        xw_amplitude = np.sqrt(xwt.real**2 + xwt.imag**2)
        xw_phase = np.arctan2(xwt.imag, xwt.real)

        return xwt, xw_amplitude, xw_phase

    def wavelet_coherence(self, coeff1, coeff2, freq, tau, smooth_factor=0.25):
        ''' Return the cross wavelet coherence.

        Args
        ----

        coeff1 : array
            the first of two sets of wavelet transform coefficients **in the form of a1 + a2*1j**
        coeff2 : array
            the second of two sets of wavelet transform coefficients **in the form of a1 + a2*1j**
        freq : array
            vector of frequency
        tau : array'
            the evenly-spaced time points, namely the time shift for wavelet analysis

        Returns
        -------

        xw_coherence : array
            the cross wavelet coherence

        References
        ----------

        1. Grinsted, A., Moore, J. C. & Jevrejeva, S. Application of the cross wavelet transform and
            wavelet coherence to geophysical time series. Nonlin. Processes Geophys. 11, 561–566 (2004).
        2. Matlab code by Grinsted (https://github.com/grinsted/wavelet-coherence)
        3. Python code by Sebastian Krieger (https://github.com/regeirk/pycwt)

        '''
        def rect(length, normalize=False):
            """ Rectangular function adapted from https://github.com/regeirk/pycwt/blob/master/pycwt/helpers.py

            Args:
                length (int): length of the rectangular function
                normalize (bool): normalize or not

            Returns:
                rect (array): the (normalized) rectangular function

            """
            rect = np.zeros(length)
            rect[0] = rect[-1] = 0.5
            rect[1:-1] = 1

            if normalize:
                rect /= rect.sum()

            return rect

        def smoothing(coeff, snorm, dj, smooth_factor=smooth_factor):
            """ Soothing function adapted from https://github.com/regeirk/pycwt/blob/master/pycwt/helpers.py

            Args
            ----

            coeff : array
                the wavelet coefficients get from wavelet transform **in the form of a1 + a2*1j**
            snorm : array
                normalized scales
            dj : float
                it satisfies the equation [ Sj = S0 * 2**(j*dj) ]

            Returns
            -------

            rect : array
                the (normalized) rectangular function

            """
            def fft_kwargs(signal, **kwargs):
                return {'n': np.int(2 ** np.ceil(np.log2(len(signal))))}

            W = coeff.transpose()
            m, n = np.shape(W)

            # Smooth in time
            k = 2 * np.pi * fft.fftfreq(fft_kwargs(W[0, :])['n'])
            k2 = k ** 2
            # Notes by Smoothing by Gaussian window (absolute value of wavelet function)
            # using the convolution theorem: multiplication by Gaussian curve in
            # Fourier domain for each scale, outer product of scale and frequency
            F = np.exp(-smooth_factor * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product
            smooth = fft.ifft(F * fft.fft(W, axis=1, **fft_kwargs(W[0, :])),
                              axis=1,  # Along Fourier frequencies
                              **fft_kwargs(W[0, :], overwrite_x=True))
            T = smooth[:, :n]  # Remove possibly padded region due to FFT
            if np.isreal(W).all():
                T = T.real

            # Smooth in scale
            wsize = 0.6 / dj * 2
            win = rect(np.int(np.round(wsize)), normalize=True)
            T = signal.convolve2d(T, win[:, np.newaxis], 'same')
            S = T.transpose()

            return S

        xwt = coeff1 * np.conj(coeff2)
        power1 = np.abs(coeff1)**2
        power2 = np.abs(coeff2)**2

        scales = 1/freq  # `scales` here is the `Period` axis in the wavelet plot
        dt = np.median(np.diff(tau))
        snorm = scales / dt  # normalized scales

        # with WWZ method, we don't have a constant dj, so we will just take the average over the whole scale range
        N = np.size(scales)
        s0 = scales[-1]
        sN = scales[0]
        dj = np.log2(sN/s0) / N

        S12 = Smoothing(xwt/scales, snorm, dj)
        S1 = Smoothing(power1/scales, snorm, dj)
        S2 = Smoothing(power2/scales, snorm, dj)
        xw_coherence = np.abs(S12)**2 / (S1*S2)
        wcs = S12 / (np.sqrt(S1)*np.sqrt(S2))
        xw_phase = np.angle(wcs)

        return xw_coherence, xw_phase

    def reconstruct_ts(self, coeff, freq, tau, t, len_bd=0):
        ''' Reconstruct the normalized time series from the wavelet coefficients.

        Args
        ----

        coeff : array
            the coefficients of the corresponding basis functions (a0, a1, a2)
        freq : array
            vector of frequency of the basis functions
        tau : array
            the evenly-spaced time points of the basis functions
        t : array
            the specified evenly-spaced time points of the reconstructed time series
        len_bd : int
            the number of the ghost grids want to creat on each boundary

        Returns
        -------

        rec_ts : array
            the reconstructed normalized time series
        t : array
            the evenly-spaced time points of the reconstructed time series
        '''
        omega = 2*np.pi*freq
        nf = np.size(freq)

        dt = np.median(np.diff(t))
        if len_bd > 0:
            t_left_bd = np.linspace(t[0]-dt*len_bd, t[0]-dt, len_bd)
            t_right_bd = np.linspace(t[-1]+dt, t[-1]+dt*len_bd, len_bd)
            t = np.concatenate((t_left_bd, t, t_right_bd))

        ntau = np.size(tau)
        a_0, a_1, a_2 = coeff

        rec_ts = np.zeros(np.size(t))
        for k in range(nf):
            for j in range(ntau):
                if np.isnan(a_0[j, k]) or np.isnan(a_1[j, k]) or np.isnan(a_1[j, k]):
                    continue
                else:
                    dz = omega[k] * (t - tau[j])
                    phi_1 = np.cos(dz)
                    phi_2 = np.sin(dz)

                    rec_ts += (a_0[j, k] + a_1[j, k]*phi_1 + a_2[j, k]*phi_2)

        rec_ts = self.preprocess(rec_ts, t, detrend=False, gaussianize=False, standardize=True)

        return rec_ts, t

    def wavelet_evenly():
        #TODO
        return


class AliasFilter(object):
    '''Performing anti-alias filter on a psd @author: fzhu
    '''

    def alias_filter(self, freq, pwr, fs, fc, f_limit, avgs):
        ''' anti_alias filter

        Args
        ----

        freq : array
            vector of frequencies in power spectrum
        pwr : array
            vector of spectral power corresponding to frequencies "freq"
        fs : float
            sampling frequency
        fc : float
            corner frequency for 1/f^2 steepening of power spectrum
        f_limit : float
            lower frequency limit for estimating misfit of model-plus-alias spectrum vs. measured power
        avgs : int
            flag for whether spectrum is derived from instantaneous point measurements (avgs<>1)
            OR from measurements averaged over each sampling interval (avgs==1)

        Returns
        -------

        alpha : float
            best-fit exponent of power-law model
        filtered_pwr : array
            vector of alias-filtered spectral power
        model_pwr : array
            vector of modeled spectral power
        aliased_pwr : array
            vector of modeled spectral power, plus aliases

        References
        ----------

        1. Kirchner, J. W. Aliasing in 1/f(alpha) noise spectra: origins, consequences, and remedies.
                Phys Rev E Stat Nonlin Soft Matter Phys 71, 66110 (2005).

        '''
        log_pwr = np.log(pwr)
        freq_mask = (freq > f_limit)*1  # convert True & False to 1 & 0

        alpha_upper_bound = 5

        if avgs == 1:
            alpha_lower_bound = -2.9  # if measurements are time-averaged
        else:
            alpha_lower_bound = -0.9  # if measurements are point samples

        alpha = optimize.fminbound(self.misfit, alpha_lower_bound, alpha_upper_bound,
                                   args=(fs, fc, freq, log_pwr, freq_mask, avgs), xtol=1e-4)

        model_pwr, aliased_pwr, RMSE = self.alias(alpha, fs, fc, freq, log_pwr, freq_mask, avgs)
        filtered_pwr = pwr * model_pwr / aliased_pwr

        return alpha, filtered_pwr, model_pwr, aliased_pwr

    def misfit(self, alpha, fs, fc, freq, log_pwr, freq_mask, avgs):
        model, aliased_pwr, RMSE = self.alias(alpha, fs, fc, freq, log_pwr, freq_mask, avgs)
        return RMSE

    def alias(self, alpha, fs, fc, freq, log_pwr, freq_mask, avgs):
        model_pwr = self.model(alpha, fs, fc, freq, avgs)
        aliased_pwr = np.copy(model_pwr)
        if avgs == 1:
            aliased_pwr = aliased_pwr * np.sinc(freq/fs) ** 2

        for k in range(1, 11):
            alias_minus = self.model(alpha, fs, fc, k*fs-freq, avgs)
            if avgs == 1:
                alias_minus = alias_minus * np.sinc((k*fs-freq)/fs) ** 2

            aliased_pwr = aliased_pwr + alias_minus

            alias_plus = self.model(alpha, fs, fc, k*fs+freq, avgs)  # notice the + in (k*fs+freq)
            if avgs == 1:
                alias_plus = alias_plus * np.sinc((k*fs+freq)/fs) ** 2

            aliased_pwr = aliased_pwr + alias_plus

        if avgs == 1:
            beta = alpha + 3
            const = 1 / (2*np.pi**2*beta/fs)
        else:
            beta = alpha + 1
            const = 1 / (beta*fs)

        zo_minus = (11*fs-freq)**(-beta)
        dz_minus = zo_minus / 20

        for j in range(1, 21):
            aliased_pwr = aliased_pwr + const / ((j*dz_minus)**(2/beta) + 1/fc**2)*dz_minus

        zo_plus = (11*fs+freq)**(-beta)
        dz_plus = zo_plus / 20

        for j in range(1, 21):
            aliased_pwr = aliased_pwr + const / ((j*dz_plus)**(2/beta) + 1/fc**2)*dz_plus

        log_aliased = np.log(aliased_pwr)

        prefactor = np.sum((log_pwr - log_aliased) * freq_mask) / np.sum(freq_mask)

        log_aliased = log_aliased + prefactor
        aliased_pwr = aliased_pwr * np.exp(prefactor)
        model_pwr = model_pwr * np.exp(prefactor)

        RMSE = np.sqrt(np.sum((log_aliased-log_pwr)*(log_aliased-log_pwr)*freq_mask)) / np.sum(freq_mask)

        return model_pwr, aliased_pwr, RMSE

    def model(self, alpha, fs, fc, freq, avgs):
        spectr = freq**(-alpha) / (1 + (freq/fc)**2)

        return spectr


class Filter(object):
    """Group various Filters under a class
    """
    @staticmethod
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

    @staticmethod
    def tsPad(ys,ts,method = 'reflect', params=(1,0,0), reflect_type = 'odd',padFrac=0.1):
        """ tsPad: pad a timeseries based on timeseries model predictions

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

    @staticmethod
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
            yp,tp = Filter.tsPad(ys,ts,method = 'ARIMA', params=params, padFrac=padFrac)
        elif pad=='reflect':
            yp,tp = Filter.tsPad(ys,ts,method = 'reflect', reflect_type=reflect_type, padFrac=padFrac)
        elif pad is None:
            yp = ys; tp = ts
        else:
            raise ValueError('Not a valid argument. Enter "ARIMA", "reflect" or None')

        ypf = signal.filtfilt(b, a, yp)
        yf  = ypf[np.isin(tp,ts)]

        return yf

class Correlation(object):
    """ Estimates the significance of correlations
    """
    def corr_sig(self, y1, y2, nsim=1000, method='isospectral', alpha=0.05):
        """ Estimates the significance of correlations between non IID time series by 3 independent methods:
        1) 'ttest': T-test where d.o.f are corrected for the effect of serial correlation
        2) 'isopersistent': AR(1) modeling of x and y.
        3) 'isospectral': phase randomization of original inputs. (default)
        The T-test is parametric test, hence cheap but usually wrong except in idyllic circumstances.
        The others are non-parametric, but their computational requirements scales with nsim.

        Args
        ----

        y1 : array
            vector of (real) numbers of same length as y2, no NaNs allowed
        y2 : array
            vector of (real) numbers of same length as y1, no NaNs allowed
        nsim : int
            the number of simulations [default: 1000]
        method : string
            methods 1-3 above [default: 'isospectral']
        alpha : float
            significance level for critical value estimation [default: 0.05]

        Returns
        -------

         r : float
             correlation between x and y
         signif : bool
             true (1) if significant; false (0) otherwise
         p : float
             Fraction of time series with higher correlation coefficents than observed (approximates the p-value).
             Note that signif = True if and only if p <= alpha.
        """
        y1 = np.array(y1, dtype=float)
        y2 = np.array(y2, dtype=float)

        assert np.size(y1) == np.size(y2), 'The size of X and the size of Y should be the same!'

        if method == 'ttest':
            (r, signif, p) = self.corr_ttest(y1, y2, alpha=alpha)
        elif method == 'isopersistent':
            (r, signif, p) = self.corr_isopersist(y1, y2, alpha=alpha, nsim=nsim)
        elif method == 'isospectral':
            (r, signif, p) = self.corr_isospec(y1, y2, alpha=alpha, nsim=nsim)

        return r, signif, p

    def corr_ttest(self, y1, y2, alpha=0.05):
        """ Estimates the significance of correlations between 2 time series using
        the classical T-test with degrees of freedom modified for autocorrelation.
        This function creates 'nsim' random time series that have the same power
        spectrum as the original time series but with random phases.

        Args
        ----

        y1 : array
            vectors of (real) numbers with identical length, no NaNs allowed
        y2 : array
            vectors of (real) numbers with identical length, no NaNs allowed
        alpha : float
            significance level for critical value estimation [default: 0.05]

        Returns
        -------

        r : float
             correlation between x and y
        signif : bool
            true (1) if significant; false (0) otherwise
        pval : float
            test p-value (the probability of the test statstic exceeding the observed one by chance alone)
        """
        r = pearsonr(y1, y2)[0]

        g1 = self.ar1_fit(y1)
        g2 = self.ar1_fit(y2)

        N = np.size(y1)

        Ney1 = N * (1-g1) / (1+g1)
        Ney2 = N * (1-g2) / (1+g2)

        Ne = gmean([Ney1+Ney2])
        assert Ne >= 10, 'Too few effective d.o.f. to apply this method!'

        df = Ne - 2
        t = np.abs(r) * np.sqrt(df/(1-r**2))

        pval = 2 * stu.cdf(-np.abs(t), df)

        signif = pval <= alpha

        return r, signif, pval

    def corr_isopersist(self, y1, y2, alpha=0.05, nsim=1000):
        ''' Computes correlation between two timeseries, and their significance.
        The latter is gauged via a non-parametric (Monte Carlo) simulation of
        correlations with nsim AR(1) processes with identical persistence
        properties as x and y ; the measure of which is the lag-1 autocorrelation (g).

        Args
        ----

        y1 : array
            vectors of (real) numbers with identical length, no NaNs allowed
        y2 : array
            vectors of (real) numbers with identical length, no NaNs allowed
        alpha : float
            significance level for critical value estimation [default: 0.05]
        nsim : int
            number of simulations [default: 1000]

        Returns
        -------

        r : float
            correlation between x and y
        signif : bool
            true (1) if significant; false (0) otherwise
        pval : float
            test p-value (the probability of the test statstic exceeding the observed one by chance alone)

        Notes
        -----

        The probability of obtaining a test statistic at least as extreme as the one actually observed,
        assuming that the null hypothesis is true.
        The test is 1 tailed on |r|: Ho = { |r| = 0 }, Ha = { |r| > 0 }
        The test is rejected (signif = 1) if pval <= alpha, otherwise signif=0;
        (Some Rights Reserved) Hepta Technologies, 2009
        v1.0 USC, Aug 10 2012, based on corr_signif.
        '''

        r = pearsonr(y1, y2)[0]
        ra = np.abs(r)

        y1_red, g1 = self.isopersistent_rn(y1, nsim)
        y2_red, g2 = self.isopersistent_rn(y2, nsim)

        rs = np.zeros(nsim)
        for i in np.arange(nsim):
            rs[i] = pearsonr(y1_red[:, i], y2_red[:, i])[0]

        rsa = np.abs(rs)

        xi = np.linspace(0, 1.1*np.max([ra, np.max(rsa)]), 200)
        kde = gaussian_kde(rsa)
        prob = kde(xi).T

        diff = np.abs(ra - xi)
        #  min_diff = np.min(diff)
        pos = np.argmin(diff)

        pval = np.trapz(prob[pos:], xi[pos:])

        rcrit = np.percentile(rsa, 100*(1-alpha))
        signif = ra >= rcrit

        return r, signif, pval

    def isopersistent_rn(self, X, p):
        ''' Generates p realization of a red noise [i.e. AR(1)] process
        with same persistence properties as X (Mean and variance are also preserved).

        Args
        ----

        X : array
            vector of (real) numbers as a time series, no NaNs allowed
        p : int
            number of simulations

        Returns
        -------

        red : numpy array
            n rows by p columns matrix of an AR1 process, where n is the size of X
        g :float
            lag-1 autocorrelation coefficient

        Notes
        -----

        (Some Rights Reserved) Hepta Technologies, 2008

        '''
        n = np.size(X)
        sig = np.std(X, ddof=1)

        g = self.ar1_fit(X)
        #  red = red_noise(N, M, g)
        red = self.ar1_sim(n, p, g, sig)

        return red, g

    def ar1_fit(self, ts):
        ''' Return the lag-1 autocorrelation from ar1 fit.

        Args
        ----

        ts : array
            vector of (real) numbers as a time series

        Returns
        -------

        g :float
            lag-1 autocorrelation coefficient
        '''


        ar1_mod = sm.tsa.AR(ts, missing='drop').fit(maxlag=1)
        g = ar1_mod.params[1]

        return g

    def ar1_sim(self, n, p, g, sig):
        ''' Produce p realizations of an AR1 process of length n with lag-1 autocorrelation g

        Args
        ----

        n : int
            row dimensions
        p : int
            column dimensions

        g : float
            lag-1 autocorrelation coefficient
        sig : float
            the standard deviation of the original time series

        Returns
        -------

        red : numpy matrix
            n rows by p columns matrix of an AR1 process
        '''
        # specify model parameters (statsmodel wants lag0 coefficents as unity)
        ar = np.r_[1, -g]  # AR model parameter
        ma = np.r_[1, 0.0] # MA model parameters
        sig_n = sig*np.sqrt(1-g**2) # theoretical noise variance for red to achieve the same variance as X

        red = np.empty(shape=(n, p)) # declare array

        # simulate AR(1) model for each column
        for i in np.arange(p):
            red[:, i] = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=50, sigma=sig_n)

        return red

    def red_noise(self, N, M, g):
        ''' Produce M realizations of an AR1 process of length N with lag-1 autocorrelation g

        Args
        ----

        N : int
            row dimensions
        M : int
            column dimensions
        g : float
            lag-1 autocorrelation coefficient

        Returns
        -------

        red : numpy array
            N rows by M columns matrix of an AR1 process

        Notes
        -----

        (Some Rights Reserved) Hepta Technologies, 2008
        J.E.G., GaTech, Oct 20th 2008
        '''
        red = np.zeros(shape=(N, M))
        red[0, :] = np.random.randn(1, M)
        for i in np.arange(1, N):
            red[i, :] = g * red[i-1, :] + np.random.randn(1, M)

        return red

    def corr_isospec(self, y1, y2, alpha=0.05, nsim=1000):
        ''' Phase randomization correltation estimates

        Estimates the significance of correlations between non IID
        time series by phase randomization of original inputs.
        This function creates 'nsim' random time series that have the same power
        spectrum as the original time series but random phases.

        Args
        ----

        y1 : array
            vectors of (real) numbers with identical length, no NaNs allowed
        y2 : array
            vectors of (real) numbers with identical length, no NaNs allowed
        alpha : float
            significance level for critical value estimation [default: 0.05]
        nsim : int
            number of simulations [default: 1000]

        Returns
        -------

        r : float
            correlation between y1 and y2
        signif : bool
            true (1) if significant; false (0) otherwise
        F : float
            Fraction of time series with higher correlation coefficents than observed (approximates the p-value).

        Reference
        ---------

        - Ebisuzaki, W, 1997: A method to estimate the statistical
        significance of a correlation when the data are serially correlated.
        J. of Climate, 10, 2147-2153.
        - Prichard, D., Theiler, J. Generating Surrogate Data for Time Series
        with Several Simultaneously Measured Variables (1994)
        Physical Review Letters, Vol 73, Number 7
        (Some Rights Reserved) USC Climate Dynamics Lab, 2012.
        '''
        r = pearsonr(y1, y2)[0]

        # generate phase-randomized samples using the Theiler & Prichard method
        Y1surr = self.phaseran(y1, nsim)
        Y2surr = self.phaseran(y2, nsim)

        # compute correlations
        Y1s = preprocessing.scale(Y1surr)
        Y2s = preprocessing.scale(Y2surr)

        n = np.size(y1)
        C = np.dot(np.transpose(Y1s), Y2s) / (n-1)
        rSim = np.diag(C)

        # compute fraction of values higher than observed
        F = np.sum(np.abs(rSim) >= np.abs(r)) / nsim

        # establish significance
        signif = F < alpha  # significant or not?

        return r, signif, F

    def phaseran(self, recblk, nsurr):
        ''' Phaseran by Carlos Gias

        http://www.mathworks.nl/matlabcentral/fileexchange/32621-phase-randomization/content/phaseran.m

        Args
        ----

        recblk : numpy array
            2D array , Row: time sample. Column: recording.
            An odd number of time samples (height) is expected.
            If that is not the case, recblock is reduced by 1 sample before the surrogate data is created.
            The class must be double and it must be nonsparse.
        nsurr : int
            is the number of image block surrogates that you want to generate.

        Returns
        -------

        surrblk : numpy array
            3D multidimensional array image block with the surrogate datasets along the third dimension

        Reference
        ---------

        Prichard, D., Theiler, J. Generating Surrogate Data for Time Series with Several Simultaneously Measured Variables (1994)
        Physical Review Letters, Vol 73, Number 7
        '''
        # Get parameters
        nfrms = recblk.shape[0]

        if nfrms % 2 == 0:
            nfrms = nfrms-1
            recblk = recblk[0:nfrms]

        len_ser = int((nfrms-1)/2)
        interv1 = np.arange(1, len_ser+1)
        interv2 = np.arange(len_ser+1, nfrms)

        # Fourier transform of the original dataset
        fft_recblk = np.fft.fft(recblk)

        surrblk = np.zeros((nfrms, nsurr))

        #  for k in tqdm(np.arange(nsurr)):
        for k in np.arange(nsurr):
            ph_rnd = np.random.rand(len_ser)

            # Create the random phases for all the time series
            ph_interv1 = np.exp(2*np.pi*1j*ph_rnd)
            ph_interv2 = np.conj(np.flipud(ph_interv1))

            # Randomize all the time series simultaneously
            fft_recblk_surr = np.copy(fft_recblk)
            fft_recblk_surr[interv1] = fft_recblk[interv1] * ph_interv1
            fft_recblk_surr[interv2] = fft_recblk[interv2] * ph_interv2

            # Inverse transform
            surrblk[:, k] = np.real(np.fft.ifft(fft_recblk_surr))

        return surrblk

class Causality(object):
    def granger_causality(self,y1, y2, maxlag=1,addconst=True,verbose=True):
        '''
        statsmodels granger causality tests

        Four tests for granger non causality of 2 time series.

        All four tests give similar results. params_ftest and ssr_ftest are equivalent based on F test which is identical to lmtest:grangertest in R.

        Args
        ----

        x : array
            The data for test whether the time series in the second column Granger causes the time series in the first column. Missing values are not supported.
        maxlag : int or int iterable
            If an integer, computes the test for all lags up to maxlag. If an iterable, computes the tests only for the lags in maxlag.
        addconst : bool
            Include a constant in the model.
        verbose : bool
            Print results

        Returns
        -------

        dict
            All test results, dictionary keys are the number of lags. For each lag the values are a tuple, with the first element a dictionary with test statistic,
            pvalues, degrees of freedom, the second element are the OLS estimation results for the restricted model, the unrestricted model and the restriction (contrast)
            matrix for the parameter f_test.
        '''

        if len(y1)!=len(y2):
            raise ValueError('Timeseries must be of same length')

        x=np.matrix([y1,y2]).T
        return grangercausalitytests(x,maxlag=maxlag,addconst=addconst,verbose=verbose)

    def liang_causality(self, y1, y2, npt=1):
        '''
        Estimate the Liang information transfer from series y2 to series y1


        Args
        ----

        y1, y2 : array
            vectors of (real) numbers with identical length, no NaNs allowed

        npt : int  >=1
            time advance in performing Euler forward differencing,
            e.g., 1, 2. Unless the series are generated with a highly chaotic deterministic system,
            npt=1 should be used

        Returns
        -------

        T21 : float
            info flow from y2 to y1 (Note: not y1 -> y2!)
        tau21 : float
            the standardized info flow fro y2 to y1
        Z : float
            the total info

        References
        ----------

        - Liang, X.S. (2013) The Liang-Kleeman Information Flow: Theory and
                Applications. Entropy, 15, 327-360, doi:10.3390/e15010327
        - Liang, X.S. (2014) Unraveling the cause-efect relation between timeseries.
            Physical review, E 90, 052150
        - Liang, X.S. (2015) Normalizing the causality between time series.
            Physical review, E 92, 022126
        - Liang, X.S. (2016) Information flow and causality as rigorous notions ab initio.
            Physical review, E 94, 052201

        '''
        dt=1
        nm = np.size(y1)

        grad1 = (y1[0+npt:] - y1[0:-npt]) / (npt)
        grad2 = (y2[0+npt:] - y2[0:-npt]) / (npt)

        y1 = y1[:-npt]
        y2 = y2[:-npt]

        N = nm - npt
        C = np.cov(y1, y2)
        detC = np.linalg.det(C)

        dC = np.ndarray((2, 2))
        dC[0, 0] = np.sum((y1-np.mean(y1))*(grad1-np.mean(grad1)))
        dC[0, 1] = np.sum((y1-np.mean(y1))*(grad2-np.mean(grad2)))
        dC[1, 0] = np.sum((y2-np.mean(y2))*(grad1-np.mean(grad1)))
        dC[1, 1] = np.sum((y2-np.mean(y2))*(grad2-np.mean(grad2)))

        dC /= N-1

        a11 = C[1, 1]*dC[0, 0] - C[0, 1]*dC[1, 0]
        a12 = -C[0, 1]*dC[0, 0] + C[0, 0]*dC[1, 0]

        a11 /= detC
        a12 /= detC

        f1 = np.mean(grad1) - a11*np.mean(y1) - a12*np.mean(y2)
        R1 = grad1 - (f1 + a11*y1 + a12*y2)
        Q1 = np.sum(R1*R1)
        b1 = np.sqrt(Q1*dt/N)

        NI = np.ndarray((4, 4))
        NI[0, 0] = N*dt/b1**2
        NI[1, 1] = dt/b1**2*np.sum(y1*y1)
        NI[2, 2] = dt/b1**2*np.sum(y2*y2)
        NI[3, 3] = 3*dt/b1**4*np.sum(R1*R1) - N/b1**2
        NI[0, 1] = dt/b1**2*np.sum(y1)
        NI[0, 2] = dt/b1**2*np.sum(y2)
        NI[0, 3] = 2*dt/b1**3*np.sum(R1)
        NI[1, 2] = dt/b1**2*np.sum(y1*y2)
        NI[1, 3] = 2*dt/b1**3*np.sum(R1*y1)
        NI[2, 3] = 2*dt/b1**3*np.sum(R1*y2)

        NI[1, 0] = NI[0, 1]
        NI[2, 0] = NI[0, 2]
        NI[2, 1] = NI[1, 2]
        NI[3, 0] = NI[0, 3]
        NI[3, 1] = NI[1, 3]
        NI[3, 2] = NI[2, 3]

        invNI = np.linalg.pinv(NI)
        var_a12 = invNI[2, 2]
        T21 = C[0, 1]/C[0, 0] * (-C[1, 0]*dC[0, 0] + C[0, 0]*dC[1, 0]) / detC
        var_T21 = (C[0, 1]/C[0, 0])**2 * var_a12

        dH1_star= a11
        dH1_noise = b1**2 / (2*C[0, 0])

        Z = np.abs(T21) + np.abs(dH1_star) + np.abs(dH1_noise)

        tau21 = T21 / Z
        dH1_star = dH1_star / Z
        dH1_noise = dH1_noise / Z

        res_dict = {
            'T21': T21,
            'tau21': tau21,
            'Z': Z,
            'dH1_star': dH1_star,
            'dH1_noise': dH1_noise
        }

        return res_dict

    def signif_isopersist(self, y1, y2, method='liang',
                          nsim=1000, qs=[0.005, 0.025, 0.05, 0.95, 0.975, 0.995],
                          **kwargs):
        ''' significance test with AR(1) with same persistence

        Args
        ----

        y1, y2 : array
            vectors of (real) numbers with identical length, no NaNs allowed
        method : str
            only "liang" for now
        npt : int>=1
            time advance in performing Euler forward differencing,
            e.g., 1, 2. Unless the series are generated with a highly chaotic deterministic system,
            npt=1 should be used.
        nsim : int
            the number of AR(1) surrogates for significance test
        qs : list
            the quantiles for significance test

        Returns
        -------

        res_dict : dict
            A dictionary with the following information:
              T21_noise_qs : list
                the quantiles of the information flow from noise2 to noise1 for significance testing
              tau21_noise_qs : list
                the quantiles of the standardized information flow from noise2 to noise1 for significance testing

        '''
        stat = stats.Correlation()
        g1 = stat.ar1_fit(y1)
        g2 = stat.ar1_fit(y2)
        sig1 = np.std(y1)
        sig2 = np.std(y2)
        n = np.size(y1)
        noise1 = stat.ar1_sim(n, nsim, g1, sig1)
        noise2 = stat.ar1_sim(n, nsim, g2, sig2)

        if method == 'liang':
            npt = kwargs['npt'] if 'npt' in kwargs else 1
            T21_noise = []
            tau21_noise = []
            for i in tqdm(range(nsim), desc='Calculating causality between surrogates'):
                res_noise = self.liang_causality(noise1[:, i], noise2[:, i], npt=npt)
                tau21_noise.append(res_noise['tau21'])
                T21_noise.append(res_noise['T21'])
            tau21_noise = np.array(tau21_noise)
            T21_noise = np.array(T21_noise)
            tau21_noise_qs = mquantiles(tau21_noise, qs)
            T21_noise_qs = mquantiles(T21_noise, qs)

            res_dict = {
                'tau21_noise_qs': tau21_noise_qs,
                'T21_noise_qs': T21_noise_qs,
            }
        else:
            raise KeyError(f'{method} is not a valid method')

        return res_dict

    def signif_isospec(self, y1, y2, method='liang',
                       nsim=1000, qs=[0.005, 0.025, 0.05, 0.95, 0.975, 0.995],
                       **kwargs):
        ''' significance test with surrogates with randomized phases

        Args
        ----

        y1, y2 : array
                vectors of (real) numbers with identical length, no NaNs allowed
        method : str
                only "liang" for now
        npt : int>=1
             time advance in performing Euler forward differencing,
             e.g., 1, 2. Unless the series are generated with a highly chaotic deterministic system,
             npt=1 should be used.
        nsim : int
              the number of surrogates for significance test
        qs : list
            the quantiles for significance test

        Returns
        -------

        res_dict : dict
            A dictionary with the following information:
              T21_noise_qs : list
                            the quantiles of the information flow from noise2 to noise1 for significance testing
              tau21_noise_qs : list
                              the quantiles of the standardized information flow from noise2 to noise1 for significance testing
        '''
        corr_obj = Correlation()
        noise1 = corr_obj.phaseran(y1, nsim)
        noise2 = corr_obj.phaseran(y2, nsim)

        if method == 'liang':
            npt = kwargs['npt'] if 'npt' in kwargs else 1
            T21_noise = []
            tau21_noise = []
            for i in tqdm(range(nsim), desc='Calculating causality between surrogates'):
                res_noise = self.liang_causality(noise1[:, i], noise2[:, i], npt=npt)
                tau21_noise.append(res_noise['tau21'])
                T21_noise.append(res_noise['T21'])
            tau21_noise = np.array(tau21_noise)
            T21_noise = np.array(T21_noise)
            tau21_noise_qs = mquantiles(tau21_noise, qs)
            T21_noise_qs = mquantiles(T21_noise, qs)

            res_dict = {
                'tau21_noise_qs': tau21_noise_qs,
                'T21_noise_qs': T21_noise_qs,
            }
        else:
            raise KeyError(f'{method} is not a valid method')

        return res_dict


class Decomposition(object):
    def pca(x,n_components=None,copy=True,whiten=False, svd_solver='auto',tol=0.0,iterated_power='auto',random_state=None):
        '''
        scikit-learn PCA

        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

        Args
        ----
        x : array
            timeseries
        n_components : int,None,or str
             [default: None]
            Number of components to keep. if n_components is not set all components are kept:
            If n_components == 'mle' and svd_solver == 'full', Minka’s MLE is used to guess the dimension. Use of n_components == 'mle' will interpret svd_solver == 'auto' as svd_solver == 'full'.
            If 0 < n_components < 1 and svd_solver == 'full', select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.
            If svd_solver == 'arpack', the number of components must be strictly less than the minimum of n_features and n_samples.
        copy : bool,optional
            [default: True]
            If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, use fit_transform(X) instead.
        whiten : bool,optional
            [default: False]
            When True (False by default) the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
        svd_solver : str {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
            If auto :
                The solver is selected by a default policy based on X.shape and n_components: if the input data is larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension of the data, then the more efficient ‘randomized’ method is enabled.
                Otherwise the exact full SVD is computed and optionally truncated afterwards.

            If full :
                run exact full SVD calling the standard LAPACK solver via scipy.linalg.svd and select the components by postprocessing

            If arpack :
                run SVD truncated to n_components calling ARPACK solver via scipy.sparse.linalg.svds. It requires strictly 0 < n_components < min(X.shape)

            If randomized :
                run randomized SVD by the method of Halko et al.
        tol : float >= 0 ,optional
            [default: 0]
            Tolerance for singular values computed by svd_solver == ‘arpack’.
        iterated_power : int >= 0, or string {'auto'}
            [default: 'auto']
            Number of iterations for the power method computed by svd_solver == ‘randomized’.
        random_state : int, RandomState instance, or None, optional
            [default: None]
            If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
            Used when svd_solver == ‘arpack’ or ‘randomized’.

        Returns
        -------

        dict
            Sklearn PCA object dictionary of all attributes and values.


        '''
        if np.any(np.isnan(x)):
            raise ValueError('matrix may not have null values.')
        pca=PCA(n_components=n_components,copy=copy,whiten=whiten,svd_solver=svd_solver,tol=tol,iterated_power=iterated_power,random_state=random_state)
        return pca.fit(x).__dict__

    def ssa(self, ys, ts, M, MC=1000, f=0.3, method='SSA', prep_args={}):
        '''
        Args
        ----

        ys : array
            time series
        ts: array
           time axis
        M : int
           window size
        MC : int
            Number of iteration in the Monte-Carlo process
        f : float
           fraction (0<f<=1) of good data points for identifying
        method (str, {'SSA', 'MSSA'}) : str({'SSA','MSSA'})
                                       perform SSA or MSSA

        prep_args : dict
                  the arguments for preprocess, including
                    detrend : str
                             'none' - the original time series is assumed to have no trend;
                             'linear' - a linear least-squares fit to `ys` is subtracted;
                             'constant' - the mean of `ys` is subtracted
                             'savitzy-golay' - ys is filtered using the Savitzky-Golay
                                                 filters and the resulting filtered series is subtracted from y.
                              'hht' - detrending with Hilbert-Huang Transform
                    params  : list
                               The paramters for the Savitzky-Golay filters. The first parameter
                               corresponds to the window size (default it set to half of the data)
                               while the second parameter correspond to the order of the filter
                              (default is 4). The third parameter is the order of the derivative
                              (the default is zero, which means only smoothing.)
                    gaussianize : bool
                                 If True, gaussianizes the timeseries
                    standardize : bool
                                 If True, standardizes the timeseries

        Returns
        -------

        res_dict : dictionary
                  the result dictionary, including
                     deval : array
                            eigenvalue spectrum
                     eig_vec : array
                              eigenvalue vector
                     q05 : float
                          The 5% percentile of eigenvalues
                     q95 : float
                          The 95% percentile of eigenvalues
                     pc: 2D array
                        matrix of principal components
                     rc: 2D array
                        matrix of RCs (nrec,N,nrec*M) (only if K>0)
        '''

        wa = WaveletAnalysis()
        ys, ts = Timeseries.clean_ts(ys, ts)
        ys = wa.preprocess(ys, ts, **prep_args)

        ssa_func = {
            'SSA': self.ssa_all,
            'MSSA': self.MSSA,
        }
        deval, eig_vec, q05, q95, pc, rc = ssa_func[method](ys, M, MC=MC, f=f)

        res_dict = {
            'deval': deval,
            'eig_vec': eig_vec,
            'q05': q05,
            'q95': q95,
            'pc': pc,
            'rc': rc,
        }

        return res_dict

    def standardize(self, x):
        if np.any(np.isnan(x)):
            x_ex = x[np.logical_not(np.isnan(x))]
            xm = np.mean(x_ex)
            xs = np.std(x_ex, ddof=1)
        else:
            xm = np.mean(x)
            xs = np.std(x, ddof=1)
        xstd = (x - xm) / xs
        return xstd

    def mssa(self, data, M, MC=1000, f=0.3):
        '''Multi-channel SSA analysis
        (applicable for data including missing values)
        and test the significance by Monte-Carlo method

        Args
        ----

        data : array
              multiple time series (dimension: length of time series x total number of time series)
        M : int
           window size
        MC : int
           Number of iteration in the Monte-Carlo process
        f : float
           fraction (0<f<=1) of good data points for identifying significant PCs [f = 0.3]

        Returns
        -------

        deval : array
               eigenvalue spectrum
        q05 : float
             The 5% percentile of eigenvalues
        q95 : float
             The 95% percentile of eigenvalues
        PC : 2D array
             matrix of principal components
        RC : 2D array
            matrix of RCs (nrec,N,nrec*M) (only if K>0)

        '''
        N = len(data[:, 0])
        nrec = len(data[0, :])
        Y = np.zeros((N - M + 1, nrec * M))
        for irec in np.arange(nrec):
            for m in np.arange(0, M):
                Y[:, m + irec * M] = data[m:N - M + 1 + m, irec]

        C = np.dot(np.nan_to_num(np.transpose(Y)), np.nan_to_num(Y)) / (N - M + 1)
        eig_val, eig_vec = eigh(C)

        sort_tmp = np.sort(eig_val)
        deval = sort_tmp[::-1]
        sortarg = np.argsort(-eig_val)

        eig_vec = eig_vec[:, sortarg]

        # test the signifiance using Monte-Carlo
        Ym = np.zeros((N - M + 1, nrec * M))
        noise = np.zeros((nrec, N, MC))
        for irec in np.arange(nrec):
            noise[irec, 0, :] = data[0, irec]
        Lamda_R = np.zeros((nrec * M, MC))
        # estimate coefficents of ar1 processes, and then generate ar1 time series (noise)
        for irec in np.arange(nrec):
            Xr = data[:, irec]
            coefs_est, var_est = alg.AR_est_YW(Xr[~np.isnan(Xr)], 1)
            sigma_est = np.sqrt(var_est)

            for jt in range(1, N):
                noise[irec, jt, :] = coefs_est * noise[irec, jt - 1, :] + sigma_est * np.random.randn(1, MC)

        for m in range(MC):
            for irec in np.arange(nrec):
                noise[irec, :, m] = (noise[irec, :, m] - np.mean(noise[irec, :, m])) / (
                    np.std(noise[irec, :, m], ddof=1))
                for im in np.arange(0, M):
                    Ym[:, im + irec * M] = noise[irec, im:N - M + 1 + im, m]
            Cn = np.dot(np.nan_to_num(np.transpose(Ym)), np.nan_to_num(Ym)) / (N - M + 1)
            # Lamda_R[:,m] = np.diag(np.dot(np.dot(eig_vec,Cn),np.transpose(eig_vec)))
            Lamda_R[:, m] = np.diag(np.dot(np.dot(np.transpose(eig_vec), Cn), eig_vec))

        q95 = np.percentile(Lamda_R, 95, axis=1)
        q05 = np.percentile(Lamda_R, 5, axis=1)


        # determine principal component time series
        PC = np.zeros((N - M + 1, nrec * M))
        PC[:, :] = np.nan
        for k in np.arange(nrec * M):
            for i in np.arange(0, N - M + 1):
                #   modify for nan
                prod = Y[i, :] * eig_vec[:, k]
                ngood = sum(~np.isnan(prod))
                #   must have at least m*f good points
                if ngood >= M * f:
                    PC[i, k] = sum(prod[~np.isnan(prod)])  # the columns of this matrix are Ak(t), k=1 to M (T-PCs)

        # compute reconstructed timeseries
        Np = N - M + 1

        RC = np.zeros((nrec, N, nrec * M))

        for k in np.arange(nrec):
            for im in np.arange(M):
                x2 = np.dot(np.expand_dims(PC[:, im], axis=1), np.expand_dims(eig_vec[0 + k * M:M + k * M, im], axis=0))
                x2 = np.flipud(x2)

                for n in np.arange(N):
                    RC[k, n, im] = np.diagonal(x2, offset=-(Np - 1 - n)).mean()

        return deval, eig_vec, q95, q05, PC, RC

    def ssa_all(self, data, M, MC=1000, f=0.3):
        '''SSA analysis for a time series
        (applicable for data including missing values)
        and test the significance by Monte-Carlo method

        Args
        ----

        data : array
              time series
        M : int
           window size
        MC : int
            Number of iteration in the Monte-Carlo process
        f : fraction
           fraction (0<f<=1) of good data points for identifying significant PCs [f = 0.3]

        Returns
        -------

        deval : array
               eigenvalue spectrum
        q05 : float
             The 5% percentile of eigenvalues
        q95 : float
             The 95% percentile of eigenvalues
        PC : 2D array
            matrix of principal components
        RC : 2D array
            matrix of RCs (N*M, nmode) (only if K>0)
        '''


        Xr = self.standardize(data)
        N = len(data)
        c = np.zeros(M)

        for j in range(M):
            prod = Xr[0:N - j] * Xr[j:N]
            c[j] = sum(prod[~np.isnan(prod)]) / (sum(~np.isnan(prod)) - 1)


        C = toeplitz(c[0:M])

        eig_val, eig_vec = eigh(C)

        sort_tmp = np.sort(eig_val)
        deval = sort_tmp[::-1]
        sortarg = np.argsort(-eig_val)

        eig_vec = eig_vec[:, sortarg]

        coefs_est, var_est = alg.AR_est_YW(Xr[~np.isnan(Xr)], 1)
        sigma_est = np.sqrt(var_est)

        noise = np.zeros((N, MC))
        noise[0, :] = Xr[0]
        Lamda_R = np.zeros((M, MC))

        for jt in range(1, N):
            noise[jt, :] = coefs_est * noise[jt - 1, :] + sigma_est * np.random.randn(1, MC)

        for m in range(MC):
            noise[:, m] = (noise[:, m] - np.mean(noise[:, m])) / (np.std(noise[:, m], ddof=1))
            Gn = np.correlate(noise[:, m], noise[:, m], "full")
            lgs = np.arange(-N + 1, N)
            Gn = Gn / (N - abs(lgs))
            Cn = toeplitz(Gn[N - 1:N - 1 + M])
            # Lamda_R[:,m] = np.diag(np.dot(np.dot(eig_vec,Cn),np.transpose(eig_vec)))
            Lamda_R[:, m] = np.diag(np.dot(np.dot(np.transpose(eig_vec), Cn), eig_vec))

        q95 = np.percentile(Lamda_R, 95, axis=1)
        q05 = np.percentile(Lamda_R, 5, axis=1)

        # determine principal component time series
        PC = np.zeros((N - M + 1, M))
        PC[:, :] = np.nan
        for k in np.arange(M):
            for i in np.arange(0, N - M + 1):
                #   modify for nan
                prod = Xr[i:i + M] * eig_vec[:, k]
                ngood = sum(~np.isnan(prod))
                #   must have at least m*f good points
                if ngood >= M * f:
                    PC[i, k] = sum(
                        prod[~np.isnan(prod)]) * M / ngood  # the columns of this matrix are Ak(t), k=1 to M (T-PCs)

        # compute reconstructed timeseries
        Np = N - M + 1

        RC = np.zeros((N, M))

        for im in np.arange(M):
            x2 = np.dot(np.expand_dims(PC[:, im], axis=1), np.expand_dims(eig_vec[0:M, im], axis=0))
            x2 = np.flipud(x2)

            for n in np.arange(N):
                RC[n, im] = np.diagonal(x2, offset=-(Np - 1 - n)).mean()

        return deval, eig_vec, q05, q95, PC, RC


class FDR:
    ''' The FDR procedures translated from fdr.R by Dr. Chris Paciorek (https://www.stat.berkeley.edu/~paciorek/research/code/code.html)
    '''
    def fdr_basic(self, pvals,qlevel=0.05):
        ''' The basic FDR of Benjamini & Hochberg (1995).

        Args
        ----

        pvals : list or array
            A vector of p-values on which to conduct the multiple testing.

        qlevel : float
            The proportion of false positives desired.

        Returns
        -------

        fdr_res : array or None
            A vector of the indices of the significant tests; None if no significant tests

        '''

        n = len(pvals)
        sorted_pvals = np.sort(pvals)
        sort_index = np.argsort(pvals)
        indices = np.arange(1, n+1)*(sorted_pvals <= qlevel*np.arange(1, n+1)/n)
        num_reject = np.max(indices)

        if num_reject:
            indices = np.arange(num_reject)
            fdr_res =  np.sort(sort_index[indices])
        else:
            fdr_res = None

        return fdr_res

    def fdr_master(self, pvals, qlevel=0.05, method='original'):
        ''' Perform various versions of the FDR procedure, but without the modification

        Args
        ----

        pvals : list or array
            A vector of p-values on which to conduct the multiple testing.

        qlevel : float
            The proportion of false positives desired.

        method : {'original', 'general'}
            Method for performing the testing.
            - 'original' follows Benjamini & Hochberg (1995);
            - 'general' is much more conservative, requiring no assumptions on the p-values (see Benjamini & Yekutieli (2001)).
            We recommend using 'original', and if desired, using 'adj_method="mean"' to increase power.

        Returns
        -------

        fdr_res : array or None
            A vector of the indices of the significant tests; None if no significant tests

        '''
        if method == 'general':
            n = len(pvals)
            qlevel = qlevel / np.sum(1/np.arange(1, n+1))

        fdr_res = self.fdr_basic(pvals, qlevel)
        return fdr_res

    def storey(self, edf_quantile, pvals):
        ''' The basic Storey (2002) estimator of a, the proportion of alternative hypotheses.

        Args
        ----

        edf_quantile : float
            The quantile of the empirical distribution function at which to estimate a.

        pvals : list or array
            A vector of p-values on which to estimate a

        Returns
        -------

        a : int
            estimate of a, the number of alternative hypotheses

        '''
        if edf_quantile >= 1 or edf_quantile <= 0:
            raise ValueError(f'Wrong edf_quantile: {edf_quantile}; must be within (0, 1)!')

        pvals = np.array(pvals)
        a = (np.mean(pvals<=edf_quantile) - edf_quantile) / (1 - edf_quantile)
        a = np.max(a, 0)  # set to 0 if a is negative
        return a

    def prop_alt(self, pvals, adj_method='mean', adj_args={'edf_lower': 0.8, 'num_steps': 20}):
        ''' Calculate an estimate of a, the proportion of alternative hypotheses, using one of several methods

        Args
        ----

        pvals : list or array
            A vector of p-values on which to estimate a


        adj_method: {'mean', 'storey', 'two-stage'}
            Method for increasing the power of the procedure by estimating the proportion of alternative p-values.
            - 'mean', the modified Storey estimator that we suggest in Ventura et al. (2004)
            - 'storey', the method of Storey (2002)
            - 'two-stage', the iterative approach of Benjamini et al. (2001)

        adj_args : dict
            - for "mean", specify "edf_lower", the smallest quantile at which to estimate a, and "num_steps", the number of quantiles to use
              the approach uses the average of the Storey (2002) estimator for the num_steps quantiles starting at "edf_lower" and finishing just less than 1
            - for "storey", specify "edf_quantile", the quantile at which to calculate the estimator
            - for "two-stage", the method uses a standard FDR approach to estimate which p-values are significant
              this number is the estimate of a; therefore the method requires specification of qlevel,
              the proportion of false positives and "fdr_method" ('original' or 'general'), the FDR method to be used.
              We do not recommend 'general' as this is very conservative and will underestimate a.

        Returns
        -------

        a : int
            estimate of a, the number of alternative hypotheses

        '''
        n = len(pvals)
        if adj_method == 'two-stage':
            fdr_res = self.fdr_master(pvals, adj_method['qlevel'], adj_args['fdr_method'])
            a = len(fdr_res)/n
            return a

        elif adj_method == 'storey':
            if 'edf_quantile' not in adj_args:
                raise ValueError('`edf_quantile` must be specified in `adj_args`!')

            a = self.storey(adj_args['edf_quantile'], pvals)
            return a

        elif adj_method == 'mean':
            if adj_args['edf_lower']>=1 or adj_args['edf_lower']<=0:
                raise ValueError(f'Wrong edf_lower: {adj_args["edf_lower"]}; must be within (0, 1)!')

            if adj_args['num_steps']<1 or type(adj_args['num_steps']) is not int:
                raise ValueError(f'Wrong num_steps: {adj_args["num_steps"]}; must be an integer >= 1')

            stepsize = (1 - adj_args['edf_lower']) / adj_args['num_steps']

            edf_quantiles = np.linspace(adj_args['edf_lower'], adj_args['edf_lower']+stepsize*(adj_args['num_steps']-1), adj_args['num_steps'])
            a_vec = [self.storey(edf_q, pvals) for edf_q in edf_quantiles]
            a = np.mean(a_vec)
            return a

        else:
            raise ValueError(f'Wrong method: {method}!')

    def fdr(self, pvals, qlevel=0.05, method='original', adj_method=None, adj_args={}):
        ''' Determine significance based on the FDR approach

        Args
        ----

        pvals : list or array
            A vector of p-values on which to conduct the multiple testing.

        qlevel : float
            The proportion of false positives desired.

        method : {'original', 'general'}
            Method for performing the testing.
            - 'original' follows Benjamini & Hochberg (1995);
            - 'general' is much more conservative, requiring no assumptions on the p-values (see Benjamini & Yekutieli (2001)).
            We recommend using 'original', and if desired, using 'adj_method="mean"' to increase power.

        adj_method: {'mean', 'storey', 'two-stage'}
            Method for increasing the power of the procedure by estimating the proportion of alternative p-values.
            - 'mean', the modified Storey estimator that we suggest in Ventura et al. (2004)
            - 'storey', the method of Storey (2002)
            - 'two-stage', the iterative approach of Benjamini et al. (2001)

        adj_args : dict
            Arguments for adj_method; see prop_alt() for description,
            but note that for "two-stage", qlevel and fdr_method are taken from the qlevel and method arguments for fdr()

        Returns
        -------

        fdr_res : array or None
            A vector of the indices of the significant tests; None if no significant tests

        '''
        n = len(pvals)

        a = 0
        if adj_method is not None:
            if adj_method == 'two-stage':
                qlevel = qlevel / (1+qlevel)  # see Benjamini et al. (2001) for proof that this controls the FDR at level qlevel
                adj_args['qlevel'] = qlevel
                adj_args['fdr_method'] = method
                print(f'Adjusting cutoff using two-stage method, with method: {adj_args["fdr_method"]}; qlevel: {adj_args["qlevel"]}')

            elif adj_method == 'mean':
                if adj_args == {}:
                     # default arguments for "mean" method of Ventura et al. (2004)
                    adj_args['edf_lower'] = 0.8
                    adj_args['num_steps'] = 20
                print(f'Adjusting cutoff using mean method, with edf_lower: {adj_args["edf_lower"]}; num_steps: {adj_args["num_steps"]}')

            a = self.prop_alt(pvals, adj_method, adj_args)

        if a == 1:
            # all hypotheses are estimated to be alternatives
            fdr_res = np.arange(n)
        else:
            qlevel = qlevel / (1-a)  # adjust for estimate of a; default is 0
            fdr_res = self.fdr_master(pvals, qlevel, method)

        return fdr_res


class Timeseries:
    def simpleStats(y, axis=None):
        """ Computes simple statistics

        Computes the mean, median, min, max, standard deviation, and interquartile
        range of a numpy array y, ignoring NaNs.

        Args
        ----

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


    def binvalues(x, y, bin_size=None, start=None, end=None):
        """ Bin the values

        Args
        ----

        x : array
            The x-axis series.
        y : array
            The y-axis series.
        bin_size : float
            The size of the bins. Default is the average resolution
        start : float
            Where/when to start binning. Default is the minimum
        end : float
            When/where to stop binning. Defulat is the maximum

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

        return bins, binned_values, n, error


    def interp(x,y, interp_type='linear', interp_step=None,start=None,end=None):
        """ Linear interpolation onto a new x-axis

        Args
        ----

        x : array
           The x-axis
        y : array
           The y-axis
        interp_step : float
                     The interpolation step. Default is mean resolution.
        start : float
               where/when to start the interpolation. Default is min..
        end : float
             where/when to stop the interpolation. Default is max.

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

        interp_values = interpolate.interp1d(data['x-axis'],data['y-axis'],kind=interp_type)(xi)

        return xi, interp_values


    def onCommonAxis(x1, y1, x2, y2, method = 'interpolation', step=None, start=None, end=None):
        """Places two timeseries on a common axis

        Args
        ----
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
            Valid entries: 'interpolation' (default), 'bin', 'None'. 'None' only
            cuts the timeseries to the common period but does not attempt
            to generate a common time axis
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
            xi1, interp_values1, n, error = binvalues(x1, y1, bin_size=step, start=start,
                                    end=end)
            xi2, interp_values2, n, error = binvalues(x2, y2, bin_size=step, start=start,
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

        Args
        ----

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

        Args
        ----

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

        @author: fzhu
        '''

        ys, ts = Timeseries.clean_ts(ys, ts)

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

        Args
        ----
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

        Args
        ----
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


    def detrend(y, x = None, method = "emd", params = ["default",4,0,1]):
        """Detrend a timeseries according to three methods

        Detrending methods include, "linear", "constant", and using a low-pass
            Savitzky-Golay filters (default).

        Args
        ----

        y : array
           The series to be detrended.
        x : array
           The time axis for the timeseries. Necessary for use with
           the Savitzky-Golay filters method since the series should be evenly spaced.
        method : str
            The type of detrending:
            - linear: the result of a linear least-squares fit to y is subtracted from y.
            - constant: only the mean of data is subtrated.
            - "savitzy-golay", y is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
            - "emd" (default): Empiracal mode decomposition
        params : list
            The paramters for the Savitzky-Golay filters. The first parameter
            corresponds to the window size (default it set to half of the data)
            while the second parameter correspond to the order of the filter
            (default is 4). The third parameter is the order of the derivative
            (the default is zero, which means only smoothing.)

        Returns
        -------

        ys : array
            The detrended timeseries.
        """
        y = np.array(y)

        if x is not None:
            x = np.array(x)

        if method == "linear":
            ys = signal.detrend(y,type='linear')
        elif method == 'constant':
            ys = signal.detrend(y,type='constant')
        elif method == "savitzy-golay":
            # Check that the timeseries is uneven and interpolate if needed
            if x is None:
                raise ValueError("A time axis is needed for use with the Savitzky-Golay filters method")
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
            y_filt = Filter.savitzky_golay(y_interp,l,o,d,e)
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


    def detect_outliers(ts, ys, args={}):
        ''' Function to detect outliers in the given timeseries
        Args
        ----

        ts : array
             time axis of time series
        ys : array
             y values of time series
        args : dict
             arguments for the DBSCAN function from sklearn,
             for more details, see: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

        Returns
        -------

        is_outlier : array
                    a list of boolean values indicating whether the point is an outlier or not
        '''

        if args == {}:
            args = {'eps': 0.2, 'metric': 'euclidean'}

        outlier_detection = DBSCAN(**args)

        clusters = outlier_detection.fit_predict(ys.values.reshape(-1,1))
        is_outlier = []

        for value in clusters:
            if value == -1:
                is_outlier.append(True)
            else:
                is_outlier.append(False)

        return is_outlier

def causality_est(y1, y2, method='liang', signif_test='isospec', nsim=1000,\
                  qs=[0.005, 0.025, 0.05, 0.95, 0.975, 0.995], **kwargs):
    '''Information flow, estimate the information transfer from series y2 to series y1

    Args
    ----

    y1, y2 : array
        vectors of (real) numbers with identical length, no NaNs allowed
    method : array
        only "liang" for now
    signif_test : str
        the method for significance test
    nsim : int
        the number of AR(1) surrogates for significance test
    qs : list
        the quantiles for significance test
    kwargs : includes
        npt : int
            the number of time advance in performing Euler forward differencing in "liang" method

    Returns
    -------

    res_dict : dictionary
        The result of the dictionary including
    T21 : float
        The information flow from y2 to y1
    tau21 : float
        The standardized info flow from y2 to y1, tau21 = T21/Z
    Z : float
       The total information flow
    qs  : list
        significance test  of quantile levels
    t21_noise : list
        The quantiles of the information flow from noise2 to noise1 for significance testing
    tau21_noise : list
        The quantiles of the standardized information flow from noise2 to noise1 for significance testing
    '''
    ca = Causality()
    if method == 'liang':
        npt = kwargs['npt'] if 'npt' in kwargs else 1
        res_dict = ca.liang_causality(y1, y2, npt=npt)
        tau21 = res_dict['tau21']
        T21 = res_dict['T21']
        Z = res_dict['Z']

        signif_test_func = {
            'isopersist': ca.signif_isopersist,
            'isospec': ca.signif_isospec,
        }

        signif_dict = signif_test_func[signif_test](y1, y2, nsim=nsim, qs=qs, npt=npt)

        T21_noise_qs = signif_dict['T21_noise_qs']
        tau21_noise_qs = signif_dict['tau21_noise_qs']
        res_dict = {
            'T21': T21,
            'tau21': tau21,
            'Z': Z,
            'signif_qs': qs,
            'T21_noise': T21_noise_qs,
            'tau21_noise': tau21_noise_qs,
        }
    else:
        raise KeyError(f'{method} is not a valid method')

    return res_dict

def corrsig(y1, y2, nsim=1000, method='isospectral', alpha=0.05):
    """
    Estimates the significance of correlations between non IID time series by 3 independent methods:
        1) 'ttest': T-test where d.o.f are corrected for the effect of serial correlation
        2) 'isopersistent': AR(1) modeling of x and y.
        3) 'isospectral': phase randomization of original inputs. (default)
        The T-test is parametric test, hence cheap but usually wrong except in idyllic circumstances.
        The others are non-parametric, but their computational requirements scales with nsim.

    Args
    ----

    y1 : array
        vector of (real) numbers of identical length, no NaNs allowed
    y2 : array
        vector of (real) numbers of identical length, no NaNs allowed
    nsim : int
        the number of simulations [default: 1000]
    method : string
        methods 1-3 above [default: 'isospectral']
    alpha : float
        ignificance level for critical value estimation [default: 0.05]

    Returns
    -------

     r : float
         correlation between x and y
     signif : int
         true  if significant; false otherwise
     p : float
         Fraction of time series with higher correlation coefficents than observed (approximates the p-value).
         Note that signif = True if and only if p <= alpha.
"""
    corr = Correlation()
    r, signif, p = corr.corr_sig(y1,y2, nsim = nsim, method = method,
                                 alpha = alpha)

    return r, signif, p

def ar1_fit(ys, ts=None, detrend= None, params=["default", 4, 0, 1]):
    ''' Returns the lag-1 autocorrelation from ar1 fit OR persistence from tauest.

    Args
    ----

    ys : array
        the time series
    ts : array
        the time axis of that series
    detrend : string
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
    params : list
        The paramters for the Savitzky-Golay filters. The first parameter
        corresponds to the window size (default it set to half of the data)
        while the second parameter correspond to the order of the filter
        (default is 4). The third parameter is the order of the derivative
        (the default is zero, which means only smoothing.)

    Returns
    -------

    g : float
        lag-1 autocorrelation coefficient (for evenly-spaced time series)
        OR estimated persistence (for unevenly-spaced time series)
    '''

    wa = WaveletAnalysis()

    if wa.is_evenly_spaced(ts):
        g = wa.ar1_fit_evenly(ys, ts, detrend=detrend, params=params)
    else:
        g = wa.tau_estimation(ys, ts, detrend=detrend, params=params)

    return g

def ar1_sim(ys, n, p, ts=None, detrend=False, params=["default", 4, 0, 1]):
    ''' Produce p realizations of an AR1 process of length n with lag-1 autocorrelation g calculated from `ys` and `ts`

    Args
    ----

    ys : array
        a time series
    n : int
        row dimensions
    p : int
        column dimensions
    ts : array
        the time axis of that series
    detrend : string
        None - the original time series is assumed to have no trend;
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
    params : list
        The paramters for the Savitzky-Golay filters. The first parameter
        corresponds to the window size (default it set to half of the data)
        while the second parameter correspond to the order of the filter
        (default is 4). The third parameter is the order of the derivative
        (the default is zero, which means only smoothing.)

    Returns
    -------

    red : array
        n rows by p columns matrix of an AR1 process

    '''
    red = np.empty(shape=(n, p))  # declare array

    wa = WaveletAnalysis()
    if wa.is_evenly_spaced(ts):
        g = ar1_fit(ys, ts=ts, detrend=detrend, params=params)
        sig = np.std(ys)

        # specify model parameters (statsmodel wants lag0 coefficents as unity)
        ar = np.r_[1, -g]  # AR model parameter
        ma = np.r_[1, 0.0]  # MA model parameters
        sig_n = sig*np.sqrt(1-g**2)  # theoretical noise variance for red to achieve the same variance as ys

        # simulate AR(1) model for each column
        for i in np.arange(p):
            red[:, i] = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=50, sigma=sig_n)

    else:
        tau_est = ar1_fit(ys, ts=ts, detrend=detrend, params=params)
        for i in np.arange(p):
            red[:, i] = wa.ar1_model(ts, tau_est, n=n)

    if p == 1:
        red = red[:, 0]

    return red


def wwz(ys, ts, tau=None, freq=None, c=1/(8*np.pi**2), Neff=3, Neff_coi=3,\
        nMC=200, nproc=8, detrend=False, params=['default', 4, 0, 1],\
        gaussianize=False, standardize=True, method='default', len_bd=0,\
        bc_mode='reflect', reflect_type='odd'):
    ''' Return the weighted wavelet amplitude (WWA) with phase, AR1_q, and cone of influence, as well as WT coefficients

    Args
    ----

    ys : array
        a time series, NaNs will be deleted automatically
    ts : array
        the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
    tau : array
        the evenly-spaced time points
    freq : array
        vector of frequency
    c : float
        the decay constant, the default value 1/(8*np.pi**2) is good for most of the cases
    Neff : int
        effective number of points
    nMC : int
        the number of Monte-Carlo simulations
    nproc : int
        the number of processes for multiprocessing
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
    method : string
        'Foster' - the original WWZ method;
        'Kirchner' - the method Kirchner adapted from Foster;
        'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py
    len_bd : int
        the number of the ghost grids want to creat on each boundary
    bc_mode : string
        {'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect' , 'symmetric', 'wrap'}
        For more details, see np.lib.pad()
    reflect_type : string
         {‘even’, ‘odd’}, optional
         Used in ‘reflect’, and ‘symmetric’. The ‘even’ style is the default with an unaltered reflection around the edge value.
         For the ‘odd’ style, the extented part of the array is created by subtracting the reflected values from two times the edge value.
         For more details, see np.lib.pad()

    Returns
    -------

    wwa : array
        the weighted wavelet amplitude.
    AR1_q : array
        AR1 simulations
    coi : array
        cone of influence
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points, namely the time shift for wavelet analysis
    Neffs : array
        the matrix of effective number of points in the time-scale coordinates
    coeff : array
        the wavelet transform coefficents

    '''
    wa = WaveletAnalysis()
    assert isinstance(nMC, int) and nMC >= 0, "nMC should be larger than or equal to 0."

    ys_cut, ts_cut, freq, tau = wa.prepare_wwz(ys, ts, freq=freq, tau=tau,
                                                len_bd=len_bd, bc_mode=bc_mode, reflect_type=reflect_type)

    wwz_func = wa.get_wwz_func(nproc, method)
    wwa, phase, Neffs, coeff = wwz_func(ys_cut, ts_cut, freq, tau, Neff=Neff, c=c, nproc=nproc,
                                        detrend=detrend, params=params,
                                        gaussianize=gaussianize, standardize=standardize)

    # Monte-Carlo simulations of AR1 process
    nt = np.size(tau)
    nf = np.size(freq)

    wwa_red = np.ndarray(shape=(nMC, nt, nf))
    AR1_q = np.ndarray(shape=(nt, nf))

    if nMC >= 1:
        #  tauest = wa.tau_estimation(ys_cut, ts_cut, detrend=detrend, gaussianize=gaussianize, standardize=standardize)

        for i in tqdm(range(nMC), desc='Monte-Carlo simulations'):
            #  r = wa.ar1_model(ts_cut, tauest)
            r = ar1_sim(ys_cut, np.size(ts_cut), 1, ts=ts_cut)
            wwa_red[i, :, :], _, _, _ = wwz_func(r, ts_cut, freq, tau, c=c, Neff=Neff, nproc=nproc,
                                                 detrend=detrend, params=params,
                                                 gaussianize=gaussianize, standardize=standardize)

        for j in range(nt):
            for k in range(nf):
                AR1_q[j, k] = mquantiles(wwa_red[:, j, k], 0.95)

    else:
        AR1_q = None

    # calculate the cone of influence
    coi = wa.make_coi(tau, Neff=Neff_coi)

    Results = collections.namedtuple('Results', ['amplitude', 'phase', 'AR1_q', 'coi', 'freq', 'time', 'Neffs', 'coeff'])
    res = Results(amplitude=wwa, phase=phase, AR1_q=AR1_q, coi=coi, freq=freq, time=tau, Neffs=Neffs, coeff=coeff)

    return res

def lomb_scargle(ys, ts, freq=None, detrend=False, gaussianize=False,standardize=True, params=['default', 4, 0, 1], args={"precenter" : False, "normalize" : False, "make_freq_method" : "nfft"}):
    return SpectralAnalysis.lombs_cargle(ys, ts, freq=freq, detrend=detrend, gaussianize=gaussianize, standardize=standardize, params=params, args=args)

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
    wa = WaveletAnalysis()
    ys_cut, ts_cut, freq, tau = wa.prepare_wwz(ys, ts, freq=freq, tau=tau)

    # get wwa but AR1_q is not needed here so set nMC=0
    #  wwa, _, _, coi, freq, _, Neffs, _ = wwz(ys_cut, ts_cut, freq=freq, tau=tau, c=c, nproc=nproc, nMC=0,
    res_wwz = wwz(ys_cut, ts_cut, freq=freq, tau=tau, c=c, nproc=nproc, nMC=0,
              detrend=detrend, params=params,
              gaussianize=gaussianize, standardize=standardize, method=method)

    psd = wa.wwa2psd(res_wwz.amplitude, ts_cut, res_wwz.Neffs, freq=res_wwz.freq, Neff=Neff, anti_alias=anti_alias, avgs=avgs)
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


def xwt(ys1, ts1, ys2, ts2,
        tau=None, freq=None, c=1/(8*np.pi**2), Neff=3, Neff_coi=6, nproc=8,
        detrend=False, params=['default', 4, 0, 1],
        gaussianize=False, standardize=True,
        method='default'):
    ''' Return the cross-wavelet transform of two time series.

    Args
    ----

    ys1 : array
        first of two time series
    ys2 : array
        second of the two time series
    ts1 : array
        time axis of first time series
    ts2 : array
        time axis of the second time series
    tau : array
        the evenly-spaced time points
    freq : array
        vector of frequency
    c : float
        the decay constant, the default value 1/(8*np.pi**2) is good for most of the cases
    Neff : int
        effective number of points
    nproc : int
        the number of processes for multiprocessing
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

    Returns
    -------

    xw_amplitude : array
        the cross wavelet amplitude
    xw_phase : array
        the cross wavelet phase
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points
    AR1_q : array
        AR1 simulations
    coi : array
        cone of influence

    '''
    wa = WaveletAnalysis()

    wwz_func = wa.get_wwz_func(nproc, method)

    ys1_cut, ts1_cut, freq, tau = wa.prepare_wwz(ys1, ts1, freq=freq, tau=tau)
    ys2_cut, ts2_cut, freq, tau = wa.prepare_wwz(ys2, ts2, freq=freq, tau=tau)

    _, _, _, coeff1 = wwz_func(ys1_cut, ts1_cut, freq, tau, Neff=Neff, c=c, nproc=nproc, detrend=detrend,
                               params=params, gaussianize=gaussianize, standardize=standardize)
    _, _, _, coeff2 = wwz_func(ys2_cut, ts2_cut, freq, tau, Neff=Neff, c=c, nproc=nproc, detrend=detrend,
                               params=params, gaussianize=gaussianize, standardize=standardize)

    tauest1 = wa.tau_estimation(ys1_cut, ts1_cut, detrend=detrend, params=params,
                                gaussianize=gaussianize, standardize=standardize)
    tauest2 = wa.tau_estimation(ys2_cut, ts2_cut, detrend=detrend, params=params,
                                gaussianize=gaussianize, standardize=standardize)
    r1 = wa.ar1_model(ts1_cut, tauest1)
    r2 = wa.ar1_model(ts2_cut, tauest2)
    #  r1 = ar1_sim(ys1_cut, np.size(ts1_cut), 1, ts=ts1_cut)
    #  r2 = ar1_sim(ys2_cut, np.size(ts2_cut), 1, ts=ts2_cut)

    #  wwa_red1, _, Neffs_red1, _ = wwz_func(r1, ts1_cut, freq, tau, c=c, Neff=Neff, nproc=nproc, detrend=detrend,
    #                                        gaussianize=gaussianize, standardize=standardize)
    #  wwa_red2, _, Neffs_red2, _ = wwz_func(r2, ts2_cut, freq, tau, c=c, Neff=Neff, nproc=nproc, detrend=detrend,
    #                                        gaussianize=gaussianize, standardize=standardize)
    #  psd1_ar1 = wa.wwa2psd(wwa_red1, ts1_cut, Neffs_red1, freq=freq, Neff=Neff, anti_alias=False, avgs=1)
    #  psd2_ar1 = wa.wwa2psd(wwa_red2, ts2_cut, Neffs_red2, freq=freq, Neff=Neff, anti_alias=False, avgs=1)
    dt1 = np.median(np.diff(ts1))
    dt2 = np.median(np.diff(ts2))
    f_sampling_1 = 1/dt1
    f_sampling_2 = 1/dt2
    psd1_ar1 = wa.psd_ar(np.var(r1), freq, tauest1, f_sampling_1)
    psd2_ar1 = wa.psd_ar(np.var(r2), freq, tauest2, f_sampling_2)

    wt_coeff1 = coeff1[1] + coeff1[2]*1j
    wt_coeff2 = coeff2[1] + coeff2[2]*1j
    xwt, xw_amplitude, xw_phase = wa.cross_wt(wt_coeff1, wt_coeff2, freq, tau)

    sigma_1 = np.std(ys1_cut)
    sigma_2 = np.std(ys2_cut)
    nu, Znu = 2, 3.9999  # according to `xwt.m` from Grinsted's MATLAB code

    signif = sigma_1*sigma_2 * np.sqrt(psd1_ar1*psd2_ar1) * Znu/nu  # Eq. (5) of Grinsted et al 2004
    AR1_q = np.tile(signif, (np.size(tau), 1))

    coi = wa.make_coi(tau, Neff=Neff_coi)

    return xwt, xw_amplitude, xw_phase, freq, tau, AR1_q, coi


def xwc(ys1, ts1, ys2, ts2, smooth_factor=0.25,
        tau=None, freq=None, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend=False,
        nMC=200, params=['default', 4, 0, 1],
        gaussianize=False, standardize=True, method='default'):
    ''' Return the cross-wavelet coherence of two time series.

    Args
    ----

    ys1 : array
        first of two time series
    ys2 : array
        second of the two time series
    ts1 : array
        time axis of first time series
    ts2 : array
        time axis of the second time series
    tau : array
        the evenly-spaced time points
    freq : array
        vector of frequency
    c : float
        the decay constant, the default value 1/(8*np.pi**2) is good for most of the cases
    Neff : int
        effective number of points
    nproc : int
        the number of processes for multiprocessing
    nMC : int
        the number of Monte-Carlo simulations
    detrend : string
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

    Returns
    -------

    res : dict
        contains the cross wavelet coherence, cross-wavelet phase,
        vector of frequency, evenly-spaced time points, AR1 sims, cone of influence

    '''
    wa = WaveletAnalysis()
    assert isinstance(nMC, int) and nMC >= 0, "nMC should be larger than or eaqual to 0."

    if tau is None:
        lb1, ub1 = np.min(ts1), np.max(ts1)
        lb2, ub2 = np.min(ts2), np.max(ts2)
        lb = np.max([lb1, lb2])
        ub = np.min([ub1, ub2])

        inside = ts1[(ts1>=lb) & (ts1<=ub)]
        tau = np.linspace(lb, ub, np.size(inside)//10)
        print(f'Setting tau={tau[:3]}...{tau[-3:]}, ntau={np.size(tau)}')

    if freq is None:
        s0 = 2*np.median(np.diff(ts1))
        nv = 12
        a0 = 2**(1/nv)
        noct = np.floor(np.log2(np.size(ts1)))-1
        scale = s0*a0**(np.arange(noct*nv+1))
        freq = 1/scale[::-1]
        print(f'Setting freq={freq[:3]}...{freq[-3:]}, nfreq={np.size(freq)}')

    ys1_cut, ts1_cut, freq1, tau1 = wa.prepare_wwz(ys1, ts1, freq=freq, tau=tau)
    ys2_cut, ts2_cut, freq2, tau2 = wa.prepare_wwz(ys2, ts2, freq=freq, tau=tau)

    if np.any(tau1 != tau2):
        print('inconsistent `tau`, recalculating...')
        tau_min = np.min([np.min(tau1), np.min(tau2)])
        tau_max = np.max([np.max(tau1), np.max(tau2)])
        ntau = np.max([np.size(tau1), np.size(tau2)])
        tau = np.linspace(tau_min, tau_max, ntau)
    else:
        tau = tau1

    if np.any(freq1 != freq2):
        print('inconsistent `freq`, recalculating...')
        freq_min = np.min([np.min(freq1), np.min(freq2)])
        freq_max = np.max([np.max(freq1), np.max(freq2)])
        nfreq = np.max([np.size(freq1), np.size(freq2)])
        freq = np.linspace(freq_min, freq_max, nfreq)
    else:
        freq = freq1

    if freq[0] == 0:
        freq = freq[1:] # delete 0 frequency if present

    res_wwz1 = wwz(ys1_cut, ts1_cut, tau=tau, freq=freq, c=c, Neff=Neff, nMC=0,
                   nproc=nproc, detrend=detrend, params=params,
                   gaussianize=gaussianize, standardize=standardize, method=method)
    res_wwz2 = wwz(ys2_cut, ts2_cut, tau=tau, freq=freq, c=c, Neff=Neff, nMC=0,
                   nproc=nproc, detrend=detrend, params=params,
                   gaussianize=gaussianize, standardize=standardize, method=method)

    wt_coeff1 = res_wwz1.coeff[1] - res_wwz1.coeff[2]*1j
    wt_coeff2 = res_wwz2.coeff[1] - res_wwz2.coeff[2]*1j

    xw_coherence, xw_phase = wa.wavelet_coherence(wt_coeff1, wt_coeff2, freq, tau, smooth_factor=smooth_factor)
    xwt, xw_amplitude, _ = wa.cross_wt(wt_coeff1, wt_coeff2)

    # Monte-Carlo simulations of AR1 process
    nt = np.size(tau)
    nf = np.size(freq)

    coherence_red = np.ndarray(shape=(nMC, nt, nf))
    AR1_q = np.ndarray(shape=(nt, nf))

    if nMC >= 1:

        for i in tqdm(range(nMC), desc='Monte-Carlo simulations'):
            r1 = ar1_sim(ys1_cut, np.size(ts1_cut), 1, ts=ts1_cut)
            r2 = ar1_sim(ys2_cut, np.size(ts2_cut), 1, ts=ts2_cut)
            res_wwz_r1 = wwz(r1, ts1_cut, tau=tau, freq=freq, c=c, Neff=Neff, nMC=0, nproc=nproc,
                                                     detrend=detrend, params=params,
                                                     gaussianize=gaussianize, standardize=standardize)
            res_wwz_r2 = wwz(r2, ts2_cut, tau=tau, freq=freq, c=c, Neff=Neff, nMC=0, nproc=nproc,
                                                     detrend=detrend, params=params,
                                                     gaussianize=gaussianize, standardize=standardize)

            wt_coeffr1 = res_wwz_r1.coeff[1] - res_wwz_r2.coeff[2]*1j
            wt_coeffr2 = res_wwz_r1.coeff[1] - res_wwz_r2.coeff[2]*1j
            coherence_red[i, :, :], phase_red = wa.wavelet_coherence(wt_coeffr1, wt_coeffr2, freq, tau, smooth_factor=smooth_factor)

        for j in range(nt):
            for k in range(nf):
                AR1_q[j, k] = mquantiles(coherence_red[:, j, k], 0.95)

    else:
        AR1_q = None

    coi = wa.make_coi(tau, Neff=Neff)
    Results = collections.namedtuple('Results', ['xw_coherence', 'xw_amplitude', 'xw_phase', 'xwt', 'freq', 'time', 'AR1_q', 'coi'])
    res = Results(xw_coherence=xw_coherence, xw_amplitude=xw_amplitude, xw_phase=xw_phase, xwt=xwt,
                  freq=freq, time=tau, AR1_q=AR1_q, coi=coi)

    return res


def plot_wwa(wwa, freq, tau, AR1_q=None, coi=None, levels=None, tick_range=None,
             yticks=None, yticks_label=None, ylim=None, xticks=None, xlabels=None,
             figsize=[20, 8], clr_map='OrRd',cbar_drawedges=False, cone_alpha=0.5,
             plot_signif=False, signif_style='contour', title=None, font_scale=1.5,
             plot_cbar=True, plot_cone=False, ax=None, xlabel='Year CE',
             ylabel='Period (years)', cbar_orientation='vertical',
             cbar_pad=0.05, cbar_frac=0.15, cbar_labelsize=None):
    """ Plot the wavelet amplitude

    Args
    ----

    wwa : array
        the weighted wavelet amplitude.
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points, namely the time shift for wavelet analysis
    AR1_q : array
        AR1 simulations
    coi : array
        cone of influence
    levels : array
        levels of values to plot
    tick_range : array
        levels of ticks to show on the colorbar
    yticks : list
        ticks on y-axis
    ylim : list
        limitations for y-axis
    xticks : list
        ticks on x-axis
    figsize : list
        the size for the figure
    clr_map : string
        the name of the colormap
    cbar_drawedges : bool
        whether to draw edges on the colorbar or not
    cone_alpha : float
        the alpha value for the area covered by cone of influence
    plot_signif : bool
        plot 95% significant area or not
    signif_style : string
        plot 95% significant area with `contour` or `shade`
    title : string
        Title for the plot
    plot_cbar : bool
        Plot the color scale bar
    plot_cone : bool
        plot cone of influence
    ax : axis, optional
        Return as axis instead of figure (useful to integrate plot into a subplot)
    xlabel : string
        The x-axis label
    ylabel : string
        The y-axis label
    cbar_pad : float
        the pad for the colorbar
    cbar_frac : float
        the frac for the colorbar
    cbar_labelsize : float
        the font size of the colorbar label

    Returns
    -------

    fig : figure
        the 2-D plot of wavelet analysis

    """
    sns.set(style="ticks", font_scale=font_scale)
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    if levels is None:
        q95 = mquantiles(wwa, 0.95)
        if np.nanmax(wwa) > 2*q95:
            warnings.warn("There are outliers in the input amplitudes, " +
                          "and the `levels` have been set so that the outpliers will be ignored! " +
                          "One might want to use `analysis.plot_wwadist(wwa)` to plot the distribution of " +
                          "the amplitudes with the 95% quantile line to check if the levels are appropriate.", stacklevel=2)

            max_level = np.round(2*q95, decimals=1)
            if max_level == 0:
                max_level = 0.01
            levels = np.linspace(0, max_level, 11)

    origin = 'lower'

    if levels is not None:
        plt.contourf(tau, 1/freq, wwa.T, levels, cmap=clr_map, origin=origin)
    else:
        plt.contourf(tau, 1/freq, wwa.T, cmap=clr_map, origin=origin)

    if plot_cbar:
        cb = plt.colorbar(drawedges=cbar_drawedges, orientation=cbar_orientation, fraction=cbar_frac, pad=cbar_pad,
                          ticks=tick_range)

        if cbar_labelsize is not None:
            cb.ax.tick_params(labelsize=cbar_labelsize)

    plt.yscale('log', nonposy='clip')

    if yticks is not None:
        if np.min(yticks) < 1e3:
            yticks_label = list(map(str, yticks))
        else:
            yticks_label = list(map(str, np.asarray(yticks)/1e3))
            ylabel='Period (kyrs)'

        plt.yticks(yticks, yticks_label)


    #  xticks = ax.get_xticks()
    #  if np.abs(np.min(xticks)) < 1e3:
        #  xticks_label = list(map(str, xticks))
    #  else:
        #  xticks_label = list(map(str, xticks/1e3))
    #  plt.xticks(xticks, xticks_label)

    if yticks_label is None:
        ax.get_yaxis().set_major_formatter(ScalarFormatter())

    if ylim is None:
        if coi is None:
            ylim = [np.min(yticks), np.max(yticks)]
        else:
            ylim = [np.min(yticks), np.max(coi)]

    plt.ylim(ylim)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if plot_signif:
        assert AR1_q is not None, "Please set values for `AR1_q`!"
        signif = wwa / AR1_q
        if signif_style == 'contour':
            plt.contour(tau, 1/freq, signif.T, [-99, 1], colors='k')
        elif signif_style == 'shade':
            plt.contourf(tau, 1/freq, signif.T, [-99, 1], colors='k', alpha=0.1)  # significant if not shaded

    if plot_cone:
        assert coi is not None, "Please set values for `coi`!"
        plt.plot(tau, coi, 'k--')
        ax.fill_between(tau, coi, ylim[1], color='white', alpha=cone_alpha)

    if title is not None:
        plt.title(title)

    return ax


def plot_coherence(res_xwc, pt=0.5,
                   levels=None, tick_range=None, basey=2,
                   yticks=None, ylim=None, xticks=None, xlabels=None,
                   figsize=[20, 8], clr_map='OrRd',
                   skip_x=5, skip_y=5, scale=30, width=0.004,
                   cbar_drawedges=False, cone_alpha=0.5, plot_signif=False,
                   signif_style='contour', title=None,
                   plot_cone=False, ax=None, xlabel='Year', ylabel='Period',
                   cbar_orientation='vertical', font_scale=1.5,
                   cbar_pad=0.05, cbar_frac=0.15, cbar_labelsize=None):
    """ Plot the wavelet coherence

    Args
    ----

    res_xwc : dict
        contains the cross wavelet coherence, cross-wavelet phase,
        vector of frequency, evenly-spaced time points, AR1 sims,
        cone of influence. See xwc
    pt : float
        plot arrows above pt value
    levels : array
        levels of values to plot
    tick_range : array
        levels of ticks to show on the colorbar
    basey : int
        log base for y. Default is 2.
    yticks : list
        ticks on y-axis
    ylim : list
        limitations for y-axis
    xticks : list
        ticks on x-axis
    xlabels : list
        List of labels for the x-axis ticks
    figsize : list
        the size for the figure
    clr_map : string
        the name of the colormap
    skip_x : float
        plot every x points
    skip_y : float
        plot every y points
    scale : int
        Scale factor for arrows
    width : float
        Width of the arrows
    cbar_drawedges : bool
        whether to draw edges on the colorbar or not
    cone_alpha : float
        the alpha value for the area covered by cone of influence
    plot_signif : bool
        plot 95% significant area
    signif_style : string
        plot 95% significant area with `contour` or `shade`
    title : string
        Add a title to the plot
    plot_cone : bool
        plot cone of influence
    ax : axis, optional
        Return as axis instead of figure (useful to integrate plot into a subplot)
    xlabel : string
        The x-axis label
    ylabel : string
        The y-axis label
    cbar_orientation : string
        the orientation of the colorbar. Default is vertical
    cbar_pad : float
        the pad for the colorbar
    cbar_frac : float
        the frac for the colorbar
    cbar_labelsize : float
        the font size of the colorbar label


    Returns
    -------

    fig : figure
        the 2-D plot of wavelet analysis

    """
    xw_coherence = res_xwc.xw_coherence
    xw_phase = res_xwc.xw_phase
    freq = res_xwc.freq
    tau = res_xwc.tau
    AR1_q = res_xwc.AR1_q
    coi = res_xwc.coi

    sns.set(style="ticks", font_scale=font_scale)
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # plot coherence with significance test
    if levels is None:
        levels = np.linspace(0, 1, 11)

    origin = 'lower'

    plt.contourf(tau, 1/freq, xw_coherence.T, levels, cmap=clr_map, origin=origin)

    cb = plt.colorbar(drawedges=cbar_drawedges, orientation=cbar_orientation, fraction=cbar_frac, pad=cbar_pad,
                      ticks=tick_range)

    if cbar_labelsize is not None:
        cb.ax.tick_params(labelsize=cbar_labelsize)

    plt.yscale('log', nonposy='clip', basey=basey)

    if yticks is not None:
        plt.yticks(yticks)

    if xticks is not None:
        plt.xticks(xticks, xlabels)

    ax.get_yaxis().set_major_formatter(ScalarFormatter())

    if ylim is not None:
        plt.ylim(ylim)

    else:
        ylim = ax.get_ylim()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if plot_signif:
        assert AR1_q is not None, "Set values for `AR1_q`!"
        signif = xw_coherence / AR1_q
        if signif_style == 'contour':
            plt.contour(tau, 1/freq, signif.T, [-99, 1], colors='k')
        elif signif_style == 'shade':
            plt.contourf(tau, 1/freq, signif.T, [-99, 1], colors='k', alpha=0.1)  # significant if not shaded

    if plot_cone:
        assert coi is not None, "Please set values for `coi`!"
        plt.plot(tau, coi, 'k--')
        ax.fill_between(tau, coi, ylim[1], color='white', alpha=cone_alpha)

    if title is not None:
        plt.title(title)

    ax.set_ylim(ylim)

    # plot phase
    phase = np.copy(xw_phase)
    phase[xw_coherence < pt] = np.nan

    X, Y = np.meshgrid(tau, 1/freq)
    U, V = np.cos(phase).T, np.sin(phase).T

    ax.quiver(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x],
              U[::skip_y, ::skip_x], V[::skip_y, ::skip_x],
              scale=scale, width=width)

    return ax


def plot_wwadist(wwa, ylim=None, font_scale=1.5):
    ''' Plot the distribution of wwa with the 95% quantile line.

    Args
    ----

    wwa : array
        the weighted wavelet amplitude.
    ylim :list
        limitations for y-axis

    Returns
    -------

    fig : figure
        the 2-D plot of wavelet analysis

    '''
    sns.set(style="darkgrid", font_scale=font_scale)
    plt.subplots(figsize=[20, 4])
    q95 = mquantiles(wwa, 0.95)
    fig = sns.distplot(np.nan_to_num(wwa.flat))
    fig.axvline(x=q95, ymin=0, ymax=0.5, linewidth=2, linestyle='-')

    if ylim is not None:
        plt.ylim(ylim)

    return fig


def plot_psd(psd, freq, lmstyle='-', linewidth=None,
             color=sns.xkcd_rgb["denim blue"], ar1_lmstyle='-',
             ar1_linewidth=None, period_ticks=None, period_tickslabel=None,
             psd_lim=None, period_lim=None, alpha=1,
             figsize=[20, 8], label='PSD', plot_ar1=False,
             psd_ar1_q95=None, title=None, legend=True, font_scale=1.5,
             psd_ar1_color=sns.xkcd_rgb["pale red"],
             ax=None, vertical=False, plot_gridlines=True,
             period_label='Period (years)', psd_label='Spectral Density',
             zorder=None):
    """ Plot power spectral density

    Args
    ----

    psd : array
        power spectral density
    freq : array
        vector of frequency
    period_ticks : list
        ticks for period
    period_tickslabel : list
        Labels for the period ticks
    psd_lim : list
        limits for spectral density axis
    label : str
        the label for the PSD
    plot_ar1 : bool
        plot the ar1 curve
    psd_ar1_q95 : array
        the 95% quantile of the AR1 PSD
    psd_ar1_color : str
        the color for the 95% quantile of the AR1 PSD
    title : str
        the title for the figure
    period_lim : list
        limits for period axis
    figsize : list
        the size for the figure
    ax : axis
        Return as axis instead of figure (useful to integrate plot into a subplot)
    vertical : bool
        plot in vertical layout or not
    legend : bool
        plot legend
    lmstyle : str
        the line style
    linewidth : float
        the line width
    color : str
        Color of the line
    ar1_lmstyle : str
        line style for the AR1 ensemble
    ar1_linewidth : int
        line width for AR1 ensemble
    period_label : str
        the label for period
    psd_label : str
        the label for psd
    zorder : int
        the order of the layer
    period_tickslabel : str
        label for the period tick
    alpha : float
        set transparency
    label : str
        label for the figure
    plot_gridlines : bool
        Plot gridlines
    period_label : str
        label for the period axis
    psd_label : str
        label for the PSD axis

    Returns
    -------

    ax : figure
        the 2-D plot of wavelet analysis

    Examples
    --------

    Perform WWZ and plot the PSD:

    .. plot::
        :context: close-figs

        >>> from pyleoclim import spectral
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> # make up a sine wave
        >>> time = np.arange(2001)
        >>> f = 1/50
        >>> signal = np.cos(2*np.pi*f*time)
        >>> # WWZ
        >>> tau = np.linspace(np.min(time), np.max(time), 51)
        >>> res_wwz = analysis.wwz_psd(signal, time, tau=tau, c=1e-3, standardize=False, nMC=0)
        >>> # plot
        >>> fig = analysis.plot_psd(
        ...           res_wwz.amplitude,
        ...           res_wwz.freq,
        ...           period_ticks=[2, 5, 10, 20, 50, 100],
        ...           figsize=[10, 8],
        ...       )

    """
    sns.set(style="ticks", font_scale=font_scale)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if title is not None:
        ax.set_title(title)

    if vertical:
        x_data = psd
        y_data = 1 / freq
        x_data_ar1 = psd_ar1_q95
        y_data_ar1 = 1 / freq
    else:
        x_data = 1 / freq
        y_data = psd
        x_data_ar1 = 1 / freq
        y_data_ar1 = psd_ar1_q95

    if zorder is not None:
        ax.plot(x_data, y_data, lmstyle, linewidth=linewidth, label=label, zorder=zorder, color=color, alpha=alpha)
        if plot_ar1:
            assert psd_ar1_q95 is not None, "psd_ar1_q95 is required!"
            ax.plot(x_data_ar1, y_data_ar1, ar1_lmstyle, linewidth=ar1_linewidth,
                     label='AR(1) 95%', color=psd_ar1_color, zorder=zorder-1)
    else:
        ax.plot(x_data, y_data, lmstyle, linewidth=linewidth, label=label, color=color, alpha=alpha)
        if plot_ar1:
            assert psd_ar1_q95 is not None, "psd_ar1_q95 is required!"
            ax.plot(x_data_ar1, y_data_ar1, ar1_lmstyle, linewidth=ar1_linewidth,
                     label='AR(1) 95%', color=psd_ar1_color)

    ax.set_xscale('log', nonposx='clip')
    ax.set_yscale('log', nonposy='clip')

    if vertical:
        ax.set_ylabel(period_label)
        ax.set_xlabel(psd_label)

        if period_lim is not None:
            ax.set_ylim(period_lim)

        if psd_lim is not None:
            ax.set_xlim(psd_lim)

        if period_ticks is not None:
            ax.set_yticks(period_ticks)
            ax.get_yaxis().set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        else:
            ax.set_aspect('equal')
    else:
        ax.set_xlabel(period_label)
        ax.set_ylabel(psd_label)

        if period_lim is not None:
            ax.set_xlim(period_lim)

        if psd_lim is not None:
            ax.set_ylim(psd_lim)

        if period_tickslabel is None:
            if period_ticks is not None:
                ax.set_xticks(period_ticks)
                ax.get_xaxis().set_major_formatter(ScalarFormatter())
                ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
                plt.gca().invert_xaxis()
            else:
                ax.set_aspect('equal')
        else:
            ax.set_xticks(period_ticks)
            ax.set_xticklabels(period_tickslabel)
            plt.gca().invert_xaxis()

    if legend:
        ax.legend()

    if plot_gridlines:
        ax.grid()

    return ax


def plot_summary(ys, ts, freq=None, tau=None, c1=1/(8*np.pi**2), c2=1e-3,
                 nMC=200, nproc=1, detrend=False, params=["default", 4, 0, 1],
                 gaussianize=False, standardize=True, levels=None, method='default',
                 anti_alias=False, period_ticks=None, ts_color=None, ts_style='-o',
                 title=None, ts_ylabel=None, wwa_xlabel=None, wwa_ylabel=None,
                 psd_lmstyle='-', psd_lim=None, font_scale=1.5,
                 period_S_str='beta_I', period_S=[1/8, 1/2],
                 period_L_str='beta_D', period_L=[1/200, 1/20]):
    """ Plot the time series with the wavelet analysis and psd

    Args
    ----

    ys : array
        a time series
    ts : array
        time axis of the time series
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points, namely the time shift for wavelet analysis
    c1 : float
        the decay constant (wwz method)
    c2 : float
        the decay constant (wwz_psd method)
    nMC : int
        Number of Monte-Carlo simulations
    nproc : int
        fake argument, just for convenience
    detrend : string
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
    levels : array
        levels of values to plot
    method : string
        method for the WWZ transform. Default is Kirchner_f2py
    anti_alias : bool
        If True, uses anti-aliasing
    period_ticks : list
        ticks for period
    ts_color : string
        the color for the time series curve
    ts_style : string
        Style for the line
    title : string
        the title for the time series plot
    ts_ylabel : string
        label for y-axis in the time series plot
    wwa_xlabel : string
        label for x-axis in the wwa plot
    wwa_ylabel : string
        label for y-axis in the wwa plot
    psd_lmstyle : string
        the line style in the psd plot
    psd_lim : list
        the limits for psd
    font_scale : float
        Scaling factor for the font on the plot
    period_S : list
        the ranges for beta estimation
    period_L : list
        the ranges for beta estimation
    period_S_str : string
        String for beta estimation
    period_L_str : string
        String for beta estimation

    Returns
    --------

    fig : figure
        the summary plot

    """
    title_font = {'fontname': 'Arial', 'size': '24', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}

    if period_ticks is not None:
        period_ticks = np.asarray(period_ticks)
        period_tickslabel = list(map(str, period_ticks))

    ylim_min = np.min(period_ticks)

    gs = gridspec.GridSpec(6, 12)
    gs.update(wspace=0, hspace=0)

    fig = plt.figure(figsize=(15, 15))

    # plot the time series
    sns.set(style="ticks", font_scale=font_scale)
    ax1 = plt.subplot(gs[0:1, :-3])
    plt.plot(ts, ys, ts_style, color=ts_color)

    if title is not None:
        plt.title(title, **title_font)

    plt.xlim([np.min(ts), np.max(ts)])

    if ts_ylabel is not None:
        plt.ylabel(ts_ylabel)

    plt.grid()
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    # plot wwa
    ax2 = plt.subplot(gs[1:5, :-3])

    #  wwa, phase, AR1_q, coi, freq, tau, Neffs, coeff = \
    res_wwz = wwz(ys, ts, freq=freq, tau=tau, c=c1, nMC=nMC, nproc=nproc, detrend=detrend, method=method,
                  gaussianize=gaussianize, standardize=standardize)

    if wwa_xlabel is not None and wwa_ylabel is not None:
        plot_wwa(res_wwz.amplitude, res_wwz.freq, res_wwz.tau, coi=res_wwz.coi, AR1_q=res_wwz.AR1_q,
                 yticks=period_ticks, yticks_label=period_tickslabel,
                 ylim=[ylim_min, np.max(res_wwz.coi)],
                 plot_cone=True, plot_signif=True, xlabel=wwa_xlabel, ylabel=wwa_ylabel, ax=ax2, levels=levels,
                 cbar_orientation='horizontal', cbar_labelsize=15, cbar_pad=0.1, cbar_frac=0.15,
                 )
    else:
        plot_wwa(res_wwz.amplitude, res_wwz.freq, res_wwz.tau, coi=res_wwz.coi, AR1_q=res_wwz.AR1_q,
                 yticks=period_ticks, yticks_label=period_tickslabel,
                 ylim=[ylim_min, np.max(res_wwz.coi)],
                 plot_cone=True, plot_signif=True, ax=ax2,
                 cbar_orientation='horizontal', cbar_labelsize=15, cbar_pad=0.1, cbar_frac=0.15, levels=levels,
                 )

    # plot psd
    ax3 = plt.subplot(gs[1:4, 9:])
    res_psd = wwz_psd(ys, ts, freq=None, tau=tau, c=c2, nproc=nproc, nMC=nMC, method=method,
                      detrend=detrend, gaussianize=gaussianize, standardize=standardize,
                      anti_alias=anti_alias)

    # TODO: deal with period_ticks
    plot_psd(res_psd.psd, res_psd.freq, plot_ar1=True, psd_ar1_q95=res_psd.psd_ar1_q95,
             period_ticks=period_ticks[period_ticks < np.max(res_wwz.coi)],
             period_lim=[np.min(period_ticks), np.max(res_wwz.coi)], psd_lim=psd_lim,
             lmstyle=psd_lmstyle, ax=ax3, period_label='', label='Estimated spectrum', vertical=True)

    if period_S is not None:
        res_beta1 = beta_estimation(res_psd.psd, res_psd.freq, period_S[0], period_S[1])

        if period_L is not None:
            res_beta2 = beta_estimation(res_psd.psd, res_psd.freq, period_L[0], period_L[1])
            ax3.plot(res_beta1.Y_reg, 1/res_beta1.f_binned, color='k',
                    label=r'$\{}$ = {:.2f}$\pm${:.2f}'.format(period_S_str, res_beta1.beta, res_beta1.std_err) + ', ' + r'$\{}$ = {:.2f}$\pm${:.2f}'.format(period_L_str, res_beta2.beta, res_beta2.std_err))
            ax3.plot(res_beta2.Y_reg, 1/res_beta2.f_binned, color='k')
        else:
            ax3.plot(res_beta1.Y_reg, 1/res_beta1.f_binned, color='k',
                    label=r'$\{}$ = {:.2f}$\pm${:.2f}'.format(period_S_str, res_beta1.beta, res_beta1.std_err))

    plt.tick_params(axis='y', which='both', labelleft='off')
    plt.legend(fontsize=15, bbox_to_anchor=(0, 1.2), loc='upper left', ncol=1)

    return fig


def calc_plot_psd(ys, ts, ntau=501, dcon=1e-3, standardize=False,
                  anti_alias=False, plot_fig=True, method='default', nproc=8,
                  period_ticks=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], color=None,
                  figsize=[10, 6], font_scale=1.5, lw=3, label='PSD', zorder=None,
                  xlim=None, ylim=None, loc='upper right', bbox_to_anchor=None):
    """ Calculate the PSD and plot the result

    Args
    ----

    ys : array
        a time series
    ts : array
        time axis of the time series
    natu : int
        the length of tau, the evenly-spaced time points,
        namely the time shift for wavelet analysis
    dcon : float
        the decay constant
    standardize : bool
        perform standardization or not
    anti_alias : bool
        perform anti-alising procedure or not
    plot_fig : bool
        plot the result or not
    method : string
        the WWZ method to use
    nproc : int
        the number of threads
    period_ticks : list
        List of period ticks
    color : string
        set color
    figsize : list
        Size of the figure
    font_scale : float
        Scale of the font
    lw : float
        For plotting purposes
    label : string
        Labeld for the y-axis
    zorder : int
        the order of the layer
    xlim : list
        x-axis limits
    ylim : list
        y-axis limits
    loc : string
        location for the legend
    bbox_to_anchor : list
        gives a great degree of control for manual
        legend placement. For example, if you want your axes legend
        located at the figure’s top right-hand corner instead of the axes’
        corner, simply specify the corner’s location, and the coordinate
        system of that location

    Returns
    -------

    fig : figure
        the summary plot
    psd : array
        the spectral density
    freq : array
        the frequency vector

    """
    if color is None:
        color = sns.xkcd_rgb['denim blue']

    tau = np.linspace(np.min(ts), np.max(ts), ntau)
    res_psd = wwz_psd(ys, ts, freq=None, tau=tau, c=dcon, standardize=standardize, nMC=0,
                      method=method, anti_alias=anti_alias, nproc=nproc)
    if plot_fig:
        sns.set(style='ticks', font_scale=font_scale)
        fig, ax = plt.subplots(figsize=figsize)
        ax.loglog(1/res_psd.freq, res_psd.psd, lw=lw, color=color, label=label,
                  zorder=zorder)
        ax.set_xticks(period_ticks)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.invert_xaxis()
        ax.set_ylabel('Spectral Density')
        ax.set_xlabel('Period (years)')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)
        ax.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, frameon=False)
        return fig, res_psd.psd, res_psd.freq
    else:
        return res_psd.psd, res_psd.freq


# some alias
wa = WaveletAnalysis()
beta_estimation = wa.beta_estimation
tau_estimation = wa.tau_estimation

def spectral(ys, ts, method='mtm', nMC=0, qs=0.95, kwargs={}):
    ''' Call periodogram from scipy

    Args
    ----

    ys : array
        a time series
    ts : array
        time axis of the time series
    method : string
        {'mtm', 'periodogram', 'welch', 'lombscargle'}): the available methods
    nMC : int
        the number of surrogates of AR(1) process for significance test; 0 means no test
    qs : float
        the quantile used for significance test
    kwargs : dict
        the dictionary for arguments, for the details please see the methods under class SpectralAnalysis()

    Returns
    -------

    res_dict : dict
        the result dictionary, including
        - freq (array): the frequency vector
        - psd (array): the spectral density vector
        - psd_ar1_q95 (array): the spectral density vector

    '''

    # spectral analysis
    sa = SpectralAnalysis()
    spec_func = {
        'mtm': sa.mtm,
        'periodogram': sa.periodogram,
        'welch': sa.welch,
        'lombscargle': sa.lombscargle,
    }

    res_dict = spec_func[method](ys, ts, **kwargs)

    # significance test with AR(1)
    if nMC >= 1:
        nf = np.size(res_dict['freq'])
        psd_ar1 = np.ndarray(shape=(nMC, nf))
        for i in tqdm(range(nMC), desc='Monte-Carlo simulations'):
            ar1_noise = ar1_sim(ys, np.size(ts), 1, ts=ts)
            res_red = spec_func[method](ar1_noise, ts, **kwargs)
            psd_ar1[i, :] = res_red['psd']

        psd_ar1_q95 = mquantiles(psd_ar1, qs, axis=0)[0]
        res_dict['psd_ar1_q95'] = psd_ar1_q95

    return res_dict
