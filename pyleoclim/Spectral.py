#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:23:06 2017

@author: deborahkhider, fengzhu

Spectral module for pyleoclim
"""

import numpy as np
import statsmodels.api as sm

from scipy import optimize
from scipy import signal
from scipy.stats.mstats import mquantiles
import scipy.fftpack as fft

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib import gridspec

from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

import warnings

from pyleoclim import Timeseries
import sys
#import platform

from math import factorial

if sys.platform.startswith('darwin'):
    from . import f2py_wwz as f2py

'''
Core functions below, focusing on algorithms
'''


class WaveletAnalysis(object):
    '''Performing wavelet analysis @author: fzhu
    '''
    def is_evenly_spaced(self, ts):
        ''' Check if a time axis is evenly spaced.

        Args:
            ts (array): the time axis of a time series

        Returns:
            check (bool): True - evenly spaced; False - unevenly spaced.

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

    def ar1_fit_evenly(self, ys, ts, detrend='no', params=["default", 4, 0, 1], gaussianize=False):
        ''' Returns the lag-1 autocorrelation from ar1 fit.

        Args:
            ys (array): vector of (float) numbers as a time series
            ts (array): The time axis for the timeseries. Necessary for use with
                the Savitzky-Golay filters method since the series should be evenly spaced.
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
                           'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            g (float): lag-1 autocorrelation coefficient

        '''
        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize)
        ar1_mod = sm.tsa.AR(pd_ys, missing='drop').fit(maxlag=1)
        g = ar1_mod.params[0]

        return g

    def preprocess(self, ys, ts, detrend='no', params=["default", 4, 0, 1], gaussianize=False, standardize=True):
        ''' Return the processed time series using (detrend and) standardization.

        Args:
            ys (array): a time series
            ts (array): The time axis for the timeseries. Necessary for use with
                the Savitzky-Golay filters method since the series should be evenly spaced.
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
                           'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            res (array): the processed time series

        '''

        if detrend is not 'no':
            ys_d = Timeseries.detrend(ys, ts, method=detrend, params=params)
        else:
            ys_d = ys

        if standardize:
            res, _, _ = Timeseries.standardize(ys_d)
        else:
            res = ys_d

        if gaussianize:
            res = Timeseries.gaussianize(res)

        return res

    def tau_estimation(self, ys, ts, detrend='no', params=["default", 4, 0, 1], gaussianize=False, standardize=True):
        ''' Return the estimated persistence of a givenevenly/unevenly spaced time series.

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
                           'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            tau_est (float): the estimated persistence

        References:
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

        Args:
            ts (array): time axis of the time series
            tau (float): the averaged persistence
            n (int): the length of the AR1 process

        Returns:
            r (array): the AR1 time series

        References:
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

    def wwz_opt2(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend='no', params=["default", 4, 0, 1],
                 gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA).

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): fake argument, just for convenience
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
                           'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients (a0, a1, a2)

        References:
            Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
            Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
                Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        assert nproc == 1, "wwz_basic() only supports nproc=1"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freqs)

        Neffs = np.ndarray(shape=(nt, nf))
        ywave_1 = np.ndarray(shape=(nt, nf))
        ywave_2 = np.ndarray(shape=(nt, nf))
        ywave_3 = np.ndarray(shape=(nt, nf))

        for k in range(nf):
            for j in range(nt):
                dz = omega[k] * (ts - tau[j])
                weights = np.exp(-c*dz**2)

                sum_w = np.sum(weights)
                Neffs[j, k] = sum_w**2 / np.sum(weights**2)  # local number of effective dof

                if Neffs[j, k] <= Neff:
                    ywave_2[j, k] = np.nan  # the coefficients cannot be estimated reliably when Neff_loc <= Neff
                    ywave_3[j, k] = np.nan
                else:
                    phi2 = np.cos(dz)
                    phi3 = np.sin(dz)

                    weighted_one = np.sum(weights*pd_ys) / sum_w
                    weighted_phi2 = np.sum(weights*phi2*pd_ys) / sum_w
                    weighted_phi3 = np.sum(weights*phi3*pd_ys) / sum_w

                    ywave_1[j, k] = weighted_one
                    ywave_2[j, k] = 2*weighted_phi2
                    ywave_3[j, k] = 2*weighted_phi3

        wwa = np.sqrt(ywave_2**2 + ywave_3**2)
        phase = np.arctan2(ywave_3, ywave_2)
        #  coeff = ywave_2 + ywave_3*1j
        coeff = (ywave_1, ywave_2, ywave_3)

        return wwa, phase, Neffs, coeff

    def wwz_opt1(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend='no', params=["default", 4, 0, 1],
                 gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA).

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): fake argument, just for convenience
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
                           'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients (a0, a1, a2)

        References:
            Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
            Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
                Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        assert nproc == 1, "wwz_basic() only supports nproc=1"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freqs)

        Neffs = np.ndarray(shape=(nt, nf))
        ywave_1 = np.ndarray(shape=(nt, nf))
        ywave_2 = np.ndarray(shape=(nt, nf))
        ywave_3 = np.ndarray(shape=(nt, nf))

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

                    weighted_phi2 = np.sum(weights*phi2*pd_ys) / sum_w
                    weighted_phi3 = np.sum(weights*phi3*pd_ys) / sum_w
                    weighted_one = np.sum(weights*pd_ys) / sum_w
                    cos_shift_one = np.sum(weights*phi2) / sum_w
                    sin_shift_one = np.sum(weights*phi3) / sum_w

                    ywave_1[j, k] = weighted_one
                    ywave_2[j, k] = 2*(weighted_phi2-weighted_one*cos_shift_one)
                    ywave_3[j, k] = 2*(weighted_phi3-weighted_one*sin_shift_one)

        wwa = np.sqrt(ywave_2**2 + ywave_3**2)
        phase = np.arctan2(ywave_3, ywave_2)
        #  coeff = ywave_2 + ywave_3*1j
        coeff = (ywave_1, ywave_2, ywave_3)

        return wwa, phase, Neffs, coeff

    def wwz_basic(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend='no', params=['default', 4, 0, 1],
                  gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA).

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): fake argument, just for convenience
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
                           'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients (a0, a1, a2)

        References:
            Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
            Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
                Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        assert nproc == 1, "wwz_basic() only supports nproc=1"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freqs)

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

    def wwz_nproc(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=8,  detrend='no', params=['default', 4, 0, 1],
                  gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA).

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): the number of processes for multiprocessing
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
                           'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients (a0, a1, a2)

        '''
        assert nproc >= 2, "wwz_nproc() should use nproc >= 2, if want serial run, please use wwz_basic()"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freqs)

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

    def kirchner_basic(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend='no', params=["default", 4, 0, 1],
                       gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): fake argument, just for convenience
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
                           'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients (a0, a1, a2)

        References:
            Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
            Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
                Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        assert nproc == 1, "wwz_basic() only supports nproc=1"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nts = np.size(ts)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freqs)

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

    def kirchner_opt(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend='no', params=["default", 4, 0, 1],
                     gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): fake argument, just for convenience
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
                           'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients (a0, a1, a2)

        References:
            Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
            Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
                Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        assert nproc == 1, "kirchner_opt() only supports nproc=1"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nts = np.size(ts)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freqs)

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
                    ys_one = w_prod(pd_ys, one_v)
                    sin_tau_center = np.sin(omega[k]*(time_shift - tau[j]))
                    cos_tau_center = np.cos(omega[k]*(time_shift - tau[j]))

                    ys_cos_shift = w_prod(pd_ys, cos_shift)
                    ys_sin_shift = w_prod(pd_ys, sin_shift)

                    A = 2*ys_cos_shift
                    B = 2*ys_sin_shift

                    a0[j, k] = ys_one
                    a1[j, k] = cos_tau_center*A - sin_tau_center*B  # Eq. (S6)
                    a2[j, k] = sin_tau_center*A + cos_tau_center*B  # Eq. (S7)

        wwa = np.sqrt(a1**2 + a2**2)
        phase = np.arctan2(a2, a1)
        #  coeff = a1 + a2*1j
        coeff = (a0, a1, a2)

        return wwa, phase, Neffs, coeff

    def kirchner_nproc(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend='no', params=['default', 4, 0, 1],
                       gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): the number of processes for multiprocessing
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
            'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients (a0, a1, a2)

        '''
        assert nproc >= 2, "wwz_nproc() should use nproc >= 2, if want serial run, please use wwz_basic()"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nts = np.size(ts)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freqs)

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

    def kirchner_f2py(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend='no', params=['default', 4, 0, 1],
                      gaussianize=False, standardize=True):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): fake argument, just for convenience
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted
                           'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)
            gaussionize (bool): If True, gaussianizes the timeseries
            standardize (bool): If True, standardizes the timeseries

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients (a0, a1, a2)

        '''
        self.assertPositiveInt(Neff, nproc)

        nt = np.size(tau)
        nts = np.size(ts)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)

        omega = self.make_omega(ts, freqs)

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

        Args:
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
            Neff (int): the threshold of the number of effective samples

        Returns:
            coi (array): cone of influence

        References:
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

    def make_omega(self, ts, freqs):
        ''' Return the angular frequency based on the time axis and given frequency vector

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency

        Returns:
            omega (array): the angular frequency vector

        '''
        # for the frequency band larger than f_Nyquist, the wwa will be marked as NaNs
        f_Nyquist = 0.5 / np.median(np.diff(ts))
        freqs_with_nan = np.copy(freqs)
        freqs_with_nan[freqs > f_Nyquist] = np.nan
        omega = 2*np.pi*freqs_with_nan

        return omega

    def wwa2psd(self, wwa, ts, Neffs, freqs=None, Neff=3, anti_alias=False, avgs=1):
        """ Return the power spectral density (PSD) using the weighted wavelet amplitude (WWA).

        Args:
            wwa (array): the weighted wavelet amplitude.
            ts (array): the time points, should be pre-truncated so that the span is exactly what is used for wwz
            Neffs (array):  the matrix of effective number of points in the time-scale coordinates obtained from wwz from wwz
            freqs (array): vector of frequency from wwz
            Neff (int): the threshold of the number of effective samples
            anti_alias (bool): whether to apply anti-alias filter
            avgs (int): flag for whether spectrum is derived from instantaneous point measurements (avgs<>1)
                        OR from measurements averaged over each sampling interval (avgs==1)

        Returns:
            psd (array): power spectral density

        References:
            Kirchner's C code for weighted psd calculation

        """
        af = AliasFilter()

        # weighted psd calculation start
        dt = np.median(np.diff(ts))

        power = wwa**2 * 0.5 * dt * Neffs

        Neff_diff = Neffs - Neff
        #  Neff_diff[Neff_diff < 0] = 0

        sum_power = np.nansum(power * Neff_diff, axis=0)
        sum_eff = np.nansum(Neff_diff, axis=0)

        psd = sum_power / sum_eff
        # weighted psd calculation end

        if anti_alias:
            assert freqs is not None, "freqs is required for alias filter!"
            dt = np.median(np.diff(ts))
            f_sampling = 1/dt
            psd_copy = psd[1:]
            freqs_copy = freqs[1:]
            alpha, filtered_pwr, model_pwer, aliased_pwr = af.alias_filter(
                freqs_copy, psd_copy, f_sampling, f_sampling*1e3, np.min(freqs), avgs)

            psd[1:] = np.copy(filtered_pwr)

        return psd

    def freq_vector_lomb_scargle(self, ts, nf=None, ofac=4, hifac=1):
        ''' Return the frequency vector based on the Lomb-Scargle algorithm.

        Args:
            ts (array): time axis of the time series
            ofac (float): Oversampling rate that influences the resolution of the frequency axis,
                         when equals to 1, it means no oversamling (should be >= 1).
                         The default value 4 is usaually a good value.

            hifac (float): fhi/fnyq (should be >= 1), where fhi is the highest frequency that
                          can be analyzed by the Lomb-Scargle algorithm and fnyq is the Nyquist frequency.

        Returns:
            freqs (array): the frequency vector

        References:
            Trauth, M. H. MATLAB® Recipes for Earth Sciences. (Springer, 2015). pp 181.

        '''
        assert ofac >= 1 and hifac <= 1, "`ofac` should be >= 1, and `hifac` should be <= 1"

        dt = np.median(np.diff(ts))
        flo = (1/(2*dt)) / (np.size(ts)*ofac)
        fhi = hifac / (2*dt)

        if nf is None:
            df = flo
            nf = (fhi - flo) / df + 1

        freqs = np.linspace(flo, fhi, nf)

        return freqs

    def freq_vector_welch(self, ts):
        ''' Return the frequency vector based on the Weltch's method.

        Args:
            ts (array): time axis of the time series

        Returns:
            freqs (array): the frequency vector

        References:
            https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/spectral.py

        '''
        nt = np.size(ts)
        dt = np.median(np.diff(ts))
        fs = 1 / dt
        if nt % 2 == 0:
            n_freqs = nt//2 + 1
        else:
            n_freqs = (nt+1) // 2

        freqs = np.arange(n_freqs) * fs / nt

        return freqs

    def freq_vector_nfft(self, ts):
        ''' Return the frequency vector based on NFFT

        Args:
            ts (array): time axis of the time series

        Returns:
            freqs (array): the frequency vector

        '''
        nt = np.size(ts)
        dt = np.median(np.diff(ts))
        fs = 1 / dt
        n_freqs = nt//2 + 1

        freqs = np.linspace(0, fs/2, n_freqs)

        return freqs

    def make_freq_vector(self, ts):
        ''' Make frequency vector

        Args:
            ts (array): time axis of the time series

        Returns:
            freqs (array): the frequency vector

        '''
        freqs = self.freq_vector_nfft(ts)
        #  freqs = freqs[1:]  # discard the first element 0

        return freqs

    def beta_estimation(self, psd, freqs, fmin, fmax):
        ''' Estimate the power slope of a 1/f^beta process.

        Args:
            psd (array): the power spectral density
            freqs (array): the frequency vector
            fmin, fmax (float): the frequency range for beta estimation

        Returns:
            beta (float): the estimated slope
            f_binned (array): binned frequency vector
            psd_binned (array): binned power spectral density
            Y_reg (array): prediction based on linear regression

        '''
        # drop the PSD at frequency zero
        if freqs[0] == 0:
            psd = psd[1:]
            freqs = freqs[1:]

        if np.max(freqs) < fmax or np.min(freqs) > fmin:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        # frequency binning start
        fminindx = np.where(freqs >= fmin)[0][0]
        fmaxindx = np.where(freqs <= fmax)[0][-1]

        if fminindx >= fmaxindx:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        logf = np.log(freqs)
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

        return beta, f_binned, psd_binned, Y_reg, std_err

    def beta2HurstIndex(self, beta):
        ''' Translate psd slope to Hurst index

        Args:
            beta (float): the estimated slope of a power spectral density curve

        Returns:
            H (float): Hurst index, should be in (0, 1)

        References:
            Equation 2 in http://www.bearcave.com/misl/misl_tech/wavelets/hurst/

        '''
        H = (beta-1)/2

        return H

    def psd_ar(self, var_noise, freqs, ar_params, f_sampling):
        ''' Return the theoretical power spectral density (PSD) of an autoregressive model

        Args:
            var_noise (float): the variance of the noise of the AR process
            freqs (array): vector of frequency
            ar_params (array): autoregressive coefficients, not including zero-lag
            f_sampling (float): sampling frequency

        Returns:
            psd (array): power spectral density

        '''
        p = np.size(ar_params)

        tmp = np.ndarray(shape=(p, np.size(freqs)), dtype=complex)
        for k in range(p):
            tmp[k, :] = np.exp(-1j*2*np.pi*(k+1)*freqs/f_sampling)

        psd = var_noise / np.absolute(1-np.sum(ar_params*tmp, axis=0))**2

        return psd

    def fBMsim(self, N=128, H=0.25):
        '''Simple method to generate fractional Brownian Motion

        Args:
            N (int): the length of the simulated time series
            H (float): Hurst index, should be in (0, 1)

        Returns:
            xfBm (array): the simulated fractional Brownian Motion time series

        References:
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

    def psd_fBM(self, freqs, ts, H):
        ''' Return the theoretical psd of a fBM

        Args:
            freqs (array): vector of frequency
            ts (array): the time axis of the time series
            H (float): Hurst index, should be in (0, 1)

        Returns:
            psd (array): power spectral density

        References:
            Flandrin, P. On the spectrum of fractional Brownian motions.
                IEEE Transactions on Information Theory 35, 197–199 (1989).

        '''
        nf = np.size(freqs)
        psd = np.ndarray(shape=(nf))
        T = np.max(ts) - np.min(ts)

        omega = 2 * np.pi * freqs

        for k in range(nf):
            tmp = 2 * omega[k] * T
            psd[k] = (1 - 2**(1 - 2*H)*np.sin(tmp)/tmp) / np.abs(omega[k])**(1 + 2*H)

        return psd

    def get_wwz_func(self, nproc, method):
        ''' Return the wwz function to use.

        Args:
            nproc (int): the number of processes for multiprocessing
            method (str): 'Foster' - the original WWZ method;
                          'Kirchner' - the method Kirchner adapted from Foster;
                          'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py
        Returns:
            wwz_func (function): the wwz function to use

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

        elif method == 'Kirchner_opt':
            wwz_func = wa.kirchner_opt

        elif method == 'Foster_opt1':
            wwz_func = wa.wwz_opt1

        elif method == 'Foster_opt2':
            wwz_func = wa.wwz_opt2

        else:
            wwz_func = wa.kirchner_f2py

        return wwz_func

    def prepare_wwz(self, ys, ts, freqs=None, tau=None, len_bd=0, bc_mode='reflect', reflect_type='odd'):
        ''' Return the truncated time series with NaNs deleted

        Args:
            ys (array): a time series, NaNs will be deleted automatically
            ts (array): the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
                if the boundaries of tau are not exactly on two of the time axis points, then tau will be adjusted to be so
            len_bd (int): the number of the ghost grids want to creat on each boundary
            bc_mode (str): see np.lib.pad()
            reflect_type (str): see np.lib.pad()

        Returns:
            ys_cut (array): the truncated time series with NaNs deleted
            ts_cut (array): the truncated time axis of the original time series with NaNs deleted
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis

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

        if freqs is None:
            freqs = self.make_freq_vector(ts_cut)

        return ys_cut, ts_cut, freqs, tau

    def cross_wt(self, coeff1, coeff2, freqs, tau):
        ''' Return the cross wavelet transform.

        Args:
            coeff1, coeff2 (array): the two sets of wavelet transform coefficients **in the form of a1 + a2*1j**
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis

        Returns:
            xw_amplitude (array): the cross wavelet amplitude
            xw_phase (array): the cross wavelet phase

        References:
            1.Grinsted, A., Moore, J. C. & Jevrejeva, S. Application of the cross wavelet transform and
                wavelet coherence to geophysical time series. Nonlin. Processes Geophys. 11, 561–566 (2004).

        '''
        xwt = coeff1 * np.conj(coeff2)
        xw_amplitude = np.sqrt(xwt.real**2 + xwt.imag**2)
        xw_phase = np.arctan2(xwt.imag, xwt.real)

        return xwt, xw_amplitude, xw_phase

    def wavelet_coherence(self, coeff1, coeff2, freqs, tau):
        ''' Return the cross wavelet transform.

        Args:
            coeff1, coeff2 (array): the two sets of wavelet transform coefficients **in the form of a1 + a2*1j**
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis

        Returns:
            xw_coherence (array): the cross wavelet coherence

        References:
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

        def Smoothing(coeff, snorm, dj):
            """ Soothing function adapted from https://github.com/regeirk/pycwt/blob/master/pycwt/helpers.py

            Args:
                coeff (array): the wavelet coefficients get from wavlet transform **in the form of a1 + a2*1j**
                snorm (array): normalized scales
                dj (float): it satisfies the equation [ Sj = S0 * 2**(j*dj) ]

            Returns:
                rect (array): the (normalized) rectangular function

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
            F = np.exp(-0.5 * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product
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
        scales = 1/freqs  # `scales` here is the `Period` axis in the wavelet plot
        dt = np.median(np.diff(tau))
        snorm = scales / dt  # normalized scales

        scale = 1/freqs

        # with WWZ method, we don't have a constant dj, so we will just take the average over the whole scale range
        N = np.size(scale)
        s0 = scale[-1]
        sN = scale[0]
        dj = np.log2(sN/s0) / N

        S12 = Smoothing(xwt/scale, snorm, dj)
        S1 = Smoothing(power1/scale, snorm, dj)
        S2 = Smoothing(power2/scale, snorm, dj)
        xw_coherence = np.abs(S12)**2 / (S1*S2)

        return xw_coherence

    def reconstruct_ts(self, coeff, freqs, tau, t, len_bd=0):
        ''' Reconstruct the normalized time series from the wavelet coefficients.
        Args:
            coeff (array): the coefficients of the corresponding basis functions (a0, a1, a2)
            freqs (array): vector of frequency of the basis functions
            tau (array): the evenly-spaced time points of the basis functions
            t (array): the specified evenly-spaced time points of the reconstructed time series
            len_bd (int): the number of the ghost grids want to creat on each boundary

        Returns:
            rec_ts (array): the reconstructed normalized time series
            t (array): the evenly-spaced time points of the reconstructed time series
        '''
        omega = 2*np.pi*freqs
        nf = np.size(freqs)

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

        rec_ts = self.preprocess(rec_ts, t, detrend='no', gaussianize=False, standardize=True)

        return rec_ts, t


class AliasFilter(object):
    '''Performing anti-alias filter on a psd @author: fzhu
    '''

    def alias_filter(self, freq, pwr, fs, fc, f_limit, avgs):
        '''
        Args:
            freq (array): vector of frequencies in power spectrum
            pwr (array): vector of spectral power corresponding to frequencies "freq"
            fs (float): sampling frequency
            fc (float): corner frequency for 1/f^2 steepening of power spectrum
            f_limit (float): lower frequency limit for estimating misfit of model-plus-alias spectrum vs. measured power
            avgs (int): flag for whether spectrum is derived from instantaneous point measurements (avgs<>1)
                        OR from measurements averaged over each sampling interval (avgs==1)

        Returns:
            alpha (float): best-fit exponent of power-law model
            filtered_pwr (array): vector of alias-filtered spectral power
            model_pwr (array): vector of modeled spectral power
            aliased_pwr (array): vector of modeled spectral power, plus aliases

        References:
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

        Args:
            y (array): the values of the time history of the signal.
            window_size (int) : the length of the window. Must be an odd integer number.
            order (int) : the order of the polynomial used in the filtering. Must be less then `window_size` - 1.
            deriv (int): the order of the derivative to compute (default = 0 means only smoothing)

        Returns:

            ys - ndarray of shape (N), the smoothed signal (or it's n-th derivative).

        Reference:

            - A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
                Data by Simplified Least Squares Procedures. Analytical
                Chemistry, 1964, 36 (8), pp 1627-1639.
            - Numerical Recipes 3rd Edition: The Art of Scientific Computing
                W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
                Cambridge University Press ISBN-13: 9780521880688
            - SciPy Cookbook: shttps://github.com/scipy/scipy-cookbook/blob/master/ipython/SavitzkyGolay.ipynb
        """
        if type(window_size)is not int:
            sys.exit("window_size should be of type int")
        if type(order) is not int:
            sys.exit("order should be of type int")
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

'''
Interface for the users below, more checks about the input will be performed here
'''


def ar1_fit(ys, ts=None, detrend='no', params=["default", 4, 0, 1]):
    ''' Returns the lag-1 autocorrelation from ar1 fit OR persistence from tauest.

    Args:
        ys (array): the time series
        ts (array): the time axis of that series
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted
                       'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
            params (list): The paramters for the Savitzky-Golay filters. The first parameter
                corresponds to the window size (default it set to half of the data)
                while the second parameter correspond to the order of the filter
                (default is 4). The third parameter is the order of the derivative
                (the default is zero, which means only smoothing.)

    Returns:
        g (float): lag-1 autocorrelation coefficient (for evenly-spaced time series)
        OR estimated persistence (for unevenly-spaced time series)
    '''

    wa = WaveletAnalysis()

    if wa.is_evenly_spaced(ts):
        g = wa.ar1_fit_evenly(ys, ts, detrend=detrend, params=params)
    else:
        g = wa.tau_estimation(ys, ts, detrend=detrend, params=params)

    return g


def ar1_sim(ys, n, p, ts=None, detrend='no', params=["default", 4, 0, 1]):
    ''' Produce p realizations of an AR1 process of length n with lag-1 autocorrelation g calculated from `ys` and `ts`

    Args:
        ys (array): a time series
        n, p (int): dimensions as n rows by p columns
        ts (array): the time axis of that series
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted
                       'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
        params (list): The paramters for the Savitzky-Golay filters. The first parameter
            corresponds to the window size (default it set to half of the data)
            while the second parameter correspond to the order of the filter
            (default is 4). The third parameter is the order of the derivative
            (the default is zero, which means only smoothing.)

    Returns:
        red (matrix): n rows by p columns matrix of an AR1 process

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


def wwz(ys, ts, tau=None, freqs=None, c=1/(8*np.pi**2), Neff=3, Neff_coi=3, nMC=200, nproc=8,
        detrend='no', params=['default', 4, 0, 1], gaussianize=False, standardize=True,
        method='Kirchner_f2py', len_bd=0, bc_mode='reflect', reflect_type='odd'):
    ''' Return the weighted wavelet amplitude (WWA) with phase, AR1_q, and cone of influence, as well as WT coeeficients

    Args:
        ys (array): a time series, NaNs will be deleted automatically
        ts (array): the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
        tau (array): the evenly-spaced time points
        freqs (array): vector of frequency
        c (float): the decay constant, the default value 1/(8*np.pi**2) is good for most of the cases
        Neff (int): effective number of points
        nMC (int): the number of Monte-Carlo simulations
        nproc (int): the number of processes for multiprocessing
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted
                       'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
        params (list): The paramters for the Savitzky-Golay filters. The first parameter
            corresponds to the window size (default it set to half of the data)
            while the second parameter correspond to the order of the filter
            (default is 4). The third parameter is the order of the derivative
            (the default is zero, which means only smoothing.)
        method (str): 'Foster' - the original WWZ method;
                      'Kirchner' - the method Kirchner adapted from Foster;
                      'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py
        len_bd (int): the number of the ghost grids want to creat on each boundary
        bc_mode (str): see np.lib.pad()
        reflect_type (str): see np.lib.pad()

    Returns:
        wwa (array): the weighted wavelet amplitude.
        AR1_q (array): AR1 simulations
        coi (array): cone of influence
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
        Neffs (array): the matrix of effective number of points in the time-scale coordinates
        coeff (array): the wavelet transform coefficents

    '''
    if method == 'Kirchner_f2py':
        if not sys.platform.startswith('darwin'):
            warnings.warn("WWZ method: the f2py version is only supported on macOS right now; will use python version instead.")
            method = 'Kirchner'

    wa = WaveletAnalysis()
    assert isinstance(nMC, int) and nMC >= 0, "nMC should be larger than or eaqual to 0."

    ys_cut, ts_cut, freqs, tau = wa.prepare_wwz(ys, ts, freqs=freqs, tau=tau,
                                                len_bd=len_bd, bc_mode=bc_mode, reflect_type=reflect_type)

    wwz_func = wa.get_wwz_func(nproc, method)
    wwa, phase, Neffs, coeff = wwz_func(ys_cut, ts_cut, freqs, tau, Neff=Neff, c=c, nproc=nproc,
                                        detrend=detrend, params=params,
                                        gaussianize=gaussianize, standardize=standardize)

    # Monte-Carlo simulations of AR1 process
    nt = np.size(tau)
    nf = np.size(freqs)

    wwa_red = np.ndarray(shape=(nMC, nt, nf))
    AR1_q = np.ndarray(shape=(nt, nf))

    if nMC >= 1:
        #  tauest = wa.tau_estimation(ys_cut, ts_cut, detrend=detrend, gaussianize=gaussianize, standardize=standardize)

        for i in tqdm(range(nMC), desc='Monte-Carlo simulations'):
            #  r = wa.ar1_model(ts_cut, tauest)
            r = ar1_sim(ys_cut, np.size(ts_cut), 1, ts=ts_cut)
            wwa_red[i, :, :], _, _, _ = wwz_func(r, ts_cut, freqs, tau, c=c, Neff=Neff, nproc=nproc,
                                                 detrend=detrend, params=params,
                                                 gaussianize=gaussianize, standardize=standardize)

        for j in range(nt):
            for k in range(nf):
                AR1_q[j, k] = mquantiles(wwa_red[:, j, k], 0.95)

    else:
        AR1_q = None

    # calculate the cone of influence
    coi = wa.make_coi(tau, Neff=Neff_coi)

    return wwa, phase, AR1_q, coi, freqs, tau, Neffs, coeff


def wwz_psd(ys, ts, freqs=None, tau=None, c=1e-3, nproc=8, nMC=200,
            detrend='no', params=["default", 4, 0, 1], gaussianize=False, standardize=True,
            Neff=3, anti_alias=False, avgs=1, method='Kirchner_f2py'):
    ''' Return the psd of a timeseries directly using wwz method.

    Args:
        ys (array): a time series, NaNs will be deleted automatically
        ts (array): the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
        c (float): the decay constant, the default value 1e-3 is good for most of the cases
        nproc (int): the number of processes for multiprocessing
        nMC (int): the number of Monte-Carlo simulations
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted
                       'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
        params (list): The paramters for the Savitzky-Golay filters. The first parameter
            corresponds to the window size (default it set to half of the data)
            while the second parameter correspond to the order of the filter
            (default is 4). The third parameter is the order of the derivative
            (the default is zero, which means only smoothing.)
        gaussionize (bool): If True, gaussianizes the timeseries
        standardize (bool): If True, standardizes the timeseries
        method (str): 'Foster' - the original WWZ method;
                      'Kirchner' - the method Kirchner adapted from Foster;
                      'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py

    Returns:
        psd (array): power spectral density
        freqs (array): vector of frequency
        psd_ar1_q95 (array): the 95% quantile of the psds of AR1 processes
        psd_ar1 (array): the psds of AR1 processes

    '''
    wa = WaveletAnalysis()
    ys_cut, ts_cut, freqs, tau = wa.prepare_wwz(ys, ts, freqs=freqs, tau=tau)

    # get wwa but AR1_q is not needed here so set nMC=0
    wwa, _, _, coi, freqs, _, Neffs, _ = wwz(ys_cut, ts_cut, freqs=freqs, tau=tau, c=c, nproc=nproc, nMC=0,
                                             detrend=detrend, params=params,
                                             gaussianize=gaussianize, standardize=standardize, method=method)

    psd = wa.wwa2psd(wwa, ts_cut, Neffs, freqs=freqs, Neff=Neff, anti_alias=anti_alias, avgs=avgs)
    #  psd[1/freqs > np.max(coi)] = np.nan  # cut off the unreliable part out of the coi
    #  psd = psd[1/freqs <= np.max(coi)] # cut off the unreliable part out of the coi
    #  freqs = freqs[1/freqs <= np.max(coi)]

    # Monte-Carlo simulations of AR1 process
    nf = np.size(freqs)

    psd_ar1 = np.ndarray(shape=(nMC, nf))

    if nMC >= 1:
        #  tauest = wa.tau_estimation(ys_cut, ts_cut, detrend=detrend)

        for i in tqdm(range(nMC), desc='Monte-Carlo simulations'):
            #  r = wa.ar1_model(ts_cut, tauest)
            r = ar1_sim(ys_cut, np.size(ts_cut), 1, ts=ts_cut)
            wwa_red, _, _, coi_red, freqs_red, _, Neffs_red, _ = wwz(r, ts_cut, freqs=freqs, tau=tau, c=c, nproc=nproc, nMC=0,
                                                                     detrend=detrend, params=params,
                                                                     gaussianize=gaussianize, standardize=standardize,
                                                                     method=method)
            psd_ar1[i, :] = wa.wwa2psd(wwa_red, ts_cut, Neffs_red, freqs=freqs, Neff=Neff, anti_alias=anti_alias, avgs=avgs)
            #  psd_ar1[i, 1/freqs_red > np.max(coi_red)] = np.nan  # cut off the unreliable part out of the coi
            #  psd_ar1 = psd_ar1[1/freqs_red <= np.max(coi_red)] # cut off the unreliable part out of the coi

        psd_ar1_q95 = mquantiles(psd_ar1, 0.95, axis=0)[0]

    else:
        psd_ar1_q95 = None

    return psd, freqs, psd_ar1_q95, psd_ar1


def xwt(ys1, ts1, ys2, ts2,
        tau=None, freqs=None, c=1/(8*np.pi**2), Neff=3, Neff_coi=6, nproc=8, detrend='no', params=['default', 4, 0, 1],
        gaussianize=False, standardize=True,
        method='Kirchner_f2py'):
    ''' Return the crosse wavelet transform of two time series.

    Args:
        ys1, ys2 (array): the two time series
        ts1, ts2 (array): the time axis of the two time series
        tau (array): the evenly-spaced time points
        freqs (array): vector of frequency
        c (float): the decay constant, the default value 1/(8*np.pi**2) is good for most of the cases
        Neff (int): effective number of points
        nproc (int): the number of processes for multiprocessing
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted
                       'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
        params (list): The paramters for the Savitzky-Golay filters. The first parameter
            corresponds to the window size (default it set to half of the data)
            while the second parameter correspond to the order of the filter
            (default is 4). The third parameter is the order of the derivative
            (the default is zero, which means only smoothing.)
        gaussionize (bool): If True, gaussianizes the timeseries
        standardize (bool): If True, standardizes the timeseries
        method (str): 'Foster' - the original WWZ method;
                      'Kirchner' - the method Kirchner adapted from Foster;
                      'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py

    Returns:
        xw_amplitude (array): the cross wavelet amplitude
        xw_phase (array): the cross wavelet phase
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points
        AR1_q (array): AR1 simulations
        coi (array): cone of influence

    '''
    wa = WaveletAnalysis()

    wwz_func = wa.get_wwz_func(nproc, method)

    ys1_cut, ts1_cut, freqs, tau = wa.prepare_wwz(ys1, ts1, freqs=freqs, tau=tau)
    ys2_cut, ts2_cut, freqs, tau = wa.prepare_wwz(ys2, ts2, freqs=freqs, tau=tau)

    wwa, phase, Neffs, coeff1 = wwz_func(ys1_cut, ts1_cut, freqs, tau, Neff=Neff, c=c, nproc=nproc, detrend=detrend,
                                         params=params, gaussianize=gaussianize, standardize=standardize)
    wwa, phase, Neffs, coeff2 = wwz_func(ys2_cut, ts2_cut, freqs, tau, Neff=Neff, c=c, nproc=nproc, detrend=detrend,
                                         params=params, gaussianize=gaussianize, standardize=standardize)

    tauest1 = wa.tau_estimation(ys1_cut, ts1_cut, detrend=detrend, params=params,
                                gaussianize=gaussianize, standardize=standardize)
    tauest2 = wa.tau_estimation(ys2_cut, ts2_cut, detrend=detrend, params=params,
                                gaussianize=gaussianize, standardize=standardize)
    r1 = wa.ar1_model(ts1_cut, tauest1)
    r2 = wa.ar1_model(ts2_cut, tauest2)
    #  r1 = ar1_sim(ys1_cut, np.size(ts1_cut), 1, ts=ts1_cut)
    #  r2 = ar1_sim(ys2_cut, np.size(ts2_cut), 1, ts=ts2_cut)

    #  wwa_red1, _, Neffs_red1, _ = wwz_func(r1, ts1_cut, freqs, tau, c=c, Neff=Neff, nproc=nproc, detrend=detrend,
    #                                        gaussianize=gaussianize, standardize=standardize)
    #  wwa_red2, _, Neffs_red2, _ = wwz_func(r2, ts2_cut, freqs, tau, c=c, Neff=Neff, nproc=nproc, detrend=detrend,
    #                                        gaussianize=gaussianize, standardize=standardize)
    #  psd1_ar1 = wa.wwa2psd(wwa_red1, ts1_cut, Neffs_red1, freqs=freqs, Neff=Neff, anti_alias=False, avgs=1)
    #  psd2_ar1 = wa.wwa2psd(wwa_red2, ts2_cut, Neffs_red2, freqs=freqs, Neff=Neff, anti_alias=False, avgs=1)
    dt1 = np.median(np.diff(ts1))
    dt2 = np.median(np.diff(ts2))
    f_sampling_1 = 1/dt1
    f_sampling_2 = 1/dt2
    psd1_ar1 = wa.psd_ar(np.var(r1), freqs, tauest1, f_sampling_1)
    psd2_ar1 = wa.psd_ar(np.var(r2), freqs, tauest2, f_sampling_2)

    wt_coeff1 = coeff1[1] + coeff1[2]*1j
    wt_coeff2 = coeff2[1] + coeff2[2]*1j
    xwt, xw_amplitude, xw_phase = wa.cross_wt(wt_coeff1, wt_coeff2, freqs, tau)

    sigma_1 = np.std(ys1_cut)
    sigma_2 = np.std(ys2_cut)
    nu, Znu = 2, 3.9999  # according to `xwt.m` from Grinsted's MATLAB code

    signif = sigma_1*sigma_2 * np.sqrt(psd1_ar1*psd2_ar1) * Znu/nu  # Eq. (5) of Grinsted et al 2004
    AR1_q = np.tile(signif, (np.size(tau), 1))

    coi = wa.make_coi(tau, Neff=Neff_coi)

    return xwt, xw_amplitude, xw_phase, freqs, tau, AR1_q, coi


def xwc(ys1, ts1, ys2, ts2,
        tau=None, freqs=None, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend='no',
        nMC=200, params=['default', 4, 0, 1],
        gaussianize=False, standardize=True, method='Kirchner_f2py'):
    ''' Return the crosse wavelet coherence of two time series.

    Args:
        ys1, ys2 (array): the two time series
        ts1, ts2 (array): the time axis of the two time series
        tau (array): the evenly-spaced time points
        freqs (array): vector of frequency
        c (float): the decay constant, the default value 1/(8*np.pi**2) is good for most of the cases
        Neff (int): effective number of points
        nproc (int): the number of processes for multiprocessing
        nMC (int): the number of Monte-Carlo simulations
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted
                       'savitzy-golay' - ys is filtered using the Savitzky-Golay
                               filters and the resulting filtered series is subtracted from y.
        params (list): The paramters for the Savitzky-Golay filters. The first parameter
            corresponds to the window size (default it set to half of the data)
            while the second parameter correspond to the order of the filter
            (default is 4). The third parameter is the order of the derivative
            (the default is zero, which means only smoothing.)
        gaussionize (bool): If True, gaussianizes the timeseries
        standardize (bool): If True, standardizes the timeseries
        method (str): 'Foster' - the original WWZ method;
                      'Kirchner' - the method Kirchner adapted from Foster;
                      'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py

    Returns:
        xw_coherence (array): the cross wavelet coherence
        xw_phase (array): the cross wavelet phase
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points
        AR1_q (array): AR1 simulations
        coi (array): cone of influence

    '''
    if sys.platform.startswith('linux') and method == 'Kirchner_f2py':
        warnings.warn("The f2py version is not supported for Linux right now; will use python version instead.")
        method = 'Kirchner'

    wa = WaveletAnalysis()
    assert isinstance(nMC, int) and nMC >= 0, "nMC should be larger than or eaqual to 0."

    ys1_cut, ts1_cut, freqs1, tau1 = wa.prepare_wwz(ys1, ts1, freqs=freqs, tau=tau)
    ys2_cut, ts2_cut, freqs2, tau2 = wa.prepare_wwz(ys2, ts2, freqs=freqs, tau=tau)

    if np.any(tau1 != tau2):
        print('inconsistent `tau`, recalculating...')
        tau_min = np.min([np.min(tau1), np.min(tau2)])
        tau_max = np.max([np.max(tau1), np.max(tau2)])
        ntau = np.max([np.size(tau1), np.size(tau2)])
        tau = np.linspace(tau_min, tau_max, ntau)
    else:
        tau = tau1

    if np.any(freqs1 != freqs2):
        print('inconsistent `freqs`, recalculating...')
        freqs_min = np.min([np.min(freqs1), np.min(freqs2)])
        freqs_max = np.max([np.max(freqs1), np.max(freqs2)])
        nfreqs = np.max([np.size(freqs1), np.size(freqs2)])
        freqs = np.linspace(freqs_min, freqs_max, nfreqs)
    else:
        freqs = freqs1

    wwa1, phase1, AR1_q, coi, freqs, tau, Neffs, coeff1 = wwz(ys1_cut, ts1_cut, tau=tau, freqs=freqs, c=c, Neff=Neff, nMC=0,
                                                              nproc=nproc, detrend=detrend, params=params,
                                                              gaussianize=gaussianize, standardize=standardize, method=method)
    wwa2, phase2, AR1_q, coi, freqs, tau, Neffs, coeff2 = wwz(ys2_cut, ts2_cut, tau=tau, freqs=freqs, c=c, Neff=Neff, nMC=0,
                                                              nproc=nproc, detrend=detrend, params=params,
                                                              gaussianize=gaussianize, standardize=standardize, method=method)

    wt_coeff1 = coeff1[1] + coeff1[2]*1j
    wt_coeff2 = coeff2[1] + coeff2[2]*1j
    xw_coherence = wa.wavelet_coherence(wt_coeff1, wt_coeff2, freqs, tau)
    xwt, xw_amplitude, xw_phase = wa.cross_wt(wt_coeff1, wt_coeff2, freqs, tau)

    # Monte-Carlo simulations of AR1 process
    nt = np.size(tau)
    nf = np.size(freqs)

    coherence_red = np.ndarray(shape=(nMC, nt, nf))
    AR1_q = np.ndarray(shape=(nt, nf))

    if nMC >= 1:

        for i in tqdm(range(nMC), desc='Monte-Carlo simulations'):
            r1 = ar1_sim(ys1_cut, np.size(ts1_cut), 1, ts=ts1_cut)
            r2 = ar1_sim(ys2_cut, np.size(ts2_cut), 1, ts=ts2_cut)
            _, _, _, _, freqs, tau, _, coeffr1 = wwz(r1, ts1_cut, tau=tau, freqs=freqs, c=c, Neff=Neff, nMC=0, nproc=nproc,
                                                     detrend=detrend, params=params,
                                                     gaussianize=gaussianize, standardize=standardize)
            _, _, _, _, freqs, tau, _, coeffr2 = wwz(r2, ts2_cut, tau=tau, freqs=freqs, c=c, Neff=Neff, nMC=0, nproc=nproc,
                                                     detrend=detrend, params=params,
                                                     gaussianize=gaussianize, standardize=standardize)

            wt_coeffr1 = coeffr1[1] + coeffr1[2]*1j
            wt_coeffr2 = coeffr2[1] + coeffr2[2]*1j
            coherence_red[i, :, :] = wa.wavelet_coherence(wt_coeffr1, wt_coeffr2, freqs, tau)

        for j in range(nt):
            for k in range(nf):
                AR1_q[j, k] = mquantiles(coherence_red[:, j, k], 0.95)

    else:
        AR1_q = None

    return xw_coherence, xw_amplitude, xw_phase, freqs, tau, AR1_q, coi


def plot_wwa(wwa, freqs, tau, AR1_q=None, coi=None, levels=None, tick_range=None,
             yticks=None, yticks_label=None, ylim=None, xticks=None, xlabels=None, figsize=[20, 8], clr_map='OrRd',
             cbar_drawedges=False, cone_alpha=0.5, plot_signif=False, signif_style='contour', title=None,
             plot_cone=False, ax=None, xlabel='Year', ylabel='Period', cbar_orientation='vertical',
             cbar_pad=0.05, cbar_frac=0.15, cbar_labelsize=None):
    """ Plot the wavelet amplitude

    Args:
        wwa (array): the weighted wavelet amplitude.
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
        AR1_q (array): AR1 simulations
        coi (array): cone of influence
        levels (array): levels of values to plot
        tick_range (array): levels of ticks to show on the colorbar
        yticks (list): ticks on y-axis
        ylim (list): limitations for y-axis
        xticks (list): ticks on x-axis
        figsize (list): the size for the figure
        clr_map (str): the name of the colormap
        cbar_drawedges (bool): whether to draw edges on the colorbar or not
        cone_alpha (float): the alpha value for the area covered by cone of influence
        plot_signif (bool): plot 95% significant area or not
        signif_style (str): plot 95% significant area with `contour` or `shade`
        plot_cone (bool): plot cone of influence or not
        ax: Return as axis instead of figure (useful to integrate plot into a subplot)
        xlabel (str): The x-axis label
        ylabel (str): The y-axis label
        cbar_pad (float): the pad for the colorbar
        cbar_frac (float): the frac for the colorbar
        cbar_labelsize (float): the font size of the colorbar label

    Returns:
        fig (figure): the 2-D plot of wavelet analysis

    """
    sns.set(style="ticks", font_scale=2)
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    if levels is None:
        q95 = mquantiles(wwa, 0.95)
        if np.nanmax(wwa) > 2*q95:
            warnings.warn("There are outliers in the input amplitudes, " +
                          "and the `levels` have been set so that the outpliers will be ignored! " +
                          "One might want to use `Spectral.plot_wwadist(wwa)` to plot the distribution of " +
                          "the amplitudes with the 95% quantile line to check if the levels are appropriate.", stacklevel=2)

            max_level = np.round(2*q95, decimals=1)
            if max_level == 0:
                max_level = 0.01
            levels = np.linspace(0, max_level, 11)

    origin = 'lower'

    if levels is not None:
        plt.contourf(tau, 1/freqs, wwa.T, levels, cmap=clr_map, origin=origin)
    else:
        plt.contourf(tau, 1/freqs, wwa.T, cmap=clr_map, origin=origin)

    cb = plt.colorbar(drawedges=cbar_drawedges, orientation=cbar_orientation, fraction=cbar_frac, pad=cbar_pad,
                      ticks=tick_range)

    if cbar_labelsize is not None:
        cb.ax.tick_params(labelsize=cbar_labelsize)

    plt.yscale('log', nonposy='clip')

    if yticks is not None:
        yticks_label = list(map(str, yticks))
        plt.yticks(yticks, yticks_label)

    if xticks is not None:
        plt.xticks(xticks)

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
            plt.contour(tau, 1/freqs, signif.T, [-99, 1], colors='k')
        elif signif_style == 'shade':
            plt.contourf(tau, 1/freqs, signif.T, [-99, 1], colors='k', alpha=0.1)  # significant if not shaded

    if plot_cone:
        assert coi is not None, "Please set values for `coi`!"
        plt.plot(tau, coi, 'k--')
        ax.fill_between(tau, coi, ylim[1], color='white', alpha=cone_alpha)

    if title is not None:
        plt.title(title)

    return ax


def plot_coherence(xw_coherence, xw_phase, freqs, tau, AR1_q=None, coi=None, levels=None, tick_range=None, basey=2,
                   yticks=None, ylim=None, xticks=None, xlabels=None, figsize=[20, 8], clr_map='OrRd',
                   exg=5, scale=30, width=0.004,
                   cbar_drawedges=False, cone_alpha=0.5, plot_signif=False, signif_style='contour', title=None,
                   plot_cone=False, ax=None, xlabel='Year', ylabel='Period', cbar_orientation='vertical',
                   cbar_pad=0.05, cbar_frac=0.15, cbar_labelsize=None):
    """ Plot the wavelet amplitude

    Args:
        xw_coherence (array): the wavelet cohernce
        xw_phase (array): the wavelet cohernce phase
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
        AR1_q (array): AR1 simulations
        coi (array): cone of influence
        levels (array): levels of values to plot
        tick_range (array): levels of ticks to show on the colorbar
        yticks (list): ticks on y-axis
        ylim (list): limitations for y-axis
        xticks (list): ticks on x-axis
        figsize (list): the size for the figure
        clr_map (str): the name of the colormap
        cbar_drawedges (bool): whether to draw edges on the colorbar or not
        cone_alpha (float): the alpha value for the area covered by cone of influence
        plot_signif (bool): plot 95% significant area or not
        signif_style (str): plot 95% significant area with `contour` or `shade`
        plot_cone (bool): plot cone of influence or not
        ax: Return as axis instead of figure (useful to integrate plot into a subplot)
        xlabel (str): The x-axis label
        ylabel (str): The y-axis label
        cbar_pad (float): the pad for the colorbar
        c)bar_frac (float): the frac for the colorbar
        cbar_labelsize (float): the font size of the colorbar label

    Returns:
        fig (figure): the 2-D plot of wavelet analysis

    """
    sns.set(style="ticks", font_scale=2)
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # plot coherence with significance test
    if levels is None:
        levels = np.linspace(0, 1, 11)

    origin = 'lower'

    plt.contourf(tau, 1/freqs, xw_coherence.T, levels, cmap=clr_map, origin=origin)

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
        assert AR1_q is not None, "Please set values for `AR1_q`!"
        signif = xw_coherence / AR1_q
        if signif_style == 'contour':
            plt.contour(tau, 1/freqs, signif.T, [-99, 1], colors='k')
        elif signif_style == 'shade':
            plt.contourf(tau, 1/freqs, signif.T, [-99, 1], colors='k', alpha=0.1)  # significant if not shaded

    if plot_cone:
        assert coi is not None, "Please set values for `coi`!"
        plt.plot(tau, coi, 'k--')
        ax.fill_between(tau, coi, ylim[1], color='white', alpha=cone_alpha)

    if title is not None:
        plt.title(title)

    ax.set_ylim(ylim)

    # plot phase
    phase = np.copy(xw_phase)
    phase[xw_coherence < .5] = np.nan

    X, Y = np.meshgrid(tau, 1/freqs)
    U, V = np.cos(phase).T, np.sin(phase).T

    ax.quiver(X[::exg, ::exg], Y[::exg, ::exg], U[::exg, ::exg], V[::exg, ::exg], scale=scale, width=width)

    return ax


def plot_wwadist(wwa, ylim=None):
    ''' Plot the distribution of wwa with the 95% quantile line.

    Args:
        wwa (array): the weighted wavelet amplitude.
        ylim (list): limitations for y-axis

    Returns:
        fig (figure): the 2-D plot of wavelet analysis

    '''
    sns.set(style="darkgrid", font_scale=2)
    plt.subplots(figsize=[20, 4])
    q95 = mquantiles(wwa, 0.95)
    fig = sns.distplot(np.nan_to_num(wwa.flat))
    fig.axvline(x=q95, ymin=0, ymax=0.5, linewidth=2, linestyle='-')

    if ylim is not None:
        plt.ylim(ylim)

    return fig


def plot_psd(psd, freqs, lmstyle='-', linewidth=None, color=sns.xkcd_rgb["denim blue"], ar1_lmstyle='-', ar1_linewidth=None,
             period_ticks=None, period_tickslabel=None, psd_lim=None, period_lim=None,
             figsize=[20, 8], label='PSD', plot_ar1=False, psd_ar1_q95=None, title=None, legend=True,
             psd_ar1_color=sns.xkcd_rgb["pale red"], ax=None, vertical=False, plot_gridlines=True,
             period_label='Period (years)', psd_label='Spectral Density', zorder=None):
    """ Plot the wavelet amplitude

    Args:
        psd (array): power spectral density
        freqs (array): vector of frequency
        period_ticks (list): ticks for period
        psd_lim (list): limits for spectral density axis
        label (str): the label for the PSD
        plot_ar1 (bool): plot the ar1 curve or not
        psd_ar1_q95 (array): the 95% quantile of the AR1 PSD
        psd_ar1_color (str): the color for the 95% quantile of the AR1 PSD
        title (str): the title for the figure
        period_lim (list): limits for period axis
        figsize (list): the size for the figure
        ax (axis): Return as axis instead of figure (useful to integrate plot into a subplot)
        vertical (bool): plot in vertical layout or not
        legend (bool): plot legend or not
        lmstyle (str): the line style
        linewidth (float): the line width
        period_label (str): the label for period
        psd_label (str): the label for psd
        zorder (int): the order of the layer

    Returns:
        ax (figure): the 2-D plot of wavelet analysis

    """
    sns.set(style="ticks", font_scale=2)

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    if title is not None:
        ax.set_title(title)

    if vertical:
        x_data = psd
        y_data = 1 / freqs
        x_data_ar1 = psd_ar1_q95
        y_data_ar1 = 1 / freqs
    else:
        x_data = 1 / freqs
        y_data = psd
        x_data_ar1 = 1 / freqs
        y_data_ar1 = psd_ar1_q95

    if zorder is not None:
        ax.plot(x_data, y_data, lmstyle, linewidth=linewidth, label=label, zorder=zorder, color=color)
        if plot_ar1:
            assert psd_ar1_q95 is not None, "psd_ar1_q95 is required!"
            ax.plot(x_data_ar1, y_data_ar1, ar1_lmstyle, linewidth=ar1_linewidth,
                     label='AR(1) 95%', color=psd_ar1_color, zorder=zorder-1)
    else:
        ax.plot(x_data, y_data, lmstyle, linewidth=linewidth, label=label, color=color)
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


def plot_summary(ys, ts, freqs=None, tau=None, c1=1/(8*np.pi**2), c2=1e-3, nMC=200, nproc=8, detrend='no',
                 gaussianize=False, standardize=True, levels=None, method='Kirchner_f2py',
                 anti_alias=False, period_ticks=None, ts_color=None,
                 title=None, ts_ylabel=None, wwa_xlabel=None, wwa_ylabel=None,
                 psd_lmstyle='-', psd_lim=None, period_I=[1/8, 1/2], period_D=[1/200, 1/20]):
    """ Plot the time series with the wavelet analysis and psd

    Args:
        ys (array): a time series
        ts (array): time axis of the time series
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
        c (float): the decay constant
        Neff (int): the threshold of the number of effective degree of freedom
        nproc (int): fake argument, just for convenience
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted
        ts_color (str): the color for the time series curve
        title (str): the title for the time series plot
        ts_ylabel (str): label for y-axis in the time series plot
        wwa_xlabel (str): label for x-axis in the wwa plot
        wwa_ylabel (str): label for y-axis in the wwa plot
        psd_lmstyle (str): the line style in the psd plot
        psd_lim (list): the limits for psd
        period_I, period_D (list): the ranges for beta estimation

    Returns:
        fig (figure): the summary plot

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
    sns.set(style="ticks", font_scale=1.5)
    ax1 = plt.subplot(gs[0:1, :-3])
    plt.plot(ts, ys, '-o', color=ts_color)

    if title is not None:
        plt.title(title, **title_font)

    plt.xlim([np.min(ts), np.max(ts)])

    if ts_ylabel is not None:
        plt.ylabel(ts_ylabel)

    plt.grid()
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    # plot wwa
    sns.set(style="ticks", font_scale=1.5)
    ax2 = plt.subplot(gs[1:5, :-3])

    wwa, phase, AR1_q, coi, freqs, tau, Neffs, coeff = \
        wwz(ys, ts, freqs=freqs, tau=tau, c=c1, nMC=nMC, nproc=nproc, detrend=detrend, method=method,
            gaussianize=gaussianize, standardize=standardize)

    if wwa_xlabel is not None and wwa_ylabel is not None:
        plot_wwa(wwa, freqs, tau, coi=coi, AR1_q=AR1_q, yticks=period_ticks, yticks_label=period_tickslabel,
                 ylim=[ylim_min, np.max(coi)],
                 plot_cone=True, plot_signif=True, xlabel=wwa_xlabel, ylabel=wwa_ylabel, ax=ax2, levels=levels,
                 cbar_orientation='horizontal', cbar_labelsize=15, cbar_pad=0.1, cbar_frac=0.15,
                 )
    else:
        plot_wwa(wwa, freqs, tau, coi=coi, AR1_q=AR1_q, yticks=period_ticks, yticks_label=period_tickslabel,
                 ylim=[ylim_min, np.max(coi)],
                 plot_cone=True, plot_signif=True, ax=ax2,
                 cbar_orientation='horizontal', cbar_labelsize=15, cbar_pad=0.1, cbar_frac=0.15, levels=levels,
                 )

    # plot psd
    sns.set(style="ticks", font_scale=1.5)
    ax3 = plt.subplot(gs[1:4, 9:])
    psd, freqs, psd_ar1_q95, psd_ar1 = wwz_psd(ys, ts, freqs=freqs, tau=tau, c=c2, nproc=nproc, nMC=nMC, method=method,
                                      detrend=detrend, gaussianize=gaussianize, standardize=standardize,
                                      anti_alias=anti_alias)

    # TODO: deal with period_ticks
    plot_psd(psd, freqs, plot_ar1=True, psd_ar1_q95=psd_ar1_q95, period_ticks=period_ticks[period_ticks < np.max(coi)],
             period_lim=[np.min(period_ticks), np.max(coi)], psd_lim=psd_lim,
             lmstyle=psd_lmstyle, ax=ax3, period_label='', label='Estimated spectrum', vertical=True)

    beta_1, f_binned_1, psd_binned_1, Y_reg_1, stderr_1 = beta_estimation(psd, freqs, period_I[0], period_I[1])
    beta_2, f_binned_2, psd_binned_2, Y_reg_2, stderr_2 = beta_estimation(psd, freqs, period_D[0], period_D[1])
    ax3.plot(Y_reg_1, 1/f_binned_1, color='k',
             label=r'$\beta_I$ = {:.2f}'.format(beta_1) + ', ' + r'$\beta_D$ = {:.2f}'.format(beta_2))
    ax3.plot(Y_reg_2, 1/f_binned_2, color='k')
    plt.tick_params(axis='y', which='both', labelleft='off')
    plt.legend(fontsize=15, bbox_to_anchor=(0, 1.2), loc='upper left', ncol=1)

    return fig


# some alias
wa = WaveletAnalysis()
beta_estimation = wa.beta_estimation
tau_estimation = wa.tau_estimation
