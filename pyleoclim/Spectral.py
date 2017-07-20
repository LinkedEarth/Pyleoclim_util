#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:23:06 2017

@author: deborahkhider

Spectral module for pyleoclim
"""

import numpy as np
import statsmodels.api as sm

from scipy import optimize
from scipy import signal
from scipy.stats.mstats import mquantiles

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

import warnings

from pyleoclim import Timeseries
import sys


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

    def ar1_fit_evenly(self, ys, detrend='no'):
        ''' Returns the lag-1 autocorrelation from ar1 fit.

        Args:
            ys (array): vector of (flaot) numbers as a time series
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted

        Returns:
            g (float): lag-1 autocorrelation coefficient

        '''
        pd_ys = self.preprocess(ys, detrend=detrend)
        ar1_mod = sm.tsa.AR(pd_ys, missing='drop').fit(maxlag=1)
        g = ar1_mod.params[0]

        return g

    def preprocess(self, ys, detrend='no'):
        ''' Return the processed time series using (detrend and) standardization.

        Args:
            ys (array): a time series
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted

        Returns:
            res (array): the processed time series

        '''

        if detrend == 'linear':
            ys_d = signal.detrend(ys, type='linear')
        elif detrend == 'constant':
            ys_d = signal.detrend(ys, type='constant')
        else:
            ys_d = ys

        res, _, _ = Timeseries.standardize(ys_d)

        return res

    def tau_estimation(self, ys, ts, detrend='no'):
        ''' Return the estimated persistence of a givenevenly/unevenly spaced time series.

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            detrend (str): 'no' - the original time series is assumed to have no trend;
                           'linear' - a linear least-squares fit to `ys` is subtracted;
                           'constant' - the mean of `ys` is subtracted

        Returns:
            tau_est (float): the estimated persistence

        References:
            Mudelsee, M. TAUEST: A Computer Program for Estimating Persistence in Unevenly Spaced Weather/Climate Time Series.
                Comput. Geosci. 28, 69–72 (2002).

        '''
        pd_ys = self.preprocess(ys, detrend=detrend)
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

    def wwz_opt2(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend='no'):
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

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients

        References:
            Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
            Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
                Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        assert nproc == 1, "wwz_basic() only supports nproc=1"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, detrend=detrend)

        omega = 2*np.pi*freqs

        Neffs = np.ndarray(shape=(nt, nf))
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

                    weighted_phi2 = np.sum(weights*phi2*pd_ys) / sum_w
                    weighted_phi3 = np.sum(weights*phi3*pd_ys) / sum_w

                    ywave_2[j, k] = 2*weighted_phi2
                    ywave_3[j, k] = 2*weighted_phi3

        wwa = np.sqrt(ywave_2**2 + ywave_3**2)
        phase = np.arctan2(ywave_3, ywave_2)
        coeff = ywave_2 + ywave_3*1j

        return wwa, phase, Neffs, coeff

    def wwz_opt1(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend='no'):
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

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients

        References:
            Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
            Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
                Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        assert nproc == 1, "wwz_basic() only supports nproc=1"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, detrend=detrend)

        omega = 2*np.pi*freqs

        Neffs = np.ndarray(shape=(nt, nf))
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

                    weighted_phi2 = np.sum(weights*phi2*pd_ys) / sum_w
                    weighted_phi3 = np.sum(weights*phi3*pd_ys) / sum_w
                    weighted_one = np.sum(weights*pd_ys) / sum_w
                    cos_shift_one = np.sum(weights*phi2) / sum_w
                    sin_shift_one = np.sum(weights*phi3) / sum_w

                    ywave_2[j, k] = 2*(weighted_phi2-weighted_one*cos_shift_one)
                    ywave_3[j, k] = 2*(weighted_phi3-weighted_one*sin_shift_one)

        wwa = np.sqrt(ywave_2**2 + ywave_3**2)
        phase = np.arctan2(ywave_3, ywave_2)
        coeff = ywave_2 + ywave_3*1j

        return wwa, phase, Neffs, coeff

    def wwz_basic(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend='no'):
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

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients

        References:
            Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
            Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
                Nonlinear Processes in Geophysics 12, 345–352 (2005).

        '''
        assert nproc == 1, "wwz_basic() only supports nproc=1"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, detrend=detrend)

        omega = 2*np.pi*freqs

        Neffs = np.ndarray(shape=(nt, nf))
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
                    ywave_2[j, k] = np.nan  # the coefficients cannot be estimated reliably when Neff_loc <= Neff
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

                    ywave_2[j, k] = S_inv[1, 0]*weighted_phi1 + S_inv[1, 1]*weighted_phi2 + S_inv[1, 2]*weighted_phi3
                    ywave_3[j, k] = S_inv[2, 0]*weighted_phi1 + S_inv[2, 1]*weighted_phi2 + S_inv[2, 2]*weighted_phi3

        wwa = np.sqrt(ywave_2**2 + ywave_3**2)
        phase = np.arctan2(ywave_3, ywave_2)
        coeff = ywave_2 + ywave_3*1j

        return wwa, phase, Neffs, coeff

    def wwz_nproc(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=8,  detrend='no'):
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

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients

        '''
        assert nproc >= 2, "wwz_nproc() should use nproc >= 2, if want serial run, please use wwz_basic()"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, detrend=detrend)

        omega = 2*np.pi*freqs

        Neffs = np.ndarray(shape=(nt, nf))
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

                ywave_2_1g = S_inv[1, 0]*weighted_phi1 + S_inv[1, 1]*weighted_phi2 + S_inv[1, 2]*weighted_phi3
                ywave_3_1g = S_inv[2, 0]*weighted_phi1 + S_inv[2, 1]*weighted_phi2 + S_inv[2, 2]*weighted_phi3

            return Neff_loc, ywave_2_1g, ywave_3_1g

        tf_mesh = np.meshgrid(tau, omega)
        list_of_grids = list(zip(*(grid.flat for grid in tf_mesh)))
        tau_grids, omega_grids = zip(*list_of_grids)

        with Pool(nproc) as pool:
            res = pool.map(wwa_1g, tau_grids, omega_grids)
            res_array = np.asarray(res)
            Neffs = res_array[:, 0].reshape((np.size(omega), np.size(tau))).T
            ywave_2 = res_array[:, 1].reshape((np.size(omega), np.size(tau))).T
            ywave_3 = res_array[:, 2].reshape((np.size(omega), np.size(tau))).T

        wwa = np.sqrt(ywave_2**2 + ywave_3**2)
        phase = np.arctan2(ywave_3, ywave_2)
        coeff = ywave_2 + ywave_3*1j

        return wwa, phase, Neffs, coeff

    def kirchner_basic(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend='no'):
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

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients

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

        pd_ys = self.preprocess(ys, detrend=detrend)

        omega = 2*np.pi*freqs

        Neffs = np.ndarray(shape=(nt, nf))
        a1 = np.ndarray(shape=(nt, nf))
        a2 = np.ndarray(shape=(nt, nf))

        for k in range(nf):
            for j in range(nt):
                dz = omega[k] * (ts - tau[j])
                weights = np.exp(-c*dz**2)

                sum_w = np.sum(weights)
                Neffs[j, k] = sum_w**2 / np.sum(weights**2)  # local number of effective dof

                if Neffs[j, k] <= Neff:
                    a1[j, k] = np.nan  # the coefficients cannot be estimated reliably when Neff_loc <= Neff
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

                    a1[j, k] = cos_tau_center*A - sin_tau_center*B  # Eq. (S6)
                    a2[j, k] = sin_tau_center*A + cos_tau_center*B  # Eq. (S7)

        wwa = np.sqrt(a1**2 + a2**2)
        phase = np.arctan2(a2, a1)
        coeff = a1 + a2*1j

        return wwa, phase, Neffs, coeff

    def kirchner_opt(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend='no'):
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

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients

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

        pd_ys = self.preprocess(ys, detrend=detrend)

        omega = 2*np.pi*freqs

        Neffs = np.ndarray(shape=(nt, nf))
        a1 = np.ndarray(shape=(nt, nf))
        a2 = np.ndarray(shape=(nt, nf))

        for k in range(nf):
            for j in range(nt):
                dz = omega[k] * (ts - tau[j])
                weights = np.exp(-c*dz**2)

                sum_w = np.sum(weights)
                Neffs[j, k] = sum_w**2 / np.sum(weights**2)  # local number of effective dof

                if Neffs[j, k] <= Neff:
                    a1[j, k] = np.nan  # the coefficients cannot be estimated reliably when Neff_loc <= Neff
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

                    A = 2*ys_cos_shift
                    B = 2*ys_sin_shift

                    a1[j, k] = cos_tau_center*A - sin_tau_center*B  # Eq. (S6)
                    a2[j, k] = sin_tau_center*A + cos_tau_center*B  # Eq. (S7)

        wwa = np.sqrt(a1**2 + a2**2)
        phase = np.arctan2(a2, a1)
        coeff = a1 + a2*1j

        return wwa, phase, Neffs, coeff

    def kirchner_nproc(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend='no'):
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

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients

        '''
        assert nproc >= 2, "wwz_nproc() should use nproc >= 2, if want serial run, please use wwz_basic()"
        self.assertPositiveInt(Neff)

        nt = np.size(tau)
        nts = np.size(ts)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, detrend=detrend)

        omega = 2*np.pi*freqs

        Neffs = np.ndarray(shape=(nt, nf))
        a1 = np.ndarray(shape=(nt, nf))
        a2 = np.ndarray(shape=(nt, nf))

        def wwa_1g(tau, omega):
            dz = omega * (ts - tau)
            weights = np.exp(-c*dz**2)

            sum_w = np.sum(weights)
            Neff_loc = sum_w**2 / np.sum(weights**2)

            if Neff_loc <= Neff:
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

                a1_1g = cos_tau_center*A - sin_tau_center*B  # Eq. (S6)
                a2_1g = sin_tau_center*A + cos_tau_center*B  # Eq. (S7)

            return Neff_loc, a1_1g, a2_1g

        tf_mesh = np.meshgrid(tau, omega)
        list_of_grids = list(zip(*(grid.flat for grid in tf_mesh)))
        tau_grids, omega_grids = zip(*list_of_grids)

        with Pool(nproc) as pool:
            res = pool.map(wwa_1g, tau_grids, omega_grids)
            res_array = np.asarray(res)
            Neffs = res_array[:, 0].reshape((np.size(omega), np.size(tau))).T
            a1 = res_array[:, 1].reshape((np.size(omega), np.size(tau))).T
            a2 = res_array[:, 2].reshape((np.size(omega), np.size(tau))).T

        wwa = np.sqrt(a1**2 + a2**2)
        phase = np.arctan2(a2, a1)
        coeff = a1 + a2*1j

        return wwa, phase, Neffs, coeff

    def kirchner_f2py(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend='no'):
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

        Returns:
            wwa (array): the weighted wavelet amplitude
            phase (array): the weighted wavelet phase
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients

        '''
        self.assertPositiveInt(Neff, nproc)

        nt = np.size(tau)
        nts = np.size(ts)
        nf = np.size(freqs)

        pd_ys = self.preprocess(ys, detrend=detrend)

        omega = 2*np.pi*freqs

        from . import f2py_wwz as f2py
        Neffs, a1, a2 = f2py.f2py_wwz.wwa(tau, omega, c, Neff, ts, pd_ys, nproc, nts, nt, nf)

        undef = -99999.
        a1[a1 == undef] = np.nan
        a2[a2 == undef] = np.nan
        wwa = np.sqrt(a1**2 + a2**2)
        phase = np.arctan2(a2, a1)

        coeff = a1 + a2*1j

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

        dt = np.mean(np.diff(tau))
        nt_half = (nt+1)//2 - 1

        A = np.append(0.00001, np.arange(nt_half)+1)
        B = A[::-1]

        if nt % 2 == 0:
            C = np.append(A, B)
        else:
            C = np.append(A, B[1:])

        coi = coi_const * dt * C

        return coi

    def wwa2psd(self, wwa, ts, Neffs, freqs=None, Neff=3, anti_alias=False, avgs=2):
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
        dt = np.mean(np.diff(ts))

        power = wwa**2 * 0.5 * dt * Neffs

        Neff_diff = Neffs - Neff
        Neff_diff[Neff_diff < 0] = 0

        sum_power = np.nansum(power * Neff_diff, axis=0)
        sum_eff = np.nansum(Neff_diff, axis=0)

        psd = sum_power / sum_eff
        # weighted psd calculation end

        if anti_alias:
            assert freqs is not None, "freqs is required for alias filter!"
            dt = np.mean(np.diff(ts))
            f_sampling = 1/dt
            alpha, filtered_pwr, model_pwer, aliased_pwr = af.alias_filter(
                freqs, psd, f_sampling, f_sampling*1e3, np.min(freqs), avgs)

            psd = np.copy(filtered_pwr)

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

        dt = np.mean(np.diff(ts))
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
        dt = np.mean(np.diff(ts))
        fs = 1 / dt
        if nt % 2 == 0:
            n_freqs = nt//2 + 1
        else:
            n_freqs = (nt+1) // 2

        freqs = np.arange(n_freqs) * fs / nt

        return freqs

    def make_freq_vector(self, ts):
        ''' Make frequency vector

        Args:
            ts (array): time axis of the time series

        Returns:
            freqs (array): the frequency vector

        '''
        freqs_welch = self.freq_vector_welch(ts)
        freqs = freqs_welch[1:]  # discard the first element 0

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
        # frequency binning start
        fminindx = np.where(freqs >= fmin)[0][0]
        fmaxindx = np.where(freqs <= fmax)[0][-1]

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

        beta = -results.params[1]  # the slope we want

        Y_reg = 10**model.predict(results.params)  # prediction based on linear regression

        return beta, f_binned, psd_binned, Y_reg

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

        References:
            1. http://cours-physique.lps.ens.fr/index.php/TD11_Correlated_Noise_2011
            2. https://www.wikiwand.com/en/Fractional_Brownian_motion

        @authors: jeg

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

    def prepare_wwz(self, ys, ts, freqs=None, tau=None):
        ''' Return the truncated time series with NaNs deleted

        Args:
            ys (array): a time series, NaNs will be deleted automatically
            ts (array): the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
                if the boundaries of tau are not exactly on two of the time axis points, then tau will be adjusted to be so

        Returns:
            ys_cut (array): the truncated time series with NaNs deleted
            ts_cut (array): the truncated time axis of the original time series with NaNs deleted
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis

        '''
        # delete NaNs if there is any
        ys_tmp = np.copy(ys)
        ys = ys[~np.isnan(ys_tmp)]
        ts = ts[~np.isnan(ys_tmp)]
        ts_tmp = np.copy(ts)
        ys = ys[~np.isnan(ts_tmp)]
        ts = ts[~np.isnan(ts_tmp)]

        if np.mean(np.diff(ts)) < 0:
            warnings.warn("The original time axis is decreasing, and it has been reversed.")
            ys = ys[::-1]
            ts = ts[::-1]

        if tau is None:
            med_res = np.size(ts) // np.median(np.diff(ts))
            tau = np.linspace(np.min(ts), np.max(ts), np.max([np.size(ts)//10, 50, med_res]))

        elif np.min(tau) < np.min(ts) and np.max(tau) > np.max(ts):
            warnings.warn("tau should be within the time span of the time series. \
                          Note that sometimes if the leading points of the time series are NaNs, \
                          they will be deleted and cause np.min(tau) < np.min(ts). \
                          A new tau with the same size of the input tau will be generated.")
            tau = np.linspace(np.min(ts), np.max(ts), np.size(tau))

        elif np.min(tau) not in ts or np.max(tau) not in ts:
            warnings.warn("The boundaries of tau are not exactly on two of the time axis points, \
                          and it will be adjusted to be so.")
            tau_lb = np.min(ts[ts > np.min(tau)])
            tau_ub = np.max(ts[ts < np.max(tau)])
            tau = np.linspace(tau_lb, tau_ub, np.size(tau))

        # truncate the time series when the range of tau is smaller than that of the time series
        ts_cut = ts[(np.min(tau) <= ts) & (ts <= np.max(tau))]
        ys_cut = ys[(np.min(tau) <= ts) & (ts <= np.max(tau))]

        if freqs is None:
            freqs = self.make_freq_vector(ts_cut)

        return ys_cut, ts_cut, freqs, tau

    def cross_wt(self, coeff1, coeff2, freqs, tau):
        ''' Return the cross wavelet transform.

        Args:
            coeff1, coeff2 (array): the two sets of wavelet transform coefficients
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

        return xw_amplitude, xw_phase

    def cross_coherence(self, coeff1, coeff2, freqs, tau, c1, c2):
        ''' Return the cross wavelet transform.

        Args:
            coeff1, coeff2 (array): the two sets of wavelet transform coefficients
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
            c1, c2 (float): normalization constants

        Returns:
            xw_coherence (array): the cross wavelet coherence

        References:
            1.Grinsted, A., Moore, J. C. & Jevrejeva, S. Application of the cross wavelet transform and
                wavelet coherence to geophysical time series. Nonlin. Processes Geophys. 11, 561–566 (2004).

        '''
        xwt = coeff1 * np.conj(coeff2)
        power1 = np.abs(coeff1)**2
        power2 = np.abs(coeff2)**2

        nt = np.size(tau)
        nf = np.size(freqs)

        omega = 2*np.pi*freqs

        xw_coherence = np.ndarray(shape=(nt, nf))

        def Smooth_time(coeff, c1):
            return None

        def Smooth_scale(coeff, c2):
            return None

        def Smoothing(coeff, c1, c2):
            S = Smooth_scale(Smooth_time(coeff, c1), c2)
            return S

        for j in range(nf):
            xw_coherence[:, j] = np.abs(Smoothing(xwt[:, j]/omega[j], c1, c2))**2 / \
                Smoothing(power1[:, j]/omega[j], c1, c2) / Smoothing(power2[:, j]/omega[j], c1, c2)

        return xw_coherence


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


'''
Interface for the users below, more checks about the input will be performed here
'''


def ar1_fit(ys, ts=None, detrend='no'):
    ''' Returns the lag-1 autocorrelation from ar1 fit OR persistence from tauest.

    Args:
        ys (array): the time series
        ts (array): the time axis of that series
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted

    Returns:
        g (float): lag-1 autocorrelation coefficient (for evenly-spaced time series)
        OR estimated persistence (for unevenly-spaced time series)
    '''

    wa = WaveletAnalysis()

    if wa.is_evenly_spaced(ts):
        g = wa.ar1_fit_evenly(ys, detrend=detrend)
    else:
        g = wa.tau_estimation(ys, ts, detrend=detrend)

    return g


def ar1_sim(ys, n, p, ts=None, detrend='no'):
    ''' Produce p realizations of an AR1 process of length n with lag-1 autocorrelation g calculated from `ys` and `ts`

    Args:
        ys (array): a time series
        n, p (int): dimensions as n rows by p columns
        ts (array): the time axis of that series
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted

    Returns:
        red (matrix): n rows by p columns matrix of an AR1 process

    '''
    red = np.empty(shape=(n, p))  # declare array

    wa = WaveletAnalysis()
    if wa.is_evenly_spaced(ts):
        g = ar1_fit(ys, ts=ts, detrend=detrend)
        sig = np.std(ys)

        # specify model parameters (statsmodel wants lag0 coefficents as unity)
        ar = np.r_[1, -g]  # AR model parameter
        ma = np.r_[1, 0.0]  # MA model parameters
        sig_n = sig*np.sqrt(1-g**2)  # theoretical noise variance for red to achieve the same variance as ys

        # simulate AR(1) model for each column
        for i in np.arange(p):
            red[:, i] = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=50, sigma=sig_n)

    else:
        tau_est = ar1_fit(ys, ts=ts, detrend=detrend)
        for i in np.arange(p):
            red[:, i] = wa.ar1_model(ts, tau_est, n=n)

    if p == 1:
        red = red[:, 0]

    return red


def wwz(ys, ts, tau=None, freqs=None, c=1/(8*np.pi**2), Neff=3, nMC=200, nproc=8, detrend='no', method='Kirchner_f2py'):
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
        method (str): 'Foster' - the original WWZ method;
                      'Kirchner' - the method Kirchner adapted from Foster;
                      'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py

    Returns:
        wwa (array): the weighted wavelet amplitude.
        AR1_q (array): AR1 simulations
        coi (array): cone of influence
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
        Neffs (array): the matrix of effective number of points in the time-scale coordinates
        coeff (array): the wavelet transform coefficents

    '''
    if sys.platform.startswith('linux') and method == 'Kirchner_f2py':
        warnings.warn("The f2py version is not supported for Linux right now; will use python version instead.")
        method = 'Kirchner'

    wa = WaveletAnalysis()
    assert isinstance(nMC, int) and nMC >= 0, "nMC should be larger than or eaqual to 0."

    ys_cut, ts_cut, freqs, tau = wa.prepare_wwz(ys, ts, freqs=freqs, tau=tau)

    wwz_func = wa.get_wwz_func(nproc, method)
    wwa, phase, Neffs, coeff = wwz_func(ys_cut, ts_cut, freqs, tau, Neff=Neff, c=c, nproc=nproc, detrend=detrend)

    # Monte-Carlo simulations of AR1 process
    nt = np.size(tau)
    nf = np.size(freqs)

    wwa_red = np.ndarray(shape=(nMC, nt, nf))
    AR1_q = np.ndarray(shape=(nt, nf))

    if nMC >= 1:
        tauest = wa.tau_estimation(ys_cut, ts_cut, detrend=detrend)

        for i in tqdm(range(nMC), desc='Monte-Carlo simulations...'):
            r = wa.ar1_model(ts_cut, tauest)
            wwa_red[i, :, :], _, _, _ = wwz_func(r, ts_cut, freqs, tau, c=c, Neff=Neff, nproc=nproc, detrend=detrend)

        for j in range(nt):
            for k in range(nf):
                AR1_q[j, k] = mquantiles(wwa_red[:, j, k], 0.95)

    else:
        AR1_q = None

    # calculate the cone of influence
    coi = wa.make_coi(tau)

    return wwa, phase, AR1_q, coi, freqs, tau, Neffs, coeff


def wwz_psd(ys, ts, freqs=None, tau=None, c=1e-3, nproc=8, nMC=200,
            detrend='no', Neff=3, anti_alias=False, avgs=2, method='Kirchner_f2py'):
    ''' Return the psd of a timeseires directly using wwz method.

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
        method (str): 'Foster' - the original WWZ method;
                      'Kirchner' - the method Kirchner adapted from Foster;
                      'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py

    Returns:
        psd (array): power spectral density
        freqs (array): vector of frequency
        psd_ar1_q95 (array): the 95% quantile of the psds of AR1 processes

    '''
    wa = WaveletAnalysis()
    ys_cut, ts_cut, freqs, tau = wa.prepare_wwz(ys, ts, freqs=freqs, tau=tau)

    # get wwa but AR1_q is not needed here so set nMC=0
    wwa, _, _, _, freqs, _, Neffs, _ = wwz(ys_cut, ts_cut, freqs=freqs, tau=tau, c=c, nproc=nproc, nMC=0,
                                           detrend=detrend, method=method)

    psd = wa.wwa2psd(wwa, ts_cut, Neffs, freqs=freqs, Neff=Neff, anti_alias=anti_alias, avgs=avgs)

    # Monte-Carlo simulations of AR1 process
    nf = np.size(freqs)

    psd_ar1 = np.ndarray(shape=(nMC, nf))

    if nMC >= 1:
        tauest = wa.tau_estimation(ys_cut, ts_cut, detrend=detrend)

        for i in tqdm(range(nMC), desc='Monte-Carlo simulations...'):
            r = wa.ar1_model(ts_cut, tauest)
            wwa_red, _, _, _, _, _, Neffs_red, _ = wwz(r, ts_cut, freqs=freqs, tau=tau, c=c, nproc=nproc, nMC=0,
                                                       detrend=detrend, method=method)
            psd_ar1[i, :] = wa.wwa2psd(wwa_red, ts_cut, Neffs_red, freqs=freqs, Neff=Neff, anti_alias=anti_alias, avgs=avgs)

        psd_ar1_q95 = mquantiles(psd_ar1, 0.95, axis=0)[0]

    else:
        psd_ar1_q95 = None

    return psd, freqs, psd_ar1_q95


def xwt(ys1, ts1, ys2, ts2,
        tau=None, freqs=None, c=1/(8*np.pi**2), Neff=3, nMC=200, nproc=8, detrend='no', method='Kirchner_f2py'):
    ''' Return the crosse wavelet transform of two time series.

    Args:
        ys1, ys2 (array): the two time series
        ts1, ts2 (array): the time axis of the two time series
        tau (array): the evenly-spaced time points
        freqs (array): vector of frequency
        c (float): the decay constant, the default value 1/(8*np.pi**2) is good for most of the cases
        Neff (int): effective number of points
        nMC (int): the number of Monte-Carlo simulations
        nproc (int): the number of processes for multiprocessing
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted
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
    assert isinstance(nMC, int) and nMC >= 0, "nMC should be larger than or eaqual to 0."

    wwz_func = wa.get_wwz_func(nproc, method)

    ys1_cut, ts1_cut, freqs, tau = wa.prepare_wwz(ys1, ts1, freqs=freqs, tau=tau)
    ys2_cut, ts2_cut, freqs, tau = wa.prepare_wwz(ys2, ts2, freqs=freqs, tau=tau)

    wwa, phase, Neffs, coeff1 = wwz_func(ys1_cut, ts1_cut, freqs, tau, Neff=Neff, c=c, nproc=nproc, detrend=detrend)
    wwa, phase, Neffs, coeff2 = wwz_func(ys2_cut, ts2_cut, freqs, tau, Neff=Neff, c=c, nproc=nproc, detrend=detrend)

    xw_amplitude, xw_phase = wa.cross_wt(coeff1, coeff2, freqs, tau)

    AR1_q = None
    coi = wa.make_coi(tau)

    return xw_amplitude, xw_phase, freqs, tau, AR1_q, coi


def plot_wwa(wwa, freqs, tau, Neff=3, AR1_q=None, coi=None, levels=None, tick_range=None,
             yticks=None, ylim=None, xticks=None, xlabels=None, figsize=[20, 8], clr_map='OrRd',
             cbar_drawedges=False, cone_alpha=0.5, plot_signif=False, signif_style='contour',
             plot_cone=False, ax=None, xlabel='Year', ylabel='Period'):
    """ Plot the wavelet amplitude

    Args:
        wwa (array): the weighted wavelet amplitude.
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
        Neff (int): effective number of points
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

    Returns:
        fig (figure): the 2-D plot of wavelet analysis

    """
    assert isinstance(Neff, int) and Neff >= 1

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

    frac = 0.15
    pad = 0.05
    origin = 'lower'

    if levels is not None:
        plt.contourf(tau, 1/freqs, wwa.T, levels, cmap=clr_map, origin=origin)
    else:
        plt.contourf(tau, 1/freqs, wwa.T, cmap=clr_map, origin=origin)

    plt.colorbar(drawedges=cbar_drawedges, orientation='vertical', fraction=frac, pad=pad,
                 ticks=tick_range)

    plt.yscale('log', nonposy='clip')

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
        signif = wwa / AR1_q
        if signif_style == 'contour':
            plt.contour(tau, 1/freqs, signif.T, [-99, 1], colors='k')
        elif signif_style == 'shade':
            plt.contourf(tau, 1/freqs, signif.T, [-99, 1], colors='k', alpha=0.1)  # significant if not shaded

    if plot_cone:
        assert coi is not None, "Please set values for `coi`!"
        plt.plot(tau, coi, 'k--')
        ax.fill_between(tau, coi, ylim[1], color='white', alpha=cone_alpha)

    ax.set_ylim(ylim)

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


def plot_psd(psd, freqs, lmstyle=None, linewidth=None, xticks=None, xlim=None, ylim=None,
             figsize=[20, 8], label='PSD', plot_ar1=False, psd_ar1_q95=None,
             psd_ar1_color=sns.xkcd_rgb["pale red"], ax=None,
             xlabel='Period', ylabel='Spectral Density'):
    """ Plot the wavelet amplitude

    Args:
        psd (array): power spectral density
        freqs (array): vector of frequency
        xticks (list): ticks on x-axis
        xlim (list): limits for x-axis
        figsize (list): the size for the figure
        ax: Return as axis instead of figure (useful to integrate plot into a subplot)
        xlabel (str): The x-axis label
        ylabel (str): The y-axis label

    Returns:
        fig (figure): the 2-D plot of wavelet analysis

    """
    sns.set(style="ticks", font_scale=2)

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    if lmstyle is not None:
        plt.plot(1/freqs, psd, lmstyle, linewidth=linewidth, label=label)
        if plot_ar1:
            assert psd_ar1_q95 is not None, "psd_ar1_q95 is required!"
            plt.plot(1/freqs, psd_ar1_q95, lmstyle, linewidth=linewidth,  label='AR1 95%', color=psd_ar1_color)
    else:
        plt.plot(1/freqs, psd, linewidth=linewidth,  label=label)
        if plot_ar1:
            assert psd_ar1_q95 is not None, "psd_ar1_q95 is required!"
            plt.plot(1/freqs, psd_ar1_q95, linewidth=linewidth,  label='AR1 95%', color=psd_ar1_color)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.xscale('log', nonposy='clip')
    plt.yscale('log', nonposy='clip')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    if xticks is not None:
        plt.xticks(xticks)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
    else:
        ax.set_aspect('equal')

    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid()

    return ax


# some alias
wa = WaveletAnalysis()
beta_estimation = wa.beta_estimation
tau_estimation = wa.tau_estimation
