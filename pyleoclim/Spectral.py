#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:23:06 2017

@author: deborahkhider

Spectral module for pyleoclim
"""

import numpy as np
import statsmodels.api as sm

import scipy.optimize as optimize
import scipy.signal as signal
from scipy.stats import mstats

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

import warnings

from pyleoclim import Timeseries


def ar1_fit(ys, ts=None, detrend=False):
    ''' Returns the lag-1 autocorrelation from ar1 fit OR persistence from tauest.

    Args:
        ys (array): the time series
        ts (array): the time axis of that series
        detrend (bool): whether to detrend the time series or not

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


def ar1_sim(ys, n, p, ts=None, detrend=False):
    ''' Produce p realizations of an AR1 process of length n with lag-1 autocorrelation g calculated from `ys` and `ts`

    Args:
        ys (array): a time series
        n, p (int): dimensions as n rows by p columns
        ts (array): the time axis of that series
        detrend (bool): whether to detrend the time series or not

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
        wa = WaveletAnalysis()
        for i in np.arange(p):
            red[:, i] = wa.ar1_model(ts, tau_est, n=n)

    if p == 1:
        red = red[:, 0]

    return red


class WaveletAnalysis(object):
    '''Performing wavelet analysis
    @author: fzhu
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

    def ar1_fit_evenly(self, ys, detrend=False):
        ''' Returns the lag-1 autocorrelation from ar1 fit.

        Args:
            ys (array): vector of (flaot) numbers as a time series
            detrend (bool): whether to detrend the time series or not

        Returns:
            g (float): lag-1 autocorrelation coefficient

        '''
        pd_ys = self.preprocess(ys, detrend=detrend)
        ar1_mod = sm.tsa.AR(pd_ys, missing='drop').fit(maxlag=1)
        g = ar1_mod.params[0]

        return g

    def preprocess(self, ys, detrend=False):
        ''' Return the processed time series using (detrend and) standardization.

        Args:
            ys (array): a time series
            detrend (bool): whether to detrend the time series or not

        Returns:
            res (array): the processed time series

        '''

        if detrend:
            ys_d = signal.detrend(ys)
        else:
            ys_d = ys

        res, _, _ = Timeseries.standardize(ys_d)

        return res

    def tau_estimation(self, ys, ts, detrend=False):
        ''' Return the estimated persistence of a givenevenly/unevenly spaced time series.

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            detrend (bool): whether detrend the time series or not

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

    def wwz_basic(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend=False):
        ''' Return the weighted wavelet amplitude (WWA).

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): fake argument, just for convenience
            detrend (bool): whether to detrend the time series or not

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

        wwa = np.ndarray(shape=(nt, nf))
        phase = np.ndarray(shape=(nt, nf))
        Neffs = np.ndarray(shape=(nt, nf))
        coeff = np.ndarray(shape=(nt, nf), dtype=complex)

        S = np.zeros(shape=(3, 3))

        for k in range(nf):
            for j in range(nt):
                dz = omega[k] * (ts - tau[j])
                weights = np.exp(-c*dz**2)

                sum_w = np.sum(weights)
                Neffs[j, k] = sum_w**2 / np.sum(weights**2)  # local number of effective dof

                if Neffs[j, k] <= Neff:
                    wwa[j, k] = np.nan  # the amplitude cannot be estimated reliably when Neff_loc <= Neff
                    phase[j, k] = np.nan
                    coeff[j, k] = np.nan + np.nan*1j
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

                    ywave_2 = S_inv[1, 0]*weighted_phi1 + S_inv[1, 1]*weighted_phi2 + S_inv[1, 2]*weighted_phi3
                    ywave_3 = S_inv[2, 0]*weighted_phi1 + S_inv[2, 1]*weighted_phi2 + S_inv[2, 2]*weighted_phi3

                    wwa[j, k] = np.sqrt(ywave_2**2 + ywave_3**2)
                    phase[j, k] = np.arctan2(ywave_2, ywave_3)
                    coeff[j, k] = ywave_2 + ywave_3*1j

        return wwa, phase, Neffs, coeff

    def wwz_nproc(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=8,  detrend=False):
        ''' Return the weighted wavelet amplitude (WWA).

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): the number of processes for multiprocessing
            detrend (bool): whether to detrend the time series or not

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

        wwa = np.ndarray(shape=(nt, nf))
        phase = np.ndarray(shape=(nt, nf))
        Neffs = np.ndarray(shape=(nt, nf))
        coeff = np.ndarray(shape=(nt, nf), dtype=complex)

        def wwa_1g(tau, omega):
            dz = omega * (ts - tau)
            weights = np.exp(-c*dz**2)

            sum_w = np.sum(weights)
            Neff_loc = sum_w**2 / np.sum(weights**2)

            S = np.zeros(shape=(3, 3))

            if Neff_loc <= Neff:
                amplitude_1g = np.nan  # the amplitude cannot be estimated reliably when Neff_loc <= Neff
                phase_1g = np.nan
                coeff_1g = np.nan + np.nan*1j
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

                ywave_2 = S_inv[1, 0]*weighted_phi1 + S_inv[1, 1]*weighted_phi2 + S_inv[1, 2]*weighted_phi3
                ywave_3 = S_inv[2, 0]*weighted_phi1 + S_inv[2, 1]*weighted_phi2 + S_inv[2, 2]*weighted_phi3

                amplitude_1g = np.sqrt(ywave_2**2 + ywave_3**2)
                phase_1g = np.arctan2(ywave_2, ywave_3)
                coeff_1g = ywave_2 + ywave_3*1j

            return amplitude_1g, phase_1g, Neff_loc, coeff_1g

        tf_mesh = np.meshgrid(tau, omega)
        list_of_grids = list(zip(*(grid.flat for grid in tf_mesh)))
        tau_grids, omega_grids = zip(*list_of_grids)

        with Pool(nproc) as pool:
            res = pool.map(wwa_1g, tau_grids, omega_grids)
            res_array = np.asarray(res)
            wwa = res_array[:, 0].reshape((np.size(omega), np.size(tau))).T
            phase = res_array[:, 1].reshape((np.size(omega), np.size(tau))).T
            Neffs = res_array[:, 2].reshape((np.size(omega), np.size(tau))).T
            coeff = res_array[:, 3].reshape((np.size(omega), np.size(tau))).T

        return wwa, phase, Neffs, coeff

    def kirchner_basic(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend=False):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): fake argument, just for convenience
            detrend (bool): whether to detrend the time series or not

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

        wwa = np.ndarray(shape=(nt, nf))
        phase = np.ndarray(shape=(nt, nf))
        Neffs = np.ndarray(shape=(nt, nf))
        coeff = np.ndarray(shape=(nt, nf), dtype=complex)

        for k in range(nf):
            for j in range(nt):
                dz = omega[k] * (ts - tau[j])
                weights = np.exp(-c*dz**2)

                sum_w = np.sum(weights)
                Neffs[j, k] = sum_w**2 / np.sum(weights**2)  # local number of effective dof

                if Neffs[j, k] <= Neff:
                    wwa[j, k] = np.nan  # the amplitude cannot be estimated reliably when Neff_loc <= Neff
                    phase[j, k] = np.nan
                    coeff[j, k] = np.nan + np.nan*1j
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
                    denominator = cos_cos - cos_one**2 - (sin_sin - sin_one)**2
                    time_shift = np.arctan2(numerator, denominator) / (2*omega[k])  # Eq. (S5)

                    sin_shift = np.sin(omega[k]*(ts - time_shift))
                    cos_shift = np.cos(omega[k]*(ts - time_shift))
                    sin_tau = np.sin(omega[k]*time_shift)
                    cos_tau = np.cos(omega[k]*time_shift)

                    ys_cos_shift = w_prod(pd_ys, cos_shift)
                    ys_sin_shift = w_prod(pd_ys, sin_shift)
                    ys_one = w_prod(pd_ys, one_v)
                    cos_shift_one = w_prod(cos_shift, one_v)
                    sin_shift_one = w_prod(sin_shift, one_v)

                    A = ys_cos_shift-ys_one*cos_shift_one
                    B = ys_sin_shift-ys_one*sin_shift_one

                    a1 = 2*(cos_tau*A - sin_tau*B)  # Eq. (S6)
                    a2 = 2*(sin_tau*A + cos_tau*B)  # Eq. (S7)

                    wwa[j, k] = np.sqrt(a1**2 + a2**2)
                    phase[j, k] = np.arctan2(a1, a2)
                    coeff[j, k] = a1 + a2*1j

        return wwa, phase, Neffs, coeff

    def kirchner_nproc(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend=False):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): the number of processes for multiprocessing
            detrend (bool): whether to detrend the time series or not

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

        wwa = np.ndarray(shape=(nt, nf))
        phase = np.ndarray(shape=(nt, nf))
        Neffs = np.ndarray(shape=(nt, nf))
        coeff = np.ndarray(shape=(nt, nf), dtype=complex)

        def wwa_1g(tau, omega):
            dz = omega * (ts - tau)
            weights = np.exp(-c*dz**2)

            sum_w = np.sum(weights)
            Neff_loc = sum_w**2 / np.sum(weights**2)

            if Neff_loc <= Neff:
                amplitude_1g = np.nan  # the amplitude cannot be estimated reliably when Neff_loc <= Neff
                phase_1g = np.nan
                coeff_1g = np.nan + np.nan*1j
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
                sin_tau = np.sin(omega*time_shift)
                cos_tau = np.cos(omega*time_shift)

                ys_cos_shift = w_prod(pd_ys, cos_shift)
                ys_sin_shift = w_prod(pd_ys, sin_shift)
                ys_one = w_prod(pd_ys, one_v)
                cos_shift_one = w_prod(cos_shift, one_v)
                sin_shift_one = w_prod(sin_shift, one_v)

                A = ys_cos_shift - ys_one*cos_shift_one
                B = ys_sin_shift - ys_one*sin_shift_one

                a1 = 2*(cos_tau*A - sin_tau*B)  # Eq. (S6)
                a2 = 2*(sin_tau*A + cos_tau*B)  # Eq. (S7)

                amplitude_1g = np.sqrt(a1**2 + a2**2)
                phase_1g = np.arctan2(a1, a2)
                coeff_1g = a1 + a2*1j

            return amplitude_1g, phase_1g, Neff_loc, coeff_1g

        tf_mesh = np.meshgrid(tau, omega)
        list_of_grids = list(zip(*(grid.flat for grid in tf_mesh)))
        tau_grids, omega_grids = zip(*list_of_grids)

        with Pool(nproc) as pool:
            res = pool.map(wwa_1g, tau_grids, omega_grids)
            res_array = np.asarray(res)
            wwa = res_array[:, 0].reshape((np.size(omega), np.size(tau))).T
            phase = res_array[:, 1].reshape((np.size(omega), np.size(tau))).T
            Neffs = res_array[:, 2].reshape((np.size(omega), np.size(tau))).T
            coeff = res_array[:, 3].reshape((np.size(omega), np.size(tau))).T

        return wwa, phase, Neffs, coeff

    def kirchner_f2py(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend=False):
        ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points
            c (float): the decay constant
            Neff (int): the threshold of the number of effective degree of freedom
            nproc (int): fake argument, just for convenience
            detrend (bool): whether to detrend the time series or not

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
        wwa, phase, Neffs, coeff_real, coeff_imag = f2py.f2py_wwz.wwa(tau, omega, c, Neff, ts, pd_ys, nproc, nts, nt, nf)

        undef = -99999.
        wwa[wwa == undef] = np.nan
        phase[phase == undef] = np.nan
        Neffs[Neffs < Neff] = 0

        coeff_real[coeff_real == undef] = np.nan
        coeff_imag[coeff_imag == undef] = np.nan
        coeff = coeff_real + coeff_imag*1j

        return wwa, phase, Neffs, coeff
    def make_coi(self, tau, Neff=3):
        ''' Return the cone of influence.

        Args:
            tau (array): the evenly-spaced time points
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

    def wavelet_analysis(self, ys, ts, freqs, tau, c=1/(8*np.pi**2), Neff=3, nMC=200, nproc=8,
                         detrend=False, method='Kirchner_f2py'):
        ''' Return the weighted wavelet amplitude (WWA).

        Args:
            ys (array): a time series
            ts (array): time axis of the time series
            freqs (array): vector of frequency
            tau (array): the evenly-spaced time points
            c: the decay constant
            nMC (int): the number of Monte-Carlo simulations
            nproc (int): the number of processes for multiprocessing
            detrend (bool): whether to detrend the time series or not
            method (str): 'Foster' - the original WWZ method;
                          'Kirchner' - the method Kirchner adapted from Foster;
                          'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py

        Returns:
            wwa (array): the weighted wavelet amplitude.
            AR1_q (array): AR1 simulations
            coi (array): cone of influence
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            coeff (array): the wavelet transform coefficients

        '''
        self.assertPositiveInt(nproc)
        assert isinstance(nMC, int) and nMC >= 0, "nMC should be larger than or eaqual to 0."

        nt = np.size(tau)
        nf = np.size(freqs)

        if method == 'Foster':
            if nproc == 1:
                wwz_func = self.wwz_basic
            else:
                wwz_func = self.wwz_nproc

        elif method == 'Kirchner':
            if nproc == 1:
                wwz_func = self.kirchner_basic
            else:
                wwz_func = self.kirchner_nproc

        else:
            wwz_func = self.kirchner_f2py

        wwa, phase, Neffs, coeff = wwz_func(ys, ts, freqs, tau, Neff=Neff, c=c, nproc=nproc, detrend=detrend)

        wwa_red = np.ndarray(shape=(nMC, nt, nf))
        AR1_q = np.ndarray(shape=(nt, nf))

        tauest = self.tau_estimation(ys, ts, detrend=detrend)

        if nMC >= 1:
            for i in tqdm(range(nMC), desc='Monte-Carlo simulations...'):
                r = self.ar1_model(ts, tauest)
                wwa_red[i, :, :], _, _, _ = wwz_func(r, ts, freqs, tau, c=c, Neff=Neff, nproc=nproc, detrend=detrend)

            for j in range(nt):
                for k in range(nf):
                    AR1_q[j, k] = mstats.mquantiles(wwa_red[:, j, k], 0.95, alphap=0.5, betap=0.5)

        else:
            AR1_q = None

        coi = self.make_coi(tau)

        return wwa, phase, AR1_q, coi, Neffs, coeff

    def wwa2psd_weighted(self, wwa, ts, Neffs, Neff=3):
        ''' Return the power spectral density (PSD) using the weighted wavelet amplitude (WWA).

        Args:
            wwa (array): the weighted wavelet amplitude.
            ts (array): time axis of the time series
            Neffs (array): the matrix of effective number of points in the time-scale coordinates
            Neff (int): the threshold of the number of effective samples

        Returns:
            psd (array): power spectral density

        References:
            Kirchner's C code

        '''
        dt = np.mean(np.diff(ts))

        power = wwa**2 * 0.5 * dt * Neffs
        sum_power = np.sum(power * (Neffs - Neff), axis=0)
        sum_eff = np.sum(Neffs - Neff, axis=0)

        psd = sum_power / sum_eff

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


class AliasFilter(object):
    '''Performing anti-alias filter on a psd
    @author: fzhu
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
Interface for users
'''


def wwz(ys, ts, tau=None, freqs=None, c=1/(8*np.pi**2), Neff=3, nMC=200, nproc=8,
        detrend=False, method='Kirchner_f2py'):
    ''' Return the weighted wavelet amplitude (WWA) with phase, AR1_q, and cone of influence

    Args:
        ys (array): a time series, NaNs will be deleted automatically
        ts (array): the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
        tau (array): the evenly-spaced time points
        freqs (array): vector of frequency
        c (float): the decay constant, the default value 1/(8*np.pi**2) is good for most of the cases
        Neff (int): effective number of points
        nMC (int): the number of Monte-Carlo simulations
        nproc (int): the number of processes for multiprocessing
        detrend (bool): whether to detrend the time series or not
        method (str): 'Foster' - the original WWZ method;
                      'Kirchner' - the method Kirchner adapted from Foster;
                      'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py

    Returns:
        wwa (array): the weighted wavelet amplitude.
        AR1_q (array): AR1 simulations
        coi (array): cone of influence
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points
        Neffs (array): the matrix of effective number of points in the time-scale coordinates
        coeff (array): the wavelet transform coefficents

    '''
    wa = WaveletAnalysis()

    # delete NaNs if there is any
    ys_tmp = np.copy(ys)
    ys = ys[~np.isnan(ys_tmp)]
    ts = ts[~np.isnan(ys_tmp)]

    if tau is None:
        med_res = np.size(ts) // np.median(np.diff(ts))
        tau = np.linspace(np.min(ts), np.max(ts), np.max([np.size(ts)//10, 50, med_res]))

    if freqs is None:
        freqs = wa.make_freq_vector(ts)

    wwa, phase, AR1_q, coi, Neffs, coeff = wa.wavelet_analysis(ys, ts, freqs, tau, c=c, Neff=Neff, nMC=nMC, nproc=nproc,
                                                               detrend=detrend, method=method)

    return wwa, phase, AR1_q, coi, freqs, tau, Neffs, coeff


def wwa2psd(wwa, ts, freqs, tau, Neffs, Neff=3, anti_alias=False, avgs=2):
    """ Return the power spectral density (PSD) using the weighted wavelet amplitude (WWA).

    Args:
        wwa (array): the weighted wavelet amplitude.
        ts (array): the time points
        freqs (array): vector of frequency from wwz
        Neffs (array):  the matrix of effective number of points in the time-scale coordinates obtained from wwz from wwz
        Neff (int): the threshold of the number of effective samples
        tau (array): the evenly-spaced time points
        anti_alias (bool): whether to apply anti-alias filter
        avgs (int): flag for whether spectrum is derived from instantaneous point measurements (avgs<>1)
                    OR from measurements averaged over each sampling interval (avgs==1)

    Returns:
        psd (array): power spectral density

    """
    wa = WaveletAnalysis()
    af = AliasFilter()
    psd = wa.wwa2psd_weighted(wwa, ts, Neffs=Neffs, Neff=Neff)

    if anti_alias:
        dt = np.mean(np.diff(ts))
        f_sampling = 1/dt
        alpha, filtered_pwr, model_pwer, aliased_pwr = af.alias_filter(
            freqs, psd, f_sampling, f_sampling*1e3, np.min(freqs), avgs)

        psd = np.copy(filtered_pwr)

    return psd


def wwz_psd(ys, ts, freqs=None, tau=None, c=1e-3, nproc=8, nMC=0,
            detrend=False, Neff=3, anti_alias=False, avgs=2, method='Kirchner_f2py'):
    ''' Return the psd of a timeseires directly using wwz method.

    Args:
        ys (array): a time series, NaNs will be deleted automatically
        ts (array): the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points
        c (float): the decay constant, the default value 1e-3 is good for most of the cases
        nproc (int): the number of processes for multiprocessing
        nMC (int): the number of Monte-Carlo simulations
        detrend (bool): whether to detrend the time series or not
        method (str): 'Foster' - the original WWZ method;
                      'Kirchner' - the method Kirchner adapted from Foster;
                      'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py

    Returns:
        psd (array): power spectral density
        freqs (array): vector of frequency

    '''
    # delete NaNs if there is any
    ys_tmp = np.copy(ys)
    ys = ys[~np.isnan(ys_tmp)]
    ts = ts[~np.isnan(ys_tmp)]

    if tau is None:
        med_res = np.size(ts) // np.median(np.diff(ts))
        tau = np.linspace(np.min(ts), np.max(ts), np.max([np.size(ts)//10, 50, med_res]))

    wwa, phase, AR1_q, coi, freqs, tau, Neffs, coeff = wwz(ys, ts, tau, freqs, c=c, nproc=nproc, nMC=nMC,
                                                           detrend=detrend, method=method)

    psd = wwa2psd(wwa, ts, freqs, tau, Neffs, Neff=Neff, anti_alias=anti_alias, avgs=avgs)

    return psd, freqs


def plot_wwa(wwa, freqs, tau, Neff=3, AR1_q=None, coi=None, levels=None, tick_range=None,
             yticks=None, ylim=None, xticks=None, xlabels=None, figsize=[20, 8], clr_map='OrRd',
             cbar_drawedges=False, cone_alpha=0.5, plot_signif=False, signif_style='contour',
             plot_cone=False,
             ):
    """ Plot the wavelet amplitude

    Args:
        wwa (array): the weighted wavelet amplitude.
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points
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

    Returns:
        fig (figure): the 2-D plot of wavelet analysis

    """
    assert isinstance(Neff, int) and Neff >= 1

    sns.set(style="ticks", font_scale=2)
    fig, ax = plt.subplots(figsize=figsize)

    q95 = mstats.mquantiles(wwa, 0.95, alphap=0.5, betap=0.5)

    if levels is None:
        if np.nanmax(wwa) > 2*q95:
            warnings.warn("There are outliers in the input amplitudes, " +
                          "and the `levels` have been set so that the outpliers will be ignored! " +
                          "One might want to use `Spectral.plot_wwadist(wwa)` to plot the distribution of " +
                          "the amplitudes with the 95% quantile line to check if the levels are appropriate.", stacklevel=2)

            max_level = np.round(2*q95, decimals=1)
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

    plt.xlabel('Year (CE)')
    plt.ylabel('Period (years)')

    if plot_signif:
        assert AR1_q is not None, "Please set values for `AR1_q`!"
        signif = wwa / AR1_q
        if signif_style == 'countour':
            plt.contour(tau, 1/freqs, signif.T, [-99, 1], linewidth=3)
        elif signif_style == 'shade':
            plt.contourf(tau, 1/freqs, signif.T, [-99, 1], colors='k', alpha=0.1)

    if plot_cone:
        assert coi is not None, "Please set values for `coi`!"
        plt.plot(tau, coi, 'k--')
        ax.fill_between(tau, coi, ylim[1], color='white', alpha=cone_alpha)

    return fig


def plot_wwadist(wwa):
    ''' Plot the distribution of wwa with the 95% quantile line.

    Args:
        wwa (array): the weighted wavelet amplitude.

    Returns:
        fig (figure): the 2-D plot of wavelet analysis

    '''
    sns.set(style="darkgrid", font_scale=2)
    plt.subplots(figsize=[20, 4])

    q95 = mstats.mquantiles(wwa, 0.95, alphap=0.5, betap=0.5)
    fig = sns.distplot(np.nan_to_num(wwa.flat))
    fig.axvline(x=q95, ymin=0, ymax=0.5, linewidth=2, linestyle='-')

    return fig


def plot_psd(psd, freqs, lmstyle=None, linewidth=None, xticks=None, xlim=None, ylim=None,
             figsize=[20, 8], label=None):
    """ Plot the wavelet amplitude

    Args:
        psd (array): power spectral density
        freqs (array): vector of frequency
        xticks (list): ticks on x-axis
        xlim (list): limitations for x-axis
        figsize (list): the size for the figure

    Returns:
        fig (figure): the 2-D plot of wavelet analysis

    """
    sns.set(style="ticks", font_scale=2)
    fig, ax = plt.subplots(figsize=figsize)

    if lmstyle is not None:
        plt.plot(1/freqs, psd, lmstyle, linewidth=linewidth, label=label)
    else:
        plt.plot(1/freqs, psd, linewidth=linewidth,  label=label)

    plt.ylabel('Spectral Density')
    plt.xlabel('Period (years)')

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

    plt.gca().invert_xaxis()
    plt.grid()

    return fig


wa = WaveletAnalysis()
beta_estimation = wa.beta_estimation
tau_estimation = wa.tau_estimation
