#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 08:14:31 2020

@author: deborahkhider

Functions concerning wavelet analysis
"""


__all__ = [
    #'cwt',
    'wwz',
    'xwc',
]

import numpy as np
import statsmodels.api as sm
from scipy import signal
from pathos.multiprocessing import ProcessingPool as Pool
import numba as nb
from numba.core.errors import NumbaPerformanceWarning
import warnings
import collections
import scipy.fftpack as fft
from scipy import optimize
from scipy.optimize import fminbound
from scipy.special._ufuncs import gamma, gammainc

#from .tsmodel import ar1_sim
from .tsutils import preprocess
from .tsbase import (
    clean_ts,
    is_evenly_spaced,
)

from .filter import ts_pad

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

#---------------
#Wrapper functions
#---------------

#----------------
#Main Functions
#----------------

class AliasFilter(object):
    '''Performing anti-alias filter on a psd

    experimental: Use at your own risk

    @author: fzhu
    '''

    def alias_filter(self, freq, pwr, fs, fc, f_limit, avgs):
        ''' anti_alias filter

        Parameters
        ----------

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

        Kirchner, J. W. Aliasing in 1/f(alpha) noise spectra: origins, consequences, and remedies.
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

# def cwt(ys,ts,scales,wavelet='morl',sampling_period=1.0,method='conv',axis=-1):
#     '''Continous wavelet transform for evenly spaced data

#     pywavelet documentation: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html

#     Parameters
#     ----------
#     ys : array
#         signal
#     ts : array
#         time
#     scales : array (float)
#         different wavelet scales to use
#     wavelet : str
#         types of wavelet options in function documentation link. The default is 'morl' for a morlet wavelet.
#     sampling_period : float, optional
#         sampling period for frequencies output. The default is 1.0.
#     method : str, optional
#         cwt computation  method. 'conv','fft'. or 'auto' The default is 'conv'.
#     axis : int, optional
#         axis over which to compute cwt. The default is -1, the last axis.


#     Returns
#     -------
#     res : dictionary
#         'freq' - array(float)
#             frequencies
#         'time' - array(float)
#         'amplitude' - array(float)
#         'coi' - array(float)
#             cone of inference

#     '''
#     coeff,freq=pywt.cwt(data=ys,scales=scales,wavelet=wavelet,sampling_period=sampling_period,method=method,axis=axis)
#     amplitude=abs(coeff).T
#     if wavelet=='morl' or wavelet[:4]=='cmor':
#         coi=make_coi(tau=ts,Neff=6)
#     else:
#         coi=make_coi(tau=ts)
#     Results = collections.namedtuple('Results', ['amplitude','coi', 'freq', 'time', 'coeff'])
#     res = Results(amplitude=amplitude, coi=coi, freq=freq, time=ts, coeff=coeff)
#     return res

def assertPositiveInt(*args):
    ''' Assert that the arguments are all positive integers.

    Parameters
    ----------

    args

    '''
    for arg in args:
        assert isinstance(arg, int) and arg >= 1

def wwz_basic(ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend=False, sg_kwargs=None,
              gaussianize=False, standardize=False):
    ''' Return the weighted wavelet amplitude (WWA).

    The Weighted wavelet Z-transform (WWZ) is based on Morlet wavelet estimation, using
    least squares minimization to suppress the energy leakage caused by the data gaps.
    WWZ does not rely on interpolation or detrending, and is appropriate for unevenly-spaced datasets.
    In particular, we use the variant of Kirchner & Neal (2013), in which basis rotations mitigate the
    numerical instability that occurs in pathological cases with the original algorithm (Foster, 1996).
    The WWZ method has one adjustable parameter, a decay constant `c` that balances the time and frequency
    resolutions of the analysis. This application uses the larger value (8π2)−1, justified elsewhere
    (Witt & Schumann, 2005).

    No multiprocessing is applied by Default.

    Parameters
    ----------

    ys : array
        a time series
    ts : array
        time axis of the time series
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points, namely the time shift for wavelet analysis
    c : float
        the decay constant that determines the analytical resolution of frequency for analysis, the smaller the higher resolution;
        the default value 1/(8*np.pi**2) is good for most of the wavelet analysis cases
    Neff : int
        the threshold of the number of effective degrees of freedom
    nproc :int
        fake argument, just for convenience
    detrend : string
        None - the original time series is assumed to have no trend;
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
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

    - Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
    - Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
    Nonlinear Processes in Geophysics 12, 345–352 (2005).
    - Kirchner, J. W. and Neal, C. (2013). Universal fractal scaling in stream chemistry and its implications for solute transport and water quality trend detection. Proc Natl Acad Sci USA 110:12213–12218.

    See also
    --------

    pyleoclim.utils.wavelet.wwz_nproc : Returns the weighted wavelet amplitude using the original method from Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_basic : Return the weighted wavelet amplitude (WWA) modified by Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.kirchner_nproc : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_numba : Return the weighted wavelet amplitude (WWA) modified by Kirchner using Numba package.

    pyleoclim.utils.wavelet.kirchner_f2py : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Uses Fortran. Fastest method but requires a compiler.

    pyleoclim.utils.filter.savitzky_golay : Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    '''
    assert nproc == 1, "wwz_basic() only supports nproc=1"
    assertPositiveInt(Neff)

    nt = np.size(tau)
    nf = np.size(freq)

    pd_ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs, gaussianize=gaussianize, standardize=standardize)

    omega = make_omega(ts, freq)

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

def wwz_nproc(ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend=False, sg_kwargs=None,
              gaussianize=False, standardize=False):
    ''' Return the weighted wavelet amplitude (WWA).

    Original method from Foster (1996). Supports multiprocessing.

    Parameters
    ----------

    ys : array
        a time series
    ts : array
        time axis of the time series
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points, namely the time shift for wavelet analysis
    c : float
        the decay constant that determines the analytical resolution of frequency for analysis, the smaller the higher resolution;
        the default value 1/(8*np.pi**2) is good for most of the wavelet analysis cases
    Neff : int
        the threshold of the number of effective degrees of freedom
    nproc : int
        the number of processes for multiprocessing
    detrend : string
        None - the original time series is assumed to have no trend;
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
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

    See also
    --------

    pyleoclim.utils.wavelet.wwz_basic : Returns the weighted wavelet amplitude using the original method from Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.kirchner_basic : Return the weighted wavelet amplitude (WWA) modified by Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.kirchner_nproc : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_numba : Return the weighted wavelet amplitude (WWA) modified by Kirchner using Numba package.

    pyleoclim.utils.wavelet.kirchner_f2py : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Uses Fortran. Fastest method but requires a compiler.

    pyleoclim.utils.filter.savitzky_golay : Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    '''
    assert nproc >= 2, "wwz_nproc() should use nproc >= 2, if want serial run, please use wwz_basic()"
    assertPositiveInt(Neff)

    nt = np.size(tau)
    nf = np.size(freq)

    pd_ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs, gaussianize=gaussianize, standardize=standardize)

    omega = make_omega(ts, freq)

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

def kirchner_basic(ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, nproc=1, detrend=False, sg_kwargs=None,
                   gaussianize=False, standardize=False):
    ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

    Method modified by Kirchner. No multiprocessing.

    Parameters
    ----------

    ys : array
        a time series
    ts : array
        time axis of the time series
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points, namely the time shift for wavelet analysis
    c : float
        the decay constant that determines the analytical resolution of frequency for analysis, the smaller the higher resolution;
        the default value 1/(8*np.pi**2) is good for most of the wavelet analysis cases
    Neff : int
        the threshold of the number of effective degrees of freedom
    nproc : int
        fake argument for convenience, for parameter consistency between functions, does not need to be specified
    detrend : string
        None - the original time series is assumed to have no trend;
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
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

    - Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
    - Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores.
    Nonlinear Processes in Geophysics 12, 345–352 (2005).

    See also
    --------

    pyleoclim.utils.wavelet.wwz_basic : Returns the weighted wavelet amplitude using the original method from Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.wwz_nproc : Returns the weighted wavelet amplitude using the original method from Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_nproc : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_numba : Return the weighted wavelet amplitude (WWA) modified by Kirchner using Numba package.

    pyleoclim.utils.wavelet.kirchner_f2py : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Uses Fortran. Fastest method but requires a compiler.

    pyleoclim.utils.filter.savitzky_golay : Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    '''
    assert nproc == 1, "wwz_basic() only supports nproc=1"
    assertPositiveInt(Neff)

    nt = np.size(tau)
    nts = np.size(ts)
    nf = np.size(freq)

    pd_ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs, gaussianize=gaussianize, standardize=standardize)

    omega = make_omega(ts, freq)

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

def kirchner_nproc(ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend=False, sg_kwargs=None,
                   gaussianize=False, standardize=False):
    ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

    Method modified by kirchner. Supports multiprocessing.

    Parameters
    ----------

    ys : array
        a time series
    ts : array
        time axis of the time series
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points, namely the time shift for wavelet analysis
    c : float
        the decay constant that determines the analytical resolution of frequency for analysis, the smaller the higher resolution;
        the default value 1/(8*np.pi**2) is good for most of the wavelet analysis cases
    Neff : int
        the threshold of the number of effective degrees of freedom
    nproc : int
        the number of processes for multiprocessing
    detrend : string
        None - the original time series is assumed to have no trend;
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
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

    See also
    --------

    pyleoclim.utils.wavelet.wwz_basic : Returns the weighted wavelet amplitude using the original method from Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.wwz_nproc : Returns the weighted wavelet amplitude using the original method from Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_basic : Return the weighted wavelet amplitude (WWA) modified by Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.kirchner_numba : Return the weighted wavelet amplitude (WWA) modified by Kirchner using Numba package.

    pyleoclim.utils.wavelet.kirchner_f2py : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Uses Fortran. Fastest method but requires a compiler.

    pyleoclim.utils.filter.savitzky_golay : Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    '''
    assert nproc >= 2, "wwz_nproc() should use nproc >= 2, if want serial run, please use wwz_basic()"
    assertPositiveInt(Neff)

    nt = np.size(tau)
    nts = np.size(ts)
    nf = np.size(freq)

    pd_ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs, gaussianize=gaussianize, standardize=standardize)

    omega = make_omega(ts, freq)

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

def kirchner_numba(ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, detrend=False, sg_kwargs=None,
                   gaussianize=False, standardize=False, nproc=1):
    ''' Return the weighted wavelet amplitude (WWA) modified by Kirchner.

    Using numba.

    Parameters
    ----------

    ys : array
        a time series
    ts : array
        time axis of the time series
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points, namely the time shift for wavelet analysis
    c : float
        the decay constant that determines the analytical resolution of frequency for analysis, the smaller the higher resolution;
        the default value 1/(8*np.pi**2) is good for most of the wavelet analysis cases
    Neff : int
        the threshold of the number of effective degrees of freedom
    nproc : int
        fake argument, just for convenience
    detrend : string
        None - the original time series is assumed to have no trend;
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
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

    See also
    --------

    pyleoclim.utils.wavelet.wwz_basic : Returns the weighted wavelet amplitude using the original method from Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.wwz_nproc : Returns the weighted wavelet amplitude using the original method from Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_basic : Return the weighted wavelet amplitude (WWA) modified by Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.kirchner_nproc : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_f2py : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Uses Fortran. Fastest method but requires a compiler.

    pyleoclim.utils.filter.savitzky_golay : Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    '''
    assertPositiveInt(Neff)
    nt = np.size(tau)
    nts = np.size(ts)
    nf = np.size(freq)

    pd_ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs, gaussianize=gaussianize, standardize=standardize)

    omega = make_omega(ts, freq)

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

def kirchner_f2py(ys, ts, freq, tau, c=1/(8*np.pi**2), Neff=3, nproc=8, detrend=False, sg_kwargs=None,
                  gaussianize=False, standardize=False):
    ''' Returns the weighted wavelet amplitude (WWA) modified by Kirchner.

    Fastest method. Calls Fortran libraries.

    Parameters
    ----------

    ys : array
        a time series
    ts : array
        time axis of the time series
    freq : array
        vector of frequency
    tau : array
        the evenly-spaced time points, namely the time shift for wavelet analysis
    c : float
        the decay constant that determines the analytical resolution of frequency for analysis, the smaller the higher resolution;
        the default value 1/(8*np.pi**2) is good for most of the wavelet analysis cases
    Neff : int
        the threshold of the number of effective degrees of freedom
    nproc : int
        fake argument, just for convenience
    detrend : string
        None - the original time series is assumed to have no trend;
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
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

    See also
    --------

    pyleoclim.utils.wavelet.wwz_basic : Returns the weighted wavelet amplitude using the original method from Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.wwz_nproc : Returns the weighted wavelet amplitude using the original method from Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_basic : Return the weighted wavelet amplitude (WWA) modified by Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.kirchner_nproc : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_numba : Return the weighted wavelet amplitude (WWA) modified by Kirchner using Numba package.

    pyleoclim.utils.filter.savitzky_golay : Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    '''
    from . import f2py_wwz as f2py
    assertPositiveInt(Neff, nproc)

    nt = np.size(tau)
    nts = np.size(ts)
    nf = np.size(freq)

    pd_ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs, gaussianize=gaussianize, standardize=standardize)

    omega = make_omega(ts, freq)

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

def make_coi(tau, Neff=3):
    ''' Return the cone of influence.

    Parameters
    ----------

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

def make_omega(ts, freq):
    ''' Return the angular frequency based on the time axis and given frequency vector

    Parameters
    ----------

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

def wwa2psd(wwa, ts, Neffs, freq=None, Neff=3, anti_alias=False, avgs=2):
    """ Return the power spectral density (PSD) using the weighted wavelet amplitude (WWA).

    Parameters
    ----------

    wwa : array
        the weighted wavelet amplitude.
    ts : array
        the time points, should be pre-truncated so that the span is exactly what is used for wwz
    Neffs : array
        the matrix of effective number of points in the time-scale coordinates obtained from wwz
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

def wwz(ys, ts, tau=None, ntau=None, freq=None, freq_method='log', freq_kwargs={}, c=1/(8*np.pi**2), Neff=3, Neff_coi=3,
        nMC=200, nproc=8, detrend=False, sg_kwargs=None,
        gaussianize=False, standardize=False, method='Kirchner_numba', len_bd=0,
        bc_mode='reflect', reflect_type='odd'):
    ''' Weighted wavelet amplitude (WWA) for unevenly-spaced data

    Parameters
    ----------

    ys : array
        a time series, NaNs will be deleted automatically
    ts : array
        the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
    tau : array
        the evenly-spaced time vector for the analysis, namely the time shift for wavelet analysis
    freq : array
        vector of frequency
    freq_method : str
        Method to generate the frequency vector if not set directly. The following options are avialable:

        - 'log' (default)
        - 'lomb_scargle'
        - 'welch'
        - 'scale'
        - 'nfft'
        See :func:`pyleoclim.utils.wavelet.make_freq_vector()` for details

    freq_kwargs : str
        used when freq=None for certain methods
    c : float
        the decay constant that determines the analytical resolution of frequency for analysis, the smaller the higher resolution;
        the default value 1/(8*np.pi**2) is good for most of the wavelet analysis cases
    Neff : int
        effective number of points
    nMC : int
        the number of Monte-Carlo simulations
    nproc : int
        the number of processes for multiprocessing

    detrend : string, {None, 'linear', 'constant', 'savitzy-golay'}
        available methods for detrending, including

        - None: the original time series is assumed to have no trend;
        - 'linear': a linear least-squares fit to `ys` is subtracted;
        - 'constant': the mean of `ys` is subtracted
        - 'savitzy-golay': ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series

    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. See :func:`pyleoclim.utils.filter.savitzky_golay()` for details.

    method : string, {'Foster', 'Kirchner', 'Kirchner_f2py', 'Kirchner_numba'}
        available specific implementation of WWZ, including

        - 'Foster': the original WWZ method;
        - 'Kirchner': the method Kirchner adapted from Foster;
        - 'Kirchner_f2py': the method Kirchner adapted from Foster, implemented with f2py for acceleration;
        - 'Kirchner_numba': the method Kirchner adapted from Foster, implemented with Numba for acceleration (default);

    len_bd : int
        the number of the ghost grids want to creat on each boundary

    bc_mode : string, {'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect' , 'symmetric', 'wrap'}
        For more details, see np.lib.pad()

    reflect_type : string, optional, {‘even’, ‘odd’}
         Used in ‘reflect’, and ‘symmetric’. The ‘even’ style is the default with an unaltered reflection around the edge value.
         For the ‘odd’ style, the extented part of the array is created by subtracting the reflected values from two times the edge value.
         For more details, see np.lib.pad()

    Returns
    -------

    res : namedtuple
        a namedtuple that includes below items

        wwa : array
            the weighted wavelet amplitude.

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

    See also
    --------

    pyleoclim.utils.wavelet.wwz_basic : Returns the weighted wavelet amplitude using the original method from Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.wwz_nproc : Returns the weighted wavelet amplitude using the original method from Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_basic : Return the weighted wavelet amplitude (WWA) modified by Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.kirchner_nproc : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_numba : Return the weighted wavelet amplitude (WWA) modified by Kirchner using Numba package.

    pyleoclim.utils.wavelet.kirchner_f2py : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Uses Fortran. Fastest method but requires a compiler.

    pyleoclim.utils.filter.savitzky_golay : Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    pyleoclim.utils.wavelet.make_freq_vector : Make frequency vector

    Examples
    --------

    We perform an ideal test below.
    We use a sine wave with a period of 50 yrs as the signal for test.
    Then performing wavelet analysis should return an energy band around period of 50 yrs in the time-period scalogram domain.

    .. ipython:: python
        :okwarning:

        from pyleoclim import utils
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
        import numpy as np

        # Create a signal
        time = np.arange(2001)
        f = 1/50  # the period is then 1/f = 50
        signal = np.cos(2*np.pi*f*time)

        # Wavelet Analysis
        res = utils.wwz(signal, time)

        # Visualization
        fig, ax = plt.subplots()
        contourf_args = {'cmap': 'magma', 'origin': 'lower', 'levels': 11}
        cbar_args = {'drawedges': False, 'orientation': 'vertical', 'fraction': 0.15, 'pad': 0.05}
        cont = ax.contourf(res.time, 1/res.freq, res.amplitude.T, **contourf_args)
        ax.plot(res.time, res.coi, 'k--')  # plot the cone of influence
        ax.set_yscale('log')
        ax.set_yticks([2, 5, 10, 20, 50, 100, 200, 500, 1000])
        ax.set_ylim([2, 1000])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.set_xlabel('Time (yr)')
        ax.set_ylabel('Period (yrs)')
        cb = plt.colorbar(cont, **cbar_args)
        @savefig wwa_wwz.png
        plt.show()

    '''
    assert isinstance(nMC, int) and nMC >= 0, "nMC should be larger than or equal to 0."

    ys_cut, ts_cut, freq, tau = prepare_wwz(
        ys, ts, freq=freq, freq_method=freq_method, freq_kwargs=freq_kwargs,
        tau=tau, len_bd=len_bd,
        bc_mode=bc_mode, reflect_type=reflect_type
    )

    wwz_func = get_wwz_func(nproc, method)
    wwa, phase, Neffs, coeff = wwz_func(ys_cut, ts_cut, freq, tau, Neff=Neff, c=c, nproc=nproc,
                                        detrend=detrend, sg_kwargs=sg_kwargs,
                                        gaussianize=gaussianize, standardize=standardize)

    # Monte-Carlo simulations of AR1 process
    nt = np.size(tau)
    nf = np.size(freq)

    #  wwa_red = np.ndarray(shape=(nMC, nt, nf))
    #  AR1_q = np.ndarray(shape=(nt, nf))

    #  if nMC >= 1:
        #  for i in tqdm(range(nMC), desc='Monte-Carlo simulations'):
            #  r = ar1_sim(ys_cut, np.size(ts_cut), 1, ts=ts_cut)
            #  wwa_red[i, :, :], _, _, _ = wwz_func(r, ts_cut, freq, tau, c=c, Neff=Neff, nproc=nproc,
                                                 #  detrend=detrend, sg_kwargs=sg_kwargs,
                                                 #  gaussianize=gaussianize, standardize=standardize)

        #  for j in range(nt):
            #  for k in range(nf):
                #  AR1_q[j, k] = mquantiles(wwa_red[:, j, k], 0.95)

    #  else:
        #  AR1_q = None
    # AR1_q = None

    # calculate the cone of influence
    coi = make_coi(tau, Neff=Neff_coi)

    # Results = collections.namedtuple('Results', ['amplitude', 'phase', 'AR1_q', 'coi', 'freq', 'time', 'Neffs', 'coeff'])
    # res = Results(amplitude=wwa, phase=phase, AR1_q=AR1_q, coi=coi, freq=freq, time=tau, Neffs=Neffs, coeff=coeff)
    Results = collections.namedtuple('Results', ['amplitude', 'phase', 'coi', 'freq', 'time', 'Neffs', 'coeff'])
    res = Results(amplitude=wwa, phase=phase, coi=coi, freq=freq, time=tau, Neffs=Neffs, coeff=coeff)

    return res

def xwc(ys1, ts1, ys2, ts2, smooth_factor=0.25,
        tau=None, freq=None, freq_method='log', freq_kwargs=None,
        c=1/(8*np.pi**2), Neff=3, nproc=8, detrend=False, sg_kwargs=None,
        nMC=200,
        gaussianize=False, standardize=False, method='Kirchner_numba',
        verbose=False):
    ''' Return the cross-wavelet coherence of two time series.

    Parameters
    ----------

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
        the decay constant that determines the analytical resolution of frequency for analysis, the smaller the higher resolution;
        the default value 1/(8*np.pi**2) is good for most of the wavelet analysis cases
    Neff : int
        effective number of points
    nproc : int
        the number of processes for multiprocessing
    nMC : int
        the number of Monte-Carlo simulations
    detrend : string
        - None: the original time series is assumed to have no trend;
        - 'linear': a linear least-squares fit to `ys` is subtracted;
        - 'constant': the mean of `ys` is subtracted
        - 'savitzy-golay': ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
        Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
    gaussianize : bool
        If True, gaussianizes the timeseries
    standardize : bool
        If True, standardizes the timeseries
    method : string
        - 'Foster': the original WWZ method;
        - 'Kirchner': the method Kirchner adapted from Foster;
        - 'Kirchner_f2py': the method Kirchner adapted from Foster with f2py
        - 'Kirchner_numba': Kirchner's algorithm with Numba support for acceleration (default)
    verbose : bool
        If True, print warning messages

    Returns
    -------

    res : dict
        contains the cross wavelet coherence, cross-wavelet phase,
        vector of frequency, evenly-spaced time points, AR1 sims, cone of influence

    See also
    --------

    pyleoclim.utils.wavelet.wwz_basic : Returns the weighted wavelet amplitude using the original method from Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.wwz_nproc : Returns the weighted wavelet amplitude using the original method from Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_basic : Return the weighted wavelet amplitude (WWA) modified by Kirchner. No multiprocessing

    pyleoclim.utils.wavelet.kirchner_nproc : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Supports multiprocessing

    pyleoclim.utils.wavelet.kirchner_numba : Return the weighted wavelet amplitude (WWA) modified by Kirchner using Numba package.

    pyleoclim.utils.wavelet.kirchner_f2py : Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Uses Fortran. Fastest method but requires a compiler.

    pyleoclim.utils.filter.savitzky_golay : Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    pyleoclim.utils.wavelet.make_freq_vector : Make frequency vector

    '''
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
        freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
        freq = make_freq_vector(ts1, method=freq_method, **freq_kwargs)
        print(f'Setting freq={freq[:3]}...{freq[-3:]}, nfreq={np.size(freq)}')

    ys1_cut, ts1_cut, freq1, tau1 = prepare_wwz(ys1, ts1, freq=freq, tau=tau)
    ys2_cut, ts2_cut, freq2, tau2 = prepare_wwz(ys2, ts2, freq=freq, tau=tau)

    if np.any(tau1 != tau2):
        if verbose: print('inconsistent `tau`, recalculating...')
        tau_min = np.min([np.min(tau1), np.min(tau2)])
        tau_max = np.max([np.max(tau1), np.max(tau2)])
        ntau = np.max([np.size(tau1), np.size(tau2)])
        tau = np.linspace(tau_min, tau_max, ntau)
    else:
        tau = tau1

    if np.any(freq1 != freq2):
        if verbose: print('inconsistent `freq`, recalculating...')
        freq_min = np.min([np.min(freq1), np.min(freq2)])
        freq_max = np.max([np.max(freq1), np.max(freq2)])
        nfreq = np.max([np.size(freq1), np.size(freq2)])
        freq = np.linspace(freq_min, freq_max, nfreq)
    else:
        freq = freq1

    if freq[0] == 0:
        freq = freq[1:] # delete 0 frequency if present

    res_wwz1 = wwz(ys1_cut, ts1_cut, tau=tau, freq=freq, c=c, Neff=Neff, nMC=0,
                   nproc=nproc, detrend=detrend, sg_kwargs=sg_kwargs,
                   gaussianize=gaussianize, standardize=standardize, method=method)
    res_wwz2 = wwz(ys2_cut, ts2_cut, tau=tau, freq=freq, c=c, Neff=Neff, nMC=0,
                   nproc=nproc, detrend=detrend, sg_kwargs=sg_kwargs,
                   gaussianize=gaussianize, standardize=standardize, method=method)

    wt_coeff1 = res_wwz1.coeff[1] - res_wwz1.coeff[2]*1j
    wt_coeff2 = res_wwz2.coeff[1] - res_wwz2.coeff[2]*1j

    xw_coherence, xw_phase = wavelet_coherence(wt_coeff1, wt_coeff2, freq, tau, smooth_factor=smooth_factor)
    xwt, xw_amplitude, _ = cross_wt(wt_coeff1, wt_coeff2)

    # Monte-Carlo simulations of AR1 process
    nt = np.size(tau)
    nf = np.size(freq)

    #  coherence_red = np.ndarray(shape=(nMC, nt, nf))
    #  AR1_q = np.ndarray(shape=(nt, nf))
    coherence_red = None
    AR1_q = None

    #  if nMC >= 1:

        #  for i in tqdm(range(nMC), desc='Monte-Carlo simulations'):
            #  r1 = ar1_sim(ys1_cut, np.size(ts1_cut), 1, ts=ts1_cut)
            #  r2 = ar1_sim(ys2_cut, np.size(ts2_cut), 1, ts=ts2_cut)
            #  res_wwz_r1 = wwz(r1, ts1_cut, tau=tau, freq=freq, c=c, Neff=Neff, nMC=0, nproc=nproc,
                                                     #  detrend=detrend, sg_kwargs=sg_kwrags,
                                                     #  gaussianize=gaussianize, standardize=standardize)
            #  res_wwz_r2 = wwz(r2, ts2_cut, tau=tau, freq=freq, c=c, Neff=Neff, nMC=0, nproc=nproc,
                                                     #  detrend=detrend, sg_kwargs=sg_kwargs,
                                                     #  gaussianize=gaussianize, standardize=standardize)

            #  wt_coeffr1 = res_wwz_r1.coeff[1] - res_wwz_r2.coeff[2]*1j
            #  wt_coeffr2 = res_wwz_r1.coeff[1] - res_wwz_r2.coeff[2]*1j
            #  coherence_red[i, :, :], phase_red = wavelet_coherence(wt_coeffr1, wt_coeffr2, freq, tau, smooth_factor=smooth_factor)

        #  for j in range(nt):
            #  for k in range(nf):
                #  AR1_q[j, k] = mquantiles(coherence_red[:, j, k], 0.95)

    #  else:
        #  AR1_q = None

    coi = make_coi(tau, Neff=Neff)
    Results = collections.namedtuple('Results', ['xw_coherence', 'xw_amplitude', 'xw_phase', 'xwt', 'freq', 'time', 'AR1_q', 'coi'])
    res = Results(xw_coherence=xw_coherence, xw_amplitude=xw_amplitude, xw_phase=xw_phase, xwt=xwt,
                  freq=freq, time=tau, AR1_q=AR1_q, coi=coi)

    return res
def freq_vector_lomb_scargle(ts, dt= None, nf=None, ofac=4, hifac=1):
    ''' Return the frequency vector based on the REDFIT recommendation.

    Parameters
    ----------

    ts : array
        time axis of the time series
    dt : float
        The resolution of the data. If None, uses the median resolution. Defaults to None.
    nf : int
        Number of frequency points.
        If None, calculated as the difference between the highest and lowest frequencies (set by hifac and ofac) divided by resolution. Defaults to None
    ofac : float
        Oversampling rate that influences the resolution of the frequency axis,
                 when equals to 1, it means no oversamling (should be >= 1).
                 The default value 4 is usually a good value.
    hifac : float
        fhi/fnyq (should be <= 1), where fhi is the highest frequency that
        can be analyzed by the Lomb-Scargle algorithm and fnyq is the Nyquist frequency.

    Returns
    -------

    freq : array
        the frequency vector

    References
    ----------

    Trauth, M. H. MATLAB® Recipes for Earth Sciences. (Springer, 2015). pp 181.


    See also
    --------

    pyleoclim.utils.wavelet.freq_vector_welch : Return the frequency vector based on the Welch's method.

    pyleoclim.utils.wavelet.freq_vector_nfft : Return the frequency vector based on NFFT

    pyleoclim.utils.wavelet.freq_vector_scale : Return the frequency vector based on scales

    pyleoclim.utils.wavelet.freq_vector_log : Return the frequency vector based on logspace

    pyleoclim.utils.wavelet.make_freq_vector : Make frequency vector
    '''
    assert ofac >= 1 and hifac <= 1, "`ofac` should be >= 1, and `hifac` should be <= 1"

    if dt is None:
        dt = np.median(np.diff(ts))
    flo = (1/(2*dt)) / (np.size(ts)*ofac)
    fhi = hifac / (2*dt)

    if nf is None:
        df = flo
        nf = int((fhi - flo) / df + 1)

    freq = np.linspace(flo, fhi, nf)

    return freq

def freq_vector_welch(ts):
    ''' Return the frequency vector based on the Welch's method.

    Parameters
    ----------

    ts : array
        time axis of the time series

    Returns
    -------

    freq : array
        the frequency vector

    References
    ----------

    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/Spectral.py

    See also
    --------

    pyleoclim.utils.wavelet.freq_vector_lomb_scargle : Return the frequency vector based on the REDFIT
        recommendation.

    pyleoclim.utils.wavelet.freq_vector_nfft : Return the frequency vector based on NFFT

    pyleoclim.utils.wavelet.freq_vector_scale : Return the frequency vector based on scales

    pyleoclim.utils.wavelet.freq_vector_log : Return the frequency vector based on logspace

    pyleoclim.utils.wavelet.make_freq_vector : Make frequency vector

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

def freq_vector_nfft(ts):
    ''' Return the frequency vector based on NFFT

    Parameters
    ----------

    ts : array
        time axis of the time series

    Returns
    -------

    freq : array
        the frequency vector

    See also
    --------

    pyleoclim.utils.wavelet.freq_vector_lomb_scargle : Return the frequency vector based on the REDFIT
        recommendation.

    pyleoclim.utils.wavelet.freq_vector_welch : Return the frequency vector based on the Welch's method.

    pyleoclim.utils.wavelet.freq_vector_scale : Return the frequency vector based on scales

    pyleoclim.utils.wavelet.freq_vector_log : Return the frequency vector based on logspace

    pyleoclim.utils.wavelet.make_freq_vector : Make frequency vector

    '''
    nt = np.size(ts)
    dt = np.median(np.diff(ts))
    fs = 1 / dt
    n_freq = nt//2 + 1

    freq = np.linspace(0, fs/2, n_freq)

    return freq

def freq_vector_scale(ts, nv=12, fourier_factor=1):
    ''' Return the frequency vector based on scales for wavelet analysis

    Parameters
    ----------

    ts : array
        time axis of the time series

    nv : int
        the parameter that controls the number of freq points

    Returns
    -------

    freq : array
        the frequency vector

    See also
    --------

    pyleoclim.utils.wavelet.freq_vector_lomb_scargle : Return the frequency vector based on the REDFIT
        recommendation.

    pyleoclim.utils.wavelet.freq_vector_welch : Return the frequency vector based on the Welch's method.

    pyleoclim.utils.wavelet.freq_vector_nfft : Return the frequency vector based on NFFT

    pyleoclim.utils.wavelet.freq_vector_log : Return the frequency vector based on logspace

    pyleoclim.utils.wavelet.make_freq_vector : Make frequency vector

    '''

    s0 = 2*np.median(np.diff(ts))
    a0 = 2**(1/nv)
    noct = np.floor(np.log2(np.size(ts)))-1  # number of octave
    scale = s0*a0**(np.arange(noct*nv+1))
    freq = 1/(scale[::-1]*fourier_factor)

    return freq

def freq_vector_log(ts, nfreq=None):
    ''' Return the frequency vector based on logspace

    Parameters
    ----------

    ts : array
        time axis of the time series

    nv : int
        the parameter that controls the number of freq points

    Returns
    -------

    freq : array
        the frequency vector

    See also
    --------

    pyleoclim.utils.wavelet.freq_vector_lomb_scargle : Return the frequency vector based on the REDFIT
        recommendation.

    pyleoclim.utils.wavelet.freq_vector_welch : Return the frequency vector based on the Welch's method.

    pyleoclim.utils.wavelet.freq_vector_nfft : Return the frequency vector based on NFFT

    pyleoclim.utils.wavelet.freq_vector_scale : Return the frequency vector based on scales

    pyleoclim.utils.wavelet.make_freq_vector : Make frequency vector
    '''

    nt = np.size(ts)
    dt = np.median(np.diff(ts))
    fs = 1 / dt
    if nfreq is None:
        nfreq = nt//10 + 1

    fmin = 2/(np.max(ts)-np.min(ts))
    fmax = fs/2
    start = np.log2(fmin)
    stop = np.log2(fmax)

    freq = np.logspace(start, stop, nfreq, base=2)

    return freq

def make_freq_vector(ts, method='log', **kwargs):
    ''' Make frequency vector

    This function selects among five methods to obtain the frequency
    vector.

    Parameters
    ----------

    ts : array
        Time axis of the time series
    method : string
        The method to use. Options are 'log' (default), 'nfft', 'lomb_scargle', 'welch', and 'scale'
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

    See also
    --------

    pyleoclim.utils.wavelet.freq_vector_lomb_scargle : Return the frequency vector based on the REDFIT
        recommendation.

    pyleoclim.utils.wavelet.freq_vector_welch : Return the frequency vector based on the Welch's method.

    pyleoclim.utils.wavelet.freq_vector_nfft : Return the frequency vector based on NFFT

    pyleoclim.utils.wavelet.freq_vector_scale : Return the frequency vector based on scales

    pyleoclim.utils.wavelet.freq_vector_log : Return the frequency vector based on logspace

    '''

    if method == 'lomb_scargle':
        freq = freq_vector_lomb_scargle(ts,**kwargs)
    elif method == 'welch':
        freq = freq_vector_welch(ts)
    elif method == 'nfft':
        freq = freq_vector_nfft(ts)
    elif method == 'scale':
        freq = freq_vector_scale(ts, **kwargs)
    elif method == 'log':
        freq = freq_vector_log(ts, **kwargs)
    else:
        raise ValueError('This method is not supported')
    #  freq = freq[1:]  # discard the first element 0

    return freq

def beta_estimation(psd, freq, fmin=None, fmax=None, logf_binning_step='max', verbose=False):
    ''' Estimate the power slope of a 1/f^beta process.

    Parameters
    ----------

    psd : array
        the power spectral density
    freq : array
        the frequency vector
    fmin : float
        the min of frequency range for beta estimation
    fmax : float
        the max of frequency range for beta estimation
    verbose : bool
         if True, will print out debug information

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
        if verbose:
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
    if logf_binning_step == 'max':
        logf_step = np.max(np.diff(logf))
    elif logf_binning_step == 'first':
        logf_step = logf[fminindx+1] - logf[fminindx]
    else:
        raise ValueError('the option for logf_binning_step is unknown')

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

    # note below: 'drop' is used for missing, so NaNs will be removed, and we need to put it back in the end
    model = sm.OLS(Y, X_ex, missing='drop')
    results = model.fit()

    if np.size(results.params) < 2:
        beta = np.nan
        Y_reg = np.nan
        std_err = np.nan
    else:
        beta = -results.params[1]  # the slope we want
        Y_reg_raw = 10**model.predict(results.params)  # prediction based on linear regression
        # handeling potential NaNs in psd_binned
        Y_reg = []
        i = 0
        for psd in psd_binned:
            if np.isnan(psd):
                Y_reg.append(np.nan)
            else:
                Y_reg.append(Y_reg_raw[i])
                i += 1

        Y_reg = np.array(Y_reg)

        std_err = results.bse[1]

    res = Results(beta=beta, f_binned=f_binned, psd_binned=psd_binned, Y_reg=Y_reg, std_err=std_err)

    return res

def beta2HurstIndex(beta):
    ''' Translate psd slope to Hurst index

    Parameters
    ----------

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

def psd_ar(var_noise, freq, ar_params, f_sampling):
    ''' Return the theoretical power spectral density (PSD) of an autoregressive model

    Parameters
    ----------

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

def fBMsim(N=128, H=0.25):
    '''Simple method to generate fractional Brownian Motion

    Parameters
    ----------

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

def psd_fBM(freq, ts, H):
    ''' Return the theoretical psd of a fBM

    Parameters
    ----------

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

def get_wwz_func(nproc, method):
    ''' Return the wwz function to use.

    Parameters
    ----------

    nproc : int
        the number of processes for multiprocessing
    method : string
        'Foster' - the original WWZ method;
        'Kirchner' - the method Kirchner adapted from Foster;
        'Kirchner_f2py' - the method Kirchner adapted from Foster with f2py
        'Kirchner_numba' - Kirchner's algorithm with Numba support for acceleration (default)

    Returns
    -------

    wwz_func : function
        the wwz function to use

    '''
    assertPositiveInt(nproc)

    if method == 'Foster':
        if nproc == 1:
            wwz_func = wwz_basic
        else:
            wwz_func = wwz_nproc

    elif method == 'Kirchner':
        if nproc == 1:
            wwz_func = kirchner_basic
        else:
            wwz_func = kirchner_nproc
    elif method == 'Kirchner_f2py':
        wwz_func = kirchner_f2py
    elif method == 'Kirchner_numba':
        wwz_func = kirchner_numba
    else:
        raise ValueError('Wrong specific method name for WWZ. Should be one of {"Foster", "Kirchner", "Kirchner_f2py", "Kirchner_numba"}')

    return wwz_func

def prepare_wwz(ys, ts, freq=None, freq_method='log', freq_kwargs=None, tau=None, len_bd=0, bc_mode='reflect', reflect_type='odd', **kwargs):
    ''' Return the truncated time series with NaNs deleted and estimate frequency vector and tau

    Parameters
    ----------

    ys : array
        a time series, NaNs will be deleted automatically
    ts : array
        the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
    freq : array
        vector of frequency. If None, will be ganerated according to freq_method.
        may be set.
    freq_method : str
        when freq=None, freq will be ganerated according to freq_method
    freq_kwargs : str
        used when freq=None for certain methods
    tau : array
        The evenly-spaced time points, namely the time shift for wavelet analysis.
        If the boundaries of tau are not exactly on two of the time axis points, then tau will be adjusted to be so.
        If None, at most 50 tau points will be generated from the input time span.
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
    ys, ts = clean_ts(ys, ts)

    if tau is None:
        ntau = np.min([np.size(ts), 50])
        tau = np.linspace(np.min(ts), np.max(ts), ntau)

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
        freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
        freq = make_freq_vector(ts_cut, method=freq_method, **freq_kwargs)

    # remove 0 in freq vector
    freq = freq[freq != 0]

    return ys_cut, ts_cut, freq, tau

def cross_wt(coeff1, coeff2):
    ''' Return the cross wavelet transform.

    Parameters
    ----------

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

    Grinsted, A., Moore, J. C. & Jevrejeva, S. Application of the cross wavelet transform and
    wavelet coherence to geophysical time series. Nonlin. Processes Geophys. 11, 561–566 (2004).

    '''
    xwt = coeff1 * np.conj(coeff2)
    xw_amplitude = np.sqrt(xwt.real**2 + xwt.imag**2)
    xw_phase = np.arctan2(xwt.imag, xwt.real)

    return xwt, xw_amplitude, xw_phase

def wavelet_coherence(coeff1, coeff2, freq, tau, smooth_factor=0.25):
    ''' Return the cross wavelet coherence.

    Parameters
    ----------

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
        """ Smoothing function adapted from https://github.com/regeirk/pycwt/blob/master/pycwt/helpers.py

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

    S12 = smoothing(xwt/scales, snorm, dj)
    S1 = smoothing(power1/scales, snorm, dj)
    S2 = smoothing(power2/scales, snorm, dj)
    xw_coherence = np.abs(S12)**2 / (S1*S2)
    wcs = S12 / (np.sqrt(S1)*np.sqrt(S2))
    xw_phase = np.angle(wcs)

    return xw_coherence, xw_phase

def reconstruct_ts(coeff, freq, tau, t, len_bd=0):
    ''' Reconstruct the normalized time series from the wavelet coefficients.

    Parameters
    ----------

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

    rec_ts = preprocess(rec_ts, t, detrend=False, gaussianize=False, standardize=False)

    return rec_ts, t

# ## Methods for Torrence and compo

# # This is the main function, which has been rewritten to work with functionalities in Pyleoclim

# def cwt(ys,ts,mother='morlet',param=None,freq=None,freq_method='scale',
#         freq_kwargs={},detrend=False, sg_kwargs={}, gaussianize=False,
#         standardize=False,pad=False,pad_kwargs={}):

#     ys=np.array(ys)
#     ts=np.array(ts)

#     ys, ts = clean_ts(ys, ts) #clean up time

#     #make sure that the time series is evenly-spaced
#     if is_evenly_spaced(ts) == True:
#         dt = np.mean(np.diff(ts))
#     else:
#         raise ValueError('Time series must be evenly spaced in time')

#     # prepare the time series
#     pd_ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs,
#                        gaussianize=gaussianize, standardize=standardize)

#     # Get the fourier factor
#     if mother.lower() == 'morlet':
#         if param is None:
#             param = 6.
#         fourier_factor = 4 * np.pi / (param + np.sqrt(2 + param**2))
#     elif mother.lower() == 'paul':
#         if param is None:
#             param = 4.
#         fourier_factor = 4 * np.pi / (2 * param + 1)
#     elif mother.lower() == 'dog':
#         if param is None:
#             param = 2.
#         fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * param + 1))
#     else:
#         fourier_factor = 1

#     #get the frequency/scale information
#     if freq is None:
#         freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
#         if freq_method == 'scale':
#             freq_kwargs.update({'fourier_factor':fourier_factor})
#         freq = make_freq_vector(ts, method=freq_method, **freq_kwargs)
#     # Use scales
#     scale = np.sort(1/(freq*fourier_factor))

#     #Normalize
#     #n_ys = pd_ys-np.mean(pd_ys)

#     #pad if wanted
#     if pad == True:
#         pad_kwargs = {} if pad_kwargs is None else pad_kwargs.copy()
#         yp,tp = ts_pad(pd_ys,ts,**pad_kwargs)
#     else:
#         yp=pd_ys
#         tp=ts

#     # Wave calculation
#     n = len(yp)

#     # construct wavenumber array used in transform [Eqn(5)]
#     kplus = np.arange(1, int(n / 2) + 1)
#     kplus = (kplus * 2 * np.pi / (n * dt))
#     kminus = np.arange(1, int((n - 1) / 2) + 1)
#     kminus = np.sort((-kminus * 2 * np.pi / (n * dt)))
#     k = np.concatenate(([0.], kplus, kminus))

#     # compute FFT of the (padded) time series
#     f = np.fft.fft(yp)

#     # define the wavelet array
#     wave = np.zeros(shape=(len(scale), n), dtype=complex)

#     # loop through all scales and compute transform
#     for a1 in range(0, len(scale)):
#         daughter, fourier_factor, coi, _ = \
#             wave_bases(mother, k, scale[a1], param)
#         wave[a1, :] = np.fft.ifft(f * daughter)  # wavelet transform[Eqn(4)]

#     #COI
#     coi = coi * dt * np.concatenate((
#         np.insert(np.arange(int((len(ys) + 1) / 2) - 1), [0], [1E-5]),
#         np.insert(np.flipud(np.arange(0, int(len(ys) / 2) - 1)), [-1], [1E-5])))

#     #Remove the padding
#     if pad == True:
#         idx = np.in1d(tp,ts)
#         wave = wave[:,idx]

#     res = {}

#     return res


# def wave_bases(mother, k, scale, param):
#     '''


#     Parameters
#     ----------
#     mother : string, {}
#         DESCRIPTION.
#     k : TYPE
#         DESCRIPTION.
#     scale : TYPE
#         DESCRIPTION.
#     param : TYPE
#         DESCRIPTION.

#     Raises
#     ------
#     KeyError
#         DESCRIPTION.

#     Returns
#     -------
#     daughter : TYPE
#         DESCRIPTION.
#     fourier_factor : TYPE
#         DESCRIPTION.
#     coi : TYPE
#         DESCRIPTION.
#     dofmin : TYPE
#         DESCRIPTION.

#     '''

#     n = len(k)
#     kplus = np.array(k > 0., dtype=float)

#     if mother == 'morlet':  # -----------------------------------  Morlet

#         if param == -1:
#             param = 6.

#         k0 = np.copy(param)
#         # calc psi_0(s omega) from Table 1
#         expnt = -(scale * k - k0) ** 2 / 2. * kplus
#         norm = np.sqrt(scale * k[1]) * (np.pi ** (-0.25)) * np.sqrt(n)
#         daughter = norm * np.exp(expnt)
#         daughter = daughter * kplus  # Heaviside step function
#         # Scale-->Fourier [Sec.3h]
#         fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
#         coi = fourier_factor / np.sqrt(2)  # Cone-of-influence [Sec.3g]
#         dofmin = 2  # Degrees of freedom
#     elif mother == 'paul':  # --------------------------------  Paul
#         if param == -1:
#             param = 4.
#         m = param
#         # calc psi_0(s omega) from Table 1
#         expnt = -scale * k * kplus
#         norm_bottom = np.sqrt(m * np.prod(np.arange(1, (2 * m))))
#         norm = np.sqrt(scale * k[1]) * (2 ** m / norm_bottom) * np.sqrt(n)
#         daughter = norm * ((scale * k) ** m) * np.exp(expnt) * kplus
#         fourier_factor = 4 * np.pi / (2 * m + 1)
#         coi = fourier_factor * np.sqrt(2)
#         dofmin = 2
#     elif mother == 'dog':  # --------------------------------  DOG
#         if param == -1:
#             param = 2.
#         m = param
#         # calc psi_0(s omega) from Table 1
#         expnt = -(scale * k) ** 2 / 2.0
#         norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
#         daughter = -norm * (1j ** m) * ((scale * k) ** m) * np.exp(expnt)
#         fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
#         coi = fourier_factor / np.sqrt(2)
#         dofmin = 1
#     else:
#         raise KeyError('Mother must be one of "morlet", "paul", "dog"')

#     return daughter, fourier_factor, coi, dofmin


# def chisquare_inv(P, V):

#     if (1 - P) < 1E-4:
#         print('P must be < 0.9999')

#     if P == 0.95 and V == 2:  # this is a no-brainer
#         X = 5.9915
#         return X

#     MINN = 0.01  # hopefully this is small enough
#     MAXX = 1  # actually starts at 10 (see while loop below)
#     X = 1
#     TOLERANCE = 1E-4  # this should be accurate enough

#     while (X + TOLERANCE) >= MAXX:  # should only need to loop thru once
#         MAXX = MAXX * 10.
#     # this calculates value for X, NORMALIZED by V
#         X = fminbound(chisquare_solve, MINN, MAXX, args=(P, V), xtol=TOLERANCE)
#         MINN = MAXX

#     X = X * V  # put back in the goofy V factor

#     return X

# def chisquare_solve(XGUESS, P, V):

#     PGUESS = gammainc(V / 2, V * XGUESS / 2)  # incomplete Gamma function

#     PDIFF = np.abs(PGUESS - P)            # error in calculated P

#     TOL = 1E-4
#     if PGUESS >= 1 - TOL:  # if P is very close to 1 (i.e. a bad guess)
#         PDIFF = XGUESS   # then just assign some big number like XGUESS

#     return PDIFF
