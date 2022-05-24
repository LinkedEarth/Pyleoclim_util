#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for spectral analysis, including WWZ, CWT, Lomb-Scargle, MTM, and Welch.
Designed for NumPy arrays, either evenly spaced or not (method-dependent).

All spectral methods must return a dictionary containing one vector for the 
frequency axis and the power spectral density (PSD).

Additional utilities help compute an optimal frequency vector or estimate scaling exponents.
"""

import numpy as np
from scipy import signal
import nitime.algorithms as nialg
import statsmodels.api as sm
import collections
import warnings

__all__ = [
    'wwz_psd',
    'cwt_psd',
    'mtm',
    'lomb_scargle',
    'welch',
    'periodogram'
]

from .tsbase import (
    is_evenly_spaced,
    clean_ts
)

from .tsutils import preprocess

from .wavelet import (
    make_freq_vector,
    prepare_wwz,
    wwz,
    wwa2psd,
    cwt,
)

#-----------
#Wrapper
#-----------

#---------
#Main functions
#---------


def welch(ys, ts, window='hann',nperseg=None, noverlap=None, nfft=None,
           return_onesided=True, detrend = None, sg_kwargs = None,
           gaussianize=False, standardize=False,
           scaling='density', average='mean'):
    '''Estimate power spectral density using Welch's periodogram

    Wrapper for the function implemented in scipy.signal.welch
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html for details.

    Welch's method is an approach for spectral density estimation. It computes 
    an estimate of the power spectral density by dividing the data into overlapping
    segments, computing a modified periodogram for each segment and averaging 
    the periodograms to lower the estimator's variance.

    Parameters
    ----------

    ys : array
        a time series
    ts : array
        time axis of the time series
    window : string or tuple
        Desired window to use. Possible values:
            - boxcar
            - triang
            - blackman
            - hamming
            - hann (default)
            - bartlett
            - flattop
            - parzen
            - bohman
            - blackmanharris
            - nuttail
            - barthann
            - kaiser (needs beta)
            - gaussian (needs standard deviation)
            - general_gaussian (needs power, width)
            - slepian (needs width)
            - dpss (needs normalized half-bandwidth)
            - chebwin (needs attenuation)
            - exponential (needs decay scale)
            - tukey (needs taper fraction)
        If the window requires no parameters, then window can be a string.
        If the window requires parameters, then window must be a tuple with the first argument the string name of the window, and the next arguments the needed parameters.
        If window is a floating point number, it is interpreted as the beta parameter of the kaiser window.
      nperseg : int
          Length of each segment. If none, nperseg=len(ys)/2. Default to None This will give three segments with 50% overlap
      noverlap : int
          Number of points to overlap. If None, noverlap=nperseg//2. Defaults to None, represents 50% overlap
      nfft: int
          Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg
      return_onesided : bool
          If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Defaults to True, but for complex data, a two-sided spectrum is always returned.
      detrend : str
          If None, no detrending is applied. Available detrending methods:
              - None - no detrending will be applied (default);
              - linear - a linear least-squares fit to `ys` is subtracted;
              - constant - the mean of `ys` is subtracted
              - savitzy-golay - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
              - emd - Empirical mode decomposition
      sg_kwargs : dict
          The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
      gaussianize : bool
          If True, gaussianizes the timeseries
      standardize : bool
          If True, standardizes the timeseries
      scaling : {"density,"spectrum}
          Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz and computing the power spectrum (‘spectrum’) where Pxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density'
      average : {'mean','median'}
          Method to use when combining periodograms. Defaults to ‘mean’.

    Returns
    -------
    res_dict : dict
        the result dictionary, including
        - freq (array): the frequency vector
        - psd (array): the spectral density vector


    See also
    --------
    pyleoclim.utils.spectral.periodogram : Spectral density estimation using a Blackman-Tukey periodogram
    pyleoclim.utils.spectral.mtm : Spectral density estimation using the multi-taper method
    pyleoclim.utils.spectral.lomb_scargle : Lomb-scargle priodogram
    pyleoclim.utils.spectral.wwz_psd : Spectral estimation using the Weighted Wavelet Z-transform
    pyleoclim.utils.spectral.cwt_psd : Spectral estimation using the Continuous Wavelet Transform
    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay
    pyleoclim.utils.tsutils.detrend : detrending functionalities using 4 possible methods  
    pyleoclim.utils.tsutils.gaussianize_1d: Quantile maps a 1D array to a Gaussian distribution 
    pyleoclim.utils.tsutils.standardize: Centers and normalizes a given time series.

    References
    ----------
    P. Welch, “The use of the fast Fourier transform for the estimation of power spectra: 
        A method based on time averaging over short, modified periodograms”, 
        IEEE Trans. Audio Electroacoust. vol. 15, pp. 70-73, 1967.

    '''

    ts = np.array(ts)
    ys = np.array(ys)

    if len(ts) != len(ys):
        raise ValueError('Time and value axis should be the same length')

    if nperseg == None:
        nperseg = len(ys/2)

    # remove NaNs
    ys, ts = clean_ts(ys,ts)
    # check for evenly-spaced
    check = is_evenly_spaced(ts)
    if check == False:
        raise ValueError('For the Welch method, data should be evenly spaced')
    # preprocessing
    ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs,
               gaussianize=gaussianize, standardize=standardize)


    # calculate sampling frequency fs
    dt = np.median(np.diff(ts))
    fs = 1 / dt

    # spectral analysis with scipy welch
    freq, psd = signal.welch(ys, fs=fs, window=window,nperseg=nperseg,noverlap=noverlap,
                             nfft=nfft, return_onesided=return_onesided, scaling=scaling,
                             average=average, detrend = False, axis=-1)

    # fix zero frequency point
    if freq[0] == 0:
        psd[0] = np.nan

    # output result
    res_dict = {
        'freq': np.asarray(freq),
        'psd' : np.asarray(psd),
    }

    return res_dict


def mtm(ys, ts, NW=None, BW=None, detrend = None, sg_kwargs=None,
           gaussianize=False, standardize=False, adaptive=False, jackknife=True,
           low_bias=True, sides='default', nfft=None):
    ''' Spectral density using the multi-taper method.


    Based on the nitime package: http://nipy.org/nitime/api/generated/nitime.algorithms.spectral.html

    Parameters
    ----------

    ys : array
        a time series
    ts : array
        time axis of the time series
    NW : float
        The normalized half-bandwidth of the data tapers, indicating a
        multiple of the fundamental frequency of the DFT (Fs/N).
        Common choices are n/2, for n >= 4.
    BW : float
        The sampling-relative bandwidth of the data tapers
    detrend : str
          If None, no detrending is applied. Available detrending methods:
              - None - no detrending will be applied (default);
              - linear - a linear least-squares fit to `ys` is subtracted;
              - constant - the mean of `ys` is subtracted
              - savitzy-golay - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
              - emd - Empirical mode decomposition
      sg_kwargs : dict
          The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
      gaussianize : bool
          If True, gaussianizes the timeseries
      standardize : bool
          If True, standardizes the timeseries
      adaptive : {True/False}
          Use an adaptive weighting routine to combine the PSD estimates of
          different tapers.
      jackknife : {True/False}
          Use the jackknife method to make an estimate of the PSD variance
          at each point.
      low_bias : {True/False}
          Rather than use 2NW tapers, only use the tapers that have better than
          90% spectral concentration within the bandwidth (still using
          a maximum of 2NW tapers)
      sides : str (optional)   [ 'default' | 'onesided' | 'twosided' ]
          This determines which sides of the spectrum to return.
          For complex-valued inputs, the default is two-sided, for real-valued
          inputs, default is one-sided Indicates whether to return a one-sided
          or two-sided

    Returns
    -------

    res_dict : dict
        the result dictionary, including
        - freq (array): the frequency vector
        - psd (array): the spectral density vector

    See Also
    --------
    pyleoclim.utils.spectral.periodogram : Spectral density estimation using a Blackman-Tukey periodogram
    pyleoclim.utils.spectral.welch : spectral estimation using Welch's periodogram
    pyleoclim.utils.spectral.lomb_scargle : Lomb-scargle priodogram
    pyleoclim.utils.spectral.wwz_psd : Spectral estimation using the Weighted Wavelet Z-transform
    pyleoclim.utils.spectral.cwt_psd : Spectral estimation using the Continuous Wavelet Transform
    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay
    pyleoclim.utils.tsutils.detrend : detrending functionalities using 4 possible methods  
    pyleoclim.utils.tsutils.gaussianize_1d: Quantile maps a 1D array to a Gaussian distribution 
    pyleoclim.utils.tsutils.standardize: Centers and normalizes a given time series.

    '''
    # preprocessing
    ts = np.array(ts)
    ys = np.array(ys)

    if len(ts) != len(ys):
        raise ValueError('Time and value axis should be the same length')

    # remove NaNs
    ys, ts = clean_ts(ys,ts)
    # check for evenly-spaced
    check = is_evenly_spaced(ts)
    if check == False:
        raise ValueError('For the MTM method, data should be evenly spaced')
    # preprocessing
    ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs,
               gaussianize=gaussianize, standardize=standardize)


    # calculate sampling frequency fs
    dt = np.median(np.diff(ts))
    fs = 1 / dt

    # spectral analysis
    freq, psd, nu = nialg.multi_taper_psd(ys, Fs=fs, NW=NW, BW=BW,adaptive=adaptive,
                                          jackknife=jackknife, low_bias=low_bias,
                                          sides=sides,NFFT=nfft)  # call nitime func

    # fix the zero frequency point
    if freq[0] == 0:
        psd[0] = np.nan

    # output result
    res_dict = {
        'freq': np.asarray(freq),
        'psd': np.asarray(psd),
    }

    return res_dict


def lomb_scargle(ys, ts, freq=None, freq_method='lomb_scargle',
                 freq_kwargs=None, n50=3, window='hann',
                 detrend = None, sg_kwargs=None,
                 gaussianize=False,
                 standardize=False,
                 average='mean'):
    """ Lomb-scargle priodogram

    Appropriate for unevenly-spaced arrays.
    Uses the lomb-scargle implementation from scipy.signal: https://scipy.github.io/devdocs/generated/scipy.signal.lombscargle.html#scipy.signal.lombscargle

    Parameters
    ----------

    ys : array
        a time series
    ts : array
        time axis of the time series
    freq : str or array
        vector of frequency.
        If string, uses the following method:
    freq_method : str
        Method to generate the frequency vector if not set directly. The following options are avialable:
            - log
            - lomb_scargle (default)
            - welch
            - scale
            - nfft
        See utils.wavelet.make_freq_vector for details
    freq_kwargs : dict
        Arguments for the method chosen in freq_method. See specific functions in utils.wavelet for details
        By default, uses dt=median(ts), ofac=4 and hifac=1 for Lomb-Scargle
    n50: int
        The number of 50% overlapping segment to apply
    window : str or tuple
        Desired window to use. Possible values:
            - boxcar
            - triang
            - blackman
            - hamming
            - hann (default)
            - bartlett
            - flattop
            - parzen
            - bohman
            - blackmanharris
            - nuttail
            - barthann
            - kaiser (needs beta)
            - gaussian (needs standard deviation)
            - general_gaussian (needs power, width)
            - slepian (needs width)
            - dpss (needs normalized half-bandwidth)
            - chebwin (needs attenuation)
            - exponential (needs decay scale)
            - tukey (needs taper fraction)
        If the window requires no parameters, then window can be a string.
        If the window requires parameters, then window must be a tuple with the first argument the string name of the window, and the next arguments the needed parameters.
        If window is a floating point number, it is interpreted as the beta parameter of the kaiser window.
     detrend : str
          If None, no detrending is applied. Available detrending methods:
              - None - no detrending will be applied (default);
              - linear - a linear least-squares fit to `ys` is subtracted;
              - constant - the mean of `ys` is subtracted
              - savitzy-golay - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
              - emd - Empirical mode decomposition
      sg_kwargs : dict
          The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
      gaussianize : bool
          If True, gaussianizes the timeseries
      standardize : bool
          If True, standardizes the timeseriesprep_args : dict
      average : {'mean','median'}
          Method to use when averaging periodograms. Defaults to ‘mean’.

    Returns
    -------

    res_dict : dict
        the result dictionary, including
        - freq (array): the frequency vector
        - psd (array): the spectral density vector

    See Also
    --------
    
    pyleoclim.utils.spectral.periodogram : Estimate power spectral density using a periodogram
    pyleoclim.utils.spectral.mtm : Retuns spectral density using a multi-taper method
    pyleoclim.utils.spectral.welch : Returns power spectral density using the Welch method
    pyleoclim.utils.spectral.wwz_psd : Spectral estimation using the Weighted Wavelet Z-transform
    pyleoclim.utils.spectral.cwt_psd : Spectral estimation using the Continuous Wavelet Transform
    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay
    pyleoclim.utils.tsutils.detrend : detrending functionalities using 4 possible methods  
    pyleoclim.utils.tsutils.gaussianize_1d: Quantile maps a 1D array to a Gaussian distribution 
    pyleoclim.utils.tsutils.standardize: Centers and normalizes a given time series.

    References
    ----------
    Lomb, N. R. (1976). Least-squares frequency analysis of unequally spaced data. Astrophysics and Space Science 39, 447-462.

    Scargle, J. D. (1982). Studies in astronomical time series analysis. II. Statistical aspects of spectral analyis of unvenly spaced data. The Astrophysical Journal, 263(2), 835-853.

    Scargle, J. D. (1982). Studies in astronomical time series analysis. II. Statistical aspects of spectral analyis of unvenly spaced data. The Astrophysical Journal, 263(2), 835-853.

    """
    ts = np.array(ts)
    ys = np.array(ys)

    if len(ts) != len(ys):
        raise ValueError('Time and value axis should be the same length')

    if n50<=0:
        raise ValueError('Number of overlapping segments should be greater than 1')

    # remove NaNs
    ys, ts = clean_ts(ys,ts)

    # preprocessing
    ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs,
               gaussianize=gaussianize, standardize=standardize)

    # divide into segments
    nseg=int(np.floor(2*len(ts)/(n50+1)))
    index=np.array(np.arange(0,len(ts),nseg/2),dtype=int)
    if len(index) == n50+2:
        index[-1] = len(ts)
    else:
        index=np.append(index,len(ts)) #make it ends at the time series

    ts_seg=[]
    ys_seg=[]


    if n50>1:
        for idx,i in enumerate(np.arange(0,len(index)-2,1)):
            ts_seg.append(ts[index[idx]:index[idx+2]])
            ys_seg.append(ys[index[idx]:index[idx+2]])
    else:
        ts_seg.append(ts)
        ys_seg.append(ys)

    if freq is None:
        freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
        if 'dt' not in freq_kwargs.keys():
            dt = np.median(np.diff(ts))
            freq_kwargs.update({'dt':dt})
        freq = make_freq_vector(ts_seg[0],
                                method=freq_method,
                                **freq_kwargs)
            #remove zero freq
    if freq[0]==0:
        freq=np.delete(freq,0)

    freq_angular = 2 * np.pi * freq

    psd_seg=[]

    for idx,item in enumerate(ys_seg):
    # calculate the frequency vector if needed
        win=signal.get_window(window,len(ts_seg[idx]))
        scale = len(ts_seg[idx])*2*np.mean(np.diff(ts_seg[idx]))/((win*win).sum())
        psd_seg.append(signal.lombscargle(ts_seg[idx],
                                          item*win,
                                          freq_angular,precenter=True)*scale)
    # average them up
    if average=='mean':
        psd=np.mean(psd_seg,axis=0)
    elif average=='median':
        psd=np.median(psd_seg,axis=0)
    else:
        raise ValueError('Average should either be set to mean or median')

    # Fix possible problems at the edge
    if psd[0]<psd[1]:
        if abs(1-abs(psd[1]-psd[0])/psd[1])<1.e-2:
            # warnings.warn("Unstability at the beginning of freq vector, removing point")
            # psd=psd[1:]
            # freq=freq[1:]
            warnings.warn("Unstability at the beginning of freq vector, setting the point to NaN")
            psd[0] = np.nan
    else:
        if abs(1-abs(psd[0]-psd[1])/psd[0])<1.e-2:
            # warnings.warn("Unstability at the beginning of freq vector, removing point")
            # psd=psd[1:]
            # freq=freq[1:]
            warnings.warn("Unstability at the beginning of freq vector, setting the point to NaN")
            psd[0] = np.nan
    if psd[-1]>psd[-2]:
        if abs(1-abs(psd[-1]-psd[-2])/psd[-1])<1.e-2:
            warnings.warn("Unstability at the end of freq vector, removing point")
            # psd=psd[0:-2]
            # freq=freq[0:-2]
            psd[-1] = np.nan
            psd[-2] = np.nan
    else:
        if abs(1-abs(psd[-2]-psd[-1])/psd[-2])<1.e-2:
            # warnings.warn("Unstability at the end of freq vector, removing point")
            # psd=psd[0:-2]
            # freq=freq[0:-2]
            warnings.warn("Unstability at the end of freq vector, setting the point point to NaN")
            psd[-1] = np.nan
            psd[-2] = np.nan

    # output result
    res_dict = {
        'freq': np.asarray(freq),
        'psd': np.asarray(psd),
    }

    return res_dict


def periodogram(ys, ts, window='hann', nfft=None,
           return_onesided=True, detrend = None, sg_kwargs=None,
           gaussianize=False, standardize=False,
           scaling='density'):
    ''' Spectral density estimation using a Blackman-Tukey periodogram

    Based on the function from scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html

    Parameters
    ----------

    ys : array
        a time series
    ts : array
        time axis of the time series
    window : string or tuple
        Desired window to use. Possible values:
            - boxcar (default)
            - triang
            - blackman
            - hamming
            - hann
            - bartlett
            - flattop
            - parzen
            - bohman
            - blackmanharris
            - nuttail
            - barthann
            - kaiser (needs beta)
            - gaussian (needs standard deviation)
            - general_gaussian (needs power, width)
            - slepian (needs width)
            - dpss (needs normalized half-bandwidth)
            - chebwin (needs attenuation)
            - exponential (needs decay scale)
            - tukey (needs taper fraction)
        If the window requires no parameters, then window can be a string.
        If the window requires parameters, then window must be a tuple with the first argument the string name of the window, and the next arguments the needed parameters.
        If window is a floating point number, it is interpreted as the beta parameter of the kaiser window.
      nfft: int
          Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg
      return_onesided : bool
          If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Defaults to True, but for complex data, a two-sided spectrum is always returned.
      detrend : str
          If None, no detrending is applied. Available detrending methods:
              - None - no detrending will be applied (default);
              - linear - a linear least-squares fit to `ys` is subtracted;
              - constant - the mean of `ys` is subtracted
              - savitzy-golay - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
              - emd - Empirical mode decomposition
      sg_kwargs : dict
          The parameters for the Savitzky-Golay filters. see pyleoclim.utils.filter.savitzy_golay for details.
      gaussianize : bool
          If True, gaussianizes the timeseries
      standardize : bool
          If True, standardizes the timeseries
      scaling : {"density,"spectrum}
          Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz and computing the power spectrum (‘spectrum’) where Pxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density'

    Returns
    -------

    res_dict : dict
        the result dictionary, including
        - freq (array): the frequency vector
        - psd (array): the spectral density vector

    See Also
    --------
    pyleoclim.utils.spectral.welch : Estimate power spectral density using the welch method
    pyleoclim.utils.spectral.mtm : Retuns spectral density using a multi-taper method
    pyleoclim.utils.spectral.lomb_scargle : Return the computed periodogram using lomb-scargle algorithm
    pyleoclim.utils.spectral.wwz_psd : Spectral estimation using the Weighted Wavelet Z-transform
    pyleoclim.utils.spectral.cwt_psd : Spectral estimation using the Continuous Wavelet Transform
    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay
    pyleoclim.utils.tsutils.detrend : detrending functionalities using 4 possible methods  
    pyleoclim.utils.tsutils.gaussianize_1d: Quantile maps a 1D array to a Gaussian distribution 
    pyleoclim.utils.tsutils.standardize: Centers and normalizes a given time series.

    '''
    ts = np.array(ts)
    ys = np.array(ys)

    if len(ts) != len(ys):
        raise ValueError('Time and value axis should be the same length')

        # remove NaNs
    ys, ts = clean_ts(ys,ts)
    # check for evenly-spaced
    check = is_evenly_spaced(ts)
    if check == False:
        raise ValueError('For the Periodogram method, data should be evenly spaced')
    # preprocessing
    ys = preprocess(ys, ts, detrend=detrend, sg_kwargs=sg_kwargs,
               gaussianize=gaussianize, standardize=standardize)

    # calculate sampling frequency fs
    dt = np.median(np.diff(ts))
    fs = 1 / dt

    # spectral analysis
    freq, psd = signal.periodogram(ys, fs, window=window, nfft=nfft,
                                   detrend=False, return_onesided=return_onesided,
                                   scaling=scaling, axis=-1)

    # fix the zero frequency point
    if freq[0] == 0:
        psd[0] = np.nan

    # output result
    res_dict = {
        'freq': np.asarray(freq),
        'psd': np.asarray(psd),
    }

    return res_dict


def wwz_psd(ys, ts, freq=None, freq_method='log', freq_kwargs=None,
            tau=None, c=1e-3, nproc=8,
            detrend=False, sg_kwargs=None, gaussianize=False,
            standardize=False, Neff_threshold=3, anti_alias=False, avgs=2,
            method='Kirchner_numba', wwa=None, wwz_Neffs=None, wwz_freq=None):
    ''' Spectral estimation using the Weighted Wavelet Z-transform
    
    The Weighted wavelet Z-transform (WWZ) is based on Morlet wavelet spectral estimation, using
    least squares minimization to suppress the energy leakage caused by data gaps.
    WWZ does not rely on interpolation or detrending, and is appropriate for unevenly-spaced datasets.
    In particular, we use the variant of Kirchner & Neal (2013), in which basis rotations mitigate the
    numerical instability that occurs in pathological cases with the original algorithm (Foster, 1996).
    The WWZ method has one adjustable parameter, a decay constant `c` that balances the time and frequency
    resolutions of the analysis. The smaller this constant is, the sharper the peaks.
    The default value is 1e-3 to obtain smooth spectra that lend themselves to better scaling exponent
    estimation, while still capturing the main periodicities. 

     Note that scalogram applications use the larger value (8π2)−1, justified elsewhere (Foster, 1996).

    Parameters
    ----------

    ys : array
        a time series, NaNs will be deleted automatically
    ts : array
        the time points, if `ys` contains any NaNs, some of the time points will be deleted accordingly
    freq : array
        vector of frequency
    freq_method : str, {'log', 'lomb_scargle', 'welch', 'scale', 'nfft'}
        Method to generate the frequency vector if not set directly. The following options are avialable:

        - 'log' (default)
        - 'lomb_scargle'
        - 'welch'
        - 'scale'
        - 'nfft'
        See :func:`pyleoclim.utils.wavelet.make_freq_vector` for details

    freq_kwargs : dict
        Arguments for the method chosen in freq_method. See specific functions in pyleoclim.utils.wavelet for details
    tau : array
        the evenly-spaced time vector for the analysis, namely the time shift for wavelet analysis
    c : float
        the decay constant that will determine the analytical resolution of frequency for analysis, the smaller the higher resolution;
        the default value 1e-3 is good for most of the spectral analysis cases
    nproc : int
        the number of processes for multiprocessing

    detrend : str, {None, 'linear', 'constant', 'savitzy-golay'}
        available methods for detrending, including

        - None: the original time series is assumed to have no trend;
        - 'linear': a linear least-squares fit to `ys` is subtracted;
        - 'constant': the mean of `ys` is subtracted
        - 'savitzy-golay': ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.

    sg_kwargs : dict
        The parameters for the Savitzky-Golay filters. See :func:`pyleoclim.utils.filter.savitzky_golay()` for details.
    gaussianize : bool
        If True, gaussianizes the timeseries
    standardize : bool
        If True, standardizes the timeseries

    method : string, {'Foster', 'Kirchner', 'Kirchner_f2py', 'Kirchner_numba'}
        available specific implementation of WWZ, including

        - 'Foster': the original WWZ method;
        - 'Kirchner': the method Kirchner adapted from Foster;
        - 'Kirchner_f2py':  the method Kirchner adapted from Foster, implemented with f2py for acceleration;
        - 'Kirchner_numba':  the method Kirchner adapted from Foster, implemented with Numba for acceleration (default);

    Neff_threshold : int
        threshold for the effective number of points
    anti_alias : bool
        If True, uses anti-aliasing
    avgs : int
        flag for whether spectrum is derived from instantaneous point measurements (avgs<>1)
        OR from measurements averaged over each sampling interval (avgs==1)

    wwa : array
        the weighted wavelet amplitude, returned from pyleoclim.utils.wavelet.wwz

    wwz_Neffs : array
        the matrix of effective number of points in the time-scale coordinates,
        returned from pyleoclim.utils.wavelet.wwz

    wwz_freq : array
        the returned frequency vector from pyleoclim.utils.wavelet.wwz

    Returns
    -------

    res : namedtuple
        a namedtuple that includes below items

        psd : array
            power spectral density
        freq : array
            vector of frequency

    See Also
    --------
    pyleoclim.utils.spectral.periodogram : Estimate power spectral density using a periodogram
    pyleoclim.utils.spectral.mtm : Retuns spectral density using a multi-taper method
    pyleoclim.utils.spectral.lomb_scargle : Return the computed periodogram using lomb-scargle algorithm
    pyleoclim.utils.spectral.welch : Estimate power spectral density using the Welch method
    pyleoclim.utils.spectral.cwt_psd : Spectral estimation using the Continuous Wavelet Transform
    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay
    pyleoclim.utils.tsutils.detrend : detrending functionalities using 4 possible methods  
    pyleoclim.utils.tsutils.gaussianize_1d: Quantile maps a 1D array to a Gaussian distribution 
    pyleoclim.utils.tsutils.standardize: Centers and normalizes a given time series.

    References
    ----------
    - Foster, G. (1996). Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal, 112(4), 1709-1729.
    - Kirchner, J. W. (2005). Aliasin in 1/f^a noise spectra: origins, consequences, and remedies. Physical Review E covering statistical, nonlinear, biological, and soft matter physics, 71, 66110.
    - Kirchner, J. W. and Neal, C. (2013). Universal fractal scaling in stream chemistry and its impli-cations for solute transport and water quality trend detection. Proc Natl Acad Sci USA 110:12213–12218.
    '''
    ys_cut, ts_cut, freq, tau = prepare_wwz(ys, ts, freq=freq,
                                            freq_method=freq_method,
                                            freq_kwargs=freq_kwargs,tau=tau)

    # get wwa but AR1_q is not needed here so set nMC=0
    #  wwa, _, _, coi, freq, _, Neffs, _ = wwz(ys_cut, ts_cut, freq=freq, tau=tau, c=c, nproc=nproc, nMC=0,
    if wwa is None or wwz_Neffs is None or wwz_freq is None:
        res_wwz = wwz(ys_cut, ts_cut, freq=freq, tau=tau, c=c, nproc=nproc,
                  detrend=detrend, sg_kwargs=sg_kwargs,
                  gaussianize=gaussianize, standardize=standardize, method=method)
        wwa = res_wwz.amplitude
        wwz_Neffs = res_wwz.Neffs
        wwz_freq = res_wwz.freq
        
    psd = wwa2psd(wwa, ts_cut, wwz_Neffs, freq=wwz_freq, Neff_threshold=Neff_threshold, anti_alias=anti_alias, avgs=avgs)
    Results = collections.namedtuple('Results', ['psd', 'freq'])
    res = Results(psd=psd, freq=freq)

    return res

def cwt_psd(ys, ts, freq=None, freq_method='log', freq_kwargs=None,scale = None, 
            detrend=False,sg_kwargs={}, gaussianize=False, standardize =False, pad=False, 
            mother='MORLET',param=None, cwt_res=None):
    ''' Spectral estimation using the continuous wavelet transform
    Uses the Torrence and Compo [1998] continuous wavelet transform implementation

    Parameters
    ----------
    ys : numpy.array
        the time series.
    ts : numpy.array
        the time axis.
    freq : numpy.array, optional
        The frequency vector. The default is None, which will prompt the use of one the underlying functions
    freq_method : string, optional
        The method by which to obtain the frequency vector. The default is 'log'.
        Options are 'log' (default), 'nfft', 'lomb_scargle', 'welch', and 'scale'
    freq_kwargs : dict, optional
        Optional parameters for the choice of the frequency vector. See make_freq_vector and additional methods for details. The default is {}.
    scale : numpy.array
        Optional scale vector in place of a frequency vector. Default is None. If scale is not None, frequency method and attached arguments will be ignored. 
    detrend : bool, string, {'linear', 'constant', 'savitzy-golay', 'emd'}
        Whether to detrend and with which option. The default is False.
    sg_kwargs : dict, optional
        Additional parameters for the savitzy-golay method. The default is {}.
    gaussianize : bool, optional
        Whether to gaussianize. The default is False.
    standardize : bool, optional
        Whether to standardize. The default is False.     
    pad : bool, optional
        Whether or not to pad the timeseries. with zeroes to get N up to the next higher power of 2. 
        This prevents wraparound from the end of the time series to the beginning, and also speeds up the FFT's used to do the wavelet transform.
        This will not eliminate all edge effects. The default is False.
    mother : string, optional
        the mother wavelet function. The default is 'MORLET'. Options are: 'MORLET', 'PAUL', or 'DOG'
    param : flaot, optional
        the mother wavelet parameter. The default is None since it varies for each mother
            - For 'MORLET' this is k0 (wavenumber), default is 6.
            - For 'PAUL' this is m (order), default is 4.
            - For 'DOG' this is m (m-th derivative), default is 2.
    cwt_res : dict
        Results from pyleoclim.utils.wavelet.cwt

    Returns
    -------
    res : dict
        Dictionary containing:
            - psd: the power density function
            - freq: frequency vector
            - scale: the scale vector
            - mother: the mother wavelet
            - param : the wavelet parameter
            
    See also
    --------
    
    pyleoclim.utils.wavelet.make_freq_vector : make the frequency vector with various methods    
    pyleoclim.utils.wavelet.cwt: Torrence and Compo implementation of the continuous wavelet transform 
    pyleoclim.utils.spectral.periodogram : Spectral estimation using Blackman-Tukey's periodogram
    pyleoclim.utils.spectral.mtm : Spectral estimation using the multi-taper method
    pyleoclim.utils.spectral.lomb_scargle : Spectral estimation using the lomb-scargle periodogram
    pyleoclim.utils.spectral.welch : Spectral estimation using Welch's periodogram
    pyleoclim.utils.spectral.wwz_psd : Spectral estimation using the Weighted Wavelet Z-transform
    pyleoclim.utils.tsutils.detrend : detrending functionalities using 4 possible methods  
    pyleoclim.utils.tsutils.gaussianize_1d: Quantile maps a 1D array to a Gaussian distribution 
    pyleoclim.utils.tsutils.standardize: Centers and normalizes a given time series.
    
    References
    ----------
    
    Torrence, C. and G. P. Compo, 1998: A Practical Guide to Wavelet Analysis. Bull. Amer. Meteor. Soc., 79, 61-78.
    Python routines available at http://paos.colorado.edu/research/wavelets/
    
    '''
    
    
        #get the wavelet:
    if cwt_res is None:
        cwt_res = cwt(ys,ts,freq=freq, freq_method=freq_method, freq_kwargs=freq_kwargs,
              scale = scale, detrend=detrend,sg_kwargs=sg_kwargs, gaussianize=gaussianize, 
              standardize = standardize, pad=pad, mother=mother, param=param) 
        n= len(ts)
    else:
        n=len(cwt_res.time)
    
    psd = np.sum(cwt_res.amplitude.T**2,axis=1)/n
    
    
    Results = collections.namedtuple('Results', ['psd', 'freq','scale','mother','param'])
    res = Results(psd=psd, freq=cwt_res.freq, scale=cwt_res.scale, mother=cwt_res.mother,param=cwt_res.param)

    return res

def beta_estimation(psd, freq, fmin=None, fmax=None, logf_binning_step='max', verbose=False):
    ''' Estimate the scaling exponent of a power spectral density.
    
    Models the spectrum as :math: `S(f) \propto 1/f^{\beta}`. For instance:
    - :math: `\beta = 0` corresponds to white noise
    - :math: `\beta = 1` corresponds to pink noise
    - :math: `\beta = 2` corresponds to red noise (Brownian motion)

    
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

def beta2Hurst(beta):
    ''' Translates spectral slope to Hurst exponent

    Parameters
    ----------

    beta : float
        the estimated slope of a power spectral density :math: `S(f) \propto 1/f^{\beta}`
        
    Returns
    -------

    H : float
        Hurst index, should be in (0, 1)

    References
    ----------

    Equation 2 in http://www.bearcave.com/misl/misl_tech/wavelets/hurst/
    
    See also
    --------
    pyleoclim.utils.spectral.beta_estimation: Estimate the scaling exponent of a power spectral density.
    
    '''
    H = (beta-1)/2

    return H

def psd_ar(var_noise, freq, ar_params, f_sampling):
    ''' Theoretical power spectral density (PSD) of an autoregressive model

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



def psd_fBM(freq, ts, H):
    ''' Theoretical power spectral density of a fractional Brownian motion

    Parameters
    ----------

    freq : array
        vector of frequency
    ts : array
        the time axis of the time series
    H : float
        Hurst exponent, should be in (0, 1)

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
