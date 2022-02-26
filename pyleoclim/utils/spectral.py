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
import warnings

__all__ = [
    'wwz_psd',
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
)
#from .tsutils import clean_ts, interp, bin

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
    '''Estimate power spectral density using Welch's method

    Wrapper for the function implemented in scipy.signal.welch
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html for details.

    Welch's method is an approach for spectral density estimation. It computes an estimate of the power spectral density by dividing the data into overlapping segments, computing a modified periodogram for each segment and averaging the periodograms.

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
          Method to use when averaging periodograms. Defaults to ‘mean’.

    Returns
    -------
    res_dict : dict
        the result dictionary, including
        - freq (array): the frequency vector
        - psd (array): the spectral density vector


    See also
    --------
    pyleoclim.utils.spectral.periodogram : Estimate power spectral density using a periodogram
    pyleoclim.utils.spectral.mtm : Retuns spectral density using a multi-taper method
    pyleoclim.utils.spectral.lomb_scargle : Return the computed periodogram using lomb-scargle algorithm
    pyleoclim.utils.spectral.wwz_psd : Return the psd of a timeseries using wwz method.
    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay
    pyleoclim.utils.tsutils.detrend : Detrending method

    References
    ----------
    P. Welch, “The use of the fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms”, IEEE Trans. Audio Electroacoust. vol. 15, pp. 70-73, 1967.

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
    ''' Retuns spectral density using a multi-taper method.


    Based on the function in the time series analysis for neuroscience toolbox: http://nipy.org/nitime/api/generated/nitime.algorithms.spectral.html

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
    pyleoclim.utils.spectral.periodogram : Estimate power spectral density using a periodogram
    pyleoclim.utils.spectral.welch : Retuns spectral density using the welch method
    pyleoclim.utils.spectral.lomb_scargle : Return the computed periodogram using lomb-scargle algorithm
    pyleoclim.utils.spectral.wwz_psd : Return the psd of a timeseries using wwz method.
    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay
    pyleoclim.utils.tsutils.detrend : Detrending method

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
    """ Return the computed periodogram using lomb-scargle algorithm

    Uses the lombscargle implementation from scipy.signal: https://scipy.github.io/devdocs/generated/scipy.signal.lombscargle.html#scipy.signal.lombscargle

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
    pyleoclim.utils.spectral.wwz_psd : Return the psd of a timeseries using wwz method.
    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay
    pyleoclim.utils.tsutils.detrend : Detrending method

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
    ''' Estimate power spectral density using a periodogram

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
    pyleoclim.utils.spectral.wwz_psd : Return the psd of a timeseries using wwz method.
    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay
    pyleoclim.utils.tsutils.detrend : Detrending method

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
            standardize=False, Neff=3, anti_alias=False, avgs=2,
            method='Kirchner_numba', wwa=None, wwz_Neffs=None, wwz_freq=None):
    ''' Returns the power spectral density (PSD) of a timeseries using the Weighted Wavelet Z-transform

    The Weighted wavelet Z-transform (WWZ) is based on Morlet wavelet spectral estimation, using
    least squares minimization to suppress the energy leakage caused by the data gaps.
    WWZ does not rely on interpolation or detrending, and is appropriate for unevenly-spaced datasets.
    In particular, we use the variant of Kirchner & Neal (2013), in which basis rotations mitigate the
    numerical instability that occurs in pathological cases with the original algorithm (Foster, 1996).
    The WWZ method has one adjustable parameter, a decay constant `c` that balances the time and frequency
    resolutions of the analysis. The smaller this constant is, the sharper the peaks.
     We choose the value 1e-3 to obtain smooth spectra that lend themselves to better scaling exponent estimation,
     while still capturing the main periodicities.

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

    Neff : int
        effective number of points
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
    pyleoclim.utils.filter.savitzy_golay : Filtering using Savitzy-Golay
    pyleoclim.utils.tsutils.detrend : Detrending method

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
        res_wwz = wwz(ys_cut, ts_cut, freq=freq, tau=tau, c=c, nproc=nproc, nMC=0,
                  detrend=detrend, sg_kwargs=sg_kwargs,
                  gaussianize=gaussianize, standardize=standardize, method=method)
        wwa = res_wwz.amplitude
        wwz_Neffs = res_wwz.Neffs
        wwz_freq = res_wwz.freq

    psd = wwa2psd(wwa, ts_cut, wwz_Neffs, freq=wwz_freq, Neff=Neff, anti_alias=anti_alias, avgs=avgs)
    #  psd[1/freqs > np.max(coi)] = np.nan  # cut off the unreliable part out of the coi
    #  psd = psd[1/freqs <= np.max(coi)] # cut off the unreliable part out of the coi
    #  freqs = freqs[1/freqs <= np.max(coi)]

    # Monte-Carlo simulations of AR1 process
    #nf = np.size(freq)

    #  psd_ar1 = np.ndarray(shape=(nMC, nf))

    #  if nMC >= 1:
        #  #  tauest = wa.tau_estimation(ys_cut, ts_cut, detrend=detrend)

        #  for i in tqdm(range(nMC), desc='Monte-Carlo simulations'):
            #  #  r = wa.ar1_model(ts_cut, tauest)
            #  r = ar1_sim(ys_cut, np.size(ts_cut), 1, ts=ts_cut)
            #  res_red = wwz(r, ts_cut, freq=freq, tau=tau, c=c, nproc=nproc, nMC=0,
                                                                     #  detrend=detrend, params=params,
                                                                     #  gaussianize=gaussianize, standardize=standardize,
                                                                     #  method=method)
            #  psd_ar1[i, :] = wa.wwa2psd(res_red.wwa, ts_cut, res_red.Neffs,
                                       #  freq=res_red.freq, Neff=Neff, anti_alias=anti_alias, avgs=avgs)
            #  #  psd_ar1[i, 1/freqs_red > np.max(coi_red)] = np.nan  # cut off the unreliable part out of the coi
            #  #  psd_ar1 = psd_ar1[1/freqs_red <= np.max(coi_red)] # cut off the unreliable part out of the coi

        #  psd_ar1_q95 = mquantiles(psd_ar1, 0.95, axis=0)[0]

    #  else:
        #  psd_ar1_q95 = None

    # Results = collections.namedtuple('Results', ['psd', 'freq', 'psd_ar1_q95', 'psd_ar1'])
    # res = Results(psd=psd, freq=freq, psd_ar1_q95=psd_ar1_q95, psd_ar1=psd_ar1)
    Results = collections.namedtuple('Results', ['psd', 'freq'])
    res = Results(psd=psd, freq=freq)

    return res
