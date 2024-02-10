#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:45:50 2024

@author: deborahkhider

Utilities for the `emd` function.

Copyright: pyhht, https://github.com/jaidevd/pyhht/blob/dev/pyhht/utils.py#L66

Copyright notice:
    
Copyright (c) 2007–2017 The PyHHT developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the PyHHT Developers nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission. 


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

"""

import numpy as np
from scipy.signal import argrelmax, argrelmin
from scipy import interpolate


def inst_freq(x, t=None):
    """
    Compute the instantaneous frequency of an analytic signal at specific time
    instants using the trapezoidal integration rule.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        The input analytic signal.
    t : array-like, shape (n_samples,), optional
        The time instants at which to calculate the instantaneous frequency.
        Defaults to `np.arange(2, n_samples)`

    Returns
    -------
    array-like
        Normalized instantaneous frequencies of the input signal

    """
    if x.ndim != 1:
        if 1 not in x.shape:
            raise TypeError("Input should be a one dimensional array.")
        else:
            x = x.ravel()
    if t is not None:
        if t.ndim != 1:
            if 1 not in t.shape:
                raise TypeError("Time instants should be a one dimensional "
                                "array.")
            else:
                t = t.ravel()
    else:
        t = np.arange(2, len(x))

    fnorm = 0.5 * (np.angle(-x[t] * np.conj(x[t - 2])) + np.pi) / (2 * np.pi)
    return fnorm, t


def boundary_conditions(signal, time_samples, z=None, nbsym=2):
    """
    Extend a 1D signal by mirroring its extrema on either side.

    Parameters
    ----------
    signal : array-like, shape (n_samples,)
        The input signal.
    time_samples : array-like, shape (n_samples,)
        Timestamps of the signal samples
    z : array-like, shape (n_samples,), optional
        A proxy signal on whose extrema the interpolation is evaluated.
        Defaults to `signal`.
    nbsym : int, optional
        The number of extrema to consider on either side of the signal.
        Defaults to 2

    Returns
    -------
    tuple
        A tuple of four arrays which represent timestamps of the minima of the
        extended signal, timestamps of the maxima of the extended signal,
        minima of the extended signal and maxima of the extended signal.
        signal, minima of the extended signal and maxima of the extended
        signal.

    """
    tmax = argrelmax(signal)[0]
    maxima = signal[tmax]
    tmin = argrelmin(signal)[0]
    minima = signal[tmin]

    if tmin.shape[0] + tmax.shape[0] < 3:
        raise ValueError("Not enough extrema.")

    loffset_max = time_samples[tmax[:nbsym]] - time_samples[0]
    roffset_max = time_samples[-1] - time_samples[tmax[-nbsym:]]
    new_tmax = np.r_[time_samples[0] - loffset_max[::-1],
                     time_samples[tmax], roffset_max[::-1] + time_samples[-1]]
    new_vmax = np.r_[maxima[:nbsym][::-1], maxima, maxima[-nbsym:][::-1]]

    loffset_min = time_samples[tmin[:nbsym]] - time_samples[0]
    roffset_min = time_samples[-1] - time_samples[tmin[-nbsym:]]

    new_tmin = np.r_[time_samples[0] - loffset_min[::-1],
                     time_samples[tmin], roffset_min[::-1] + time_samples[-1]]
    new_vmin = np.r_[minima[:nbsym][::-1], minima, minima[-nbsym:][::-1]]
    return new_tmin, new_tmax, new_vmin, new_vmax


def get_envelops(x, t=None):
    """
    Get the upper and lower envelopes of an array, as defined by its extrema.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        The input array.
    t : array-like, shape (n_samples,), optional
        Timestamps of the signal. Defaults to `np.arange(n_samples,)`

    Returns
    -------
    tuple
        A tuple of arrays representing the upper and the lower envelopes
        respectively.

    """
    if t is None:
        t = np.arange(x.shape[0])
    maxima = argrelmax(x)[0]
    minima = argrelmin(x)[0]

    # consider the start and end to be extrema

    ext_maxima = np.zeros((maxima.shape[0] + 2,), dtype=int)
    ext_maxima[1:-1] = maxima
    ext_maxima[0] = 0
    ext_maxima[-1] = t.shape[0] - 1

    ext_minima = np.zeros((minima.shape[0] + 2,), dtype=int)
    ext_minima[1:-1] = minima
    ext_minima[0] = 0
    ext_minima[-1] = t.shape[0] - 1

    tck = interpolate.splrep(t[ext_maxima], x[ext_maxima])
    upper = interpolate.splev(t, tck)
    tck = interpolate.splrep(t[ext_minima], x[ext_minima])
    lower = interpolate.splev(t, tck)
    return upper, lower


def extr(x):
    """
    Extract the indices of the extrema and zero crossings.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Input signal.

    Returns
    -------
    tuple
        A tuple of three arrays representing the minima, maxima and zero
        crossings of the signal respectively.

    """
    m = x.shape[0]

    x1 = x[:m - 1]
    x2 = x[1:m]
    indzer = np.where(x1 * x2 < 0)[0]
    if np.any(x == 0):
        iz = np.where(x == 0)[0]
        indz = []
        if np.any(np.diff(iz) == 1):
            zer = x == 0
            dz = np.diff(np.r_[0, zer, 0])
            debz = np.where(dz == 1)[0]
            finz = np.where(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2)
        else:
            indz = iz
        indzer = np.sort(np.hstack([indzer, indz]))

    indmax = argrelmax(x)[0]
    indmin = argrelmin(x)[0]

    return indmin, indmax, indzer


