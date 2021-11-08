#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 09:52:10 2021

@author: deborahkhider

Basic functionalities to clean timeseries data. 
"""

import numpy as np
from typing import OrderedDict

#APIs
__all__ = [
    'clean_ts',
    'dropna',
    'sort_ts',
    'is_evenly_spaced',
    'reduce_duplicated_timestamps',
]


def clean_ts(ys, ts, verbose=False):
    ''' Cleaning the timeseries

    Delete the NaNs in the time series and sort it with time axis ascending,
    duplicate timestamps will be reduced by averaging the values.

    Parameters
    ----------
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
    ys, ts = dropna(ys, ts, verbose=verbose)
    ys, ts = sort_ts(ys, ts, verbose=verbose)
    ys, ts = reduce_duplicated_timestamps(ys, ts, verbose=verbose)

    return ys, ts


def dropna(ys, ts, verbose=False):
    ''' Remove entries of ys or ts that bear NaNs

    Parameters
    ----------
    ys : array
        A time series, NaNs allowed
    ts : array
        The time axis of the time series, NaNs allowed
    verbose : bool
        If True, will print a warning message

    Returns
    -------
    ys : array
        The time series without nans
    ts : array
        The time axis of the time series without nans

    '''
    ys = np.asarray(ys, dtype=float)
    ts = np.asarray(ts, dtype=float)
    assert ys.size == ts.size, 'The size of time axis and data value should be equal!'

    ys_tmp = np.copy(ys)
    ys = ys[~np.isnan(ys_tmp)]
    ts = ts[~np.isnan(ys_tmp)]
    ts_tmp = np.copy(ts)
    ys = ys[~np.isnan(ts_tmp)]
    ts = ts[~np.isnan(ts_tmp)]

    if verbose and any(np.isnan(ys_tmp)):
        print('NaNs have been detected and dropped.')

    return ys, ts

def sort_ts(ys, ts, verbose=False):
    ''' Sort ts values in ascending order

    Parameters
    ----------
    ys : array
        Dependent variable
    ts : array
        Independent variable
    verbose : bool
        If True, will print a warning message

    Returns
    -------
    ys : array
        Dependent variable
    ts : array
        Independent variable, sorted in ascending order

    '''
    ys = np.asarray(ys, dtype=float)
    ts = np.asarray(ts, dtype=float)
    assert ys.size == ts.size, 'time and value arrays must be of equal length'

    # sort the time series so that the time axis will be ascending
    dt = np.median(np.diff(ts))
    if dt < 0:
        sort_ind = np.argsort(ts)
        ys = ys[sort_ind]
        ts = ts[sort_ind]
        if verbose:
            print('The time axis has been adjusted to be prograde')

    return ys, ts

def reduce_duplicated_timestamps(ys, ts, verbose=False):
    ''' Reduce duplicated timestamps in a timeseries by averaging the values

    Parameters
    ----------
    ys : array
        Dependent variable
    ts : array
        Independent variable
    verbose : bool
        If True, will print a warning message

    Returns
    -------
    ys : array
        Dependent variable
    ts : array
        Independent variable, with duplicated timestamps reduced by averaging the values

    '''
    ys = np.asarray(ys, dtype=float)
    ts = np.asarray(ts, dtype=float)
    assert ys.size == ts.size, 'The size of time axis and data value should be equal!'

    if len(ts) != len(set(ts)):
        value = OrderedDict()
        for t, y in zip(ts, ys):
            if t not in value:
                value[t] = [y]
            else:
                value[t].append(y)

        ts = []
        ys = []
        for k, v in value.items():
            ts.append(k)
            ys.append(np.mean(v))

        ts = np.array(ts)
        ys = np.array(ys)

        if verbose:
            print('Duplicate timestamps have been combined by averaging values.')
    return ys, ts

def is_evenly_spaced(ts, tol=1e-4):
    ''' Check if a time axis is evenly spaced, within a given tolerance

    Parameters
    ----------

    ts : array
        the time axis of a time series
        
    tol : float64
        numerical tolerance for the relative difference

    Returns
    -------

    check : bool
        True - evenly spaced; False - unevenly spaced.

    '''
    if ts is None:
        check = True
    else:
        dts = np.diff(ts)
        dt_mean = dts.mean()   
        check = all(np.abs((dt - dt_mean)/dt_mean) < tol for dt in np.diff(ts)) # compare relative spacing to the mean
        
    return check


