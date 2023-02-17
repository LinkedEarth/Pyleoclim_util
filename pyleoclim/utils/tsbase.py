#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic functionalities to clean timeseries data prior to analysis
"""

import numpy as np
from typing import OrderedDict
import operator
import warnings
import pandas as pd


#APIs

__all__ = [
    'clean_ts',
    'dropna',
    'sort_ts',
    'is_evenly_spaced',
    'reduce_duplicated_timestamps',
]

SECONDS_PER_YEAR = 365.25 * 60  * 60 * 24 ## TODO: generalize to different calendars using cftime

MATCH_A  = frozenset(['y', 'yr', 'yrs', 'year', 'years'])
MATCH_KA = frozenset(['ka', 'ky', 'kyr', 'kyrs', 'kiloyear', 'kiloyr', 'kiloyrs']) 
MATCH_MA = frozenset(['ma', 'my','myr','myrs'])
MATCH_GA = frozenset(['ga', 'gy', 'gyr', 'gyrs'])

def time_unit_to_datum_exp_dir(time_unit, time_name=None, verbose=False):
         
    tu = time_unit.lower().split()
    
    # deal with statements explicit about exponents, and take a first guess at datum/direction        
    if tu[0] in MATCH_A:
        exponent = 0  
        datum = 0
        direction = 'prograde'
    elif tu[0] in MATCH_KA:
        datum = 1950
        exponent = 3
        direction = 'retrograde'
    elif tu[0] in MATCH_MA:
        datum = 1950
        exponent = 6
        direction = 'retrograde'
    elif tu[0] in MATCH_GA:
        datum = 1950
        exponent = 9
        direction = 'retrograde'
    elif tu[0].replace('.','') in ['ad', 'ce']:
        exponent = 0  
        datum = 0
        direction = 'prograde'    
    else:
        warnings.warn(f'Time unit "{time_unit}" unknown; triggering defaults', stacklevel=4)    
        exponent = 0  
        datum = 0
        direction = 'prograde'
    
    # if provided, deal with statements about datum/direction, like kyr BP, years CE, etc
    if len(tu) > 1: 
        datum_str = tu[1].replace('.','') # make lowercase + strip stops, so "B.P." --> "bp"
        if datum_str == 'b2k':
            datum = 2000
            direction = 'retrograde'
        elif datum_str in ['bp', 'bnf', 'b1950']:
            datum = 1950
            direction = 'retrograde'
        elif datum_str in ['ad', 'ce']:
            datum = 0
            direction = 'prograde'
    
    if time_name is not None:
        if time_name.lower() == 'age':
            direction = 'retrograde'
        
    if verbose:
        print(f'Provided time medata translated to {direction} flow, 10^{exponent} year units, and year {datum} datum')    
  
    return (datum, exponent, direction)

def convert_datetime_index_to_time(datetime_index, time_unit, time_name):
    datum, exponent, direction = time_unit_to_datum_exp_dir(time_unit, time_name)
    #import operator
    if direction == 'prograde':
        multiplier = 1
    elif direction == 'retrograde':
        multiplier = -1
    else:
        raise ValueError(f'Expected one of {"prograde", "retrograde"}, got {direction}')
        
    if not isinstance(datetime_index, pd.DatetimeIndex): 
        raise ValueError('The provided index is not a proper DatetimeIndex object')
    if datetime_index.unit != 's':
        raise ValueError(
            "Only 'second' resolution is currently supported. "
            "Please cast to second resolution with `.as_unit('s')`"
        )
    year_diff = (datetime_index.year - int(datum))
    numpy_datetime_index = datetime_index.to_numpy()
    years_floor = numpy_datetime_index.astype('datetime64[Y]').astype('datetime64[s]')
    seconds_diff = (numpy_datetime_index - years_floor).astype('int')
    diff = year_diff + seconds_diff / SECONDS_PER_YEAR
    time = multiplier * diff / 10**exponent

    return time

def time_to_datetime(time, datum=0, exponent=0, direction='prograde', unit='s'):
    '''
    Converts a vector of time values to a pandas datetime object

    Parameters
    ----------
    time : array-like
        the time axis to be converted
    datum : int, optional
        origin point for the time scale. The default is 0.
    exponent : int, optional
        Base-10 exponent for year multiplier. Dates in kyr should use 3, dates in Myr should use 6, etc.
        The default is 0. 
    direction : str, optional
        Direction of time flow, 'prograde' [default] or 'retrograde'.
    unit : str, optional
        Units of the datetime. Default is 's', corresponding to seconds. 
        Only change if you have an excellent reason to use finer resolution!

    Returns
    -------
    index, a datetime64[unit] object

    '''
    if direction == 'prograde':
        op = operator.add
    elif direction == 'retrograde':
        op = operator.sub
    else:
        raise ValueError(f'Expected one of {"prograde", "retrograde"}, got {direction}')
    
    timedelta = np.array(time) * 10**exponent
    years = timedelta.astype('int')
    seconds = ((timedelta - timedelta.astype('int')) * SECONDS_PER_YEAR).astype('timedelta64[s]') # incorporate unit here
    index = op(op(int(datum), years).astype(str).astype('datetime64[s]'), seconds)  # incorporate unit here?
    
    return index


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
    verbose : bool
        If True, will print a warning message

    Returns
    -------
    ys : array
        The time series without nans
    ts : array
        The time axis of the time series without nans

    See also
    --------

    pyleoclim.utils.tsbase.dropna : Drop NaN values

    pyleoclim.utils.tsbase.sort_ts : Sort timeseries

    pyleoclim.utils.tsbase.reduce_duplicated_timestamps : Consolidate duplicated timestamps

    '''
    ys, ts = dropna(ys, ts, verbose=verbose)
    ys, ts = sort_ts(ys, ts, verbose=verbose)
    ys, ts = reduce_duplicated_timestamps(ys, ts, verbose=verbose)

    return ys, ts


def dropna(ys, ts, verbose=False):
    '''Drop NaN values
    
    Remove entries of ys or ts that bear NaNs

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
    ''' Sort timeseries
    
    Sort ts values in ascending order

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
    ''' Consolidate duplicated timestamps
    
    Reduce duplicated timestamps in a timeseries by averaging the values

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
        The time axis of a time series
        
    tol : float64
        Numerical tolerance for the relative difference

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


