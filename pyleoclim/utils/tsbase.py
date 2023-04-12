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
import scipy.stats as st

#APIs

__all__ = [
    'clean_ts',
    'dropna',
    'sort_ts',
    'is_evenly_spaced',
    'reduce_duplicated_timestamps',
]

# UDUNITS, see: http://cfconventions.org/cf-conventions/cf-conventions#time-coordinate
SECONDS_PER_YEAR = 31556925.974592  # 86400 * 365.24219878

MATCH_A  = frozenset(['y', 'yr', 'yrs', 'year', 'years'])
MATCH_KA = frozenset(['ka', 'ky', 'kyr', 'kyrs', 'kiloyear', 'kiloyr', 'kiloyrs'])
MATCH_MA = frozenset(['ma', 'my','myr','myrs'])
MATCH_GA = frozenset(['ga', 'gy', 'gyr', 'gyrs'])

def time_unit_to_datum_exp_dir(time_unit, time_name=None, verbose=False):
    """Convert time unit (yr, ka, ma, etc) to a datum, exponent, direction 
    triplet. Based on the time_unit (and optionally, the time_name) the datum
    (year zero), exponent (10^x year units), and direction (prograde/retrograde)
    can be inferred. A verbose option is included here for users who want to 
    confirm the resulting inference.

    Parameters
    ----------
    time_unit: str
        Time unit indicates the major unit of time. Examples: annum (yr), 
        kiloyear (ka, ky), milayear (ma, my), gigayear (ga, gy)
    time_name: str
        (Optional) If 'age', direction is always 'retrograde'. Defaults to None,
        which is effectively unused.
    verbose: bool
        (Optional) If True, includes a print statement explaining the 
        conversion.

    Returns
    -------
    datum : int, optional
        origin point for the time scale.
    exponent : int, optional
        Base-10 exponent for year multiplier. Dates in kyr should use 3, dates in Myr should use 6, etc.
    direction: str
        Direction of time flow, 'prograde' or 'retrograde'.

    Examples
    --------
    >>> from pyleoclim.utils.tsbase import time_unit_to_datum_exp_dir
    >>> (datum, exponent, direction) = time_unit_to_datum_exp_dir(time_unit)
    (1950, 3, 'retrograde')
    """

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
        print(f'Provided time metadata translated to {direction} flow, 10^{exponent} year units, and year {datum} datum')    
  
    return (datum, exponent, direction)

def convert_datetime_index_to_time(datetime_index, time_unit, time_name):
    """ Convert a Pandas DatetimeIndex into a numpy array of floats.
    
    The general formula is:

        datetime_index = datum +/- time*10**exponent
    
    where we assume ``time`` to use the Gregorian calendar. If dealing with other
    calendars, then conversions need to happen before reaching pyleoclim.

    Parameters
    ----------
    datetime_index: pd.DatetimeIndex
        Index to covert to floats
    time_unit: str
        Time unit indicates the major unit of time. Examples: annum (yr), 
        kiloyear (ka, ky), milayear (ma, my), gigayear (ga, gy)
    time_name: str
        If 'age', direction is always 'retrograde'. 

    Returns
    -------
    np.array((float,)) of converted times

    Examples
    --------
    >>> from pyleoclim.utils.tsbase import convert_datetime_index_to_time
    >>> time_unit = 'ga'
    >>> time_name = None
    >>> dti = pd.date_range("2018-01-01", periods=5, freq="Y", unit='s')
    >>> df = pd.DataFrame(np.array(range(5)), index=dti)
    >>> time = convert_datetime_index_to_time(
        df.index, 
        time_unit, 
        time_name=time_name,
        )
    np.array([-6.89965777e-08, -6.99965777e-08, -7.09993155e-08, -7.19965777e-08, -7.29965777e-08])

    """
    datum, exponent, direction = time_unit_to_datum_exp_dir(time_unit, time_name)
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
    
    time = (
        multiplier * (datetime_index.to_numpy() - np.datetime64(str(datum), "s"))
    ).astype(float) / (10**exponent * SECONDS_PER_YEAR)
    return pd.Index(time)


def time_to_datetime(time, datum=0, exponent=0, direction='prograde', unit='s'):
    '''
    Converts a vector of time values to a pandas datetime object

    The general formula is:

        datetime_index = datum +/- time*10**exponent
    
    where we assume ``time`` to use the Gregorian calendar. If dealing with other
    calendars, then conversions need to happen before reaching pyleoclim.

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
    if direction not in ('prograde', 'retrograde'):
        raise ValueError(f'Expected one of {"prograde", "retrograde"}, got {direction}')
    
    if direction == 'prograde':
        op = operator.add
    elif direction == 'retrograde':
        op = operator.sub
        
    index = op(
        np.datetime64(str(datum), 's'),
        (time*SECONDS_PER_YEAR*10**exponent).astype('timedelta64[s]')
    )
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

    See Also
    --------
    https://pandas.pydata.org/docs/reference/api/pandas.Series.dropna.html

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

def sort_ts(ys, ts, ascending = True, verbose=False):
    ''' Sort timeseries

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

    sort_ind = np.argsort(ts)

    ys = ys[sort_ind]
    ts = ts[sort_ind]

    if ascending:
        if verbose:
            print('Time axis values sorted in ascending order')
    else:
        ys = ys[::-1] # flip the series
        ts = ts[::-1]
        if verbose:
            print('Time axis values sorted in descending order')

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

def is_evenly_spaced(x, tol=1e-4):
    ''' Check if an axis x is evenly spaced, within a given tolerance

    Parameters
    ----------

    x : array

    tol : float64
        Numerical tolerance for the relative difference

    Returns
    -------

    check : bool
        True - evenly spaced; False - unevenly spaced.

    '''
    if x is None:
        check = True
    else:
        dx = np.diff(x)
        dx_mean = dx.mean()
        check = all(np.abs((dx - dx_mean)/dx_mean) < tol for dx in np.diff(x)) # compare relative spacing to the mean

    return check

def resolution(x):
    '''
    Computes the resolution (increments) of an axis, and returns its descriptive statistics

    Parameters
    ----------
    x : array

    Returns
    -------
    res : array
        array of time increments

    stats : DescribeResult
        descriptive statistics of res
        
    sign : str
        sign of the resolution    
        'positive' if all values of res are > 0
        'negative' if all values of res are < 0
        'mixed' otherwise 

    See Also
    --------

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html
    '''
    res = np.diff(x)
    stats = st.describe(res)
        
    if all(res > 0):
        sign = 'positive'
    elif all(res < 0):
        sign = 'negative'
    else:
        sign = 'mixed'
        
    return (res, stats, sign)
