''' The module for timeseries models
'''

import numpy as np
import statsmodels.api as sm
from .tsutils import is_evenly_spaced
from .tsutils import preprocess
from scipy import optimize

__all__ = [
    'ar1_sim',
]

def ar1_model(ts, tau, n=None):
    ''' Return a time series with the AR1 process

    Args
    ----

    ts : array
        time axis of the time series
    tau : float
        the averaged persistence
    n : int
        the length of the AR1 process

    Returns
    -------

    r : array
        the AR1 time series

    References
    ----------

    Schulz, M. & Mudelsee, M. REDFIT: estimating red-noise spectra directly from unevenly spaced
        paleoclimatic time series. Computers & Geosciences 28, 421–426 (2002).

    '''
    if n is None:
        n = np.size(ts)

    r = np.zeros(n)

    r[0] = 1
    for i in range(1, n):
        scaled_dt = (ts[i] - ts[i-1]) / tau
        rho = np.exp(-scaled_dt)
        err = np.random.normal(0, np.sqrt(1 - rho**2), 1)
        r[i] = r[i-1]*rho + err

    return r

def ar1_fit(ys, ts=None, detrend= None, params=["default", 4, 0, 1]):
    ''' Returns the lag-1 autocorrelation from ar1 fit OR persistence from tauest.

    Args
    ----

    ys : array
        the time series
    ts : array
        the time axis of that series
    detrend : string
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
    params : list
        The paramters for the Savitzky-Golay filters. The first parameter
        corresponds to the window size (default it set to half of the data)
        while the second parameter correspond to the order of the filter
        (default is 4). The third parameter is the order of the derivative
        (the default is zero, which means only smoothing.)

    Returns
    -------

    g : float
        lag-1 autocorrelation coefficient (for evenly-spaced time series)
        OR estimated persistence (for unevenly-spaced time series)
    '''

    if is_evenly_spaced(ts):
        g = ar1_fit_evenly(ys, ts, detrend=detrend, params=params)
    else:
        g = tau_estimation(ys, ts, detrend=detrend, params=params)

    return g

def ar1_sim(ys, n, p, ts=None, detrend=False, params=["default", 4, 0, 1]):
    ''' Produce p realizations of an AR1 process of length n with lag-1 autocorrelation g calculated from `ys` and `ts`

    Args
    ----

    ys : array
        a time series
    n : int
        row dimensions
    p : int
        column dimensions
    ts : array
        the time axis of that series
    detrend : string
        None - the original time series is assumed to have no trend;
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
    params : list
        The paramters for the Savitzky-Golay filters. The first parameter
        corresponds to the window size (default it set to half of the data)
        while the second parameter correspond to the order of the filter
        (default is 4). The third parameter is the order of the derivative
        (the default is zero, which means only smoothing.)

    Returns
    -------

    red : array
        n rows by p columns matrix of an AR1 process

    '''
    red = np.empty(shape=(n, p))  # declare array

    if is_evenly_spaced(ts):
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
            red[:, i] = ar1_model(ts, tau_est, n=n)

    if p == 1:
        red = red[:, 0]

    return red

def ar1_fit_evenly(ys, ts, detrend=False, params=["default", 4, 0, 1],
                   gaussianize=False):
    ''' Returns the lag-1 autocorrelation from ar1 fit.

    Args
    ----

    ys : array
        vector of (float) numbers as a time series
    ts : array
        The time axis for the timeseries. Necessary for use with the Savitzky-Golay filters method since the series should be evenly spaced.
    detrend : string
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
    params : list
        The paramters for the Savitzky-Golay filters. The first parameter
        corresponds to the window size (default it set to half of the data)
        while the second parameter correspond to the order of the filter
        (default is 4). The third parameter is the order of the derivative
        (the default is zero, which means only smoothing.)
    gaussianize : bool
        If True, gaussianizes the timeseries
    standardize : bool
        If True, standardizes the timeseries

    Returns
    -------

    g : float
        lag-1 autocorrelation coefficient

    '''
    pd_ys = preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize)
    ar1_mod = sm.tsa.AR(pd_ys, missing='drop').fit(maxlag=1)
    g = ar1_mod.params[1]

    return g

def tau_estimation(ys, ts, detrend=False, params=["default", 4, 0, 1], 
                   gaussianize=False, standardize=True):
    ''' Return the estimated persistence of a givenevenly/unevenly spaced time series.

    Args
    ----

    ys : array
        a time series
    ts : array
        time axis of the time series
    detrend : string
        'linear' - a linear least-squares fit to `ys` is subtracted;
        'constant' - the mean of `ys` is subtracted
        'savitzy-golay' - ys is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
    params : list
        The paramters for the Savitzky-Golay filters. The first parameter
        corresponds to the window size (default it set to half of the data)
        while the second parameter correspond to the order of the filter
        (default is 4). The third parameter is the order of the derivative
        (the default is zero, which means only smoothing.)
    gaussianize : bool
        If True, gaussianizes the timeseries
    standardize : bool
        If True, standardizes the timeseries

    Returns
    -------

    tau_est : float
        the estimated persistence

    References
    ----------

    Mudelsee, M. TAUEST: A Computer Program for Estimating Persistence in Unevenly Spaced Weather/Climate Time Series.
        Comput. Geosci. 28, 69–72 (2002).

    '''
    pd_ys = preprocess(ys, ts, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)
    dt = np.diff(ts)
    #  assert dt > 0, "The time points should be increasing!"

    def ar1_fun(a):
        return np.sum((pd_ys[1:] - pd_ys[:-1]*a**dt)**2)

    a_est = optimize.minimize_scalar(ar1_fun, bounds=[0, 1], method='bounded').x
    #  a_est = optimize.minimize_scalar(ar1_fun, method='brent').x

    tau_est = -1 / np.log(a_est)

    return tau_est

