''' The module for timeseries models
'''

import numpy as np
import statmodels.api as sm
from .tutils import is_evenly_spaced
from .tutils import preprocess
from scipy import optimize

__all__ = [
    'ar1_sim',
]

def ar1_model(t, tau, n=None):
    ''' Simulate a (possibly irregularly-sampled) AR(1) process with given decay
        constant tau, à la REDFIT.
    Args
    ----

    t :  array
        time axis of the time series
    tau : float
        the averaged persistence
    n : int
        the length of the AR(1) process

    Returns
    -------

    y : array
        the AR(1) time series

    References
    ----------

    Schulz, M. & Mudelsee, M. REDFIT: estimating red-noise spectra directly from unevenly spaced
        paleoclimatic time series. Computers & Geosciences 28, 421–426 (2002).

    '''
    if n is None:
        n = np.size(t)

    y    = np.zeros(n)
    y[0] = 0  # initializing

    for i in range(1, n):
        scaled_dt = (t[i] - t[i-1]) / tau
        rho = np.exp(-scaled_dt)
        err = np.random.normal(0, np.sqrt(1 - rho**2), 1)
        y[i] = y[i-1]*rho + err

    return y

#  def ar1_fit(y, t=None, detrend= None, params=["default", 4, 0, 1]):
def ar1_fit(y, t=None):
    ''' Returns the lag-1 autocorrelation from AR(1) fit OR persistence from tauest.

    Args
    ----

    y : array
        the time series
    t : array
        the time axis of that series

    Returns
    -------

    g : float
        lag-1 autocorrelation coefficient (for evenly-spaced time series)
        OR estimated persistence (for unevenly-spaced time series)
    '''

    if is_evenly_spaced(t):
        #  g = ar1_fit_evenly(y, t, detrend=detrend, params=params)
        g = ar1_fit_evenly(y, t)
    else:
        #  g = tau_estimation(y, t, detrend=detrend, params=params)
        g = tau_estimation(y, t)

    return g

def ar1_sim(y, n , p, t=None):
#  def ar1_sim(y, n, p, t=None, detrend=False, params=["default", 4, 0, 1]):
    ''' Produce p realizations of an AR(1) process of length n with lag-1 autocorrelation g calculated from `y` and (if provided) `t`

    Args
    ----

    y : array
        a time series
    n : int
        row dimension  (number of samples)
    p : int
        column dimension (number of surrogates)
    t : array
        the time axis of the series

    Returns
    -------

    Yr : array
        n by p matrix of simulated AR(1) vector

    '''

    Yr = np.empty(shape=(n, p))  # declare array

    if is_evenly_spaced(t):
        #  g = ar1_fit(y, t=t, detrend=detrend, params=params)
        g = ar1_fit(y, t=t)
        sig = np.std(y)

        # specify model parameters (statmodel want lag0 coefficent as unity)
        ar = np.r_[1, -g]  # AR model parameter
        ma = np.r_[1, 0.0]  # MA model parameters
        sig_n = sig*np.sqrt(1-g**2)  # theoretical noise variance for Yr to achieve the same variance as y

        # simulate AR(1) model for each column
        for i in np.arange(p):
            Yr[:, i] = sm.ta.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=50, sigma=sig_n)

    else:
        #  tau_est = ar1_fit(y, t=t, detrend=detrend, params=params)
        tau_est = ar1_fit(y, t=t)
        for i in np.arange(p):
            Yr[:, i] = ar1_model(t, tau_est, n=n)

    if p == 1:
        Yr = Yr[:, 0]

    return Yr

def ar1_fit_evenly(y, t):
#  def ar1_fit_evenly(y, t, detrend=False, params=["default", 4, 0, 1], gaussianize=False):
    ''' Returns the lag-1 autocorrelation from AR(1) fit.

    Args
    ----

    y : array
        vector of (float) numbers as a time series
    t : array
        The time axis for the timeseries. Necessary for use with the Savitzky-Golay filters method since the series should be evenly spaced.
    detrend : string
        'linear' - a linear least-squares fit to `y` is subtracted;
        'constant' - the mean of `y` is subtracted
        'savitzy-golay' - y is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
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
    #  pd_y = preprocess(y, t, detrend=detrend, params=params, gaussianize=gaussianize)
    #  ar1_mod = sm.ta.AR(pd_y, missing='drop').fit(maxlag=1)
    ar1_mod = sm.ta.AR(y, missing='drop').fit(maxlag=1)
    g = ar1_mod.params[1]

    if g > 1:
        print('Warning: AR(1) fitted autocorrelation greater than 1; setting to 1')
        g = 1

    return g

def tau_estimation(y, t):
#  def tau_estimation(y, t, detrend=False, params=["default", 4, 0, 1], gaussianize=False, standardize=True):
    ''' Return the estimated persistence of a givenevenly/unevenly spaced time series.

    Args
    ----

    y : array
        a time series
    t : array
        time axis of the time series
    detrend : string
        'linear' - a linear least-squares fit to `y` is subtracted;
        'constant' - the mean of `y` is subtracted
        'savitzy-golay' - y is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
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
    #  pd_y = preprocess(y, t, detrend=detrend, params=params, gaussianize=gaussianize, standardize=standardize)
    dt = np.diff(t)
    #  assert dt > 0, "The time point should be increasing!"

    def ar1_fun(a):
        #  return np.sum((pd_y[1:] - pd_y[:-1]*a**dt)**2)
        return np.sum((y[1:] - y[:-1]*a**dt)**2)

    a_est = optimize.minimize_scalar(ar1_fun, bounds=[0, 1], method='bounded').x
    #  a_est = optimize.minimize_scalar(ar1_fun, method='brent').x

    tau_est = -1 / np.log(a_est)

    return tau_est
