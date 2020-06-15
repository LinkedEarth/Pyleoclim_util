''' The module for timeseries models
'''

import numpy as np
import statsmodels.api as sm
from .tsutils import (
    is_evenly_spaced,
    clean_ts,
)
#from .tsutils import preprocess   # no longer used here
from scipy import optimize

__all__ = [
    'ar1_sim',
]

def ar1_model(t, tau):
    ''' Simulate a (possibly irregularly-sampled) AR(1) process with given decay
        constant tau, à la REDFIT.
    Args
    ----

    t :  array
        time axis of the time series
    tau : float
        the averaged persistence

    Returns
    -------

    y : array
        the AR(1) time series

    References
    ----------

    Schulz, M. & Mudelsee, M. REDFIT: estimating red-noise spectra directly from unevenly spaced
        paleoclimatic time series. Computers & Geosciences 28, 421–426 (2002).

    '''
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

def ar1_sim(y, p, t=None):
    ''' Produce p realizations of an AR(1) process of length n with lag-1 autocorrelation g calculated from `y` and (if provided) `t`

    Args
    ----

    y : array
        a time series; NaNs not allowed
    p : int
        column dimension (number of surrogates)
    t : array
        the time axis of the series

    Returns
    -------

    Yr : array
        n by p matrix of simulated AR(1) vector

    '''
    n = np.size(y)
    Yr = np.empty(shape=(n, p))  # declare array

    if is_evenly_spaced(t):
        #  g = ar1_fit(y, t=t, detrend=detrend, params=params)
        g = ar1_fit(y, t=t)
        sig = np.std(y)

        # specify model parameters (statmodel want lag0 coefficent as unity)
        ar = np.r_[1, -g]  # AR model parameter
        ma = np.r_[1, 0.0]  # MA model parameters
        sig_n = sig*np.sqrt(1-g**2)  # theoretical noise variance for the process to achieve the same variance as y

        # simulate AR(1) model for each column
        for i in np.arange(p):
            #Yr[:, i] = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=50, sigma=sig_n) # old statsmodels syntax
            Yr[:, i] = sm.tsa.ArmaProcess(ar, ma).generate_sample(nsample=n, scale=sig_n, burnin=50) # statsmodels v0.11.1-?
    else:
        #  tau_est = ar1_fit(y, t=t, detrend=detrend, params=params)
        tau_est = ar1_fit(y, t=t)
        for i in np.arange(p):
            Yr[:, i] = ar1_model(t, tau_est)

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
        The time axis for the timeseries.

    Returns
    -------
    g : float
        lag-1 autocorrelation coefficient

    '''
    #  pd_y = preprocess(y, t, detrend=detrend, params=params, gaussianize=gaussianize)
    #  ar1_mod = sm.tsa.AR(pd_y, missing='drop').fit(maxlag=1)
    #ar1_mod = sm.tsa.AR(y, missing='drop').fit(maxlag=1)
    #g = ar1_mod.params[1]

    # syntax compatible with statsmodels v0.11.1
    ar1_mod = sm.tsa.ARMA(y, (1, 0), missing='drop').fit(trend='nc', disp=0)
    g = ar1_mod.params[0]

    if g > 1:
        print('Warning: AR(1) fitted autocorrelation greater than 1; setting to 1-eps^{1/4}')
        eps = np.spacing(1.0)
        g = 1.0 - eps**(1/4)

    return g

def tau_estimation(y, t):
#  def tau_estimation(y, t, detrend=False, params=["default", 4, 0, 1], gaussianize=False, standardize=True):
    ''' Estimates the  temporal decay scale of an (un)evenly spaced time series.

    Args
    ----

    y : array
        a time series
    t : array
        time axis of the time series

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
