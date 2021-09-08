''' Module for timeseries modeling
'''

import numpy as np
# new for statsmodels v0.12
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA

from .tsbase import (
    is_evenly_spaced
)
#from .tsutils import preprocess   # no longer used here
from scipy import optimize

__all__ = [
    'ar1_sim',
    'colored_noise',
    'colored_noise_2regimes',
    'gen_ar1_evenly',
]

def ar1_model(t, tau, output_sigma=1):
    ''' Simulate a (possibly irregularly-sampled) AR(1) process with given decay
        constant tau, à la REDFIT.

    Parameters
    ----------

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
    y = np.zeros(n)
    y[0] = 0  # initializing

    for i in range(1, n):
        scaled_dt = (t[i] - t[i-1]) / tau
        rho = np.exp(-scaled_dt)
        err = np.random.normal(0, np.sqrt(1 - rho**2)*output_sigma, 1)
        y[i] = y[i-1]*rho + err

    return y


def ar1_fit(y, t=None):
    ''' Returns the lag-1 autocorrelation from AR(1) fit OR persistence from tauest.

    Parameters
    ----------

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
        g = ar1_fit_evenly(y)
    else:
        #  g = tau_estimation(y, t, detrend=detrend, params=params)
        g = tau_estimation(y, t)

    return g

def ar1_sim(y, p, t=None):
    ''' Produce p realizations of an AR(1) process of length n with lag-1 autocorrelation g calculated from `y` and (if provided) `t`

    Parameters
    ----------

    y : array
        a time series; NaNs not allowed
    p : int
        column dimension (number of surrogates)
    t : array
        the time axis of the series

    Returns
    -------

    ysim : array
        n by p matrix of simulated AR(1) vector

    See Also
    --------

    pyleoclim.utils.tsmodel.ar1_model : Simulates a (possibly irregularly-sampled) AR(1) process with given decay constant tau, à la REDFIT.

    pyleoclim.utils.tsmodel.ar1_fit : Returns the lag-1 autocorrelation from AR(1) fit OR persistence from tauest.

    pyleoclim.utils.tsmodel.ar1_fit_evenly : Returns the lag-1 autocorrelation from AR(1) fit assuming even temporal spacing.

    pyleoclim.utils.tsmodel.tau_estimation : Estimates the  temporal decay scale of an (un)evenly spaced time series.

    '''
    n = np.size(y)
    ysim = np.empty(shape=(n, p))  # declare array

    sig = np.std(y)
    if is_evenly_spaced(t):
        g = ar1_fit_evenly(y)

        # specify model parameters (statmodel want lag0 coefficent as unity)
        ar = np.r_[1, -g]  # AR model parameter
        ma = np.r_[1, 0.0]  # MA model parameters
        sig_n = sig*np.sqrt(1-g**2)  # theoretical noise variance for the process to achieve the same variance as y

        # simulate AR(1) model for each column
        for i in np.arange(p):
            #ysim[:, i] = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=50, sigma=sig_n) # old statsmodels syntax
            #ysim[:, i] = sm.tsa.ArmaProcess(ar, ma).generate_sample(nsample=n, scale=sig_n, burnin=50) # statsmodels v0.11.1-?
            ysim[:, i] = arma_generate_sample(ar, ma, nsample=n, scale=sig_n, burnin=50) # statsmodels v0.12+
    else:
        #  tau_est = ar1_fit(y, t=t, detrend=detrend, params=params)
        tau_est = tau_estimation(y, t)
        for i in np.arange(p):
            # the output of ar1_model has unit variance,
            # multiply by sig to be consistent with the original input timeseries
            ysim[:, i] = ar1_model(t, tau_est, output_sigma=sig)

    if p == 1:
        ysim = ysim[:, 0]

    return ysim

def gen_ar1_evenly(t, g, scale=1, burnin=50):
    ''' Generate AR(1) series samples

    Parameters
    ----------

    t : array
        the time axis
    
    g : float
        lag-1 autocorrelation

    scale : float
        The standard deviation of noise.

    burnin : int
        Number of observation at the beginning of the sample to drop. Used to reduce dependence on initial values.

    Returns
    -------
    y : array
        the generated AR(1) series


    See also
    --------
    statsmodels.tsa.arima_process.arma_generate_sample: Simulate data from an ARMA. (https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.arma_generate_sample.html)

    '''
    ar = np.r_[1, -g]  # AR model parameter
    ma = np.r_[1, 0.0]  # MA model parameters
    y = arma_generate_sample(ar, ma, nsample=np.size(t), scale=scale, burnin=burnin)
    return y


def ar1_fit_evenly(y):
    ''' Returns the lag-1 autocorrelation from AR(1) fit.

    Parameters
    ----------
    y : array
        vector of (float) numbers as a time series

    Returns
    -------
    g : float
        lag-1 autocorrelation coefficient

    '''
    # syntax compatible with statsmodels v0.11.1
    #ar1_mod = sm.tsa.ARMA(y, (1, 0), missing='drop').fit(trend='nc', disp=0)
    # syntax compatible with statsmodels v0.12
    ar1_mod = ARIMA(y, order = (1, 0, 0), missing='drop',trend='ct').fit()
    g = ar1_mod.params[2]

    if g > 1:
        print('Warning: AR(1) fitted autocorrelation greater than 1; setting to 1-eps^{1/4}')
        eps = np.spacing(1.0)
        g = 1.0 - eps**(1/4)

    return g

def tau_estimation(y, t):
#  def tau_estimation(y, t, detrend=False, params=["default", 4, 0, 1], gaussianize=False, standardize=True):
    ''' Estimates the  temporal decay scale of an (un)evenly spaced time series.

    Parameters
    ----------

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


def colored_noise(alpha, t, f0=None, m=None, seed=None):
    ''' Generate a colored noise timeseries

    Parameters
    ----------
    alpha : float
        exponent of the 1/f^alpha noise

    t : float
        time vector of the generated noise

    f0 : float
        fundamental frequency

    m : int
        maximum number of the waves, which determines the highest frequency of the components in the synthetic noise

    Returns
    -------

    y : array
        the generated 1/f^alpha noise

    References
    ----------

    Eq. (15) in Kirchner, J. W. Aliasing in 1/f(alpha) noise spectra: origins, consequences, and remedies.
        Phys Rev E Stat Nonlin Soft Matter Phys 71, 066110 (2005).
    '''
    n = np.size(t)  # number of time points
    y = np.zeros(n)

    if f0 is None:
        f0 = 1/n  # fundamental frequency
    if m is None:
        m = n//2

    k = np.arange(m) + 1  # wave numbers

    if seed is not None:
        np.random.seed(seed)

    theta = np.random.rand(int(m))*2*np.pi  # random phase
    for j in range(n):
        coeff = (k*f0)**(-alpha/2)
        sin_func = np.sin(2*np.pi*k*f0*t[j] + theta)
        y[j] = np.sum(coeff*sin_func)

    return y

def colored_noise_2regimes(alpha1, alpha2, f_break, t, f0=None, m=None, seed=None):
    ''' Generate a colored noise timeseries with two regimes

    Parameters
    ----------

    alpha1, alpha2 : float
        the exponent of the 1/f^alpha noise

    f_break : float
        the frequency where the scaling breaks

    t : float
        time vector of the generated noise
    f0 : float
        fundamental frequency
    m : int
        maximum number of the waves, which determines the highest frequency of the components in the synthetic noise

    Returns
    -------

    y : array
        the generated 1/f^alpha noise

    References
    ----------

     Eq. (15) in Kirchner, J. W. Aliasing in 1/f(alpha) noise spectra: origins, consequences, and remedies.
         Phys Rev E Stat Nonlin Soft Matter Phys 71, 066110 (2005).
    '''
    n = np.size(t)  # number of time points
    y = np.zeros(n)

    if f0 is None:
        f0 = 1/n  # fundamental frequency
    if m is None:
        m = n//2  # so the aliasing is limited

    k = np.arange(m) + 1  # wave numbers

    if seed is not None:
        np.random.seed(seed)

    theta = np.random.rand(int(m))*2*np.pi  # random phase

    f_vec = k*f0
    regime1= k*f0>=f_break
    regime2= k*f0<=f_break
    f_vec1 = f_vec[regime1]
    f_vec2 = f_vec[regime2]
    s = np.exp(alpha1/alpha2*np.log(f_vec1[0])) / f_vec2[-1]

    for j in range(n):
        coeff = np.ndarray((np.size(f_vec)))
        coeff[regime1] = f_vec1**(-alpha1/2)
        coeff[regime2] = (s*f_vec2)**(-alpha2/2)
        sin_func = np.sin(2*np.pi*k*f0*t[j] + theta)
        y[j] = np.sum(coeff*sin_func)

    return y
