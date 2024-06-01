''' Module for timeseries modeling
'''

import numpy as np
# new for statsmodels v0.12
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
# from tqdm import tqdm
from .tsutils import standardize
# from stochastic.processes.noise import ColoredNoise
# from stochastic.processes.noise import FractionalGaussianNoise

from .tsbase import (
    is_evenly_spaced
)
from scipy import optimize
from scipy.optimize import minimize # for MLE estimation of tau


__all__ = [
    'ar1_fit',
    'ar1_sim',
    'uar1_fit',
    'uar1_sim',
    'colored_noise',
    'colored_noise_2regimes',
    'gen_ar1_evenly',
    'gen_ts',
    'tau_estimation',
    'random_time_axis',
    'inverse_cumsum'
]


def ar1_model(t, tau, output_sigma=1):
    ''' Simulate AR(1) process with REDFIT
    
    Simulate a (possibly irregularly-sampled) AR(1) process with given decay constant tau, à la REDFIT.

    Parameters
    ----------
    t :  array
        Time axis of the time series
    tau : float
        The averaged persistence

    Returns
    -------
    y : array
        The AR(1) time series

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


def ar1_fit(y, t=None):   ## is this still used anywhere? Looks redundant
    ''' Return lag-1 autocorrelation
    
    Returns the lag-1 autocorrelation from AR(1) fit OR persistence from tauest.

    Parameters
    ----------
    y : array
        The time series
    t : array
        The time axis of that series

    Returns
    -------
    g : float
        Lag-1 autocorrelation coefficient (for evenly-spaced time series)
        OR estimated persistence (for unevenly-spaced time series)

    See also
    --------

    pyleoclim.utils.tsbase.is_evenly_spaced : Check if a time axis is evenly spaced, within a given tolerance

    pyleoclim.utils.tsmodel.tau_estimation : Estimates the  temporal decay scale of an (un)evenly spaced time series.


    '''

    if is_evenly_spaced(t):
        g = ar1_fit_evenly(y)
    else:
        g = tau_estimation(y, t)

    return g




def ar1_sim(y, p, t=None):
    '''Simulate AR(1) process(es) with sample autocorrelation value
    
    Produce p realizations of an AR(1) process of length n with lag-1 autocorrelation g calculated from `y` and (if provided) `t`

    Will be replaced by uar1_sim in a future release


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

    See also
    --------

    pyleoclim.utils.tsmodel.ar1_model : Simulates a (possibly irregularly-sampled) AR(1) process with given decay constant tau, à la REDFIT.

    pyleoclim.utils.tsmodel.ar1_fit : Returns the lag-1 autocorrelation from AR(1) fit OR persistence from tauest.

    pyleoclim.utils.tsmodel.ar1_fit_evenly : Returns the lag-1 autocorrelation from AR(1) fit assuming even temporal spacing.

    pyleoclim.utils.tsmodel.tau_estimation : Estimates the  temporal decay scale of an (un)evenly spaced time series.

    '''
    n = np.size(y)
    ysim = np.empty(shape=(n, p))  # declare array

    sig = np.std(y) # Not MLE estimate in any case
    if is_evenly_spaced(t):
        g = ar1_fit_evenly(y)

        # specify model parameters (statmodel want lag0 coefficent as unity)
        ar = np.r_[1, -g]  # AR model parameter
        ma = np.r_[1, 0.0]  # MA model parameters
        sig_n = sig*np.sqrt(1-g**2)  # theoretical noise variance for the process to achieve the same variance as y

        # simulate AR(1) model for each column
        for i in np.arange(p):
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
    
    MARK FOR DEPRECATION once uar1_fit is adopted

    Wrapper for the function `statsmodels.tsa.arima_process.arma_generate_sample <https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.arma_generate_sample.html>`_.
    used to generate an ARMA

    Parameters
    ----------
    t : array
        the time axis
    
    g : float
        lag-1 autocorrelation

    scale : float
        The standard deviation of the noise.

    burnin : int
        Number of observation at the beginning of the sample to drop. Used to reduce dependence on initial values.

    Returns
    -------
    y : array
        the generated AR(1) series

    See also
    --------

    pyleoclim.utils.tsmodel.gen_ts : Generate pyleoclim.Series with timeseries models

    '''
    ar = np.r_[1, -g]  # AR model parameter
    ma = np.r_[1, 0.0]  # MA model parameters
    y = arma_generate_sample(ar, ma, nsample=np.size(t), scale=scale, burnin=burnin)
    return y


def ar1_fit_evenly(y):
    ''' Returns the lag-1 autocorrelation from AR(1) fit.

    Uses `statsmodels.tsa.arima.model.ARIMA <https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html>`_. to
    calculate lag-1 autocorrelation

    MARK FOR DEPRECATION once uar1_fit is adopted
    
    Parameters
    ----------
    y : array
        Vector of (float) numbers as a time series

    Returns
    -------
    g : float
        Lag-1 autocorrelation coefficient

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
    ''' Estimates the  temporal decay scale of an (un)evenly spaced time series.

    Esimtates the temporal decay scale of an (un)evenly spaced time series. 
    Uses `scipy.optimize.minimize_scalar <https://docs.scipy.org/doc/scipy/reference/optimize.minimize_scalar-bounded.html>`_.

    Parameters
    ----------
    y : array
        A time series
    t : array
        Time axis of the time series

    Returns
    -------
    tau_est : float
        The estimated persistence

    References
    ----------

    Mudelsee, M. TAUEST: A Computer Program for Estimating Persistence in Unevenly Spaced Weather/Climate Time Series.
        Comput. Geosci. 28, 69–72 (2002).

    '''
    dt = np.diff(t)

    def ar1_fun(a):
        return np.sum((y[1:] - y[:-1]*a**dt)**2)
    a_est = optimize.minimize_scalar(ar1_fun, bounds=[0, 1], method='bounded').x
    tau_est = -1 / np.log(a_est)

    return tau_est




def isopersistent_rn(y, p):
    ''' Generates p realization of a red noise [i.e. AR(1)] process
    with same persistence properties as y (Mean and variance are also preserved).

    Parameters
    ----------
    X : array
        vector of (real) numbers as a time series, no NaNs allowed
    p : int
        number of simulations

    Returns
    -------
    red : numpy array
        n rows by p columns matrix of an AR1 process, where n is the size of X
    g :float
        lag-1 autocorrelation coefficient
    
    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    pyleoclim.utils.correlation.fdr : Determine significance based on the false discovery rate

    Notes
    -----

    (Some Rights Reserved) Hepta Technologies, 2008

    '''
    n = np.size(y)
    sig = np.std(y, ddof=1)

    g = ar1_fit_evenly(y)
    red = sm_ar1_sim(n, p, g, sig)

    return red, g


def sm_ar1_sim(n, p, g, sig):
    ''' Produce p realizations of an AR1 process of length n with lag-1 autocorrelation g using statsmodels

    Parameters
    ----------
    n : int
        row dimensions
    p : int
        column dimensions

    g : float
        lag-1 autocorrelation coefficient
        
    sig : float
        the standard deviation of the original time series

    Returns
    -------
    red : numpy matrix
        n rows by p columns matrix of an AR1 process

    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    pyleoclim.utils.correlation.fdr : Determine significance based on the false discovery rate

    '''
    # specify model parameters (statsmodel wants lag0 coefficents as unity)
    ar = np.r_[1, -g]  # AR model parameter
    ma = np.r_[1, 0.0] # MA model parameters
    sig_n = sig*np.sqrt(1-g**2) # theoretical noise variance for red to achieve the same variance as X

    red = np.empty(shape=(n, p)) # declare array

    # simulate AR(1) model for each column
    for i in np.arange(p):
        red[:, i] = arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=50, scale=sig_n)

    return red


def colored_noise(alpha, t, std = 1.0, f0=None, m=None, seed=None):
    ''' Generate a colored noise timeseries

    Parameters
    ----------
    alpha : float
        exponent of the 1/f^alpha noise

    t : float
        time vector of the generated noise
        
    std : float
        standard deviation of the series. defaults to 1.0

    f0 : float
        fundamental frequency

    m : int
        maximum number of the waves, which determines the highest frequency of the components in the synthetic noise

    seed : int
        seed for the random number generator
        
        
    Returns
    -------
    y : array
        the generated 1/f^alpha noise

    See also
    --------

    pyleoclim.utils.tsmodel.gen_ts : Generate pyleoclim.Series with timeseries models

    References
    ----------

    Eq. (15) in Kirchner, J. W. Aliasing in 1/f(alpha) noise spectra: origins, consequences, and remedies. Phys Rev E Stat Nonlin Soft Matter Phys 71, 066110 (2005).

    '''
    n = np.size(t)  # number of time points
    y = np.zeros(n)

    if f0 is None:
        f0 = 1/np.ptp(t)  # fundamental frequency
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
    
    if std is not None:
        ys, _, _ = standardize(y,scale=std) # rescale 
    else:
        ys = y
        
    return ys

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

    See also
    --------

    pyleoclim.utils.tsmodel.gen_ts : Generate pyleoclim.Series with timeseries models

    References
    ----------

     Eq. (15) in Kirchner, J. W. Aliasing in 1/f(alpha) noise spectra: origins, consequences, and remedies. Phys Rev E Stat Nonlin Soft Matter Phys 71, 066110 (2005).
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

def gen_ts(model, t=None, nt=1000, **kwargs):
    ''' Generate pyleoclim.Series with timeseries models

    Parameters
    ----------
    model : str, {'colored_noise', 'colored_noise_2regimes', 'ar1'}

        the timeseries model to use
        
        - colored_noise : colored noise with one scaling slope
        - colored_noise_2regimes : colored noise with two regimes of two different scaling slopes
        - ar1 : AR(1) series, with default autocorrelation of 0.5

    t : array
        the time axis

    nt : number of time points
        only works if 't' is None, and it will use an evenly-spaced vector with nt points

    kwargs : dict
        the keyward arguments for the specified timeseries model

    Returns
    -------
    t, v : NumPy arrays
        time axis and values

    See also
    --------

    pyleoclim.utils.tsmodel.colored_noise : Generate a colored noise timeseries

    pyleoclim.utils.tsmodel.colored_noise_2regimes : Generate a colored noise timeseries with two regimes

    pyleoclim.utils.tsmodel.gen_ar1_evenly : Generate an AR(1) series

    '''

    if t is None:
        t = np.arange(nt)

    tsm = {
        'colored_noise': colored_noise,
        'colored_noise_2regimes': colored_noise_2regimes,
        'ar1': gen_ar1_evenly,
    }

    tsm_args = {}
    tsm_args['colored_noise'] = {'alpha': 1}
    tsm_args['colored_noise_2regimes'] = {'alpha1': 1/2, 'alpha2': 2, 'f_break': 1/20}
    tsm_args['ar1'] = {'g': 0.5}
    tsm_args[model].update(kwargs)

    v = tsm[model](t=t, **tsm_args[model])

    return t, v

# def parametric_surrogates(y, p, model, param, seed):
#     ''' 
#     Generate `p` surrogates of array X according to a given
#     parametric model 
        
#     Parameters
#     ----------
    
#     model : str
#         Stochastic model for the temporal behavior. Accepted choices are:
#         - 'unif': resample uniformly from the posterior distribution
#         - 'ar': autoregressive model, see  https://www.statsmodels.org/dev/tsa.html#univariate-autoregressive-processes-ar
#         - 'fGn': fractional Gaussian noise, see https://stochastic.readthedocs.io/en/stable/noise.html#stochastic.processes.noise.FractionalGaussianNoise 
#         - 'power-law': aka Colored Noise, see https://stochastic.readthedocs.io/en/stable/noise.html#stochastic.processes.noise.ColoredNoise
        
#     param : variable type [default is None]
#         parameter of the model. 
#         - 'unif': no parameter 
#         - 'ar': param is the result from fitting Statsmodels Autoreg.fit() (with zero-lag term)
#         - 'fGn': param is the Hurst exponent, H (float)
#         - 'power-law': param is the spectral exponent beta (float)
        
#     Under allowable values, 'fGn' and 'power-law' should return equivalent results as long as H = (beta+1)/2 is in [0, 1)
        
#     p : int
#         number of series to export
        
#     trend : array, length self.nt
#         general trend of the ensemble. 
#         If None, it is calculated as the ensemble mean.
#         If provided, it will be added to the ensemble. 
          
#     seed : int
#         seed for the random generator (provided for reproducibility)

#     Returns
#     -------
#     surr :  N x p array containing surrogates
#     '''
    
#     if seed is not None:
#         np.random.seed(seed)
#     N = len(y)      
        
#     paths = np.ndarray((N, p))
    
#     if model == 'ar':
#         coeffs = param[1:] # ignore the zero-lag term
#         arparams = np.r_[1, -coeffs]
#         maparams = np.r_[1, np.zeros_like(coeffs)]
       
#         for j in tqdm(range(p)):
#             y = arma_generate_sample(arparams, maparams, N)
#             z, _, _ = standardize(y)
#             paths[:,j] = z

#     elif model == 'power-law':
#         for j in tqdm(range(p)):
#             CN = ColoredNoise(beta=param,t=N)
#             z, _, _ = standardize(CN.sample(N-1))
#             paths[:,j] = z
             
#     elif model == 'fGn':
#         for j in tqdm(range(p)):
#             fgn = FractionalGaussianNoise(hurst=param, t=N)
#             z, _, _ = standardize(fgn.sample(N, algorithm='daviesharte')) 
#             paths[:,j] = z
        
def n_ll_unevenly_spaced_ar1(theta, y, t):
  """
    Compute the negative log-likelihood of an evenly/unevenly spaced AR(1) model.
    It is assumed that the vector theta is initialized with log of tau and log of sigma 2.
      
    Parameters
    ----------
    theta: array, length 2 
        the first value is tau, the second value sigma^2.
    y: array,length n 
        The vector of observations.
        
    t: array,length n,
        the vector of time values.
  
    Returns
    -------
      float. The value of the negative log likelihood evalued with the arguments provided (theta, y, t).
  """
  # define n
  n = len(y)
  log_tau = theta[0]
  log_sigma_2 = theta[1]
  tau = np.exp(log_tau)
  sigma_2 = np.exp(log_sigma_2)
  delta = np.diff(t)
  phi = np.exp((-delta / tau))
  term_1 =  y[1:n] - (phi * y[0:(n-1)])
  term_2 = pow(term_1, 2)
  term_3 = term_2 / (1- pow(phi, 2))
  term_4 = 1/(sigma_2 * n) *sum(term_3)
  term_5 = 1/n * sum(np.log(1-pow(phi,2)))
  nll = np.log(2*np.pi) + np.log(sigma_2) + term_5 + term_4
  return(nll)

def uar1_fit(y, t):
    ''' Maximum Likelihood Estimation of parameters tau and sigma_2

    Parameters
    ----------
    y : An array of the values of the time series
        Values of the times series.
    t : An array of the time index values of the time series
        Time index values of the time series

    Returns
    -------
    theta_hat : An array containing the estimated parameters tau_hat and sigma_2_hat, first entry is tau_hat, second entry is sigma_2_hat

    '''
    
    # obtain initial value for tau
    tau_initial_value = tau_estimation(y= y, t = t)
    # obtain initial value for sifma_2_0
    sigma_2_initial_value = np.var(y)
    # obtain MLE 
    optim_res = minimize(n_ll_unevenly_spaced_ar1, x0=[np.log(tau_initial_value), np.log(sigma_2_initial_value)], args=(y,t), method='nelder-mead', options={'xatol': 1e-10, 'disp': False, 'maxiter': 1000})
    # transform back parameters
    theta_hat = np.exp(optim_res.x)
    
    return theta_hat

def uar1_sim(t, tau, sigma_2=1):  
                               
    """
    Generate a time series of length n from an autoregressive process of order 1 with evenly/unevenly spaced time points.
    
    Parameters
    ----------
    t : array
        Time axis 
        
    tau : float
        Time decay parameter of the  AR(1) model ($\phi = e^{-\tau}$)
        
    sigma_2 : float
        Variance of the innovations      
  
    Returns
    -------
    ys : n 
        matrix of simulated AR(1) vector
      
          
    See also
    --------
  
    pyleoclim.utils.tsmodel.uar1_fit : Maximumum likelihood estimate of AR(1) parameters 
        
    """
    if t.ndim == 1: # add extraneous dimension if t is 1d, to write only one loop. 
        n = len(t); p = 1
        t = t[:,np.newaxis]
    else:
        n, p = t.shape
    
    # generate innovations
    z = np.random.normal(loc=0, scale=1, size=(n,p))
    y = np.copy(z) # initialize AR(1) vectors
    # fill the array
    for j in range(p):  # Note: this shouldn't work but it does!
        for i in range(1, n): 
            delta_i = t[i,j] - t[i-1,j] 
            phi = np.exp(-delta_i / tau)
            sigma_i = np.sqrt(sigma_2 * (1-phi**2))
            y[i,j] = phi * y[i-1,j] + sigma_i * z[i,j]  
    
    y = np.squeeze(y) # squeeze superfluous dimensions
    return y

def inverse_cumsum(arr):
    return np.diff(np.concatenate(([0], arr)))

def random_time_axis(n, delta_t_dist = "exponential", param = [1.0]):
    '''
    Generate a random time axis according to a specific probability model

    Parameters
    ----------
    n: integer
        The length of the time series 
        
    delta_t_dist: str
        the probability distribution of the random time increments.
        possible choices include 'exponential', 'poisson', 'pareto', or 'random_choice'.
        
        if 'exponential', `param` is expected to be a single scale parameter (traditionally denoted \lambda)
        if 'poisson', `param` is expected to be a single parameter (rate)
        if 'pareto', expects a 2-list with 2 scalar shape & scale parameters (in that order)
        if 'random_choice', expects a 2-list containing the arrays:      
            value_random_choice: 
                elements from which the random sample is generated (e.g. [1,2])
            prob_random_choice: 
                probabilities associated with each entry value_random_choice  (e.g. [.95,.05])
            (These two arrays must be of the same size)
   
        
    Returns
    -------
    t : 1D array of random time axis obtained by taking the cumulative sum of the sampled random time increments, length n


    '''
    # check for a valid distribution 
    valid_distributions = ["exponential", "poisson", "pareto", "random_choice"]
    if delta_t_dist not in valid_distributions:
        raise ValueError("delta_t_dist must be one of: 'exponential', 'poisson', 'pareto', 'random_choice'.")    
    

    param = np.array(param) # coerce array type
    
    if delta_t_dist == "exponential":
        # make sure that param is of len 1
        if len(param) != 1:
            raise ValueError('The Exponential law takes a single scale parameter.')       
        delta_t = np.random.exponential(scale = param, size=n)
        
    elif delta_t_dist == "poisson":
        if len(param) != 1:
            raise ValueError('The Poisson law takes a single parameter.')       
        delta_t = np.random.poisson(lam = param, size = n) + 1
    elif delta_t_dist == "pareto":
        if len(param) != 2:
            raise ValueError('The Pareto law takes a shape and a scale parameter (in that order) ')
        else:
            delta_t = (np.random.pareto(param[0], n) + 1) * param[1]
    elif delta_t_dist == "random_choice":
        if len(param)<2 or len(param[0]) != len(param[1]):
            raise ValueError("value_random_choice and prob_random_choice must have the same size.")
        delta_t = np.random.choice(param[0], size=n, p=param[1])
    return np.cumsum(delta_t)





# def fBMsim(N=128, H=0.25):
#     '''Simple method to generate fractional Brownian Motion

#     Parameters
#     ----------

#     N : int
#         the length of the simulated time series
#     H : float
#         Hurst index, should be in (0, 1). The relationship between H and the scaling exponent beta is
#         H = (beta-1) / 2

#     Returns
#     -------

#     xfBm : array
#         the simulated fractional Brownian Motion time series

#     References
#     ----------

#     1. http://cours-physique.lps.ens.fr/index.php/TD11_Correlated_Noise_2011
#     2. https://www.wikiwand.com/en/Fractional_Brownian_motion

#     @authors: jeg, fzhu
#     '''
#     assert isinstance(N, int) and N >= 1
#     assert H > 0 and H < 1, "H should be in (0, 1)!"

#     HH = 2 * H

#     ns = N-1  # number of steps
#     covariance = np.ones((ns, ns))

#     for i in range(ns):
#         for j in range(i, ns):
#             x = np.abs(i-j)
#             covariance[i, j] = covariance[j, i] = (np.abs(x-1)**HH + (x+1)**HH - 2*x**HH) / 2.

#     w, v = np.linalg.eig(covariance)

#     A = np.zeros((ns, ns))
#     for i in range(ns):
#         for j in range(i, ns):
#             A[i, j] = A[j, i] = np.sum(np.sqrt(w) * v[i, :] * v[j, :])

#     xi = np.random.randn((ns))
#     eta = np.dot(A, xi)

#     xfBm = np.zeros(N)
#     xfBm[0] = 0
#     for i in range(1, N):
#         xfBm[i] = xfBm[i-1] + eta[i-1]

#     return xfBm
