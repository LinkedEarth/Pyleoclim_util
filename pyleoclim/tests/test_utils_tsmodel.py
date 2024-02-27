#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:36:37 2024

@author: julieneg
"""

import pytest
import numpy as np
import pyleoclim.utils.tsmodel
from pyleoclim.utils import tsmodel

# for MLE estimation of tau_0
from scipy import optimize
from scipy.optimize import minimize

#@Julien, I initially deifned these function in ts.model.py, however, the test didnt's pass as the defined function where not found when running the test, hence, I defined all functions in the test file

def tau_estimation(y, t):
    '''
    @Julien I had to redefine this function in the test to ensure the test is working when running pytest pyleoclim/tests/test_utils_tsmodel.py

    Estimates the  temporal decay scale of an (un)evenly spaced time series.

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


def gen_evenly_unevenly_spaced_ar1(n, tau_0, sigma_2_0, seed=12345, scale_parameter_delta_t = None, evenly_spaced = False):
  """
  @Julien, I just added this way of generating a evenly/unevenly spaced AR1 so I can use it on the test both evenly and not evenly spaced AR1, plan is to remove it later and use your generation function
  
  Generate a time series of length n from an autoregressive process of order 1 with evenly/unevenly spaced time points.
  In case parameter evenly_spaced is True, delta_t  (spacing between time points) is a vector of 1, if evenly_spaced is False, delta_t are generated from an exponential distribution.
  Args:
    n: An integer. The length of the time series, 
    tau_0: Parameter of the unevenly AR1 model.
    sigma_0: Parameter of the unevenly AR1 model.
    seed: An integer. Random seed for reproducible results.
    scale_parameter_delta_t: parameter of the exponential distribution for generating the delta_t.
  
  Returns: A tuple with three array, the y value, the index t and the delta_t of the generated time series.
  
  """
  # generate delta-t depending on the parameter evenly_spaced
  if evenly_spaced:
      t_delta = [1]*n
  else:
      np.random.seed(seed)
      # consider generation of the delta_t from an exponential distribution
      t_delta= np.random.exponential(scale = scale_parameter_delta_t, size=n)
  # obtain the time index from the delta_t distribution
  t = np.cumsum(t_delta)
    
  # create empty vector
  y = np.empty(n)
  
  # generate unevenly spaced AR1
  np.random.seed(seed)
  z = np.random.normal(loc=0, scale=1, size=n)
  y[0] = z[0]
  for i in range(1,n):
    delta_i = t[i] - t[i-1] 
    phi_i = np.exp(-delta_i / tau_0)
    sigma_2_i = sigma_2_0 * (1-pow(phi_i, 2))
    sigma_i = np.sqrt(sigma_2_i)
    y[i] = phi_i * y[i-1] + sigma_i * z[i]
  return y, t, t_delta




def n_ll_unevenly_spaced_ar1(theta, y, t):
  """
    Compute the negative log-likelihood of an evenly/unevenly spaced AR1 model.
    It is assumed that the vector theta is initialized with log of tau and log of sigma 2.
      
    Args:
        theta: A vector of length 2, with the first value being the log of tau_0, and the second value the log of sigma^2.
        y: The vector of observations.
        t: the vector of time value.
  
    Returns:
      A double. The value of the negative log likelihood evalued with the arguments provided (theta, y, t).
  """
  # define n
  n = len(y)
  log_tau_0 = theta[0]
  log_sigma_2_0 = theta[1]
  tau_0 = np.exp(log_tau_0)
  sigma_2 = np.exp(log_sigma_2_0)
  delta = np.diff(t)
  phi = np.exp((-delta / tau_0))
  term_1 =  y[1:n] - (phi * y[0:(n-1)])
  term_2 = pow(term_1, 2)
  term_3 = term_2 / (1- pow(phi, 2))
  term_4 = 1/(sigma_2 * n) *sum(term_3)
  term_5 = 1/n * sum(np.log(1-pow(phi,2)))
  nll = np.log(2*np.pi) + np.log(sigma_2) + term_5 + term_4
  return(nll)



def ar1fit_ml(y, t):
    '''
   Estimate parameters tau_0 and sigma_2_0 based on the MLE

    Parameters
    ----------
    y : An array of the values of the time series
        Values of the times series.
    t : An array of the time index values of the time series
        Time index values of the time series

    Returns:
        An array containing the estimated parameters tau_0_hat and sigma_2_hat, first entry is tau_0_hat, second entry is sigma_2_hat
    -------
    None.

    '''
    # obtain initial value for tau_0
    tau_initial_value = tau_estimation(y= y, t = t)
    # obtain initial value for sifma_2_0
    sigma_2_initial_value = np.var(y)
    # obtain MLE 
    optim_res = minimize(n_ll_unevenly_spaced_ar1, x0=[np.log(tau_initial_value), np.log(sigma_2_initial_value)], args=(y,t), method='nelder-mead', options={'xatol': 1e-10, 'disp': False, 'maxiter': 1000})
    # transform back parameters
    theta_hat = np.exp(optim_res.x)
    
    return theta_hat


def test_ar1fit_ml_evenly_spaced():
    '''
    Tests whether this method works well on an evenly-spaced AR(1) process

    '''
    tol = .5
    
    y, t, t_delta  = gen_evenly_unevenly_spaced_ar1(n = 200, tau_0=5, sigma_2_0=2, seed=123, scale_parameter_delta_t=1, evenly_spaced=True)
    theta_hat = ar1fit_ml(y, t)
    
    # test that 
    
    assert np.abs(theta_hat[0]-5) < tol
    assert np.abs(theta_hat[1]-2) < tol
    
    

def test_ar1fit_ml_unevenly_spaced():
    '''
    Tests whether this method works well on an unevenly-spaced AR(1) process

    '''
    tol = .5
    
    y, t, t_delta  = gen_evenly_unevenly_spaced_ar1(n = 200, tau_0=5, sigma_2_0=2, seed=123, scale_parameter_delta_t=1, evenly_spaced=False)
    theta_hat = ar1fit_ml(y, t)
    
    # test that 
    
    assert np.abs(theta_hat[0]-5) < tol
    assert np.abs(theta_hat[1]-2) < tol
    