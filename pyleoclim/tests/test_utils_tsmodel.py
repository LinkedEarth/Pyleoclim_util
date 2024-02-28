#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:36:37 2024

@author: julieneg
"""

import pytest
import numpy as np
from pyleoclim.utils import tsmodel

def gen_evenly_unevenly_spaced_ar1(n=200, tau_0=5, sigma_2_0=2, seed=123, scale_parameter_delta_t=1, evenly_spaced = False):
  """
  
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

@pytest.mark.parametrize('evenly_spaced', [True, False])
def test_ar1fit_ml(evenly_spaced):
    '''
    Tests whether this method works well on an AR(1) process with known parameters

    '''
    tol = .3
    
    y, t, t_delta  = gen_evenly_unevenly_spaced_ar1(evenly_spaced=evenly_spaced)
    theta_hat = tsmodel.ar1_fit_ml(y, t)
    
    # test that 
    
    assert np.abs(theta_hat[0]-5) < tol
    assert np.abs(theta_hat[1]-2) < tol
    
    


    