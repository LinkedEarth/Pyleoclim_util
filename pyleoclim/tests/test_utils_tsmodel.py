#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:36:37 2024

@author: julieneg
"""


import pytest
import numpy as np
from pyleoclim.utils import tsmodel
#from scipy.stats import poisson

@pytest.mark.parametrize('model', ["exponential", "poisson"])
def test_time_index_0(model):
    '''
    Generate time increments with 1-parameter models
    '''
    delta_t = tsmodel.random_time_axis(n=20, param=[1], delta_t_dist = model)
    assert all(np.cumsum(delta_t)>0)

def test_time_index_1():
    '''
    Generate time increments with Pareto
    '''
    delta_t = tsmodel.random_time_axis(n=20, param=[4.2,2.5], delta_t_dist = "pareto")
    assert all(np.cumsum(delta_t)>0)
    
def test_time_index_2():
    '''
    Generate time increments with random choice
    '''
    delta_t = tsmodel.random_time_axis(n=20, delta_t_dist = "random_choice",
                                        param=[[1,2],[.95,.05]] )
    assert all(np.cumsum(delta_t)>0)
    
@pytest.mark.parametrize(('p', 'evenly_spaced'), [(1, True), (10, True), (1, False), (10, False)])
def test_uar1_fit(p, evenly_spaced, tol = 0.45):
    '''
    Tests whether this method works well on an AR(1) process with known parameters and evenly spaced time points

    '''
    # define tolerance
    tau = 2
    sigma_2 = 1
    n = 500
    np.random.seed(108)
    
    if evenly_spaced:
       t_arr = np.tile(range(1,n), (p, 1)).T 
    else:
        t_arr = np.zeros((n, p))  # Initialize matrix to store time increments
        for j in range(p):
            # Generate random time increment
            t_arr[:, j] = tsmodel.random_time_axis(n=n, param=[1])
   
    # Create an empty matrix to store estimated parameters
    theta_hat_matrix = np.empty((p, 2))
    # estimate parameters for each time series
    for j in range(p):
        ys = tsmodel.uar1_sim(t = t_arr[:, j], tau = tau, sigma_2=sigma_2)
        theta_hat_matrix[j,:]  = tsmodel.uar1_fit(ys, t_arr[:, j])
    # compute mean of estimated param for each simulate ts
    theta_hat_bar = np.mean(theta_hat_matrix, axis=0)
  
    # test that 
    assert np.abs(theta_hat_bar[0]-tau) < tol
    assert np.abs(theta_hat_bar[1]-sigma_2) < tol
    
@pytest.mark.parametrize('std',[None,2])
def test_colored_noise(std, nt=100, eps = 0.1):
    t = np.arange(nt)
    v = tsmodel.colored_noise(alpha=1.0, t=t, std = std)
    if std is not None:
        assert np.abs(v.std() - std) < eps

