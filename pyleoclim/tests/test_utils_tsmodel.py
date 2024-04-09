#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:36:37 2024

@author: julieneg
"""


import pytest
import numpy as np
from pyleoclim.utils import tsmodel
import pyleoclim as pyleo
from scipy.stats import expon
#from scipy.stats import poisson

@pytest.mark.parametrize('model', ["exponential", "poisson"])
def test_time_index_0(model):
    '''
    Generate time increments with 1-parameter models
    '''
    delta_t = tsmodel.random_time_index(n=20, param=[1], delta_t_dist = model)
    assert all(np.cumsum(delta_t)>0)

def test_time_index_1():
    '''
    Generate time increments with Pareto
    '''
    delta_t = tsmodel.random_time_index(n=20, param=[4.2,2.5], delta_t_dist = "pareto")
    assert all(np.cumsum(delta_t)>0)
    
def test_time_index_2():
    '''
    Generate time increments with random choice
    '''
    delta_t = tsmodel.random_time_index(n=20, delta_t_dist = "random_choice",
                                      param=[[1,2],[.95,.05]] )
    assert all(np.cumsum(delta_t)>0)
    

    

@pytest.mark.parametrize(('p', 'evenly_spaced'), [(1, True), (10, True), (1, False), (10, False)])
def test_uar1_fit(p, evenly_spaced):
    '''
    Tests whether this method works well on an AR(1) process with known parameters and evenly spaced time points

    '''
    # define tolerance
    tol = .4
    tau = 2
    sigma_2 = 1
    n = 500
    
    # create an array of evenly spaced points    
    # if p==1:
    #     if evenly_spaced:
    #         t = np.arange(n)
    #     else:
    #         t = tsmodel.random_time_index(n = n, param=[1])
    #     ys = tsmodel.uar1_sim(t, tau_0 = tau, sigma_2_0=sigma_2)
    #     theta_hat = tsmodel.uar1_fit(ys, t)
    #     assert np.abs(theta_hat[0]-tau) < tol
    #     assert np.abs(theta_hat[1]-sigma_2) < tol
    # elif p>1:
    if evenly_spaced:
       t_arr = np.tile(range(1,n), (p, 1)).T 
    else:
        t_arr = np.zeros((n, p))  # Initialize matrix to store time increments
        for j in range(p):
            # Generate random time increment
            t_arr[:, j] = tsmodel.random_time_index(n=n, param=[1])
   
    # Create an empty matrix to store estimated parameters
    theta_hat_matrix = np.empty((p, 2))
    # estimate parameters for each time series
    for j in range(p):
        ys = tsmodel.uar1_sim(t = t_arr[:, j], tau_0 = tau, sigma_2_0=sigma_2)
        theta_hat_matrix[j,:]  = tsmodel.uar1_fit(ys, t_arr[:, j])
    # compute mean of estimated param for each simulate ts
    theta_hat_bar = np.mean(theta_hat_matrix, axis=0)
  
    # test that 
    assert np.abs(theta_hat_bar[0]-tau) < tol
    assert np.abs(theta_hat_bar[1]-sigma_2) < tol

  
@pytest.mark.parametrize('p', [1, 50])
def test_surrogates_uar1_match(p):
    tau = 2
    sigma_2 = 1
    # ys, t_sim = uar1_sim(n=200, tau_0=tau, sigma_2_0=sigma_2, evenly_spaced=True, p = 1)
    n = 500
    # generate time index
    t_arr = np.arange(1,(n+1))
    # create time series
    ys, t_sim = tsmodel.uar1_sim(t_arr = t_arr, tau_0=tau, sigma_2_0=sigma_2)
    ts = pyleo.Series(time = t_sim, value=ys)
    # generate surrogates
    surr = ts.surrogates(method = 'uar1', number = p, time_pattern ="match")
    if p ==1:
        assert(all(surr.series_list[0].time == t_sim))
    if p> 1:
        for i in range(p):
            assert(all(surr.series_list[i].time == t_sim))



@pytest.mark.parametrize('p', [1, 50])
def test_surrogates_uar1_even(p):
    tau = 2
    sigma_2 = 1
    time_incr = 2
    n = 500
    # generate time index
    t = np.arange(1,(n+1))
    # create time series
    ys = tsmodel.uar1_sim(t, tau_0=tau, sigma_2_0=sigma_2)
    ts = pyleo.Series(time = t, value=ys)
    # generate surrogates
    surr = ts.surrogates(method = 'uar1', number = p, time_pattern ="even", settings={"time_increment" :time_incr})
    if p ==1:
        assert(all(tsmodel.inverse_cumsum(surr.series_list[0].time) == time_incr))
    if p> 1:
        for i in range(p):
           assert(all(tsmodel.inverse_cumsum(surr.series_list[i].time) == time_incr))


@pytest.mark.parametrize('p', [1, 50])
def test_surrogates_uar1_uneven(p):
    tol = 0.5
    tau = 2
    sigma_2 = 1
    n = 500
    # generate time index
    t_arr = np.arange(1,(n+1))
    # create time series
    ys, t_sim = tsmodel.uar1_sim(t_arr = t_arr, tau_0=tau, sigma_2_0=sigma_2)
    ts = pyleo.Series(time = t_sim, value=ys)
    # generate surrogates default is exponential with parameter value 1
    surr = ts.surrogates(method = 'uar1', number = p, time_pattern ="uneven")
    #surr = ts.surrogates(method = 'uar1', number = p, time_pattern ="uneven",settings={"delta_t_dist" :"poisson","param":[1]} )
    if p ==1:
        delta_t = tsmodel.inverse_cumsum(surr.series_list[0].time)
        # Compute the empirical cumulative distribution function (CDF) of the generated data
        empirical_cdf, bins = np.histogram(delta_t, bins=100, density=True)
        empirical_cdf = np.cumsum(empirical_cdf) * np.diff(bins)

        # Compute the theoretical CDF of the Exponential distribution
        theoretical_cdf = expon.cdf(bins[1:], scale=1)

        # Trim theoretical_cdf to match the size of empirical_cdf
        theoretical_cdf = theoretical_cdf[:len(empirical_cdf)]

        # Compute the L2 norm (Euclidean distance) between empirical and theoretical CDFs
        l2_norm = np.linalg.norm(empirical_cdf - theoretical_cdf)

        assert(l2_norm<tol)
    if p> 1:
        for i in range(p):
            delta_t = tsmodel.inverse_cumsum(surr.series_list[i].time)
            # Compute the empirical cumulative distribution function (CDF) of the generated data
            empirical_cdf, bins = np.histogram(delta_t, bins=100, density=True)
            empirical_cdf = np.cumsum(empirical_cdf) * np.diff(bins)

            # Compute the theoretical CDF of the Exponential distribution
            theoretical_cdf = expon.cdf(bins[1:], scale=1)

            # Trim theoretical_cdf to match the size of empirical_cdf
            theoretical_cdf = theoretical_cdf[:len(empirical_cdf)]

            # Compute the L2 norm (Euclidean distance) between empirical and theoretical CDFs
            l2_norm = np.linalg.norm(empirical_cdf - theoretical_cdf)

            assert(l2_norm<tol)
            
