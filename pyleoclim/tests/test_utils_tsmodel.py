#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:36:37 2024

@author: julieneg
"""

import pytest
import numpy as np
from pyleoclim.utils import tsmodel

@pytest.mark.parametrize('evenly_spaced', [True, False])
def test_ar1fit_ml(evenly_spaced):
    '''
    Tests whether this method works well on an AR(1) process with known parameters

    '''
    # define tolerance
    tol = .2
    tau = 2
    sigma_2 = 1
    
    # create p=50 time series
    y_sim, t_sim = tsmodel.ar1_sim_geneva(tau_0=tau, sigma_2_0=sigma_2, evenly_spaced=evenly_spaced, p = 10)

    # Create an empty matrix to store estimated parameters
    theta_hat_matrix = np.empty((y_sim.shape[1], 2))
    
    # estimate parameters for each time series
    for j in range(y_sim.shape[1]):
        theta_hat_matrix[j,:]  = tsmodel.ar1_fit_ml(y_sim[:, j], t_sim[:, j])
    
    # compute mean of estimated param for each simulate ts
    theta_hat_bar = np.mean(theta_hat_matrix, axis=0)
    
    # test that 
    
    assert np.abs(theta_hat_bar[0]-tau) < tol
    assert np.abs(theta_hat_bar[1]-sigma_2) < tol
    
    


    