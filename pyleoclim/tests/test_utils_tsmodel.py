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
    tol = .3
    
    y, t, t_delta  = tsmodel.ar1_sim_geneva(evenly_spaced=evenly_spaced)
    theta_hat = tsmodel.ar1_fit_ml(y, t)
    
    # test that 
    
    assert np.abs(theta_hat[0]-5) < tol
    assert np.abs(theta_hat[1]-2) < tol
    
    


    