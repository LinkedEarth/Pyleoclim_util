#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:36:37 2024

@author: julieneg
"""

import pytest
import numpy as np
from pyleoclim.utils import tsmodel


def uneven_ar1(nt =100):  # define your function here

    return t, v

def test_ar1fit_ml_t0():
    '''
    Tests whether this method works well on an evenly-spaced AR(1) process

    '''
    tol = 1e-2
    t,v = tsmodel.gen_ts(model='ar1',nt=200) # should have gamma close to 0.5
    tau, sigma2 = tsmodel.ar1fit_ml(v,t)
    
    # tau should be close to np.log(2)
    
    assert np.abs(sigma2 - 1) < tol
    assert np.abs(tau + np.log(2)) < tol
    
    
    