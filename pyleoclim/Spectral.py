#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:23:06 2017

@author: deborahkhider

Spectral module for pyleoclim
"""

import numpy as np
import statsmodels.api as sm

'''
Evenly-spaced methods
'''

def ar1_fit(self, ts):
    """ Returns the lag-1 autocorrelation from ar1 fit.
    
    Args:
        ts (array): vector of (real) numbers as a time series
        
    Returns:
        g (real): lag-1 autocorrelation coefficient
    """


    ar1_mod = sm.tsa.AR(ts, missing='drop').fit(maxlag=1, trend='nc')
    g = ar1_mod.params[0]

    return g


def ar1_sim(self, n, p, g, sig):
    ''' Produce p realizations of an AR1 process of length n with lag-1 autocorrelation g
    
    Args:
        n, p (int): dimensions as n rows by p columns
        g (real): lag-1 autocorrelation coefficient
        sig (real): the standard deviation of the original time series
        
    Returns:
        red (matrix): n rows by p columns matrix of an AR1 process
    '''
    # specify model parameters (statsmodel wants lag0 coefficents as unity)
    ar = np.r_[1, -g]  # AR model parameter
    ma = np.r_[1, 0.0] # MA model parameters
    sig_n = sig*np.sqrt(1-g**2) # theoretical noise variance for red to achieve the same variance as X

    red = np.empty(shape=(n, p)) # declare array

    # simulate AR(1) model for each column
    for i in np.arange(p):
        red[:, i] = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=50, sigma=sig_n)

    return red
    
    


'''
Unenvenly spaced methods
'''

