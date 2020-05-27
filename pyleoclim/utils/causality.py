#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 05:47:03 2020

@author: deborahkhider

Contains all relevant functions for causality analysis
"""

__all__ = [
    'causality_est',
]

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm
from .correlation import sm_ar1_fit, sm_ar1_sim, phaseran
from scipy.stats.mstats import mquantiles

#--------
#Wrappers
#_________

def causality_est(y1, y2, method='liang', signif_test='isospec', nsim=1000,\
                  qs=[0.005, 0.025, 0.05, 0.95, 0.975, 0.995], **kwargs):
    '''Information flow, estimate the information transfer from series y2 to series y1

    Args
    ----

    y1, y2 : array
        vectors of (real) numbers with identical length, no NaNs allowed
    method : array
        only "liang" for now
    signif_test : str
        the method for significance test
    nsim : int
        the number of AR(1) surrogates for significance test
    qs : list
        the quantiles for significance test
    kwargs : includes
        npt : int
            the number of time advance in performing Euler forward differencing in "liang" method

    Returns
    -------

    res_dict : dictionary
        The result of the dictionary including
    T21 : float
        The information flow from y2 to y1
    tau21 : float
        The standardized info flow from y2 to y1, tau21 = T21/Z
    Z : float
       The total information flow
    qs  : list
        significance test  of quantile levels
    t21_noise : list
        The quantiles of the information flow from noise2 to noise1 for significance testing
    tau21_noise : list
        The quantiles of the standardized information flow from noise2 to noise1 for significance testing
    '''
    if method == 'liang':
        npt = kwargs['npt'] if 'npt' in kwargs else 1
        res_dict = liang_causality(y1, y2, npt=npt)
        tau21 = res_dict['tau21']
        T21 = res_dict['T21']
        Z = res_dict['Z']

        signif_test_func = {
            'isopersist': signif_isopersist,
            'isospec': signif_isospec,
        }

        signif_dict = signif_test_func[signif_test](y1, y2, method=method, nsim=nsim, qs=qs, npt=npt)

        T21_noise_qs = signif_dict['T21_noise_qs']
        tau21_noise_qs = signif_dict['tau21_noise_qs']
        res_dict = {
            'T21': T21,
            'tau21': tau21,
            'Z': Z,
            'signif_qs': qs,
            'T21_noise': T21_noise_qs,
            'tau21_noise': tau21_noise_qs,
        }
    else:
        raise KeyError(f'{method} is not a valid method')

    return res_dict

#-------
#Main functions
#--------
def granger_causality(y1, y2, maxlag=1,addconst=True,verbose=True):
    '''
    statsmodels granger causality tests

    Four tests for granger non causality of 2 time series.

    All four tests give similar results. params_ftest and ssr_ftest are equivalent based on F test which is identical to lmtest:grangertest in R.

    Args
    ----

    y1, y2: array
        vectors of (real) numbers with identical length, no NaNs allowed
    maxlag : int or int iterable
        If an integer, computes the test for all lags up to maxlag. If an iterable, computes the tests only for the lags in maxlag.
    addconst : bool
        Include a constant in the model.
    verbose : bool
        Print results

    Returns
    -------

    dict
        All test results, dictionary keys are the number of lags. For each lag the values are a tuple, with the first element a dictionary with test statistic,
        pvalues, degrees of freedom, the second element are the OLS estimation results for the restricted model, the unrestricted model and the restriction (contrast)
        matrix for the parameter f_test.
    '''

    if len(y1)!=len(y2):
        raise ValueError('Timeseries must be of same length')

    x=np.matrix([y1,y2]).T
    res_dict = grangercausalitytests(x,maxlag=maxlag,addconst=addconst,verbose=verbose)
    return res_dict

def liang_causality(y1, y2, npt=1):
    '''
    Estimate the Liang information transfer from series y2 to series y1


    Args
    ----

    y1, y2 : array
        vectors of (real) numbers with identical length, no NaNs allowed

    npt : int  >=1
        time advance in performing Euler forward differencing,
        e.g., 1, 2. Unless the series are generated with a highly chaotic deterministic system,
        npt=1 should be used

    Returns
    -------

    T21 : float
        info flow from y2 to y1 (Note: not y1 -> y2!)
    tau21 : float
        the standardized info flow fro y2 to y1
    Z : float
        the total info

    References
    ----------

    - Liang, X.S. (2013) The Liang-Kleeman Information Flow: Theory and
            Applications. Entropy, 15, 327-360, doi:10.3390/e15010327
    - Liang, X.S. (2014) Unraveling the cause-efect relation between timeseries.
        Physical review, E 90, 052150
    - Liang, X.S. (2015) Normalizing the causality between time series.
        Physical review, E 92, 022126
    - Liang, X.S. (2016) Information flow and causality as rigorous notions ab initio.
        Physical review, E 94, 052201

    '''
    dt=1
    nm = np.size(y1)

    grad1 = (y1[0+npt:] - y1[0:-npt]) / (npt)
    grad2 = (y2[0+npt:] - y2[0:-npt]) / (npt)

    y1 = y1[:-npt]
    y2 = y2[:-npt]

    N = nm - npt
    C = np.cov(y1, y2)
    detC = np.linalg.det(C)

    dC = np.ndarray((2, 2))
    dC[0, 0] = np.sum((y1-np.mean(y1))*(grad1-np.mean(grad1)))
    dC[0, 1] = np.sum((y1-np.mean(y1))*(grad2-np.mean(grad2)))
    dC[1, 0] = np.sum((y2-np.mean(y2))*(grad1-np.mean(grad1)))
    dC[1, 1] = np.sum((y2-np.mean(y2))*(grad2-np.mean(grad2)))

    dC /= N-1

    a11 = C[1, 1]*dC[0, 0] - C[0, 1]*dC[1, 0]
    a12 = -C[0, 1]*dC[0, 0] + C[0, 0]*dC[1, 0]

    a11 /= detC
    a12 /= detC

    f1 = np.mean(grad1) - a11*np.mean(y1) - a12*np.mean(y2)
    R1 = grad1 - (f1 + a11*y1 + a12*y2)
    Q1 = np.sum(R1*R1)
    b1 = np.sqrt(Q1*dt/N)

    NI = np.ndarray((4, 4))
    NI[0, 0] = N*dt/b1**2
    NI[1, 1] = dt/b1**2*np.sum(y1*y1)
    NI[2, 2] = dt/b1**2*np.sum(y2*y2)
    NI[3, 3] = 3*dt/b1**4*np.sum(R1*R1) - N/b1**2
    NI[0, 1] = dt/b1**2*np.sum(y1)
    NI[0, 2] = dt/b1**2*np.sum(y2)
    NI[0, 3] = 2*dt/b1**3*np.sum(R1)
    NI[1, 2] = dt/b1**2*np.sum(y1*y2)
    NI[1, 3] = 2*dt/b1**3*np.sum(R1*y1)
    NI[2, 3] = 2*dt/b1**3*np.sum(R1*y2)

    NI[1, 0] = NI[0, 1]
    NI[2, 0] = NI[0, 2]
    NI[2, 1] = NI[1, 2]
    NI[3, 0] = NI[0, 3]
    NI[3, 1] = NI[1, 3]
    NI[3, 2] = NI[2, 3]

    invNI = np.linalg.pinv(NI)
    var_a12 = invNI[2, 2]
    T21 = C[0, 1]/C[0, 0] * (-C[1, 0]*dC[0, 0] + C[0, 0]*dC[1, 0]) / detC
    var_T21 = (C[0, 1]/C[0, 0])**2 * var_a12

    dH1_star= a11
    dH1_noise = b1**2 / (2*C[0, 0])

    Z = np.abs(T21) + np.abs(dH1_star) + np.abs(dH1_noise)

    tau21 = T21 / Z
    dH1_star = dH1_star / Z
    dH1_noise = dH1_noise / Z

    res_dict = {
        'T21': T21,
        'tau21': tau21,
        'Z': Z,
        'dH1_star': dH1_star,
        'dH1_noise': dH1_noise
    }

    return res_dict

def signif_isopersist(y1, y2, method,
                      nsim=1000, qs=[0.005, 0.025, 0.05, 0.95, 0.975, 0.995],
                      **kwargs):
    ''' significance test with AR(1) with same persistence

    Args
    ----

    y1, y2 : array
        vectors of (real) numbers with identical length, no NaNs allowed
    method : str
        only "liang" for now
    npt : int>=1
        time advance in performing Euler forward differencing,
        e.g., 1, 2. Unless the series are generated with a highly chaotic deterministic system,
        npt=1 should be used.
    nsim : int
        the number of AR(1) surrogates for significance test
    qs : list
        the quantiles for significance test

    Returns
    -------

    res_dict : dict
        A dictionary with the following information:
          T21_noise_qs : list
            the quantiles of the information flow from noise2 to noise1 for significance testing
          tau21_noise_qs : list
            the quantiles of the standardized information flow from noise2 to noise1 for significance testing

    '''
    g1 = sm_ar1_fit(y1)
    g2 = sm_ar1_fit(y2)
    sig1 = np.std(y1)
    sig2 = np.std(y2)
    n = np.size(y1)
    noise1 = sm_ar1_sim(n, nsim, g1, sig1)
    noise2 = sm_ar1_sim(n, nsim, g2, sig2)

    if method == 'liang':
        npt = kwargs['npt'] if 'npt' in kwargs else 1
        T21_noise = []
        tau21_noise = []
        for i in tqdm(range(nsim), desc='Calculating causality between surrogates'):
            res_noise = liang_causality(noise1[:, i], noise2[:, i], npt=npt)
            tau21_noise.append(res_noise['tau21'])
            T21_noise.append(res_noise['T21'])
        tau21_noise = np.array(tau21_noise)
        T21_noise = np.array(T21_noise)
        tau21_noise_qs = mquantiles(tau21_noise, qs)
        T21_noise_qs = mquantiles(T21_noise, qs)

        res_dict = {
            'tau21_noise_qs': tau21_noise_qs,
            'T21_noise_qs': T21_noise_qs,
        }
    #TODO add granger method
    else:
        raise KeyError(f'{method} is not a valid method')

    return res_dict

def signif_isospec(y1, y2, method,
                   nsim=1000, qs=[0.005, 0.025, 0.05, 0.95, 0.975, 0.995],
                   **kwargs):
    ''' significance test with surrogates with randomized phases

    Args
    ----

    y1, y2 : array
            vectors of (real) numbers with identical length, no NaNs allowed
    method : str
            only "liang" for now
    npt : int>=1
         time advance in performing Euler forward differencing,
         e.g., 1, 2. Unless the series are generated with a highly chaotic deterministic system,
         npt=1 should be used.
    nsim : int
          the number of surrogates for significance test
    qs : list
        the quantiles for significance test

    Returns
    -------

    res_dict : dict
        A dictionary with the following information:
          T21_noise_qs : list
                        the quantiles of the information flow from noise2 to noise1 for significance testing
          tau21_noise_qs : list
                          the quantiles of the standardized information flow from noise2 to noise1 for significance testing
    '''
    
    noise1 = phaseran(y1, nsim)
    noise2 = phaseran(y2, nsim)

    if method == 'liang':
        npt = kwargs['npt'] if 'npt' in kwargs else 1
        T21_noise = []
        tau21_noise = []
        for i in tqdm(range(nsim), desc='Calculating causality between surrogates'):
            res_noise = liang_causality(noise1[:, i], noise2[:, i], npt=npt)
            tau21_noise.append(res_noise['tau21'])
            T21_noise.append(res_noise['T21'])
        tau21_noise = np.array(tau21_noise)
        T21_noise = np.array(T21_noise)
        tau21_noise_qs = mquantiles(tau21_noise, qs)
        T21_noise_qs = mquantiles(T21_noise, qs)

        res_dict = {
            'tau21_noise_qs': tau21_noise_qs,
            'T21_noise_qs': T21_noise_qs,
        }
    #TODO add Granger
    else:
        raise KeyError(f'{method} is not a valid method')

    return res_dict
