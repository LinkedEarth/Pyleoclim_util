#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:01:55 2020

@author: deborahkhider

Contains all relevant functions for correlation analysis
"""

__all__ = [
    'corr_sig',
    'fdr',
]

import numpy as np
from scipy.stats import pearsonr
from scipy.stats.mstats import gmean
from scipy.stats import t as stu
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from sklearn import preprocessing
from .tsmodel import ar1_fit_evenly


def corr_sig(y1, y2, nsim=1000, method='isospectral', alpha=0.05):
    """ Estimates the Pearson's correlation and associated significance between two non IID time series
    
    The significance of the correlation is assessed using one of the following methods:
        
    1) 'ttest': T-test adjusted for effective sample size. 
        This is a parametric test (data are Gaussian and identically distributed) with a rather ad-hoc adjustment. 
        It is instantaneous but makes a lot of assumptions about the data, many of which may not be met.
    2) 'isopersistent': AR(1) modeling of x and y.
        This is a parametric test as well (series follow an AR(1) model) but 
        solves the issue by direct simulation. 
    3) 'isospectral': phase randomization of original inputs. (default)
        This is a non-parametric method, assuming only wide-sense stationarity.
        
    
    For 2 and 3, computational requirements scale with nsim.
    When possible, nsim should be at least 1000. 

    Parameters
    ----------

    y1 : array
        vector of (real) numbers of same length as y2, no NaNs allowed
        
    y2 : array
        vector of (real) numbers of same length as y1, no NaNs allowed
        
    nsim : int
        the number of simulations [default: 1000]
        
    method : {'ttest','isopersistent','isospectral' (default)}
        method for significance testing
        
    alpha : float
        significance level for critical value estimation [default: 0.05]

    Returns
    -------
    res : dict 
        the result dictionary, containing

        - r : float
            correlation coefficient
        - p : float 
            the p-value
        - signif : bool
            true if significant; false otherwise
            Note that signif = True if and only if p <= alpha.
         
    See Also
    --------

    pyleoclim.utils.correlation.corr_ttest : Estimates the significance of correlations between 2 time series using the classical T-test adjusted for effective sample size.
    
    pyleoclim.utils.correlation.corr_isopersist : Computes correlation between two timeseries, and their significance using Ar(1) modeling.
    
    pyleoclim.utils.correlation.corr_isospec : Estimates the significance of the correlation using phase randomization
     
    """
    y1 = np.array(y1, dtype=float)
    y2 = np.array(y2, dtype=float)

    assert np.size(y1) == np.size(y2), 'The size of X and the size of Y should be the same!'

    if method == 'ttest':
        (r, signif, p) = corr_ttest(y1, y2, alpha=alpha)
    elif method == 'isopersistent':
        (r, signif, p) = corr_isopersist(y1, y2, alpha=alpha, nsim=nsim)
    elif method == 'isospectral':
        (r, signif, p) = corr_isospec(y1, y2, alpha=alpha, nsim=nsim)

    res={'r':r,'signif':signif,'p':p}    
    
    return res

def fdr(pvals, qlevel=0.05, method='original', adj_method=None, adj_args={}):
    ''' Determine significance based on the FDR approach
    
    The false discovery rate is a method of conceptualizing the rate of type I errors in null hypothesis testing when conducting multiple comparisons. 
    Translated from fdr.R by Dr. Chris Paciorek 
    
    Parameters
    ----------

    pvals : list or array
        A vector of p-values on which to conduct the multiple testing.

    qlevel : float
        The proportion of false positives desired.

    method : {'original', 'general'}
        Method for performing the testing.
            - 'original' follows Benjamini & Hochberg (1995);
            - 'general' is much more conservative, requiring no assumptions on the p-values (see Benjamini & Yekutieli (2001)).
            'original' is recommended, and if desired, using 'adj_method="mean"' to increase power.

    adj_method: {'mean', 'storey', 'two-stage'}
        Method for increasing the power of the procedure by estimating the proportion of alternative p-values.
            - 'mean', the modified Storey estimator in Ventura et al. (2004)
            - 'storey', the method of Storey (2002)
            - 'two-stage', the iterative approach of Benjamini et al. (2001)

    adj_args : dict
        Arguments for adj_method; see prop_alt() for description,
        but note that for "two-stage", qlevel and fdr_method are taken from the qlevel and method arguments for fdr()

    Returns
    -------

    fdr_res : array or None
        A vector of the indices of the significant tests; None if no significant tests

    References
    ----------

    - fdr.R by Dr. Chris Paciorek: https://www.stat.berkeley.edu/~paciorek/research/code/code.html

    '''
    n = len(pvals)

    a = 0
    if adj_method is not None:
        if adj_method == 'two-stage':
            qlevel = qlevel / (1+qlevel)  # see Benjamini et al. (2001) for proof that this controls the FDR at level qlevel
            adj_args['qlevel'] = qlevel
            adj_args['fdr_method'] = method
            print(f'Adjusting cutoff using two-stage method, with method: {adj_args["fdr_method"]}; qlevel: {adj_args["qlevel"]}')

        elif adj_method == 'mean':
            if adj_args == {}:
                 # default arguments for "mean" method of Ventura et al. (2004)
                adj_args['edf_lower'] = 0.8
                adj_args['num_steps'] = 20
            print(f'Adjusting cutoff using mean method, with edf_lower: {adj_args["edf_lower"]}; num_steps: {adj_args["num_steps"]}')

        a = prop_alt(pvals, adj_method, adj_args)

    if a == 1:
        # all hypotheses are estimated to be alternatives
        fdr_res = np.arange(n)
    else:
        qlevel = qlevel / (1-a)  # adjust for estimate of a; default is 0
        fdr_res = fdr_master(pvals, qlevel, method)

    return fdr_res

#-----------
# Utilities
#-----------

def corr_ttest(y1, y2, alpha=0.05):
    """ Estimates the significance of correlations between 2 time series using
    the classical T-test adjusted for effective sample size.
    
    The degrees of freedom are adjusted following n_eff=n(1-g)/(1+g) where g is the lag-1 autocorrelation. 
    
    
    Parameters
    ----------

    y1 : array
        vectors of (real) numbers with identical length, no NaNs allowed
        
    y2 : array
        vectors of (real) numbers with identical length, no NaNs allowed
        
    alpha : float
        significance level for critical value estimation [default: 0.05]

    Returns
    -------

    r : float
         correlation between x and y
         
    signif : bool
        true (1) if significant; false (0) otherwise
        
    pval : float
        test p-value (the probability of the test statistic exceeding the observed one by chance alone)
        
    See Also
    --------
    
    pyleoclim.utils.correlation.corr_isopersist : Estimate Pearson's correlation and associated significance using AR(1)
    
    pyleoclim.utils.correlation.corr_isospec : Estimate Pearson's correlation and associated significance using phase randomization
    
    """
    r = pearsonr(y1, y2)[0]

    g1 = ar1_fit_evenly(y1)
    g2 = ar1_fit_evenly(y2)

    N = np.size(y1)

    Ney1 = N * (1-g1) / (1+g1)
    Ney2 = N * (1-g2) / (1+g2)

    Ne = gmean([Ney1+Ney2])
    assert Ne >= 10, 'Too few effective d.o.f. to apply this method!'

    df = Ne - 2
    t = np.abs(r) * np.sqrt(df/(1-r**2))

    pval = 2 * stu.cdf(-np.abs(t), df)

    signif = pval <= alpha

    return r, signif, pval

def corr_isopersist(y1, y2, alpha=0.05, nsim=1000):
    ''' Computes the Pearson's correlation between two timeseries, and their significance using Ar(1) modeling.
    
    The significance is gauged via a non-parametric (Monte Carlo) simulation of
    correlations with nsim AR(1) processes with identical persistence
    properties as x and y ; the measure of which is the lag-1 autocorrelation (g).

    Parameters
    ----------

    y1 : array
        vectors of (real) numbers with identical length, no NaNs allowed
        
    y2 : array
        vectors of (real) numbers with identical length, no NaNs allowed
        
    alpha : float
        significance level for critical value estimation [default: 0.05]
        
    nsim : int
        number of simulations [default: 1000]

    Returns
    -------

    r : float
        correlation between x and y
        
    signif : bool
        true (1) if significant; false (0) otherwise
        
    pval : float
        test p-value (the probability of the test statstic exceeding the observed one by chance alone)

    Notes
    -----

    The probability of obtaining a test statistic at least as extreme as the one actually observed,
    assuming that the null hypothesis is true.
    The test is 1 tailed on |r|: Ho = { |r| = 0 }, Ha = { |r| > 0 }
    The test is rejected (signif = 1) if pval <= alpha, otherwise signif=0;
    (Some Rights Reserved) Hepta Technologies, 2009
    v1.0 USC, Aug 10 2012, based on corr_signif.
    
    See Also
    --------
    
    pyleoclim.utils.correlation.corr_ttest: Estimates Pearson's correlation and associated significance using a t-test.
    
    pyleoclim.utils.correlation.corr_isospec : Estimates Pearson's correlation and associated significance using 
    
    '''

    r = pearsonr(y1, y2)[0]
    ra = np.abs(r)

    y1_red, g1 = isopersistent_rn(y1, nsim)
    y2_red, g2 = isopersistent_rn(y2, nsim)

    rs = np.zeros(nsim)
    for i in np.arange(nsim):
        rs[i] = pearsonr(y1_red[:, i], y2_red[:, i])[0]

    rsa = np.abs(rs)

    xi = np.linspace(0, 1.1*np.max([ra, np.max(rsa)]), 200)
    kde = gaussian_kde(rsa)
    prob = kde(xi).T

    diff = np.abs(ra - xi)
    #  min_diff = np.min(diff)
    pos = np.argmin(diff)

    pval = np.trapz(prob[pos:], xi[pos:])

    rcrit = np.percentile(rsa, 100*(1-alpha))
    signif = ra >= rcrit

    return r, signif, pval

def isopersistent_rn(X, p):
    ''' Generates p realization of a red noise [i.e. AR(1)] process
    with same persistence properties as X (Mean and variance are also preserved).

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

    Notes
    -----

    (Some Rights Reserved) Hepta Technologies, 2008

    '''
    n = np.size(X)
    sig = np.std(X, ddof=1)

    g = ar1_fit_evenly(X)
    #  red = red_noise(N, M, g)
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
    '''
    # specify model parameters (statsmodel wants lag0 coefficents as unity)
    ar = np.r_[1, -g]  # AR model parameter
    ma = np.r_[1, 0.0] # MA model parameters
    sig_n = sig*np.sqrt(1-g**2) # theoretical noise variance for red to achieve the same variance as X

    red = np.empty(shape=(n, p)) # declare array

    # simulate AR(1) model for each column
    for i in np.arange(p):
        red[:, i] = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=50, scale=sig_n)

    return red

def red_noise(N, M, g):
    ''' Produce M realizations of an AR1 process of length N with lag-1 autocorrelation g

    Parameters
    ----------

    N : int
        row dimensions
        
    M : int
        column dimensions
        
    g : float
        lag-1 autocorrelation coefficient

    Returns
    -------

    red : numpy array
        N rows by M columns matrix of an AR1 process

    Notes
    -----

    (Some Rights Reserved) Hepta Technologies, 2008
    J.E.G., GaTech, Oct 20th 2008
    '''
    red = np.zeros(shape=(N, M))
    red[0, :] = np.random.randn(1, M)
    for i in np.arange(1, N):
        red[i, :] = g * red[i-1, :] + np.random.randn(1, M)

    return red

def corr_isospec(y1, y2, alpha=0.05, nsim=1000):
    ''' Estimates the significance of the correlation using phase randomization

    Estimates the significance of correlations between non IID
    time series by phase randomization of original inputs.
    This function creates 'nsim' random time series that have the same power
    spectrum as the original time series but random phases.

    Parameters
    ----------

    y1 : array
        vectors of (real) numbers with identical length, no NaNs allowed
        
    y2 : array
        vectors of (real) numbers with identical length, no NaNs allowed
        
    alpha : float
        significance level for critical value estimation [default: 0.05]
        
    nsim : int
        number of simulations [default: 1000]

    Returns
    -------

    r : float
        correlation between y1 and y2
        
    signif : bool
        true (1) if significant; false (0) otherwise
        
    F : float
        Fraction of time series with higher correlation coefficents than observed (approximates the p-value).

    References
    ---------

    - Ebisuzaki, W, 1997: A method to estimate the statistical
    significance of a correlation when the data are serially correlated.
    J. of Climate, 10, 2147-2153.
    
    - Prichard, D., Theiler, J. Generating Surrogate Data for Time Series
    with Several Simultaneously Measured Variables (1994)
    Physical Review Letters, Vol 73, Number 7
    (Some Rights Reserved) USC Climate Dynamics Lab, 2012.
    
    See Also
    --------
    
    pyleoclim.utils.correlation.corr_ttest : Estimates Pearson's correlation and associated significance using a t-test
    
    pyleoclim.utils.correlation.corr_isopersist : Estimates Pearson's correlation and associated significance using AR(1) simulations
    
    '''
    r = pearsonr(y1, y2)[0]

    # generate phase-randomized samples using the Theiler & Prichard method
    Y1surr = phaseran(y1, nsim)
    Y2surr = phaseran(y2, nsim)

    # compute correlations
    Y1s = preprocessing.scale(Y1surr)
    Y2s = preprocessing.scale(Y2surr)

    n = np.size(y1)
    C = np.dot(np.transpose(Y1s), Y2s) / (n-1)
    rSim = np.diag(C)

    # compute fraction of values higher than observed
    F = np.sum(np.abs(rSim) >= np.abs(r)) / nsim

    # establish significance
    signif = F < alpha  # significant or not?

    return r, signif, F

def phaseran(recblk, nsurr):
    ''' Simultaneous phase randomization of a set of time series
    
    It creates blocks of surrogate data with the same second order properties as the original
    time series dataset by transforming the oriinal data into the frequency domain, randomizing the
    phases simultaneoulsy across the time series and converting the data back into the time domain. 
    
    Written by Carlos Gias for MATLAB

    http://www.mathworks.nl/matlabcentral/fileexchange/32621-phase-randomization/content/phaseran.m

    Parameters
    ----------

    recblk : numpy array
        2D array , Row: time sample. Column: recording.
        An odd number of time samples (height) is expected.
        If that is not the case, recblock is reduced by 1 sample before the surrogate data is created.
        The class must be double and it must be nonsparse.
    
    nsurr : int
        is the number of image block surrogates that you want to generate.

    Returns
    -------

    surrblk : numpy array
        3D multidimensional array image block with the surrogate datasets along the third dimension

    References
    ----------

    - Prichard, D., Theiler, J. Generating Surrogate Data for Time Series with Several Simultaneously Measured Variables (1994)
    Physical Review Letters, Vol 73, Number 7
    
    - Carlos Gias (2020). Phase randomization, MATLAB Central File Exchange
    '''
    # Get parameters
    nfrms = recblk.shape[0]

    if nfrms % 2 == 0:
        nfrms = nfrms-1
        recblk = recblk[0:nfrms]

    len_ser = int((nfrms-1)/2)
    interv1 = np.arange(1, len_ser+1)
    interv2 = np.arange(len_ser+1, nfrms)

    # Fourier transform of the original dataset
    fft_recblk = np.fft.fft(recblk)

    surrblk = np.zeros((nfrms, nsurr))

    #  for k in tqdm(np.arange(nsurr)):
    for k in np.arange(nsurr):
        ph_rnd = np.random.rand(len_ser)

        # Create the random phases for all the time series
        ph_interv1 = np.exp(2*np.pi*1j*ph_rnd)
        ph_interv2 = np.conj(np.flipud(ph_interv1))

        # Randomize all the time series simultaneously
        fft_recblk_surr = np.copy(fft_recblk)
        fft_recblk_surr[interv1] = fft_recblk[interv1] * ph_interv1
        fft_recblk_surr[interv2] = fft_recblk[interv2] * ph_interv2

        # Inverse transform
        surrblk[:, k] = np.real(np.fft.ifft(fft_recblk_surr))

    return surrblk

''' The FDR procedures translated from fdr.R by Dr. Chris Paciorek (https://www.stat.berkeley.edu/~paciorek/research/code/code.html)
'''
def fdr_basic(pvals,qlevel=0.05):
    ''' The basic FDR of Benjamini & Hochberg (1995).

    Parameters
    ----------

    pvals : list or array
        A vector of p-values on which to conduct the multiple testing.

    qlevel : float
        The proportion of false positives desired.

    Returns
    -------

    fdr_res : array or None
        A vector of the indices of the significant tests; None if no significant tests

    References
    ----------
    
    - Benjamini, Yoav; Hochberg, Yosef (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing". Journal of the Royal Statistical Society, Series B. 57 (1): 289–300. MR 1325392
    

    '''

    n = len(pvals)
    sorted_pvals = np.sort(pvals)
    sort_index = np.argsort(pvals)
    indices = np.arange(1, n+1)*(sorted_pvals <= qlevel*np.arange(1, n+1)/n)
    num_reject = np.max(indices)

    if num_reject:
        indices = np.arange(num_reject)
        fdr_res =  np.sort(sort_index[indices])
    else:
        fdr_res = None

    return fdr_res

def fdr_master(pvals, qlevel=0.05, method='original'):
    ''' Perform various versions of the FDR procedure

    Parameters
    ----------

    pvals : list or array
        A vector of p-values on which to conduct the multiple testing.

    qlevel : float
        The proportion of false positives desired.

    method : {'original', 'general'}
        Method for performing the testing.
        - 'original' follows Benjamini & Hochberg (1995);
        - 'general' is much more conservative, requiring no assumptions on the p-values (see Benjamini & Yekutieli (2001)).
        We recommend using 'original', and if desired, using 'adj_method="mean"' to increase power.

    Returns
    -------

    fdr_res : array or None
        A vector of the indices of the significant tests; None if no significant tests

    References
    ----------
    
    - Benjamini, Yoav; Hochberg, Yosef (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing". Journal of the Royal Statistical Society, Series B. 57 (1): 289–300. MR 1325392
    
    - Benjamini, Yoav; Yekutieli, Daniel (2001). "The control of the false discovery rate in multiple testing under dependency". Annals of Statistics. 29 (4): 1165–1188. doi:10.1214/aos/1013699998 

    '''
    if method == 'general':
        n = len(pvals)
        qlevel = qlevel / np.sum(1/np.arange(1, n+1))

    fdr_res = fdr_basic(pvals, qlevel)
    return fdr_res

def storey(edf_quantile, pvals):
    ''' The basic Storey (2002) estimator of a, the proportion of alternative hypotheses.

    Parameters
    ----------

    edf_quantile : float
        The quantile of the empirical distribution function at which to estimate a.

    pvals : list or array
        A vector of p-values on which to estimate a

    Returns
    -------

    a : int
        estimate of a, the number of alternative hypotheses
        
    References
    ----------
    
    - Storey, J.D., 2002, A direct approach to False Discovery Rates. Journal of the Royal Statistical Society, Series B, 64, 3, 479-498

    '''
    if edf_quantile >= 1 or edf_quantile <= 0:
        raise ValueError(f'Wrong edf_quantile: {edf_quantile}; must be within (0, 1)!')

    pvals = np.array(pvals)
    a = (np.mean(pvals<=edf_quantile) - edf_quantile) / (1 - edf_quantile)
    a = np.max(a, 0)  # set to 0 if a is negative
    return a

def prop_alt(pvals, adj_method='mean', adj_args={'edf_lower': 0.8, 'num_steps': 20}):
    ''' Calculate an estimate of a, the proportion of alternative hypotheses, using one of several methods

    Parameters
    ----------

    pvals : list or array
        A vector of p-values on which to estimate a


    adj_method: {'mean', 'storey', 'two-stage'}
        Method for increasing the power of the procedure by estimating the proportion of alternative p-values.
        - 'mean', the modified Storey estimator that we suggest in Ventura et al. (2004)
        - 'storey', the method of Storey (2002)
        - 'two-stage', the iterative approach of Benjamini et al. (2001)

    adj_args : dict
        - for "mean", specify "edf_lower", the smallest quantile at which to estimate a, and "num_steps", the number of quantiles to use
          the approach uses the average of the Storey (2002) estimator for the num_steps quantiles starting at "edf_lower" and finishing just less than 1
        - for "storey", specify "edf_quantile", the quantile at which to calculate the estimator
        - for "two-stage", the method uses a standard FDR approach to estimate which p-values are significant
          this number is the estimate of a; therefore the method requires specification of qlevel,
          the proportion of false positives and "fdr_method" ('original' or 'general'), the FDR method to be used.
          We do not recommend 'general' as this is very conservative and will underestimate a.

    Returns
    -------

    a : int
        estimate of a, the number of alternative hypotheses
    
    References
    ----------
    
    - Storey, J.D. (2002). A direct approach to False Discovery Rates. Journal of the Royal Statistical Society, Series B, 64, 3, 479-498
    
    - Benjamini, Yoav; Yekutieli, Daniel (2001). "The control of the false discovery rate in multiple testing under dependency". Annals of Statistics. 29 (4): 1165–1188. doi:10.1214/aos/1013699998 
    
    - Ventura, V., Paciorek, C., Risbey, J.S. (2004). Controlling the proportion of falsely rejected hypotheses when conducting multiple tests with climatological data. Journal of climate, 17, 4343-4356

    '''
    n = len(pvals)
    if adj_method == 'two-stage':
        fdr_res = fdr_master(pvals, adj_method['qlevel'], adj_args['fdr_method'])
        a = len(fdr_res)/n
        return a

    elif adj_method == 'storey':
        if 'edf_quantile' not in adj_args:
            raise ValueError('`edf_quantile` must be specified in `adj_args`!')

        a = storey(adj_args['edf_quantile'], pvals)
        return a

    elif adj_method == 'mean':
        if adj_args['edf_lower']>=1 or adj_args['edf_lower']<=0:
            raise ValueError(f'Wrong edf_lower: {adj_args["edf_lower"]}; must be within (0, 1)!')

        if adj_args['num_steps']<1 or type(adj_args['num_steps']) is not int:
            raise ValueError(f'Wrong num_steps: {adj_args["num_steps"]}; must be an integer >= 1')

        stepsize = (1 - adj_args['edf_lower']) / adj_args['num_steps']

        edf_quantiles = np.linspace(adj_args['edf_lower'], adj_args['edf_lower']+stepsize*(adj_args['num_steps']-1), adj_args['num_steps'])
        a_vec = [storey(edf_q, pvals) for edf_q in edf_quantiles]
        a = np.mean(a_vec)
        return a

    else:
        raise ValueError('Wrong method: {method}!')