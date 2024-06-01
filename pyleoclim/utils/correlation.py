#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relevant functions for correlation analysis
"""

__all__ = [
    'corr_sig',
    'fdr',
    'association'
]

import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
from .tsmodel import ar1_fit_evenly, isopersistent_rn
from .tsutils import phaseran


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
        This is a non-parametric method, assuming only wide-sense stationarity
        
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
        
    method : str; {'ttest','isopersistent','isospectral' (default)}
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
         
    See also
    --------

    pyleoclim.utils.correlation.corr_ttest : Estimates the significance of correlations between 2 time series using the classical T-test adjusted for effective sample size
    pyleoclim.utils.correlation.corr_isopersist : Computes correlation between two timeseries, and their significance using Ar(1) modeling
    pyleoclim.utils.correlation.corr_isospec : Estimates the significance of the correlation using phase randomization
    pyleoclim.utils.correlation.fdr : Determine significance based on the false discovery rate
     
    """
    y1 = np.array(y1, dtype=float)
    y2 = np.array(y2, dtype=float)

    assert np.size(y1) == np.size(y2), 'The size of y1 and y2 should be the same'

    if method == 'ttest':
        (r, signif, p) = corr_ttest(y1, y2, alpha=alpha)
    elif method == 'isopersistent':
        
        (r, signif, p) = corr_isopersist(y1, y2, alpha=alpha, nsim=nsim)
    elif method == 'isospectral':
        (r, signif, p) = corr_isospec(y1, y2, alpha=alpha, nsim=nsim)
        
        
    # apply this syntax:
    # wave_res = wave_func[method](self.value, self.time, **args[method])

    res={'r':r,'signif':signif,'p':p}    
    
    return res

def fdr(pvals, qlevel=0.05, method='original', adj_method=None, adj_args={}):
    ''' Determine significance based on the false discovery rate
    
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

    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    

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
        
    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    pyleoclim.utils.correlation.corr_isopersist : Estimate Pearson's correlation and associated significance using AR(1)
    pyleoclim.utils.correlation.corr_isospec : Estimate Pearson's correlation and associated significance using phase randomization
    pyleoclim.utils.correlation.fdr : Determine significance based on the false discovery rate

    """
    r = stats.pearsonr(y1, y2)[0]

    g1 = ar1_fit_evenly(y1)
    g2 = ar1_fit_evenly(y2)

    N = np.size(y1)

    Ney1 = N * (1-g1) / (1+g1)
    Ney2 = N * (1-g2) / (1+g2)

    Ne = stats.mstats.gmean([Ney1+Ney2])
    assert Ne >= 10, 'Too few effective d.o.f. to apply this method!'

    df = Ne - 2
    t = np.abs(r) * np.sqrt(df/(1-r**2))

    pval = 2 * stats.t.cdf(-np.abs(t), df)

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
    
    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    pyleoclim.utils.correlation.corr_ttest: Estimates Pearson's correlation and associated significance using a t-test
    pyleoclim.utils.correlation.corr_isospec : Estimates Pearson's correlation and associated significance using 
    pyleoclim.utils.correlation.fdr : Determine significance based on the false discovery rate

    '''

    r = stats.pearsonr(y1, y2)[0]
    ra = np.abs(r)

    y1_red, g1 = isopersistent_rn(y1, nsim)
    y2_red, g2 = isopersistent_rn(y2, nsim)

    rs = np.zeros(nsim)
    for i in np.arange(nsim):
        rs[i] = stats.pearsonr(y1_red[:, i], y2_red[:, i])[0]

    rsa = np.abs(rs)

    xi = np.linspace(0, 1.1*np.max([ra, np.max(rsa)]), 200)
    kde = stats.gaussian_kde(rsa)
    prob = kde(xi).T

    diff = np.abs(ra - xi)
    #  min_diff = np.min(diff)
    pos = np.argmin(diff)

    pval = np.trapz(prob[pos:], xi[pos:])

    rcrit = np.percentile(rsa, 100*(1-alpha))
    signif = ra >= rcrit

    return r, signif, pval


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

    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    pyleoclim.utils.correlation.corr_ttest : Estimates Pearson's correlation and associated significance using a t-test
    pyleoclim.utils.correlation.corr_isopersist : Estimates Pearson's correlation and associated significance using AR(1) simulations
    pyleoclim.utils.correlation.fdr : Determine significance based on the false discovery rate
    
    References
    ----------

    - Ebisuzaki, W, 1997: A method to estimate the statistical significance of a correlation when the data are serially correlated. J. of Climate, 10, 2147-2153.
    
    - Prichard, D., Theiler, J. Generating Surrogate Data for Time Series with Several Simultaneously Measured Variables (1994) Physical Review Letters, Vol 73, Number 7 (Some Rights Reserved) USC Climate Dynamics Lab, 2012.
    '''
    r = stats.pearsonr(y1, y2)[0]

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

    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    pyleoclim.utils.correlation.fdf : Determine significance based on the false discovery rate

    References
    ----------
    
    Benjamini, Yoav; Hochberg, Yosef (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing". Journal of the Royal Statistical Society, Series B. 57 (1): 289–300. MR 1325392
    

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

    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    pyleoclim.utils.correlation.fdf : Determine significance based on the false discovery rate

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

    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    pyleoclim.utils.correlation.fdf : Determine significance based on the false discovery rate

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


    adj_method: str; {'mean', 'storey', 'two-stage'}
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

    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Estimates the Pearson's correlation and associated significance between two non IID time series
    pyleoclim.utils.correlation.fdf : Determine significance based on the false discovery rate

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

def cov_shrink_rblw(S, n):
    """Compute a shrinkage estimate of the covariance matrix using
    the Rao-Blackwellized Ledoit-Wolf estimator described by Chen et al. 2011 [1]

    Contributed by `Robert McGibbon <https://rmcgibbo.org/>`_.

    Parameters
    ----------
    S : array, shape=(n, n)
        Sample covariance matrix (e.g. estimated with np.cov(X.T))

    n : int
        Number of data points used in the estimate of S.

    Returns
    -------
    sigma : array, shape=(p, p)
        Estimated shrunk covariance matrix

    shrinkage : float
        The applied covariance shrinkage intensity.

    Notes
    -----

    See the `covar documentation <https://pythonhosted.org/covar/generated/covar.cov_shrink_rblw.html>`_ for 
    math details.


    References
    ----------

    - Y. Chen, A. Wiesel and A. O. Hero (2011), Robust Shrinkage Estimation of High-Dimensional Covariance Matrices, IEEE Transactions on Signal Processing, vol. 59, no. 9, pp. 4097-4107, doi:10.1109/TSP.2011.2138698

    See Also
    --------
  
    sklearn.covariance.ledoit_wolf : very similar approach using the same
        shrinkage target, :math:`T`, but a different method for estimating the
        shrinkage intensity, :math:`gamma`.
    """

    p = S.shape[0]

    if S.shape[1] != p:
        raise ValueError('S must be a (p x p) matrix')

    alpha = (n-2)/(n*(n+2))
    beta = ((p+1)*n - 2) / (n*(n+2))

    trace_S  = 0  # np.trace(S)
    trace_S2 = 0  # np.trace(S.dot(S))

    for i in range(p):
        trace_S += S[i,i]
        for j in range(p):
            trace_S2 += S[i,j]*S[i,j]

    U = ((p * trace_S2 / (trace_S*trace_S)) - 1)
    rho = min(alpha + beta/U, 1)

    F = (trace_S / p) * np.eye(p)

    return (1-rho)*np.asarray(S) + rho*F, rho

def association(y1, y2, statistic='pearsonr',settings=None):
    '''
    Quantify the strength of a relationship (e.g. linear) between paired observations y1 and y2.

    Parameters
    ----------
    y1 : array, length n
        vector of (real) numbers of same length as y2, no NaNs allowed  
    y2 : array, length n
        vector of (real) numbers of same length as y1, no NaNs allowed
    statistic : str, optional
        The statistic used to measure the association, to be chosen from a subset of
        https://docs.scipy.org/doc/scipy/reference/stats.html#association-correlation-tests
        ['pearsonr','spearmanr','pointbiserialr','kendalltau','weightedtau']
        The default is 'pearsonr'.
    settings : dict, optional
        optional arguments to modify the behavior of the SciPy association functions

    Raises
    ------
    ValueError
        Complains loudly if the requested statistic is not from the list above. 

    Returns
    -------
    res : instance result class
        structure containing the result. The first element (res[0]) is always the statistic.
    '''
    y1 = np.array(y1, dtype=float)
    y2 = np.array(y2, dtype=float)
    assert np.size(y1) == np.size(y2), 'The size of y1 and y2 should be the same'
    
    args = {} if settings is None else settings.copy()
    acceptable_methods = ['linregress','pearsonr','spearmanr','pointbiserialr','kendalltau','weightedtau']
    if statistic in acceptable_methods:
        func = getattr(stats, statistic) 
        res = func(y1,y2,**args)
    else:
        raise ValueError(f'Wrong statistic: {statistic}; acceptable choices are {acceptable_methods}')
       
    return res
        
        
        