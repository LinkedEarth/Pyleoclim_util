#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:46:59 2017

@author: deborahkhider

Statistics toolbox for pyleoclim

"""

import numpy as np
from scipy.stats import pearsonr
from scipy.stats.mstats import gmean
from scipy.stats import t as stu
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from sklearn import preprocessing
from tqdm import tqdm


"""
Simple stats
"""


def simpleStats(y, axis=None):
    """ Computes simple statistics

    Computes the mean, median, min, max, standard deviation, and interquartile
    range of a numpy array y.

    Args:
        y (array): A Numpy array
        axis (int, typle of ints): Optional. Axis or Axes along which the means
            are computed, the default is to compute the mean of the flattened
            array. If a tuple of ints, performed over multiple axes

    Returns:
        The mean, median, min, max, standard deviation and IQR by columns

    """
    # make sure that y is an array
    y = np.array(y, dtype='float64')

    # Perform the various calculations
    mean = np.nanmean(y, axis=axis)
    std = np.nanstd(y, axis=axis)
    median = np.nanmedian(y, axis=axis)
    min_ = np.nanmin(y, axis=axis)
    max_ = np.nanmax(y, axis=axis)
    IQR = np.nanpercentile(y, 75, axis=axis) - np.nanpercentile(y, 25, axis=axis)

    return mean, median, min_, max_, std, IQR

"""
Correlation
"""


class Correlation(object):
    """ Estimates the significance of correlations
    """
    def corr_sig(self, y1, y2, nsim=1000, method='isospectral', alpha=0.5):
        """ Estimates the significance of correlations between non IID time series by 3 independent methods:
        1) 'ttest': T-test where d.o.f are corrected for the effect of serial correlation
        2) 'isopersistent': AR(1) modeling of x and y.
        3) 'isospectral': phase randomization of original inputs. (default)
        The T-test is parametric test, hence cheap but usually wrong except in idyllic circumstances.
        The others are non-parametric, but their computational requirements scales with nsim.

        Args:
            y1, y2 (array)- vector of (real) numbers of identical length, no NaNs allowed
            nsim (int)- the number of simulations [1000]
            method (str)- methods 1-3 above ['isospectral']
            alpha (float)- significance level for critical value estimation [0.05]

        Returns:
             r (real): correlation between x and y \n
             signif (boolean): true (1) if significant; false (0) otherwise \n
             p (real): Fraction of time series with higher correlation coefficents than observed (approximates the p-value). \n
                Note that signif = True if and only if p <= alpha.
        """
        y1 = np.array(y1, dtype=float)
        y2 = np.array(y2, dtype=float)

        assert np.size(y1) == np.size(y2), 'The size of X and the size of Y should be the same!'

        if method == 'ttest':
            (r, signif, p) = self.corr_ttest(y1, y2, alpha=alpha)
        elif method == 'isopersistent':
            (r, signif, p) = self.corr_isopersist(y1, y2, alpha=alpha, nsim=nsim)
        elif method == 'isospectral':
            (r, signif, p) = self.corr_isospec(y1, y2, alpha=alpha, nsim=nsim)

        return r, signif, p

    def corr_ttest(self, y1, y2, alpha=0.05):
        """ Estimates the significance of correlations between 2 time series using
        the classical T-test with degrees of freedom modified for autocorrelation.
        This function creates 'nsim' random time series that have the same power
        spectrum as the original time series but with random phases.

        Args:
            y1, y2 (array): vectors of (real) numbers with identical length, no NaNs allowed
            alpha (real): significance level for critical value estimation [default: 0.05]

        Returns:
            r (real)- correlation between x and y \n
            signif (boolean)- true (1) if significant; false (0) otherwise \n
            pval (real)- test p-value (the probability of the test statstic exceeding the observed one by chance alone)
        """
        r = pearsonr(y1, y2)[0]

        g1 = self.ar1_fit(y1)
        g2 = self.ar1_fit(y2)

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

    def corr_isopersist(self, y1, y2, alpha=0.05, nsim=1000):
        ''' Computes correlation between two timeseries, and their significance.
        The latter is gauged via a non-parametric (Monte Carlo) simulation of
        correlations with nsim AR(1) processes with identical persistence
        properties as x and y ; the measure of which is the lag-1 autocorrelation (g).

        Args:
            y1, y2 (array): vectors of (real) numbers with identical length, no NaNs allowed
            alpha (real): significance level for critical value estimation [default: 0.05]
            nsim (int): number of simulations [default: 1000]

        Returns:
            r (real) - correlation between x and y \n
            signif (boolean) - true (1) if significant; false (0) otherwise \n
            pval (real) - test p-value (the probability of the test statstic exceeding the observed one by chance alone)

        Remarks:
            The probability of obtaining a test statistic at least as extreme as the one actually observed,
            assuming that the null hypothesis is true. \n
            The test is 1 tailed on |r|: Ho = { |r| = 0 }, Ha = { |r| > 0 } \n
            The test is rejected (signif = 1) if pval <= alpha, otherwise signif=0; \n
            (Some Rights Reserved) Hepta Technologies, 2009 \n
            v1.0 USC, Aug 10 2012, based on corr_signif.m
        '''

        r = pearsonr(y1, y2)[0]
        ra = np.abs(r)

        y1_red, g1 = self.isopersistent_rn(y1, nsim)
        y2_red, g2 = self.isopersistent_rn(y2, nsim)

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

    def isopersistent_rn(self, X, p):
        ''' Generates p realization of a red noise [i.e. AR(1)] process
        with same persistence properties as X (Mean and variance are also preserved).

        Args:
            X (array): vector of (real) numbers as a time series, no NaNs allowed
            p (int): number of simulations

        Returns:
            red (matrix) - n rows by p columns matrix of an AR1 process, where n is the size of X \n
            g (real) - lag-1 autocorrelation coefficient

        Remarks:
            (Some Rights Reserved) Hepta Technologies, 2008
        '''
        n = np.size(X)
        sig = np.std(X, ddof=1)

        g = self.ar1_fit(X)
        #  red = red_noise(N, M, g)
        red = self.ar1_sim(n, p, g, sig)

        return red, g

    def ar1_fit(self, ts):
        ''' Return the lag-1 autocorrelation from ar1 fit.

        Args:
            ts (array): vector of (real) numbers as a time series

        Returns:
            g (real): lag-1 autocorrelation coefficient
        '''


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

    def red_noise(self, N, M, g):
        ''' Produce M realizations of an AR1 process of length N with lag-1 autocorrelation g

        Args:
            N, M (int): dimensions as N rows by M columns
            g (real): lag-1 autocorrelation coefficient

        Returns:
            red (matrix): N rows by M columns matrix of an AR1 process

        Remarks:
            (Some Rights Reserved) Hepta Technologies, 2008
            J.E.G., GaTech, Oct 20th 2008
        '''
        red = np.zeros(shape=(N, M))
        red[0, :] = np.random.randn(1, M)
        for i in np.arange(1, N):
            red[i, :] = g * red[i-1, :] + np.random.randn(1, M)

        return red

    def corr_isospec(self, y1, y2, alpha=0.05, nsim=1000):
        ''' Phase randomization correltation estimates

        Estimates the significance of correlations between non IID
        time series by phase randomization of original inputs.
        This function creates 'nsim' random time series that have the same power
        spectrum as the original time series but random phases.

        Args:
            y1, y2 (array): vectors of (real) numbers with identical length, no NaNs allowed
            alpha (real): significance level for critical value estimation [default: 0.05]
            nsim (int): number of simulations [default: 1000]

        Returns:
            r (real): correlation between y1 and y2 \n
            signif (boolean): true (1) if significant; false (0) otherwise \n
            F : Fraction of time series with higher correlation coefficents than observed (approximates the p-value).

        References:
            - Ebisuzaki, W, 1997: A method to estimate the statistical
            significance of a correlation when the data are serially correlated.
            J. of Climate, 10, 2147-2153.
            - Prichard, D., Theiler, J. Generating Surrogate Data for Time Series
            with Several Simultaneously Measured Variables (1994)
            Physical Review Letters, Vol 73, Number 7
            (Some Rights Reserved) USC Climate Dynamics Lab, 2012.
        '''
        r = pearsonr(y1, y2)[0]

        # generate phase-randomized samples using the Theiler & Prichard method
        Y1surr = self.phaseran(y1, nsim)
        Y2surr = self.phaseran(y2, nsim)

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

    def phaseran(self, recblk, nsurr):
        ''' Phaseran by Carlos Gias

        http://www.mathworks.nl/matlabcentral/fileexchange/32621-phase-randomization/content/phaseran.m

        Args:
            recblk (2D array): Row: time sample. Column: recording.
                An odd number of time samples (height) is expected.
                If that is not the case, recblock is reduced by 1 sample before the surrogate data is created.
                The class must be double and it must be nonsparse.
            nsurr (int): is the number of image block surrogates that you want to generate.

        Returns:
            surrblk: 3D multidimensional array image block with the surrogate datasets along the third dimension

        Reference:
            Prichard, D., Theiler, J. Generating Surrogate Data for Time Series with Several Simultaneously Measured Variables (1994)
            Physical Review Letters, Vol 73, Number 7
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

        for k in tqdm(np.arange(nsurr)):
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


def corrsig(y1, y2, nsim=1000, method='isospectral', alpha=0.5):
    """
    Estimates the significance of correlations between non IID time series by 3 independent methods:
        1) 'ttest': T-test where d.o.f are corrected for the effect of serial correlation
        2) 'isopersistent': AR(1) modeling of x and y.
        3) 'isospectral': phase randomization of original inputs. (default)
        The T-test is parametric test, hence cheap but usually wrong except in idyllic circumstances.
        The others are non-parametric, but their computational requirements scales with nsim.

    Args:
        y1, y2 (array)- vector of (real) numbers of identical length, no NaNs allowed
        nsim (int)- the number of simulations [1000]
        method (str)- methods 1-3 above ['isospectral']
        alpha (float)- significance level for critical value estimation [0.05]

    Returns:
         r (real): correlation between x and y \n
         signif (int): true  if significant; false otherwise \n
         p (real): Fraction of time series with higher correlation coefficents than observed (approximates the p-value). \n
            Note that signif = True if and only if p <= alpha.
"""
    corr = Correlation()
    r, signif, p = corr.corr_sig(y1,y2, nsim = nsim, method = method,
                                 alpha = alpha)

    return r, signif, p
