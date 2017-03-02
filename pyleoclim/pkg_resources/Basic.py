# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 2017

@author: deborahkhider

Basic functionalities for science rather than file manipulation.
Make sure this module is imported for the use of static methods. 

"""

import lipd as lpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sys
import os
from matplotlib import gridspec
from scipy.stats import pearsonr
from scipy.stats.mstats import gmean
from scipy.stats import t as stu
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from sklearn import preprocessing
import progressbar

#Import internal packages to pyleoclim
from .LiPDutils import *

class Basic(object):
    """ Basic manipulation of timeseries for scientific purpose.
    
    Calculates statistics of timeseries, bin or interpolate data
    
    """
    
    def __init__(self,timeseries_list):

        self.TS = timeseries_list

    @staticmethod
    def getValues(timeseries):
        """Get the paleoData values from the timeseries object
        
        Args:
            timeseries: a single timeseries object. Use getTSO() to get
                one from the dictionary
        """
        values_key =[]
        for key, val in timeseries.items():
            if "values" in key.lower():
                values_key.append(key)
        
        values = timeseries[values_key[0]]

        return values             
    
        
    def simpleStats(self, timeseries=""):
        """ Compute the mean and standard deviation of a time series
        
        Args:
            timeseries: a single timeseries. Will prompt for one 
                if not available

        Returns:
            the mean, median, min, max, standard deviation and IQR of a timeseries.
                
        """        
        # get the values
        if not timeseries:
            timeseries = getTs(self.TS)

        values = Basic.getValues(timeseries)    
     
        mean = np.nanmean(values)
        std = np.nanstd(values)
        median = np.nanmedian(values)
        min_ = np.nanmin(values)
        max_ = np.nanmax(values)
        IQR = np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
        
        return mean, median, min_, max_, std, IQR
    
    @staticmethod    
    def bin_Ts(timeseries, x_axis = "", bin_size = "", start = "", end = ""):
        """Bin the PaleoData values
        
        Args:
            timeseries: a single timeseries object. Use getTSO() to get one.
            x-axis (str): The representation against which to plot the 
                paleo-data. Options are "age", "year", and "depth". Default
                is to let the system choose if only one available or prompt 
                the user. 
            bin_size (float): the size of the bins to be used. 
                By default, will prompt for one
            start (float): Start time/age/depth. Default is the minimum 
            end (float): End time/age/depth. Default is the maximum
            
        Returns:
            binned_data - the binned output \n
            bins - the bins (centered on the median, i.e. the 100-200 bin is 150) \n
            n - number of data points in each bin \n
            error - the standard error on the mean in each bin
            
        """
        
        # Get the values
        values = Basic.getValues(timeseries)
        
        # Get the time (or depth) representation
        if not x_axis:
            time, label = xAxisTs(timeseries)
        elif x_axis == 'age':
            time = timeseries['age']
            label = 'age'
        elif x_axis == "year":
            time = timeseries['year']
            label = 'year'
        elif x_axis == 'depth':
            time = timeseries['depth']
            label = 'depth'
        else:
            sys.exit("Enter either 'age', 'year', or 'depth'")
            
        # Get the units
        if label == "age":
            if "ageUnits" in timeseries.keys():
                units = timeseries["ageUnits"]
            else: units = 'NA'    
        elif label == "year":
            if "yearUnits" in timeseries.keys():
                units = timeseries["yearUnits"]
            else: units = 'NA'
        elif label == "depth":
            if "depthUnits" in timeseries.keys():
                units = timeseries["depthUnits"]
            else: units = 'NA'

        # Check for bin_size, startdate and enddate
        if not bin_size:
            print("The " + label + " is expressed in " + units)
            bin_size = float(input("What bin size would you like to use? "))
        if type(start) is str:
            start = np.nanmin(np.asarray(time))
        if type(end) is str:
            end = np.nanmax(np.asarray(time))
        
        # set the bin medians
        bins = np.arange(start+bin_size/2, end + bin_size/2, bin_size)
        
        # Calculation
        binned_data=[]
        n = []
        error = []
        for x in np.nditer(bins):
            idx = [idx for idx,c in enumerate(time) if c>=(x-bin_size/2) and c<(x+bin_size/2)]     
            if np.asarray(values)[idx].size==0:
                binned_data.append(np.nan)
                n.append(np.nan)
                error.append(np.nan)
            else:                      
                binned_data.append(np.nanmean(np.asarray(values)[idx]))
                n.append(np.asarray(values)[idx].size)
                error.append(np.nanstd(np.asarray(values)[idx]))
        
        return bins, binned_data, n, error

    @staticmethod
    def interp_Ts(timeseries, x_axis = "", interp_step="",start ="",\
                    end =""):
        """Linear interpolation of the PaleoData values
        
        Args:
            timeseries: a timeseries object. Use getTSO() to get one.
            x-axis (str): The representation against which to plot the 
                paleo-data. Options are "age", "year", and "depth". 
                Default is to let the system choose if only one available 
                or prompt the user. 
            interp_step (float): the step size. By default, will prompt the user. 
            start (float): Start time/age/depth. Default is the minimum 
            end (float): End time/age/depth. Default is the maximum
            
        Returns:
            interp_age - the interpolated age/year/depth according to 
            the end/start and time step \n
            interp_values - the interpolated values
            
        """
            
        # Get the values and age
        values = Basic.getValues(timeseries)
        # Get the time (or depth) representation
        if not x_axis:
            time, label = xAxisTs(timeseries)
        elif x_axis == 'age':
            time = timeseries['age']
            label = 'age'
        elif x_axis == "year":
            time = timeseries['year']
            label = 'year'
        elif x_axis == 'depth':
            time = timeseries['depth']
            label = 'depth'
        else:
            sys.exit("Enter either 'age', 'year', or 'depth'")
            
        # Get the units
        if label == "age":
            if "ageUnits" in timeseries.keys():
                units = timeseries["ageUnits"]
            else: units = 'NA'    
        elif label == "year":
            if "yearUnits" in timeseries.keys():
                units = timeseries["yearUnits"]
            else: units = 'NA'
        elif label == "depth":
            if "depthUnits" in timeseries.keys():
                units = timeseries["depthUnits"]
            else: units = 'NA'

        # Check for interp_step, startdate and enddate
        if not interp_step:
            print("The " + label + " is expressed in " + units)
            interp_step = float(input("What interpolation step would you like to use? "))
        if type(start) is str:
            start = np.nanmin(np.asarray(time))
        if type(end) is str:
            end = np.nanmax(np.asarray(time))
        
        # Calculation
        interp_age = np.arange(start,end,interp_step)
        data = pd.DataFrame({"x-axis": time, "y-axis": values}).sort_values('x-axis') 
        val_idx = valuesLoc(data)
        data = data.iloc[val_idx,:]
        interp_values =np.interp(interp_age,data['x-axis'],data['y-axis'])
        
        return interp_age, interp_values

    @staticmethod
    def onCommonXAxis(timeseries1, timeseries2, x_axis = "", interp_step="",\
                      start ="", end =""):
        """ Places two timeseries on a common x-axis
        
        Interpolate the PaleoDataValues of two timeseries on a common x-axis.
        
        Args:
            timeseries1: a timeseries object. Use getTSO() to get one.
            timeseries2: a timeseries object. Use getTSO() to get one.
            x-axis (str): The representation against which to express the 
                paleo-data. Options are "age", "year", and "depth". 
                Default is to let the system choose if only one available 
                or prompt the user. 
            interp_step (float): the step size. By default, will prompt the user. 
            start (float): Start time/age/depth. Default is the maximum of 
                the minima of the two timeseries
            end (float): End time/age/depth. Default is the minimum of the 
                maxima of the two timeseries
                
        Returns:
            interp_age - the interpolated age/year/depth according to 
            the end/start and time step \n
            interp_values1 - the interpolated values for the first timeseries \n
            interp_values2 - the interpolated values for the second timeseries \n
        
        """
        # Make sure there is an x-axis representation
        if not x_axis:
            # get the axis from the first timeseries
            x_axis_val, x_axis = xAxisTs(timeseries1)
            # make sure it's available in the other timeseries
            if not x_axis in timeseries2.keys():
                sys.exit(x_axis+" is not available in the second series, select another\
                         representation")
        elif not x_axis in timeseries1.keys():
            sys.exit(x_axis+" is not available in the first series, select another\
                     representation")
        elif not x_axis  in timeseries2.keys():
            sys.exit(x_axis+" is not available in the second series, select another\
                     representation")
        
        # Get the values
        data1 = TsToDf(timeseries1,x_axis = x_axis)
        data2 = TsToDf(timeseries2,x_axis = x_axis)
        
        # get the index where the values are
        val_idx1 = valuesLoc(data1)
        val_idx2 = valuesLoc(data2)
        
        # remove the missing values        
        data1 = data1.iloc[val_idx1,:] 
        data2 = data2.iloc[val_idx2,:] 

        # find the min/max age
        if type(start) is str:
            start = np.max([np.min(data1[x_axis]), np.min(data2[x_axis])])
        if type(end) is str:
            end = np.min([np.max(data1[x_axis]), np.max(data2[x_axis])])
        
        # find the average resolution for interpolation step
        if type(interp_step) is str:
            interp_step = np.max([np.mean(data1[x_axis].diff(1)),\
                                   np.mean(data2[x_axis].diff(1))])
        
        # perform the interpolation
        # perform the interpolation
        interp_age = np.arange(start,end,interp_step)
        interp_values1 =np.interp(interp_age,data1[x_axis],data1['y-axis'])
        interp_values2 =np.interp(interp_age,data2[x_axis],data2['y-axis'])
        
        return interp_age, interp_values1, interp_values2

class Correlation(object):
    """ Perform correlation analysis between two timeseries
    """

    def __init__(self, timeseries_list):
        #Pass all the needed variables for the functions
        self.TS = timeseries_list

    def corr_sig(self,timeseries1 = "", timeseries2 = "", x_axis = "", \
                 interp_step = "", start = "", end = "", nsim = 1000, \
                 method = 'isospectral', alpha = 0.5):
        """ Estimates the significance of correlations between non IID timeseries.

        Function written by. F. Zhu.

        Args:
            timeseries1, timeseries2: timeseries object. Default is blank.
            x-axis (str): The representation against which to express the
                paleo-data. Options are "age", "year", and "depth".
                Default is to let the system choose if only one available
                or prompt the user.
            interp_step (float): the step size. By default, will prompt the user.
            start (float): Start time/age/depth. Default is the maximum of
                the minima of the two timeseries
            end (float): End time/age/depth. Default is the minimum of the
                maxima of the two timeseries
            nsim (int): the number of simulations. Default is 1000
            method (str): method use to estimate the correlation and significance
                Available methods include:
                    - 'ttest': T-test where the degrees of freedom are corrected for
                    the effect of serial correlation.\n
                    - 'isopersistant': AR(1) modeling of the two timeseries\n
                    - 'isospectral' (default): phase randomization of original
                    inputs.
                The T-test is parametric test, hence cheap but usually wrong 
                    except in idyllic circumstances.
                The others are non-parametric, but their computational 
                    requirements scales with nsim.
            alpha (float): significance level for critical value estimation. Default is 0.05

        Returns:
            r (float) - correlation between the two timeseries \n
            sig (bool) -  Returns True if sifnificant, False otherwise \n
            p (real) - the p-value
        """
        
        if not timeseries1:
            timeseries1 = getTs(self.TS)
        
        if not timeseries2:
            timeseries2 = getTs(self.TS)
            
        # check that both timeseries have the same number of datapoints and same limits
        if len(timeseries1['paleoData_values']) != len(timeseries2['paleoData_values']):
            print("The two timeseries do not contain the same number of points.\
            Interpolating...")
            age, values1, values2 = Basic.onCommonXAxis(timeseries1,timeseries2,\
                                                              x_axis=x_axis, \
                                                              interp_step = interp_step,\
                                                              start = start, end = end)
        elif not x_axis:
            x_axis_val, x_axis = xAxisTs(timeseries1)
            
            # make sure it's available in the other timeseries
            
            if not x_axis in timeseries2.keys():
                sys.exit(x_axis+" is not available in the second series, select another\
                                 representation") 
                
            # Check that the starting point is the same
            if not start:
                print("No start date given, finding the best interpolation scheme between\
                      the two timeseries")    
                age, values1, values2 = Basic.onCommonXAxis(timeseries1,timeseries2,\
                                                              x_axis=x_axis, \
                                                              interp_step = interp_step,\
                                                              start = start, end = end)
            elif np.nanmin(timeseries1[x_axis])!= start:
                print("The minimum age in the first timeseries doesn't conform to the\
                      given start date. Interpolating...")
                age, values1, values2 = Basic.onCommonXAxis(timeseries1,timeseries2,\
                                                              x_axis=x_axis, \
                                                              interp_step = interp_step,\
                                                              start = start, end = end)
            elif np.nanmin(timeseries2[x_axis])!= start:
                print("The minimum age in the second timeseries doesn't conform to the\
                      given start date. Interpolating...")
                age, values1, values2 = Basic.onCommonXAxis(timeseries1,timeseries2,\
                                                              x_axis=x_axis, \
                                                              interp_step = interp_step,\
                                                              start = start, end = end)
            
            # Check that the end point is correct     
            if not end:
                print("No end date given, finding the best interpolation scheme between\
                      the two timeseries")    
                age, values1, values2 = Basic.onCommonXAxis(timeseries1,timeseries2,\
                                                              x_axis=x_axis, \
                                                              interp_step = interp_step,\
                                                              start = start, end = end)
            elif np.nanmin(timeseries1[x_axis])!= end:
                print("The maximum age in the first timeseries doesn't conform to the\
                      given end date. Interpolating...")
                age, values1, values2 = Basic.onCommonXAxis(timeseries1,timeseries2,\
                                                              x_axis=x_axis, \
                                                              interp_step = interp_step,\
                                                              start = start, end = end)
            elif np.nanmin(timeseries2[x_axis])!= end:
                print("The maximum age in the second timeseries doesn't conform to the\
                      given end date. Interpolating...")
                age, values1, values2 = Basic.onCommonXAxis(timeseries1,timeseries2,\
                                                              x_axis=x_axis, \
                                                              interp_step = interp_step,\
                                                              start = start, end = end)       
        else:
            age = timeseries1[x_axis]
            values1 = timeseries1['paleoData_values']
            values2 = timeseries2['paleoData_values']

        values1 = np.array(values1, dtype = float)
        values2 = np.array(values2, dtype = float)
        
        assert np.size(values1) == np.size(values2)
        
        if method == 'ttest':
            (r, signif, p) = self.corr_ttest(values1, values2, alpha=alpha)
        elif method == 'isopersistent':
            (r, signif, p) = self.corr_isopersist(values1, values2, alpha=alpha, nsim=nsim)
        elif method == 'isospectral':
            (r, signif, p) = self.corr_isospec(values1, values2, alpha=alpha, nsim=nsim)
        else:
            sys.exist("Method should be either 'ttest', or 'isopersistent',\
            or 'isospectral'")
       
        return r, signif, p
        
        
    def corr_ttest(self,x, y, alpha=0.05):
        """ Estimates the significance of correlations between 2 time series.
        
        This function uses the classical T-test with degrees of freedom 
        modified for autocorrelation. This function creates 'nsim' random time 
        series that have the same power spectrum as the original time series 
        but with random phases.
        
        Args:
            x, y: vectors of (real) numbers with identical length, no NaNs 
                allowed
            alpha: significance level for critical value estimation 
                [default: 0.05]
                
        Returns:
            r - correlation between x and y \n
            signif - true (1) if significant; false (0) otherwise \n
            pval - test p-value (the probability of the test statstic exceeding 
            the observed one by chance alone)
        """
        r = pearsonr(x, y)[0]
    
        g1 = self.ar1_fit(x)
        g2 = self.ar1_fit(y)
    
        N = np.size(x)
    
        Nex = N * (1-g1) / (1+g1)
        Ney = N * (1-g2) / (1+g2)
    
        Ne = gmean([Nex+Ney])
        assert Ne >= 10, 'Too few effective d.o.f. to apply this method!'
    
        df = Ne - 2
        t = np.abs(r) * np.sqrt(df/(1-r**2))
    
        pval = 2 * stu.cdf(-np.abs(t), df)
    
        signif = pval <= alpha
    
        return r, signif, pval
    
    
    def corr_isopersist(self, x, y, alpha=0.05, nsim=1000):
        """Computes correlation between two timeseries, and their significance.
        
        Th e significance is gauged via a non-parametric (Monte Carlo) simulation of
        correlations with nsim AR(1) processes with identical persistence
        properties as x and y ; the measure of which is the lag-1 autocorrelation (g).
        
        Args:
            x, y (real): vectors of (real) numbers with identical length, no NaNs allowed
            alpha (real): significance level for critical value estimation [default: 0.05]
            nsim (int): number of simulations [default: 1000]
            
        Returns:
            r (real) - correlation between x and y \n
            signif (bool) - true (1) if significant; false (0) otherwise
            pval (real) - test p-value (the probability of the test statistic
            exceeding the observed one by chance alone)
            
        Notes:
            The probability of obtaining a test statistic at least as extreme as the one actually observed,
            assuming that the null hypothesis is true.
            The test is 1 tailed on |r|: Ho = { |r| = 0 }, Ha = { |r| > 0 }
            The test is rejected (signif = 1) if pval <= alpha, otherwise signif=0;
            (Some Rights Reserved) Hepta Technologies, 2009
            v1.0 USC, Aug 10 2012, based on corr_signif.m
        """
    
        r = pearsonr(x, y)[0]
        ra = np.abs(r)
    
        x_red, g1 = self.isopersistent_rn(x, nsim)
        y_red, g2 = self.isopersistent_rn(y, nsim)
    
        rs = np.zeros(nsim)
        for i in np.arange(nsim):
            rs[i] = pearsonr(x_red[:, i], y_red[:, i])[0]
    
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
    
    
    def isopersistent_rn(self, X, M):
        """ Generates M realization of a red noise [i.e. AR(1)] process
        with same persistence properties as X (Mean and variance are also preserved).
        
        Args:
            X (array): vector of (real) numbers as a time series, no NaNs allowed
            M (int): number of simulations
        
        Returns:
            red (matrix) - N rows by M columns matrix of an AR1 process\n
            g (real) - lag-1 autocorrelation coefficient
        
        Notes:
            (Some Rights Reserved) Hepta Technologies, 2008
        """
        N = np.size(X)
        mu = np.mean(X)
        sig = np.std(X, ddof=1)
    
        g = self.ar1_fit(X)
        red = self.red_noise(N, M, g)
        m = np.mean(red)
        s = np.std(red, ddof=1)
    
        red_n = (red - m) / s
        red_z = red_n * sig + mu
    
        return red_z, g
    
    
    def ar1_fit(self, ts):
        """ Return the lag-1 autocorrelation from ar1 fit.
        
        Args:
            ts (array): vector of (real) numbers as a time series
            
        Returns:
            g (real) - lag-1 autocorrelation coefficient
        """
        ar1_mod = sm.tsa.ARMA(ts, (1, 0)).fit()
        g = ar1_mod.params[1]
    
        return g
    
    
    def red_noise(self, N, M, g):
        """ Produce AR1 process matrix with nv = [N M] and lag-1 autocorrelation g.
        Args:
            N, M (int): dimensions as N rows by M columns
            g (real): lag-1 autocorrelation coefficient
            
        Returns:
            red (matrix): N rows by M columns matrix of an AR1 process
            
        Remarks:
            (Some Rights Reserved) Hepta Technologies, 2008
            J.E.G., GaTech, Oct 20th 2008
        """
        red = np.zeros(shape=(N, M))
        red[0, :] = np.random.randn(1, M)
        for i in np.arange(1, N):
            red[i, :] = g * red[i-1, :] + np.random.randn(1, M)
    
        return red
    
    
    def corr_isospec(self, x, y, alpha=0.05, nsim=1000):
        """ Estimates the significance of correlations between non IID
        time series by phase randomization of original inputs.
        
        This function creates 'nsim' random time series that have the same power
        spectrum as the original time series but random phases.
        
        Args:
            x, y (array): vectors of (real) numbers with identical length, no NaNs allowed
            alpha (real): significance level for critical value estimation [default: 0.05]
            nsim (int): number of simulations [default: 1000]
            
        Returns:
            r (real) - correlation between x and y \n
            signif (bool) - true (1) if significant; false (0) otherwise \n
            F - Fraction of time series with higher correlation coefficents than observed (approximates the p-value).
        
        Notes:
            See the following references: \n
            Ebisuzaki, W, 1997: A method to estimate the statistical
            significance of a correlation when the data are serially correlated.
            J. of Climate, 10, 2147-2153.\n
            Prichard, D., Theiler, J. Generating Surrogate Data for Time Series
            with Several Simultaneously Measured Variables (1994)
            Physical Review Letters, Vol 73, Number 7\n
            (Some Rights Reserved) USC Climate Dynamics Lab, 2012.
        """
        r = pearsonr(x, y)[0]
    
        # generate phase-randomized samples using the Theiler & Prichard method
        Xsurr = self.phaseran(x, nsim)
        Ysurr = self.phaseran(y, nsim)
    
        # compute correlations
        Xs = preprocessing.scale(Xsurr)
        Ys = preprocessing.scale(Ysurr)
    
        n = np.size(x)
        C = np.dot(np.transpose(Xs), Ys) / (n-1)
        rSim = np.diag(C)
    
        # compute fraction of values higher than observed
        F = np.sum(np.abs(rSim) >= np.abs(r)) / nsim
    
        # establish significance
        signif = F < alpha  # significant or not?
    
        return r, signif, F
    
    
    def phaseran(self, recblk, nsurr):
        """ Phaseran by Carlos Gias
        
            Full code available at: http://www.mathworks.nl/matlabcentral/fileexchange/32621-phase-randomization/content/phaseran.m
        
        Args:
            recblk (2D array): Row: time sample. Column: recording.
                An odd number of time samples (height) is expected.
                If that is not the case, recblock is reduced by 1 sample before the surrogate data is created.
                The class must be double and it must be nonsparse.
            nsurr (int): is the number of image block surrogates that you want to generate.
       
        Returns:
            surrblk - 3D multidimensional array image block with the surrogate datasets along the third dimension
            
        Notes:
            Reference: \n
            Prichard, D., Theiler, J. Generating Surrogate Data for Time Series with Several Simultaneously Measured Variables (1994)
            Physical Review Letters, Vol 73, Number 7
        """
        
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
        
        pbar = progressbar.ProgressBar(
                widgets=[
                    ' Surrogates generating... (',
                    progressbar.SimpleProgress(),
                    ') [',
                    progressbar.Percentage(), '] ',
                    progressbar.Bar(),
                    ' (', progressbar.ETA(), ') '
                ],
                redirect_stdout=True
            )
    
        for k in pbar(np.arange(nsurr)):
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
            
            