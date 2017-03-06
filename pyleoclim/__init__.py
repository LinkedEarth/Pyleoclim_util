# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:42:34 2016

@author: deborahkhider

Initializes the Pyleoclim package

"""
#Import all the needed packages


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
import seaborn as sns
import progressbar

#Import internal packages to pyleoclim
from .pkg_resources.Map import *
from .pkg_resources.TSPlot import *
from .pkg_resources.LiPDutils import *
from .pkg_resources.Basic import *
from .pkg_resources.SummaryPlots import *

# Load the LiPDs present in the directory
def openLipds(path="",ts_list=""):
    """Load and extract timeseries objects from LiPD files.
    
    Allows to load and extract timeseries objects into the workspace for use
    with Pyleoclim. This can be done by the user previously, using the LiPD 
    utilities and passed into the function's argumenta. If no timeseries objects
    are found by other functions, this function will be triggered automatically
    without arguments.
    
    Args:  
        path (string): the path to the LiPD file. If not specified, will 
            trigger the LiPD utilities GUI.
        ts_list (list): the list of available timeseries objects 
            obtained from lipd.extractTs(). 

    Warning: 
        if specifying a list, path should also be specified. 
    
    Examples:
        >>> pyleoclim.openLipds(path = "/Users/deborahkhider/Documents/LiPD")
        Found: 12 LiPD file(s)
        processing: Crystal.McCabe-Glynn.2013.lpd
        processing: MD01-2412.Harada.2006.lpd
        processing: MD98-2170.Stott.2004.lpd
        processing: MD982176.Stott.2004.lpd
        processing: O2kLR-EmeraldBasin.Sachs.2007.lpd
        processing: Ocean2kHR-AtlanticBahamasTOTORosenheim2005.lpd
        processing: Ocean2kHR-AtlanticCapeVerdeMoses2006.lpd
        processing: Ocean2kHR-AtlanticMontegoBayHaaseSchramm2003.lpd
        processing: Ocean2kHR-AtlanticPrincipeSwart1998.lpd
        processing: Ocean2kHR-PacificClippertonClipp2bWu2014.lpd
        processing: Ocean2kHR-PacificNauruGuilderson1999.lpd
        processing: ODP1098B.lpd
        extracting: ODP1098B.lpd
        extracting: MD98-2170.Stott.2004.lpd
        extracting: Ocean2kHR-PacificClippertonClipp2bWu2014.lpd
        extracting: Ocean2kHR-AtlanticBahamasTOTORosenheim2005.lpd
        extracting: Ocean2kHR-AtlanticPrincipeSwart1998.lpd
        extracting: Ocean2kHR-AtlanticMontegoBayHaaseSchramm2003.lpd
        extracting: MD982176.Stott.2004.lpd
        extracting: Ocean2kHR-PacificNauruGuilderson1999.lpd
        extracting: O2kLR-EmeraldBasin.Sachs.2007.lpd
        extracting: Crystal.McCabe-Glynn.2013.lpd
        extracting: Ocean2kHR-AtlanticCapeVerdeMoses2006.lpd
        extracting: MD01-2412.Harada.2006.lpd
        Finished time series: 31 objects
        Process Complete

    """
    global lipd_path
    global timeseries_list
    if not path and not ts_list:
        lipd_path = lpd.readLipds()
        timeseries_list = lpd.extractTs()
    elif not ts_list:
        lipd_path = lpd.readLipds(path)
        timeseries_list = lpd.extractTs()
    elif not path:
        sys.exit("If specifying a list of timeseries, also need to specify path")
    else:
        lipd_path = path
        timeseries_list = ts_list       
        
# Set the default palette for plots

plot_default = {'ice/rock': ['#FFD600','h'],
                'coral': ['#FF8B00','o'],
                'documents':['k','p'],
                'glacier ice':['#86CDFA', 'd'],
                'hybrid': ['#00BEFF','*'],
                'lake sediment': ['#4169E0','s'],
                'marine sediment': ['#8A4513', 's'],
                'sclerosponge' : ['r','o'],
                'speleothem' : ['FF1492','d'], 
                'wood' : ['#32CC32','^'],
                'molluskshells' : ['#FFD600','h'],
                'peat' : ['#2F4F4F','*']} 

# Mapping
def mapAll(markersize = 50, saveFig = False, dir="", format='eps'):
    """Map all the available records loaded into the workspace by archiveType.
    
    Map of all the records into the workspace by archiveType. 
        Uses the default color palette. Enter pyleoclim.plot_default for detail. 
    
    Args:
        markersize (int): The size of the markers. Default is 50
        saveFig (bool): Default is to not save the figure
        dir (str): The absolute path of the directory in which to save the 
            figure. If not provided, creates a default folder called 'figures' 
            in the LiPD working directory (lipd.path). 
        format (str): One of the file extensions supported by the active 
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg. 
    
    Returns:
        The figure
    
    Examples:
        >>> fig = pyleoclim.mapAll()
        
    """
    # Make sure there are LiPD files to plot
    if not 'timeseries_list' in globals():
        openLipds()
        
    map1 = Map(plot_default)
    fig =  map1.map_all(markersize=markersize, saveFig = saveFig, dir=dir, format=format)

    return fig

def mapLipd(name="", countries = True, counties = False, \
        rivers = False, states = False, background = "shadedrelief",\
        scale = 0.5, markersize = 50, marker = "default", \
        saveFig = False, dir = "", format="eps"):
    """ Create a Map for a single record
    
    Orthographic projection map of a single record.
    
    Args:
        name(str): the name of the LiPD file. **WITH THE .LPD EXTENSION!**. 
            If not provided, will prompt the user for one
        countries (bool): Draws the country borders. Default is on (True).
        counties (bool): Draws the USA counties. Default is off (False).
        states (bool): Draws the American and Australian states borders. 
            Default is off (False)
        background (str): Plots one of the following images on the map: 
            bluemarble, etopo, shadedrelief, or none (filled continents). 
            Default is shadedrelief
        scale (float): useful to downgrade the original image resolution to 
            speed up the process. Default is 0.5.
        markersize (int): default is 100
        marker (str): a string (or list) containing the color and shape of the
            marker. Default is by archiveType. Type pyleo.plot_default to see
            the default palette. 
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).  
        format (str): One of the file extensions supported by the active 
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.
        
    Returns:
        The figure
        
    Examples:
        >>> fig = pyleoclim.mapLipd(markersize=100)               
        
    """
    # Make sure there are LiPD files to plot

    if not 'timeseries_list' in globals():
        openLipds()
        
    map1 = Map(plot_default)
    fig =  map1.map_Lipd(name=name,countries = countries, counties = counties, \
        rivers = rivers, states = states, background = background,\
        scale = scale, markersize = markersize, marker = marker, \
        saveFig = saveFig, dir = dir, format=format)

    return fig

# Plotting

def plotTs(timeseries = "", x_axis = "", markersize = 50,\
            marker = "default", saveFig = False, dir = "",\
            format="eps"):
    """Plot a single time series. 
    
    Args:
        A timeseries: By default, will prompt the user for one. 
        x_axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the 
            system choose if only one available or prompt the user. 
        markersize (int): default is 50. 
        marker (str): a string (or list) containing the color and shape of the
            marker. Default is by archiveType. Type pyleo.plot_default to see 
            the default palette.
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the 
            LiPD working directory (lipd.path). 
        format (str): One of the file extensions supported by the active 
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.
    
    Returns:
        The figure. 
    
    Examples:
        >>> fig = pyleoclim.plotTs(marker = 'rs')
        
    """
    if not 'timeseries_list' in globals():
        openLipds()
        
    plot1 = Plot(plot_default, timeseries_list)
    fig = plot1.plot_Ts(timeseries = timeseries, x_axis = x_axis,\
                   markersize = markersize,\
                   marker = marker, saveFig = saveFig, dir = dir,\
                   format=format)

    return fig

def histTs(timeseries = "", bins = None, hist = True, \
             kde = True, rug = False, fit = None, hist_kws = {"label":"hist"},\
             kde_kws = {"label":"kde"}, rug_kws = {"label":"rug"}, \
             fit_kws = {"label":"fit"}, color = None, vertical = False, \
             norm_hist = True, legend = True, saveFig = False, format ="eps",\
             dir = ""):
    """ Plot a univariate distribution of the PaleoData values
    
    This function is based on the seaborn displot function, which is
    itself a combination of the matplotlib hist function with the 
    seaborn kdeplot() and rugplot() functions. It can also fit 
    scipy.stats distributions and plot the estimated PDF over the data.
    
    Args:
        timeseries: A timeseries. By default, will prompt the user for one.
        bins (int): Specification of hist bins following matplotlib(hist), 
            or None to use Freedman-Diaconis rule
        hist (bool): Whether to plot a (normed) histogram    
        kde (bool): Whether to plot a gaussian kernel density estimate
        rug (bool): Whether to draw a rugplot on the support axis
        fit: Random variable object. An object with fit method, returning 
            a tuple that can be passed to a pdf method of positional 
            arguments following a grid of values to evaluate the pdf on.
        {hist, kde, rug, fit}_kws: Dictionaries. Keyword arguments for 
            underlying plotting functions. If modifying the dictionary, make
            sure the labels "hist", "kde", "rug" and "fit" are still passed.
        color (str): matplotlib color. Color to plot everything but the
            fitted curve in.
        vertical (bool): if True, oberved values are on y-axis.
        norm_hist (bool): If True (default), the histrogram height shows
            a density rather than a count. This is implied if a KDE or 
            fitted density is plotted
        legend (bool): If true, plots a default legend
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the 
            LiPD working directory (lipd.path). 
        format (str): One of the file extensions supported by the active 
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.
            
    Returns
        fig - The figure
    
    Examples:
        >>> fig = pyleoclim.histTs(vertical = True)    
    """    
    if not 'timeseries_list' in globals():
        openLipds()
        
    plot1 = Plot(plot_default, timeseries_list)
    fig = plot1.plot_hist(timeseries = timeseries, bins = bins, hist = hist, \
                 kde = kde, rug = rug, fit = fit, hist_kws = hist_kws,\
                 kde_kws = kde_kws, rug_kws = rug_kws, \
                 fit_kws = fit_kws, color = color, vertical = vertical, \
                 norm_hist = norm_hist, legend = legend)
    
    return fig

# Statistics

def statsTs(timeseries=""):
    """ Calculate the mean and standard deviation of a timeseries
    
    Args:
        timeseries: sytem will prompt for one if not given
        
    Returns:
        the mean, median, min, max, standard deviation and the
        inter-quartile range (IQR) of a timeseries.
        
    Examples:
        >>> mean, median, min_, max_, std, IQR = pyleo.statsTs()
        0 :  Ocean2kHR-AtlanticCapeVerdeMoses2006 :  d18O
        1 :  ODP1098B :  SST
        2 :  ODP1098B :  TEX86
        3 :  Ocean2kHR-AtlanticBahamasTOTORosenheim2005 :  d18O
        4 :  Ocean2kHR-AtlanticBahamasTOTORosenheim2005 :  Sr_Ca
        5 :  Ocean2kHR-PacificClippertonClipp2bWu2014 :  Sr_Ca
        6 :  Crystal.McCabe-Glynn.2013 :  sst.anom
        7 :  Crystal.McCabe-Glynn.2013 :  s180carbVPDB
        8 :  Ocean2kHR-AtlanticMontegoBayHaaseSchramm2003 :  d18O
        9 :  Ocean2kHR-AtlanticMontegoBayHaaseSchramm2003 :  Sr_Ca
        10 :  Ocean2kHR-AtlanticPrincipeSwart1998 :  d13C
        11 :  Ocean2kHR-AtlanticPrincipeSwart1998 :  d18O
        12 :  MD01-2412.Harada.2006 :  uk37
        13 :  MD01-2412.Harada.2006 :  sst
        14 :  MD01-2412.Harada.2006 :  calyrbp
        15 :  MD98-2170.Stott.2004 :  mg
        16 :  MD98-2170.Stott.2004 :  RMSE
        17 :  MD98-2170.Stott.2004 :  d18ow
        18 :  MD98-2170.Stott.2004 :  d18o
        19 :  MD982176.Stott.2004 :  d18Ow-s
        20 :  MD982176.Stott.2004 :  sst
        21 :  MD982176.Stott.2004 :  d18Ob.rub
        22 :  MD982176.Stott.2004 :  Mg/Ca-g.rub
        23 :  Ocean2kHR-PacificNauruGuilderson1999 :  d13C
        24 :  Ocean2kHR-PacificNauruGuilderson1999 :  d18O
        25 :  O2kLR-EmeraldBasin.Sachs.2007 :  temperature
        26 :  O2kLR-EmeraldBasin.Sachs.2007 :  notes
        27 :  O2kLR-EmeraldBasin.Sachs.2007 :  Uk37
        28 :  O2kLR-EmeraldBasin.Sachs.2007 :  temperature
        29 :  O2kLR-EmeraldBasin.Sachs.2007 :  notes
        30 :  O2kLR-EmeraldBasin.Sachs.2007 :  Uk37
        Enter the number of the variable you wish to use: 13
        
        >>> print(mean)
        10.6708933718
        
        >>> print(median)
        11.0
        
        >>> print(min_)
        5.0
        
        >>> print(max_)
        16.2
        
        >>> print(std)
        2.41519924361
        
        >>> print(IQR)
        3.9
            
    """
    if not 'timeseries_list' in globals():
        openLipds()
     
    basic1 = Basic(timeseries_list)
    mean, median, min_, max_, std, IQR = basic1.simpleStats(timeseries = timeseries)
    return mean, median, min_, max_, std, IQR

def corrSigTs(timeseries1 = "", timeseries2 = "", x_axis = "", \
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
            method (str): method use to estimate the correlation and significance.
                Available methods include:
                    - 'ttest': T-test where the degrees of freedom are corrected for
                    the effect of serial correlation \n
                    - 'isopersistant': AR(1) modeling of the two timeseries \n
                    - 'isospectral' (default): phase randomization of original
                    inputs.
                The T-test is parametric test, hence cheap but usually wrong 
                    except in idyllic circumstances.
                The others are non-parametric, but their computational 
                    requirements scales with nsim.
            alpha (float): significance level for critical value estimation. Default is 0.05

        Returns:
            r (float) - correlation between the two timeseries \n
            sig (bool) -  Returns True if significant, False otherwise \n
            p (real) - the p-value
            
        Examples:
            >>> r, sig, p = pyleoclim.corrSigTs()
            0 :  Ocean2kHR-PacificClippertonClipp2bWu2014 :  Sr_Ca
            1 :  Ocean2kHR-PacificNauruGuilderson1999 :  d18O
            2 :  Ocean2kHR-PacificNauruGuilderson1999 :  d13C
            3 :  MD982176.Stott.2004 :  sst
            4 :  MD982176.Stott.2004 :  d18Ob.rub
            5 :  MD982176.Stott.2004 :  d18Ow-s
            6 :  MD982176.Stott.2004 :  Mg/Ca-g.rub
            7 :  ODP1098B :  TEX86
            8 :  ODP1098B :  SST
            9 :  Ocean2kHR-AtlanticCapeVerdeMoses2006 :  d18O
            10 :  O2kLR-EmeraldBasin.Sachs.2007 :  temperature
            11 :  O2kLR-EmeraldBasin.Sachs.2007 :  Uk37
            12 :  O2kLR-EmeraldBasin.Sachs.2007 :  notes
            13 :  O2kLR-EmeraldBasin.Sachs.2007 :  temperature
            14 :  O2kLR-EmeraldBasin.Sachs.2007 :  Uk37
            15 :  O2kLR-EmeraldBasin.Sachs.2007 :  notes
            16 :  Ocean2kHR-AtlanticMontegoBayHaaseSchramm2003 :  d18O
            17 :  Ocean2kHR-AtlanticMontegoBayHaaseSchramm2003 :  Sr_Ca
            18 :  MD98-2170.Stott.2004 :  d18o
            19 :  MD98-2170.Stott.2004 :  RMSE
            20 :  MD98-2170.Stott.2004 :  d18ow
            21 :  MD98-2170.Stott.2004 :  mg
            22 :  Ocean2kHR-AtlanticPrincipeSwart1998 :  d18O
            23 :  Ocean2kHR-AtlanticPrincipeSwart1998 :  d13C
            24 :  Ocean2kHR-AtlanticBahamasTOTORosenheim2005 :  Sr_Ca
            25 :  Ocean2kHR-AtlanticBahamasTOTORosenheim2005 :  d18O
            26 :  MD01-2412.Harada.2006 :  uk37
            27 :  MD01-2412.Harada.2006 :  sst
            28 :  MD01-2412.Harada.2006 :  calyrbp
            29 :  Crystal.McCabe-Glynn.2013 :  sst.anom
            30 :  Crystal.McCabe-Glynn.2013 :  s180carbVPDB
            Enter the number of the variable you wish to use: 19
            0 :  Ocean2kHR-PacificClippertonClipp2bWu2014 :  Sr_Ca
            1 :  Ocean2kHR-PacificNauruGuilderson1999 :  d18O
            2 :  Ocean2kHR-PacificNauruGuilderson1999 :  d13C
            3 :  MD982176.Stott.2004 :  sst
            4 :  MD982176.Stott.2004 :  d18Ob.rub
            5 :  MD982176.Stott.2004 :  d18Ow-s
            6 :  MD982176.Stott.2004 :  Mg/Ca-g.rub
            7 :  ODP1098B :  TEX86
            8 :  ODP1098B :  SST
            9 :  Ocean2kHR-AtlanticCapeVerdeMoses2006 :  d18O
            10 :  O2kLR-EmeraldBasin.Sachs.2007 :  temperature
            11 :  O2kLR-EmeraldBasin.Sachs.2007 :  Uk37
            12 :  O2kLR-EmeraldBasin.Sachs.2007 :  notes
            13 :  O2kLR-EmeraldBasin.Sachs.2007 :  temperature
            14 :  O2kLR-EmeraldBasin.Sachs.2007 :  Uk37
            15 :  O2kLR-EmeraldBasin.Sachs.2007 :  notes
            16 :  Ocean2kHR-AtlanticMontegoBayHaaseSchramm2003 :  d18O
            17 :  Ocean2kHR-AtlanticMontegoBayHaaseSchramm2003 :  Sr_Ca
            18 :  MD98-2170.Stott.2004 :  d18o
            19 :  MD98-2170.Stott.2004 :  RMSE
            20 :  MD98-2170.Stott.2004 :  d18ow
            21 :  MD98-2170.Stott.2004 :  mg
            22 :  Ocean2kHR-AtlanticPrincipeSwart1998 :  d18O
            23 :  Ocean2kHR-AtlanticPrincipeSwart1998 :  d13C
            24 :  Ocean2kHR-AtlanticBahamasTOTORosenheim2005 :  Sr_Ca
            25 :  Ocean2kHR-AtlanticBahamasTOTORosenheim2005 :  d18O
            26 :  MD01-2412.Harada.2006 :  uk37
            27 :  MD01-2412.Harada.2006 :  sst
            28 :  MD01-2412.Harada.2006 :  calyrbp
            29 :  Crystal.McCabe-Glynn.2013 :  sst.anom
            30 :  Crystal.McCabe-Glynn.2013 :  s180carbVPDB
            Enter the number of the variable you wish to use: 3
            The two timeseries do not contain the same number of points. Interpolating...
            Do you want to use time or depth?
            Enter 0 for time and 1 for depth: 0
        
            >>> print(p)
            0.004
        
            >>> print(r)
            0.723075099334
        
            >>> print(sig)
            True
    """    
    if not 'timeseries_list' in globals():
        openLipds()
        
    corr1 = Correlation(timeseries_list)
    r, sig, p = corr1.corr_sig(timeseries1 = timeseries1, \
                               timeseries2 = timeseries2, x_axis = x_axis, \
                 interp_step = interp_step, start = start, end = end, \
                 nsim = 1000, method = 'isospectral', alpha = 0.5)

    return r, sig, p    
    
def binTs(timeseries="", x_axis = "", bin_size = "", start = "", end = ""):
    """Bin the paleoData values of the timeseries
    
    Args:
        timeseries. By default, will prompt the user for one.
        x-axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the 
            system  choose if only one available or prompt the user. 
        bin_size (float): the size of the bins to be used. By default, 
            will prompt for one
        start (float): Start time/age/depth. Default is the minimum 
        end (float): End time/age/depth. Default is the maximum
        
    Returns:
        binned_data- the binned output,\n
        bins-  the bins (centered on the median, i.e. the 100-200 bin is 150),\n
        n-  number of data points in each bin,\n
        error- the standard error on the mean in each bin\n
    
    Example:
        >>> ts = pyleoclim.timeseries_list[28]
        >>> bin_size = 200
        >>> bins, binned_data, n, error = pyleoclim.binTs(timeseries = ts, bin_size = bin_size)
        Do you want to plot vs time or depth?
        Enter 0 for time and 1 for depth: 0
        
        >>> print(bins)
        [   239.3    439.3    639.3 ...,  14439.3  14639.3  14839.3]
        
        >>> print(binned_data)
        [28.440000000000005, 28.920000000000005, 28.657142857142862, 
        28.939999999999998, 28.733333333333334, 28.949999999999999, 28.75, 
        28.899999999999999, 28.75, 28.566666666666663, 28.800000000000001, 
        29.049999999999997, 29.233333333333334, 29.274999999999999, 
        29.057142857142857, 28.699999999999999, 29.433333333333334, 
        28.575000000000003, 28.733333333333331, 28.48, 28.733333333333331,
        28.766666666666666, 29.166666666666668, 29.18, 29.600000000000001, 
        29.300000000000001, 28.949999999999999, 29.475000000000001,
        29.333333333333332, 29.800000000000001, 29.016666666666666, 
        29.349999999999998, 29.485714285714288, 28.850000000000001,
        29.366666666666664, 28.699999999999999, 29.233333333333334, 
        29.366666666666664, 29.5, 29.350000000000001, 29.699999999999999, 
        29.300000000000001, 29.233333333333334, 29.300000000000001, 
        29.300000000000001, 29.600000000000001, 28.950000000000003,
        29.166666666666668, 28.799999999999997, 28.975000000000001,
        29.033333333333331, 28.649999999999999, 28.450000000000003,
        28.533333333333331, 28.599999999999998, 28.25, 28.0, 
        28.550000000000001, 28.799999999999997, 28.350000000000001, 
        27.699999999999999, 27.149999999999999, 27.666666666666668, 
        26.800000000000001, 26.700000000000003, 26.800000000000001, 
        26.5, 26.850000000000001, 26.5, 26.5, 26.0, 26.899999999999999,
        26.5, 26.100000000000001]
        
    """
    if not 'timeseries_list' in globals():
        openLipds()

    if not timeseries:
        timeseries = getTs(timeseries_list)
                
    bins, binned_data, n, error = Basic.bin_Ts(timeseries = timeseries,\
                x_axis = x_axis, bin_size = bin_size, start = start, end = end)
    
    return bins, binned_data, n, error

def interpTs(timeseries="", x_axis = "", interp_step = "", start = "", end = ""):
    """Simple linear interpolation
    
    Simple linear interpolation of the data using the numpy.interp method
    
    Args:
        timeseries. Default is blank, will prompt for it
        x-axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the 
            system choose if only one available or prompt the user. 
        interp_step (float): the step size. By default, will prompt the user. 
        start (float): Start time/age/depth. Default is the minimum 
        end (float): End time/age/depth. Default is the maximum
        
    Returns:
        interp_age - the interpolated age/year/depth according to the end/start 
        and time step, \n
        interp_values - the interpolated values
        
    Examples:
        >>> ts = pyleoclim.timeseries_list[28]
        >>> interp_step = 200
        >>> interp_age, interp_values = pyleoclim.interpTs(timeseries = ts, interp_step = interp_step)
        Do you want to plot vs time or depth?
        Enter 0 for time and 1 for depth: 0
        
        >>> print(interp_age)
        [   139.3    339.3    539.3 ...,  14339.3  14539.3  14739.3]
        
        >>> print(interp_values)
        [ 0.188       0.05981567 -0.04020261 ...,  1.20834663  1.47751854
         1.16054494]
      
    """
    if not 'timeseries_list' in globals():
        openLipds()

    if not timeseries:
        timeseries = getTs(timeseries_list)
        
    interp_age, interp_values = Basic.interp_Ts(timeseries = timeseries,\
                x_axis = x_axis, interp_step = interp_step, start= start, end=end)
    
    return interp_age, interp_values
    
# SummaryPlots
def basicSummary(timeseries = "", x_axis="", saveFig = False,
                     format = "eps", dir = ""):
    """ Makes a basic summary plot
    
    Plots the following information: the time series, location map, 
    Age-Depth profile if both are available from the paleodata, Metadata

    Notes: 
        The plots use default settings from the MapLiPD and plotTS methods.
    
    Arguments:
        timeseries: By default, will prompt for one.
        x-axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the 
            system choose if only one available or prompt the user.
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the 
            LiPD working directory (lipd.path). 
        format (str): One of the file extensions supported by the active 
            backend. Default is "eps". Most backend support png, pdf, ps, eps, 
            and svg.
        
    Returns:
        The figure.
        
    Examples:
        >>> fig = pyleoclim.basicSummary()
        
    """
    if not 'timeseries_list' in globals():
        openLipds()
        
    plt1 = SummaryPlots(timeseries_list, plot_default)
    fig = plt1.basic(x_axis=x_axis, timeseries = timeseries, saveFig=saveFig,\
               format = format, dir = dir)

    return fig

def basicSummary2(timeseries = "", x_axis="", saveFig = False,
                     format = "eps", dir = ""):
    """ Second type of basic summary plot
    
    Plots the following information: the time series, a histogram of
    the PaleoData_values, location map, age-depth profile if both are
    available from the paleodata, metadata about the record.

    Notes: 
        The plots use default settings from the MapLiPD and plotTS methods.
    
    Arguments:
        timeseries: By default, will prompt for one.
        x-axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the 
            system choose if only one available or prompt the user.
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the 
            LiPD working directory (lipd.path). 
        format (str): One of the file extensions supported by the active 
            backend. Default is "eps". Most backend support png, pdf, ps, eps, 
            and svg.
        
    Returns:
        The figure.
        
    Examples:
        >>> fig = pyleoclim.basicSummary2()
        
    """
    if not 'timeseries_list' in globals():
        openLipds()
        
    plt1 = SummaryPlots(timeseries_list, plot_default)
    fig = plt1.basic2(x_axis=x_axis, timeseries = timeseries, saveFig=saveFig,\
               format = format, dir = dir)

    return fig