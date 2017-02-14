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
    if not path and not ts_list:
        global lipd_path
        lipd_path = lpd.readLipds()
        global timeseries_list
        timeseries_list = lpd.extractTs()
    elif not ts_list:
        global lipd_path
        lipd_path = lpd.readLipds(path)
        global timeseries_list
        timeseries_list = lpd.extractTs()
    elif not path:
        sys.exit("If specifying a list of timeseries, also need to specify path")
    else:
        global lipd_path
        lipd_path = path
        global timeseries_list
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

# Statistics

def statsTs(timeseries=""):
    """ Calculate the mean and standard deviation of a timeseries
    
    Args:
        timeseries: sytem will prompt for one if not given
        
    Returns:
        The mean and standard deviation
        
    Examples:
        >>> mean,std = pyleoclim.statsTs()
        0 :  Ocean2kHR-AtlanticMontegoBayHaaseSchramm2003 :  Sr_Ca
        1 :  Ocean2kHR-AtlanticMontegoBayHaaseSchramm2003 :  d18O
        2 :  O2kLR-EmeraldBasin.Sachs.2007 :  notes
        3 :  O2kLR-EmeraldBasin.Sachs.2007 :  temperature
        4 :  O2kLR-EmeraldBasin.Sachs.2007 :  Uk37
        5 :  O2kLR-EmeraldBasin.Sachs.2007 :  notes
        6 :  O2kLR-EmeraldBasin.Sachs.2007 :  temperature
        7 :  O2kLR-EmeraldBasin.Sachs.2007 :  Uk37
        8 :  ODP1098B :  SST
        9 :  ODP1098B :  TEX86
        10 :  MD01-2412.Harada.2006 :  calyrbp
        11 :  MD01-2412.Harada.2006 :  sst
        12 :  MD01-2412.Harada.2006 :  uk37
        13 :  Crystal.McCabe-Glynn.2013 :  s180carbVPDB
        14 :  Crystal.McCabe-Glynn.2013 :  sst.anom
        15 :  Ocean2kHR-AtlanticCapeVerdeMoses2006 :  d18O
        16 :  Ocean2kHR-PacificNauruGuilderson1999 :  d13C
        17 :  Ocean2kHR-PacificNauruGuilderson1999 :  d18O
        18 :  Ocean2kHR-AtlanticBahamasTOTORosenheim2005 :  d18O
        19 :  Ocean2kHR-AtlanticBahamasTOTORosenheim2005 :  Sr_Ca
        20 :  Ocean2kHR-AtlanticPrincipeSwart1998 :  d13C
        21 :  Ocean2kHR-AtlanticPrincipeSwart1998 :  d18O
        22 :  MD98-2170.Stott.2004 :  d18o
        23 :  MD98-2170.Stott.2004 :  RMSE
        24 :  MD98-2170.Stott.2004 :  mg
        25 :  MD98-2170.Stott.2004 :  d18ow
        26 :  Ocean2kHR-PacificClippertonClipp2bWu2014 :  Sr_Ca
        27 :  MD982176.Stott.2004 :  Mg/Ca-g.rub
        28 :  MD982176.Stott.2004 :  sst
        29 :  MD982176.Stott.2004 :  d18Ob.rub
        30 :  MD982176.Stott.2004 :  d18Ow-s
        Enter the number of the variable you wish to use: 12
        
        >>> print(mean)
        0.401759365994
        
        >>> print(std)
        0.0821452359532
            
    """
    if not 'timeseries_list' in globals():
        openLipds()
     
    basic1 = Basic(timeseries_list)
    mean, std = basic1.simpleStats(timeseries = timeseries)
    return mean, std

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

