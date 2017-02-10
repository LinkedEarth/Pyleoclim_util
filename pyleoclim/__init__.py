# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:42:34 2016

@author: deborahkhider

Initializes the Pyleoclim module

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
def openLiPDs(path="",timeseries_list=""):
    """
    Load and extract the timeseries object from the LiPD files
    Arguments:
    - Path to LiPD files. Default is blank which will prompt the LiPD GUI
    - TS: a dictionary of timeseries obtained from the function lipd.extracTS ran outside of pyleoclim
    WARNING: When specifying a TS, 
    """
    if not path and not timeseries_list:
        global lipd_path
        lipd_path = lpd.readLipds()
        global time_series
        time_series = lpd.extractTs()
    elif not timeseries_list:
        global lipd_path
        lipd_path = lpd.readLipds(path)
        global time_series
        time_series = lpd.extractTs()
    elif not path:
        sys.exit("If specifying a dictionary of timeseries, also need to specify path")
    else:
        global lipd_path
        lipd_path = path
        global time_series
        time_series = timeseries_list       
        
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
def MapAll(markersize = 50, saveFig = False, dir="", format='eps'):
    """
    Map all the available records loaded into the LiPD working directory by archiveType.
    Arguments:
      - path: the path where the liPD files are saved. If not given, will trigger the LiPD GUI
      - markersize: default is 50
      - saveFig: default is to not save the figure
      - dir: the full path of the directory in which to save the figure. If not provided, creates
      a default folder called 'figures' in the LiPD working directory (lipd.path). 
      - format: One of the file extensions supported by the active backend. Default is "eps".
      Most backend support png, pdf, ps, eps, and svg. 
    """
    # Make sure there are LiPD files to plot
    if not 'time_series' in globals():
        openLiPDs()
        
    map1 = Map(plot_default)
    fig =  map1.map_all(markersize=markersize, saveFig = saveFig, dir=dir, format=format)

    return fig

def MapLiPD(name="", countries = True, counties = False, \
        rivers = False, states = False, background = "shadedrelief",\
        scale = 0.5, markersize = 50, marker = "default", \
        saveFig = False, dir = "", format="eps"):
    """
    Makes a map for a single record. 
    Arguments:
         - name: the name of the LiPD file. **WITH THE .LPD EXTENSION!**.
         If not provided, will prompt the user for one.
         - countries: Draws the country borders. Default is on (True).
         - counties: Draws the USA counties. Default is off (False).
         - states: Draws the American and Australian states borders. Default is off (False)
         - background: Plots one of the following images on the map: bluemarble, etopo, shadedrelief,
         or none (filled continents). Default is shadedrelief
         - scale: useful to downgrade the original image resolution to speed up the process. Default is 0.5.
         - markersize: default is 100
         - marker: a string (or list) containing the color and shape of the marker. Default is by archiveType.
         Type pyleo.plot_default to see the default palette. 
         - saveFig: default is to not save the figure
         - dir: the full path of the directory in which to save the figure. If not provided, creates
          a default folder called 'figures' in the LiPD working directory (lipd.path).  
         - format: One of the file extensions supported by the active backend. Default is "eps".
          Most backend support png, pdf, ps, eps, and svg.
    """
    # Make sure there are LiPD files to plot

    if not 'time_series' in globals():
        openLiPDs()
        
    map1 = Map(plot_default)
    fig =  map1.map_one(name=name,countries = countries, counties = counties, \
        rivers = rivers, states = states, background = background,\
        scale = scale, markersize = markersize, marker = marker, \
        saveFig = saveFig, dir = dir, format=format)

    return fig

# Plotting

def plotTS(timeseries = "", x_axis = "", markersize = 50,\
            marker = "default", saveFig = False, dir = "",\
            format="eps"):
    """
    Plot a single time series. 
    Arguments:
    - A timeseries: By default, will prompt the user for one. 
    - x_axis: The representation against which to plot the paleo-data. Options are "age",
    "year", and "depth". Default is to let the system choose if only one available or prompt
    the user. 
    - markersize: default is 50. 
    - marker: a string (or list) containing the color and shape of the marker. Default is by archiveType.
     Type pyleo.plot_default to see the default palette.
    - saveFig: default is to not save the figure
    - dir: the full path of the directory in which to save the figure. If not provided, creates
      a default folder called 'figures' in the LiPD working directory (lipd.path). 
    - format: One of the file extensions supported by the active backend. Default is "eps".
      Most backend support png, pdf, ps, eps, and svg.
    """
    if not 'time_series' in globals():
        openLiPDs()
        
    plot1 = Plot(plot_default, time_series)
    fig = plot1.plotoneTSO(new_timeseries = timeseries, x_axis = x_axis, markersize = markersize,\
                   marker = marker, saveFig = saveFig, dir = dir,\
                   format=format)

    return fig

# Statistics

def TSstats(timeseries=""):
    """
    Return the mean and standard deviation of the paleoData values of a timeseries
    Arguments:
    - Timeseries: sytem will prompt for one if not given
    """
    if not 'time_series' in globals():
        openLiPDs()
     
    basic1 = Basic(time_series)
    mean, std = basic1.simpleStats(timeseries = timeseries)
    return mean, std

def TSbin(timeseries="", x_axis = "", bin_size = "", start = "", end = ""):
    """
    Bin the paleoData values of the timeseries
    Arguments:
      - Timeseries. By default, will prompt the user for one.
      - x-axis: The representation against which to plot the paleo-data. Options are "age",
    "year", and "depth". Default is to let the system choose if only one available or prompt
    the user. 
      - bin_size: the size of the bins to be used. By default, will prompt for one
      - start: Start time/age/depth. Default is the minimum 
      - end: End time/age/depth. Default is the maximum
    Outputs:
      - binned_data: the binned output
      - bins: the bins (centered on the median, i.e. the 100-200 bin is 150)
      - n: number of data points in each bin
      - error: the standard error on the mean in each bin
    """
    if not 'time_series' in globals():
        openLiPDs()

    if not timeseries:
        timeseries = getTSO(time_series)
                
    bins, binned_data, n, error = Basic.bin_data(timeseries = timeseries,\
                x_axis = x_axis, bin_size = bin_size, start = start, end = end)
    return bins, binned_data, n, error

def TSinterp(timeseries="", x_axis = "", interp_step = "", start = "", end = ""):
    """
    Simple linear interpolation
    Arguments:
      - Timeseries. Default is blank, will prompt for it
      - x-axis: The representation against which to plot the paleo-data. Options are "age",
    "year", and "depth". Default is to let the system choose if only one available or prompt
    the user. 
      - interp_step: the step size. By default, will prompt the user. 
      - start: Start time/age/depth. Default is the minimum 
      - end: End time/age/depth. Default is the maximum
    Outputs:
      - interp_age: the interpolated age/year/depth according to the end/start and time step
      - interp_values: the interpolated values
    """
    if not 'time_series' in globals():
        openLiPDs()

    if not timeseries:
        timeseries = getTSO(time_series)
        
    interp_age, interp_values = Basic.interp_data(timeseries = timeseries,\
                x_axis = x_axis, interp_step = interp_step, start= start, end=end)
    return interp_age, interp_values
    
# SummaryPlots
def BasicSummary(timeseries = "", x_axis="", saveFig = False,
                     format = "eps", dir = ""):
    """
    Makes a basic summary plot
    1. The time series
    2. Location map
    3. Age-Depth profile if both are available from the paleodata
    4. Metadata

    **Note**: The plots use default setting from the MapLiPD and plotTS method.
    
    Arguments:
      - timeseries: By default, will prompt for one.
      - x-axis: The representation against which to plot the paleo-data. Options are "age",
    "year", and "depth". Default is to let the system choose if only one available or prompt
    the user.
      - saveFig: default is to not save the figure
      - dir: the full path of the directory in which to save the figure. If not provided, creates
      a default folder called 'figures' in the LiPD working directory (lipd.path). 
      - format: One of the file extensions supported by the active backend. Default is "eps".
      Most backend support png, pdf, ps, eps, and svg.
    """
    if not 'time_series' in globals():
        openLiPDs()
        
    plt1 = SummaryPlots(time_series, plot_default)
    fig = plt1.basic(x_axis=x_axis, new_timeseries = timeseries, saveFig=saveFig,\
               format = format, dir = dir)

    return fig

