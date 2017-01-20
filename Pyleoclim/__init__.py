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
import cartopy
import cartopy.crs as ccrs
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
lpd.loadLipds()

# Get the timeseries objects
time_series = lpd.extractTs()

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
def MapAll(markersize = int(50), saveFig = True, dir="", format='eps'):
    """
    Map all the available LiPD files by archiveType.
    Arguments:
      - marersize: default is 50
      - saveFig: default is to save the figure
      - dir: the name of the folder in the current directory. If not provided, creates
      a folder called 'figures'
      - format: the format in which to save the map. The default is eps
    """
    map1 = Map(plot_default)
    map1.map_all(markersize=markersize, saveFig = saveFig, dir=dir, format=format)

def MapLiPD(name="",gridlines = False, borders = True, \
        topo = True, markersize = int(100), marker = "default", \
        saveFig = True, dir = "", format="eps"):
    """
    Makes a map of a single record
    Arguments:
     - name: the name of the LiPD file. WITH THE .LPD extension. If not provided
     will prompt the user for one
     - gridlines: default is none. Change to TRUE to get gridlines
     - borders: The adminsitrative borders. Default is TRUE
     - topo: The topography. Default is TRUE
     - markersize: defualt is 100
     - marker: a list containing the color and shape of the marker. Default is by archiveType.
     Type pyleo.plot_default to get a list by archiveType
     - saveFig: default is to save the figure
     - dir: the name of the folder in the current directory. If not provided, creates
      a folder called 'figures'
     - format: the format in which to save the map. The default is eps.
    """
    map1 = Map(plot_default)
    map1.map_one(name=name,gridlines = gridlines, borders = borders, \
        topo = topo, markersize = markersize, marker = marker, \
        saveFig = saveFig, dir = dir, format=format)

# Plotting

def plotTS(timeseries = "", x_axis = "", markersize = 50,\
            marker = "default", saveFig = True, dir = "figures",\
            format="eps"):
    """
    Plot a single time series
    Arguments:
    - A timeseries. Be default, will prompt the user for one
    - x_axis: The representation against which to plot the paleo-data. Options are "age",
    "year", and "depth". Default is to let the system choose if only available or prompt
    the user
    - markersize: default is 50
    - marker: a list of color and shape. Default uses the archive palette. Enter
    pyleo.plot_default for details
    - saveFig: default is to save the figure
    - dir: the name of the folder in the current directory. If not provided, creates
      a folder called 'figures'
    - format: the format in which to save the map. The default is eps.
    """
    plot1 = Plot(plot_default, time_series)
    plot1.plotoneTSO(new_timeseries = timeseries, x_axis = x_axis, markersize = markersize,\
                   marker = marker, saveFig = saveFig, dir = dir,\
                   format=format)

# Statistics

def TSstats(timeseries=""):
    """
    Return the mean and standard deviation of the timeseries
    Arguments:
    - Timeseries: sytem will prompt for one if not given
    """
    basic1 = Basic(time_series)
    mean, std = basic1.simpleStats(new_timeseries = timeseries)
    return mean, std

def TSbin(timeseries="", x_axis = "", bin_size = "", start = "", end = ""):
    """
    Bin the values of the timeseries
    Arguments:
            - Timeseries. Default is blank, will prompt for it
            - x-axis: the time or depth index to use for binning. Valid keys
            inlude: depth, age, and year. 
            - bin_size: the size of the bins to be used. If not given, 
            the function will prompt the user
            - start: where the bins should start. Default is the minimum 
            - end: where the bins should end. Default is the maximum
    Outputs:
           - binned_data: the binned output
           - bins: the bins (centered on the median, i.e. the 100-200 bin is 150)
           - n: number of data points in each bin
           - error: the standard error on the mean in each bin
    """
    if not timeseries:
        timeseries = getTSO(time_series)
    bins, binned_data, n, error = Basic.bin_data(new_timeseries = timeseries,\
                x_axis = x_axis, bin_size = bin_size, start = start, end = end)
    return bins, binned_data, n, error

def TSinterp(timeseries="", x_axis = "", interp_step = "", start = "", end = ""):
    """
    Simple linear interpolation
    Arguments:
            - Timeseries. Default is blank, will prompt for it
            - x-axis: the time or depth index to use for binning. Valid keys
            inlude: depth, age, and year. 
            - interp_step: the step size. If not given, 
            the function will prompt the user
            - start: where the interpolation should start. Default is the minimum 
            - end: where the interpolation should end. Default is the maximum
    Outputs:
           - interp_age: the interpolated age according to the end/start and time step
           - interp_values: the interpolated values
    """
    if not timeseries:
        timeseries = getTSO(time_series)
    interp_age, interp_values = Basic.interp_data(new_timeseries = timeseries,\
                x_axis = x_axis, interp_step = interp_step, start= start, end=end)
    return interp_age, interp_values
    
# SummaryPlots
def BasicSummary(timeseries = "", x_axis="", saveFig = True,
                     format = "eps", dir = "figures"):
    """
    Makes a basic summary plot
    1. The time series
    2. Location map
    3. Age-Depth profile if both are available from the paleodata
    4. Metadata

    Save the figures into a local directory if prompted    
    """
    plt1 = SummaryPlots(time_series, plot_default)
    plt1.basic(x_axis=x_axis, new_timeseries = timeseries, saveFig=saveFig,\
               format = format, dir = dir)

