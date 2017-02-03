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
import cartopy
import cartopy.crs as ccrs
import sys
import os
from matplotlib import gridspec

#Import internal packages to pyleoclim
from .LiPDutils import *

class Basic(object):
    
    def __init__(self,timeseries_dict):
        """
        Pass the time_series dictionary into the Basic object
        """
        self.TS = timeseries_dict

    @staticmethod
    def getValues(timeseries):
        """
        Get the paleoData values from the timeseries object
        Arguments:
        - new_timeseries: a single timeseries object. Use getTSO() to get one from the dictionary
        """
        values_key =[]
        for key, val in timeseries.items():
            if "values" in key.lower():
                values_key.append(key)
        
        values = timeseries[values_key[0]]

        return values             
    
        
    def simpleStats(self, timeseries=""):
        """
        Compute the mean and standard deviation of a time series
        Arguments:
        - new_timeseries: a single timeseries. Will prompt for one if not available
        """        
        # get the values
        if not timeseries:
            timeseries = getTSO(self.TS)

        values = Basic.getValues(timeseries)    
     
        mean = np.nanmean(values)
        std = np.nanstd(values) 
        
        return mean, std
    
    @staticmethod    
    def bin_data(timeseries, x_axis = "", bin_size = "", start = "", end = ""):
        """
        Bin the data.
        Arguments:
         - new_timeseries: a single timeseries object. Use getTSO() to get one.
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
        
        # Get the values
        values = Basic.getValues(timeseries)
        
        # Get the time (or depth) representation
        if not x_axis:
            time, label = TSOxaxis(timeseries)
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
    def interp_data(timeseries, x_axis = "", interp_step="",start ="",\
                    end =""):
        """
        Linear interpolation of the paleodata
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
            
        # Get the values and age
        values = Basic.getValues(timeseries)
        # Get the time (or depth) representation
        if not x_axis:
            time, label = TSOxaxis(timeseries)
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
        interp_values =np.interp(interp_age,data['x-axis'],data['y-axis'])
        
        return interp_age, interp_values
