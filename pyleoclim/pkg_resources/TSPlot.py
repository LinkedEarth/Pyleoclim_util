# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:14:26 2016

@author: deborahkhider

Plot timeseries

"""

import lipd as lpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sys
import os
from matplotlib import gridspec

#Import internal packages to pyleoclim
from .LiPDutils import *
from .Basic import *

class Plot(object):
    """ Plot a timeseries
    """

    def __init__(self, plot_default,timeseries_list):
        

        self.TS= timeseries_list
        self.default = plot_default    
    
    def plot_Ts(self, timeseries = "", x_axis = "", markersize = 50,\
                   marker = "default", saveFig = False, dir = "",\
                   format="eps"):
        """ Plot a timeseries object
        
        Args:
            timeseries: A timeseries. By default, will prompt the user for one. 
            x_axis (str): The representation against which to plot the 
                paleo-data. Options are "age", "year", and "depth". 
                Default is to let the system choose if only one available or
                prompt the user. 
            markersize (int): default is 50. 
            marker (str): a string (or list) containing the color and shape of
                the marker. Default is by archiveType.Type pyleo.plot_default 
                to see the default palette.
            saveFig (bool): default is to not save the figure
            dir (str): the full path of the directory in which to save the 
                figure. If not provided, creates a default folder called 
                'figures' in the LiPD working directory (lipd.path). 
            format (str): One of the file extensions supported by the active
                backend. Default is "eps". Most backend support png, pdf, ps, eps, 
                and svg.
            
        Returns:
            The figure
            
        """
        # Get the data
        if not timeseries:
            timeseries = getTs(self.TS)
        
        dataframe = TsToDf(timeseries, x_axis)
        val_idx = valuesLoc(dataframe)
        dataframe = dataframe.iloc[val_idx,:]

        # Get the archiveType and make sure it aligns with the ontology
        archiveType = LipdToOntology(timeseries["archiveType"])
        
        # Get the labels for the axis
        # x-axis label
        headers = []
        for key, value in dataframe.items():
            headers.append(key)

        if "age" in headers:
            if "ageUnits" in timeseries.keys():
                x_axis_label = timeseries["ageUnits"]
            else:
                x_axis_label = "Age"
        elif "year" in headers:
            if "yearUnits" in timeseries.keys():
                x_axis_label = timeseries["yearUnits"]
            else:
                x_axis_label = "Year"
        else:
            if "depthUnits" in timeseries.keys():
                x_axis_label = "Depth (" + timeseries["depthUnits"] + ")"
            else:
                x_axis_label = "Depth"
        
        # get the y-axis label                           
        if "paleoData_onInferredVariableProperty" in timeseries.keys():
            if "paleoData_units" in timeseries.keys():
                y_axis_label = timeseries["paleoData_onInferredVariableProperty"] \
                    + " (" + timeseries["paleoData_units"] + ")" 
            else:
                y_axis_label = timeseries["paleoData_onInferredVariableProperty"]             
        elif "paleoData_onProxyObservationProperty" in timeseries.keys():
            if "paleoData_units" in timeseries.keys():
                y_axis_label = timeseries["paleoData_onProxyObservationProperty"] \
                    + " (" + timeseries["paleoData_units"] + ")"  
            else: 
                y_axis_label = timeseries["paleoData_onProxyObservationProperty"]     
        else:
            if "paleoData_units" in timeseries.keys():
                y_axis_label = timeseries["paleoData_variableName"] \
                    + " (" + timeseries["paleoData_units"] + ")"  
            else:
                y_axis_label = timeseries["paleoData_variableName"]
        
        #Make the plot
        fig = plt.figure()
        plt.style.use("ggplot")
        if marker == "default":
            marker = [self.default[archiveType][0],self.default[archiveType][1]]
            
        if "year" in headers:
            plt.scatter(dataframe['year'],dataframe['y-axis'],
                        s = markersize,
                        facecolor = 'none',
                        edgecolor = marker[0],
                        marker = marker[1],
                        label = 'original')
            plt.plot(dataframe['year'],dataframe['y-axis'],
                    color = marker[0],
                    linewidth = 1.0,
                    label = 'interpolated')
        elif "age" in headers:
            plt.scatter(dataframe['age'],dataframe['y-axis'],
                        s = markersize,
                        facecolor = 'none',
                        edgecolor = marker[0],
                        marker = marker[1],
                        label = 'original')
            plt.plot(dataframe['age'],dataframe['y-axis'],
                    color = marker[0],
                    linewidth = 1.0,
                    label = 'interpolated')
        
        plt.tight_layout()
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.legend(loc=3, scatterpoints=1, fancybox=True, shadow=True, \
                   fontsize=10)
        
        #Save the figure if asked
        if saveFig == True:
            name = 'plot_timeseries_'+timeseries["dataSetName"]+\
                "_"+y_axis_label
            saveFigure(name,format,dir)
        else:
            plt.show()

        return fig
    
    def plot_agemodel(self, timeseries = "", markersize = 50,\
                   marker = "default", saveFig = True, dir = "",
                   format="eps" ):
        """ Make a simple age-depth profile
        
        Args:
            timeseries: A timeseries. By default, will prompt the user for one. 
            markersize (int): default is 50. 
            marker (str): a string (or list) containing the color and shape of
                the marker. Default is by archiveType. Type pyleo.plot_default 
                to see the default palette.
            saveFig (bool): default is to not save the figure
            dir (str): the full path of the directory in which to save the figure. 
                If not provided, creates a default folder called 'figures' in
                the LiPD working directory (lipd.path). 
            format (str): One of the file extensions supported by the active 
                backend. Default is "eps". Most backend support png, pdf, ps, 
                eps, and svg.
                
        Returns:
            The figure 
            
        """
        # Get the data
        if not timeseries:
            timeseries = getTs(self.TS)
            
        if not "age" in timeseries.keys() and not "year" in timeseries.keys():
            sys.exit("No time information")
        elif not "depth" in timeseries.keys():
            sys.exit("No depth information")
        else:
            if "age" in timeseries.keys() and "year" in timeseries.keys():
                print("Do you want to use age or year?")
                choice = int(input("Enter 0 for age and 1 for year: "))
                if choice == 0:
                    y = timeseries['age']
                    if "ageUnits" in timeseries.keys():
                        y_axis_label = "Calendar Age (" +\
                                        timeseries["ageUnits"] +")"
                    else:
                        y_axis_label = "Calendar Age"
                elif choice == 1:
                    y = timeseries['year']
                    if "yearUnits" in timeseries.keys():
                        y_axis_label = "Year (" +\
                                        timeseries["yearUnits"] +")"
                    else:
                        y_axis_label = "Year"
                else:
                    sys.exit("Enter 0 or 1")
                    
            if "age" in timeseries.keys():
                y = timeseries['age']
                if "ageUnits" in timeseries.keys():
                    y_axis_label = "Calendar Age (" +\
                            timeseries["ageUnits"] +")"
                else:
                    y_axis_label = "Calendar Age"
                    
            if "year" in timeseries.keys():
                y = timeseries['year']
                if "yearUnits" in timeseries.keys():
                    y_axis_label = "Year (" +\
                            timeseries["ageUnits"] +")"
                else:
                    y_axis_label = "Year"
                    
            x = timeseries['depth']
            if "depthUnits" in timeseries.keys():
                x_axis_label = "Depth (" + timeseries["depthUnits"] + ")"
            else:
                x_axis_label = "Depth"
        
        # Get the archiveType and make sure it aligns with the ontology
        archiveType = LipdToOntology(timeseries["archiveType"]) 
        
        # Make the plot
        fig = plt.figure()
        plt.style.use("ggplot")
        if marker == "default":
            plt.plot(x,y,
                 color = self.default[archiveType][0],
                 linewidth = 1.0)
        else:
            plt.plot(x,y,
                 color = marker[0],
                 linewidth = 1.0)
        plt.tight_layout()
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)    
        
        #Save the figure if asked
        if saveFig == True:
            name = 'plot_agemodel_'+timeseries["dataSetName"]
            saveFigure(name,format,dir)
        else:
            plt.show()

        return fig    
