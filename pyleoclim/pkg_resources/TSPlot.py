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
import cartopy
import cartopy.crs as ccrs
import sys
import os
from matplotlib import gridspec

#Import internal packages to pyleoclim
from .LiPDutils import *
from .Basic import *

class Plot(object):

    def __init__(self, plot_default,time_series):
        self.TS= time_series
        self.default = plot_default    
    
    def plotoneTSO(self, new_timeseries = "", x_axis = "", markersize = 50,\
                   marker = "default", saveFig = True, dir = "figures",\
                   format="eps"):
        # Get the data
        if not new_timeseries:
            new_timeseries = getTSO(self.TS)
        
        dataframe = TStoDF(new_timeseries, x_axis)
        val_idx = valuesloc(dataframe)
        dataframe = dataframe.iloc[val_idx,:]

        # Get the archiveType and make sure it aligns with the ontology
        archiveType = LiPDtoOntology(new_timeseries["archiveType"])
        
        # Get the labels for the axis
        # x-axis label
        headers = []
        for key, value in dataframe.items():
            headers.append(key)

        if "age" in headers:
            if "ageUnits" in new_timeseries.keys():
                x_axis_label = new_timeseries["ageUnits"]
            else:
                x_axis_label = "Age"
        elif "year" in headers:
            if "yearUnits" in new_timeseries.keys():
                x_axis_label = new_timeseries["yearUnits"]
            else:
                x_axis_label = "Year"
        else:
            if "depthUnits" in new_timeseries.keys():
                x_axis_label = "Depth (" + new_timeseries["depthUnits"] + ")"
            else:
                x_axis_label = "Depth"
        
        # get the y-axis label                           
        if "paleoData_onInferredVariableProperty" in new_timeseries.keys():
            if "paleoData_units" in new_timeseries.keys():
                y_axis_label = new_timeseries["paleoData_onInferredVariableProperty"] \
                    + " (" + new_timeseries["paleoData_units"] + ")" 
            else:
                y_axis_label = new_timeseries["paleoData_onInferredVariableProperty"]             
        elif "paleoData_onProxyObservationProperty" in new_timeseries.keys():
            if "paleoData_units" in new_timeseries.keys():
                y_axis_label = new_timeseries["paleoData_onProxyObservationProperty"] \
                    + " (" + new_timeseries["paleoData_units"] + ")"  
            else: 
                y_axis_label = new_timeseries["paleoData_onProxyObservationProperty"]     
        else:
            if "paleoData_units" in new_timeseries.keys():
                y_axis_label = new_timeseries["paleoData_variableName"] \
                    + " (" + new_timeseries["paleoData_units"] + ")"  
            else:
                y_axis_label = new_timeseries["paleoData_variableName"]
        
        #Make the plot
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
            name = '/plot_timeseries_'+new_timeseries["dataSetName"]+\
                "_"+y_axis_label
            saveFigure(name,format,dir)
        else:
            plt.show()

    def agemodelplot(self, new_timeseries = "", markersize = 50,\
                   marker = "default", saveFig = True, dir = "figures",
                   format="eps" ):
        # Get the data
        if not new_timeseries:
            new_timeseries = getTSO(self.TS)
            
        if not "age" in new_timeseries.keys() and not "year" in new_timeseries.keys():
            sys.exit("No time information")
        elif not "depth" in new_timeseries.keys():
            sys.exit("No depth information")
        else:
            if "age" in new_timeseries.keys() and "year" in new_timeseries.keys():
                print("Do you want to use age or year?")
                choice = int(input("Enter 0 for age and 1 for year: "))
                if choice == 0:
                    y = new_timeseries['age']
                    if "ageUnits" in new_timeseries.keys():
                        y_axis_label = "Calendar Age (" +\
                                        new_timeseries["ageUnits"] +")"
                    else:
                        y_axis_label = "Calendar Age"
                elif choice == 1:
                    y = new_timeseries['year']
                    if "yearUnits" in new_timeseries.keys():
                        y_axis_label = "Year (" +\
                                        new_timeseries["yearUnits"] +")"
                    else:
                        y_axis_label = "Year"
                else:
                    sys.exit("Enter 0 or 1")
                    
            if "age" in new_timeseries.keys():
                y = new_timeseries['age']
                if "ageUnits" in new_timeseries.keys():
                    y_axis_label = "Calendar Age (" +\
                            new_timeseries["ageUnits"] +")"
                else:
                    y_axis_label = "Calendar Age"
                    
            if "year" in new_timeseries.keys():
                y = new_timeseries['year']
                if "yearUnits" in new_timeseries.keys():
                    y_axis_label = "Year (" +\
                            new_timeseries["ageUnits"] +")"
                else:
                    y_axis_label = "Year"
                    
            x = new_timeseries['depth']
            if "depthUnits" in new_timeseries.keys():
                x_axis_label = "Depth (" + new_timeseries["depthUnits"] + ")"
            else:
                x_axis_label = "Depth"
        
        # Get the archiveType and make sure it aligns with the ontology
        archiveType = LiPDtoOntology(new_timeseries["archiveType"]) 
        
        # Make the plot
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
            name = '/plot_agemodel_'+new_timeseries["dataSetName"]
            saveFigure(name,format,dir)
        else:
            plt.show()
