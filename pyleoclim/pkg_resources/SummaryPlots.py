# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 2017

@author: deborahkhider

Summary plots

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

class SummaryPlots(object):
    """Plots various summary figures for a LiPD record
    
    """
    
    def __init__(self,timeseries_list,plot_default):
       
        self.TS = timeseries_list
        self.default = plot_default
        
    def getMetadata(self, timeseries):
        """ Get the necessary metadata to be printed out automatically
        
        Args:
            timeseries: a specific timeseries object
            
        Returns:
            A dictionary containing the following metadata:\n
            archiveType \n
            Authors (if more than 2, replace by et al. \n
            PublicationYear \n
            Publication DOI \n
            Variable Name \n
            Units \n
            Climate Interpretation \n
            Calibration Equation \n
            Calibration References \n
            Calibration Notes \n
            
        """
        # Get all the necessary information
        # Top level information
        if "archiveType" in timeseries.keys():
            archiveType = timeseries["archiveType"]
        else:
            archiveType = "NA"
            
        if "pub1_author" in timeseries.keys():
            authors = timeseries["pub1_author"]
        else:
            authors = "NA"
        
        #Truncate if more than two authors
        idx = [pos for pos, char in enumerate(authors) if char == ";"]
        if  len(idx)>2:
            authors = authors[0:idx[1]+1] + "et al."
        
        if "pub1_pubYear" in timeseries.keys():
            Year = str(timeseries["pub1_pubYear"])
        else:
            Year = "NA"
        
        if "pub1_DOI" in timeseries.keys():
            DOI = timeseries["pub1_DOI"]  
        else:
            DOI = "NA"
        
        if "paleoData_onInferredVariableProperty" in timeseries.keys():
            Variable = timeseries["paleoData_onInferredVariableProperty"]
        elif "paleoData_onProxyObservationProperty" in timeseries.keys():
            Variable = timeseries["paleoData_onProxyObservationProperty"]
        else:
            Variable = timeseries["paleoData_variableName"]
        
        if "paleoData_units" in timeseries.keys():
            units = timeseries["paleoData_units"]
        else:
            units = "NA"
        
        #Climate interpretation information
        if "paleoData_isotopeInterpretation" in timeseries.keys():
            interpretation = timeseries["paleoData_isotopeInterpretation"]
            ClimateVar = interpretation["name"]
            Detail = interpretation["detail"]
            Seasonality = interpretation["seasonality"]
            Direction = interpretation["interpretationDirection"]
        elif "paleoData_climateInterpretation" in timeseries.keys():
            interpretation = timeseries["paleoData_climateInterpretation"]
            ClimateVar = interpretation["name"]
            Detail = interpretation["detail"]
            Seasonality = interpretation["seasonality"]
            Direction = interpretation["interpretationDirection"]    
        else:
            ClimateVar = "NA"
            Detail = "NA"
            Seasonality = "NA"
            Direction = "NA"
        
        # Check that it's not in the old format
        
        if ClimateVar == "NA" and "climateInterpretation_variable" in timeseries.keys():
            ClimateVar = timeseries["climateInterpretation_variable"] 
        if Detail == "NA" and "climateInterpretation_variableDetail" in timeseries.keys():
            Detail = timeseries["climateInterpretation_variableDetail"]
        if Seasonality == "NA" and "climateInterpretation_seasonality" in timeseries.keys(): 
            Seasonality = timeseries["climateInterpretation_seasonality"]
        if Direction == "NA" and "climateInterpretation_interpDirection" in timeseries.keys():
            Direction = timeseries["climateInterpretation_interpDirection"]       
            
        # Calibration information
        if "calibration_equation" in timeseries.keys():
            Calibration_equation = timeseries["calibration_equation"]
        else:
            Calibration_equation = "NA"
            
        if  "calibration_calibrationReferences" in timeseries.keys():
            ref = timeseries["calibration_calibrationReferences"]
            if "author" in ref.keys():
                ref_author = ref["author"][0] # get the first author
            else:
                ref_author = "NA"
            if  "publicationYear" in ref.keys():
                ref_year = str(ref["publicationYear"])
            else: ref_year="NA"
            Calibration_notes = ref_author +"."+ref_year
        elif "calibration_notes" in timeseries.keys():
            Calibration_notes = timeseries["calibration_notes"]  
        else: Calibration_notes = "NA"
        
        #Truncate the notes if too long
        charlim = 30;
        if len(Calibration_notes)>charlim:
            Calibration_notes = Calibration_notes[0:charlim] + " ..."
            
        dict_out = {"archiveType" : archiveType,
                    "authors" : authors,
                    "Year": Year,
                    "DOI": DOI,
                    "Variable": Variable,
                    "units": units,
                    "Climate_Variable" : ClimateVar,
                    "Detail" : Detail,
                    "Seasonality" : Seasonality,
                    "Interpretation_Direction" : Direction,
                    "Calibration_equation" : Calibration_equation,
                    "Calibration_notes" : Calibration_notes}
        
        return dict_out
    
    def TsData(self,timeseries ="", x_axis = ""):
        """ Get the PaleoData with age/depth information
        
        Get the necessary information for the TS plots/necessary to allow for
        axes specification
        
        Args:
            timeseries: a single timeseries object. 
                By default, will prompt the user
            x-axis (str): The representation against which to plot the 
                paleo-data. Options are "age", "year", and "depth". 
                Default is to let the system choose if only one available 
                or prompt the user.
        Returns:
            dataframe - a dataframe containg the x- and y-values \n
            archiveType - the archiveType (for plot settings) \n
            x_axis_label - the label for the x-axis \n
            y_axis_label - the label for the y-axis \n
            headers - the headers of the dataframe
            
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

        return dataframe, archiveType, x_axis_label, y_axis_label, headers

    def agemodelData(self, timeseries =""):
        """Get the necessary information for the agemodel plot
        
        Args:
            timeseries: a single timeseries object. By default, will 
                prompt the user
                
        Returns:
            depth - the depth values \n
            age - the age values \n
            x_axis_label - the label for the x-axis \n
            y_axis_label - the label for the y-axis \n
            archiveType - the archiveType (for default plot settings)
            
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
                    age = timeseries['age']
                    if "ageUnits" in timeseries.keys():
                        y_axis_label = "Calendar Age (" +\
                                        timeseries["ageUnits"] +")"
                    else:
                        y_axis_label = "Calendar Age"
                elif choice == 1:
                    age = timeseries['year']
                    if "yearUnits" in timeseries.keys():
                        y_axis_label = "Year (" +\
                                        timeseries["yearUnits"] +")"
                    else:
                        y_axis_label = "Year"
                else:
                    sys.exit("Enter 0 or 1")
                    
            if "age" in timeseries.keys():
                age = timeseries['age']
                if "ageUnits" in timeseries.keys():
                    y_axis_label = "Calendar Age (" +\
                            timeseries["ageUnits"] +")"
                else:
                    y_axis_label = "Calendar Age"
                    
            if "year" in timeseries.keys():
                age = timeseries['year']
                if "yearUnits" in timeseries.keys():
                    y_axis_label = "Year (" +\
                            timeseries["ageUnits"] +")"
                else:
                    y_axis_label = "Year"
                    
            depth = timeseries['depth']
            if "depthUnits" in timeseries.keys():
                x_axis_label = "Depth (" + timeseries["depthUnits"] + ")"
            else:
                x_axis_label = "Depth"
        
        # Get the archiveType and make sure it aligns with the ontology
        archiveType = LipdToOntology(timeseries["archiveType"]) 
        
        return depth, age, x_axis_label, y_axis_label, archiveType
        
    def basic(self,x_axis="", timeseries = "", saveFig = False,
                     format = "eps", dir = ""):
        """ Makes a basic summary plot
        
        Plots the following information: the time series, location map, 
        Age-Depth profile if both are available from the paleodata, Metadata

        Notes: 
            The plots use default settings from the MapLiPD and plotTS methods.
    
        Arguments:
            new_timeseries: By default, will prompt for one.
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
        
        """
    
        # get the timeseries
        if not timeseries:
            timeseries = getTs(self.TS)
        
        # Create the figure
        fig = plt.figure(figsize=(11,8))
        gs = gridspec.GridSpec(2, 3)
        gs.update(left=0, right=1.1)
        
        # Get the necessary metadata
        dataframe, archiveType, x_axis_label, y_axis_label, headers = \
            self.TsData(timeseries=timeseries, x_axis=x_axis)
        
        # Make the time series plot
        ax1 = fig.add_subplot(gs[0,:-1])
        plt.style.use("ggplot")
        marker = [self.default[archiveType][0],self.default[archiveType][1]]
        markersize = 50
        if "year" in headers:
            ax1.scatter(dataframe['year'],dataframe['y-axis'],
                        s = markersize,
                        facecolor = 'none',
                        edgecolor = 'k',
                        marker = 'o',
                        label = 'original')
            ax1.plot(dataframe['year'],dataframe['y-axis'],
                    color = marker[0],
                    linewidth = 1.0,
                    label = 'interpolated')
        elif "age" in headers:
            ax1.scatter(dataframe['age'],dataframe['y-axis'],
                        s = markersize,
                        facecolor = 'none',
                        edgecolor = 'k',
                        marker = 'o',
                        label = 'original')
            ax1.plot(dataframe['age'],dataframe['y-axis'],
                    color = marker[0],
                    linewidth = 1.0,
                    label = 'interpolated')
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)  
        ax1.legend(loc=3, scatterpoints=1, fancybox=True, shadow=True, \
                   fontsize=10)
        plt.title(timeseries["dataSetName"], fontsize = 14, \
                  fontweight = "bold")

        # Plot the map
        # Get the coordinates. 
        
        lat = timeseries["geo_meanLat"]  
        lon = timeseries["geo_meanLon"]
    
        #Make the map
        ax2 = fig.add_subplot(gs[1,0])
        map = Basemap(projection='ortho', lon_0=lon, lat_0=lat)
        map.drawcoastlines()
        map.shadedrelief(scale=0.5)
        map.drawcountries()
        X,Y = map(lon,lat)
        map.scatter(X,Y,
                   s = 150,
                   color = self.default[archiveType][0],
                   marker = self.default[archiveType][1])
        
        #Make the age model plot
        if "age" in timeseries.keys() and "depth" in timeseries.keys()\
          or "year" in timeseries.keys() and "depth" in timeseries.keys():
            #Get the metadata
            x, y, x_axis_label2, y_axis_label2, archiveType = \
              self.agemodelData(timeseries=timeseries)
            #Make the plot
            ax3 = fig.add_subplot(gs[1,1])
            plt.style.use("ggplot")
            ax3.plot(x,y, color = marker[0], linewidth = 1.0)
            plt.xlabel(x_axis_label2)
            plt.ylabel(y_axis_label2)
        else:
            print("No age or depth information available, skipping age model plot")
        
        # Add the metadata
        dict_out = self.getMetadata(timeseries)
        textstr = "archiveType: " + dict_out["archiveType"]+"\n"+"\n"+\
                  "Authors: " + dict_out["authors"]+"\n"+"\n"+\
                  "Year: " + dict_out["Year"]+"\n"+"\n"+\
                  "DOI: " + dict_out["DOI"]+"\n"+"\n"+\
                  "Variable: " + dict_out["Variable"]+"\n"+"\n"+\
                  "units: " + dict_out["units"]+"\n"+"\n"+\
                  "Climate Interpretation: " +"\n"+\
                  "    Climate Variable: " + dict_out["Climate_Variable"] +"\n"+\
                  "    Detail: " + dict_out["Detail"]+"\n"+\
                  "    Seasonality: " + dict_out["Seasonality"]+"\n"+\
                  "    Direction: " + dict_out["Interpretation_Direction"]+"\n \n"+\
                  "Calibration: \n" + \
                  "    Equation: " + dict_out["Calibration_equation"] + "\n" +\
                  "    Notes: " + dict_out["Calibration_notes"]  
        plt.figtext(0.75, 0.4, textstr, fontsize = 12)
                                         
        
        if saveFig == True:
            name = 'SummaryPlot_'+timeseries["dataSetName"]+"_"+\
                timeseries["paleoData_variableName"]
            saveFigure(name,format,dir)
        else:
            plt.show()

        return fig    
