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

    def __init__(self,timeseries_dict,plot_default):
        """
        Passes the dictionary of timeseries and the color palette
        """
        self.TS = timeseries_dict
        self.default = plot_default
        
    def getMetadata(self, time_series):
        """
        Get the necessary metadata to be printed out automatically
        Arguments:
          - time_series: a specific timeseries object
        Outputs:
          - A dictionary containing the following metadata:
            - archiveType
            - Authors (if more than 2, replace by et al.
            - PublicationYear
            - Publication DOI
            - Variable Name
            - Units
            - Climate Interpretation
            - Calibration Equation
            - Calibration References
            - Calibration Notes
        """
        # Get all the necessary information
        # Top level information
        if "archiveType" in time_series.keys():
            archiveType = time_series["archiveType"]
        else:
            archiveType = "NA"
            
        if "pub1_author" in time_series.keys():
            authors = time_series["pub1_author"]
        else:
            authors = "NA"
        
        #Truncate if more than two authors
        idx = [pos for pos, char in enumerate(authors) if char == ";"]
        if  len(idx)>2:
            authors = authors[0:idx[1]+1] + "et al."
        
        if "pub1_pubYear" in time_series.keys():
            Year = str(time_series["pub1_pubYear"])
        else:
            Year = "NA"
        
        if "pub1_DOI" in time_series.keys():
            DOI = time_series["pub1_DOI"]  
        else:
            DOI = "NA"
        
        if "paleoData_onInferredVariableProperty" in time_series.keys():
            Variable = time_series["paleoData_onInferredVariableProperty"]
        elif "paleoData_onProxyObservationProperty" in time_series.keys():
            Variable = time_series["paleoData_onProxyObservationProperty"]
        else:
            Variable = time_series["paleoData_variableName"]
        
        if "paleoData_units" in time_series.keys():
            units = time_series["paleoData_units"]
        else:
            units = "NA"
        
        #Climate interpretation information
        if "paleoData_isotopeInterpretation" in time_series.keys():
            interpretation = time_series["paleoData_isotopeInterpretation"]
            ClimateVar = interpretation["name"]
            Detail = interpretation["detail"]
            Seasonality = interpretation["seasonality"]
            Direction = interpretation["interpretationDirection"]
        elif "paleoData_climateInterpretation" in time_series.keys():
            interpretation = time_series["paleoData_climateInterpretation"]
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
        
        if ClimateVar == "NA" and "climateInterpretation_variable" in time_series.keys():
            ClimateVar = time_series["climateInterpretation_variable"] 
        if Detail == "NA" and "climateInterpretation_variableDetail" in time_series.keys():
            Detail = time_series["climateInterpretation_variableDetail"]
        if Seasonality == "NA" and "climateInterpretation_seasonality" in time_series.keys(): 
            Seasonality = time_series["climateInterpretation_seasonality"]
        if Direction == "NA" and "climateInterpretation_interpDirection" in time_series.keys():
            Direction = time_series["climateInterpretation_interpDirection"]       
            
        # Calibration information
        if "calibration_equation" in time_series.keys():
            Calibration_equation = time_series["calibration_equation"]
        else:
            Calibration_equation = "NA"
            
        if  "calibration_calibrationReferences" in time_series.keys():
            ref = time_series["calibration_calibrationReferences"]
            if "author" in ref.keys():
                ref_author = ref["author"][0] # get the first author
            else:
                ref_author = "NA"
            if  "publicationYear" in ref.keys():
                ref_year = str(ref["publicationYear"])
            else: ref_year="NA"
            Calibration_notes = ref_author +"."+ref_year
        elif "calibration_notes" in time_series.keys():
            Calibration_notes = time_series["calibration_notes"]  
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
    
    def TSdata(self,new_timeseries ="", x_axis = ""):
        """
        Get the necessary information for the TS plots/necessary to allow for
        axes spec
        Arguments:
         - a single timeseries object. By default, will prompt the user
         - x-axis: The representation against which to plot the paleo-data. Options are "age",
    "year", and "depth". Default is to let the system choose if only one available or prompt
    the user.
        """
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

        return dataframe, archiveType, x_axis_label, y_axis_label, headers

    def agemodelData(self, new_timeseries =""):
        """
        Get the necessary information for the agemodel plot
        Arguments:
          - new_timeseries: a single timeseries object. By default, will prompt the user
        """
    
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
        
        return x, y, x_axis_label, y_axis_label, archiveType
        
    def basic(self,x_axis="", new_timeseries = "", saveFig = True,
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
    
        # get the timeseries
        if not new_timeseries:
            new_timeseries = getTSO(self.TS)
        
        # Create the figure
        fig = plt.figure(figsize=(11,8))
        gs = gridspec.GridSpec(2, 3)
        gs.update(left=0, right=1.1)
        
        # Get the necessary metadata
        dataframe, archiveType, x_axis_label, y_axis_label, headers = \
            self.TSdata(new_timeseries=new_timeseries, x_axis=x_axis)
        
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
        plt.title(new_timeseries["dataSetName"], fontsize = 14, \
                  fontweight = "bold")

        # Plot the map
        # Get the coordinates. 
        
        lat = new_timeseries["geo_meanLat"]  
        lon = new_timeseries["geo_meanLon"]
    
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
        if "age" in new_timeseries.keys() and "depth" in new_timeseries.keys()\
          or "year" in new_timeseries.keys() and "depth" in new_timeseries.keys():
            #Get the metadata
            x, y, x_axis_label2, y_axis_label2, archiveType = \
              self.agemodelData(new_timeseries=new_timeseries)
            #Make the plot
            ax3 = fig.add_subplot(gs[1,1])
            plt.style.use("ggplot")
            ax3.plot(x,y, color = marker[0], linewidth = 1.0)
            plt.xlabel(x_axis_label2)
            plt.ylabel(y_axis_label2)
        else:
            print("No age or depth information available, skipping age model plot")
        
        # Add the metadata
        dict_out = self.getMetadata(new_timeseries)
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
            name = 'SummaryPlot_'+new_timeseries["dataSetName"]+"_"+\
                new_timeseries["paleoData_variableName"]
            saveFigure(name,format,dir)
        else:
            plt.show()

        return fig    
