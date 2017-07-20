#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:26:47 2017

@author: deborahkhider

Methods necessary to make summary plots. These methods REQUIRE the use of a 
LiPD file

"""

import numpy as np
import sys

# Internal packages
import pyleoclim.LipdUtils as LipdUtils

#Methods

def getMetadata(timeseries):
    
    """ Get the necessary metadata to be printed out automatically
    
    Args:
        timeseries: a specific timeseries object. 
        
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
    
    if "paleoData_InferredVariableType" in timeseries.keys():
        Variable = timeseries["paleoData_InferredVariableType"]
    elif "paleoData_ProxyObservationType" in timeseries.keys():
        Variable = timeseries["paleoData_ProxyObservationType"]
    else:
        Variable = timeseries["paleoData_variableName"]
    
    if "paleoData_units" in timeseries.keys():
        units = timeseries["paleoData_units"]
    else:
        units = "NA"
    
    #Climate interpretation information
    if "paleoData_interpretation" in timeseries.keys():
        interpretation = timeseries["paleoData_interpretation"][0]
        if "name" in interpretation.keys():
            ClimateVar = interpretation["name"]
        elif "variable" in interpretation.keys():
            ClimateVar = interpretation["variable"]
        else:
            ClimateVar = "NA"
        if "detail" in interpretation.keys(): 
            Detail = interpretation["detail"]
        elif "variableDetail" in interpretation.keys():
            Detail = interpretation['variableDetail']
        else:
            Detail = "NA"
        if "scope" in interpretation.keys():
            Scope = interpretation['scope']
        else:
            Scope = "NA"
        if "seasonality" in interpretation.keys():    
            Seasonality = interpretation["seasonality"]
        else:
            Seasonality = "NA"
        if "interpdirection" in interpretation.keys():    
            Direction = interpretation["interpdirection"]
        else:
            Direction = "NA"
    else:
        ClimateVar = "NA"
        Detail = "NA"
        Scope = "NA"
        Seasonality = "NA"
        Direction = "NA"
        
    # Calibration information
    if "paleoData_calibration" in timeseries.keys():
        calibration = timeseries['paleoData_calibration'][0]
        if "equation" in calibration.keys():
            Calibration_equation = calibration["equation"]
        else:
            Calibration_equation = "NA"
        if  "calibrationReferences" in calibration.keys():
            ref = calibration["calibrationReferences"]
            if "author" in ref.keys():
                ref_author = ref["author"][0] # get the first author
            else:
                ref_author = "NA"
            if  "publicationYear" in ref.keys():
                ref_year = str(ref["publicationYear"])
            else: ref_year="NA"
            Calibration_notes = ref_author +"."+ref_year
        elif "notes" in calibration.keys():
            Calibration_notes = calibration["notes"]
        else: Calibration_notes = "NA"    
    else:
        Calibration_equation = "NA"
        Calibration_notes = "NA"
    
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
                "Scope":Scope,
                "Seasonality" : Seasonality,
                "Interpretation_Direction" : Direction,
                "Calibration_equation" : Calibration_equation,
                "Calibration_notes" : Calibration_notes}
    
    return dict_out    

def TsData(timeseries, x_axis=""):
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
        x - the x-valus \n
        y - the y-values \n
        archiveType - the archiveType (for plot settings) \n
        x_label - the label for the x-axis \n
        y_label - the label for the y-axis \n
        label - the results of the x-axis query. Either depth, year, or age
        
    """
    # Grab the x and y values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')   
    x, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)

    # Remove NaNs
    y_temp = np.copy(y)
    y = y[~np.isnan(y_temp)]
    x = x[~np.isnan(y_temp)]

    # Grab the archiveType
    archiveType = LipdUtils.LipdToOntology(timeseries["archiveType"])

    # x_label
    if label+"Units" in timeseries.keys():
        x_label = label[0].upper()+label[1:]+ " ("+timeseries[label+"Units"]+")"
    else:
        x_label = label[0].upper()+label[1:]   
    # ylabel
    if "paleoData_InferredVariableType" in timeseries.keys():
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_InferredVariableType"] + \
                      " (" + timeseries["paleoData_units"]+")" 
        else:
            y_label = timeseries["paleoData_InferredVariableType"]
    elif "paleoData_ProxyObservationType" in timeseries.keys():
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_ProxyObservationType"] + \
                      " (" + timeseries["paleoData_units"]+")" 
        else:
            y_label = timeseries["paleoData_ProxyObservationType"]
    else:
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_variableName"] + \
                      " (" + timeseries["paleoData_units"]+")" 
        else:
            y_label = timeseries["paleoData_variableName"]                

    return x,y,archiveType,x_label,y_label    

def agemodelData(timeseries):
    """Get the necessary information for the agemodel plot

    Args:
        timeseries: a single timeseries object. By default, will
            prompt the user

    Returns:
        depth - the depth values \n
        age - the age values \n
        x_label - the label for the x-axis \n
        y_label - the label for the y-axis \n
        archiveType - the archiveType (for default plot settings)

    """
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
                    age_label = "Calendar Age (" +\
                                    timeseries["ageUnits"] +")"
                else:
                    age_label = "Calendar Age"
            elif choice == 1:
                age = timeseries['year']
                if "yearUnits" in timeseries.keys():
                    age_label = "Year (" +\
                                    timeseries["yearUnits"] +")"
                else:
                    age_label = "Year"
            else:
                sys.exit("Enter 0 or 1")

        if "age" in timeseries.keys():
            age = timeseries['age']
            if "ageUnits" in timeseries.keys():
                age_label = "Calendar Age (" +\
                        timeseries["ageUnits"] +")"
            else:
                age_label = "Calendar Age"

        if "year" in timeseries.keys():
            age = timeseries['year']
            if "yearUnits" in timeseries.keys():
                age_label = "Year (" +\
                        timeseries["ageUnits"] +")"
            else:
                age_label = "Year"

        depth = timeseries['depth']
        if "depthUnits" in timeseries.keys():
            depth_label = "Depth (" + timeseries["depthUnits"] + ")"
        else:
            depth_label = "Depth"

    # Get the archiveType and make sure it aligns with the ontology
    archiveType = LipdUtils.LipdToOntology(timeseries["archiveType"])

    return depth, age, depth_label, age_label, archiveType