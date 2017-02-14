# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:07:07 2016

@author: deborahkhider

LiPD file manipulations. Except for maps, most manipulations are done on the timeseries objects.

See the LiPD documentation for more information on timeseries objects (TSO)

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
import pyleoclim 

"""
The following functions handle creating new directories and saving figures and logs
"""

def createDir(path, foldername):
    """Create a new folder in a working directory
    
    Create a new folder in a working directory to save outputs from Pyleoclim.
    
    Args:
        path(str): the path to the new folder.
        foldername(str): the name of the folder to be created
        
    Returns:
        newdir - the full path to the new directory
        
    """

    if not os.path.exists(path+'/'+foldername):
        os.makedirs(path+'/'+foldername)
    
    newdir = path+'/'+foldername    

    return newdir 

def saveFigure(name, format="eps",dir=""):
    """Save a figure
    
    Save the figure in the directory. If not given, creates a folder in the 
    lipd.path directory. 
    
    Args:
        name (str): name of the file
        format (str): One of the file extensions supported by the active 
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.
        dir (str): the name of the folder in the LiPD working directory.
            If not provided, creates a default folder called 'figures'.
            
    """
    if not dir:
        newdir = createDir(pyleoclim.lipd_path,"figures")            
        plt.savefig(newdir+'/'+name+'.'+format,\
                    bbox_inches='tight',pad_inches = 0.25)
    else:
        plt.savefig(dir+'/'+name+'.'+format,\
                    bbox_inches='tight',pad_inches = 0.25)           
    
""" 
The following functions handle the LiPD files
"""
    
def enumerateLipds():
    """Enumerate the LiPD files loaded in the workspace
    
    """
    lipd_in_directory = lpd.getLipdNames()
    print("Below are the available records")
    for idx, val in enumerate(lipd_in_directory):
        print(idx,': ',val)   

def promptForLipd():
    """Prompt for a LiPD file
    
    Ask the user to select a LiPD file from a list
    Use this function in conjunction with enumerateLipds()
    
    Returns:
        The index of the LiPD file
        
    """
    select_lipd = int(input("Enter the number of the file you wish to analyze: "))
    return select_lipd 
                                   
"""
The following functions work at the variables level
"""
        
def promptForVariable():
    """Prompt for a specific variable
    
    Ask the user to select the variable they are interested in.
    Use this function in conjunction with readHeaders() or getTSO()
    
    Returns:
        The index of the variable
        
    """
    select_var = int(input("Enter the number of the variable you wish to use: ")) 
    return select_var
                      
def valuesLoc(dataframe, missing_value = "NaN", var_idx = 1):
    """Remove missing values flag
    
    Look for the indexes where there are no missing values for the variable
    
    Args:
        dataframe: a Pandas Dataframe
        missing_value (str or float): how are the missing value represented. 
            Default is NaN
        var_idx (int): the column number in which to look for the missing 
            values (default is the second column)
            
    Returns:
        val_idx - the indices of the lines in the dataframe containing the actual values
            
    """
    
    # Get the index with the appropriate values
    val_idx = [] #Initiate the counter
    if missing_value == 'NaN':
        for n,i in enumerate(dataframe.iloc[:,var_idx]):
            if not np.isnan(i):
                val_idx.append(n)
    else:
        for n,i in enumerate(dataframe.iloc[:,var_idx]):
            if i!=float(missing_value):
                val_idx.append(n)

    return val_idx

def xAxisTs(timeseries):
    """ Prompt the user to choose a x-axis representation for the timeseries.
    
    Args:
        timeseries: a timeseries object
        
    Returns:
        x_axis - the values for the x-axis representation, \n
        label - returns either "age", "year", or "depth"
        
    """
    if "depth" in timeseries.keys() and "age" in timeseries.keys() or\
            "depth" in timeseries.keys() and "year" in timeseries.keys():
        print("Do you want to plot vs time or depth?")
        choice = int(input("Enter 0 for time and 1 for depth: "))
        if choice == 0:
            if "age" in timeseries.keys() and "year" in timeseries.keys():
                print("Do you want to use age or year?")
                choice2 = int(input("Enter 0 for age and 1 for year: "))
                if choice2 == 0:
                    x_axis = timeseries["age"]
                    label = "age"
                elif choice2 == 1:
                    x_axis = timeseries["year"]
                    label = "year"
                else:
                    sys.exit("Enter 0 or 1")
            elif "age" in timeseries.keys():
                x_axis = timeseries["age"]
                label = "age"
            elif "year" in timeseries.keys():
                x_axis = timeseries["year"]
                label = "year"            
        elif choice == 1:
            x_axis = timeseries["depth"]
            label = "depth"
        else: 
            sys.exit("Enter 0 or 1")
    elif "depth" in timeseries.keys():
        x_axis =  timeseries["depth"]
        label = "depth"
    elif "age" in timeseries.keys():
        x_axis = timeseries["age"]
        label = "age"
    elif "year" in timeseries.keys():
        x_axis = timeseries["year"]
        label = "year" 
    else: 
        sys.exist("No age or depth information available")
        
    return x_axis, label    
    
"""
The following functions handle the time series objects
"""
def enumerateTs(timeseries_list):
    """Enumerate the available time series objects
    
    Args:
        timeseries_list: a  list of available timeseries objects. 
            To use the timeseries loaded upon initiation of the 
            pyleoclim package, use pyleo.time_series.
            
    """
    available_y = []
    dataSetName =[]
    for index,val in enumerate(timeseries_list):
        for key, value in val.items():
            if 'dataSetName' in key:
                dataSetName.append(value)
            if 'variableName' in key:
                available_y.append(value)
             
    for idx,val in enumerate(available_y):
        print(idx,': ',dataSetName[idx], ': ', val)     

def getTs(timeseries_list):
    """Get a specific timeseries object from a dictionary of timeseries
    
    Args:
        timeseries_list: a  list of available timeseries objects. 
            To use the timeseries loaded upon initiation of the 
            pyleoclim package, use pyleo.time_series.
            
    Returns:
        A single timeseries object 
        
    """        
    enumerateTs(timeseries_list)
    select_TSO = promptForVariable()
    timeseries = timeseries_list[select_TSO]

    return timeseries

def TsToDf(timeseries, x_axis = ""):
    """ Timeseries to Dataframe
    
    Create a dataframe from a timeseries object with two colums: 
    depth/age representation and the paleoData values
    
    Args:
        timeseries: A timeseries object
        x-axis (str): The representation against which to plot the paleo-data. 
            Options are "age", "year", and "depth". Default is to let the 
            system choose if only one available or prompt the user.
    
    Returns:
        A Pandas Dataframe with two columns - the x-axis representation
        ("year", "age", or "depth") and the PaleoDataValues        
            
    """
    if not x_axis:
        x_axis, label = xAxisTs(timeseries)
    elif x_axis == "year":
        x_axis = timeseries["year"]
        label = "year"
    elif x_axis == "age":
        x_axis = timeseries["age"]
        label= "age"
    elif x_axis == "depth":
        x_axis = timeseries["depth"]
        label= "depth"
    else:
        sys.exit("Please enter 'depth' or 'age'")
    dataframe = pd.DataFrame({label:x_axis,
                          'y-axis':timeseries["paleoData_values"]}) 

    return dataframe    

""" 
Handle mapping to LinkedEarth Ontology if needed
"""    
def LipdToOntology(archiveType):
    """ standardize archiveType
    
    Transform the archiveType from their LiPD name to their ontology counterpart
    
    Args:
        archiveType (STR): name of the archiveType from the LiPD file
        
    Returns:
        archiveType according to the ontology
        
    """
    #Align with the ontology
    if archiveType.lower()== "ice core":
        archiveType = 'glacier ice'
    elif archiveType.lower()== 'tree':
        archiveType = 'wood'
    elif archiveType.lower() == 'borehole':
        archiveType = 'ice/rock'
    elif archiveType.lower() == 'bivalve':
        archiveType = 'molluskshells'
    
    return archiveType
