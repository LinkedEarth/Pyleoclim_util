# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:07:07 2016

@author: deborahkhider

LiPD file manipulations. Except for maps, most manipulations are done on the time series objects

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

"""
The following functions handle creating new directories and saving figures/logs
"""

def createdir(path, foldername):
    """
    create a new folder in the working directory
    """

    if not os.path.exists(path+'/'+foldername):
        os.makedirs(path+'/'+foldername)
    
    newdir = path+'/'+foldername    

    return newdir 

def saveFigure(name, format="eps",dir=""):
    """
    Save the figures if asked
    Input:
        name: name of the file
        format: the chosen format (default is eps)
        dir: the directory in which to save the file (default is a "figures" folder in the LiPD file directory)
    """
    if not dir:
        newdir = createdir(lpd.path,"figures")            
        plt.savefig(newdir+'/'+name+'.'+format,\
                    bbox_inches='tight',pad_inches = 0.25)
    else:
        plt.savefig(dir+'/'+name+'.'+format,\
                    bbox_inches='tight',pad_inches = 0.25)           
    
""" 
The following functions handle the LiPD files
"""
    
def enumerateLipds():
    """
    enumerate the LiPDs loaded in the workspace
    """
    lipd_in_directory = lpd.getLipdNames()
    print("Below are the available records")
    for idx, val in enumerate(lipd_in_directory):
        print(idx,': ',val)   

def promptforLipd():
    """
    Ask the user to select a LiPD file from a list
    Use this function in conjunction with enumerateLipds()
    """
    select_lipd = int(input("Enter the number of the file you wish to analyze: "))
    return select_lipd 
                                   
"""
The following functions work at the variables level
"""
        
def promptforVariable():
    """
    Ask the user to select the variable they are interested in.
    Use this function in conjunction with readHeaders or getTSO
    """
    select_var = int(input("Enter the number of the variable you wish to use: ")) 
    return select_var
                      
def valuesloc(dataframe, missing_value = "NaN", var_idx = 1):
    """
    Look for the indexes where there are no missing values for the variable
    Inputs:
        - Dataframe
        - missing_value: how are the missing value represented. Default is NaN
        - var_idx: the column number (default is the second column)
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

def TSOxaxis(time_series): 
    if "depth" in time_series.keys() and "age" in time_series.keys() or\
            "depth" in time_series.keys() and "year" in time_series.keys():
        print("Do you want to plot vs time or depth?")
        choice = int(input("Enter 0 for time and 1 for depth: "))
        if choice == 0:
            if "age" in time_series.keys() and "year" in time_series.keys():
                print("Do you want to use age or year?")
                choice2 = int(input("Enter 0 for age and 1 for year: "))
                if choice2 == 0:
                    x_axis = time_series["age"]
                    label = "age"
                elif choice2 == 1:
                    x_axis = time_series["year"]
                    label = "year"
                else:
                    sys.exit("Enter 0 or 1")
            elif "age" in time_series.keys():
                x_axis = time_series["age"]
                label = "age"
            elif "year" in time_series.keys():
                x_axis = time_series["year"]
                label = "year"            
        elif choice == 1:
            x_axis = time_series["depth"]
            label = "depth"
        else: 
            sys.exit("Enter 0 or 1")
    elif "depth" in time_series.keys():
        x_axis =  time_series["depth"]
        label = "depth"
    elif "age" in time_series.keys():
        x_axis = time_series["age"]
        label = "age"
    elif "year" in time_series.keys():
        x_axis = time_series["year"]
        label = "year" 
    else: 
        sys.exist("No age or depth information available")
        
    return x_axis, label    
    
"""
The following functions handle the time series objects
"""
def enumerateTSO(time_series):
    """
    Enumerate the available time series objects
    """
    available_y = []
    dataSetName =[]
    for index,val in enumerate(time_series):
        for key, value in val.items():
            if 'dataSetName' in key:
                dataSetName.append(value)
            if 'variableName' in key:
                available_y.append(value)
             
    for idx,val in enumerate(available_y):
        print(idx,': ',dataSetName[idx], ': ', val)     

def getTSO(time_series):
    """
    Get a specific time series object
    """        
    enumerateTSO(time_series)
    select_TSO = promptforVariable()
    new_TSO = time_series[select_TSO]

    return new_TSO

def TStoDF(time_series, x_axis = ""):
    if not x_axis:
        x_axis, label = TSOxaxis(time_series)
    elif x_axis == "year":
        x_axis = time_series["year"]
        label = "year"
    elif x_axis == "age":
        x_axis = time_series["age"]
        label= "age"
    elif x_axis == "depth":
        x_axis = time_series["depth"]
        label= "depth"
    else:
        sys.exit("Please enter 'depth' or 'age'")
    dataframe = pd.DataFrame({label:x_axis,
                          'y-axis':time_series["paleoData_values"]}) 

    return dataframe    

""" 
Handle mapping to LinkedEarth Ontology if needed
"""    
def LiPDtoOntology(archiveType):
    """
    Transform the archiveType from their LiPD name to their ontology counterpart
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
