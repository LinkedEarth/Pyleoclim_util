# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:00:07 2016

@author: deborahkhider

Mapping functions.

Uses the LiPD files directly rather than timeseries objects

"""

import lipd as lpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sys
import os

#Import internal packages to pyleoclim

from .LiPDutils import *
from .Basic import *

class Map(object):
    """
    Create Maps using cartopy
    """

    def __init__(self, plot_default):
        """
        Passes the default color palette
        Opens the LiPD files loaded in the workspace and grabs the following metadata:
        1. Latitude
        2. Longitude
        3. archiveType
        4. The filename
        """
        
        # Organize the data        

        self.default= plot_default

        lipd_in_directory = lpd.getLipdNames()
                     
        self.data = pd.DataFrame({'Lat': np.array([0] * len(lipd_in_directory)),
                     'Lon': np.array([0] * len(lipd_in_directory)),
                     'archive':'archive',
                     'name':'name'})                   

        for idx,val in enumerate(lipd_in_directory):
            d = lpd.getMetadata(val)
            self.data.iloc[idx,0]=d['geo']['geometry']['coordinates'][1]
            self.data.iloc[idx,1]=d['geo']['geometry']['coordinates'][0]
            # Put the archiveType in
            self.data.iloc[idx,2] = LiPDtoOntology(d['archiveType'])
            self.data.iloc[idx,3]=val
  
    def map_all(self, markersize = 50, saveFig = False, dir="", format='eps'):
        """        
        Map all the available records loaded into the LiPD working directory by archiveType.
        Arguments:
          - markersize: default is 50
          - saveFig: default is to save the figure
          - dir: the full path of the directory in which to save the figure. If not provided, creates
          a default folder called 'figures' in the LiPD working directory (lipd.path). 
          - format: One of the file extensions supported by the active backend. Default is "eps".
          Most backend support png, pdf, ps, eps, and svg.
        """
        fig = plt.figure()
        map = Basemap(projection='robin',lon_0 = 0, lat_0 = 0)
        map.fillcontinents(color='0.9')
        map.drawcoastlines()
    
        already_plotted = [] # Check if the data has already been plotted

        for archiveType in self.data['archive']:
            if archiveType in self.default and \
            archiveType not in already_plotted:
                X,Y = map(np.asarray(self.data[self.data['archive']==archiveType]['Lon']),\
                  np.asarray(self.data[self.data['archive']==archiveType]['Lat']))
                map.scatter(X,Y,
                    s = markersize,
                    color = self.default[archiveType][0],
                    marker = self.default[archiveType][1],
                    label = archiveType)
                already_plotted.append(archiveType)
            elif archiveType not in self.default:
                X,Y = map(np.asarray(self.data[self.data['archive']==archiveType]['Lon']),\
                  np.asarray(self.data[self.data['archive']==archiveType]['Lat']))
                map.scatter(X,Y,
                    s = markersize,
                    color = 'k',
                    marker = 'o',
                    label = archiveType)  
        
        plt.legend(loc = 'center', bbox_to_anchor=(1.25,0.5),scatterpoints = 1,
                   frameon = False, fontsize = 8, markerscale = 0.7)
        
        if saveFig == True:
            saveFigure('map_all_liPDs',format,dir)
        else:
            plt.show()

        return fig    

    def map_one(self, name="", countries = True, counties = False, \
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
        # Check whether the record name was provided
        if not name:
            enumerateLipds()
            selection = promptforLipd()
            dataset = self.data.iloc[selection]['name']
        else:
            dataset = name
    
        if self.data[self.data['name']==dataset].empty:
            sys.exit("ERROR: The name you have entered is " +
            "not in the current directory. Make sure you entered "+
            "the name with the .lpd extension.")
            
        record = self.data[self.data['name']==dataset]

        # Get the coordinated
        lon = record['Lon'].iloc[0]
        lat = record['Lat'].iloc[0]

       # Make the figure
        fig = plt.figure()
        map = Basemap(projection='ortho', lon_0=lon, lat_0=lat)
        map.drawcoastlines()
 
        if background  == "shadedrelief":
            map.shadedrelief(scale=scale)
        elif background == "bluemarble":
            map.bluemarble(scale=scale)
        elif background == "etopo":
            map.etopo(scale=scale)
        elif background == 'none':
            map.fillcontinents(color='0.9')
        else:
            sys.exit("Enter either 'shadedrelief', 'bluemarble', 'etopo', or 'none'")

        if countries == True:
            map.drawcountries()

        if counties == True:
            map.drawcounties()

        if rivers == True:
            map.drawrivers()
            
        if states == True:
            map.drawstates()  
         
        if marker == "default":
            if record['archive'].iloc[0] in self.default:
                X,Y = map(lon,lat)
                map.scatter(X, Y,
                    s = markersize,
                    color = self.default[record['archive'].iloc[0]][0],
                    marker = self.default[record['archive'].iloc[0]][1]) 
            else:
                X,Y = map(lon,lat)
                ax.scatter(X, Y,
                    s = markersize,
                    color = 'k',
                    marker = 'o')
        else:
            X,Y = map(lon,lat)
            map.scatter(X, Y,
                    s = markersize,
                    color = marker[0],
                    marker = marker[1])

        if saveFig == True:
            figname = '/map_'+os.path.splitext(name)[0]
            saveFigure(figname,format,dir)            
        else:
            plt.show()

        return fig    
