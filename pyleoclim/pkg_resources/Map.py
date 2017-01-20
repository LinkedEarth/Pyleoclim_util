# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:00:07 2016

@author: deborahkhider

Mapping functions 

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

class Map(object):
    """
    Create Maps using cartopy
    """

    def __init__(self, plot_default):
        
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
  
    def map_all(self, markersize = int(50), saveFig = True, dir="", format='eps'):
        """        
        Make a map of all available records
        """
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black', \
            facecolor = ( 0.9, 0.9, 0.9))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.75, box.height])
    
        already_plotted = [] # Check if the data has already been plotted

        for archiveType in self.data['archive']:
            if archiveType in self.default and \
            archiveType not in already_plotted:
                ax.scatter(self.data[self.data['archive']==archiveType]['Lon'],
                    self.data[self.data['archive']==archiveType]['Lat'],
                    s = markersize,
                    facecolor = self.default[archiveType][0],
                    edgecolor = 'k',
                    marker = self.default[archiveType][1],
                    transform=ccrs.Geodetic(),
                    label = archiveType)
                already_plotted.append(archiveType)
            elif archiveType not in self.default:
                ax.scatter(self.data[self.data['archive']==archiveType]['Lon'],
                    self.data[self.data['archive']==archiveType]['Lat'],
                    s = markersize,
                    facecolor = 'k',
                    edgecolor = 'k',
                    marker = 'o',
                    transform=ccrs.Geodetic(),
                    label = 'other')    
        
        ax.legend(loc = 'center', bbox_to_anchor=(1.25,0.5),scatterpoints = 1,
                   frameon = False, fontsize = 8, markerscale = 0.7)
        
        if saveFig == True:
            saveFigure('map_all_liPDs',format,dir)
        else:
            plt.show()

    def map_one(self, name="",gridlines = False, borders = True, \
        topo = True, markersize = int(100), marker = "default", \
        saveFig = True, dir = "", format="eps"):
        """
        Map one particular record. 
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
        ax = plt.axes(projection=ccrs.Orthographic(record['Lon'].iloc[0], \
                                               record['Lat'].iloc[0]))
        if topo == True:
            ax.stock_img()
            ax.add_feature(cartopy.feature.LAND, 
               edgecolor='black', facecolor='none')
        else:
            ax.add_feature(cartopy.feature.LAND, 
               edgecolor='black', facecolor=[0.9375, 0.9375, 0.859375],\
               zorder = 0)               
        if borders == True:
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
        if gridlines == True:   
            ax.gridlines()
        ax.set_global()
        if marker == "default":
            if record['archive'].iloc[0] in self.default:    
                ax.scatter(record['Lon'].iloc[0], record['Lat'].iloc[0],
                    s = markersize,
                    facecolor = self.default[record['archive'].iloc[0]][0],
                    edgecolor = 'k',
                    marker = self.default[record['archive'].iloc[0]][1],
                    transform=ccrs.Geodetic()) 
            else:
                ax.scatter(record['Lon'].iloc[0], record['Lat'].iloc[0],
                    s = markersize,
                    facecolor = 'k',
                    edgecolor = 'k',
                    marker = 'o',
                    transform=ccrs.Geodetic())
        else:
            ax.scatter(record['Lon'].iloc[0], record['Lat'].iloc[0],
                    s = markersize,
                    facecolor = marker[0],
                    edgecolor = 'k',
                    marker = marker[1],
                    transform=ccrs.Geodetic())
        if saveFig == True:
            figname = '/map_'+os.path.splitext(name)[0]
            saveFigure(figname,format,dir)            
        else:
            plt.show()
