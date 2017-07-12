# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:00:07 2016

@author: deborahkhider

Mapping functions.

Uses the LiPD files directly rather than timeseries objects

"""
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sys


def mapAll(lat, lon, criteria, projection = 'robin', lat_0 = "", lon_0 = "",\
           llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, \
           countries = False, counties = False, rivers = False, states = False,\
           figsize = [10,4], ax = None,\
           background = 'none', scale = 0.5, palette="", markersize = 50):
    """ Map the location of all lat/lon according to some criteria 
    
    Map the location of all lat/lon according to some criteria. The choice of 
    plotting color/marker is passed through palette according to unique 
    criteria (e.g., record name, archive type, proxy observation type).
    
    Args:
        lat (list): a list of latitude.
        lon (list): a list of longitude.
        criteria (list): a list of criteria for plotting purposes. For instance,
            a map by the types of archive present in the dataset or proxy
            observations.
        projection (string): the map projection. Refers to the Basemap
            documentation for a list of available projections. Only projections
            supporting setting the map center with a single lat/lon or with
            the coordinates of the rectangle are currently supported. 
            Default is to use a Robinson projection.
        lat_0, lon_0 (float): the center coordinates for the map. Default is
            mean latitude/longitude in the list. 
            If the chosen projection doesn't support it, Basemap will
            ignore the given values.
        llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon (float): The coordinates
            of the two opposite corners of the rectangle.
        countries (bool): Draws the countries border. Defaults is off (False). 
        counties (bool): Draws the USA counties. Default is off (False).
        rivers (bool): Draws the rivers. Default is off (False).
        states (bool): Draws the American and Australian states borders. 
            Default is off (False).
        background (string): Plots one of the following images on the map: 
            bluemarble, etopo, shadedrelief, or none (filled continents). 
            Default is none.
        scale (float): Useful to downgrade the original image resolution to
            speed up the process. Default is 0.5.
        palette (dict): A dictionary of plotting color/marker by criteria. The
            keys should correspond to ***unique*** criteria with a list of 
            associated values. The list should be in the format 
            ['color', 'marker'].
        markersize (int): The size of the marker.
        figsize (list): the size for the figure
        ax: Return as axis instead of figure (useful to integrate plot into a subplot) 
        
    Returns:
        The figure       
    """
    #Check that the lists have the same length and convert to numpy arrays
    if len(lat)!=len(lon) or len(lat)!=len(criteria) or len(lon)!=len(criteria):
        sys.exit("Latitude, Longitude, and criteria list must be the same" +\
                 "length")
    
    # Grab the center latitude/longitude 
    if not lat_0:
        lat_0 = np.mean(np.array(lat))
    
    if not lon_0:
        lon_0 = np.mean(np.array(lon))
        
    # If palette is not given, then make a random one.
    if not palette:
        marker_list = ['o','v','^','<','>','8','s','p','*','h','D']
        color_list = ['#FFD600','#FF8B00','k','#86CDFA','#00BEFF','#4169E0',\
                 '#8A4513','r','#FF1492','#32CC32','#FFD600','#2F4F4F']
        # select at random for unique entries in criteria
        marker = [random.choice(marker_list) for _ in range(len(set(criteria)))]
        color = [random.choice(color_list) for _ in range(len(set(criteria)))]
        crit_unique = [crit for crit in set(criteria)]
        #initialize the palette
        palette = {crit_unique[0]:[color[0],marker[0]]}
        for i in range(len(crit_unique)):
            d1 = {crit_unique[i]:[color[i],marker[i]]}
            palette.update(d1)
            
    #Make the figure
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
        
    map = Basemap(projection = projection, lat_0 = lat_0, lon_0 = lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon)
    map.drawcoastlines()
    
    # Background
    if background == "shadedrelief":
        map.shadedrelief(scale = scale)
    elif background == "bluemarble":
        map.bluemarble(scale=scale)
    elif background == "etopo":
        map.etopo(scale=scale)
    elif background == "none":
        map.fillcontinents(color='0.9', lake_color = 'w')
    else:
        sys.exit("Enter either 'shadedrelief','bluemarble','etopo',or'None'")
            
    #Other extra information
    if countries == True:
        map.drawcountries()
    if counties == True:
        map.drawcounties()
    if rivers == True:
        map.drawrivers()
    if states == True:
        map.drawrivers()
    
    # Get the indexes by criteria
    for crit in set(criteria):
        # Grab the indices with same criteria
        index = [i for i,x in enumerate(criteria) if x == crit]
        X,Y =map(np.array(lon)[index],np.array(lat)[index])
        map.scatter(X,Y,
                    s= markersize,
                    facecolor = palette[crit][0],
                    marker = palette[crit][1],
                    zorder = 10,
                    label = crit)
    plt.legend(loc = 'center', bbox_to_anchor=(1.25,0.5),scatterpoints = 1,
               frameon = False, fontsize = 8, markerscale = 0.7)
    
    return ax    
        
def mapOne(lat, lon, projection = 'ortho', lat_0 = "", lon_0 = "",\
           llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180,\
           countries = True, counties = False, \
           rivers = False, states = False, background = "shadedrelief",\
           scale = 0.5, markersize = 50, marker = "ro", figsize = [4,4], \
           ax = None,):
    """ Map one location on the globe
    
    Args:
        lat (float): a float number representing latitude
        lon (float): a float number representing longitude
        projection (string): the map projection. Refers to the Basemap
            documentation for a list of available projections. Only projections
            supporting setting the map center with a single lat/lon or with
            the coordinates of the rectangle are currently supported. 
            Default is to use a Robinson projection.
        lat_0, lon_0 (float): the center coordinates for the map. Default is
            mean latitude/longitude in the list. 
            If the chosen projection doesn't support it, Basemap will
            ignore the given values.
        llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon (float): The coordinates
            of the two opposite corners of the rectangle.
        countries (bool): Draws the countries border. Defaults is off (False). 
        counties (bool): Draws the USA counties. Default is off (False).
        rivers (bool): Draws the rivers. Default is off (False).
        states (bool): Draws the American and Australian states borders. 
            Default is off (False).
        background (string): Plots one of the following images on the map: 
            bluemarble, etopo, shadedrelief, or none (filled continents). 
            Default is none.
        scale (float): Useful to downgrade the original image resolution to
            speed up the process. Default is 0.5.
        markersize (int): The size of the marker.
        marker (str or list): color and type of marker. 
        figsize (list): the size for the figure
        ax: Return as axis instead of figure (useful to integrate plot into a subplot) 
    
    """
    if not lon_0:
        lon_0 = lon
    if not lat_0:
        lat_0 = lat
    
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
        
    map = Basemap(projection=projection, lat_0=lat_0, lon_0=lon_0,
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon)
    
    map.drawcoastlines()
    
    # Background
    if background == "shadedrelief":
        map.shadedrelief(scale = scale)
    elif background == "bluemarble":
        map.bluemarble(scale=scale)
    elif background == "etopo":
        map.etopo(scale=scale)
    elif background == "none":
        map.fillcontinents(color='0.9')
    else:
        sys.exit("Enter either 'shadedrelief','bluemarble','etopo',or'None'")
            
    #Other extra information
    if countries == True:
        map.drawcountries()
    if counties == True:
        map.drawcounties()
    if rivers == True:
        map.drawrivers()
    if states == True:
        map.drawrivers()
    
    #Plot the point
    X,Y = map(lon,lat)
    map.scatter(X,Y,s=markersize,facecolor=marker[0],marker=marker[1],zorder=10) 
    
    return ax