# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:42:34 2016

@author: deborahkhider

License agreement - GNU GENERAL PUBLIC LICENSE v3
https://github.com/LinkedEarth/Pyleoclim_util/blob/master/license

"""
#Import all the needed packages
import lipd as lpd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import gridspec
import seaborn as sns
import sys
from itertools import chain
from scipy.stats.mstats import mquantiles
import datetime
import os
from collections import OrderedDict


# Import internal modules to pyleoclim
from pyleoclim import Map
from pyleoclim import LipdUtils
from pyleoclim import SummaryPlots
from pyleoclim import Plot
from pyleoclim import Spectral
from pyleoclim import Stats
from pyleoclim import Timeseries
from pyleoclim import RBchron


"""
Open Lipd files and extract timeseries (set them as global variable)
 
"""

def openLipd(usr_path=""):
    """Read Lipd files into a dictionary
    
    Sets the dictionary as global variable so that it doesn't have to be provided
    as an argument for every function.
    
    Args:
        usr_path (str): The path to a directory or a single file. (Optional argument)
        
    Returns:
        lipd_dict - a dictionary containing the LiPD library
    
    """
    global lipd_dict
    lipd_dict = lpd.readLipd(usr_path=usr_path)
    return lipd_dict

def fetchTs(lipds=None):
    """Extract timeseries dictionary
    
    This function is based on the function of the same name in the LiPD utilities.
    Set the dictionary as a global variable so that it doesn't have to be
    provided as an argument for every function. 
    
    Args:
        lipds (dict): A dictionary of LiPD files obtained through the 
        readLipd function
    
    Returns:
        ts_list - A list of timeseries object
    
    """
    global ts_list
    if not lipds:
        if 'lipd_dict' not in globals():
            openLipd()
            
        ts_list = lpd.extractTs(lipd_dict)
        
    else:
        ts_list = lpd.extractTs(lipds)
    return ts_list
        

"""
Set a few global variables
"""
      
#Set the default palette for plots

plot_default = {'ice/rock': ['#FFD600','h'],
                'coral': ['#FF8B00','o'],
                'documents':['k','p'],
                'glacier ice':['#86CDFA', 'd'],
                'hybrid': ['#00BEFF','*'],
                'lake sediment': ['#4169E0','s'],
                'marine sediment': ['#8A4513', 's'],
                'sclerosponge' : ['r','o'],
                'speleothem' : ['#FF1492','d'],
                'wood' : ['#32CC32','^'],
                'molluskshells' : ['#FFD600','h'],
                'peat' : ['#2F4F4F','*'],
                'other':['k','o']}

"""
Mapping
"""
def mapAllArchive(lipds = "", markersize = 50, background = 'shadedrelief',\
                  figsize = [10,4],\
                  saveFig = False, dir="", format='eps'):
    """Map all the available records loaded into the workspace by archiveType.

    Map of all the records into the workspace by archiveType.
        Uses the default color palette. Enter pyleoclim.plot_default for detail.

    Args:
        lipds (dict): A list of LiPD files. (Optional)
        markersize (int): The size of the markers. Default is 50
        background (str): Plots one of the following images on the map:
            bluemarble, etopo, shadedrelief, or none (filled continents).
            Default is shadedrelief.
        figsize (list): the size for the figure
        ax: Return as axis instead of figure (useful to integrate plot into a subplot)     
        saveFig (bool): Default is to not save the figure
        dir (str): The absolute path of the directory in which to save the
            figure. If not provided, creates a default folder called 'figures'
            in the LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.

    Returns:
        The figure
    """
    
    # Get the dictionary of LiPD files
    if not lipds:
        if 'lipd_dict' not in globals():
            openLipd()
        lipds = lipd_dict
        
    # Initialize the various lists
    lat = []
    lon = []
    archiveType = []

    # Loop ang grab the metadata
    for idx, key in enumerate(lipds):
        d = lipds[key]
        lat.append(d['geo']['geometry']['coordinates'][1])
        lon.append(d['geo']['geometry']['coordinates'][0])
        archiveType.append(LipdUtils.LipdToOntology(d['archiveType']).lower())


    # make sure criteria is in the plot_default list
    for idx,val in enumerate(archiveType):
        if val not in plot_default.keys():
            archiveType[idx] = 'other'


    # Make the map
    fig = Map.mapAll(lat,lon,archiveType,lat_0=0,lon_0=0,palette=plot_default,\
                     background = background, markersize = markersize,\
                     figsize=figsize, ax=None)

    # Save the figure if asked
    if saveFig == True:
        LipdUtils.saveFigure('mapLipds_archive', format, dir)
    else:
        plt.show()

    return fig

def mapLipd(timeseries="", countries = True, counties = False, \
        rivers = False, states = False, background = "shadedrelief",\
        scale = 0.5, markersize = 50, marker = "default", \
        figsize = [4,4], \
        saveFig = False, dir = "", format="eps"):
    """ Create a Map for a single record

    Orthographic projection map of a single record.

    Args:
        timeseries: a LiPD timeseries object. Will prompt for one if not given
        countries (bool): Draws the country borders. Default is on (True).
        counties (bool): Draws the USA counties. Default is off (False).
        rivers (bool): Draws the rivers. Default is off (False).
        states (bool): Draws the American and Australian states borders.
            Default is off (False)
        background (str): Plots one of the following images on the map:
            bluemarble, etopo, shadedrelief, or none (filled continents).
            Default is shadedrelief
        scale (float): useful to downgrade the original image resolution to
            speed up the process. Default is 0.5.
        markersize (int): default is 50
        marker (str): a string (or list) containing the color and shape of the
            marker. Default is by archiveType. Type pyleo.plot_default to see
            the default palette.
        figsize (list): the size for the figure
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.

    Returns:
        The figure

    """
    # Make sure there are LiPD files to plot
    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list)    

    # Get latitude/longitude

    lat = timeseries['geo_meanLat']
    lon = timeseries['geo_meanLon']

    # Make sure it's in the palette
    if marker == 'default':
        archiveType = LipdUtils.LipdToOntology(timeseries['archiveType']).lower()
        if archiveType not in plot_default.keys():
            archiveType = 'other'
        marker = plot_default[archiveType]
    

    fig = Map.mapOne(lat,lon,marker=marker,markersize=markersize,\
                     countries = countries, counties = counties,rivers = rivers, \
                     states = states, background = background, scale =scale,
                     ax=None, figsize = figsize)

    # Save the figure if asked
    if saveFig == True:
        LipdUtils.saveFigure(timeseries['dataSetName']+'_map', format, dir)
    else:
        plt.show()

    return fig

class MapFilters():
    """Create the various filters for mapping purposes
    """
    
    def getData(self, lipd_dict):
        """Initializes the object and store the information into lists        
        """
        lat = []
        lon = []
        archiveType = []
        dataSetName =[]
    
        for idx, key in enumerate(lipd_dict):
            d = lipd_dict[key]
            lat.append(d['geo']['geometry']['coordinates'][1])
            lon.append(d['geo']['geometry']['coordinates'][0])
            archiveType.append(LipdUtils.LipdToOntology(d['archiveType']))  
            dataSetName.append(d['dataSetName'])
        
        # make sure criteria is in the plot_default list
        for idx,val in enumerate(archiveType):
            if val not in plot_default.keys():
                archiveType[idx] = 'other'
        
        return lat, lon, archiveType, dataSetName          

    def distSphere(self, lat1,lon1,lat2,lon2):
        """Uses the harversine formula to calculate distance on a sphere
        
        Args:
            lat1: Latitude of the first point, in radians
            lon1: Longitude of the first point, in radians
            lat2: Latitude of the second point, in radians
            lon2: Longitude of the second point, in radians
            
        Returns:
            The distance between the two point in km
        
        """
        R = 6371  #km. Earth's radius
        dlat = lat2-lat1
        dlon = lon2-lon1
        
        a = np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))   
        dist =R*c
        
        return dist
    
    def computeDist(self, lat_r, lon_r, lat_c, lon_c):
        """ Computes the distance in (km) between a reference point and an array
        of other coordinates.
        
        Args:
            lat_r (float): The reference latitude, in deg
            lon_r (float): The reference longitude, in deg
            lat_c (list): An list of latitudes for the comparison points, in deg
            lon_c (list): An list of longitudes for the comparison points, in deg
            
        Returns:
            dist - a list of distances in km. 
        """
        dist = []
        
        for idx, val in enumerate (lat_c):
            lat1 = np.radians(lat_r)
            lon1 = np.radians(lon_r)
            lat2 = np.radians(val)
            lon2 = np.radians(lon_c[idx])
            dist.append(self.distSphere(lat1,lon1,lat2,lon2))
        
        return dist
    
    def filterByArchive(self,archiveType,option):
        """Returns the indexes of the records with the matching archiveType
        
        Args:
            archiveType (list): A list of ArchiveType
            option: the matching string
            
        Returns:
            idx (list): The indices matching the query
        """        
        idx = [idx for idx,val in enumerate(archiveType) if val==option]
        if not idx:
            print("Warning: Your search criteria doesn't match any record in the database")
        
        return idx 

    def withinDistance(self, distance, radius):
        """ Returns the index of the records that are within a certain distance
        
        Args:
            distance (list): A list containing the distance
            radius (float): the radius to be considered
            
        Returns:
            idx (list): a list of index
        """ 
        idx = [idx for idx,val in enumerate(distance) if val <= radius]

        return idx

    def filterList(self, lat, lon, archiveType, dist, dataSetName, idx):
        """Filters the list by the given indexes
        
        Args:
            lat (array): Array of latitudes
            lon (array): Array of longitudes
            archiveType (array): Array of ArchiveTypes
            dist (array): Array of distances
            dataSetName (array): Array of dataset names
            idx (array): An array of indices used for the filtering
            
        Returns:
            The previous arrays filtered by indexes
            
        """
        
        lat =  np.array(lat)[idx]
        lon = np.array(lon)[idx]
        archiveType = np.array(archiveType)[idx]
        dataSetName = np.array(dataSetName)[idx]
        dist =np.array(dist)[idx]
        
        return lat, lon, archiveType, dataSetName, dist
        

def mapNearRecords(timeseries = "", lipds = "", n = 5, radius = None, \
                   sameArchive = False, projection = 'ortho', lat_0 = "", \
                   lon_0="", llcrnrlat = -90, urcrnrlat=90, llcrnrlon=-180, 
                   urcrnrlon=180, countries = True, counties = False, \
                   rivers = False, states = False, \
                   background = "shadedrelief", scale = 0.5, markersize = 200,\
                   markersize_adjust = True, marker_r = "ko", \
                   marker_c = "default", cmap = "Reds", colorbar = True,\
                   location = "right", label = "Distance in km",
                   figsize = [4,4],ax = None, saveFig = False, dir = "", \
                   format = "eps"):
    
    """ Map the nearest records from the record of interest
    
    Args:
        timeseries (dict): A timeseries object. If none given, will prompt for one
        lipds (list): A list of LiPD files. (Optional)
        n (int): the number of records to match
        radius (float): The distance (in km) to search for nearby records.
            Default is to search the entire globe
        sameArchive (bool): Returns only records with the same archiveType.
            Default is not to do so.
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
        markersize (int): the size of the marker
        markersize_adjust (bool): If True, will proportionaly adjust the size of
            the marker according to distance.
        marker_r (list or str): The color and shape of the marker for the
            reference record.
        marker_c (list or str): The color and shape of the marker for the other
            records. Default is to use the color palette by archiveType. If set
            to None then the color of the marker will represent the distance from
            the reference records.
        cmap (str): The colormap to use to represent the distance from the 
            reference record if no marker is selected.
        colorbar (bool): Create a colorbar. Default is True
        location (str): Location of the colorbar
        label (str): Label for the colorbar.
        figsize (list): the size for the figure
        ax: Return as axis instead of figure (useful to integrate plot into a subplot)
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.

        
    Returns:
        ax - The figure       
    
    """
    
    # Get the dictionary of LiPD files
    if not lipds:
        if 'lipd_dict' not in globals():
            openLipd()
        lipds = lipd_dict
    
    # Get a timeseries if not given
    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list)
    
    # Get the data
    newfilter = MapFilters()
    lat, lon, archiveType, dataSetName = newfilter.getData(lipds)
    # Calculate the distance
    dist = newfilter.computeDist(timeseries["geo_meanLat"],
                                 timeseries["geo_meanLon"],
                                 lat,
                                 lon)
    
    # Filter to remove the reference record from the list
    idx_zero = np.flatnonzero(np.array(dist))
    if len(idx_zero)==0:
        sys.exit("No matching records found. Change your search criteria!")
    lat, lon, archiveType, dataSetName, dist = newfilter.filterList(lat,lon,
                                                                    archiveType,
                                                                    dist,
                                                                    dataSetName,
                                                                    idx_zero)
    
    #Filter to be within the radius
    if radius:
       idx_radius = newfilter.withinDistance(dist, radius)
       if len(idx_radius)==0:
           sys.exit("No matching records found. Change your search criteria!")
       lat, lon, archiveType, dataSetName, dist = newfilter.filterList(lat,lon,
                                                                    archiveType,
                                                                    dist,
                                                                    dataSetName,
                                                                    idx_radius)
    
    # Same archive if asked
    if sameArchive == True:
        idx_archive = newfilter.filterByArchive(archiveType,timeseries["archiveType"])
        if len(idx_archive)==0:
            sys.exit("No matching records found. Change your search criteria!")
        lat, lon, archiveType, dataSetName, dist = newfilter.filterList(lat,lon,
                                                                    archiveType,
                                                                    dist,
                                                                    dataSetName,
                                                                    idx_archive)
    
    #Print a warning if plotting less than asked because of the filters
    if n>len(dist):    
        print("Warning: Number of matching records is less"+\
              " than the number of neighbors chosen. Including all records "+\
              " in the analysis.")
        n = len(dist)
    
    # Sort the distance array
    sort_idx = np.argsort(dist)
    lat, lon, archiveType, dataSetName, dist = newfilter.filterList(lat,lon,
                                                                    archiveType,
                                                                    dist,
                                                                    dataSetName,
                                                                    sort_idx)
    
    # Grab the right number of records
    dist = dist[0:n]
    lat = lat[0:n]
    lon = lon[0:n]
    archiveType = archiveType[0:n]
    dataSetName = dataSetName[0:n]  
    
    # Make the map
    if not lon_0:
        lon_0 = timeseries["geo_meanLon"]
        
    if not lat_0:
        lat_0 = timeseries["geo_meanLat"]
    
    if not ax:
        fig,ax =  plt.subplots(figsize=figsize)
        
    map = Basemap(projection=projection, lat_0 = lat_0, lon_0 = lon_0,\
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
    elif not background:
        map.fillcontinents(color='0.5')
    else:
        sys.exit("Enter either 'shadedrelief','bluemarble','etopo',or None")
         
    # Other extra information
    
    if countries == True:
        map.drawcountries()
    if counties == True:
        map.drawcounties()
    if rivers == True:
        map.drawrivers()
    if states == True:
        map.drawrivers()
    
        
    X_r,Y_r = map(timeseries["geo_meanLon"],timeseries["geo_meanLat"]) 
    map.scatter(X_r, Y_r, s=markersize, facecolor = marker_r[0], \
                marker = marker_r[1], zorder =10)
    
    #Either plot single color or gradient
    
    if not marker_c:
        X_c, Y_c = map(lon,lat)
        CS = map.scatter(X_c, Y_c, s=markersize, c = dist, zorder =10, cmap = cmap,\
                         marker = '^')
        if colorbar == True:
           cb = map.colorbar(CS,location)
           if not not label:
               cb.set_label(label)          
    elif marker_c == "default":
        dist_max = np.max(dist)
        dist_adj = np.ceil(dist*markersize/dist_max)
        #Use the archive specific markers
        for archive in archiveType:
           index = [i for i,x, in enumerate(archiveType) if x == archive]
           X_c, Y_c = map(lon[index],lat[index])
           if markersize_adjust == True:
               map.scatter(X_c, Y_c, s = dist_adj[index],
                       facecolor = plot_default[archive][0],
                       marker = plot_default[archive][1],
                       zorder = 10)
           else:
               map.scatter(X_c, Y_c, s = markersize,
                       facecolor = plot_default[archive][0],
                       marker = plot_default[archive][1],
                       zorder = 10)
    else:
        X_c, Y_c = map(lon,lat)
        dist_max = np.max(dist)
        dist_adj = np.ceil(dist*markersize/dist_max)
        if markersize_adjust == True:
            map.scatter(X_c, Y_c, s=dist_adj, zorder =10, marker = marker_c[1],\
                    facecolor = marker_c[0])
        else:
            map.scatter(X_c, Y_c, s=markersize, zorder =10, marker = marker_c[1],\
                    facecolor = marker_c[0])
            
    # Save the figure if asked
    if saveFig == True:
        LipdUtils.saveFigure(timeseries['dataSetName']+'_map', format, dir)
    else:
        plt.show()
            
    return ax


"""
Plotting
"""

def plotTs(timeseries = "", x_axis = "", markersize = 50,\
            marker = "default", figsize =[10,4],\
            saveFig = False, dir = "",\
            format="eps"):
    """Plot a single time series.

    Args:
        A timeseries: By default, will prompt the user for one.
        x_axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the
            system choose if only one available or prompt the user.
        markersize (int): default is 50.
        marker (str): a string (or list) containing the color and shape of the
            marker. Default is by archiveType. Type pyleo.plot_default to see
            the default palette.
        figsize (list): the size for the figure
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
    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list) 

    y = np.array(timeseries['paleoData_values'], dtype = 'float64')
    x, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)

    # remove nans and sort time axis
    y,x = Timeseries.clean_ts(y,x)

    # get the markers
    if marker == 'default':
        archiveType = LipdUtils.LipdToOntology(timeseries['archiveType']).lower()
        if archiveType not in plot_default.keys():
            archiveType = 'other'
        marker = plot_default[archiveType]

    # Get the labels
    # title
    title = timeseries['dataSetName']
    # x_label
    if label+"Units" in timeseries.keys():
        x_label = label[0].upper()+label[1:]+ " ("+timeseries[label+"Units"]+")"
    else:
        x_label = label[0].upper()+label[1:]
    # ylabel
    if "paleoData_inferredVariableType" in timeseries.keys():
        #This if loop is needed because some files appear to have two
        #different inferredVariableType/proxyObservationType, which
        #should not be possible in the ontology. Just build some checks
        # in the system
        if type(timeseries["paleoData_inferredVariableType"]) is list:
            var = timeseries["paleoData_inferredVariableType"][0]
        else:
            var = timeseries["paleoData_inferredVariableType"]
        if "paleoData_units" in timeseries.keys():
            y_label = var + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = var
    elif "paleoData_proxyObservationType" in timeseries.keys():
        if type(timeseries["paleoData_proxyObservationType"]) is list:
            var = timeseries["paleoData_proxyObservationType"][0]
        else: 
            var = timeseries["paleoData_proxyObservationType"] 
        if "paleoData_units" in timeseries.keys():
            y_label = var + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = var
    else:
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_variableName"] + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = timeseries["paleoData_variableName"]
    
    # make the plot
    fig = Plot.plot(x,y,markersize=markersize,marker=marker,x_label=x_label,\
              y_label=y_label, title=title, figsize = figsize, ax=None)

    #Save the figure if asked
    if saveFig == True:
        name = 'plot_timeseries_'+timeseries["dataSetName"]+\
            "_"+y_label
        LipdUtils.saveFigure(name,format,dir)
    else:
        plt.show()

    return fig

def plotEnsTs(timeseries = "", lipd ="", ensTableName = None, ens = None, \
              color = "default", \
              alpha = 0.005, figsize = [10,4], \
              saveFig = False, dir = "",\
              format="eps"):
    """ Plot timeseries on various ensemble ages
    
    Args:
        timeseries (dict): LiPD timeseries object. By default, will prompt for one
        lipd (dict): The LiPD dictionary. MUST be provided if timeseries is set.
        ensTableName (str): The name of the ensemble table, if known.
        ens (int): Number of ensembles to plot. By default, will plot either the 
            number of ensembles stored in the chronensembleTable or 500 of them,
            whichever is lower
        color (str): The line color. If None, uses the default color palette
        alpha (float): Transparency setting for each line. Default is 0.005.
        figsize (list): the size for the figure
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.

    Returns:
        The figure
        
    """
    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list)
    elif not lipd and type(timeseries) is dict:
        sys.exit("LiPD file should be provided when timeseries is set.")
       
    # Get the csv files
    if not lipd: 
        if 'archiveType' in lipd_dict.keys():
            csv_dict = lpd.getCsv[lipd_dict]
        else:    
            if timeseries["dataSetName"] not in lipd_dict.keys():
                # Print out warnings
                print("Dataset name and LiPD file name don't match!")
                print("Select your LiPD file from the list")
                # Get the name of the LiPD files stored in memory
                keylist = []
                for val in lipd_dict.keys():
                    keylist.append(val)
                #Print them out for the users
                for idx, val in enumerate(keylist):
                    print(idx,": ",val)
                # Ask the user to pick one    
                sel = int(input("Enter the number of the LiPD file: "))
                lipd_name = keylist[sel]
                # Grab the csv list
                csv_dict = lpd.getCsv(lipd_dict[lipd_name])
                
            else: #Just grab the csv directly
                csv_dict = lpd.getCsv(lipd_dict[timeseries["dataSetName"]])
    else:
        if 'archiveType' in lipd.keys():
            csv_dict = lpd.getCsv(lipd)
        else:            
            if timeseries["dataSetName"] not in lipd.keys():
                # Print out warnings
                print("Dataset name and LiPD file name don't match!")
                print("Select your LiPD file from the list")
                # Get the name of the LiPD files stored in memory
                keylist = []
                for val in lipd.keys():
                    keylist.append(val)
                #Print them out for the users
                for idx, val in enumerate(keylist):
                    print(idx,": ",val)
                # Ask the user to pick one    
                sel = int(input("Enter the number of the LiPD file: "))
                lipd_name = keylist[sel]
                # Grab the csv list
                csv_dict = lpd.getCsv(lipd[lipd_name])
            else: #Just grab the csv directly
                csv_dict = lpd.getCsv(lipd[timeseries["dataSetName"]]) 
                
    # Get the ensemble tables
    if not ensTableName:
        chronEnsembleTables, paleoEnsembleTables = LipdUtils.isEnsemble(csv_dict)
    elif ensTableName not in csv_dict.keys():
        print("The name of the table you entered doesn't exist, selecting...")
    else:
        chronEnsembleTables = ensTableName

    # Check the number of Chron Tables 
    if len(chronEnsembleTables) == 0:
        sys.exit("No chronEnsembleTable available")
    elif len(chronEnsembleTables) > 1: 
        print("More than one ensemble table available.")
        for idx, val in enumerate(chronEnsembleTables):
            print(idx,": ",val)
        sel = int(input("Enter the number of the table you'd like to use: "))
        ensemble_dict = csv_dict[chronEnsembleTables[sel]]
    else:
        ensemble_dict = csv_dict[chronEnsembleTables[0]]
    
    # Get depth and values    
    depth, ensembleValues = LipdUtils.getEnsembleValues(ensemble_dict)
    
    # Get the paleoData values
    ys = np.array(timeseries["paleoData_values"], dtype = 'float64')
    ds = np.array(timeseries["depth"], dtype = 'float64')
    # Remove NaNs
    ys_tmp = np.copy(ys)
    ys = ys[~np.isnan(ys_tmp)]
    ds = ds[~np.isnan(ys_tmp)]
    
    # Bring the ensemble values to common depth
    ensembleValuestoPaleo = LipdUtils.mapAgeEnsembleToPaleoData(ensembleValues, depth, ds)

    # Get plot information
    title = timeseries['dataSetName']
    x_label = "Age"
    # y_label
    if "paleoData_inferredVariableType" in timeseries.keys():
        #This if loop is needed because some files appear to have two
        #different inferredVariableType/proxyObservationType, which
        #should not be possible in the ontology. Just build some checks
        # in the system
        if type(timeseries["paleoData_inferredVariableType"]) is list:
            var = timeseries["paleoData_inferredVariableType"][0]
        else:
            var = timeseries["paleoData_inferredVariableType"]
        if "paleoData_units" in timeseries.keys():
            y_label = var + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = var
    elif "paleoData_proxyObservationType" in timeseries.keys():
        if type(timeseries["paleoData_proxyObservationType"]) is list:
            var = timeseries["paleoData_proxyObservationType"][0]
        else: 
            var = timeseries["paleoData_proxyObservationType"] 
        if "paleoData_units" in timeseries.keys():
            y_label = var + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = var
    else:
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_variableName"] + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = timeseries["paleoData_variableName"]
     
    # Get the color
    if color == "default":
        archiveType = LipdUtils.LipdToOntology(timeseries['archiveType']).lower()
        if archiveType not in plot_default.keys():
            archiveType = 'other'
        color = plot_default[archiveType][0]
    
    # Make the plot
    fig = Plot.plotEns(ensembleValuestoPaleo, ys, ens = ens, color = color,\
                       alpha = alpha, x_label = x_label, y_label = y_label,\
                       title = title, figsize = figsize, ax = None)

    # Save the figure if asked
    if saveFig == True:
        name = 'plot_ens_timeseries_'+timeseries["dataSetName"]+\
            "_"+y_label
        LipdUtils.saveFigure(name,format,dir)
    else:
        plt.show()

    return fig

def histTs(timeseries = "", bins = None, hist = True, \
             kde = True, rug = False, fit = None, hist_kws = {"label":"Histogram"},\
             kde_kws = {"label":"KDE fit"}, rug_kws = {"label":"Rug"}, \
             fit_kws = {"label":"Fit"}, color = "default", vertical = False, \
             norm_hist = True, figsize = [5,5],\
             saveFig = False, format ="eps",\
             dir = ""):
    """ Plot a univariate distribution of the PaleoData values

    This function is based on the seaborn displot function, which is
    itself a combination of the matplotlib hist function with the
    seaborn kdeplot() and rugplot() functions. It can also fit
    scipy.stats distributions and plot the estimated PDF over the data.

    Args:
        timeseries: A timeseries. By default, will prompt the user for one.
        bins (int): Specification of hist bins following matplotlib(hist),
            or None to use Freedman-Diaconis rule
        hist (bool): Whether to plot a (normed) histogram
        kde (bool): Whether to plot a gaussian kernel density estimate
        rug (bool): Whether to draw a rugplot on the support axis
        fit: Random variable object. An object with fit method, returning
            a tuple that can be passed to a pdf method of positional
            arguments following a grid of values to evaluate the pdf on.
        {hist, kde, rug, fit}_kws: Dictionaries. Keyword arguments for
            underlying plotting functions. If modifying the dictionary, make
            sure the labels "hist", "kde", "rug" and "fit" are still passed.
        color (str): matplotlib color. Color to plot everything but the
            fitted curve in. Default is to use the default paletter for each
            archive type.
        vertical (bool): if True, oberved values are on y-axis.
        norm_hist (bool): If True (default), the histrogram height shows
            a density rather than a count. This is implied if a KDE or
            fitted density is plotted
        figsize (list): the size for the figure
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.

    Returns
        fig - The figure

    """
    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list) 

    # Get the values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')

    # Remove NaNs
    index = np.where(~np.isnan(y))[0]
    y = y[index]

    # Get the y_label
    if "paleoData_inferredVariableType" in timeseries.keys():
        #This if loop is needed because some files appear to have two
        #different inferredVariableType/proxyObservationType, which
        #should not be possible in the ontology. Just build some checks
        # in the system
        if type(timeseries["paleoData_inferredVariableType"]) is list:
            var = timeseries["paleoData_inferredVariableType"][0]
        else:
            var = timeseries["paleoData_inferredVariableType"]
        if "paleoData_units" in timeseries.keys():
            y_label = var + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = var
    elif "paleoData_proxyObservationType" in timeseries.keys():
        if type(timeseries["paleoData_proxyObservationType"]) is list:
            var = timeseries["paleoData_proxyObservationType"][0]
        else: 
            var = timeseries["paleoData_proxyObservationType"] 
        if "paleoData_units" in timeseries.keys():
            y_label = var + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = var
    else:
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_variableName"] + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = timeseries["paleoData_variableName"]

    # Grab the color
    if color == 'default':
        archiveType = LipdUtils.LipdToOntology(timeseries['archiveType']).lower()
        if archiveType not in plot_default.keys():
            archiveType = 'other'
        color = plot_default[archiveType][0]
    
    # Make this histogram
    fig = Plot.plot_hist(y, bins = bins, hist = hist, \
        kde = kde, rug = rug, fit = fit, hist_kws = hist_kws,\
        kde_kws = kde_kws, rug_kws = rug_kws, \
        fit_kws = fit_kws, color = color, vertical = vertical, \
        norm_hist = norm_hist, label = y_label, figsize = figsize, ax=None)

    #Save the figure if asked
    if saveFig == True:
        name = 'plot_timeseries_'+timeseries["dataSetName"]+\
            "_"+y_label
        LipdUtils.saveFigure(name,format,dir)
    else:
        plt.show()

    return fig

"""
SummaryPlots
"""

def summaryTs(timeseries = "", x_axis = "", saveFig = False, dir = "",
               format ="eps"):
    """Basic summary plot

    Plots the following information: the time series, a histogram of
    the PaleoData_values, location map, spectral density using the wwz 
    method, and metadata about the record.

    Args:
        timeseries: a timeseries object. By default, will prompt for one
        x_axis (str): The representation against which to plot the paleo-data.
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
        The figure

    """

    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list) 

    # get the necessary metadata
    metadata = SummaryPlots.getMetadata(timeseries)
    
    # get the information about the timeseries
    x,y,archiveType,x_label,y_label = SummaryPlots.TsData(timeseries,
                                                          x_axis=x_axis)
    
    # Clean up
    y,x = Timeseries.clean_ts(y,x)
        
    # Make the figure
    fig = plt.figure(figsize=(11,8))
    plt.style.use("ggplot")
    gs = gridspec.GridSpec(2, 5)
    gs.update(left=0, right =1.1)
    
    # Plot the timeseries
    ax1 = fig.add_subplot(gs[0,:-3])
    archiveType = LipdUtils.LipdToOntology(timeseries['archiveType']).lower()
    if archiveType not in plot_default.keys():
        archiveType = 'other'
    marker = plot_default[archiveType]
    markersize = 50
    
    Plot.plot(x,y,markersize=markersize, marker = marker, x_label=x_label,\
              y_label=y_label, title = timeseries['dataSetName'], ax=ax1)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    
    # Plot the histogram and kernel density estimates
    ax2 = fig.add_subplot(gs[0,2])
    sns.distplot(y, vertical = True, color = marker[0], \
                hist_kws = {"label":"Histogram"},
                kde_kws = {"label":"KDE fit"})
    plt.xlabel('PDF')
    ax2.set_ylim([ymin,ymax])
    ax2.set_yticklabels([])
    
    # Plot the Map
    lat = timeseries["geo_meanLat"]
    lon = timeseries["geo_meanLon"]
    
    ax3 = fig.add_subplot(gs[1,0])
    map = Basemap(projection='ortho', lon_0=lon, lat_0=lat)
    map.drawcoastlines()
    map.shadedrelief(scale=0.5)
    map.drawcountries()
    X,Y = map(lon,lat)
    map.scatter(X,Y,
               s = 150,
               color = marker[0],
               marker = marker[1])
    
    # Spectral analysis
    
    if not 'age' in timeseries.keys() and not 'year' in timeseries.keys():
        print("No age or year information available, skipping spectral analysis")
    else:
        ax4 = fig.add_subplot(gs[1,2])    
        if 'depth' in x_label.lower():
            if 'age' in timeseries.keys() and 'year' in timeseries.keys():
                print("Both age and year information are available.")
                x_axis = input("Which one would you like to use? ")
                while x_axis != "year" and x_axis != "age":
                    x_axis = input("Only enter year or age: ")
            elif 'age' in timeseries.keys():
                x_axis = 'age'
            elif 'year' in timeseries.keys():
                x_axis = 'year'
            y = np.array(timeseries['paleoData_values'], dtype = 'float64')
            x, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)
            y,x = Timeseries.clean_ts(y,x)
                   
    # Perform the analysis
        default = {'tau':None,
                           'freqs': None,
                           'c':1e-3,
                           'nproc':8,
                           'nMC':200,
                           'detrend':'no',
                           'params' : ["default",4,0,1],
                           'gaussianize': False,
                           'standardize':True,
                           'Neff':3,
                           'anti_alias':False,
                           'avgs':1,
                           'method':'Kirchner_f2py',
                           }
        psd, freqs, psd_ar1_q95, psd_ar1 = Spectral.wwz_psd(y,x,**default)
        
        # Make the plot
        ax4.plot(1/freqs, psd, linewidth=1,  label='PSD', color = marker[0])
        ax4.plot(1/freqs, psd_ar1_q95, linewidth=1,  label='AR1 95%', color=sns.xkcd_rgb["pale red"])
        plt.ylabel('Spectral Density')
        plt.xlabel('Period ('+\
                    x_label[x_label.find("(")+1:x_label.find(")")][0:x_label.find(" ")-1]+')')
        
        plt.xscale('log', nonposy='clip')
        plt.yscale('log', nonposy='clip')
        
        ax4.set_aspect('equal')
        
        plt.gca().invert_xaxis()
        plt.legend()    
        
    #Add the metadata
    textstr = "archiveType: " + metadata["archiveType"]+"\n"+"\n"+\
              "Authors: " + metadata["authors"]+"\n"+"\n"+\
              "Year: " + metadata["Year"]+"\n"+"\n"+\
              "DOI: " + metadata["DOI"]+"\n"+"\n"+\
              "Variable: " + metadata["Variable"]+"\n"+"\n"+\
              "units: " + metadata["units"]+"\n"+"\n"+\
              "Climate Interpretation: " +"\n"+\
              "    Climate Variable: " + metadata["Climate_Variable"] +"\n"+\
              "    Detail: " + metadata["Detail"]+"\n"+\
              "    Seasonality: " + metadata["Seasonality"]+"\n"+\
              "    Direction: " + metadata["Interpretation_Direction"]+"\n \n"+\
              "Calibration: \n" + \
              "    Equation: " + metadata["Calibration_equation"] + "\n" +\
              "    Notes: " + metadata["Calibration_notes"]
    plt.figtext(0.7, 0.4, textstr, fontsize = 12)

    #Save the figure if asked
    if saveFig == True:
        name = 'plot_timeseries_'+timeseries["dataSetName"]+\
            "_"+y_label
        LipdUtils.saveFigure(name,format,dir)
    else:
        plt.show()

    return fig

"""
Statistics
"""

def statsTs(timeseries=""):
    """ Calculate simple statistics of a timeseries

    Args:
        timeseries: sytem will prompt for one if not given

    Returns:
        the mean, median, min, max, standard deviation and the
        inter-quartile range (IQR) of a timeseries.

    Examples:
        >>> mean, median, min_, max_, std, IQR = pyleo.statsTs(timeseries)

    """
    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list) 

    # get the values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')

    mean, median, min_, max_, std, IQR = Stats.simpleStats(y)

    return mean, median, min_, max_, std, IQR

def corrSigTs(timeseries1 = "", timeseries2 = "", x_axis = "", \
                 interp_step = "", start = "", end = "", nsim = 1000, \
                 method = 'isospectral', alpha = 0.5):
    """ Estimates the significance of correlations between non IID timeseries.

        Function written by. F. Zhu.

        Args:
            timeseries1, timeseries2: timeseries object. Default is blank.
            x-axis (str): The representation against which to express the
                paleo-data. Options are "age", "year", and "depth".
                Default is to let the system choose if only one available
                or prompt the user.
            interp_step (float): the step size. By default, will prompt the user.
            start (float): Start time/age/depth. Default is the maximum of
                the minima of the two timeseries
            end (float): End time/age/depth. Default is the minimum of the
                maxima of the two timeseries
            nsim (int): the number of simulations. Default is 1000
            method (str): method use to estimate the correlation and significance.
                Available methods include:
                    - 'ttest': T-test where the degrees of freedom are corrected for
                    the effect of serial correlation \n
                    - 'isopersistant': AR(1) modeling of the two timeseries \n
                    - 'isospectral' (default): phase randomization of original
                    inputs.
                The T-test is parametric test, hence cheap but usually wrong
                except in idyllic circumstances.
                The others are non-parametric, but their computational
                requirements scales with nsim.
            alpha (float): significance level for critical value estimation. Default is 0.05

        Returns:
            r (float) - correlation between the two timeseries \n
            sig (bool) -  Returns True if significant, False otherwise \n
            p (real) - the p-value

    """
    if not timeseries1:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries1 = LipdUtils.getTs(ts_list)
        
    if not timeseries2:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries2 = LipdUtils.getTs(ts_list)    

    # Get the first time and paleoData values
    y1 = np.array(timeseries1['paleoData_values'], dtype = 'float64')
    x1, label1 = LipdUtils.checkXaxis(timeseries1, x_axis=x_axis)

    # Get the second one
    y2 = np.array(timeseries2['paleoData_values'], dtype = 'float64')
    x2, label2 = LipdUtils.checkXaxis(timeseries2, x_axis=x_axis)

    # Remove NaNs and ordered
    y1,x1 = Timeseries.clean_ts(y1,x1)
    y2,x2 = Timeseries.clean_ts(y2,x2)
    
    # Make sure that the series have the same units:
    units1 = timeseries1[label1+'Units']
    units2 = timeseries2[label2+'Units'] 
    
    if units2!=units1:
        print('Warning: The two timeseries are on different time units!')
        print('The units of timeseries1 are '+units1)
        print('The units of timeseries2 are '+units2)
        answer = input('Enter an equation to convert the units of x2 onto x1.'+
                       " The equation should be written in Python format"+
                       " (e.g., a*x2+b or 1950-a*x2) or press enter to abort: ")
        
        if not answer:
            sys.exit("Aborted by User")
        elif "x2" not in answer: 
            answer = input("The form must be a valid python expression and contain x2!"+
                           "Enter a valid expression: ")
            while "x2" not in answer:
                answer = input("The form must be a valid python expression and contain x2!"+
                           "Enter a valid expression: ")        
        else: x2 = eval(answer)  
          

    #Check that the two timeseries have the same lenght and if not interpolate
    if len(y1) != len(y2):
        print("The two series don't have the same length. Interpolating ...")
        xi, interp_values1, interp_values2 = Timeseries.onCommonAxis(x1,y1,x2,y2,
                                                                     interp_step = interp_step,
                                                                     start =start,
                                                                     end=end)
    elif min(x1) != min(x2) and max(x1) != max(x2):
        print("The two series don't have the same length. Interpolating ...")
        xi, interp_values1, interp_values2 = Timeseries.onCommonAxis(x1,y1,x2,y2,
                                                                     interp_step = interp_step,
                                                                     start =start,
                                                                     end=end)
    else:
        #xi = x1
        interp_values1 = y1
        interp_values2 = y2

    #Make sure that these vectors are not empty, otherwise return an error
    if np.size(interp_values1) == 0 or np.size(interp_values2) == 0:
        sys.exit("No common time period between the two time series.")

    r, sig, p = Stats.corrsig(interp_values1,interp_values2,nsim=nsim,
                                 method=method,alpha=alpha)

    return r, sig, p


"""
Timeseries manipulation
"""

def binTs(timeseries="", x_axis = "", bin_size = "", start = "", end = ""):
    """Bin the paleoData values of the timeseries

    Args:
        timeseries. By default, will prompt the user for one.
        x-axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the
            system  choose if only one available or prompt the user.
        bin_size (float): the size of the bins to be used. By default,
            will prompt for one
        start (float): Start time/age/depth. Default is the minimum
        end (float): End time/age/depth. Default is the maximum

    Returns:
        binned_values- the binned output,\n
        bins-  the bins (centered on the median, i.e. the 100-200 bin is 150),\n
        n-  number of data points in each bin,\n
        error- the standard error on the mean in each bin\n

    """
    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list)

    # Get the values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')
    x, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)

    #remove nans
    y,x = Timeseries.clean_ts(y,x)
    
    #Bin the timeseries:
    bins, binned_values, n, error = Timeseries.bin(x,y, bin_size = bin_size,\
                                                   start = start, end = end)

    return bins, binned_values, n, error

def interpTs(timeseries="", x_axis = "", interp_step = "", start = "", end = ""):
    """Simple linear interpolation

    Simple linear interpolation of the data using the numpy.interp method

    Args:
        timeseries. Default is blank, will prompt for it
        x-axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the
            system choose if only one available or prompt the user.
        interp_step (float): the step size. By default, will prompt the user.
        start (float): Start year/age/depth. Default is the minimum
        end (float): End year/age/depth. Default is the maximum

    Returns:
        interp_age - the interpolated age/year/depth according to the end/start
        and time step, \n
        interp_values - the interpolated values

    """
    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list)

    # Get the values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')
    x, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)

    #remove nans
    y,x = Timeseries.clean_ts(y,x)

    #Interpolate the timeseries
    interp_age, interp_values = Timeseries.interp(x,y,interp_step = interp_step,\
                                                  start= start, end=end)

    return interp_age, interp_values

def standardizeTs(timeseries = "", scale = 1, ddof = 0, eps = 1e-3):
    """ Centers and normalizes the paleoData values of a  given time series.

    Constant or nearly constant time series not rescaled.

    Args:
        x (array): vector of (real) numbers as a time series, NaNs allowed
        scale (real): a scale factor used to scale a record to a match a given variance
        axis (int or None): axis along which to operate, if None, compute over the whole array
        ddof (int): degress of freedom correction in the calculation of the standard deviation
        eps (real): a threshold to determine if the standard deviation is too close to zero

    Returns:
        - z (array): the standardized time series (z-score), Z = (X - E[X])/std(X)*scale, NaNs allowed \n
        - mu (real): the mean of the original time series, E[X] \n
        - sig (real): the standard deviation of the original time series, std[X] \n

    References:
        1. Tapio Schneider's MATLAB code: http://www.clidyn.ethz.ch/imputation/standardize.m
        2. The zscore function in SciPy: https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py

    @author: fzhu
    """
    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list)

    # get the values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')

    # Remove NaNs
    y_temp = np.copy(y)
    y = y[~np.isnan(y_temp)]

    #Standardize
    z, mu, sig = Timeseries.standardize(y,scale=1,axis=None,ddof=0,eps=1e-3)

    return z, mu, sig

def segmentTs(timeseries = "", factor = 2):
    """Divides a time series into several segments using a gap detection algorithm
    
    Gap detection rule: If the time interval between some two data points is
    larger than some factor times the mean resolution of the timeseries, then
    a brak point is applied and the timseries is divided. 
    
    Args:
        timeseries: a LiPD timeseries object
        factor (float): factor to adjust the threshold. threshold = factor*dt_mean.
            Default is 2.
    
    Returns:
        seg_y (list) - a list of several segments with potentially different length
        seg_t (list) - A list of the time values for each y segment. 
        n_segs (int) - the number of segments
        
    
    """
    
    if not timeseries:
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list)
        
    # Get the values
    # Raise an error if age or year not in the keys
    if not 'age' in timeseries.keys() and not 'year' in timeseries.keys():
        sys.exit("No time information available")
    elif 'age' in timeseries.keys() and 'year' in timeseries.keys():
        print("Both age and year information are available.")
        x_axis = input("Which one would you like to use? ")
        while x_axis != "year" and x_axis != "age":
            x_axis = input("Only enter year or age: ")
    elif 'age' in timeseries.keys():
        x_axis = 'age'
    elif 'year' in timeseries.keys():
        x_axis = 'year'        
    
    # Get the values
    ys = np.array(timeseries['paleoData_values'], dtype = 'float64') 
    ts, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)
    
    # remove NaNs
    ys,ts = Timeseries.clean_ts(ys,ts)   

    #segment the timeseries
    seg_y, seg_t, n_segs = Timeseries.ts2segments(ys, ts, factor)

    return seg_y, seg_t, n_segs        

#"""
# Spectral Analysis
#"""

def wwzTs(timeseries = "", lim = None, wwz = False, psd = True, wwz_default = True,
          psd_default = True, wwaplot_default = True, psdplot_default = True,
          fig = True, saveFig = False, dir = "", format = "eps"):
    """Weigthed wavelet Z-transform analysis
    
    Wavelet analysis for unevenly spaced data adapted from Foster et al. (1996)
    
    Args:
        timeseries (dict): A LiPD timeseries object (Optional, will prompt for one.)
        lim (list): Truncate the timeseries between min/max time (e.g., [0,10000])
        wwz (bool): If True, will perform wavelet analysis
        psd (bool): If True, will inform the power spectral density of the timeseries
        wwz_default: If True, will use the following default parameters:
            
            wwz_default = {'tau':None,
                           'freqs':None,
                           'c':1/(8*np.pi**2),
                           'Neff':3,
                           'Neff_coi':3,
                           'nMC':200,
                           'nproc':8,
                           'detrend':'no',
                           'params' : ["default",4,0,1],
                           'gaussianize': False,
                           'standardize':True,
                           'method':'Kirchner_f2py',
                           'bc_mode':'reflect',
                           'reflect_type':'odd',
                           'len_bd':0}
                
            Modify the values for specific keys to change the default behavior.
                
        psd_default: If True, will use the following default parameters:
            
            psd_default = {'tau':None,
                       'freqs': None,
                       'c':1e-3,
                       'nproc':8,
                       'nMC':200,
                       'detrend':'no',
                       'params' : ["default",4,0,1],
                       'gaussianize': False,
                       'standardize':True,
                       'Neff':3,
                       'anti_alias':False,
                       'avgs':1,
                       'method':'Kirchner_f2py',
                       }
            
            Modify the values for specific keys to change the default behavior.
            
        wwaplot_default: If True, will use the following default parameters:
            
            wwaplot_default={'AR1_q':AR1_q,
                                 'coi':coi,
                                 'levels':None,
                                 'tick_range':None,
                                 'yticks':None,
                                 'yticks_label': None,
                                 'ylim':None,
                                 'xticks':None,
                                 'xlabels':None,
                                 'figsize':[20,8],
                                 'clr_map':'OrRd',
                                 'cbar_drawedges':False,
                                 'cone_alpha':0.5,
                                 'plot_signif':True,
                                 'signif_style':'contour',
                                 'plot_cone':True,
                                 'title':None,
                                 'ax':None,
                                 'xlabel': label.upper()[0]+label[1:]+'('+s+')',
                                 'ylabel': 'Period ('+ageunits+')',
                                 'cbar_orientation':'vertical',
                                 'cbar_pad':0.05,
                                 'cbar_frac':0.15,
                                 'cbar_labelsize':None}
            
            Modify the values for specific keys to change the default behavior.
        psdplot_default: If True, will use the following default parameters:
            
            psdplot_default={'lmstyle':'-',
                                 'linewidth':None,
                                 'color': sns.xkcd_rgb["denim blue"],
                                 'ar1_lmstyle':'-',
                                 'ar1_linewidth':1,
                                 'period_ticks':None,
                                 'period_tickslabel':None,
                                 'psd_lim':None,
                                 'period_lim':None,
                                 'figsize':[20,8],
                                 'label':'PSD',
                                 'plot_ar1':True,
                                 'psd_ar1_q95':psd_ar1_q95,
                                 'title': None,
                                 'psd_ar1_color':sns.xkcd_rgb["pale red"],
                                 'ax':None,
                                 'vertical':False,
                                 'plot_gridlines':True,
                                 'period_label':'Period ('+ageunits+')',
                                 'psd_label':'Spectral Density',
                                 'zorder' : None}    
            
            Modify the values for specific keys to change the default behavior.
            
        fig (bool): If True, plots the figure
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.
        
    Returns:
        dict_out (dict): A dictionary of outputs. 
            
            For wwz: 
            
            - wwa (array): The weights wavelet amplitude 
        
            - AR1_q (array): AR1 simulations 
        
            - coi (array): cone of influence 
        
            - freqs (array): vector for frequencies 
        
            - tau (array): the evenly-spaced time points, namely the time 
            shift for wavelet analysis. 
        
            - Neffs (array): The matrix of effective number of points in the
            time-scale coordinates.
        
            - coeff (array): The wavelet transform coefficients
        
            For psd: 
            
            - psd (array): power spectral density 
        
            - freqs (array): vector of frequency 
        
            - psd_ar1_q95 (array): the 95% quantile of the psds of AR1 processes 
        
        fig: The figure
         
        References:
            Foster, G. (1996). Wavelets for period analysis of unevenly 
            sampled time series. The Astronomical Journal, 112(4), 1709-1729.
        
        Examples:
            To run both wwz and psd: \n
            
            >>> dict_out, fig = pyleoclim.wwzTs(wwz=True)
            
            Note: This will return a single figure with wwa and psd \n
            
            To change a default behavior:\n
            
            >>> dict_out, fig = pyleoclim.wwzTs(psd_default = {'nMC':1000}) 
           
    """
    
    # Make sure there is something to compute
    if wwz is False and psd is False:
        sys.error("Set 'wwz' and/or 'psd' to True")
    
    # Get a timeseries
    if not timeseries: 
        if not 'ts_list' in globals():
            fetchTs()
        timeseries = LipdUtils.getTs(ts_list)
    
    # Raise an error if age or year not in the keys
    if not 'age' in timeseries.keys() and not 'year' in timeseries.keys():
        sys.exit("No time information available")
    elif 'age' in timeseries.keys() and 'year' in timeseries.keys():
        print("Both age and year information are available.")
        x_axis = input("Which one would you like to use? ")
        while x_axis != "year" and x_axis != "age":
            x_axis = input("Only enter year or age: ")
    elif 'age' in timeseries.keys():
        x_axis = 'age'
    elif 'year' in timeseries.keys():
        x_axis = 'year'

    # Set the defaults
    #Make sure the default have the proper type 
    if psd_default is not True and type(psd_default) is not dict:
        sys.exit('The default for the psd calculation should either be provided'+
                 ' as a dictionary are set to True')
    if psdplot_default is not True and type(psdplot_default) is not dict:
        sys.exit('The default for the psd figure should either be provided'+
                 ' as a dictionary are set to True')  
    if wwz_default is not True and type(wwz_default) is not dict:
        sys.exit('The default for the wwz calculation should either be provided'+
                 ' as a dictionary are set to True')
    if wwaplot_default is not True and type(wwaplot_default) is not dict:
        sys.exit('The default for the wwa figure should either be provided'+
                 ' as a dictionary are set to True')

    # Get the values
    ys = np.array(timeseries['paleoData_values'], dtype = 'float64') 
    ts, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)
    
    # remove NaNs
    ys,ts = Timeseries.clean_ts(ys,ts) 
    
    # Truncate the timeseries if asked
    if lim is not None:
        idx_low = np.where(ts>=lim[0])[0][0]
        idx_high = np.where(ts<=lim[1])[0][-1]
        ts = ts[idx_low:idx_high]
        ys = ys[idx_low:idx_high]
    
    #Get the time units
    s = timeseries[label+"Units"]
    ageunits = s[0:s.find(" ")]  
    
    # Perform the calculations
    if psd is True and wwz is False: # PSD only
        
        if type(psd_default) is dict:
            dict_in = psd_default
            
            psd_default = {'tau':None,
                       'freqs': None,
                       'c':1e-3,
                       'nproc':8,
                       'nMC':200,
                       'detrend':'no',
                       'params' : ["default",4,0,1],
                       'gaussianize': False,
                       'standardize':True,
                       'Neff':3,
                       'anti_alias':False,
                       'avgs':1,
                       'method':'Kirchner_f2py',
                       }
            
            for key, value in dict_in.items():
                if key in psd_default.keys():
                    psd_default[key] = value
        
        else:
          psd_default = {'tau':None,
                       'freqs': None,
                       'c':1e-3,
                       'nproc':8,
                       'nMC':200,
                       'detrend':'no',
                       'params' : ["default",4,0,1],
                       'gaussianize': False,
                       'standardize':True,
                       'Neff':3,
                       'anti_alias':False,
                       'avgs':1,
                       'method':'Kirchner_f2py',
                       }
            
        # Perform calculation
        psd, freqs, psd_ar1_q95, psd_ar1 = Spectral.wwz_psd(ys, ts, **psd_default)
        
        # Wrap up the output dictionary
        dict_out = {'psd':psd,
               'freqs':freqs,
               'psd_ar1_q95':psd_ar1_q95,
               'psd_ar1':psd_ar1}
        
        # Plot if asked
        if fig is True:

            if type(psdplot_default) is dict:
                dict_in = psdplot_default
                
                psdplot_default={'lmstyle':'-',
                                 'linewidth':None,
                                 'color': sns.xkcd_rgb["denim blue"],
                                 'ar1_lmstyle':'-',
                                 'ar1_linewidth':1,
                                 'period_ticks':None,
                                 'period_tickslabel':None,
                                 'psd_lim':None,
                                 'period_lim':None,
                                 'figsize':[20,8],
                                 'label':'PSD',
                                 'plot_ar1':True,
                                 'psd_ar1_q95':psd_ar1_q95,
                                 'title': None,
                                 'psd_ar1_color':sns.xkcd_rgb["pale red"],
                                 'ax':None,
                                 'vertical':False,
                                 'plot_gridlines':True,
                                 'period_label':'Period ('+ageunits+')',
                                 'psd_label':'Spectral Density',
                                 'zorder' : None}        
                                
                for key, value in dict_in.items():
                    if key in psdplot_default.keys():
                        psdplot_default[key] = value
                        
            else:
                   
               psdplot_default={'lmstyle':'-',
                                 'linewidth':None,
                                 'color': sns.xkcd_rgb["denim blue"],
                                 'ar1_lmstyle':'-',
                                 'ar1_linewidth':1,
                                 'period_ticks':None,
                                 'period_tickslabel':None,
                                 'psd_lim':None,
                                 'period_lim':None,
                                 'figsize':[20,8],
                                 'label':'PSD',
                                 'plot_ar1':True,
                                 'psd_ar1_q95':psd_ar1_q95,
                                 'title': None,
                                 'psd_ar1_color':sns.xkcd_rgb["pale red"],
                                 'ax':None,
                                 'vertical':False,
                                 'plot_gridlines':True,
                                 'period_label':'Period ('+ageunits+')',
                                 'psd_label':'Spectral Density',
                                 'zorder' : None}                 
                
            fig = Spectral.plot_psd(psd,freqs,**psdplot_default)
            
            if saveFig is True:
                LipdUtils.saveFigure(timeseries['dataSetName']+'_PSDplot',format,dir)
            else:
                plt.show()               
            
        else:
            fig = None
             
    elif psd is False and wwz is True: #WWZ only   
        # Set default 
        if type(wwz_default) is dict:
            dict_in = wwz_default
            
            wwz_default = {'tau':None,
                           'freqs':None,
                           'c':1/(8*np.pi**2),
                           'Neff':3,
                           'Neff_coi':3,
                           'nMC':200,
                           'nproc':8,
                           'detrend':'no',
                           'params' : ["default",4,0,1],
                           'gaussianize': False,
                           'standardize':True,
                           'method':'Kirchner_f2py',
                           'bc_mode':'reflect',
                           'reflect_type':'odd',
                           'len_bd':0}
            
            for key,value in dict_in.items():
                if key in wwz_default.keys():
                    wwz_default[key]=value
        
        else:  
            
            wwz_default = {'tau':None,
                           'freqs':None,
                           'c':1/(8*np.pi**2),
                           'Neff':3,
                           'Neff_coi':3,
                           'nMC':200,
                           'nproc':8,
                           'detrend':'no',
                           'params' : ["default",4,0,1],
                           'gaussianize': False,
                           'standardize':True,
                           'method':'Kirchner_f2py',
                           'bc_mode':'reflect',
                           'reflect_type':'odd',
                           'len_bd':0}
        
        #Perform the calculation
        wwa, phase, AR1_q, coi, freqs, tau, Neffs, coeff = Spectral.wwz(ys,ts, **wwz_default)
        
        #Wrap up the output dictionary
        dict_out = {'wwa':wwa,
                    'phase':phase,
                    'AR1_q':AR1_q,
                    'coi':coi,
                    'freqs':freqs,
                    'tau':tau,
                    'Neffs':Neffs,
                    'coeff':coeff}
        
        #PLot if asked
        if fig is True:
            # Set the plot default
            if type(wwaplot_default) is dict:
                dict_in = wwaplot_default
                wwaplot_default={'AR1_q':AR1_q,
                                 'coi':coi,
                                 'levels':None,
                                 'tick_range':None,
                                 'yticks':None,
                                 'yticks_label': None,
                                 'ylim':None,
                                 'xticks':None,
                                 'xlabels':None,
                                 'figsize':[20,8],
                                 'clr_map':'OrRd',
                                 'cbar_drawedges':False,
                                 'cone_alpha':0.5,
                                 'plot_signif':True,
                                 'signif_style':'contour',
                                 'plot_cone':True,
                                 'title':None,
                                 'ax':None,
                                 'xlabel': label.upper()[0]+label[1:]+'('+s+')',
                                 'ylabel': 'Period ('+ageunits+')',
                                 'cbar_orientation':'vertical',
                                 'cbar_pad':0.05,
                                 'cbar_frac':0.15,
                                 'cbar_labelsize':None}
                for key, value in dict_in.items():
                    if key in wwaplot_default.keys():
                        wwaplot_default[key] = value
            
            else:
                wwaplot_default={'AR1_q':AR1_q,
                                 'coi':coi,
                                 'levels':None,
                                 'tick_range':None,
                                 'yticks':None,
                                 'yticks_label': None,
                                 'ylim':None,
                                 'xticks':None,
                                 'xlabels':None,
                                 'figsize':[20,8],
                                 'clr_map':'OrRd',
                                 'cbar_drawedges':False,
                                 'cone_alpha':0.5,
                                 'plot_signif':True,
                                 'signif_style':'contour',
                                 'plot_cone':True,
                                 'title':None,
                                 'ax':None,
                                 'xlabel': label.upper()[0]+label[1:]+'('+s+')',
                                 'ylabel': 'Period ('+ageunits+')',
                                 'cbar_orientation':'vertical',
                                 'cbar_pad':0.05,
                                 'cbar_frac':0.15,
                                 'cbar_labelsize':None}
            
            fig = Spectral.plot_wwa(wwa, freqs, tau, **wwaplot_default)
            
            if saveFig is True:
                LipdUtils.saveFigure(timeseries['dataSetName']+'_PSDplot',format,dir)
            else:
                plt.show()               
            
        else:
            fig = None
    
    elif psd is True and wwz is True: # perform both
    
        # Set the defaults
        
        if type(psd_default) is dict:
            dict_in = psd_default
            
            psd_default = {'tau':None,
                       'freqs': None,
                       'c':1e-3,
                       'nproc':8,
                       'nMC':200,
                       'detrend':'no',
                       'params' : ["default",4,0,1],
                       'gaussianize': False,
                       'standardize':True,
                       'Neff':3,
                       'anti_alias':False,
                       'avgs':1,
                       'method':'Kirchner_f2py',
                       }
            
            for key, value in dict_in.items():
                if key in psd_default.keys():
                    psd_default[key] = value
        
        else:
          psd_default = {'tau':None,
                       'freqs': None,
                       'c':1e-3,
                       'nproc':8,
                       'nMC':200,
                       'detrend':'no',
                       'params' : ["default",4,0,1],
                       'gaussianize': False,
                       'standardize':True,
                       'Neff':3,
                       'anti_alias':False,
                       'avgs':1,
                       'method':'Kirchner_f2py',
                       }
           
        if type(wwz_default) is dict:
            dict_in = wwz_default
            
            wwz_default = {'tau':None,
                           'freqs':None,
                           'c':1/(8*np.pi**2),
                           'Neff':3,
                           'Neff_coi':3,
                           'nMC':200,
                           'nproc':8,
                           'detrend':'no',
                           'params' : ["default",4,0,1],
                           'gaussianize': False,
                           'standardize':True,
                           'method':'Kirchner_f2py',
                           'bc_mode':'reflect',
                           'reflect_type':'odd',
                           'len_bd':0}

            
            for key,value in dict_in.items():
                if key in wwz_default.keys():
                    wwz_default[key]=value
        
        else:  
            
            wwz_default = {'tau':None,
                           'freqs':None,
                           'c':1/(8*np.pi**2),
                           'Neff':3,
                           'Neff_coi':3,
                           'nMC':200,
                           'nproc':8,
                           'detrend':'no',
                           'params' : ["default",4,0,1],
                           'gaussianize': False,
                           'standardize':True,
                           'method':'Kirchner_f2py',
                           'bc_mode':'reflect',
                           'reflect_type':'odd',
                           'len_bd':0}

            
        # Perform the calculations
        psd, freqs, psd_ar1_q95, psd_ar1 = Spectral.wwz_psd(ys, ts, **psd_default)
        wwa, phase, AR1_q, coi, freqs, tau, Neffs, coeff = Spectral.wwz(ys,ts, **wwz_default)
          
        #Wrap up the output dictionary
        dict_out = {'wwa':wwa,
                    'phase':phase,
                    'AR1_q':AR1_q,
                    'coi':coi,
                    'freqs':freqs,
                    'tau':tau,
                    'Neffs':Neffs,
                    'coeff':coeff,
                    'psd':psd,
                    'psd_ar1_q95':psd_ar1_q95,
                    'psd_ar1':psd_ar1}
        
        # Make the plot if asked
        if fig is True:
            
            if type(wwaplot_default) is dict and type(psdplot_default)is dict:
                if 'figsize' in wwaplot_default.keys():
                    figsize = wwaplot_default['figsize']
                elif 'figsize' in psdplot_default.keys():
                    figsize = psdplot_default['figsize']
                else: figsize = [20,8]    
            elif type(wwaplot_default) is dict:
                if 'figsize' in wwaplot_default.keys():
                    figsize = wwaplot_default['figsize']
                else:
                    figsize = [20,8]
            elif type(psdplot_default) is dict:
                if 'figsize' in psdplot_default.keys():
                    figsize = psdplot_default['figsize']
                else:
                    figsize = [20,8]
            else: figsize = [20,8]    
                    
            
            fig = plt.figure(figsize = figsize)
            ax1 = plt.subplot2grid((1,3),(0,0), colspan =2)
        
            # Set the plot default
            if type(wwaplot_default) is dict:
                dict_in = wwaplot_default
                wwaplot_default={'AR1_q':AR1_q,
                                 'coi':coi,
                                 'levels':None,
                                 'tick_range':None,
                                 'yticks':None,
                                 'yticks_label': None,
                                 'ylim':None,
                                 'xticks':None,
                                 'xlabels':None,
                                 'figsize':[20,8],
                                 'clr_map':'OrRd',
                                 'cbar_drawedges':False,
                                 'cone_alpha':0.5,
                                 'plot_signif':True,
                                 'signif_style':'contour',
                                 'plot_cone':True,
                                 'title':None,
                                 'ax':None,
                                 'xlabel': label.upper()[0]+label[1:]+'('+s+')',
                                 'ylabel': 'Period ('+ageunits+')',
                                 'cbar_orientation':'vertical',
                                 'cbar_pad':0.05,
                                 'cbar_frac':0.15,
                                 'cbar_labelsize':None}
                for key, value in dict_in.items():
                    if key in wwaplot_default.keys():
                        wwaplot_default[key] = value
            
            else:
                wwaplot_default={'AR1_q':AR1_q,
                                 'coi':coi,
                                 'levels':None,
                                 'tick_range':None,
                                 'yticks':None,
                                 'yticks_label': None,
                                 'ylim':None,
                                 'xticks':None,
                                 'xlabels':None,
                                 'figsize':[20,8],
                                 'clr_map':'OrRd',
                                 'cbar_drawedges':False,
                                 'cone_alpha':0.5,
                                 'plot_signif':True,
                                 'signif_style':'contour',
                                 'plot_cone':True,
                                 'title':None,
                                 'ax':None,
                                 'xlabel': label.upper()[0]+label[1:]+'('+s+')',
                                 'ylabel': 'Period ('+ageunits+')',
                                 'cbar_orientation':'vertical',
                                 'cbar_pad':0.05,
                                 'cbar_frac':0.15,
                                 'cbar_labelsize':None}
                
            Spectral.plot_wwa(wwa, freqs, tau, **wwaplot_default)    
            
            ax2 = plt.subplot2grid((1,3),(0,2))
            
            if type(psdplot_default) is dict:
                dict_in = psdplot_default
                
                psdplot_default={'lmstyle':'-',
                                 'linewidth':None,
                                 'color': sns.xkcd_rgb["denim blue"],
                                 'ar1_lmstyle':'-',
                                 'ar1_linewidth':1,
                                 'period_ticks':None,
                                 'period_tickslabel':None,
                                 'psd_lim':None,
                                 'period_lim':None,
                                 'figsize':[20,8],
                                 'label':'PSD',
                                 'plot_ar1':True,
                                 'psd_ar1_q95':psd_ar1_q95,
                                 'title': None,
                                 'psd_ar1_color':sns.xkcd_rgb["pale red"],
                                 'ax':None,
                                 'vertical':False,
                                 'plot_gridlines':True,
                                 'period_label':'Period ('+ageunits+')',
                                 'psd_label':'Spectral Density',
                                 'zorder' : None}       
                                
                for key, value in dict_in.items():
                    if key in psdplot_default.keys():
                        psdplot_default[key] = value
                        
            else:
                   
               psdplot_default={'lmstyle':'-',
                                 'linewidth':None,
                                 'color': sns.xkcd_rgb["denim blue"],
                                 'ar1_lmstyle':'-',
                                 'ar1_linewidth':1,
                                 'period_ticks':None,
                                 'period_tickslabel':None,
                                 'psd_lim':None,
                                 'period_lim':None,
                                 'figsize':[20,8],
                                 'label':'PSD',
                                 'plot_ar1':True,
                                 'psd_ar1_q95':psd_ar1_q95,
                                 'title': None,
                                 'psd_ar1_color':sns.xkcd_rgb["pale red"],
                                 'ax':None,
                                 'vertical':False,
                                 'plot_gridlines':True,
                                 'period_label':'Period ('+ageunits+')',
                                 'psd_label':'Spectral Density',
                                 'zorder' : None} 
            
            Spectral.plot_psd(psd,freqs,**psdplot_default)
            
            if saveFig is True:
                LipdUtils.saveFigure(timeseries['dataSetName']+'_PSDplot',format,dir)
            else:
                plt.show()               
            
        else:
            fig = None
                         
    return dict_out, fig  

"""
Age model
"""

def Bchron(lipd, modelNum = None, objectName = None, rejectAges = None,\
           calCurves = None, reservoirAgeCorr = None, predictPositions = "paleo",\
           positionsThickness = None, outlierProbs =None, iterations =1000,\
           burn = 2000, thin = 8, extractDate = 1950-datetime.datetime.now().year,\
           maxExtrap = 500, thetaMhSd = 0.5, muMhSd = 0.1, psiMhSd = 0.1,\
           ageScaleVal = 1000, positionScaleVal = 100, saveLipd = True,\
           plot = True, figsize = [4,8], flipCoor = False,xlabel = None, ylabel = None,
           xlim = None, ylim = None, violinColor = '#8B008B',\
           medianLineColor = "black", medianLineWidth = 2.0,\
           CIFillColor = "Silver", samplePaths = True, samplePathNumber =10,\
           alpha = 0.5, saveFig = False, dir = "", format = "eps"):
    """ Runs Bchron and plot if asked 
    
    Fits a non-parametric chronology model to age/position data according to
    the Compound Poisson-Gamma model defined by Haslett and Parnell (2008). 
    This version used a slightly modified Markov chain Monte-Carlo fitting
    algorithm which aims to converge quicker and requires fewer iterations.
    It also a slightly modified procedure for identifying outliers.
    
    The Bchronology functions fits a compounf Poisson-Gamma distribution to the
    incrememnts between the dated levels. This involves a stochastic linear
    interpolation step where the age gaps are Gamma distributed, and the position
    gaps are Exponential. Radiocarbon and non-radiocarbon dates (including outliers)
    are updated within the fucntion also by MCMC.
    
    This function also allows to save the ensemble, distributions, and probability
    tables as well as the parameters with which the model was run into the LiPD file.
    
    Finally allows to make a plot.
    
    Args:
        lipd (dict): A dictionary containing the entry of a LiPD file. Can be
            obtained from lipd.readLipd() or pyleoclim.openLipd(). Please note
            that the Bchron function currently only allows for a single LiPD file
            (i.e., not the entire directory).
        modelNum (int): The model number in which to place the Bchron output. 
            If unknown, the function will try to make a guess and/or prompt
            based on the number of already available models.
        objectName (str): The name of the chron object in which to store the new
            model (e.g. "chron0")
        rejectAges (vector): A vector of 1/0 where 1 include the dates to be rejected. 
            Default it None.
        calCurves (list): (Optional) A vector of values containing either 'intcal13',
            'marine13', 'shcal13', or 'normal'. If none is provided, will
            prompt the user. Should be either of length =1 if using the same
            calibration for each age or the same length as the vector of ages.
        reservoirAgeCorr (array): (Optional) A list (matrix) of two floats that correspond to the
            DeltaR and DeltaR uncertainty. If already added to the ages and
            ages standard deviation, then enter [0,0] to bypass the prompt.
            Will only be applied if CalCurves is set to 'marine13'. Otherwise,
            leave to none.
        predictPositions (array): (Optional) a vector of positions 
            (e.g. depths) at which predicted age values are required. 
            Defaults to a sequence of length 100 from the top position to the
            bottom position.
        positionsThickness (array): (Optional) Thickness values for each of the positions.
            The thickness values should be the full thickness value of the
            slice. By default set to zero.
        outlierProbs (array): (Optional) A vector of prior outlier probabilities,
            one for each age. Defaults to 0.01
        iterations (int): (Optional) The number of iterations to start the procedure. 
            Default and minimum should be 10000.
        burn (int): (Optional) The number of starting iterations to discard.
            Default is 200
        thin (int): (Optional) The step size for every iteration to keep beyond
            the burnin. Default is 8.
        extractDate (float): (Optional) The top age of the core. Used for
            extrapolation purposes so that no extrapolated ages go beyond the
            top age of the core. Defaults to the current year.
        maxExtrap (int): (Optional) The maximum number of extrapolations to
            perform before giving up and setting the predicted ages to NA. 
            Useful for when large amounts of extrapolation are required, i.e.
            some of the predictPositions are a long way from the dated
            positions. Defaults to 500.
        thetaMhSd (float):  (Optional)  The Metropolis-Hastings standard
            deviation for the age parameters. Defaults to 0.5.
        muMhSd (float): (Optional)  The Metropolis-Hastings standard deviation
            for the compound Poisson-Gamma Scale. Defaults to 0.1
        psiMhSd (float): (Optional) The Metropolis-Hastings standard deviation 
            for the Compound Poisson-Gamma Scale.
        ageScaleVal (int): (Optional) A scale value for the ages. 
            Bchronology works best when the ages are scaled to be 
            approximately between 0 and 100.
            The default value is thus 1000 for ages given in years.
        positionScaleVal (int):  (Optional) A scale value for the positions. 
            Bchronology works best when the positions are scaled to be 
            approximately between 0 and 100. The default value is thus
            100 for positions given in cm.
        saveLipd (bool): If True, saves the ensemble, distribution, and probability
            tables along with the parameters used to run the model in the LiPD
            file.
        plot (bool): If True, makes a plot for the chronology
        figsize (list): The figure size. Default is [4,8]
        flipCoor (bool): If True, plots depth on the y-axis.
        xlabel (str): The label for the x-axis
        ylabel (str): The label for the y-axis
        xlim (list): Limits for the x-axis. Default corresponds to the min/max
            of the depth vector.
        ylim (list): Limits for the y-axis. Default set by matplotlib
        violinColor (str): The color for the violins. Default is purple
        medianLineColor (str): The color for the median line. Default is black.
        medianLineWidth (float): The width for the median line
        CIFillColor (str): Fill color in between the 95% confidence interval.
            Default is silver.
        samplePaths (bool): If True, draws sample paths from the distribution.
            Use the same color as the violins. 
        samplePathNumber (int): The number of sample paths to draw. Default is 10.
            Note: samplePaths need to be set to True. 
        alpha (float): The violins' transparency. Number between 0 and 1
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.
    
    Returns:
        depth - the predicted positions (either same as the user or the default) \n
        chron -  a numpy array of possible chronologies in each column.
            The number of rows is the same as the length of depth
        ageDist - the distribution of ages around each dates.
        fig - the figure        
    
    Warnings:
        This function requires R and the Bchron package and all its
            dependencies to be installed on the same machine.
            
    Reference:
        - Haslett, J., and Parnell, A. C. (2008). A simple monotone 
            process with application to radiocarbon-dated depth 
            chronologies. Journal of the Royal Statistical Society, 
            Series C, 57, 399-418. DOI:10.1111/j.1467-9876.2008.00623.x
        - Parnell, A. C., Haslett, J., Allen, J. R. M., Buck, C. E., 
            and Huntley, B. (2008). A flexible approach to assessing 
            synchroneity of past events using Bayesian reconstructions
            of sedimentation history. Quaternary Science Reviews, 
            27(19-20), 1872-1885. DOI:10.1016/j.quascirev.2008.07.009        
    """
    
    # Get the csv_list
    csv_dict = lpd.getCsv(lipd)
    # Get the list of possible measurement tables
    chronMeasurementTables, paleoMeasurementTables = LipdUtils.isMeasurement(csv_dict)
    #Check that there is a measurement table or exit
    if not chronMeasurementTables:
        sys.exit("No ChronMeasurementTables available to run BChron!")
    # Selct measurement table
    csvName = LipdUtils.whichMeasurement(chronMeasurementTables, csv_dict)
    # Get the ts-like object from selected measurement table
    ts_list = LipdUtils.getMeasurement(csvName, lipd)
    # Make sure there is no model associated with the choice
    if not modelNum and not objectName:
        model, objectName = LipdUtils.isModel(csvName, lipd)
        modelNum = LipdUtils.modelNumber(model)
    elif modelNum and not objectName:
        sys.exit("You must provide a dataObject when specifying a model.")        
    ## look for the inputs for Bchron
    # Find an age column
    print("Looking for age data...")
    match = LipdUtils.searchVar(ts_list,["radiocarbon","age14C"])
    if not match:
        sys.exit("No age data available")
    ages = ts_list[match]['values']
    ages = np.array(ages,dtype='float64')
    # Remove NaNs (can happen in mixed chronologies)
    idx = np.where(~np.isnan(ages))[0]
    ages = ages[idx]
    if "units" in ts_list[match].keys():
        agesUnit = ts_list[match]['units']
    else:
        agesUnit =[]    
    print("Age data found.")
    
    # Find a depth column
    print("Looking for a depth/position column...")
    match = LipdUtils.searchVar(ts_list,["depth"]) 
    if not match:
        sys.exit("No age data available")
    positions = ts_list[match]['values']
    positions = np.array(positions,dtype='float64')
    positions = positions[idx]
    if "units" in ts_list[match].keys():
        positionsUnits = ts_list[match]['units']
    else:
        positionsUnits = []
    print("Depth information found.")
    
    # Find the uncertainty
    print("Looking for age uncertainty...")
    match = LipdUtils.searchVar(ts_list,["uncertainty"],exact=False)
    if not match:
        sys.exit("No uncertainty data available")
    agesStd = ts_list[match]['values']
    agesStd = np.array(agesStd,dtype='float64')
    agesStd = agesStd[idx]
    print("Uncertainty data found.")
    
    # See if there is a column of reject ages
    if rejectAges == True:
        print("Looking for a column of rejected ages...")
        match = LipdUtils.searchVar(ts_list,["rejectAges"], exact=False)
        if not match:
            print("No column of rejected ages found.")
            rejectAges = None
            print("No ages rejected.")
        else:
            rejectAges = ts_list[match]['values']
            rejectAges = rejectAges[idx]
            print("Rejected ages found.")
    
    # Check if there are calibration curves
    if not calCurves:
        print("Looking for calibration curves...")
        match = LipdUtils.searchVar(ts_list,["calCurves"], exact=False)
        if not match:
            calCurves = RBchron.chooseCalCurves()
            calCurves = RBchron.verifyCalCurves(calCurves)
            if len(calCurves) == 1 and len(positions)!=1:
                calCurves = list(chain(*[[i]*len(positions) for i in calCurves]))
        else:
            calCurves = ts_list[match]['values']
            calCurves = RBchron.verifyCalCurves(calCurves)
    elif len(calCurves)==1 and len(positions)!=1:
        calCurves = RBchron.verifyCalCurves(calCurves)
        calCurves = list(chain(*[[i]*len(positions) for i in calCurves]))
    else:
        assert len(calCurves) == len(positions)
        calCurves = RBchron.verifyCalCurves(calCurves)
    print("Calibration curves found.")
                
    #Check for a reservoir age 
    
    if reservoirAgeCorr == None:
        print("Looking for a reservoir age correction.")
        match = LipdUtils.searchVar(ts_list,\
                                    ["reservoir","reservoirAge","correction"],\
                                    exact=True)
        if len(match)==1:
            ageCorr = ts_list[match]['values']
            ageCorr = ageCorr[idx]
            print("Reservoir Age correction found.")
            print("Looking for correction uncertainty...")
            match = LipdUtils.searchVar(ts_list,["reservoirUncertainty"],\
                                                 exact=True)
            if not match:
                print("No match found.")
                corrU = float(input("Enter a value for the correction uncertainty: "))
                ageCorrStd = corrU*np.ones((len(ageCorr),1))
            else:
                ageCorrStd = ts_list[match]['values']
                ageCorrStd = ageCorrStd[idx]
                    
        else:
            print("No match found.")
            ageCorr, ageCorrStd = RBchron.reservoirAgeCorrection()
        reservoirAgeCorr = np.column_stack((ageCorr,ageCorrStd))
        reservoirAgeCorr = reservoirAgeCorr.flatten()
        
    elif reservoirAgeCorr == True:
        ageCorr, ageCorrStd = RBchron.reservoirAgeCorrection()
        reservoirAgeCorr = np.column_stack((ageCorr,ageCorrStd))
        reservoirAgeCorr = reservoirAgeCorr.flatten()       
    else:
        if type(reservoirAgeCorr)!=np.ndarray:
            if type(reservoirAgeCorr) == list:
                reservoirAgeCorr = np.array(reservoirAgeCorr)
            else:
                sys.exit("The reservoir age correction should be either None, True or an array.")
         
    #Predict positions
    if predictPositions == "paleo":
        # Grab the paleomeasurementTables
        paleoCsvName = LipdUtils.whichMeasurement(paleoMeasurementTables, csv_dict)
        # Get the various timeseries
        paleots_list = LipdUtils.getMeasurement(paleoCsvName, lipd)
        #Look for a depth column
        match = LipdUtils.searchVar(paleots_list,["depth"], exact=True)
        if not match:
            print("No paleoDepth information available, using Bchron default")
            predictPositions = None
        else:
            predictPositions = paleots_list[match]['values']
            if "units" in paleots_list[match].keys():
                predictPositionsUnits = paleots_list[match]['units']
            else:
                predictPositionsUnits = []
     
    # Raise an error if the depth units don't correspond
    if predictPositionsUnits and positionsUnits:
        if predictPositionsUnits != positionsUnits:
            print("Depth units in the paleoData table and the chronData table don't match!")
            print("PaleoDepth are expressed in "+predictPositionsUnits+\
                  " while ChronDepth is expressed in "+positionsUnits)
            factor = float(input("Enter a correction value to bring Chron to Paleo scale"+\
                                 " or press Enter to exit."+\
                                 " Enter 1 if the units are in fact the same but"+\
                                 " written differently (e.g., m vs meter): "))  
            if not factor:
                sys.exit("No correction factor entered.")
            else:
                positions = positions*factor
      
    # Run Bchron
    print("Running Bchron. This could take a few minutes...")
    depth, chron, ageDist, run = RBchron.runBchron(ages, agesStd,positions,\
                                                    rejectAges = rejectAges,\
                                                    calCurves = calCurves,\
                                                    reservoirAgeCorr = reservoirAgeCorr,\
                                                    predictPositions = predictPositions,
                                                    positionsThickness= positionsThickness,\
                                                    outlierProbs=outlierProbs,\
                                                    iterations=iterations,\
                                                    burn=burn, thin=thin,\
                                                    extractDate=extractDate,\
                                                    maxExtrap=maxExtrap,\
                                                    thetaMhSd = thetaMhSd,\
                                                    muMhSd=muMhSd,psiMhSd = psiMhSd,\
                                                    ageScaleVal=ageScaleVal,\
                                                    positionScaleVal =positionScaleVal)
    
    ## Write into the LiPD file is asked
    
    print("Placing all the tables in the LiPD object...")
    if "model" in lipd["chronData"][objectName].keys():
        T = lipd["chronData"][objectName]["model"]
    else:
        T = lipd["chronData"][objectName].update({"model":{}})
    #Grab the part of the LiPD dictionary with the chronObject/model
    T = lipd["chronData"][objectName]["model"]
    
    ## methods
    inputs={"calCurves":calCurves,
            "iterations":iterations,
            "burn":burn,
            "thin":thin,
            "extractDate":extractDate,
            "maxExtrap": maxExtrap,
            "thetaMhSd": thetaMhSd,
            "muMhSd":muMhSd,
            "psiMhSd":psiMhSd,
            "ageScaleVal":ageScaleVal,
            "positionScaleVal":positionScaleVal}
    
    # Other None objects 
    if type(reservoirAgeCorr) is np.ndarray:
        reservoirAgeCorr = reservoirAgeCorr.tolist()
        inputs.update({"reservoirAgeCorr":reservoirAgeCorr})
    else:
        inputs.update({"reservoirAgeCorr":"Not provided"})    
    if type(rejectAges) is np.ndarray:
        rejectAges = rejectAges.tolist()
        inputs.update({"rejectAges":rejectAges})
    else:
        inputs.update({"rejectAges":"Not provided"})    
    if type(positionsThickness) is np.ndarray:
        positionsThickness = positionsThickness.tolist()
        inputs.update({"positionsThickness":positionsThickness})
    else:
        inputs.update({"positionsThickness":"Not provided"})
    if type(outlierProbs) is np.ndarray:
        outlierProbs = outlierProbs.tolist()
        inputs.update({"outlierProbs":outlierProbs})    
    else:
        inputs.update({"outlierProbs":"Not provided"}) 
        
    methods = {"algorithm":"Bchron", "inputs":inputs,"runEnv":"python"}
    
    #create the key for the new model
    key = str(objectName)+"model"+str(modelNum)
    # Add the method to object
    T.update({key:{}})
    T[key].update({"method":methods})
    
    ##EnsembleTable
    d = OrderedDict()
    T[key].update({"ensembleTable": d})
    key_ens = key+"ensemble0" # Key for the ensemble table
    T[key]["ensembleTable"].update({key_ens:{}})
    d = OrderedDict() #RESET VERY IMPORTANT
    T[key]["ensembleTable"][key_ens].update({"columns": d})
    
    #store age and depth info 
    number_age = np.arange(2,np.shape(chron)[1]+2,1)
    number_age = number_age.tolist()
    chron_list = np.transpose(chron).tolist()
    age_dict = {"number": number_age,
                "units":agesUnit,
                "variableName":"age",
                "values":chron_list}
    depth_dict = {"number":1,
                  "units":predictPositionsUnits,
                  "variableName":"depth",
                  "values":depth}
    T[key]["ensembleTable"][key_ens]["columns"].update({"depth":depth_dict})
    T[key]["ensembleTable"][key_ens]["columns"].update({"age":age_dict}) 
    
    ## Summary Table
    # First calculate some summary stats
    quant = mquantiles(chron,[0.025,0.5,0.975],axis=1)
    # Save as list for JSON
    lower95 = quant[:,0].tolist()
    medianAge = quant[:,1].tolist()
    upper95 = quant[:,2].tolist()
    meanVal = np.mean(chron, axis=1)
    meanVal = meanVal.tolist()
    # Put everything where it belongs
    d = OrderedDict()
    T[key].update({"summaryTable": d})
    key_sum = key+"summary0"
    T[key]["summaryTable"].update({key_sum:{}})
    d = OrderedDict()
    T[key]["summaryTable"][key_sum].update({"columns": d})
    # Place each column as separate dictionary
    T[key]["summaryTable"][key_sum]["columns"].update({"depth":{"variableName":"depth",
                                                                "units":predictPositionsUnits,
                                                                "values":depth,
                                                                "number":1}})
    T[key]["summaryTable"][key_sum]["columns"].update({"meanAge":{"variableName":"meanAge",
                                                                "units":agesUnit,
                                                                "values":meanVal,
                                                                "number":2}})
    T[key]["summaryTable"][key_sum]["columns"].update({"medianAge":{"variableName":"medianAge",
                                                                "units":agesUnit,
                                                                "values":medianAge,
                                                                "number":3}})
    T[key]["summaryTable"][key_sum]["columns"].update({"95Lower":{"variableName":"95Lower",
                                                                "units":agesUnit,
                                                                "values":lower95,
                                                                "number":4}})
    T[key]["summaryTable"][key_sum]["columns"].update({"95Higher":{"variableName":"95Higher",
                                                                "units":agesUnit,
                                                                "values":upper95,
                                                                "number":5}})
    
    ## DistributionTables
    d = OrderedDict()
    T[key].update({"distributionTable": d})
    
    for i in np.arange(0,np.shape(ageDist)[1],1):
        #Get the output in list form
         
        key_dist = key+"distribution"+str(i)
        T[key]["distributionTable"].update({key_dist:{}})
        d = OrderedDict()
        T[key]["distributionTable"][key_dist].update({"age14C":list(run[6][i][0])[0],
                                                     "calibrationCurve":calCurves[i],
                                                     "columns":d,
                                                     "depth":list(run[6][i][2])[0],
                                                     "depthUnits":predictPositionsUnits,
                                                     "sd14C":list(run[6][i][1])[0]})
        age_dict = {"number":1,
                    "variableName":"age",
                    "units":agesUnit,
                    "values":list(run[6][i][4])}
        density_dict = {"number":2,
                    "variableName":"probabilityDensity",
                    "values":list(run[6][i][5])}
        
        T[key]["distributionTable"][key_dist]["columns"].update({"age":age_dict,
                                                                 "probabilityDensity":density_dict})
        ## Finally write it out
    if saveLipd is True:        
        print("Writing LiPD file...")
        lpd.writeLipd(lipd,path=os.getcwd()) 
        
    #Plot if asked
    if plot is True:
        print("Plotting...")
        if flipCoor is True:
            if ylabel == None:
                if len(predictPositionsUnits)!=0:
                    ylabel = "Depth ("+str(predictPositionsUnits)+")"
                else:
                    ylabel = "Depth"
            if xlabel == None:
                if len(agesUnit)!=0:
                    xlabel = "Age ("+str(agesUnit)+")"
                else:
                    xlabel = "Age"
        else:    
            if xlabel == None:
                if len(predictPositionsUnits)!=0:
                    xlabel = "Depth ("+str(predictPositionsUnits)+")"
                else:
                    xlabel = "Depth"
            if ylabel == None:
                if len(agesUnit)!=0:
                    ylabel = "Age ("+str(agesUnit)+")"
                else:
                    ylabel = "Age"
        fig = RBchron.plotBchron(depth,chron,positions,ageDist, flipCoor=flipCoor,\
                                 xlabel =xlabel, ylabel=ylabel,\
                                 xlim = xlim, ylim = ylim,\
                                 violinColor = violinColor,\
                                 medianLineColor = medianLineColor,\
                                 medianLineWidth = medianLineWidth,\
                                 CIFillColor = CIFillColor,\
                                 samplePaths = samplePaths,\
                                 samplePathNumber = samplePathNumber,
                                 alpha = alpha, figsize = figsize)
        if saveFig is True:
            LipdUtils.saveFigure(lipd['dataSetName']+'_Bchron',format,dir)
        else:
            plt.show()            
    else:
        fig = None
        
    return depth, chron, positions, ageDist, fig               
        
    
