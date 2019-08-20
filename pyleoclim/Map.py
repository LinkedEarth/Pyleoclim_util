# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:00:07 2016

@author: deborahkhider

Mapping functions.


"""
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def setProj(projection='Robinson', proj_default = True): 
    """ Set the projection for Cartopy.
    
    Args:
        projection (string): the map projection. Available projections:
            'Robinson' (default), 'PlateCarree', 'AlbertsEqualArea',
            'AzimuthalEquidistant','EquidistantConic','LambertConformal',
            'LambertCylindrical','Mercator','Miller','Mollweide','Orthographic',
            'Sinusoidal','Stereographic','TransverseMercator','UTM',
            'InterruptedGoodeHomolosine','RotatedPole','OSGB','EuroPP',
            'Geostationary','NearsidePerspective','EckertI','EckertII',
            'EckertIII','EckertIV','EckertV','EckertVI','EqualEarth','Gnomonic',
            'LambertAzimuthalEqualArea','NorthPolarStereo','OSNI','SouthPolarStereo'
        proj_default (bool): If True, uses the standard projection attributes.
            Enter new attributes in a dictionary to change them. Lists of attributes
            can be found in the Cartopy documentation: 
                https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv
    Returns:
        proj - the Cartopy projection object
    """
    if proj_default is not True and type(proj_default) is not dict:
        sys.exit('The default for the projections should either be provided'+
                 ' as a dictionary or set to True')
    
    # Set the projection
    if projection == 'Robinson':
        if proj_default is True:
            proj = ccrs.Robinson() 
        else: proj = ccrs.Robinson(**proj_default) 
    elif projection == 'PlateCarree':
        if proj_default is True:
            proj = ccrs.PlateCarree() 
        else: proj = ccrs.PlateCarree(**proj_default) 
    elif projection == 'AlbersEqualArea':
        if proj_default is True:
            proj = ccrs.AlbersEqualArea() 
        else: proj = ccrs.AlbersEqualArea(**proj_default) 
    elif projection == 'AzimuthalEquidistant':
        if proj_default is True:
            proj = ccrs.AzimuthalEquidistant() 
        else: proj = ccrs.AzimuthalEquidistant(**proj_default)
    elif projection == 'EquidistantConic':
        if proj_default is True:
            proj = ccrs.EquidistantConic() 
        else: proj = ccrs.EquidistantConic(**proj_default)
    elif projection == 'LambertConformal':
        if proj_default is True:
            proj = ccrs.LambertConformal() 
        else: proj = ccrs.LambertConformal(**proj_default)
    elif projection == 'LambertCylindrical':
        if proj_default is True:
            proj = ccrs.LambertCylindrical() 
        else: proj = ccrs.LambertCylindrical(**proj_default)
    elif projection == 'Mercator':
        if proj_default is True:
            proj = ccrs.Mercator() 
        else: proj = ccrs.Mercator(**proj_default)
    elif projection == 'Miller':
        if proj_default is True:
            proj = ccrs.Miller() 
        else: proj = ccrs.Miller(**proj_default)
    elif projection == 'Mollweide':
        if proj_default is True:
            proj = ccrs.Mollweide() 
        else: proj = ccrs.Mollweide(**proj_default)
    elif projection == 'Orthographic':
        if proj_default is True:
            proj = ccrs.Orthographic() 
        else: proj = ccrs.Orthographic(**proj_default)
    elif projection == 'Sinusoidal':
        if proj_default is True:
            proj = ccrs.Sinusoidal() 
        else: proj = ccrs.Sinusoidal(**proj_default)
    elif projection == 'Stereographic':
        if proj_default is True:
            proj = ccrs.Stereographic() 
        else: proj = ccrs.Stereographic(**proj_default)
    elif projection == 'TransverseMercator':
        if proj_default is True:
            proj = ccrs.TransverseMercator() 
        else: proj = ccrs.TransverseMercator(**proj_default)
    elif projection == 'TransverseMercator':
        if proj_default is True:
            proj = ccrs.TransverseMercator() 
        else: proj = ccrs.TransverseMercator(**proj_default)
    elif projection == 'UTM':
        if proj_default is True:
            proj = ccrs.UTM() 
        else: proj = ccrs.UTM(**proj_default)
    elif projection == 'UTM':
        if proj_default is True:
            proj = ccrs.UTM() 
        else: proj = ccrs.UTM(**proj_default)
    elif projection == 'InterruptedGoodeHomolosine':
        if proj_default is True:
            proj = ccrs.InterruptedGoodeHomolosine() 
        else: proj = ccrs.InterruptedGoodeHomolosine(**proj_default)
    elif projection == 'RotatedPole':
        if proj_default is True:
            proj = ccrs.RotatedPole() 
        else: proj = ccrs.RotatedPole(**proj_default)
    elif projection == 'OSGB':
        if proj_default is True:
            proj = ccrs.OSGB() 
        else: proj = ccrs.OSGB(**proj_default)
    elif projection == 'EuroPP':
        if proj_default is True:
            proj = ccrs.EuroPP() 
        else: proj = ccrs.EuroPP(**proj_default)
    elif projection == 'Geostationary':
        if proj_default is True:
            proj = ccrs.Geostationary() 
        else: proj = ccrs.Geostationary(**proj_default)
    elif projection == 'NearsidePerspective':
        if proj_default is True:
            proj = ccrs.NearsidePerspective() 
        else: proj = ccrs.NearsidePerspective(**proj_default)
    elif projection == 'EckertI':
        if proj_default is True:
            proj = ccrs.EckertI() 
        else: proj = ccrs.EckertI(**proj_default)
    elif projection == 'EckertII':
        if proj_default is True:
            proj = ccrs.EckertII() 
        else: proj = ccrs.EckertII(**proj_default)
    elif projection == 'EckertIII':
        if proj_default is True:
            proj = ccrs.EckertIII() 
        else: proj = ccrs.EckertIII(**proj_default)
    elif projection == 'EckertIV':
        if proj_default is True:
            proj = ccrs.EckertIV() 
        else: proj = ccrs.EckertIV(**proj_default)
    elif projection == 'EckertV':
        if proj_default is True:
            proj = ccrs.EckertV() 
        else: proj = ccrs.EckertV(**proj_default)
    elif projection == 'EckertVI':
        if proj_default is True:
            proj = ccrs.EckertVI() 
        else: proj = ccrs.EckertVI(**proj_default)
    elif projection == 'EqualEarth':
        if proj_default is True:
            proj = ccrs.EqualEarth() 
        else: proj = ccrs.EqualEarth(**proj_default)
    elif projection == 'Gnomonic':
        if proj_default is True:
            proj = ccrs.Gnomonic() 
        else: proj = ccrs.Gnomonic(**proj_default)
    elif projection == 'LambertAzimuthalEqualArea':
        if proj_default is True:
            proj = ccrs.LambertAzimuthalEqualArea() 
        else: proj = ccrs.LambertAzimuthalEqualArea(**proj_default)
    elif projection == 'NorthPolarStereo':
        if proj_default is True:
            proj = ccrs.NorthPolarStereo() 
        else: proj = ccrs.NorthPolarStereo(**proj_default)
    elif projection == 'OSNI':
        if proj_default is True:
            proj = ccrs.OSNI() 
        else: proj = ccrs.OSNI(**proj_default)
    elif projection == 'OSNI':
        if proj_default is True:
            proj = ccrs.SouthPolarStereo() 
        else: proj = ccrs.SouthPolarStereo(**proj_default)
    else:
        sys.exit('Invalid projection type')
        
    return proj

def mapAll(lat, lon, criteria, projection = 'Robinson', proj_default = True,\
           background = True,borders = False, rivers = False, lakes = False,\
           figsize = [10,4], ax = None, palette=None, markersize = 50):
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
        projection (string): the map projection. Available projections:
            'Robinson' (default), 'PlateCarree', 'AlbertsEqualArea',
            'AzimuthalEquidistant','EquidistantConic','LambertConformal',
            'LambertCylindrical','Mercator','Miller','Mollweide','Orthographic',
            'Sinusoidal','Stereographic','TransverseMercator','UTM',
            'InterruptedGoodeHomolosine','RotatedPole','OSGB','EuroPP',
            'Geostationary','NearsidePerspective','EckertI','EckertII',
            'EckertIII','EckertIV','EckertV','EckertVI','EqualEarth','Gnomonic',
            'LambertAzimuthalEqualArea','NorthPolarStereo','OSNI','SouthPolarStereo'
        proj_default (bool): If True, uses the standard projection attributes.
            Enter new attributes in a dictionary to change them. Lists of attributes
            can be found in the Cartopy documentation: 
                https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv
        background (bool): If True, uses a shaded relief background (only one 
            available in Cartopy)
        borders (bool): Draws the countries border. Defaults is off (False). 
        rivers (bool): Draws major rivers. Default is off (False).
        lakes (bool): Draws major lakes. 
            Default is off (False).
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
        
    # Check that the default is set to True or in dictionary format
    if proj_default is not True and type(proj_default) is not dict:
        sys.exit('The default for the projections should either be provided'+
                 ' as a dictionary or set to True')
        
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
    
    # get the projection:
    proj = setProj(projection=projection, proj_default=proj_default)        
    # Make the figure        
    if not ax:
        fig, ax = plt.subplots(figsize=figsize,subplot_kw=dict(projection=proj))     
    # draw the coastlines    
    ax.coastlines()
    
    # Background
    if background is True:
        ax.stock_img()
            
    #Other extra information
    if borders is True:
        ax.add_feature(cfeature.BORDERS)
    if lakes is True:
        ax.add_feature(cfeature.LAKES)
    if rivers is True:
        ax.add_feature(cfeature.RIVERS)
    
    # Get the indexes by criteria
    for crit in set(criteria):
        # Grab the indices with same criteria
        index = [i for i,x in enumerate(criteria) if x == crit]
        ax.scatter(np.array(lon)[index],np.array(lat)[index],
                    s= markersize,
                    facecolor = palette[crit][0],
                    marker = palette[crit][1],
                    zorder = 10,
                    label = crit,
                    transform=ccrs.PlateCarree())
    plt.legend(loc = 'center', bbox_to_anchor=(1.1,0.5),scatterpoints = 1,
               frameon = False, fontsize = 8, markerscale = 0.7)
    
    return ax    
        
def mapOne(lat, lon, projection = 'Orthographic', proj_default = True, label = None,\
           background = True,borders = False, rivers = False, lakes = False,\
           markersize = 50, marker = "ro", figsize = [4,4], \
           ax = None):
    """ Map one location on the globe
    
    Args:
        lat (float): a float number representing latitude
        lon (float): a float number representing longitude
        projection (string): the map projection. Available projections:
            'Robinson', 'PlateCarree', 'AlbertsEqualArea',
            'AzimuthalEquidistant','EquidistantConic','LambertConformal',
            'LambertCylindrical','Mercator','Miller','Mollweide','Orthographic' (Default),
            'Sinusoidal','Stereographic','TransverseMercator','UTM',
            'InterruptedGoodeHomolosine','RotatedPole','OSGB','EuroPP',
            'Geostationary','NearsidePerspective','EckertI','EckertII',
            'EckertIII','EckertIV','EckertV','EckertVI','EqualEarth','Gnomonic',
            'LambertAzimuthalEqualArea','NorthPolarStereo','OSNI','SouthPolarStereo'
        proj_default (bool): If True, uses the standard projection attributes, including centering.
            Enter new attributes in a dictionary to change them. Lists of attributes
            can be found in the Cartopy documentation: 
                https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv
        background (bool): If True, uses a shaded relief background (only one 
            available in Cartopy)
        label (str): label for the point. Default is None. 
        borders (bool): Draws the countries border. Defaults is off (False). 
        rivers (bool): Draws major rivers. Default is off (False).
        lakes (bool): Draws major lakes. 
            Default is off (False).
        markersize (int): The size of the marker.
        marker (str or list): color and type of marker. 
        figsize (list): the size for the figure
        ax: Return as axis instead of figure (useful to integrate plot into a subplot) 
    
    """
    # get the projection:
    if proj_default is True:
        proj_default = {'central_longitude':lon}
    proj = setProj(projection=projection, proj_default=proj_default)        
    # Make the figure        
    if not ax:
        fig, ax = plt.subplots(figsize=figsize,subplot_kw=dict(projection=proj))     
    # draw the coastlines    
    ax.coastlines()
    
    # Background
    if background is True:
        ax.stock_img()  
    
    #Other extra information
    if borders is True:
        ax.add_feature(cfeature.BORDERS)
    if lakes is True:
        ax.add_feature(cfeature.LAKES)
    if rivers is True:
        ax.add_feature(cfeature.RIVERS)
    
    # Draw the point
    ax.scatter(np.array(lon),np.array(lat),
               s= markersize,
               facecolor = marker[0],
               marker = marker[1],
               zorder = 10,
               transform=ccrs.PlateCarree())
    
    # Add a label if necessary
    if label is not None:
       assert type(label) is str, 'Label should be of type string'
       ax.annotate(label,(np.array(lon),np.array(lat)),fontweight='bold')
        
    return ax