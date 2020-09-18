#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 05:46:36 2020

@author: deborahkhider

Contains all relevant mapping functions
"""
__all__=['map_all']

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import savefig, showfig 

def set_proj(projection='Robinson', proj_default = True): 
    """ Set the projection for Cartopy.
    
    Parameters
    ----------
    
    projection : string
        the map projection. Available projections:
        'Robinson' (default), 'PlateCarree', 'AlbertsEqualArea',
        'AzimuthalEquidistant','EquidistantConic','LambertConformal',
        'LambertCylindrical','Mercator','Miller','Mollweide','Orthographic',
        'Sinusoidal','Stereographic','TransverseMercator','UTM',
        'InterruptedGoodeHomolosine','RotatedPole','OSGB','EuroPP',
        'Geostationary','NearsidePerspective','EckertI','EckertII',
        'EckertIII','EckertIV','EckertV','EckertVI','EqualEarth','Gnomonic',
        'LambertAzimuthalEqualArea','NorthPolarStereo','OSNI','SouthPolarStereo'
    proj_default : bool
        If True, uses the standard projection attributes from Cartopy.
        Enter new attributes in a dictionary to change them. Lists of attributes
        can be found in the Cartopy documentation: 
            https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv
    
    Returns
    -------
        proj : the Cartopy projection object
        
    See Also
    --------
    pyleoclim.utils.mapping.map_all : mapping function making use of the projection
    
    """
    if proj_default is not True and type(proj_default) is not dict:
        raise TypeError('The default for the projections should either be provided'+
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
        raise ValueError('Invalid projection type')
    
    return proj

def map_all(lat, lon, criteria, marker=None, color =None,
            projection = 'Robinson', proj_default = True,
           background = True,borders = False, rivers = False, lakes = False,
           figsize = None, ax = None, scatter_kwargs=None, legend=True,
           lgd_kwargs=None,savefig_settings=None, mute=False):
    """ Map the location of all lat/lon according to some criteria
    
    Map the location of all lat/lon according to some criteria. Based on functions defined in the Cartopy package. 
    
    Parameters
    ----------
    
    lat : list
        a list of latitudes.
        
    lon : list
        a list of longitudes.
        
    criteria : list
        a list of unique criteria for plotting purposes. For instance,
        a map by the types of archive present in the dataset or proxy
        observations. Should have the same length as lon/lat.  
    
    marker : list
        a list of possible markers for each criterion. If None, will use pyleoclim default
    
    color : list
        a list of possible colors for each criterion. If None, will use pyleoclim default
    
    projection : string
        the map projection. Available projections:
        'Robinson' (default), 'PlateCarree', 'AlbertsEqualArea',
        'AzimuthalEquidistant','EquidistantConic','LambertConformal',
        'LambertCylindrical','Mercator','Miller','Mollweide','Orthographic',
        'Sinusoidal','Stereographic','TransverseMercator','UTM',
        'InterruptedGoodeHomolosine','RotatedPole','OSGB','EuroPP',
        'Geostationary','NearsidePerspective','EckertI','EckertII',
        'EckertIII','EckertIV','EckertV','EckertVI','EqualEarth','Gnomonic',
        'LambertAzimuthalEqualArea','NorthPolarStereo','OSNI','SouthPolarStereo'
        
    proj_default : bool
        If True, uses the standard projection attributes.
        Enter new attributes in a dictionary to change them. Lists of attributes
        can be found in the Cartopy documentation: 
            https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv
            
    background : bool
        If True, uses a shaded relief background (only one 
        available in Cartopy)
        
    borders : bool
        Draws the countries border. Defaults is off (False).
        
    rivers : bool
        Draws major rivers. Default is off (False).
        
    lakes : bool
        Draws major lakes. 
        Default is off (False).  
        
    figsize : list
        the size for the figure
        
    ax: axis,optional
        Return as axis instead of figure (useful to integrate plot into a subplot) 
        
    scatter_kwargs : dict
        Dictionary of arguments available in matplotlib.pyplot.scatter (https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.scatter.html).     
    
    legend : bool
        Whether the draw a legend on the figure
    
    lgd_kwargs : dict
        Dictionary of arguments for matplotlib.pyplot.legend (https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.legend.html)
    
    savefig_settings : dict
        Dictionary of arguments for matplotlib.pyplot.saveFig.
        - "path" must be specified; it can be any existed or non-existed path,
          with or without a suffix; if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}
    
    mute : bool
        if True, the plot will not show;
        recommend to set to true when more modifications are going to be made on ax
    
    Returns
    -------
    
    ax: The figure, or axis if ax specified 

    See Also
    --------
    pyleoclim.utils.mapping.set_proj : Set the projection for Cartopy-based maps
    """
    
    #Check that the lists have the same length and convert to numpy arrays
    if len(lat)!=len(lon) or len(lat)!=len(criteria) or len(lon)!=len(criteria):
        raise ValueError("Latitude, Longitude, and criteria list must be the same" +\
                 "length")
        
    # Check that the default is set to True or in dictionary format
    if proj_default is not True and type(proj_default) is not dict:
        raise TypeError('The default for the projections should either be provided'+
                 ' as a dictionary or set to True')
    
    # handle dict defaults
    savefig_settings={} if savefig_settings is None else savefig_settings.copy()
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()
    lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()
    
    if marker!=None:
        if 'marker' in scatter_kwargs.keys(): 
            print('marker has been set as a parameter to the map_all function, overriding scatter_kwargs')
            del scatter_kwargs['marker']
        if len(marker)!=len(criteria):
            raise ValueError('The marker vector should have the same length as the criteria vector')
    
    
    if color!=None:
        if 'facecolor' in scatter_kwargs.keys(): 
            print('facecolor has been set as a parameter to the map_all function, overriding scatter_kwargs')
            del scatter_kwargs['facecolor']
        if len(color)!=len(criteria):
            raise ValueError('The color vector should have the same length as the criteria vector')
    
    #get unique criteria/color/marker
    
    color_data=pd.DataFrame({'criteria':criteria,'color':color,'marker':marker})
    palette = color_data.drop_duplicates(subset='criteria')
    
    # get the projection:
    proj = set_proj(projection=projection, proj_default=proj_default) 
    data_crs = ccrs.PlateCarree()       
    # Make the figure        
    if ax is None:
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
    
    if color==None and marker==None:
        for crit in set(criteria):
            # Grab the indices with same criteria
            index = [i for i,x in enumerate(criteria) if x == crit]
            ax.scatter(np.array(lon)[index],np.array(lat)[index],
                        zorder = 10,
                        label = crit,
                        transform=data_crs,
                        **scatter_kwargs)
    elif color==None and marker!=None:
        for crit in set(criteria):
            # Grab the indices with same criteria
            index = [i for i,x in enumerate(criteria) if x == crit]
            ax.scatter(np.array(lon)[index],np.array(lat)[index],
                        zorder = 10,
                        label = crit,
                        transform=data_crs,
                        marker = palette[palette['criteria']==crit]['marker'].iloc[0],
                        **scatter_kwargs)
    elif color!=None and marker==None:
        for crit in set(criteria):
            # Grab the indices with same criteria
            index = [i for i,x in enumerate(criteria) if x == crit]
            ax.scatter(np.array(lon)[index],np.array(lat)[index],
                        zorder = 10,
                        label = crit,
                        transform=data_crs,
                        facecolor = palette[palette['criteria']==crit]['color'].iloc[0],
                        **scatter_kwargs)
    elif color!=None and marker!=None:
        for crit in set(criteria):
            # Grab the indices with same criteria
            index = [i for i,x in enumerate(criteria) if x == crit]
            ax.scatter(np.array(lon)[index],np.array(lat)[index],
                        zorder = 10,
                        label = crit,
                        transform=data_crs,
                        facecolor = palette[palette['criteria']==crit]['color'].iloc[0],
                        marker=palette[palette['criteria']==crit]['marker'].iloc[0],
                        **scatter_kwargs)
    
    if legend == True:
        ax.legend(**lgd_kwargs)
    else:
        ax.legend().remove()
    
    if 'fig' in locals():
        if 'path' in savefig_settings:
            savefig(fig, settings=savefig_settings)
        else:
            if not mute:
                showfig(fig)
        return fig, ax
    else:
        return ax
    
    return ax    
        

