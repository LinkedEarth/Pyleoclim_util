#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapping utilities for geolocated objects, leveraging Cartopy.
"""
__all__ = ['map', 'compute_dist']

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
from itertools import cycle
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from .plotting import savefig
from .lipdutils import PLOT_DEFAULT, LipdToOntology


def set_proj(projection='Robinson', proj_default=True):
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

    proj_default : bool; {True,False}

        If True, uses the standard projection attributes from Cartopy.
        Enter new attributes in a dictionary to change them. Lists of attributes
        can be found in the `Cartopy documentation <https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv>`_.
    
    Returns
    -------
        proj : the Cartopy projection object
        
    See Also
    --------
    pyleoclim.utils.mapping.map : mapping function making use of the projection
    
    """
    if proj_default is not True and type(proj_default) is not dict:
        raise TypeError('The default for the projections should either be provided' +
                        ' as a dictionary or set to True')

    # Set the projection
    if projection == 'Robinson':
        if proj_default is True:
            proj = ccrs.Robinson()
        else:
            proj = ccrs.Robinson(**proj_default)
    elif projection == 'PlateCarree':
        if proj_default is True:
            proj = ccrs.PlateCarree()
        else:
            proj = ccrs.PlateCarree(**proj_default)
    elif projection == 'AlbersEqualArea':
        if proj_default is True:
            proj = ccrs.AlbersEqualArea()
        else:
            proj = ccrs.AlbersEqualArea(**proj_default)
    elif projection == 'AzimuthalEquidistant':
        if proj_default is True:
            proj = ccrs.AzimuthalEquidistant()
        else:
            proj = ccrs.AzimuthalEquidistant(**proj_default)
    elif projection == 'EquidistantConic':
        if proj_default is True:
            proj = ccrs.EquidistantConic()
        else:
            proj = ccrs.EquidistantConic(**proj_default)
    elif projection == 'LambertConformal':
        if proj_default is True:
            proj = ccrs.LambertConformal()
        else:
            proj = ccrs.LambertConformal(**proj_default)
    elif projection == 'LambertCylindrical':
        if proj_default is True:
            proj = ccrs.LambertCylindrical()
        else:
            proj = ccrs.LambertCylindrical(**proj_default)
    elif projection == 'Mercator':
        if proj_default is True:
            proj = ccrs.Mercator()
        else:
            proj = ccrs.Mercator(**proj_default)
    elif projection == 'Miller':
        if proj_default is True:
            proj = ccrs.Miller()
        else:
            proj = ccrs.Miller(**proj_default)
    elif projection == 'Mollweide':
        if proj_default is True:
            proj = ccrs.Mollweide()
        else:
            proj = ccrs.Mollweide(**proj_default)
    elif projection == 'Orthographic':
        if proj_default is True:
            proj = ccrs.Orthographic()
        else:
            proj = ccrs.Orthographic(**proj_default)
    elif projection == 'Sinusoidal':
        if proj_default is True:
            proj = ccrs.Sinusoidal()
        else:
            proj = ccrs.Sinusoidal(**proj_default)
    elif projection == 'Stereographic':
        if proj_default is True:
            proj = ccrs.Stereographic()
        else:
            proj = ccrs.Stereographic(**proj_default)
    elif projection == 'TransverseMercator':
        if proj_default is True:
            proj = ccrs.TransverseMercator()
        else:
            proj = ccrs.TransverseMercator(**proj_default)
    elif projection == 'TransverseMercator':
        if proj_default is True:
            proj = ccrs.TransverseMercator()
        else:
            proj = ccrs.TransverseMercator(**proj_default)
    elif projection == 'UTM':
        if proj_default is True:
            proj = ccrs.UTM()
        else:
            proj = ccrs.UTM(**proj_default)
    elif projection == 'UTM':
        if proj_default is True:
            proj = ccrs.UTM()
        else:
            proj = ccrs.UTM(**proj_default)
    elif projection == 'InterruptedGoodeHomolosine':
        if proj_default is True:
            proj = ccrs.InterruptedGoodeHomolosine()
        else:
            proj = ccrs.InterruptedGoodeHomolosine(**proj_default)
    elif projection == 'RotatedPole':
        if proj_default is True:
            proj = ccrs.RotatedPole()
        else:
            proj = ccrs.RotatedPole(**proj_default)
    elif projection == 'OSGB':
        if proj_default is True:
            proj = ccrs.OSGB()
        else:
            proj = ccrs.OSGB(**proj_default)
    elif projection == 'EuroPP':
        if proj_default is True:
            proj = ccrs.EuroPP()
        else:
            proj = ccrs.EuroPP(**proj_default)
    elif projection == 'Geostationary':
        if proj_default is True:
            proj = ccrs.Geostationary()
        else:
            proj = ccrs.Geostationary(**proj_default)
    elif projection == 'NearsidePerspective':
        if proj_default is True:
            proj = ccrs.NearsidePerspective()
        else:
            proj = ccrs.NearsidePerspective(**proj_default)
    elif projection == 'EckertI':
        if proj_default is True:
            proj = ccrs.EckertI()
        else:
            proj = ccrs.EckertI(**proj_default)
    elif projection == 'EckertII':
        if proj_default is True:
            proj = ccrs.EckertII()
        else:
            proj = ccrs.EckertII(**proj_default)
    elif projection == 'EckertIII':
        if proj_default is True:
            proj = ccrs.EckertIII()
        else:
            proj = ccrs.EckertIII(**proj_default)
    elif projection == 'EckertIV':
        if proj_default is True:
            proj = ccrs.EckertIV()
        else:
            proj = ccrs.EckertIV(**proj_default)
    elif projection == 'EckertV':
        if proj_default is True:
            proj = ccrs.EckertV()
        else:
            proj = ccrs.EckertV(**proj_default)
    elif projection == 'EckertVI':
        if proj_default is True:
            proj = ccrs.EckertVI()
        else:
            proj = ccrs.EckertVI(**proj_default)
    elif projection == 'EqualEarth':
        if proj_default is True:
            proj = ccrs.EqualEarth()
        else:
            proj = ccrs.EqualEarth(**proj_default)
    elif projection == 'Gnomonic':
        if proj_default is True:
            proj = ccrs.Gnomonic()
        else:
            proj = ccrs.Gnomonic(**proj_default)
    elif projection == 'LambertAzimuthalEqualArea':
        if proj_default is True:
            proj = ccrs.LambertAzimuthalEqualArea()
        else:
            proj = ccrs.LambertAzimuthalEqualArea(**proj_default)
    elif projection == 'NorthPolarStereo':
        if proj_default is True:
            proj = ccrs.NorthPolarStereo()
        else:
            proj = ccrs.NorthPolarStereo(**proj_default)
    elif projection == 'OSNI':
        if proj_default is True:
            proj = ccrs.OSNI()
        else:
            proj = ccrs.OSNI(**proj_default)
    elif projection == 'SouthPolarStereo':
        if proj_default is True:
            proj = ccrs.SouthPolarStereo()
        else:
            proj = ccrs.SouthPolarStereo(**proj_default)
    else:
        raise ValueError('Invalid projection type')

    return proj


def map(lat, lon, criteria, marker=None, color=None,
        projection='Robinson', proj_default=True,
        background=True, borders=False, rivers=False, lakes=False,
        figsize=None, ax=None, scatter_kwargs=None, legend=True, legend_title=None,
        lgd_kwargs=None, savefig_settings=None):
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
        can be found in the `Cartopy documentation <https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv>`_. 
            
    background : bool
        If True, uses a shaded relief background (only one available in Cartopy)
        
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
        Dictionary of arguments available in `matplotlib.pyplot.scatter <https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.scatter.html>`_.     
    
    legend : bool
        Whether the draw a legend on the figure
    
    legend_title : str
        Use this instead of a dynamic range for legend
    
    lgd_kwargs : dict
        Dictionary of arguments for `matplotlib.pyplot.legend <https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.legend.html>`_.
    
    savefig_settings : dict

        Dictionary of arguments for matplotlib.pyplot.saveFig.

        - "path" must be specified; it can be any existed or non-existed path,
          with or without a suffix; if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}
    
    Returns
    -------
    
    ax: The figure, or axis if ax specified 

    See Also
    --------
    pyleoclim.utils.mapping.set_proj : Set the projection for Cartopy-based maps
    """

    # Take care of duplicate legends
    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), **lgd_kwargs)

    # Check that the lists have the same length and convert to numpy arrays
    if len(lat) != len(lon) or len(lat) != len(criteria) or len(lon) != len(criteria):
        raise ValueError("Latitude, Longitude, and criteria list must be the same" + \
                         "length")

    # Check that the default is set to True or in dictionary format
    if proj_default is not True and type(proj_default) is not dict:
        raise TypeError('The default for the projections should either be provided' +
                        ' as a dictionary or set to True')

    # handle dict defaults
    savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()
    lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

    if marker is not None:
        if 'marker' in scatter_kwargs.keys():
            print('marker has been set as a parameter to the map_all function, overriding scatter_kwargs')
            del scatter_kwargs['marker']
        if type(marker) == list and len(marker) != len(criteria):
            raise ValueError('The marker vector should have the same length as the lat/lon/criteria vector')

    if color is not None:
        if 'facecolor' in scatter_kwargs.keys():
            print('facecolor has been set as a parameter to the map_all function, overriding scatter_kwargs')
            del scatter_kwargs['facecolor']
        if type(color) == list and len(color) != len(criteria):
            raise ValueError('The color vector should have the same length as the lon/lat/criteria vector')

    # Prepare scatter information
    if 's' in scatter_kwargs.keys():
        if type(scatter_kwargs['s']) == list and len(scatter_kwargs['s']) != len(criteria):
            raise ValueError('If s is a list, it should have the same length as lon/lat/criteria')
    else:
        scatter_kwargs['s'] = None

    if 'edgecolors' in scatter_kwargs.keys():
        if type(scatter_kwargs['edgecolors']) == list and len(scatter_kwargs['edgecolors']) != len(criteria):
            raise ValueError('If edgecolors is a list, it should have the same length as lon/lat/criteria')
    else:
        scatter_kwargs['edgecolors'] = None

    symbols = pd.DataFrame({'criteria': criteria, 'color': color, 'marker': marker,
                            's': scatter_kwargs['s'], 'edgecolors': scatter_kwargs['edgecolors']})

    # delete extra scatter_kwargs
    del scatter_kwargs['s']
    del scatter_kwargs['edgecolors']

    # get the projection:
    proj = set_proj(projection=projection, proj_default=proj_default)
    if proj_default == True:
        proj1 = {'central_latitude': np.mean(lat),
                 'central_longitude': np.mean(lon)}
        proj2 = {'central_latitude': np.mean(lat)}
        proj3 = {'central_longitude': np.mean(lon)}
        try:
            proj = set_proj(projection=projection, proj_default=proj1)
        except:
            try:
                proj = set_proj(projection=projection, proj_default=proj3)
            except:
                proj = set_proj(projection=projection, proj_default=proj2)

    data_crs = ccrs.PlateCarree()
    # Make the figure        
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=proj))
        # draw the coastlines
    ax.add_feature(cfeature.COASTLINE)
    # Background
    if background is True:
        ax.stock_img()
        # Other extra information
    if borders is True:
        ax.add_feature(cfeature.BORDERS)
    if lakes is True:
        ax.add_feature(cfeature.LAKES)
    if rivers is True:
        ax.add_feature(cfeature.RIVERS)

    # Get the indexes by criteria
    if legend_title is not None:
        for index, crit in enumerate(criteria):
            ax.scatter(np.array(lon)[index], np.array(lat)[index],
                       zorder=10,
                       label=legend_title,
                       transform=data_crs,
                       marker=symbols['marker'].iloc[index],
                       color=symbols['color'].iloc[index],
                       s=symbols['s'].iloc[index],
                       edgecolors='white',
                       # edgecolors= symbols['edgecolors'].iloc[index],
                       **scatter_kwargs)

    else:
        for index, crit in enumerate(criteria):
            ax.scatter(np.array(lon)[index], np.array(lat)[index],
                       zorder=10,
                       label=crit,
                       transform=data_crs,
                       marker=symbols['marker'].iloc[index],
                       color=symbols['color'].iloc[index],
                       s=symbols['s'].iloc[index],
                       edgecolors='white',
                       # edgecolors= symbols['edgecolors'].iloc[index],
                       **scatter_kwargs)

    if legend == True:
        # ax.legend(**lgd_kwargs)
        legend_without_duplicate_labels(ax)
    else:
        ax.legend().remove()

    if 'fig' in locals():
        if 'path' in savefig_settings:
            savefig(fig, settings=savefig_settings)
        return fig, ax
    else:
        return ax


def scatter_map(geos, hue='archiveType', size=None, marker='archiveType', edgecolor='w', projection='Robinson', proj_default=True,
                background=True, borders=False, rivers=False, lakes=False, ocean=True, land=True,
                figsize=None, scatter_kwargs=None, legend_kwargs=None, legend=True, cmap='viridis',
                fig=None, gs_slot=None):

    def make_df(geo_ms, hue=None, marker=None, size=None):
        try:
            geo_series_list = geo_ms.series_list
        except:
            geo_series_list = [geo_ms]
        lats = [geos.lat for geos in geo_series_list]
        lons = [geos.lon for geos in geo_series_list]

        trait_d = {'hue': hue, 'marker': marker, 'size': size}
        value_d = {'lat': lats, 'lon': lons}
        for trait_key in trait_d.keys():
            trait = trait_d[trait_key]
            if trait != None:
                trait_vals = [geos.__dict__[trait] if trait in geos.__dict__.keys() else None for geos in
                              geo_series_list]
                value_d[trait] = [trait_val if trait_val != 'None' else None for trait_val in trait_vals]
        geos_df = pd.DataFrame(value_d)
        return geos_df

    def plot_scatter(df=None, x=None, y=None, hue_var=None, size_var=None, marker_var=None, edgecolor='w',
                     ax=None, proj=None, scatter_kwargs=None, legend=True, legend_kwargs=None,
                     # fig=None, gs_slot=None,
                     cmap='viridis', **kwargs):

        if type(scatter_kwargs) != dict:
            scatter_kwargs = {}
        print(scatter_kwargs)

        plot_defaults = copy.copy(PLOT_DEFAULT)
        palette = None

        _df = df
        if len(_df) == 1:
            _df = _df.reindex()

        if ax is None:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot()

        elif type(ax) == cartopy.mpl.geoaxes.GeoAxes:
            if proj is not None:
                scatter_kwargs['transform'] = proj
            else:
                scatter_kwargs['transform'] = ccrs.PlateCarree()

        missing_d = {'hue': kwargs['missing_val_hue'] if 'missing_val_hue' in kwargs else 'k',
                     'marker': kwargs['missing_val_marker'] if 'missing_val_marker' in kwargs else r'$?$',
                     'size': kwargs['missing_val_size'] if 'missing_val_size' in kwargs else 200,
                     'label': kwargs['missing_val_label'] if 'missing_val_label' in kwargs else 'missing',
                     }
        missing_val = missing_d['label']

        if 'edgecolor' not in scatter_kwargs:
                scatter_kwargs['edgecolor'] = edgecolor

        for trait_var in [hue_var, marker_var, size_var]:
            if trait_var not in _df.columns:
                trait_var = None

        trait_vars = [trait_var for trait_var in [hue_var, marker_var, size_var] if
                      ((trait_var != None) and (trait_var in _df.columns))]

        for trait_var in trait_vars:
            if type(_df[trait_var]) == pd.Series:
                _df[trait_var] = _df[trait_var].fillna(missing_val)
            else:
                if _df[trait_var] in [None, 'None']:
                    _df[trait_var] = missing_val

        if size_var == None:
            scatter_kwargs['s'] = missing_d['size']
        else:
            if len(set(_df[size_var]) - set([missing_val])) < 2:
                scatter_kwargs['s'] = missing_d['size']
                size_var = None
            else:
                # if size does vary, filter out missing values; at present no strategy to depict missing size values
                _df = _df[_df[size_var] != missing_val]
                scatter_kwargs['sizes'] = (20, 200)
        trait_vars = [trait_var for trait_var in [hue_var, marker_var, size_var] if trait_var != None]

        # mapping between marker styles and marker values
        # the difference between '.' and 'o' is a matter of size, so can't use both when size is a variable
        if size_var != None:
            marker_selection = [marker for marker in Line2D.filled_markers if
                                marker not in ['.', 'o', '', ' ', 'none', 'None']]
        else:
            marker_selection = [marker for marker in Line2D.filled_markers if
                                marker not in ['.', '', ' ', 'none', 'None']]

        # check if a mapping has been prescribed
        trait2marker = None
        if 'marker_mapping' in kwargs:
            trait2marker = kwargs['marker_mapping']
        elif marker_var == 'archiveType':
            trait2marker = {key: value[1] for key, value in plot_defaults.items()}

        if type(trait2marker) == dict:
            residual_traits = [trait for trait in _df[marker_var].unique() if
                               trait not in trait2marker.keys()]  # +set(['missing'])
            residual_markers = [marker for marker in marker_selection if marker not in trait2marker.values()]
            for trait in residual_traits:
                trait2marker[trait] = residual_markers.pop()

            for key in trait2marker:
                if type(trait2marker[key]) == str:
                    if trait2marker[key] == missing_d['marker']:
                        print('default symbol for missing values used in mapping')
                    try:
                        trait2marker[key] = mpl.markers.MarkerStyle(trait2marker[key], fillstyle=None)
                    except:
                        pass

        m_cycle = cycle(marker_selection)
        if ((marker_var != None) and (type(trait2marker) != dict)):
            if len(_df[marker_var].unique()) > 1:
                trait2marker = {trait_val: next(m_cycle) for ik, trait_val
                                in enumerate(_df[marker_var].unique())}

        if ((type(marker_var) == str) and (type(trait2marker) == dict)):
            # residual = set(_df[marker_var].unique()) - set(trait2marker.keys())
            # with missing values assigned '?'
            trait2marker['missing'] = missing_d['marker']
            scatter_kwargs['markers'] = trait2marker
            residual_traits = [trait for trait in _df[marker_var].unique() if
                               trait not in trait2marker.keys()]
            if len(residual_traits) > 0:
                print(residual_traits)

                # palette

        # use hue mapping if supplied
        if 'hue_mapping' in kwargs:
            palette = kwargs['hue_mapping']
        elif hue_var == 'archiveType':
            palette = {key: value[0] for key, value in plot_defaults.items()}
        elif type(hue_var) == str:
            palette = cmap

        if ((type(hue_var) == str) and (type(palette) == dict)):
            residual_traits = [trait for trait in _df[hue_var].unique() if
                               trait not in palette.keys()]
            # residual = set(_df[hue_var].unique()) - set(palette.keys())
            if len(residual_traits) > 0:
                print(residual_traits)

                # to get missing hue values to be missing value color (contrary to palette for available values)
        # yet be sized correctly, we plot all data with missing color, collect legend information,
        # then plot data with available hue over it, collect the legend information again and recompose the legend

        if type(hue_var) == str:
            sns.scatterplot(data=_df, x=x, y=y, hue=hue_var, size=size_var,
                            style=marker_var, palette=cycle([missing_d['hue']]), ax=ax, **scatter_kwargs)
            missing_handles, missing_labels = ax.get_legend_handles_labels()

            print(hue_var)
            hue_data = _df[_df[hue_var] != missing_val]
            sns.scatterplot(data=hue_data, x=x, y=y, hue=hue_var, size=size_var,
                            style=marker_var, palette=palette, ax=ax, **scatter_kwargs)
        else:
            scatter_kwargs['c'] = missing_d['hue']
            sns.scatterplot(data=_df, x=x, y=y, hue=hue_var, size=size_var,
                            style=marker_var, ax=ax, **scatter_kwargs)
            missing_handles, missing_labels = ax.get_legend_handles_labels()

        h, l = ax.get_legend_handles_labels()
        if len(l) == 0:
            legend = False

        if legend == True:
            han = copy.copy(h[0])
            han.set_alpha(0)
            blank_handle = han

            # find breakpoint between first plotting and overplotting
            breakpoint = len(missing_labels)
            available_handles = h[breakpoint:]
            available_labels = l[breakpoint:]

            for pair in [(available_handles, available_labels), (missing_handles, missing_labels)]:
                pair_h, pair_l = pair[0], pair[1]
                if len(pair_l) > 0:
                    if pair_l[0] not in trait_vars:
                        pair_l.insert(0, [trait_var for trait_var in set(trait_vars) if trait_var != None][0])
                        pair_h.insert(0, blank_handle)
            # legend has one section for each trait but ax.get_legend_handles_labels() yields straight lists
            # This code reorganizes legend information hierarchically starting with available and adding values
            # from missing as needed
            d_leg = {}

            for pair in [(available_handles, available_labels), (missing_handles, missing_labels)]:
                pair_h, pair_l = pair[0], pair[1]
                for ik, label in enumerate(pair_l):
                    if label in trait_vars:
                        key = label
                        if label not in d_leg.keys():
                            d_leg[key] = {'labels': [], 'handles': []}
                    else:
                        try:
                            # first pass at sig figs approach to number formatting
                            _label = np.format_float_positional(np.float16(pair_l[ik]), unique=True, precision=3)
                        except:
                            try:
                                _label = LipdToOntology(pair_l[ik])
                            except:
                                _label = pair_l[ik]
                        if _label not in d_leg[key]['labels']:
                            d_leg[key]['labels'].append(_label)
                            d_leg[key]['handles'].append(pair_h[ik])

            # Finally rebuild legend in single list with formatted section headers
            handles, labels = [], []
            for key in d_leg:
                han = copy.copy(h[0])
                han.set_alpha(0)
                if ((len(d_leg)==1) and ('label' in d_leg.keys()))==False:
                    handles.append(han)
                    labels.append('$\\bf{}$'.format('{' + key + '}'))

                tmp_labels, tmp_handles = [], []
                tmp_labels_missing, tmp_handles_missing = [], []
                for ik, label in enumerate(d_leg[key]['labels']):
                    if label == 'missing':
                        tmp_labels_missing.append(label)
                        tmp_handles_missing.append(d_leg[key]['handles'][ik])
                    else:
                        tmp_labels.append(label)
                        tmp_handles.append(d_leg[key]['handles'][ik])

                tmp_labels += tmp_labels_missing
                tmp_handles += tmp_handles_missing

                handles += tmp_handles
                labels += tmp_labels

                handles.append(han)
                labels.append('')
            if type(legend_kwargs) != dict:
                legend_kwargs = {}
            if 'loc' not in legend_kwargs:
                legend_kwargs['loc'] = 'upper left'
            if 'bbox_to_anchor' not in legend_kwargs:
                legend_kwargs['bbox_to_anchor'] = (1, 1)
            ax.legend(handles, labels, **legend_kwargs)  # loc="upper left", bbox_to_anchor=(1, 1))
        else:
            ax.legend().remove()

    if type(geos) != pd.DataFrame:
        df = make_df(geos, hue=hue, marker=marker, size=size)
    else:
        df = geos

    # newCrs = ccrs.Robinson()
    if proj_default is not True and type(proj_default) is not dict:
        raise TypeError('The default for the projections should either be provided' +
                        ' as a dictionary or set to True')

    # get the projection:
    proj = set_proj(projection=projection, proj_default=proj_default)
    if proj_default == True:
        converted = False
        if len(df[df['lon'] > 180]) > 0:
            converted = True
            mean_lon = np.mean(df['lon'].apply(lambda x: (x + 180) % 360 - 180))
        else:
            mean_lon = np.mean(df['lon'])

        proj1 = {'central_latitude': np.mean(df['lat']),
                 'central_longitude': mean_lon}
        proj2 = {'central_latitude': np.mean(df['lat'])}
        proj3 = {'central_longitude': mean_lon}

        # proj1 = {'central_latitude': np.mean(df['lat']),
        #          'central_longitude': np.mean(df['lon'])}
        # proj2 = {'central_latitude': np.mean(df['lat'])}
        # proj3 = {'central_longitude': np.mean(df['lon'])}
        try:
            proj = set_proj(projection=projection, proj_default=proj1)
        except:
            try:
                proj = set_proj(projection=projection, proj_default=proj3)
            except:
                proj = set_proj(projection=projection, proj_default=proj2)

    if fig == None:
        if figsize == None:
            figsize = (20, 7)
        fig = plt.figure(figsize=figsize)

    if gs_slot == None:
        ax = fig.add_subplot(projection=proj)
    else:
        ax = fig.add_subplot(gs_slot, projection=proj)

    # draw the coastlines
    ax.add_feature(cfeature.COASTLINE, linewidths=(1,))
    # Background
    if background is True:
        ax.stock_img()
    # Other extra information
    if borders is True:
        ax.add_feature(cfeature.BORDERS, alpha=.5)
    if lakes is True:
        ax.add_feature(cfeature.LAKES, alpha=0.25)
    if rivers is True:
        ax.add_feature(cfeature.RIVERS)
    if ocean is True:
        ax.add_feature(cfeature.OCEAN, alpha=.25)
    if land is True:
        ax.add_feature(cfeature.LAND, alpha=.5)

    # ax.set_global()

    x = 'lon'
    y = 'lat'

    if type(scatter_kwargs) != dict:
        scatter_kwargs = {}
    # print(edgecolor)
    # if 'edgecolor' not in scatter_kwargs:
    #     scatter_kwargs['edgecolor'] = edgecolor

    plot_scatter(df=df, x=x, y=y, hue_var=hue, size_var=size, marker_var=marker, ax=ax, proj=None,edgecolor=edgecolor,
                 cmap=cmap, scatter_kwargs=scatter_kwargs, legend=legend, legend_kwargs=legend_kwargs)  # , **kwargs)
    return ax


def dist_sphere(lat1, lon1, lat2, lon2):
    """Uses the haversine formula to calculate distance on a sphere
    https://en.wikipedia.org/wiki/Haversine_formula
    
    Parameters
    ----------
    lat1: float
        Latitude of the first point, in radians
    lon1: float
        Longitude of the first point, in radians
    lat2: float
        Latitude of the second point, in radians
    lon2: float
        Longitude of the second point, in radians
        
    Returns
    -------
    dist: float
        The distance between the two point in km
    """
    R = 6371  # km. Earth's radius
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R * c

    return float(dist)


def compute_dist(lat_r, lon_r, lat_c, lon_c):
    """ Computes the distance in (km) between a reference point and an array
    of other coordinates.
    
    Parameters
    ----------
    lat_r: float
        The reference latitude, in deg
    lon_r: float
        The reference longitude, in deg
    lat_c: list
        A list of latitudes for the comparison points, in deg
    lon_c: list
        A list of longitudes for the comparison points, in deg
    
    See also
    --------
    
    pyleoclim.utils.mapping.dist_sphere: calculate distance on a sphere
    
    Returns
    -------
    
    dist: list
        A list of distances in km.
    """
    dist = []

    for idx, val in enumerate(lat_c):
        lat1 = np.radians(lat_r)
        lon1 = np.radians(lon_r)
        lat2 = np.radians(val)
        lon2 = np.radians(lon_c[idx])
        dist.append(dist_sphere(lat1, lon1, lat2, lon2))

    return dist


def within_distance(distance, radius):
    """ Returns the index of the records that are within a certain distance
    
    Parameters
    ----------   
    
    distance: list
        A list containing the distance
        
    radius: float
        The radius to be considered
        
    Returns
    -------
    
    idx: list
        a list of index
    """
    idx = [idx for idx, val in enumerate(distance) if val <= radius]

    return idx
