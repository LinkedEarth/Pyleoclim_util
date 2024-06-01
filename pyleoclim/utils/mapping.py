#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapping utilities for geolocated objects, leveraging Cartopy.
"""
__all__ = ['map', 'compute_dist']

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import seaborn as sns
import copy
from itertools import cycle
# matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from .plotting import savefig, make_scalar_mappable, keep_center_colormap, consolidate_legends
from .lipdutils import PLOT_DEFAULT, LipdToOntology, CaseInsensitiveDict


def pick_proj(lat, lon, crit_dist=5000):
    '''
    Pick projection based on the degree of clustering of coordinates.
    At the moment, returns only one of two options:
        - 'Robinson' for R > crit_dist
        - 'Orthographic' for R <= crit_dist

    Parameters
    ----------
    lat : 1d array
        latitudes in [-90, 90]
    lon : 1d array
        longitudes in (-180, 180]
    crit_dist : float
        critical radius. Default: 5000 km

    Returns
    -------
    proj: str
        'Orthographic' or 'Robinson'

    '''
    lon = lon_360_to_180(lon)  # convert longitudes to [-180, 180]

    lat_c, lon_c = centroid_coords(lat, lon)  # find coordinates of centroid

    d = compute_dist(lat_c, lon_c, lat, lon)  # computes distances to centroid
    dmax = np.array(d).max()  # find maximum distance
    if dmax > crit_dist:
        proj = 'Robinson'
    else:
        proj = 'Orthographic'

    return proj


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

    See also
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
        projection='auto', proj_default=True, crit_dist=5000,
        background=True, borders=False, rivers=False, lakes=False,
        figsize=None, ax=None, scatter_kwargs=None, legend=True, legend_title=None,
        lgd_kwargs=None, savefig_settings=None):
    """ Map the location of all lat/lon according to some criteria

    DEPRECATED: use scatter_map() instead

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
        By default, projection == 'auto', so the projection will be picked
        based on the degree of clustering of the sites.

    proj_default : bool
        If True, uses the standard projection attributes.
        Enter new attributes in a dictionary to change them. Lists of attributes
        can be found in the `Cartopy documentation <https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv>`_.

    crit_dist : float
        critical radius for projection choice. Default: 5000 km
        Only active if projection == 'auto'

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

    See also
    --------

    pyleoclim.utils.mapping.set_proj : Set the projection for Cartopy-based maps
    pyleoclim.utils.mapping.pick_proj : pick the projection type based on the degree of clustering of coordinates
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
    if projection == 'auto':
        projection = pick_proj(lat, lon, crit_dist=crit_dist)

    proj = set_proj(projection=projection, proj_default=proj_default)
    if proj_default == True:
        clat, clon = centroid_coords(lat, lon)
        proj1 = {'central_latitude': clat,
                 'central_longitude': clon}
        proj2 = {'central_latitude': clat}
        proj3 = {'central_longitude': clon}
        proj4 = {}
        try:
            proj = set_proj(projection=projection, proj_default=proj1)
        except:
            try:
                proj = set_proj(projection=projection, proj_default=proj3)
            except:
                try:
                    proj = set_proj(projection=projection, proj_default=proj2)
                except:
                    proj = set_proj(projection=projection, proj_default=proj4)

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


def make_df(geo_ms, hue=None, marker=None, size=None, cols=None, d=None):
    try:
        geo_series_list = geo_ms.series_list
    except:
        geo_series_list = [geo_ms]

    lats = [geos.lat for geos in geo_series_list]
    lons = [geos.lon for geos in geo_series_list]

    traits = [hue, marker, size]
    if type(cols) == list:
        traits += cols
    value_d = {'lat': lats, 'lon': lons}

    for trait in traits:  # trait_d.keys():
        # trait = trait_d[trait_key]
        if trait != None:
            if trait == 'archiveType':
                trait_vals = []
                for geos in geo_series_list:
                    if trait in geos.__dict__.keys():
                        try:
                            trait_vals.append(LipdToOntology(geos.__dict__[trait]).lower().replace(" ", ""))
                        except:
                            trait_vals.append(None)
                    else:
                        trait_vals.append(None)
                # trait_vals = [LipdToOntology(geos.__dict__[trait]).lower().replace(" ","") if trait in geos.__dict__.keys() else None for geos in
                #          geo_series_list]
            else:
                trait_vals = [geos.__dict__[trait] if trait in geos.__dict__.keys() else None for geos in
                              geo_series_list]
            value_d[trait] = [trait_val if trait_val != 'None' else None for trait_val in trait_vals]

    geos_df = pd.DataFrame(value_d)
    if type(d) == dict:
        for trait in d.keys():
            if type(d[trait]) in [list, np.ndarray]:
                if len(d[trait]) == len(geos_df):
                    geos_df[trait] = d[trait]

    return geos_df


def scatter_map(geos, hue='archiveType', size=None, marker='archiveType', edgecolor='k',
                proj_default=True, projection='auto', crit_dist=5000,
                background=True, borders=False, coastline=True, rivers=False, lakes=False, ocean=True, land=True,
                figsize=None, scatter_kwargs=None, gridspec_kwargs=None, extent='global',
                lgd_kwargs=None, legend=True, colorbar=True, cmap=None, color_scale_type=None,
                fig=None, gs_slot=None, **kwargs):
    '''


    Parameters
    ----------
    geos : Pandas DataFrame, GeoSeries, MultipleGeoSeries, required
        If a Pandas DataFrame, expects 'lat' and 'lon' columns

    hue : string, optional
        Grouping variable that will produce points with different colors. Can be either categorical or numeric, although color mapping will behave differently in latter case.
        The default is 'archiveType'.

    size : string, optional
        Grouping variable that will produce points with different sizes. Expects to be numeric. Any data without a value for the size variable will be filtered out.
        The default is None.

    marker : string, optional
        Grouping variable that will produce points with different markers. Can have a numeric dtype but will always be treated as categorical.
        The default is 'archiveType'.

    edgecolor : color (string) or list of rgba tuples, optional
        Color of marker edge. The default is 'w'.

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
        By default, projection == 'auto', so the projection will be picked
        based on the degree of clustering of the sites.

    proj_default : bool, optional
        If True, uses the standard projection attributes.
        Enter new attributes in a dictionary to change them. Lists of attributes can be found in the `Cartopy documentation <https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv>`_.
        The default is True.

    crit_dist : float, optional
        critical radius for projection choice. Default: 5000 km
        Only active if projection == 'auto'

    background : bool, optional
        If True, uses a shaded relief background (only one available in Cartopy)

    borders : bool, optional
        Draws the countries border.
        Defaults is off (False).

    rivers : bool, optional
        Draws major rivers.
        Default is off (False).

    lakes : bool, optional
        Draws major lakes.
        Default is off (False).

    figsize : list or tuple, optional
        Size for the figure

    scatter_kwargs : dict, optional
        Dict of arguments available in `seaborn.scatterplot <https://seaborn.pydata.org/generated/seaborn.scatterplot.html>`_.
        Dictionary of arguments available in `matplotlib.pyplot.scatter <https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.scatter.html>`_.

    legend : bool, optional
        Whether the draw a legend on the figure.
        Default is True.

    colorbar : bool, optional
        Whether the draw a colorbar on the figure if the data associated with hue are numeric.
        Default is True.

    color_scale_type : str, optional
        Setting to "discrete" will force a discrete color scale with a default bin number of max(11, n) where n=number of unique values$^{\frac{1}{2}}$
        Default is None

    lgd_kwargs : dict, optional
        Dictionary of arguments for `matplotlib.pyplot.legend <https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.legend.html>`_.

    savefig_settings : dict, optional
        Dictionary of arguments for matplotlib.pyplot.saveFig.

         - "path" must be specified; it can be any existed or non-existed path,
           with or without a suffix; if the suffix is not given in "path", it will follow "format"
         - "format" can be one of {"pdf", "eps", "png", "ps"}

    extent : TYPE, optional
        DESCRIPTION.
        The default is 'global'.

    cmap : string or list, optional
        Matplotlib supported colormap id or list of colors for creating a colormap. See `choosing a matplotlib colormap <https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html>`_.
        The default is None.

    fig : matplotlib.pyplot.figure, optional
        See matplotlib.pyplot.figure <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>_.
        The default is None.

    gs_slot : Gridspec slot, optional
        If generating a map for a multi-plot, pass a gridspec slot.
        The default is None.

    gridspec_kwargs : dict, optional
        Function assumes the possibility of a colorbar, map, and legend. A list of floats associated with the keyword `width_ratios` will assume the first (index=0) is the relative width of the colorbar, the second to last (index = -2) is the relative width of the map, and the last (index = -1) is the relative width of the area for the legend.
        For information about Gridspec configuration, refer to `Matplotlib documentation <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.gridspec.GridSpec.html#matplotlib.gridspec.GridSpec>`_. The default is None.

    kwargs: dict, optional
        - 'missing_val_hue', 'missing_val_marker', 'missing_val_label' can all be used to change the way missing values are represented ('k', '?',  are default hue and marker values will be associated with the label: 'missing').
        - 'hue_mapping' and 'marker_mapping' can be used to submit dictionaries mapping hue values to colors and marker values to markers. Does not replace passing a string value for hue or marker.
        - 'scalar_mappable' can be used to pass a matplotlib scalar mappable. See pyleoclim.utils.plotting.make_scalar_mappable for documentation on using the Pyleoclim utility, or the `Matplotlib tutorial on customizing colorbars <https://matplotlib.org/stable/users/explain/colors/colorbar_only.html>`_.


    Returns
    -------
    TYPE
        fig, dictionary of ax objects which includes the as many as three items: 'cb' (colorbar ax), 'map' (scatter map), and 'leg' (legend ax)

    See also
    --------

    pyleoclim.utils.mapping.set_proj : Set the projection for Cartopy-based maps
    pyleoclim.utils.mapping.pick_proj : pick the projection type based on the degree of clustering of coordinates

    '''

    # def make_df(geo_ms, hue=None, marker=None, size=None, cols=None):
    #     try:
    #         geo_series_list = geo_ms.series_list
    #     except:
    #         geo_series_list = [geo_ms]
    #     lats = [geos.lat for geos in geo_series_list]
    #     lons = [geos.lon for geos in geo_series_list]
    #
    #     trait_d = {'hue': hue, 'marker': marker, 'size': size}
    #     traits = [hue, marker, size]
    #     if type(cols) == list:
    #         traits += cols
    #     value_d = {'lat': lats, 'lon': lons}
    #     for trait in traits:#trait_d.keys():
    #         # trait = trait_d[trait_key]
    #         if trait != None:
    #             trait_vals = [geos.__dict__[trait] if trait in geos.__dict__.keys() else None for geos in
    #                           geo_series_list]
    #             value_d[trait] = [trait_val if trait_val != 'None' else None for trait_val in trait_vals]
    #     geos_df = pd.DataFrame(value_d)
    #     return geos_df

    def plot_scatter(df=None, x=None, y=None, hue_var=None, size_var=None, marker_var=None, edgecolor='w',
                     ax=None, ax_d=None, proj=None, scatter_kwargs=None, legend=True, lgd_kwargs=None, colorbar=None,
                     fig=None, color_scale_type=None,  # gs_slot=None,
                     cmap=None, **kwargs):

        scatter_kwargs = {} if type(scatter_kwargs) != dict else scatter_kwargs
        lgd_kwargs = {} if type(lgd_kwargs) != dict else lgd_kwargs
        kwargs = {} if type(kwargs) != dict else kwargs
        norm_kwargs = kwargs.pop('norm_kwargs', {})
        ax_sm = kwargs.pop('scalar_mappable', None)

        palette = None
        hue_norm = None
        # if (color_scale_type is not None) and (colorbar is None):
        #         colorbar = True

        # plot_defaults = copy.copy(PLOT_DEFAULT)
        f = copy.copy(PLOT_DEFAULT)
        plot_defaults = CaseInsensitiveDict()
        for key, value in f.items():
            plot_defaults[key] = value

        _df = df
        if len(_df) == 1:
            _df = _df.reindex()

        ax_leg, ax_cb = None, None
        if type(ax_d) == dict:
            if 'map' in ax_d.keys():
                ax = ax_d['map']
            if 'cb' in ax_d.keys():
                ax_cb = ax_d['cb']
            if 'leg' in ax_d.keys():
                ax_leg = ax_d['leg']
        else:
            ax_d = {}
            if ax is None:
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot()

        transform = ccrs.PlateCarree()
        # if type(ax) == cartopy.mpl.geoaxes.GeoAxes:
        #     transform=ccrs.PlateCarree()
        #     if proj is not None:
        #         scatter_kwargs['transform'] = ccrs.PlateCarree()#proj
        #     else:
        #         scatter_kwargs['transform'] = ccrs.PlateCarree()

        missing_d = {'hue': kwargs['missing_val_hue'] if 'missing_val_hue' in kwargs else 'k',
                     'marker': kwargs['missing_val_marker'] if 'missing_val_marker' in kwargs else r'$?$',
                     'size': kwargs['missing_val_size'] if 'missing_val_size' in kwargs else 200,
                     'label': kwargs['missing_val_label'] if 'missing_val_label' in kwargs else 'missing',
                     }
        missing_val = missing_d['label']

        # if 'edgecolor' in scatter_kwargs:
        #     edgecolor = scatter_kwargs['edgecolor']
        #     scatter_kwargs['edgecolors'] = edgecolor
        # if 'edgecolors' not in scatter_kwargs:
        #     scatter_kwargs['edgecolors'] = edgecolor
        #
        if 'edgecolor' in scatter_kwargs:
            edgecolor = scatter_kwargs['edgecolor']# = edgecolor
        if isinstance(edgecolor, (list, np.ndarray)):
            _df['edgecolor'] = edgecolor

        hue_var = hue_var if hue_var in _df.columns else None
        hue_var_type_numeric = False
        if hue_var is not None:
            hue_var_type_numeric = all(isinstance(i, (int, float)) for i in _df[_df[hue_var] != missing_val][
                hue_var])  # pd.to_numeric(_df[hue_var], errors='coerce').notnull().all()

        marker_var = marker_var if marker_var in _df.columns else None
        marker_var_type_numeric = False
        if marker_var is not None:
            marker_var_type_numeric = pd.to_numeric(_df[marker_var], errors='coerce').notnull().all()

        size_var = size_var if size_var in _df.columns else None
        size_var_type_numeric = False
        if size_var is not None:
            size_var_type_numeric = all(isinstance(i, (int, float)) for i in _df[_df[size_var] != missing_val][
                size_var])  # pd.to_numeric(_df[size_var], errors='coerce').notnull().all()

        trait_vars = [trait_var for trait_var in [hue_var, marker_var, size_var] if
                      ((trait_var != None) and (trait_var in _df.columns))]

        for trait_var in trait_vars:
            if type(_df[trait_var]) == pd.Series:
                _df[trait_var] = _df[trait_var].fillna(missing_val)
            else:
                if _df[trait_var] in [None, 'None']:
                    _df[trait_var] = missing_val

        if size_var is None:
            scatter_kwargs['s'] = scatter_kwargs['s'] if 's' in scatter_kwargs else missing_d['size']
        else:
            sizes = [size_val for size_val in _df[size_var].values if size_val != missing_val]
            if len(sizes) < 2:
                scatter_kwargs['s'] = scatter_kwargs['s'] if 's' in scatter_kwargs else missing_d['size']
                size_var = None
            else:
                # if size does vary, filter out missing values; at present no strategy to depict missing size values
                _df = _df[_df[size_var] != missing_val]
                scatter_kwargs['sizes'] = (20, 200) if 'sizes' not in scatter_kwargs else scatter_kwargs['sizes']
                scatter_kwargs['size_norm'] = scatter_kwargs['size_norm'] if 'size_norm' in scatter_kwargs else (
                    _df[size_var].min(), _df[size_var].max())
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
            # with missing values assigned '?'
            trait2marker['missing'] = missing_d['marker']
            scatter_kwargs['markers'] = trait2marker
            residual_traits = [trait for trait in _df[marker_var].unique() if
                               trait not in trait2marker.keys()]
            if len(residual_traits) > 0:
                print(residual_traits)

        # use hue mapping if supplied
        if 'hue_mapping' in kwargs:
            palette = kwargs['hue_mapping']
        # there should be different control for discrete and continuous hue
        elif hue_var == 'archiveType':
            palette = {key: value[0] for key, value in plot_defaults.items()}
        elif isinstance(hue_var,str): #hue_var) == str:
            hue_data = _df[_df[hue_var] != missing_val]
            # If scalar mappable was passed, try to extract components.
            if ax_sm is not None:
                try:
                    palette = ax_sm.cmap
                except:
                    ax_sm = None  # if can't extract a palette, the scalar mappable is not helpful so set to None to trigger normal flow
                try:
                    hue_norm = ax_sm.norm
                except:
                    hue_norm = None

            if ax_sm is None:
                if cmap != None:
                    palette = cmap
                if len(hue_data[hue_var]) > 0:
                    if hue_var_type_numeric is not True:
                        # trait_val_types = [True if type(val) in (np.str_, str) else False for val in hue_data[hue_var]]
                        # if True in trait_val_types:
                        colorbar = False
                        if len(hue_data[hue_var].unique()) < 20:
                            palette = 'tab20'
                        else:
                            if palette is None:
                                palette = 'viridis'
                            ax_sm = make_scalar_mappable(cmap=palette, hue_vect=None, n=len(hue_data[hue_var].unique()),
                                                         norm_kwargs=norm_kwargs)
                            palette = ax_sm.cmap
                    else:
                        if ((type(palette) in [str, list]) or (palette is None)):
                            if color_scale_type == 'discrete':
                                n = max(10, int(np.ceil(np.sqrt(len(hue_data[hue_var].unique())))))
                                ax_sm = make_scalar_mappable(cmap=palette, hue_vect=hue_data[hue_var], n=n,
                                                             norm_kwargs=norm_kwargs)
                            else:
                                ax_sm = make_scalar_mappable(cmap=palette, hue_vect=hue_data[hue_var],
                                                             norm_kwargs=norm_kwargs)
                            palette = ax_sm.cmap
                            hue_norm = ax_sm.norm  # .autoscale(hue_data[hue_var])

        if ((type(hue_var) == str) and (type(palette) == dict)):
            residual_traits = [trait for trait in _df[hue_var].unique() if
                               trait not in palette.keys()]
            if len(residual_traits) > 0:
                print(residual_traits)

        # to get missing hue values to be missing value color (contrary to palette for available values)
        # we plot all data with missing color, collect legend information,
        # then plot data with available hue over it, collect the legend information again and recompose the legend
        if type(hue_var) == str:
            scatter_kwargs['zorder'] = 13
            if isinstance(edgecolor, np.ndarray):
                _df['edgecolor'] = edgecolor
                _df['neighbor'] = _df['edgecolor'].map({'k': 'target', 'w': 'neighbor'})

            hue_data = _df[_df[hue_var] == missing_val]

            if len(hue_data) > 0:
                sns.scatterplot(data=hue_data, x=x, y=y, hue=hue_var, size=size_var,
                                style=marker_var, transform=transform,edgecolor='w',
                                # change to transform=scatter_kwargs['transform']
                                palette=[missing_d['hue'] for ik in range(len(hue_data))],
                                ax=ax, **scatter_kwargs)
                missing_handles, missing_labels = ax.get_legend_handles_labels()
                if 'neighbor' in hue_data.columns:
                    if len(hue_data[hue_data['neighbor'] != 'neighbor']) > 1:
                        _edgecolor = hue_data[hue_data['neighbor'] != 'neighbor']['edgecolor'].values[0]
                    else:
                        _edgecolor = hue_data[hue_data['neighbor'] != 'neighbor']['edgecolor']
                    sns.scatterplot(data=hue_data[hue_data['neighbor'] != 'neighbor'], x=x, y=y, size=size_var,
                                    transform=transform, edgecolor=_edgecolor,
                                    style=marker_var, hue=hue_var, palette=palette, ax=ax, **scatter_kwargs)
            else:
                missing_handles, missing_labels = [], []

            scatter_kwargs['zorder'] = 14
            hue_data = _df[_df[hue_var] != missing_val]
            if hue_norm is not None:
                scatter_kwargs['hue_norm'] = hue_norm

            sns.scatterplot(data=hue_data, x=x, y=y, hue=hue_var, size=size_var, transform=transform,edgecolor='w',
                            style=marker_var, palette=palette, ax=ax, **scatter_kwargs)
            if 'neighbor' in hue_data.columns:
                sns.scatterplot(data=hue_data[hue_data['neighbor'] != 'neighbor'], x=x, y=y, size=size_var,
                            transform=transform, edgecolor=hue_data[hue_data['neighbor'] != 'neighbor']['edgecolor'].values[0],
                            style=marker_var, hue=hue_var, palette=palette, ax=ax, **scatter_kwargs)

        else:
            scatter_kwargs['zorder'] = 13
            scatter_kwargs['c'] = missing_d['hue']
            sns.scatterplot(data=_df, x=x, y=y, hue=hue_var, size=size_var, transform=transform,
                            # change to transform=scatter_kwargs['transform']
                            style=marker_var, ax=ax, **scatter_kwargs)
            missing_handles, missing_labels = ax.get_legend_handles_labels()

        # h, l= consolidate_legends([ax], hue=hue_var, style =marker_var, size=size_var, colorbar=colorbar)
        h, l = ax.get_legend_handles_labels()

        if ((len(l) == 2) and (l[-1] == 'missing')) or (len(l) < 2):
            legend = False

        # if legend is True, prep legend content
        d_leg = {}
        if legend is True:
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
                            _label = np.format_float_positional(np.float16(pair_l[ik]), unique=True, precision=2)
                        except:
                            try:
                                _label = LipdToOntology(pair_l[ik])
                            except:
                                _label = pair_l[ik]
                        if _label not in d_leg[key]['labels']:
                            d_leg[key]['labels'].append(_label)
                            d_leg[key]['handles'].append(pair_h[ik])

            # if colorbar is True, hue will be removed from legend content, or removed entirely if there is no other
            if ((colorbar is True) and (hue_var in d_leg.keys()) and (ax_sm is not None)):
                d_leg.pop(hue_var, None)
                # if len(d_leg.keys()) == 1:
                #     # ax.legend().remove()
                #     legend = False
                #     # ax_leg.remove()
                #     # ax_d.pop('leg', None)
                # else:
                #     d_leg.pop(hue_var, None)

        if (legend is True) and (len(d_leg.keys()) > 0):
            # Finally rebuild legend in single list with formatted section headers
            handles, labels = [], []
            headers = True
            if ((len(d_leg) == 1) and ('label' in d_leg.keys())):
                headers = False
            headers = lgd_kwargs.pop('headers', headers)

            for key in d_leg.keys():
                han = copy.copy(h[0])
                han.set_alpha(0)
                if headers is True:
                    handles.append(han)
                    # labels.append('$\\bf{}$'.format('{' + key + '}'))
                    labels.append(key)

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

            if 'loc' not in lgd_kwargs:
                lgd_kwargs['loc'] = 'upper left'
            if 'bbox_to_anchor' not in lgd_kwargs:
                lgd_kwargs['bbox_to_anchor'] = (-.1, 1)  # (1, 1)

            built_legend = ax_leg.legend(handles, labels, **lgd_kwargs)
            if headers is True:
                for _text in built_legend.get_texts():
                    if _text._text in d_leg.keys():
                        _text.set_weight('bold')

                ax_leg.add_artist(built_legend)

            ax_leg.set_axis_off()
            ax.legend().remove()
            # if colorbar == False:
            #     ax_cb.remove()
            #     ax_d.pop('cb', None)

        else:
            ax.legend().remove()
            # ax_d.pop('cb', None)
            ax_d.pop('leg', None)
            ax_leg.remove()
            # for _ax in [ax_cb, ax_leg]:
            #     if type(_ax) != None:
            #         _ax.remove()

        # if colorbar is true, make colorbar
        if ((colorbar is True) and (ax_sm is not None)):
            # make colorbar
            if ax_cb is not None:
                plt.colorbar(ax_sm, cax=ax_cb, orientation='vertical', label=hue_var,
                             fraction=.6,
                             shrink=.4)
                ax_cb.yaxis.set_label_position('left')
                ax_cb.yaxis.set_ticks_position('left')
            else:
                plt.colorbar(ax_sm, ax=ax, orientation='vertical', label=hue_var,
                             shrink=.6)  # ,label=colorbar_units,
        elif colorbar is False:
            ax_cb.remove()
            ax_d.pop('cb', None)

        # a little squirrely that this function might return different types
        if len(ax_d) > 0:  # == dict:
            if 'map' in ax_d.keys():
                ax_d['map'] = ax
            if 'cb' in ax_d.keys():
                ax_d['cb'] = ax_cb
            if 'leg' in ax_d.keys():
                ax_d['leg'] = ax_leg
            return fig, ax_d
        else:
            return fig, {'map': ax}

    ###### End of plot_scatter

    # from ..core.multiplegeoseries import MultipleGeoSeries
    # from ..core.geoseries import GeoSeries

    # if geos is not
    if type(geos) != pd.DataFrame:  # in [MultipleGeoSeries, GeoSeries]:
        df = make_df(geos, hue=hue, marker=marker, size=size)
    elif type(geos) == pd.DataFrame:
        df = geos
        if hue not in df.columns:
            hue = None
        if marker not in df.columns:
            marker = None

    gridspec_kwargs = {} if type(gridspec_kwargs) != dict else gridspec_kwargs
    scatter_kwargs = {} if type(scatter_kwargs) != dict else scatter_kwargs
    if 'marker_var' in scatter_kwargs:
        marker_var = scatter_kwargs.pop('marker_var')
    # scatter_kwargs['transform'] = ccrs.PlateCarree()
    lgd_kwargs = {} if type(lgd_kwargs) != dict else lgd_kwargs

    if proj_default is not True and type(proj_default) is not dict:
        raise TypeError('The default for the projections should either be provided' +
                        ' as a dictionary or set to True')

    # get the projection
    if projection == 'auto':
        projection = pick_proj(df['lat'].values,
                               df['lon'].values, crit_dist=crit_dist)
        if figsize == None:
            if projection == 'Robinson':
                figsize = (18, 6)
            if projection == 'Orthographic':
                figsize = (16, 6)

    # set the projection
    proj = set_proj(projection=projection, proj_default=proj_default)
    if proj_default == True:
        clat, clon = centroid_coords(df['lat'].values, df['lon'].values)
        proj1 = {'central_latitude': clat,
                 'central_longitude': clon}
        proj2 = {'central_latitude': clat}
        proj3 = {'central_longitude': clon}

        try:
            proj = set_proj(projection=projection, proj_default=proj1)
        except:
            try:
                proj = set_proj(projection=projection, proj_default=proj3)
            except:
                proj = set_proj(projection=projection, proj_default=proj2)

    if fig == None:
        if figsize == None:
            figsize = (18, 7)
        fig = plt.figure(figsize=figsize)

    ax_d = {}

    # use subgridspecs to encourage the slot for the map to have an aspect ratio closer to that of the projection
    if gs_slot == None:
        _gs = gridspec.GridSpec(1, 1)  # , **gridspec_kwargs)
        gs_slot = _gs[0]
    if projection == 'Robinson':
        gs_sub = gs_slot.subgridspec(1, 3)
        gs_subslot = gs_sub[0, :]
    elif projection == 'Orthographic':
        gs_sub = gs_slot.subgridspec(3, 6)
        gs_subslot = gs_sub[:, 1:5]
    else:
        gs_subslot = gs_slot

    num_subplots = 1 + legend + colorbar
    if 'width_ratios' in gridspec_kwargs:
        if len(gridspec_kwargs['width_ratios']) < num_subplots:
            print('Please respecify gridspec width_ratios. Reverting to defaults.')
            gridspec_kwargs['width_ratios'] = [.7, 16, 6]
    else:
        gridspec_kwargs['width_ratios'] = [.7, 16, 6]

    gridspec_kwargs['width_ratios'] = gridspec_kwargs['width_ratios'] if 'width_ratios' in gridspec_kwargs else [.7,
                                                                                                                 .05,
                                                                                                                 16, 5]
    gs = gs_subslot.subgridspec(1, len(gridspec_kwargs['width_ratios']), **gridspec_kwargs)

    ax_d['cb'] = fig.add_subplot(gs[0])
    ax_d['map'] = fig.add_subplot(gs[-2], projection=proj)
    ax_d['leg'] = fig.add_subplot(gs[-1])

    # draw the coastlines
    # ax_d['map'].add_feature(cfeature.COASTLINE, linewidths=(.5,))
    # Background
    if background is True:
        ax_d['map'].stock_img()

    # Other extra information
    feature_d = {'borders': borders, 'lakes': lakes, 'rivers': rivers, 'land': land, 'ocean': ocean,
                 'coastline': coastline}
    feature_d = {key: (value if type(value) == dict else {}) for key, value in feature_d.items() if value != False}
    feature_spec_defaults = {'borders': dict(alpha=.5, linewidths=(.5,)), 'lakes': dict(alpha=0.25),
                             'rivers': {}, 'land': dict(alpha=0.5), 'ocean': dict(alpha=0.25),
                             'coastline': dict(linewidths=(.5,))}
    for key in feature_d.keys():
        feature_spec_defaults[key].update(feature_d[key])

    feature_types = {'borders': cfeature.BORDERS, 'coastline': cfeature.COASTLINE, 'lakes': cfeature.LAKES,
                     'rivers': cfeature.RIVERS, 'land': cfeature.LAND, 'ocean': cfeature.OCEAN}
    for feature in feature_d.keys():
        ax_d['map'].add_feature(feature_types[feature], **feature_spec_defaults[feature])

    if extent == 'global':
        ax_d['map'].set_global()
    elif isinstance(extent, list) and len(extent) == 4:
        ax_d['map'].set_extent(extent, crs=ccrs.PlateCarree())

    x = 'lon'
    y = 'lat'
    _, ax_d = plot_scatter(df=df, x=x, y=y, hue_var=hue, size_var=size, marker_var=marker, ax_d=ax_d, proj=None,
                           edgecolor=edgecolor, colorbar=colorbar, color_scale_type=color_scale_type,
                           cmap=cmap, scatter_kwargs=scatter_kwargs, legend=legend, lgd_kwargs=lgd_kwargs,
                           **kwargs)  # , **kwargs)
    return fig, ax_d


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

    Returns
    -------
    dist: list
        A list of distances in km.

    See also
    --------

    pyleoclim.utils.mapping.dist_sphere: calculate distance on a sphere


    """
    dist = []
    lon_c = lon_360_to_180(np.array(lon_c))
    lon_r = lon_360_to_180(np.array(lon_r))
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


def lon_360_to_180(x):
    return (x + 180) % 360 - 180


def lon_180_to_360(x):
    return x % 360


def centroid_coords(lat, lon, true_centroid=False):
    '''
    Computes the centroid of the geographic coordinates via Shapely.
    
    If there aren't enough vertices to form a polygon (4), then the arithmetic
    mean of the coordinates is returned.
    
    h/t Tim Roberts, via StackOverflow: https://stackoverflow.com/a/72737621.

    Parameters
    ----------
    lat : 1d array
       latitudes in [-90, 90]
    lon : 1d array
       longitudes in (-180, 180]
    true_centroid : boolean
        if True, computes a true centroid, otherwise a representative point,
        which is guaranteed to lie within the polygon.

    Returns
    -------
    clat, clon : coordinates of the centroid

    See also
    --------

    https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html#shapely.Polygon


    '''
    lon = lon_360_to_180(np.array(lon))
    lat = np.array(lat)
    if len(lon) >= 4:
        p = Polygon([(x, y) for (x, y) in zip(lon, lat)])
        if true_centroid:
            clat = p.centroid.y;
            clon = p.centroid.x
        else:
            clat = p.representative_point().y
            clon = p.representative_point().x
    else:
        clat = lat.mean()
        clon = lon.mean()
    return clat, clon
