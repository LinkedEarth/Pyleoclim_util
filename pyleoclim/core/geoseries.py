"""
The GeoSeries class is a child of Series, with additional metadata latitude (lat) and longitude (lon)
This unlocks plotting capabilities like map() and dashboard(). 
"""
from ..utils import plotting, mapping, lipdutils, jsonutils, tsbase, tsutils
from ..core.series import Series

import matplotlib.pyplot as plt
import re
import pandas as pd

#from copy import deepcopy
from matplotlib import gridspec
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature

import warnings


class GeoSeries(Series):
    '''The GeoSeries class is a child of the Series class, and requires geolocation
    information (latitude, longitude). Elevation is optional, but can be used in mapping, if present.
    The class also allows for ancillary data and metadata, detailed below. 
    
    Parameters
    ----------
    time : list or numpy.array
        independent variable (t)

    value : list of numpy.array
        values of the dependent variable (y)
        
    lat : float
        latitude N in decimal degrees. Must be in the range [-90;+90]
        
    lon : float
        longitude East in decimal degrees. Must be in the range [-180;+360]
        No conversion is applied as mapping utilities convert to [-180,+180] internally
       
    elevation : float
        elevation of the sample, in meters above sea level. Negative numbers indicate depth below global mean sea level, therefore.                                                                                          

    time_unit : string
        Units for the time vector (e.g., 'years').
        Default is None

    time_name : string
        Name of the time vector (e.g., 'Time','Age').
        Default is None. This is used to label the time axis on plots

    value_name : string
        Name of the value vector (e.g., 'temperature')
        Default is None

    value_unit : string
        Units for the value vector (e.g., 'deg C')
        Default is None

    label : string
        Name of the time series (e.g., 'Nino 3.4')
        Default is None

    log : dict
        Dictionary of tuples documentating the various transformations applied to the object
        
    keep_log : bool
        Whether to keep a log of applied transformations. False by default
        
    importedFrom : string
        source of the dataset. If it came from a LiPD file, this could be the datasetID property 

    archiveType : string
        climate archive, one of 'Borehole', 'Coral', 'FluvialSediment', 'GlacierIce', 'GroundIce', 'LakeSediment', 'MarineSediment', 'Midden', 'MolluskShell', 'Peat', 'Sclerosponge', 'Shoreline', 'Speleothem', 'TerrestrialSediment', 'Wood'                                                                                   
        Reference: https://lipdverse.org/vocabulary/archivetype/
    
    control_archiveType  : [True, False]
        Whether to standardize the name of the archiveType agains the vocabulary from: https://lipdverse.org/vocabulary/paleodata_proxy/. 
        If set to True, will only allow for these terms and automatically convert known synonyms to the standardized name. Only standardized variable names will be automatically assigned a color scheme.  
        Default is False. 
        
    sensorType : string
        sensor, e.g. a paleoclimate proxy sensor. This property can be used to differentiate between species of foraminifera
        
    observationType : string
        observation type,  e.g. a proxy observation. See https://lipdverse.org/vocabulary/paleodata_proxy/. Note: this is preferred terminology but not enforced
        
    depth : array
        depth at which the values were collected
        
    depth_name : string
        name of the field, e.g. 'mid-depth', 'top-depth', etc   
        
    depth_unit : string
         units of the depth axis, e.g. 'cm'

    dropna : bool
        Whether to drop NaNs from the series to prevent downstream functions from choking on them
        defaults to True
        
    sort_ts : str
        Direction of sorting over the time coordinate; 'ascending' or 'descending'
        Defaults to 'ascending'
        
    verbose : bool
        If True, will print warning messages if there is any
        
    clean_ts : boolean flag
         set to True to remove the NaNs and make time axis strictly prograde with duplicated timestamps reduced by averaging the values
         Default is None (marked for deprecation)
    
    auto_time_params : bool, 
        If True, uses tsbase.disambiguate_time_metadata to ensure that time_name and time_unit are usable by Pyleoclim. This may override the provided metadata. 
        If False, the provided time_name and time_unit are used. This may break some functionalities (e.g. common_time and convert_time_unit), so use at your own risk.
        If not provided, code will set to True for internal consistency.

    Examples
    --------

    Import the EPICA Dome C deuterium record and display a quick synopsis:

     .. jupyter-execute::

         import pyleoclim as pyleo
         ts = pyleo.utils.datasets.load_dataset('EDC-dD')
         ts_interp = ts.convert_time_unit('kyr BP').interp(step=.5) # interpolate for a faster result
         fig, ax = ts_interp.dashboard()


    '''

    def __init__(self, time, value, lat, lon, elevation = None, time_unit=None, time_name=None, 
                 value_name=None, value_unit=None, label=None, importedFrom=None, 
                 archiveType = None, control_archiveType = False, 
                 sensorType = None, observationType = None,
                 log=None, keep_log=False, verbose=True,
                 depth = None, depth_name = None, depth_unit= None,
                 sort_ts = 'ascending', dropna = True,  clean_ts=False, auto_time_params = None):
        
        if auto_time_params is None:
            auto_time_params = True
            if verbose:
                warnings.warn('auto_time_params is not specified. Currently default behavior sets this to True, which might modify your supplied time metadata.  Please set to False if you want a different behavior.', UserWarning, stacklevel=2)

        if auto_time_params:
            # assign time metadata if they are not provided or provided incorrectly
            offending = [tsbase.MATCH_CE, tsbase.MATCH_BP]

            if time_unit is None:
                time_unit='years CE'
                if verbose:
                    warnings.warn(f'No time_unit parameter provided. Assuming {time_unit}.', UserWarning, stacklevel=2)
            elif time_unit.lower().replace(".","") in frozenset().union(*offending):
                # fix up time name and units for offending cases
                time_name, time_unit = tsbase.disambiguate_time_metadata(time_unit)
            else:
                # give a proper time name to those series that confuse that notion with time units
                time_name, _ = tsbase.disambiguate_time_metadata(time_unit)

            if time_name is None:
                if verbose:
                    warnings.warn('No time_name parameter provided. Assuming "Time".', UserWarning, stacklevel=2)
                time_name='Time'
            elif time_name in tsbase.MATCH_A:
                if verbose:
                    warnings.warn(f'{time_name} refers to the units, not the name of the axis. Picking "Time" instead', UserWarning, stacklevel=2)
                time_name='Time'
        else:
            pass
       
        # assign latitude
        if lat is not None:
            lat = float(lat) 
            if -90 <= lat <= 90: 
                self.lat = lat
            else:
                ValueError('Latitude must be a number in [-90; 90]')
        else:
            self.lat = None 
            
        # assign longitude
        if lon is not None:
            lon = float(lon)
            if -180 < lon <= 360:     
                self.lon = lon
            # elif 180 < lon <= 360:
            #     self.lon = mapping.lon_360_to_180(lon)
            #     if verbose:
            #         print('Longitude has been converted to the [-180,+180] range')
            else:
                ValueError('Longitude must be a number in [-180,360]')
        else:
            self.lon = None 
            
        # elevation
        self.elevation = elevation
        
        # PSM 
        self.sensorType = sensorType
        self.observationType = observationType
                
        # depth infornation
        self.depth = depth
        self.depth_name = depth_name
        self.depth_unit = depth_unit
            
        #assign all the rest
        super().__init__(time, value, time_unit, time_name, value_name,
                         value_unit, label, importedFrom, archiveType, control_archiveType,
                         log, keep_log, sort_ts, dropna, verbose, clean_ts)
                      

    @property
    def metadata(self):
        return dict(
            lat = self.lat,
            lon = self.lon,
            elevation = self.elevation,
            time_unit = self.time_unit,
            time_name = self.time_name,
            value_unit = self.value_unit,
            value_name = self.value_name,
            label = self.label,
            archiveType = self.archiveType,
            sensorType  = self.sensorType,
            observationType = self.observationType, 
            importedFrom = self.importedFrom,
            control_archiveType = self.control_archiveType,
            log = self.log,
        )
    
    @classmethod    
    def from_json(cls, path):
        ''' Creates a pyleoclim.Series from a JSON file
        
        The keys in the JSON file must correspond to the parameter associated with a GeoSeries object

        Parameters
        ----------
        path : str
            Path to the JSON file

        Returns
        -------
        ts : pyleoclim.core.series.Series
            A Pyleoclim Series object. 

        '''
        
        a = jsonutils.open_json(path)
        b = jsonutils.iterate_through_dict(a, 'GeoSeries')
        
        return cls(**b)
    
    @classmethod
    def from_Series(lat, lon, elevation=None,sensorType=None,observationType=None, 
                    depth=None, depth_name=None, depth_unit=None):
        
        print('a')
        # time, value, lat, lon, elevation = None, time_unit=None, time_name=None, 
        #              value_name=None, value_unit=None, label=None, importedFrom=None, 
        #              archiveType = None, control_archiveType = False, 
        #              sensorType = None, observationType = None,
        #              log=None, keep_log=False, verbose=True,
        #              depth = None, depth_name = None, depth_unit= None,
        #              sort_ts = 'ascending', dropna = True,  clean_ts=False, auto_time_params = None

    
    def map(self, projection='Orthographic', proj_default=True,
            background=True, borders=False, coastline=True, rivers=False, lakes=False, ocean=True,
            land=True, fig=None, gridspec_slot=None,
            figsize=None, marker='archiveType', hue='archiveType', size=None, edgecolor='w',
            markersize=None, scatter_kwargs=None, cmap=None, colorbar=False, gridspec_kwargs=None,
            legend=True, lgd_kwargs=None, savefig_settings=None):
        
        ''' Map the location of the record

        Parameters
        ----------
        projection : str, optional
            The projection to use. The default is 'Orthographic'.

        proj_default : bool; {True, False}, optional
            Whether to use the Pyleoclim defaults for each projection type. The default is True.

        background : bool, optional
            If True, uses a shaded relief background (only one available in Cartopy)
            Default is on (True).

        borders : bool or dict, optional
            Draws the countries border.
            If a dictionary of formatting arguments is supplied (e.g. linewidth, alpha), will draw according to specifications.
            Defaults is off (False).

        coastline : bool or dict, optional
            Draws the coastline.
            If a dictionary of formatting arguments is supplied (e.g. linewidth, alpha), will draw according to specifications.
            Defaults is on (True).

        land : bool or dict, optional
            Colors land masses.
            If a dictionary of formatting arguments is supplied (e.g. color, alpha), will draw according to specifications.
            Default is off (True). Overriden if background=True.

        ocean : bool or dict, optional
            Colors oceans.
            If a dictionary of formatting arguments is supplied (e.g. color, alpha), will draw according to specifications.
            Default is on (True). Overriden if background=True.

        rivers : bool or dict, optional
            Draws major rivers.
            If a dictionary of formatting arguments is supplied (e.g. linewidth, alpha), will draw according to specifications.
            Default is off (False).

        lakes : bool or dict, optional
            Draws major lakes.
            If a dictionary of formatting arguments is supplied (e.g. color, alpha), will draw according to specifications.
            Default is off (False).

        figsize : list or tuple, optional
            The size of the figure. The default is None.

        marker : str, optional
            The marker type for each archive.
            The default is None. Uses plot_default

        hue : str, optional
            Variable associated with color coding.
            The default is None. Uses plot_default.

        markersize : float, optional
            Size of the marker. The default is None.

        scatter_kwargs : dict, optional
            Parameters for the scatter plot. The default is None.

        legend :  bool; {True, False}, optional
            Whether to plot the legend. The default is True.

        lgd_kwargs : dict, optional
            Arguments for the legend. The default is None.

        savefig_settings : dict, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.

        Returns
        -------
        res : fig,ax_d

        See also
        --------

        pyleoclim.utils.mapping.scatter_map : Underlying mapping function for Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            ts = pyleo.utils.datasets.load_dataset('EDC-dD')
            fig, ax = ts.map()


        '''
        if markersize != None:
            scatter_kwargs['markersize'] = markersize

        fig, ax_d = mapping.scatter_map(self, hue=hue, size=size, marker=marker, projection=projection,
                    proj_default=proj_default,
                    background=background, borders=borders, rivers=rivers, lakes=lakes,
                    ocean=ocean, land=land, coastline=coastline,
                    figsize=figsize, scatter_kwargs=scatter_kwargs, gridspec_kwargs=gridspec_kwargs,
                    lgd_kwargs=lgd_kwargs, legend=legend, colorbar=colorbar,
                    cmap=cmap, edgecolor=edgecolor,
                    fig=fig, gs_slot=gridspec_slot)
        return fig, ax_d
    
    def map_neighbors(self, mgs, radius=3000, projection='Orthographic', proj_default=True,
            background=True, borders=False, rivers=False, lakes=False, ocean=True,
            land=True, fig=None, gridspec_slot=None,
            figsize=None, marker='archiveType', hue='archiveType', size=None, edgecolor='w',
            markersize=None, scatter_kwargs=None, cmap=None, colorbar=False, gridspec_kwargs=None,
            legend=True, lgd_kwargs=None, savefig_settings=None):
        
        '''Map all records within a given radius of the object

        Parameters
        ----------
        mgs : MultipleGeoSeries
            object containing the series to be considered as neighbors
            
        radius : float
            search radius for the record, in km. Default is 3000. 
            
        projection : str, optional
            The projection to use. The default is 'Orthographic'.

        proj_default : bool; {True, False}, optional
            Whether to use the Pyleoclim defaults for each projection type. The default is True.

        background :  bool; {True, False}, optional
            Whether to use a background. The default is True.

        borders :  bool; {True, False}, optional
            Draw borders. The default is False.

        rivers :  bool; {True, False}, optional
            Draw rivers. The default is False.

        lakes :  bool; {True, False}, optional
            Draw lakes. The default is False.

        figsize : list or tuple, optional
            The size of the figure. The default is None.

        marker : str, optional
            The marker type for each archive.
            The default is None. Uses plot_default

        hue : str, optional
            Variable associated with color coding.
            The default is None. Uses plot_default.

        markersize : float, optional
            Size of the marker. The default is None.

        scatter_kwargs : dict, optional
            Parameters for the scatter plot. The default is None.

        legend :  bool; {True, False}, optional
            Whether to plot the legend. The default is True.

        lgd_kwargs : dict, optional
            Arguments for the legend. The default is None.

        savefig_settings : dict, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.

        Returns
        -------
        res : fig,ax_d

        See also
        --------

        pyleoclim.utils.mapping.map : Underlying mapping function for Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            from pylipd.utils.dataset import load_dir
            lipd = load_dir(name='Pages2k')
            df = lipd.get_timeseries_essentials()
            dfs = df.query("archiveType in ('Wood','Documents','Coral','Lake sediment') and paleoData_variableName not in ('year')")
            
            # place in a MultipleGeoSeries object
            ts_list = []
            for _, row in dfs.iterrows():
                ts_list.append(pyleo.GeoSeries(time=row['time_values'],value=row['paleoData_values'],
                                                time_name=row['time_variableName'],value_name=row['paleoData_variableName'],
                                                time_unit=row['time_units'], value_unit=row['paleoData_units'],
                                                lat = row['geo_meanLat'], lon = row['geo_meanLon'],
                                                archiveType = row['archiveType'], verbose = False, 
                                                label=row['dataSetName']+'_'+row['paleoData_variableName'])) 

            mgs = pyleo.MultipleGeoSeries(series_list=ts_list,time_unit='years AD') 
            gs = ts_list[6] # extract one record as the target one
            gs.map_neighbors(mgs, radius=4000)
            
        '''
        from ..core.multiplegeoseries import MultipleGeoSeries
        if markersize != None:
            scatter_kwargs['markersize'] = markersize
            
        # find neighbors
        lats = [ts.lat for ts in mgs.series_list]
        lons = [ts.lon for ts in mgs.series_list]
        dist = mapping.compute_dist(self.lat, self.lon, lats, lons)
        neigh_idx = mapping.within_distance(dist, radius)

        neighbors =[mgs.series_list[i] for i in neigh_idx if i !=0]
        neighbors = MultipleGeoSeries(neighbors)

        df = mapping.make_df(neighbors, hue=hue, marker=marker, size=size)
        df_self = mapping.make_df(self, hue=hue, marker=marker, size=size)

        neighborhood = pd.concat([df, df_self], axis=0)
        # additional columns are added manually
        neighbor_coloring = ['w' for ik in range(len(neighborhood))]
        neighbor_coloring[-1] = 'k'
        neighborhood['original'] =neighbor_coloring
        # plot neighbors

        fig, ax_d = mapping.scatter_map(neighborhood, fig=fig, gs_slot=gridspec_slot, hue=hue, size=size, marker=marker, projection=projection,
                                           proj_default=proj_default,
                                           background=background, borders=borders, rivers=rivers, lakes=lakes,
                                           ocean=ocean, land=land,
                                           figsize=figsize, scatter_kwargs=scatter_kwargs, lgd_kwargs=lgd_kwargs,
                                           gridspec_kwargs=gridspec_kwargs, colorbar=colorbar,
                                           legend=legend, cmap=cmap, edgecolor=neighborhood['original'].values)
        return fig, ax_d
    
    def dashboard(self, figsize=[11, 8], gs=None, plt_kwargs=None, histplt_kwargs=None, spectral_kwargs=None,
                  spectralsignif_kwargs=None, spectralfig_kwargs=None, map_kwargs=None,
                  hue='archiveType', marker='archiveType', size=None, scatter_kwargs=None,
                  gridspec_kwargs=None,
                  savefig_settings=None):
        ''' Create a dashboard of plots for the GeoSeries object

        Parameters
        ----------
        figsize : list or tuple, optional
            Figure size. The default is [11,8].

        gs : matplotlib.gridspec object, optional
            Requires at least two rows and 4 columns.
            - top row, left: timeseries
            - top row, right: histogram
            - bottom left: map
            - bottom right: PSD
            See [matplotlib.gridspec.GridSpec](https://matplotlib.org/stable/tutorials/intermediate/gridspec.html) for details.

        plt_kwargs : dict, optional
            Optional arguments for the timeseries plot. See Series.plot() or EnsembleSeries.plot_envelope(). The default is None.

        histplt_kwargs : dict, optional
            Optional arguments for the distribution plot. See Series.histplot() or EnsembleSeries.plot_distplot(). The default is None.

        spectral_kwargs : dict, optional
            Optional arguments for the spectral method. Default is to use Lomb-Scargle method. See Series.spectral() or EnsembleSeries.spectral(). The default is None.

        spectralsignif_kwargs : dict, optional
            Optional arguments to estimate the significance of the power spectrum. See PSD.signif_test. Note that we currently do not support significance testing for ensembles. The default is None.

        spectralfig_kwargs : dict, optional
            Optional arguments for the power spectrum figure. See PSD.plot() or MultiplePSD.plot_envelope(). The default is None.

        map_kwargs : dict, optional
            Optional arguments for map configuration
            - projection: str; Optional value for map projection. Default 'auto'.
            - proj_default: bool
            - lakes, land, ocean, rivers, borders, coastline, background: bool or dict;
            - lgd_kwargs: dict; Optional values for how the map legend is configured
            - gridspec_kwargs: dict; Optional values for adjusting the arrangement of the colorbar, map and legend in the map subplot
            - legend: bool; Whether to draw a legend on the figure. Default is True
            - colorbar: bool; Whether to draw a colorbar on the figure if the data associated with hue are numeric. Default is True
            The default is None.

        hue : str, optional
            Variable associated with color coding for points plotted on map. May correspond to a continuous or categorical variable.
            The default is 'archiveType'.

        size : str, optional
            Variable associated with size. Must correspond to a continuous numeric variable.
            The default is None.

        marker : string, optional
            Grouping variable that will produce points with different markers. Can have a numeric dtype but will always be treated as categorical.
            The default is 'archiveType'.

        scatter_kwargs : dict, optional
            Optional arguments configuring how data are plotted on a map. See description of scatter_kwargs in pyleoclim.utils.mapping.scatter_map

        gridspec_kwargs : dict, optional
            Optional dictionary for configuring dashboard layout using gridspec
            For information about Gridspec configuration, refer to `Matplotlib documentation <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.gridspec.GridSpec.html#matplotlib.gridspec.GridSpec>_. The default is None.

        savefig_settings : dict, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}.
            The default is None.

        Returns
        -------
        fig : matplotlib.figure
            The figure

        ax : dict
            dictionary of matplotlib ax

        See also
        --------

        pyleoclim.core.series.Series.plot : plot a timeseries

        pyleoclim.core.ensembleseries.EnsembleSeries.plot_envelope: Envelope plots for an ensemble

        pyleoclim.core.series.Series.histplot : plot a distribution of the timeseries

        pyleoclim.core.ensembleseries.EnsembleSeries.histplot : plot a distribution of the timeseries across ensembles

        pyleoclim.core.series.Series.spectral : spectral analysis method.

        pyleoclim.core.multipleseries.MultipleSeries.spectral : spectral analysis method for multiple series.

        pyleoclim.core.psds.PSD.signif_test : significance test for timeseries analysis

        pyleoclim.core.psds.PSD.plot : plot power spectrum

        pyleoclim.core.psds.MulitplePSD.plot : plot envelope of power spectrum

        pyleoclim.core.geoseries.GeoSeries.map : map location of dataset

        pyleoclim.utils.mapping.scatter_map : Underlying mapping function for Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            ts = pyleo.utils.datasets.load_dataset('EDC-dD')
            ts_interp = ts.convert_time_unit('kyr BP').interp(step=.5) # interpolate for a faster result
            fig, ax = ts_interp.dashboard()


        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        # start plotting
        fig = plt.figure(figsize=figsize)

        if gs == None:
            gridspec_kwargs = {} if type(gridspec_kwargs) != dict else gridspec_kwargs
            gridspec_defaults = dict(wspace=0, width_ratios=[3, .25, 2, 1],
                                     height_ratios=[1, .1, 1], left=0, right=1.1)
            gridspec_defaults.update(gridspec_kwargs)
            gs = gridspec.GridSpec(len(gridspec_defaults['height_ratios']), len(gridspec_defaults['width_ratios']), **gridspec_defaults)

        ax = {}
        # Plot the timeseries
        plt_kwargs = {} if plt_kwargs is None else plt_kwargs.copy()
        ax['ts'] = fig.add_subplot(gs[0, :-1])
        plt_kwargs.update({'ax': ax['ts']})
        # use the defaults if color/markers not specified
        
        if self.archiveType is not None:
            archiveType = lipdutils.LipdToOntology(self.archiveType)
            if archiveType not in lipdutils.PLOT_DEFAULT.keys():
                archiveType = 'Other'                
        else: 
            archiveType = 'Other'
        
        if 'marker' not in plt_kwargs.keys():
            plt_kwargs.update({'marker': lipdutils.PLOT_DEFAULT[archiveType][1]})
        if 'color' not in plt_kwargs.keys():
            plt_kwargs.update({'color': lipdutils.PLOT_DEFAULT[archiveType][0]})
        ax['ts'] = self.plot(**plt_kwargs)
        
        ymin, ymax = ax['ts'].get_ylim()

        # plot the histplot
        histplt_kwargs = {} if histplt_kwargs is None else histplt_kwargs.copy()
        ax['dts'] = fig.add_subplot(gs[0, -1:])
        histplt_kwargs.update({'ax': ax['dts']})
        histplt_kwargs.update({'ylabel': 'Counts'})
        histplt_kwargs.update({'vertical': True})
        if 'color' not in histplt_kwargs.keys():            
            histplt_kwargs.update({'color': lipdutils.PLOT_DEFAULT[archiveType][0]})

        ax['dts'] = self.histplot(**histplt_kwargs)
        ax['dts'].set_ylim([ymin, ymax])
        ax['dts'].set_yticklabels([])
        ax['dts'].set_ylabel('')
        ax['dts'].set_yticks([])

        # plot map
        map_kwargs = {} if map_kwargs is None else map_kwargs.copy()

        projection = map_kwargs.pop('projection', 'Orthographic')
        proj_default = map_kwargs.pop('proj_default', True)
        lakes = map_kwargs.pop('lakes', False)
        land = map_kwargs.pop('land', False)
        ocean = map_kwargs.pop('ocean', False)
        rivers = map_kwargs.pop('rivers', False)
        borders = map_kwargs.pop('borders', True)
        coastline = map_kwargs.pop('coastline', True)
        background = map_kwargs.pop('background', True)

        map_gridspec_kwargs = map_kwargs.pop('gridspec_kwargs', {})
        lgd_kwargs = map_kwargs.pop('lgd_kwargs', {})

        if 'edgecolor' in map_kwargs.keys():
            scatter_kwargs.update({'edgecolor': map_kwargs['edgecolor']})

        cmap = map_kwargs.pop('cmap', None)
        legend = map_kwargs.pop('legend', False)
        colorbar = map_kwargs.pop('colorbar', False)

        if legend == False:
            map_gridspec_kwargs['width_ratios'] = [.5,16, 1]

        _, ax['map'] =mapping.scatter_map(self, hue=hue, size=size, marker=marker, projection=projection, proj_default=proj_default,
                    background=background, borders=borders, coastline=coastline, rivers=rivers, lakes=lakes, ocean=ocean, land=land,
                    figsize=None, scatter_kwargs=scatter_kwargs,gridspec_kwargs = map_gridspec_kwargs,
                    lgd_kwargs=lgd_kwargs, legend=legend, cmap=cmap, colorbar=colorbar,
                    fig=fig, gs_slot=gs[-1, 0:1])

        # spectral analysis
        spectral_kwargs = {} if spectral_kwargs is None else spectral_kwargs.copy()
        if 'method' in spectral_kwargs.keys():
            pass
        else:
            spectral_kwargs.update({'method': 'lomb_scargle'}) # unneeded as it is already the default 
        if 'freq' in spectral_kwargs.keys():
            pass
        else:
            spectral_kwargs.update({'freq': 'lomb_scargle'})

        ax['spec'] = fig.add_subplot(gs[-1, -2:])
        spectralfig_kwargs = {} if spectralfig_kwargs is None else spectralfig_kwargs.copy()
        spectralfig_kwargs.update({'ax': ax['spec']})

        
        ts_preprocess = self.detrend().standardize()
        psd = ts_preprocess.spectral(**spectral_kwargs)

        # Significance test
        spectralsignif_kwargs = {} if spectralsignif_kwargs is None else spectralsignif_kwargs.copy()
        psd_signif = psd.signif_test(**spectralsignif_kwargs)
        # plot
        if 'color' not in spectralfig_kwargs.keys():
            spectralfig_kwargs.update({'color': lipdutils.PLOT_DEFAULT[archiveType][0]})
        if 'signif_clr' not in spectralfig_kwargs.keys():
            spectralfig_kwargs.update({'signif_clr': 'grey'})
        ax['spec'] = psd_signif.plot(**spectralfig_kwargs)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)

        return fig, ax
    
    def segment(self, factor=10, verbose = False):
        """Gap detection

        This function segments a timeseries into n parts following a gap- detection algorithm. The rule of gap detection is very simple:
            we define the intervals between time points as dts, then if dts[i] is larger than factor * dts[i-1],
            we think that the change of dts (or the gradient) is too large, and we regard it as a breaking point
            and divide the time series into two segments here

        Parameters
        ----------
        factor : float
            The factor that adjusts the threshold for gap detection
        
        verbose : bool
            If True, will print warning messages if there is any

        Returns
        -------
        res : MultiplegGeoSeries or GeoSeries
            If gaps were detected, returns the segments in a MultipleGeoSeries object,
            else, returns the original timeseries.
            
        Examples
        --------

        .. jupyter-execute::
            
            import numpy as np
            gs = pyleo.utils.datasets.load_dataset('EDC-dD')
            gs.value[4000:5000] = np.nan # cut a large gap in the middle
            mgs = gs.segment()
            mgs.plot()

        """
        from ..core.multiplegeoseries import MultipleGeoSeries
        seg_y, seg_t, n_segs = tsutils.ts2segments(self.value,self.time,factor=factor)
        
        if len(seg_y)>1:
            s_list=[]
            for idx,s in enumerate(seg_y):
                if self.label is not None: 
                    s_lbl =  self.label + ' segment ' + str(idx+1)  
                else:
                    s_lbl =  'segment ' + str(idx+1)
                s_tmp = self.copy() # copy metadata
                s_tmp.time = seg_t[idx]
                s_tmp.value = s
                s_tmp.label = s_lbl
                s_list.append(s_tmp)
            res=MultipleGeoSeries(series_list=s_list)
        elif len(seg_y)==1:
            res=self.copy()
        else:
            raise ValueError('No timeseries detected')
        return res
    
    def resample(self, rule, keep_log = False, **kwargs):
        """
        Run analogue to pandas.Series.resample.
    
        This is a convenience method: doing
    
            ser.resample('AS').mean()
    
        will do the same thing as
    
            ser.pandas_method(lambda x: x.resample('AS').mean())
        
        but will also accept some extra resampling rules, such as `'Ga'` (see below).
    
        Parameters
        ----------
        rule : str
            The offset string or object representing target conversion.
            Can also accept pyleoclim units, such as 'ka' (1000 years),
            'Ma' (1 million years), and 'Ga' (1 billion years).
    
            Check the [pandas resample docs](https://pandas.pydata.org/docs/dev/reference/api/pandas.DataFrame.resample.html)
            for more details.
    
        kwargs : dict
            Any other arguments which will be passed to pandas.Series.resample.
        
        Returns
        -------
        SeriesResampler
            Resampler object, not meant to be used to directly. Instead,
            an aggregation should be called on it, see examples below.
        
        Examples
        --------

        .. jupyter-execute::

            ts = pyleo.utils.load_dataset('EDC-dD').convert_time_unit('ky BP')
            ts5k = ts.resample('1ka').mean()
            fig, ax = ts.plot()
            ts5k.plot(ax=ax,color='C1')
                
        """
        search = re.search(r'(\d*)([a-zA-Z]+)', rule)
        if search is None:
            raise ValueError(f"Invalid rule provided, got: {rule}")
    
        md = self.metadata
        if md['label'] is not None:
            md['label'] = md['label'] + ' (' + rule + ' resampling)'
    
        multiplier = search.group(1)
        if multiplier == '':
            multiplier = 1
        else:
            multiplier = int(multiplier)
        unit = search.group(2)
        if unit.lower() in tsbase.MATCH_A:
            rule = f'{multiplier}AS'
        elif unit.lower() in tsbase.MATCH_KA:
            rule = f'{1_000*multiplier}AS'
        elif unit.lower() in tsbase.MATCH_MA:
            rule = f'{1_000_000*multiplier}AS'
        elif unit.lower() in tsbase.MATCH_GA:
            rule = f'{1_000_000_000*multiplier}AS'
        
        ser = self.to_pandas()
        
        return GeoSeriesResampler(rule, ser, md, keep_log, kwargs)
    
class GeoSeriesResampler:
    """
    This is only meant to be used internally, and is not meant to 
    be public-facing or to be used directly by users.

    If users call

        ts.resample('1Y').mean()
    
    then they will get back a pyleoclim.GeoSeries, and `GeoSeriesResampler`
    will only be used in an intermediate step. Think of it as an
    implementation detail.
    """
    def __init__(self, rule, series, metadata, keep_log, kwargs):
        self.rule = rule
        self.series = series
        self.metadata = metadata
        self.keep_log = keep_log
        self.kwargs = kwargs
    
    def __getattr__(self, attr):
        attr = getattr(self.series.resample(self.rule,  **self.kwargs), attr)
        def func(*args, **kwargs):
            series = attr(*args, **kwargs)
            series.index = series.index + (series.index[1] - series.index[0])/2 # sample midpoints
            _, __, direction = tsbase.time_unit_to_datum_exp_dir(self.metadata['time_unit'], self.metadata['time_name'])
            if direction == 'prograde':
                from_pandas = GeoSeries.from_pandas(series, metadata=self.metadata)
            else:
                from_pandas = GeoSeries.from_pandas(series.sort_index(ascending=False), metadata=self.metadata)
            if self.keep_log == True:
                if from_pandas.log is None:
                    from_pandas.log=()
                from_pandas.log += ({len(from_pandas.log): 'resample','rule': self.rule},)
            return from_pandas
        return func

    
