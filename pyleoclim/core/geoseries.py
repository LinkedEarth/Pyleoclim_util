"""
The GeoSeries class is a child of Series, with additional metadata latitude (lat) and longitude (lon)
This unlocks plotting capabilities like map() and dashboard(). 
"""
from ..utils import plotting, mapping, lipdutils
from ..core.series import Series
from ..core.multipleseries import MultipleSeries

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from tqdm import tqdm
import warnings

class GeoSeries(Series):
    '''The GeoSeries class is a child of the Series class, and requires geolocation
    information (latitude, longitude). Elevation is optional, but can be used in mapping, if present.
    
    Parameters
    ----------

    time : list or numpy.array
        independent variable (t)

    value : list of numpy.array
        values of the dependent variable (y)
        
    lat : float
        latitude N in decimal degrees.
        
    lon : float
        longitude East in decimal degrees. Negative values will be converted to an angle in [0 , 360)
       
    elevation : float
        elevation of the sample, in meters above sea level.                                                                                          

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
        climate archive, one of ....                                                                                    

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

    Examples
    --------

    Import the EPICA Dome C deuterium record and display a quick synopsis:

    >>> edc = pyleo.utils.load_dataset('EDC-dD')
    >>> edc.metadata
    >>> edc.view()
          
    '''

    def __init__(self, time, value, lat, lon, elevation = None, time_unit=None, time_name=None, 
                 value_name=None, value_unit=None, label=None, 
                 importedFrom=None, archiveType = None, log=None, keep_log=False,
                 sort_ts = 'ascending', dropna = True, verbose=True, clean_ts=False):
        
       
        # assign latitude
        if lat is not None:
            lat = float(lat) 
            if -90 <= lat <= 90: 
                self.lat = lat
            else:
                ValueError('Latitude must be a number in [-90; 90]')
        else:
            self.lat = None # assign a default value to prevent bugs ?
            
        # assign longitude
        if lon is not None:
            lon = float(lon)
            if 0 <= lon < 360:     
                self.lon = lon
            elif -180 <= lon < 0:
                self.lon = 360 - lon
            else:
                ValueError('Longitude must be a number in [-180,360]')
        else:
            self.lon = None # assign a default value to prevent bugs ?
            
        # elevation
        self.elevation = elevation
            
        #assign all the rest
        super().__init__(time, value, time_unit, time_name, value_name,
                         value_unit, label, importedFrom, archiveType,
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
            importedFrom = self.importedFrom,
            log = self.log,
        )
    
    def map(self, projection='Orthographic', proj_default=True,
            background=True, borders=False, rivers=False, lakes=False,
            figsize=None, ax=None, marker=None, color=None,
            markersize=None, scatter_kwargs=None,
            legend=True, lgd_kwargs=None, savefig_settings=None):
        
        '''Map the location of the record

        Parameters
        ----------
        
        projection : str, optional

            The projection to use. The default is 'Robinson'.

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

        ax : matplotlib.ax, optional

            The matplotlib axis onto which to return the map. The default is None.

        marker : str, optional

            The marker type for each archive. The default is None. Uses plot_default

        color : str, optional

            Color for each archive. The default is None. Uses plot_default

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

        res : fig,ax

        See also
        --------

        pyleoclim.utils.mapping.map : Underlying mapping function for Pyleoclim

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            ts = pyleo.utils.datasets.load_dataset('EDC-dD')
            @savefig mapone.png
            fig, ax = ts.map()
            pyleo.closefig(fig)

        '''
        
        scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()
        # get the information from the timeseries
        lat = [self.lat]
        lon = [self.lon]
        
        if self.archiveType is not None:
            archiveType = lipdutils.LipdToOntology(self.archiveType).lower().replace(" ", "")
        else: 
            archiveType = 'other'

        # make sure criteria is in the plot_default list
        if archiveType not in lipdutils.PLOT_DEFAULT.keys():
            archiveType = 'other'

        if markersize is not None:
            scatter_kwargs.update({'s': markersize})

        if marker == None:
            marker = lipdutils.PLOT_DEFAULT[archiveType][1]

        if color == None:
            color = lipdutils.PLOT_DEFAULT[archiveType][0]

        if proj_default == True:
            proj1 = {'central_latitude': lat[0],
                     'central_longitude': lon[0]}
            proj2 = {'central_latitude': lat[0]}
            proj3 = {'central_longitude': lon[0]}

        archiveType = [archiveType]  # list so it will work with map
        marker = [marker]
        color = [color]

        if proj_default == True:

            try:
                res = mapping.map(lat=lat, lon=lon, criteria=archiveType,
                                  marker=marker, color=color,
                                  projection=projection, proj_default=proj1,
                                  background=background, borders=borders,
                                  rivers=rivers, lakes=lakes,
                                  figsize=figsize, ax=ax,
                                  scatter_kwargs=scatter_kwargs, legend=legend,
                                  lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings, )

            except:
                try:
                    res = mapping.map(lat=lat, lon=lon, criteria=archiveType,
                                      marker=marker, color=color,
                                      projection=projection, proj_default=proj3,
                                      background=background, borders=borders,
                                      rivers=rivers, lakes=lakes,
                                      figsize=figsize, ax=ax,
                                      scatter_kwargs=scatter_kwargs, legend=legend,
                                      lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)
                except:
                    res = mapping.map(lat=lat, lon=lon, criteria=archiveType,
                                      marker=marker, color=color,
                                      projection=projection, proj_default=proj2,
                                      background=background, borders=borders,
                                      rivers=rivers, lakes=lakes,
                                      figsize=figsize, ax=ax,
                                      scatter_kwargs=scatter_kwargs, legend=legend,
                                      lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)

        else:
            res = mapping.map(lat=lat, lon=lon, criteria=archiveType,
                              marker=marker, color=color,
                              projection=projection, proj_default=proj_default,
                              background=background, borders=borders,
                              rivers=rivers, lakes=lakes,
                              figsize=figsize, ax=ax,
                              scatter_kwargs=scatter_kwargs, legend=legend,
                              lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)
        return res
    
    def dashboard(self, figsize=[11, 8], plt_kwargs=None, histplt_kwargs=None, spectral_kwargs=None,
                  spectralsignif_kwargs=None, spectralfig_kwargs=None, map_kwargs=None,
                  savefig_settings=None):
        '''

        Parameters
        ----------
        
        figsize : list or tuple, optional

            Figure size. The default is [11,8].

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

            Optional arguments for the map. See LipdSeries.map(). The default is None.

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

        ax : matplolib.axis

            The axis

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

        pyleoclim.utils.mapping.map : Underlying mapping function for Pyleoclim

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            ts = pyleo.utils.datasets.load_dataset('EDC-dD')
            ts_interp =ts.convert_time_unit('kyr BP').interp(step=.5)
            @savefig ts_dashboard.png
            fig, ax = ts_interp.dashboard()
            pyleo.closefig(fig)

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        # start plotting
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 6, wspace=0)
        gs.update(left=0, right=1.1)


        ax = {}
        # Plot the timeseries
        plt_kwargs = {} if plt_kwargs is None else plt_kwargs.copy()
        ax['ts'] = fig.add_subplot(gs[0, :-1])
        plt_kwargs.update({'ax': ax['ts']})
        # use the defaults if color/markers not specified
        
        if self.archiveType is not None:
            archiveType = lipdutils.LipdToOntology(self.archiveType).lower().replace(" ", "")
        else: 
            archiveType = 'other'
        
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

        # make the map - brute force since projection is not being returned properly
        lat = [self.lat]
        lon = [self.lon]

        map_kwargs = {} if map_kwargs is None else map_kwargs.copy()
        if 'projection' in map_kwargs.keys():
            projection = map_kwargs['projection']
        else:
            projection = 'Orthographic'
        if 'proj_default' in map_kwargs.keys():
            proj_default = map_kwargs['proj_default']
        else:
            proj_default = True
        if proj_default == True:
            proj1 = {'central_latitude': lat[0],
                     'central_longitude': lon[0]}
            proj2 = {'central_latitude': lat[0]}
            proj3 = {'central_longitude': lon[0]}
            try:
                proj = mapping.set_proj(projection=projection, proj_default=proj1)
            except:
                try:
                    proj = mapping.set_proj(projection=projection, proj_default=proj3)
                except:
                    proj = mapping.set_proj(projection=projection, proj_default=proj2)
        if 'marker' in map_kwargs.keys():
            marker = map_kwargs['marker']
        else:
            marker = lipdutils.PLOT_DEFAULT[archiveType][1]
        if 'color' in map_kwargs.keys():
            color = map_kwargs['color']
        else:
            color = lipdutils.PLOT_DEFAULT[archiveType][0]
        if 'background' in map_kwargs.keys():
            background = map_kwargs['background']
        else:
            background = True
        if 'borders' in map_kwargs.keys():
            borders = map_kwargs['borders']
        else:
            borders = False
        if 'rivers' in map_kwargs.keys():
            rivers = map_kwargs['rivers']
        else:
            rivers = False
        if 'lakes' in map_kwargs.keys():
            lakes = map_kwargs['lakes']
        else:
            lakes = False
        if 'scatter_kwargs' in map_kwargs.keys():
            scatter_kwargs = map_kwargs['scatter_kwargs']
        else:
            scatter_kwargs = {}
        if 'markersize' in map_kwargs.keys():
            scatter_kwargs.update({'s': map_kwargs['markersize']})
        else:
            pass
        if 'lgd_kwargs' in map_kwargs.keys():
            lgd_kwargs = map_kwargs['lgd_kwargs']
        else:
            lgd_kwargs = {}
        if 'legend' in map_kwargs.keys():
            legend = map_kwargs['legend']
        else:
            legend = False
        # make the plot map

        data_crs = ccrs.PlateCarree()
        ax['map'] = fig.add_subplot(gs[1, 0:2], projection=proj)
        ax['map'].coastlines()
        if background is True:
            ax['map'].stock_img()
        # Other extra information
        if borders is True:
            ax['map'].add_feature(cfeature.BORDERS)
        if lakes is True:
            ax['map'].add_feature(cfeature.LAKES)
        if rivers is True:
            ax['map'].add_feature(cfeature.RIVERS)
        ax['map'].scatter(lon, lat, zorder=10, label=marker, facecolor=color, transform=data_crs, **scatter_kwargs)
        if legend == True:
            ax.legend(**lgd_kwargs)

        # spectral analysis
        spectral_kwargs = {} if spectral_kwargs is None else spectral_kwargs.copy()
        if 'method' in spectral_kwargs.keys():
            pass
        else:
            spectral_kwargs.update({'method': 'lomb_scargle'})
        if 'freq_method' in spectral_kwargs.keys():
            pass
        else:
            spectral_kwargs.update({'freq_method': 'lomb_scargle'})

        ax['spec'] = fig.add_subplot(gs[1, -3:])
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
        
        #gs.tight_layout(fig)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)

        return fig, ax
    
    