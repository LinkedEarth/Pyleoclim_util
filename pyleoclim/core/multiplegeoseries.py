"""
A MultipleGeoSeries object is a collection (more precisely, a 
list) of GeoSeries objects. This is handy in case you want to apply the same method 
to such a collection at once (e.g. process a bunch of series in a consistent fashion).
Compared to its parent class MultipleSeries, MultipleGeoSeries opens new possibilites regarding mapping.
"""
from ..core.multipleseries import MultipleSeries
from ..utils import mapping as mp
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from itertools import cycle
import matplotlib.lines as mlines
import numpy as np
#import warnings

#import matplotlib.pyplot as plt
#import matplotlib as mpl
#from matplotlib import cm
#from itertools import cycle
#import matplotlib.lines as mlines
#import copy


class MultipleGeoSeries(MultipleSeries):
    '''MultipleGeoSeries object.

    This object handles a collection of the type GeoSeries and can be created from a list of such objects.
    MultipleGeoSeries should be used when the need to run analysis on multiple records arises, such as running principal component analysis.
    Some of the methods automatically transform the time axis prior to analysis to ensure consistency.

    Parameters
    ----------

    series_list : list
    
        a list of pyleoclim.Series objects

    time_unit : str
    
        The target time unit for every series in the list.
        If None, then no conversion will be applied;
        Otherwise, the time unit of every series in the list will be converted to the target.

    label : str
   
        label of the collection of timeseries (e.g. 'Euro 2k')

    Examples
    --------
    .. jupyter-execute::
        
        from pylipd.utils.dataset import load_dir
        lipd = load_dir(name='Euro2k')
        df = lipd.get_timeseries_essentials()
        dfs = df.query("archiveType in ('tree')") 
        # place in a MultipleGeoSeries object
        ts_list = []
        for _, row in dfs.iterrows():
            ts_list.append(pyleo.GeoSeries(time=row['time_values'],value=row['paleoData_values'],
                                           time_name=row['time_variableName'],value_name=row['paleoData_variableName'],
                                           time_unit=row['time_units'], value_unit=row['paleoData_units'],
                                           lat = row['geo_meanLat'], lon = row['geo_meanLon'],
                                           archiveType = row['archiveType'], verbose = False, 
                                           label=row['dataSetName']+'_'+row['paleoData_variableName'])) 
    
        Euro2k = pyleo.MultipleGeoSeries(ts_list, label='Euro2k',time_unit='years AD')  
        Euro2k.map(projection='Orthographic') 
    '''

    def __init__(self, series_list, time_unit=None, label=None):
        # check that all components are GeoSeries
        self.series_list = series_list
        from ..core.geoseries import GeoSeries
        if not all([isinstance(ts, GeoSeries) for ts in series_list]):
            raise ValueError('All components must be GeoSeries objects')
        
        super().__init__(series_list, time_unit, label)
        # self.pca = super().pca

    # ============ MAP goes here ================


    def map(self, marker='archiveType', hue='archiveType', size=None, cmap=None,
            edgecolor='k', projection='auto',
            proj_default=True, crit_dist=5000,colorbar=True,
            background=True, borders=True, rivers=False, lakes=False, land=True,ocean=True,
            figsize=None, fig=None, scatter_kwargs=None, gridspec_kwargs=None, legend=True, gridspec_slot=None,
            lgd_kwargs=None, savefig_settings=None, **kwargs):
        '''
        

        Parameters
        ----------
        hue : string, optional
            Grouping variable that will produce points with different colors. Can be either categorical or numeric, although color mapping will behave differently in latter case.
            The default is 'archiveType'.

        size : string, optional
            Grouping variable that will produce points with different sizes. Expects to be numeric. Any data without a value for the size variable will be filtered out.
            The default is None.

        marker : TYPE, optional
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
            Default is on (True).

        borders : bool, optional
            Draws the countries border.
            Defaults is off (False).

        land : bool, optional
            Colors land masses.
            Default is off (False).

        ocean : bool, optional
            Colors oceans.
            Default is off (False).

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
            For information about Gridspec configuration, refer to `Matplotlib documentation <https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.gridspec.GridSpec.html#matplotlib.gridspec.GridSpec>_. The default is None.

        kwargs: dict, optional
            - 'missing_val_hue', 'missing_val_marker', 'missing_val_label' can all be used to change the way missing values are represented ('k', '?',  are default hue and marker values will be associated with the label: 'missing').
            - 'hue_mapping' and 'marker_mapping' can be used to submit dictionaries mapping hue values to colors and marker values to markers. Does not replace passing a string value for hue or marker.


        Returns
        -------
        fig, ax_d
            Matplotlib figure, dictionary of ax objects which includes the as many as three items: 'cb' (colorbar ax), 'map' (scatter map), and 'leg' (legend ax)
            
        See also
        --------
        pyleoclim.utils.mapping.scatter_map: information-rich scatterplot on Cartopy map
            
        Examples
        --------
        .. jupyter-execute::
            
            from pylipd.utils.dataset import load_dir
            lipd = load_dir(name='Pages2k') # this loads a small subset of the PAGES 2k database
            df = lipd.get_timeseries_essentials()
            dfs = df.query("paleoData_variableName in ('temperature','d18O', 'MXD', 'Uk37','trsgi')") 

            # place in a MultipleGeoSeries object
            ts_list = []
            for _, row in dfs.iterrows():
                ts_list.append(pyleo.GeoSeries(time=row['time_values'],value=row['paleoData_values'],
                                               time_name=row['time_variableName'],value_name=row['paleoData_variableName'],
                                               time_unit=row['time_units'], value_unit=row['paleoData_units'],
                                               lat = row['geo_meanLat'], lon = row['geo_meanLon'],
                                               elevation = row['geo_meanElev'], observationType = row['paleoData_proxy'],
                                               archiveType = row['archiveType'], verbose = False, 
                                               label=row['dataSetName']+'_'+row['paleoData_variableName'])) 

            Euro2k = pyleo.MultipleGeoSeries(ts_list, label='minimal PAGES 2k',time_unit='years AD')  

            Euro2k.map() 
         
        By default, a projection is picked based on the degree of geographic clustering of the sites. To focus the map on Europe and use a more local projection, do:   
            
        .. jupyter-execute::     
            
            eur_coord = {'central_latitude':45, 'central_longitude':20}
            Euro2k.map(projection='Orthographic',proj_default=eur_coord) 
            
        Same, with size to represent elevation:      
        .. jupyter-execute::
            
            Euro2k.map(projection='Orthographic',size='elevation', proj_default=eur_coord) 
        
        Same, with hue to represent elevation:    
        .. jupyter-execute::
            
            Euro2k.map(projection='Orthographic',hue='elevation',proj_default=eur_coord) 

        '''

        fig, ax_d = mp.scatter_map(self, hue=hue, size=size, marker=marker,
                    edgecolor=edgecolor, projection=projection,
                                        proj_default=proj_default,
                                        crit_dist=crit_dist,
                                        background=background, borders=borders, rivers=rivers, lakes=lakes,
                                        ocean=ocean,
                                        land=land, gridspec_kwargs=gridspec_kwargs,
                                        figsize=figsize, scatter_kwargs=scatter_kwargs,
                                        lgd_kwargs=lgd_kwargs, legend=legend, colorbar=colorbar,
                                        cmap=cmap,
                                        fig=fig, gs_slot=gridspec_slot, **kwargs)
        return fig, ax_d
        # def make_scalar_mappable(cmap, lims=None, n=None):
        #     if type(cmap) == list:
        #         if n is None:
        #             ax_cmap = mpl.colors.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"])
        #         else:
        #             ax_cmap = mpl.colors.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"], N=n)
        #     elif type(cmap) == str:
        #         if n is None:
        #             ax_cmap = plt.get_cmap(cmap)
        #         else:
        #             ax_cmap = plt.get_cmap(cmap, n)
        #     else:
        #         print('what madness is this?')
        #
        #     if type(lims) in [list, tuple]:
        #         ax_norm = mpl.colors.Normalize(vmin=min(lims), vmax=max(lims), clip=False)
        #     else:
        #         ax_norm = None
        #     ax_sm = cm.ScalarMappable(norm=ax_norm, cmap=ax_cmap)
        #
        #     return ax_sm
        #
        # mappings = {'color_pal': color_pal}
        # f = lambda m, c, s: mlines.Line2D([], [], marker=m, color=c, markersize=s, ls="none")
        # for key in ['hue_mapping', 'marker_mapping']:
        #     if key in kwargs:
        #         mappings[key] = kwargs[key]
        #
        # missing_d = {'hue': kwargs['missing_val_hue'] if 'missing_val_hue' in kwargs else 'k',
        #              'marker': kwargs['missing_val_marker'] if 'missing_val_marker' in kwargs else 'X',
        #              'size': kwargs['missing_val_size'] if 'missing_val_size' in kwargs else 10,
        #              }
        #
        # def make_cont_hue(trait_vals, mappings):
        #     color_pal = mappings['color_pal']
        #     if color_pal == None:
        #         color_pal = 'viridis'
        #
        #     hue_mappable = make_scalar_mappable(color_pal, lims=[min([val for val in trait_vals if val != None]),
        #                                                          max([val for val in trait_vals if val != None])])
        #     return dict(f=hue_mappable.to_rgba, mappable=hue_mappable)
        #
        # def make_cat_hue(trait_unique, mappings):
        #     if 'hue_mapping' in mappings:
        #         hue_mapping = mappings['hue_mapping']
        #     else:
        #         color_pal = mappings['color_pal']
        #         if color_pal is None:
        #             color_pal = 'turbo'
        #         hue_mappable = make_scalar_mappable(color_pal, lims=None, n=len(trait_unique))
        #         hue_mapping = {trait_val: hue_mappable.cmap.colors[ik] for ik, trait_val in enumerate(trait_unique)}
        #         hue_mapping['unknown'] =missing_d['hue']
        #
        #     leg_d = {'handles': [], 'labels': []}
        #     for key in hue_mapping:
        #         leg_d['handles'].append(f("s", hue_mapping[key], 8))
        #         leg_d['labels'].append(key)
        #
        #     return dict(f=lambda x: hue_mapping[x], mapping=hue_mapping, leg=leg_d)  # 'mappable': hue_mappable,
        #
        # def make_cont_size(trait_vals, mappings):
        #     lims = [min([val for val in trait_vals if val != None]),
        #             max([val for val in trait_vals if val != None])]
        #     szes = np.linspace(lims[0], lims[1], 4)
        #     size_f = lambda x: x * 10 / (lims[1] - lims[0])
        #     leg_d = {'handles': [], 'labels': []}
        #     for key in szes:
        #         leg_d['handles'].append(f('s', "k", size_f(key)))
        #         leg_d['labels'].append(int(key))
        #     return dict(f=size_f, lims=lims, leg=leg_d)
        #
        # def make_cat_marker(trait_unique, mappings):
        #     if 'marker_mapping' in mappings:
        #         marker_mapping = mappings['marker_mapping']
        #     else:
        #         # the dot and the filled circle look to much alike to be considered different markers
        #         m = cycle(mlines.Line2D.filled_markers[1:])
        #         marker_mapping = {trait_val: next(m) for ik, trait_val in enumerate(trait_unique)}
        #         marker_mapping['unknown'] =missing_d['marker']
        #     leg_d = {'handles': [], 'labels': []}
        #     for key in marker_mapping:
        #         leg_d['handles'].append(f(marker_mapping[key], "k", 8))
        #         leg_d['labels'].append(key)
        #     return dict(f=lambda x: marker_mapping[x], mapping=marker_mapping, leg=leg_d)
        #
        # continuous_d = {'hue': make_cont_hue,
        #                 'size': make_cont_size,
        #                 'marker': None
        #                 }
        #
        # categorical_d = {'hue': make_cat_hue,
        #                  'size': None,
        #                  'marker': make_cat_marker
        #                  }
        #
        # trait_d = {'hue': hue, 'marker': marker, 'size': size}
        # legend_d = {'hue': None, 'marker': None, 'size': None}
        #
        # for trait_key in trait_d.keys():
        #     trait = trait_d[trait_key]
        #     if trait is None:
        #         trait_vals = [None for ik in range(len(self.series_list))]
        #         if trait_key == 'hue':
        #             attrib_vals = [missing_d[trait_key] for ik in trait_vals]
        #         else:
        #             attrib_vals = None
        #         d = {'attrib_vals': attrib_vals}
        #     else:
        #         trait_vals = [geos.__dict__[trait] if trait in geos.__dict__.keys() else None for geos in
        #                       self.series_list]
        #         trait_vals = [val if val != 'None' else None for val in trait_vals]
        #         trait_unique = list(set([val for val in trait_vals if val != None]))
        #         trait_val_types = [True if type(val) == np.str_ else False for val in trait_unique]
        #
        #         # categorical
        #         if True in trait_val_types:
        #             if categorical_d[trait_key] != None:
        #                 d = categorical_d[trait_key](trait_unique, mappings)
        #                 d['attrib_vals'] = [d['mapping'][val] if val != None else missing_d[trait_key] for val in
        #                                     trait_vals]
        #                 # legend = True
        #                 if legend_attribute is None:
        #                     legend_attribute = trait_key
        #             else:
        #                 # attrib_vals = [missing_d[trait_key] for ik in trait_vals]
        #                 d = {'attrib_vals': None}
        #
        #         # continuous
        #         else:
        #             if continuous_d[trait_key] != None:
        #                 d = continuous_d[trait_key](trait_vals, mappings)
        #                 trait_vals = [int(val) if np.isnan(val) == False else None for val in trait_vals]
        #                 d['attrib_vals'] = [d['f'](val) if val != None else missing_d[trait_key] for val in trait_vals]
        #                 # legend = False
        #             else:
        #                 attrib_vals = [missing_d[trait_key] for ik in trait_vals]
        #                 d = {'attrib_vals': None}
        #                 # legend = False
        #
        #         trait_vals = [val if val is not None else 'unknown' for val in trait_vals]
        #         legend_d[trait_key] = trait_vals
        #     trait_d[trait_key] = d
        #
        # if type(scatter_kwargs) == dict:
        #     scatter_kwargs['s'] = trait_d['size']['attrib_vals']
        # elif scatter_kwargs is None:
        #     scatter_kwargs = {'s': trait_d['size']['attrib_vals'], 'edgecolors':'w'}
        #
        # if type(lgd_kwargs) != dict:
        # #     lgd_kwargs['bbox_to_anchor'] = (1.3, 1)
        # # else:
        #     lgd_kwargs = {'loc': 'upper left', 'bbox_to_anchor': (1.3, 1)}
        #
        # lats = [geos.lat for geos in self.series_list]
        # lons = [geos.lon for geos in self.series_list]
        # if legend_attribute is None:
        #     ptlabels = [None for ik in lats]
        # else:
        #     ptlabels = legend_d[legend_attribute]
        #
        # res = mp.map(lats, lons, ptlabels, marker=trait_d['marker']['attrib_vals'],
        #              color=trait_d['hue']['attrib_vals'],
        #              projection=projection, proj_default=proj_default,
        #              background=background, borders=borders, rivers=rivers, lakes=lakes,
        #              figsize=figsize, ax=ax, scatter_kwargs=scatter_kwargs, legend=legend,
        #              lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)
        #
        # if legend == True:
        #     handles = []
        #     labels = []
        #     for key in trait_d:
        #         if 'leg' in trait_d[key]:
        #             handles += trait_d[key]['leg']['handles']
        #             labels += trait_d[key]['leg']['labels']
        #             handles.append(copy.copy(handles[0]))
        #             handles[-1].set_alpha(0)
        #             labels.append('')
        #     res[1].legend(handles, labels,  **lgd_kwargs)#bbox_to_anchor=(1, 1),loc="upper left")
        #
        #     # if hue is not None:
        #     #     if 'mapping' not in trait_d['hue']:
        #     #         if 'mappable' in trait_d['hue']:
        #     #             cb2 = plt.colorbar(trait_d['hue']['mappable'], ax=ax, shrink=0.55, aspect=20 * 0.7,
        #     #                                orientation='vertical', label=hue)
        #
        # return res


    def pca(self, weights=None,missing='fill-em',tol_em=5e-03, max_em_iter=100,**pca_kwargs):
        '''Principal Component Analysis (Empirical Orthogonal Functions)

        Decomposition of MultipleGeoSeries object in terms of orthogonal basis functions.
        Tolerant to missing values, infilled by an EM algorithm.

        Do make sure the time axes are aligned, however! (e.g. use `common_time()`)

        Algorithm from statsmodels: https://www.statsmodels.org/stable/generated/statsmodels.multivariate.pca.PCA.html

        Parameters
        ----------

        weights : ndarray, optional

            Series weights to use after transforming data according to standardize
            or demean when computing the principal components.

        missing : {str, None}

            Method for missing data.  Choices are:

            * 'drop-row' - drop rows with missing values.
            * 'drop-col' - drop columns with missing values.
            * 'drop-min' - drop either rows or columns, choosing by data retention.
            * 'fill-em' - use EM algorithm to fill missing value [ default].  ncomp should be
              set to the number of factors required.
            * `None` raises if data contains NaN values.

        tol_em : float

            Tolerance to use when checking for convergence of the EM algorithm.

        max_em_iter : int

            Maximum iterations for the EM algorithm.

        Returns
        -------

        res: MultivariateDecomp

            Resulting pyleoclim.MultivariateDecomp object

        See also
        --------

        pyleoclim.utils.tsutils.eff_sample_size : Effective Sample Size of timeseries y

        pyleoclim.core.multivardecomp.MultivariateDecomp : The multivariate decomposition object

        pyleoclim.core.mulitpleseries.MulitpleSeries.common_time : align time axes

        Examples
        --------

        .. jupyter-execute::

            from pylipd.utils.dataset import load_dir
            lipd = load_dir(name='Pages2k') # this loads a small subset of the PAGES 2k database
            lipd_euro = lipd.filter_by_geo_bbox(-20,20,40,80)
            df = lipd_euro.get_timeseries_essentials()
            dfs = df.query("archiveType in ('tree') & paleoData_variableName not in ('year')") 
            # place in a MultipleGeoSeries object
            ts_list = []
            for _, row in dfs.iterrows():
                ts_list.append(pyleo.GeoSeries(time=row['time_values'],value=row['paleoData_values'],
                                               time_name=row['time_variableName'],value_name=row['paleoData_variableName'],
                                               time_unit=row['time_units'], value_unit=row['paleoData_units'],
                                               lat = row['geo_meanLat'], lon = row['geo_meanLon'],
                                               archiveType = row['archiveType'], verbose = False,
                                               label=row['dataSetName']+'_'+row['paleoData_variableName']))

            Euro2k = pyleo.MultipleGeoSeries(ts_list, label='Euro2k',time_unit='years AD')

            res = Euro2k.common_time().pca() # carry out PCA
            type(res) # the result is a MultivariateDecomp object

            res.screeplot() # plot the eigenvalue spectrum
            res.modeplot() # plot the first mode, equivalent to res.modeplot(index=0)
            res.modeplot(index=1) # plot the second mode (note the zero-based indexing)
        '''
        # extract geographical coordinate
        lats = np.array([ts.lat for ts in self.series_list])
        lons = np.array([ts.lon for ts in self.series_list])
        locs = np.column_stack([lats,lons])

        # apply PCA fom parent class
        pca_res = super().pca(weights=weights,missing=missing,tol_em=tol_em,
                           max_em_iter=max_em_iter,**pca_kwargs)
        # add geographical information
        # pca_res.locs = None
        pca_res.orig = self

        return pca_res
        