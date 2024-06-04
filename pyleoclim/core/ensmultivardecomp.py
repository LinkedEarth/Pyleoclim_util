"""
A MultipleMultivariateDecomp object is a collection (more precisely, a 
list) of MultivariateDecomp objects. This class currently exists primarily for the storage of MC-PCA products
"""

import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib import cm

from .series import Series
from .ensembleseries import EnsembleSeries
from ..utils import mapping, plotting

class EnsMultivarDecomp():
    def __init__(self, pca_list,label=None):
        self.pca_list = pca_list
        self.label = label

    def modeplot(self,index=0,flip=False,plot_envelope_kwargs = None, psd_envelope_kwargs = None,
                 figsize=[8, 8], fig=None, savefig_settings=None,gs=None,
                 title=None, title_kwargs=None, spec_method='mtm', cmap='coolwarm',
                 marker=None, scatter_kwargs=None,
                 map_kwargs=None, gridspec_kwargs=None,quantiles=[.25,.5,.75]):
        '''Plot relevant information about the specific mode
        
        Parameters
        ----------
        index : int
            The 1-based index of the mode to visualize.
            Default is 1, corresponding to the first mode.
            
        flip : bool
            Whether or not to flip the PC
            
        plot_envelope_kwargs : dict
            Dictionary of key word arguments for plot envelope

        plot_envelope_kwargs : dict
            Dictionary of key word arguments for psd envelope
            
        figsize : list, optional
            The figure size. The default is [8, 8].

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        title : str, optional
            text for figure title

        title_kwargs : dict
            the keyword arguments for ax.set_title()

        gs : matplotlib.gridspec object, optional
            Requires at least two rows and two columns.
            - top row, left: timeseries of principle component
            - top row, right: PSD
            - bottom row: spaghetti plot or map
            See [matplotlib.gridspec.GridSpec](https://matplotlib.org/stable/tutorials/intermediate/gridspec.html) for details.

        gridspec_kwargs : dict, optional
            Dictionary with custom gridspec values.
            - wspace changes space between columns (default: wspace=0.05)
            - hspace changes space between rows (default: hspace=0.03)
            - width_ratios: relative width of each column (default: width_ratios=[5,1,3] where middle column serves as a spacer)
            - height_ratios: relative height of each row (default: height_ratios=[2,1,5] where middle row serves as a spacer)

        spec_method: str, optional
            The name of the spectral method to be applied on the PC. Default: MTM
            Note that the data are evenly-spaced, so any spectral method that
            assumes even spacing is applicable here:  'mtm', 'welch', 'periodogram'
            'wwz' is relevant if scaling exponents need to be estimated, but ill-advised otherwise, as it is very slow.
            
        cmap: str, optional
            if 'hue' is specified, will be used for map scatter plot values.
            colormap name for the loadings (https://matplotlib.org/stable/tutorials/colors/colormaps.html)

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
            
        scatter_kwargs : dict, optional
            Optional arguments configuring how data are plotted on a map. See description of scatter_kwargs in pyleoclim.utils.mapping.scatter_map

        hue : str, optional
            (only applicable if using scatter map) Variable associated with color coding for points plotted on map. May correspond to a continuous or categorical variable.
            The default is 'EOF'.

        marker : string, optional
            (only applicable if using scatter map) Grouping variable that will produce points with different markers. Can have a numeric dtype but will always be treated as categorical.
            The default is 'archiveType'.

        quantiles : list,array
            Quantiles to use for plotting EOFs
            
        Returns
        -------
        fig : matplotlib.figure
            The figure

        ax : dict
            dictionary of matplotlib ax

        See also
        --------

        pyleoclim.core.MultipleSeries.pca : Principal Component Analysis

        pyleoclim.core.MultipleGeoSeries.pca : Principal Component Analysis
        
        pyleoclim.utils.tsutils.eff_sample_size : Effective sample size

        pyleoclim.utils.mapping.scatter_map : mapping
        
        Examples
        --------
        
        .. jupyter-execute::

            n = 100 # number of series

            soi = pyleo.utils.load_dataset('SOI')
            soi_time_axes = [pyleo.utils.random_time_axis(n=len(soi.time)) for _ in range(n)]
            soi_ens = pyleo.EnsembleGeoSeries([pyleo.GeoSeries(time=time, value=soi.value,lat=-5,lon=-85,auto_time_params=True,verbose=False) for time in soi_time_axes])

            nino3 = pyleo.utils.load_dataset('NINO3')
            nino3_time_axes = [pyleo.utils.random_time_axis(n=len(nino3.time)) for _ in range(n)]
            nino3_ens = pyleo.EnsembleGeoSeries([pyleo.GeoSeries(time=time, value=nino3.value,lat=-5,lon=-85,auto_time_params=True,verbose=False) for time in nino3_time_axes])

            mul_ens = pyleo.MulEnsGeoSeries([nino3_ens,soi_ens])
            mcpca = mul_ens.mcpca(nsim=10,seed=42)
            mcpca.modeplot()'''

        plot_envelope_kwargs = {} if plot_envelope_kwargs is None else plot_envelope_kwargs.copy()
        psd_envelope_kwargs = {} if psd_envelope_kwargs is None else psd_envelope_kwargs.copy()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        if fig ==None:
            fig = plt.figure(figsize=figsize)
            
        if gs == None:
            gridspec_kwargs = {} if type(gridspec_kwargs) != dict else gridspec_kwargs
            gridspec_defaults = dict(wspace=0.05, hspace=0.03, width_ratios=[5,1,3],
                                     height_ratios=[2,1,5])
            gridspec_defaults.update(gridspec_kwargs)
            gs = GridSpec(len(gridspec_defaults['height_ratios']), len(gridspec_defaults['width_ratios']), **gridspec_defaults)

        gs.update(left=0, right=1.1)

        ax = {}
        # plot the PC
        ax['pc'] = fig.add_subplot(gs[0, 0])
        label = rf'$PC_{index + 1}$' 

        t_list = []
        pc_list = []
        eof_array = np.empty(shape=(len(self.pca_list[0].eigvecs[:,index]),len(self.pca_list)))
        pctvar_array = np.empty(shape=len(self.pca_list))
        for idx,pca in enumerate(self.pca_list):
            if flip:
                PC = -pca.pcs[:, index]
                EOF = -pca.eigvecs[:, index]
                t = pca.orig.series_list[0].time
            else:
                PC = pca.pcs[:, index]
                EOF = pca.eigvecs[:, index]
                t = pca.orig.series_list[0].time
            t_list.append(t)
            pc_list.append(PC)
            eof_array[:,idx] = EOF
            pctvar_array[idx]= pca.pctvar[index]
        eof_quantiles = np.quantile(a=eof_array,q=quantiles,axis=1)

        time_unit = self.pca_list[0].orig.series_list[0].time_unit
        pca_ens = EnsembleSeries([Series(time=t_list[idx],value=pc_list[idx],time_unit=time_unit,verbose=False) for idx in range(len(self.pca_list))])

        if 'plot_legend' not in plot_envelope_kwargs.keys():
            plot_envelope_kwargs['plot_legend'] = False

        pca_ens.common_time().plot_envelope(ax=ax['pc'],**plot_envelope_kwargs)
        ax['pc'].set_ylabel(label)

        # plot its PSD
        ax['psd'] = fig.add_subplot(gs[0, -1])

        if 'plot_legend' not in psd_envelope_kwargs.keys():
            psd_envelope_kwargs['plot_legend'] = False
        if 'members_plot_num' not in psd_envelope_kwargs.keys():
            psd_envelope_kwargs['members_plot_num'] = 0

        psd = pca_ens.common_time().spectral(method=spec_method)
        _ = psd.plot_envelope(ax=ax['psd'],**psd_envelope_kwargs)

        # plot spatial pattern or spaghetti
        map_kwargs = {} if map_kwargs is None else map_kwargs.copy()

        projection = map_kwargs.pop('projection', 'auto')
        proj_default = map_kwargs.pop('proj_default', True)
        lakes = map_kwargs.pop('lakes', False)
        land = map_kwargs.pop('land', False)
        ocean = map_kwargs.pop('ocean', False)
        rivers = map_kwargs.pop('rivers', False)
        borders = map_kwargs.pop('borders', True)
        coastline = map_kwargs.pop('coastline', True)
        background = map_kwargs.pop('background', True)
        extent = map_kwargs.pop('extent', 'global')

        map_gridspec_kwargs = map_kwargs.pop('gridspec_kwargs', {})
        lgd_kwargs = map_kwargs.pop('lgd_kwargs', {})

        if 'edgecolor' in map_kwargs.keys():
            scatter_kwargs.update({'edgecolor': map_kwargs['edgecolor']})

        legend = map_kwargs.pop('legend', True)
        colorbar = map_kwargs.pop('colorbar', True)

        # This makes a bare bones dataframe from a MultipleGeoSeries object
        df_list = []
        for idx,q in enumerate(quantiles):
            eof_q = eof_quantiles[idx]
            df_tmp = mapping.make_df(self.pca_list[0].orig, hue='quantile', marker=marker, size='EOF')
            # additional columns are added manually
            df_tmp['EOF'] = np.abs(eof_q)
            df_tmp['quantile'] = q*100
            df_tmp['quantile'][eof_q<0] *= -1
            df_list.append(df_tmp)
        df = pd.concat(df_list)
        df = df.sort_values(['lat','EOF'],ascending=False)#Make sure large sizes plot first

        if legend == True:
            map_gridspec_kwargs['width_ratios'] = map_gridspec_kwargs['width_ratios'] if 'width_ratios' in map_gridspec_kwargs.keys() else [.7,.1, 12, 4]

        ax_norm = mpl.colors.Normalize(vmin=min(df['quantile'].to_numpy()), vmax=max(df['quantile'].to_numpy()), clip=False)
        ax_cmap = plt.get_cmap(name=cmap)
        ax_sm = cm.ScalarMappable(norm=ax_norm, cmap=ax_cmap)

        _, ax['map'] = mapping.scatter_map(df, hue='quantile', size='EOF', marker=marker, projection=projection,
                                            proj_default=proj_default,
                                            background=background, borders=borders, coastline=coastline,
                                            rivers=rivers, lakes=lakes,
                                            ocean=ocean, land=land, extent=extent,
                                            figsize=None, scatter_kwargs=scatter_kwargs, lgd_kwargs=lgd_kwargs,
                                            gridspec_kwargs=map_gridspec_kwargs, colorbar=False,
                                            legend=legend, cmap=ax_sm.cmap,
                                            fig=fig, gs_slot=gs[-1, :]) #label rf'$EOF_{index + 1}$'

        if title is None:
            title = self.label + ' mode ' + str(index + 1) + ', ' + '{:3.2f}'.format(pctvar_array.mean()) + u" \u00B1 " + '{:3.2f}'.format(pctvar_array.std()) + '% variance explained'
                      # weight='bold', y=0.92)

        title_kwargs = {} if title_kwargs is None else title_kwargs.copy()
        t_args = {'y': .92, 'weight': 'bold'}
        t_args.update(title_kwargs)
        fig.suptitle(title, **t_args)
        
        fig.tight_layout()
        
        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)

        return fig, ax
    
    def screeplot(self,clr_eig='C0', linewidth=.3, title='Screeplot', violin_kwargs = None, figsize=[8, 8],savefig_settings=None,ax=None):
        '''Function to plot the scree plot of the PCA
        
        Parameters
        ----------
        quantiles : list,array
            Quantiles to use for plotting range of pctvar for each mode.

        clr_eig : str
            Color to use for the eigenvalues. Default is 'C0'.

        linewidth : float
            Linewidth to use for the violin plot. Default is 0.3.

        violin_kwargs : dict
            Dictionary of key word arguments for violin plot.
            If exposed plot arguments {'color','linewidth','title'} are included in kwargs, kwargs will overwrite the exposed argument values.
        
        figsize : list, optional
            The figure size. The default is [8, 8].
            
        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        ax : matplotlib.axis, optional
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.
            
        Returns
        -------
        fig : matplotlib.figure
            The figure
        
        ax : dict
            dictionary of matplotlib ax
            
        Examples
        --------
        
        .. jupyter-execute::
        
            n = 100 # number of series

            soi = pyleo.utils.load_dataset('SOI')
            soi_time_axes = [pyleo.utils.random_time_axis(n=len(soi.time)) for _ in range(n)]
            soi_ens = pyleo.EnsembleGeoSeries([pyleo.GeoSeries(time=time, value=soi.value,lat=0,lon=0,auto_time_params=True,verbose=False) for time in soi_time_axes])

            nino3 = pyleo.utils.load_dataset('NINO3')
            nino3_time_axes = [pyleo.utils.random_time_axis(n=len(nino3.time)) for _ in range(n)]
            nino3_ens = pyleo.EnsembleGeoSeries([pyleo.GeoSeries(time=time, value=nino3.value,lat=0,lon=0,auto_time_params=True,verbose=False) for time in nino3_time_axes])

            mul_ens = pyleo.MulEnsGeoSeries([nino3_ens,soi_ens])
            mcpca = mul_ens.mcpca(nsim=10,seed=42)
            mcpca.screeplot()'''
        
        violin_kwargs = {} if violin_kwargs is None else violin_kwargs.copy()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        
        if ax is None:
            fig,ax = plt.subplots(figsize=figsize)

        modes = np.arange(len(self.pca_list[0].pctvar))
        #pctvar_array = np.empty(shape=(len(modes),len(self.pca_list)))
        data_dict = {}

        for mode in modes:
            data_dict[mode] = np.array([pca.pctvar[mode] for pca in self.pca_list])
            #pctvar_array[mode,:] = np.array([pca.pctvar[mode] for pca in self.pca_list])

        df = pd.DataFrame(columns=['Mode','Pctvar'],dtype='float')
        df['Mode'] = np.arange(1,len(data_dict)+1)
        df['Pctvar'] = data_dict.values()
        df = df.explode('Pctvar').astype({'Pctvar': 'float'})
        # pctvar_quantiles = np.quantile(a=pctvar_array,q=quantiles,axis=1)

        if 'color' in violin_kwargs.keys():
            clr_eig = violin_kwargs.pop('color')

        if 'linewidth' in violin_kwargs.keys():
            linewidth = violin_kwargs.pop('linewidth')
        
        sns.violinplot(data=df,x='Mode',y='Pctvar',linewidth=linewidth,color=clr_eig,ax=ax,**violin_kwargs)
        ax.set_xlabel(r'Mode index $i$')
        ax.set_ylabel(r'$\lambda_i$')

        if title:
            ax.set_title(title)
        
        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)
        
        return fig, ax