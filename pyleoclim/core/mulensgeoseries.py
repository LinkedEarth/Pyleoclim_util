"""
A MultipleEnsembleGeoSeries object is a collection (more precisely, a 
list) of EnsembleGeoSeries objects. This class currently exists primarily for the application of MC-PCA
"""

from tqdm import tqdm

import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FormatStrFormatter)
import matplotlib.transforms as transforms

from ..core.series import Series
from ..core.multiplegeoseries import MultipleGeoSeries
from ..core.ensmultivardecomp import EnsMultivarDecomp
from ..utils import plotting

class MulEnsGeoSeries():
    def __init__(self, ensemble_series_list,label=None):
        self.ensemble_series_list = ensemble_series_list
        self.label = label

    def mcpca(self,nsim=1000, seed=None, common_time_kwargs=None, pca_kwargs=None,align_method='dot',delta=0,phase_max=.5):
        '''Function to conduct Monte-Carlo PCA using ensembles included in this object
        
        Parameters
        ----------
        nsim : int
            Number of simulations to carry out. Default is 1000
            
        seed : int
            Seed to use for random calculations
            
        common_time_kwargs : dict
            Key word arguments for MultipleSeries.common_time()
            
        pca_kwargs : dict
            Key word arguments for MultipleGeoSeries.pca()

        align_method : str; {'correlation','dot','phase','cosine'}
            How to align pcs. 
            Correlation computes the correlation between individual principal components and the first set of PCs, flipping those that have a negative correlation.
            Dot computes the dot product between the eigenvectors, and flips those that have a negative dot product.
            Phase computes the phase difference between the eigenvectors, and flips those that have a phase difference greater than phase_max.
            Cosine computes the cosine similarity between the eigenvectors, and flips those that have a negative cosine similarity.

        delta : float
            A number between -1 and 1, that serves as the threshold for the correlation, dot, and cosine methods. Default is 0.

        phase_max : float
            A float between 0 and 1, that serves as the threshold for the phase method. Default is 0.5.
            
        Returns
        -------
        EnsembleMvD : pyleo.EnsMultivarDecomp
            Ensemble Multivariate Decomposition object
            
        Examples
        --------
        
        .. jupyter-execute::
        
            n = 3 # number of ensembles
            nn = 30 # number of noise realizations
            nt = 500
            ens_list = []

            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            signal = pyleo.Series(t,v)

            for _ in range(n): 
                series_list = []
                lat = np.random.randint(-90,90)
                lon = np.random.randint(-180,180)
                for idx in range(nn):  # noise
                    noise = np.random.randn(nt,nn)*100
                    ts = pyleo.GeoSeries(time=signal.time, value=signal.value+noise[:,idx], lat=lat, lon=lon, verbose=False)
                    series_list.append(ts)

                ts_ens = pyleo.EnsembleGeoSeries(series_list)
                ens_list.append(ts_ens)

            mul_ens = pyleo.MulEnsGeoSeries(ens_list)
            mul_ens.mcpca(nsim=10,seed=42)'''
        
        common_time_kwargs = {} if common_time_kwargs is None else common_time_kwargs.copy()
        pca_kwargs = {} if pca_kwargs is None else pca_kwargs.copy()

        if seed:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        pca_list = []

        for i in tqdm(range(nsim),desc='Iterating over simulations'):
            ensemble_list = []
            for ensemble in self.ensemble_series_list:
                rng_index = rng.integers(low=0,high=len(ensemble.series_list))
                ensemble_list.append(ensemble.series_list[rng_index])
            mgs_tmp = MultipleGeoSeries(ensemble_list).common_time(**common_time_kwargs)
            pca_tmp = mgs_tmp.pca(**pca_kwargs)

            #Create reference pca
            if i == 0:
                base_pca = pca_tmp
                base_pcs = pca_tmp.pcs
                base_t = pca_tmp.orig.series_list[0].time
                base_eigvecs = pca_tmp.eigvecs.T
                pca_list.append(pca_tmp)
                continue

            if align_method == 'correlation':
                for idx,pcs in enumerate(pca_tmp.pcs.T):
                    t = pca_tmp.orig.series_list[0].time
                    pc_series = Series(time=t,value=pcs,verbose=False)
                    base_pc_series = Series(time=base_t,value=base_pcs[:,idx],verbose=False)
                    r = pc_series.correlation(base_pc_series,settings={'nsim':1},seed=seed,mute_pbar=True).r
                    if r > delta:
                        pass
                    else:
                        pca_tmp.eigvecs[:,idx] *= -1
                        pca_tmp.pcs[:,idx] *= -1
            elif align_method == 'phase':
                for idx,pcs in enumerate(pca_tmp.pcs.T):
                    t = pca_tmp.orig.series_list[0].time
                    pc_series = Series(time=t,value=pcs,verbose=False)
                    base_pc_series = Series(time=base_t,value=base_pcs[:,idx],verbose=False)
                    filtered_pcs = pc_series.filter(method='lanczos',cutoff_freq=.005)
                    filtered_base_pcs = base_pc_series.filter(method='lanczos',cutoff_freq=.005)
                    correlation = sp.signal.correlate(filtered_base_pcs.value, filtered_pcs.value, mode="full")
                    lags = sp.signal.correlation_lags(filtered_base_pcs.value.size, filtered_pcs.value.size, mode="full")
                    shift = np.abs(lags[np.argmax(correlation)])/(len(lags)/2)
                    # plt.plot(np.linspace(-180,180,len(correlation)),correlation)
                    if shift < phase_max:
                        pass
                    else:
                        pca_tmp.eigvecs[:,idx] *= -1
                        pca_tmp.pcs[:,idx] *= -1
            elif align_method == 'dot':
                for idx,eigvecs in enumerate(pca_tmp.eigvecs.T):
                    dot = np.dot(eigvecs,base_eigvecs[idx])
                    if dot > delta:
                        pass
                    else:
                        pca_tmp.eigvecs[:,idx] *= -1
                        pca_tmp.pcs[:,idx] *= -1
            elif align_method == 'cosine':
                for idx,eigvecs in enumerate(pca_tmp.eigvecs.T):
                    cos = sp.spatial.distance.cosine(eigvecs,base_eigvecs[idx])
                    if cos > delta:
                        pass
                    else:
                        pca_tmp.eigvecs[:,idx] *= -1
                        pca_tmp.pcs[:,idx] *= -1
            else:
                raise ValueError('Align method not recognized. Please pass "phase", "cosine", "correlation", or "dot".')

            pca_list.append(pca_tmp)

        # assign name
        if self.label is not None:
            label = self.label + ' PCA'
        else:
            label = 'PCA of unlabelled object'

        EnsembleMvD = EnsMultivarDecomp(pca_list=pca_list,label=label)
        return EnsembleMvD

    def stackplot(self, figsize=None, savefig_settings=None, time_unit = None, 
                  xlim=None, colors=None, cmap='tab10', plot_style= 'envelope',
                  norm=None, labels='auto', ylabel_fontsize = 8, spine_lw=1.5, 
                  grid_lw=0.5, label_x_loc=-0.15, v_shift_factor=3/4, 
                  yticks_minor = False, xticks_minor = False, ylims ='auto',
                  plot_kwargs=None, common_time_kwargs=None):
        ''' Stack plot of multiple ensemble series

        Time units are harmonized prior to plotting. 
        Functionally, this method is very similar to the stackplot method of MultipleSeries, see the documentation there for more details on customization.
        Note that the plotting plot_style is uniquely designed for this one and cannot be properly reset with `pyleoclim.set_plot_style()`.

        Parameters
        ----------
        figsize : list
        
            Size of the figure.
            
        savefig_settings : dictionary
        
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.
            
        time_unit : str
        
            the target time unit, possible inputs:
            {
                'year', 'years', 'yr', 'yrs',
                'y BP', 'yr BP', 'yrs BP', 'year BP', 'years BP',
                'ky BP', 'kyr BP', 'kyrs BP', 'ka BP', 'ka',
                'my BP', 'myr BP', 'myrs BP', 'ma BP', 'ma',
            }
            default is None, in which case the code picks the most common time unit in the collection.
            If no discernible winner can be found, the unit of the first series in the collection is used. 
            
        xlim : list
        
            The x-axis limit.
            
        colors : a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)
        
            Colors for plotting.
            If None, the plotting will cycle the 'tab10' colormap;
            if only one color is specified, then all curves will be plotted with that single color;
            if a list of colors are specified, then the plotting will cycle that color list.
            
        cmap : str
        
            The colormap to use when "colors" is None.
            Note that the function will try to detect continuous or discrete colormaps, and set the norm accordingly.

        plot_style : str; {'envelope', 'traces'}

            The ensemble plotting style to use. Default is 'envelope'.

        norm : matplotlib.colors.Normalize like
        
            The normalization for the colormap.
            If None, a linear normalization will be used.
            
        labels: None, 'auto' or list
        
            If None, doesn't add labels to the subplots
            If 'auto', uses the labels passed during the creation of pyleoclim.Series
            If list, pass a list of strings for each labels.
            Default is 'auto'
            
        spine_lw : float
        
            The linewidth for the spines of the axes.
            
        grid_lw : float
        
            The linewidth for the gridlines.
            
        label_x_loc : float
        
            The x location for the label of each curve.
            
        v_shift_factor : float
        
            The factor for the vertical shift of each axis.
            The default value 3/4 means the top of the next axis will be located at 3/4 of the height of the previous one.
            
        ylabel_fontsize : int
            
            Size for ylabel font. Default is 8, to avoid crowding. 
            
        yticks_minor : bool

            Whether the y axes should contain minor ticks (use sparingly!). Default: False
        
        xticks_minor : bool 

            Whether the x axis should contain minor ticks. Default: False
        
        ylims : str {'spacious', 'auto'}

            Method for determining the limits of the y axes. 
            Default is 'spacious', which is mean +/- 4 x std
            'auto' activates the Matplotlib default
            
        plot_kwargs: dict or list of dict

            Arguments to further customize the plot from EnsembleSeries.plot_envelope or EnsembleSeries.plot_traces, depending on the chosen style.

            - Dictionary: Arguments will be applied to all lines in the stackplots
            - List of dictionaries: Allows to customize one line at a time.

        common_time_kwargs : dict

            Arguments to pass to the common_time method of the ensemble series.
            Common time is called to calculate the median of the ensemble series for tick purposes, and is also used if plot_style is set to 'envelope'.

        Returns
        -------
        fig : matplotlib.figure
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html) for details.

        ax : matplotlib.axis
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        See also
        --------

        pyleoclim.core.multipleseries.MultipleSeries.stackplot : Stack plot of multiple series

        pyleoclim.core.ensembleseries.EnsembleSeries.plot_envelope : Plotting the envelope of an ensemble of series

        pyleoclim.core.ensembleseries.EnsembleSeries.plot_traces : Plotting the traces of an ensemble of series

        pyleoclim.utils.plotting.savefig : Saving figure in Pyleoclim

        Examples
        --------
        
        .. jupyter-execute::
            
            n = 3 # number of ensembles
            nn = 30 # number of noise realizations
            nt = 500
            ens_list = []

            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            signal = pyleo.Series(t,v)

            for _ in range(n): 
                series_list = []
                lat = np.random.randint(-90,90)
                lon = np.random.randint(-180,180)
                for idx in range(nn):  # noise
                    noise = np.random.randn(nt,nn)*100
                    ts = pyleo.GeoSeries(time=signal.time, value=signal.value+noise[:,idx], lat=lat, lon=lon, verbose=False)
                    series_list.append(ts)

                ts_ens = pyleo.EnsembleGeoSeries(series_list)
                ens_list.append(ts_ens)

            mul_ens = pyleo.MulEnsGeoSeries(ens_list)
            mul_ens.stackplot()

        If you'd like to adjust the plot parameters, you can pass them via the `plot_kwargs` argument.

        .. jupyter-execute::
            
            n = 3 # number of ensembles
            nn = 30 # number of noise realizations
            nt = 500
            ens_list = []

            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            signal = pyleo.Series(t,v)

            for _ in range(n): 
                series_list = []
                lat = np.random.randint(-90,90)
                lon = np.random.randint(-180,180)
                for idx in range(nn):  # noise
                    noise = np.random.randn(nt,nn)*100
                    ts = pyleo.GeoSeries(time=signal.time, value=signal.value+noise[:,idx], lat=lat, lon=lon, verbose=False)
                    series_list.append(ts)

                ts_ens = pyleo.EnsembleGeoSeries(series_list)
                ens_list.append(ts_ens)

            mul_ens = pyleo.MulEnsGeoSeries(ens_list)
            mul_ens.stackplot(plot_kwargs={'shade_alpha':0.5})

        If you'd like to plot traces instead, you can use the modify the `plot_style` argument

        .. jupyter-execute::

            n = 3 # number of ensembles
            nn = 30 # number of noise realizations
            nt = 500
            ens_list = []

            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            signal = pyleo.Series(t,v)

            for _ in range(n): 
                series_list = []
                lat = np.random.randint(-90,90)
                lon = np.random.randint(-180,180)
                for idx in range(nn):  # noise
                    noise = np.random.randn(nt,nn)*100
                    ts = pyleo.GeoSeries(time=signal.time, value=signal.value+noise[:,idx], lat=lat, lon=lon, verbose=False)
                    series_list.append(ts)

                ts_ens = pyleo.EnsembleGeoSeries(series_list)
                ens_list.append(ts_ens)

            mul_ens = pyleo.MulEnsGeoSeries(ens_list)
            mul_ens.stackplot(plot_style='traces')
        '''
        # Create a figure with a specified size
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        common_time_kwargs = {} if common_time_kwargs is None else common_time_kwargs.copy()

        fig = plt.figure(figsize=figsize)

        n_ts = len(self.ensemble_series_list)

        #deal with time units
        self.ensemble_series_list = [ens.convert_time_unit(time_unit) for ens in self.ensemble_series_list]

        if type(labels)==list:
            if len(labels) != n_ts:
                raise ValueError("The length of the label list should match the number of timeseries to be plotted")
    
        # Deal with plotting arguments
        if type(plot_kwargs)==dict:
            plot_kwargs = [plot_kwargs.copy() for _ in range(n_ts)]

        if plot_kwargs is not None and len(plot_kwargs) != n_ts:
            raise ValueError("When passing a list of dictionaries for kwargs arguments, the number of items should be the same as the number of timeseries")

        if xlim is None:
            time_min = np.inf
            time_max = -np.inf
            for ens in self.ensemble_series_list:
                for ts in ens.series_list:
                    if np.min(ts.time) <= time_min:
                        time_min = np.min(ts.time)
                    if np.max(ts.time) >= time_max:
                        time_max = np.max(ts.time)
            xlim = [time_min, time_max]

        ax = {}
        left = 0
        width = 1
        height = 1 / n_ts
        bottom = 1

        # Iterate over each pair in preprocessed_series_dict
        for idx, ens in enumerate(self.ensemble_series_list):    
            if colors is None:
                cmap_obj = plt.get_cmap(cmap)

                #If the color map has way more colors than the number of time series, limit the number of colors used for the norm to the number of time series
                if hasattr(cmap_obj, 'colors'):
                    if len(cmap_obj.colors) > (n_ts*15):
                        nc = n_ts
                    else:
                        nc = len(cmap_obj.colors)
                else:
                    nc = n_ts
            
                if norm is None:
                    norm = mpl.colors.Normalize(vmin=0, vmax=nc-1)

                color = cmap_obj(norm(idx%nc))
            elif type(colors) is str:
                color = colors
            elif type(colors) is list:
                nc = len(colors)
                color = colors[idx%nc]
            else:
                raise TypeError('"colors" should be a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)')

            #deal with other plotting arguments
            if plot_kwargs is None:
                p_kwargs = {}
            else:
                p_kwargs = plot_kwargs[idx]

                print(p_kwargs)
            bottom -= height * v_shift_factor

            ax[idx] = fig.add_axes([left, bottom, width, height])

            #Convert time unit to target and create shared time axis version of ens
            ens_common = ens.common_time(**common_time_kwargs)

            if plot_style == 'envelope':
                # Plot the ensemble envelope
                if 'shade_clr' not in p_kwargs:
                    p_kwargs['shade_clr'] = color
                if 'curve_clr' not in p_kwargs:
                    p_kwargs['curve_clr'] = color
                print(p_kwargs)
                ens_common.plot_envelope(ax=ax[idx], **p_kwargs)
            elif plot_style == 'traces':
                # Plot the ensemble traces
                if 'color' not in p_kwargs:
                    p_kwargs['color'] = color
                ens.plot_traces(ax=ax[idx], **p_kwargs)

            # Set plot properties for the main axis
            ax[idx].patch.set_alpha(0)
            ax[idx].set_xlim(xlim)
            time_label,value_label = ens.series_list[0].make_labels()
            ax[idx].set_ylabel(value_label, weight='bold', size=ylabel_fontsize)

            median_ts = ens_common.quantiles(qs=[.5]).series_list[0]

            mu = np.nanmean(median_ts.value)
            std = np.nanstd(median_ts.value)
            trans = transforms.blended_transform_factory(ax[idx].transAxes, ax[idx].transData)
            
            if labels == 'auto':
                if ens.label is not None:
                    ax[idx].text(label_x_loc, mu, ens.label, horizontalalignment='right', transform=trans, color=color, weight='bold')
            elif type(labels) ==list:
                ax[idx].text(label_x_loc, mu, labels[idx], horizontalalignment='right', transform=trans, color=color, weight='bold')
            elif labels==None:
                pass
            
            ylim = [mu-4*std, mu+4*std]

            if ylims == 'spacious':
                ax[idx].set_ylim(ylim)                
                
            if yticks_minor is True:    
                ax[idx].yaxis.set_minor_locator(AutoMinorLocator())
                ax[idx].tick_params(which='major', length=7, width=1.5)
                ax[idx].tick_params(which='minor', length=3, width=1, color=color)
            else:
                ax[idx].set_yticks(ylim)
            ax[idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # Set spine and tick properties based on index
            if idx % 2 == 0:
                ax[idx].spines['left'].set_visible(True)
                ax[idx].spines['left'].set_linewidth(spine_lw)
                ax[idx].spines['left'].set_color(color)
                ax[idx].spines['right'].set_visible(False)
                ax[idx].yaxis.set_label_position('left')
                ax[idx].yaxis.tick_left()
            else:
                ax[idx].spines['left'].set_visible(False)
                ax[idx].spines['right'].set_visible(True)
                ax[idx].spines['right'].set_linewidth(spine_lw)
                ax[idx].spines['right'].set_color(color)
                ax[idx].yaxis.set_label_position('right')
                ax[idx].yaxis.tick_right()

            # Set additional plot properties
            ax[idx].yaxis.label.set_color(color)
            ax[idx].tick_params(axis='y', colors=color)
            ax[idx].spines['top'].set_visible(False)
            ax[idx].spines['bottom'].set_visible(False)
            ax[idx].tick_params(axis='x', which='both', length=0)
            ax[idx].set_xlabel('')
            ax[idx].set_xticklabels([])
            ax[idx].legend([])
            xt = ax[idx].get_xticks()[1:-1]
            for x in xt:
                ax[idx].axvline(x=x, color='lightgray', linewidth=grid_lw, ls='-', zorder=-1)
            ax[idx].axhline(y=0, color='lightgray', linewidth=grid_lw, ls='-', zorder=-1)

        # Set up the x-axis label at the bottom
        bottom -= height * (1 - v_shift_factor)
        ax['x_axis'] = fig.add_axes([left, bottom, width, height])
        ax['x_axis'].set_xlabel(time_label)
        ax['x_axis'].spines['left'].set_visible(False)
        ax['x_axis'].spines['right'].set_visible(False)
        ax['x_axis'].spines['bottom'].set_visible(True)
        ax['x_axis'].spines['bottom'].set_linewidth(spine_lw)
        ax['x_axis'].set_yticks([])
        ax['x_axis'].patch.set_alpha(0)
        ax['x_axis'].set_xlim(xlim)
        ax['x_axis'].grid(False)
        ax['x_axis'].tick_params(axis='x', which='both', length=3.5)

        for x in xt:
            ax['x_axis'].axvline(x=x, color='lightgray', linewidth=grid_lw,
                                   ls='-', zorder=-1)
        if xticks_minor is True:
            ax['x_axis'].xaxis.set_minor_locator(AutoMinorLocator())
            ax['x_axis'].tick_params(which='major', length=7, width=1.5)
            ax['x_axis'].tick_params(which='minor', length=3, width=1)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax