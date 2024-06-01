#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Coherence class stores the result of Series.wavelet_coherence(), whether WWZ or CWT.
It includes wavelet transform coherency and cross-wavelet transform.
"""
from ..utils import plotting
from ..utils import wavelet as waveutils
from ..core.scalograms import Scalogram, MultipleScalogram

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib import cm
from matplotlib import gridspec

from tqdm import tqdm
from scipy.stats.mstats import mquantiles
import warnings

class Coherence:
    '''Coherence object, meant to receive the WTC and XWT part of Series.wavelet_coherence()

    See also
    --------

    pyleoclim.core.series.Series.wavelet_coherence : Wavelet coherence method

    '''
    def __init__(self, frequency, scale, time, wtc, xwt, phase, coi=None,
                 wave_method=None, wave_args=None,
                 timeseries1=None, timeseries2=None, signif_qs=None, signif_method=None, qs =None,
                 freq_method=None, freq_kwargs=None, Neff_threshold=3, scale_unit=None, time_label=None):
        self.frequency = np.array(frequency)
        self.time = np.array(time)
        self.scale = np.array(scale)
        self.wtc = np.array(wtc)
        self.xwt = np.array(xwt)
        if coi is not None:
            self.coi = np.array(coi)
        else:
            self.coi = waveutils.make_coi(self.time, Neff_threshold=Neff_threshold)
        self.phase = np.array(phase)
        self.timeseries1 = timeseries1
        self.timeseries2 = timeseries2
        self.signif_qs = signif_qs
        self.signif_method = signif_method
        self.freq_method = freq_method
        self.freq_kwargs = freq_kwargs
        self.wave_method = wave_method
        if wave_args is not None:
            if 'freq' in wave_args.keys():
                wave_args['freq'] = np.array(wave_args['freq'])
            if 'tau' in wave_args.keys():
                wave_args['tau'] = np.array(wave_args['tau'])
        self.wave_args = wave_args
        self.qs        = qs

        if scale_unit is not None:
            self.scale_unit = scale_unit
        elif timeseries1 is not None:
            self.scale_unit = plotting.infer_period_unit_from_time_unit(timeseries1.time_unit)
        elif timeseries2 is not None:
            self.scale_unit = plotting.infer_period_unit_from_time_unit(timeseries2.time_unit)
        else:
            self.scale_unit = None

        if time_label is not None:
            self.time_label = time_label
        elif timeseries1 is not None:
            if timeseries1.time_unit is not None:
                self.time_label = f'{timeseries1.time_name} [{timeseries1.time_unit}]'
            else:
                self.time_label = f'{timeseries1.time_name}'
        elif timeseries2 is not None:
            if timeseries2.time_unit is not None:
                self.time_label = f'{timeseries2.time_name} [{timeseries2.time_unit}]'
            else:
                self.time_label = f'{timeseries2.time_name}'
        else:
            self.time_label = None

    def copy(self):
        '''Copy object
        '''
        return deepcopy(self)

    def plot(self, var='wtc', xlabel=None, ylabel=None, title='auto', figsize=[10, 8],
             ylim=None, xlim=None, in_scale=True, yticks=None, contourf_style={},
             phase_style={}, cbar_style={}, savefig_settings={}, ax=None,
             signif_clr='white', signif_linestyles='-', signif_linewidths=1,
             signif_thresh = 0.95, under_clr='ivory', over_clr='black', bad_clr='dimgray'):
        '''Plot the cross-wavelet results

        Parameters
        ----------
        
        var : str {'wtc', 'xwt'}
        
            variable to be plotted as color field. Default: 'wtc', the wavelet transform coherency.
            'xwt' plots the cross-wavelet transform instead.
        
        xlabel : str, optional
        
            x-axis label. The default is None.
        
        ylabel : str, optional
        
            y-axis label. The default is None.
        
        title : str, optional
        
            Title of the plot. The default is 'auto', where it is made from object metadata.
            To mute, pass title = None.
        
        figsize : list, optional
        
            Figure size. The default is [10, 8].
        
        ylim : list, optional
        
            y-axis limits. The default is None.
        
        xlim : list, optional
        
            x-axis limits. The default is None.
            
        in_scale : bool, optional
            
            Plots scales instead of frequencies The default is True.
        
        yticks : list, optional
        
            y-ticks label. The default is None.
        
        contourf_style : dict, optional
        
            Arguments for the contour plot. The default is {}.
        
        phase_style : dict, optional
        
            Arguments for the phase arrows. The default is {}. It includes:
            - 'pt': the default threshold above which phase arrows will be plotted
            - 'skip_x': the number of points to skip between phase arrows along the x-axis
            - 'skip_y':  the number of points to skip between phase arrows along the y-axis
            - 'scale': number of data units per arrow length unit (see matplotlib.pyplot.quiver)
            - 'width': shaft width in arrow units (see matplotlib.pyplot.quiver)
            - 'color': arrow color (see matplotlib.pyplot.quiver)
        
        cbar_style : dict, optional
        
            Arguments for the color bar. The default is {}.
       
        savefig_settings : dict, optional
        
            The default is {}.
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}
        
        ax : ax, optional
        
            Matplotlib axis on which to return the figure. The default is None.
       
        signif_thresh: float in [0, 1]
        
            Significance threshold. Default is 0.95. If this quantile is not
            found in the qs field of the Coherence object, the closest quantile
            will be picked.
        
        signif_clr : str, optional
        
            Color of the significance line. The default is 'white'.
       
        signif_linestyles : str, optional
        
            Style of the significance line. The default is '-'.
        
        signif_linewidths : float, optional
        
            Width of the significance line. The default is 1.
       
        under_clr : str, optional
        
            Color for under 0. The default is 'ivory'.
        
        over_clr : str, optional
        
            Color for over 1. The default is 'black'.
       
        bad_clr : str, optional
        
            Color for missing values. The default is 'dimgray'.

        Returns
        -------
        fig, ax

        See also
        --------
        
        pyleoclim.core.coherence.Coherence.dashboard : plots a a dashboard showing the coherence and the cross-wavelet transform. 

        pyleoclim.core.series.Series.wavelet_coherence : computes the coherence from two timeseries.

        matplotlib.pyplot.quiver : quiver plot

        Examples
        --------

        Calculate the wavelet coherence of NINO3 and All India Rainfall and plot it:
        .. jupyter-execute::


            import pyleoclim as pyleo
            ts_air = pyleo.utils.load_dataset('AIR')
            ts_nino = pyleo.utils.load_dataset('NINO3')

            coh = ts_air.wavelet_coherence(ts_nino)
            coh.plot()

        Establish significance against an AR(1) benchmark:

        .. jupyter-execute::

            coh_sig = coh.signif_test(number=20, qs=[.9,.95,.99])
            coh_sig.plot()

        Note that specifiying 3 significance thresholds does not take any more time as the quantiles are
        simply estimated from the same ensemble. By default, the plot function looks
        for the closest quantile to 0.95, but this is easy to adjust, e.g. for the 99th percentile:

        .. jupyter-execute::

            coh_sig.plot(signif_thresh = 0.99)

        By default, the function plots the wavelet transform coherency (WTC), which quantifies where
        two timeseries exhibit similar behavior in time-frequency space, regardless of whether this
        corresponds to regions of high common power. To visualize the latter, you want to plot the
        cross-wavelet transform (XWT) instead, like so:

        .. jupyter-execute::

            coh_sig.plot(var='xwt')

        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # handling NaNs
        mask_freq = []
        for i in range(np.size(self.frequency)):
            if all(np.isnan(self.wtc[:, i])):
                mask_freq.append(False)
            else:
                mask_freq.append(True)

        if in_scale:
            y_axis = self.scale[mask_freq]
            if ylabel is None:
                ylabel = f'Scale [{self.scale_unit}]' if self.scale_unit is not None else 'Scale'

            if yticks is None:
                yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])
                mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
                yticks = yticks_default[mask]
        else:
            y_axis = self.frequency[mask_freq]
            if ylabel is None:
                ylabel = f'Frequency [1/{self.scale_unit}]' if self.scale_unit is not None else 'Frequency'

        if signif_thresh > 1 or signif_thresh < 0:
            raise ValueError("The significance threshold must be in [0, 1] ")

        # plot color field for WTC or XWT
        contourf_args = {
            'cmap': 'magma',
            'origin': 'lower',
        }
        contourf_args.update(contourf_style)

        cmap = plt.get_cmap(contourf_args['cmap'])
        cmap.set_under(under_clr)
        cmap.set_over(over_clr)
        cmap.set_bad(bad_clr)
        contourf_args['cmap'] = cmap

        if var == 'wtc':
            lev = np.linspace(0, 1, 11)
            cont = ax.contourf(self.time, y_axis, self.wtc[:, mask_freq].T,
                               levels = lev, **contourf_args)
        elif var == 'xwt':
            cont = ax.contourf(self.time, y_axis, self.xwt[:, mask_freq].T,
                               levels = 11, **contourf_args) # just pass number of contours
        else:
            raise ValueError("Unknown variable; please choose either 'wtc' or 'xwt'")

        # plot significance levels
        if self.signif_qs is not None:
            signif_method_label = {
                'ar1': 'AR(1)',
            }
            if signif_thresh not in self.qs:
                isig = np.abs(np.array(self.qs) - signif_thresh).argmin()
                print("Significance threshold {:3.2f} not found in qs. Picking the closest, which is {:3.2f}".format(signif_thresh,self.qs[isig]))
            else:
                isig = self.qs.index(signif_thresh)

            if var == 'wtc':
                signif_coh = self.signif_qs[0].scalogram_list[isig] # extract WTC significance threshold
                signif_boundary = self.wtc[:, mask_freq].T / signif_coh.amplitude[:, mask_freq].T
            elif var == 'xwt':
                signif_coh = self.signif_qs[1].scalogram_list[isig] # extract XWT significance threshold
                signif_boundary = self.xwt[:, mask_freq].T / signif_coh.amplitude[:, mask_freq].T

            ax.contour(self.time, y_axis, signif_boundary, [-99, 1],
                       colors=signif_clr,
                       linestyles=signif_linestyles,
                       linewidths=signif_linewidths)
            if title is not None:
                ax.set_title("Lines:" + str(round(self.qs[isig]*100))+"% threshold")

        # plot colorbar
        cbar_args = {
            'label': var.upper(),
            'drawedges': False,
            'orientation': 'vertical',
            'fraction': 0.15,
            'pad': 0.05,
            'ticks': cont.levels
        }
        cbar_args.update(cbar_style)

        # assign colorbar to axis (instead of fig) : https://matplotlib.org/stable/gallery/subplots_axes_and_figures/colorbar_placement.html
        cb = plt.colorbar(cont, ax = ax, **cbar_args)

        # plot cone of influence
        ax.set_yscale('log')
        ax.plot(self.time, self.coi, 'k--')

        if ylim is None:
            ylim = [np.min(y_axis), np.min([np.max(y_axis), np.max(self.coi)])]

        ax.fill_between(self.time, self.coi, np.max(self.coi), color='white', alpha=0.5)

        if yticks is not None:
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

        if xlabel is None:
            xlabel = self.time_label

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # plot phase
        skip_x = np.max([int(np.size(self.time)//20), 1])
        skip_y = np.max([int(np.size(y_axis)//20), 1])

        phase_args = {'pt': 0.5, 'skip_x': skip_x, 'skip_y': skip_y,
                      'scale': 30, 'width': 0.004}
        phase_args.update(phase_style)

        pt = phase_args['pt']
        skip_x = phase_args['skip_x']
        skip_y = phase_args['skip_y']
        scale = phase_args['scale']
        width = phase_args['width']

        if 'color' in phase_style:
            color = phase_style['color']
        else:
            color = 'black'

        phase = np.copy(self.phase)[:, mask_freq]

        if self.signif_qs is None:
            if var == 'wtc':
                phase[self.wtc[:, mask_freq] < pt] = np.nan
            else:
                field = self.xwt[:, mask_freq]
                phase[field < pt*field.max()] = np.nan
        else:
            phase[signif_boundary.T < 1] = np.nan

        X, Y = np.meshgrid(self.time, y_axis)
        U, V = np.cos(phase).T, np.sin(phase).T

        ax.quiver(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x],
                  U[::skip_y, ::skip_x], V[::skip_y, ::skip_x],
                  scale=scale, width=width, zorder=99, color=color)

        ax.set_ylim(ylim)

        if xlim is not None:
            ax.set_xlim(xlim)

        lbl1 = self.timeseries1.label
        lbl2 = self.timeseries2.label

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            if title is not None and  title != 'auto':
                fig.suptitle(title)
            elif title == 'auto' and lbl1 is not None and lbl1 is not None:
                title = 'Wavelet coherency ('+self.wave_method.upper() +') between '+ lbl1 + ' and ' + lbl2
                fig.suptitle(title)
            return fig, ax
        else:
            return ax


    def dashboard(self, title=None, figsize=[9,12], overlap = True, phase_style = {}, 
                  line_colors = ['tab:blue','tab:orange'], savefig_settings={},
                  ts_plot_kwargs = None, wavelet_plot_kwargs= None):
         ''' Cross-wavelet dashboard, including the two series, their WTC and XWT.

             Note: this design balances many considerations, and is not easily customizable.

         Parameters
         ----------
         title : str, optional
         
             Title of the plot. The default is None.

         figsize : list, optional
         
             Figure size. The default is [9, 12], as this is an information-rich figure.
             
         overlap : boolean, optional
             whether to restrict the plot to the period of overlap between the series. Defaults to True
             
         phase_style : dict, optional
         
             Arguments for the phase arrows. The default is {}. It includes:
             - 'pt': the default threshold above which phase arrows will be plotted
             - 'skip_x': the number of points to skip between phase arrows along the x-axis
             - 'skip_y':  the number of points to skip between phase arrows along the y-axis
             - 'scale': number of data units per arrow length unit (see matplotlib.pyplot.quiver)
             - 'width': shaft width in arrow units (see matplotlib.pyplot.quiver)
             - 'color': arrow color (see matplotlib.pyplot.quiver)
          
         line_colors : list, optional
         
             Colors for the 2 traces For nomenclature, see https://matplotlib.org/stable/gallery/color/named_colors.html

         savefig_settings : dict, optional
         
             The default is {}.
             the dictionary of arguments for plt.savefig(); some notes below:
             - "path" must be specified; it can be any existed or non-existed path,
               with or without a suffix; if the suffix is not given in "path", it will follow "format"
             - "format" can be one of {"pdf", "eps", "png", "ps"}

         ts_plot_kwargs : dict
         
              arguments to be passed to the timeseries subplot, see pyleoclim.core.series.Series.plot for details

         wavelet_plot_kwargs : dict
         
              arguments to be passed to the contour subplots (XWT and WTC), [see pyleoclim.core.coherence.Coherence.plot for details]


         Returns
         -------
         fig, ax

         See also
         --------

         pyleoclim.core.coherence.Coherence.plot : creates a coherence plot

         pyleoclim.core.series.Series.wavelet_coherence : computes the coherence between two timeseries.

         pyleoclim.core.series.Series.plot: plots a timeseries

         matplotlib.pyplot.quiver: makes a quiver plot

         Examples
         --------

         Calculate the coherence of NINO3 and All India Rainfall and plot it as a dashboard:

         .. jupyter-execute::

             import pyleoclim as pyleo
             ts_air = pyleo.utils.load_dataset('AIR')
             ts_nino = pyleo.utils.load_dataset('NINO3')

             coh = ts_air.wavelet_coherence(ts_nino)
             coh_sig = coh.signif_test(number=10)

             coh_sig.dashboard()

         You may customize colors like so:
             
         .. jupyter-execute::
             
             coh_sig.dashboard(line_colors=['teal','gold'])

         To export the figure, use `savefig_settings`:
             
         .. jupyter-execute::
             
             coh_sig.dashboard(savefig_settings={'path':'./coh_dash.png','dpi':300})

         '''
         # prepare options dictionaries
         savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
         wavelet_plot_kwargs={} if wavelet_plot_kwargs is None else wavelet_plot_kwargs.copy()
         ts_plot_kwargs={} if ts_plot_kwargs is None else ts_plot_kwargs.copy()


         # create figure
         fig = plt.figure(figsize=figsize)
         gs = gridspec.GridSpec(8, 1)
         gs.update(wspace=0, hspace=0.5) # add some breathing room
         ax = {}
         
         # assess period of overlap 
         xlims = np.min(self.time), np.max(self.time) 

         # 1) plot timeseries
         #plt.rc('ytick', labelsize=8) 
         ax['ts1'] = plt.subplot(gs[0:2, 0])
         self.timeseries1.plot(ax=ax['ts1'], color=line_colors[0], **ts_plot_kwargs, legend=False)
         ax['ts1'].yaxis.label.set_color(line_colors[0])
         ax['ts1'].tick_params(axis='y', colors=line_colors[0],labelsize=8)
         ax['ts1'].spines['left'].set_color(line_colors[0])
         ax['ts1'].spines['bottom'].set_visible(False)
         ax['ts1'].grid(False)
         ax['ts1'].set_xlabel('')
         if overlap:
             ax['ts1'].set_xlim(xlims)

         ax['ts2'] = ax['ts1'].twinx()
         self.timeseries2.plot(ax=ax['ts2'], color=line_colors[1],  **ts_plot_kwargs, legend=False)
         ax['ts2'].yaxis.label.set_color(line_colors[1])
         ax['ts2'].tick_params(axis='y', colors=line_colors[1],labelsize=8)
         ax['ts2'].spines['right'].set_color(line_colors[1])
         ax['ts2'].spines['right'].set_visible(True)
         ax['ts2'].spines['left'].set_visible(False)
         ax['ts2'].grid(False)
         if overlap:
             ax['ts2'].set_xlim(xlims)

         # 2) plot WTC
         ax['wtc'] = plt.subplot(gs[2:5, 0], sharex=ax['ts1'])
         if 'cbar_style' not in wavelet_plot_kwargs:
             wavelet_plot_kwargs.update({'cbar_style':{'orientation': 'horizontal',
                                                       'pad': 0.15, 'aspect': 60}})
         self.plot(var='wtc',ax=ax['wtc'], title= None, **wavelet_plot_kwargs)
         #ax['wtc'].xaxis.set_visible(False)  # hide x axis
         ax['wtc'].set_xlabel('')

        # 3) plot XWT
         ax['xwt'] = plt.subplot(gs[5:8, 0], sharex=ax['ts1'])
         if 'phase_style' not in wavelet_plot_kwargs:
             wavelet_plot_kwargs.update({'phase_style':{'color': 'lightgray'}})
         self.plot(var='xwt',ax=ax['xwt'], title= None,
                   contourf_style={'cmap': 'viridis'},
                   cbar_style={'orientation': 'horizontal','pad': 0.2, 'aspect': 60},
                   phase_style=wavelet_plot_kwargs['phase_style'])

         #gs.tight_layout(fig) # this does nothing
         
         if 'fig' in locals():
             if 'path' in savefig_settings:
                 plotting.savefig(fig, settings=savefig_settings)
             return fig, ax
         else:
             return ax

    def signif_test(self, number=200, method='ar1sim', seed=None, qs=[0.95], settings=None, mute_pbar=False):
        '''Significance testing for Coherence objects

        The method obtains quantiles `qs` of the distribution of coherence between
        `number` pairs of Monte Carlo simulations of a process that resembles the original series.
        Currently, only AR(1) surrogates are supported.

        Parameters
        ----------
        number : int, optional
        
            Number of surrogate series to create for significance testing. The default is 200.
        
        method : {'ar1sim','phaseran','CN'}, optional
        
            Method through which to generate the surrogate series. The default is 'phaseran'.
       
        seed : int, optional
        
            Fixes the seed for NumPy's random number generator.
            Useful for reproducibility. The default is None, so fresh, unpredictable
            entropy will be pulled from the operating system.
        
        qs : list, optional
        
            Significance levels to return. The default is [0.95].
            
        settings : dict, optional
        
            Parameters for surrogate model. The default is None.
            
        mute_pbar : bool, optional
        
            Mute the progress bar. The default is False.

        Returns
        -------
        new : pyleoclim.core.coherence.Coherence

            original Coherence object augmented with significance levels signif_qs,
            a list with the following `MultipleScalogram` objects:
            * 0: MultipleScalogram for the wavelet transform coherency (WTC)
            * 1: MultipleScalogram for the cross-wavelet transform (XWT)

            Each object contains as many Scalogram objects as qs contains values

        See also
        --------

        pyleoclim.core.series.Series.wavelet_coherence : Wavelet coherence

        pyleoclim.core.scalograms.Scalogram : Scalogram object

        pyleoclim.core.scalograms.MultipleScalogram : Multiple Scalogram object

        pyleoclim.core.coherence.Coherence.plot : plotting method for Coherence objects

        Examples
        --------

        Calculate the coherence of NINO3 and All India Rainfall and assess significance:

        .. jupyter-execute::

            import pyleoclim as pyleo
            ts_air = pyleo.utils.load_dataset('AIR')
            ts_nino = pyleo.utils.load_dataset('NINO3')

            coh = ts_air.wavelet_coherence(ts_nino)
            coh_sig = coh.signif_test(number=20)
            coh_sig.plot()

        By default, significance is assessed against a 95% benchmark derived from
        an AR(1) process fit to the data, using 200 Monte Carlo simulations.
        To customize, one can increase the number of simulations
        (more reliable, but slower), and the quantile levels.

        .. jupyter-execute::

            coh_sig2 = coh.signif_test(number=100, qs=[.9,.95,.99])
            coh_sig2.plot()

        The plot() function will represent the 95% level as contours by default.
        If you need to show 99%, say, use the `signif_thresh` argument:

        .. jupyter-execute::

            coh_sig2.plot(signif_thresh=0.99)

        Note that if the 99% quantile is not present, the plot method will look
        for the closest match, but lines are always labeled appropriately.
        For reproducibility purposes, it may be good to specify the (pseudo)random number
        generator's seed, like so:

        .. jupyter-execute::

            coh_sig27 = coh.signif_test(number=20, seed=27)

        This will generate exactly the same set of draws from the
        (pseudo)random number at every execution, which may be important for marginal features
        in small ensembles. In general, however, we recommend increasing the
        number of draws to check that features are robust.
        
        One can also specifiy a different method to obtain surrogates, e.g. phase randomization:
            
        .. jupyter-execute::

            coh.signif_test(method='phaseran').plot() 
        '''
        from ..core.surrogateseries import SurrogateSeries
        
        if number == 0:
            return self
        
        new = self.copy()
        
        surr1 = SurrogateSeries(method=method,number=number, seed=seed)
        surr1.from_series(self.timeseries1)
        surr2 = SurrogateSeries(method=method,number=number, seed=seed)
        surr2.from_series(self.timeseries2)
        
        # adjust time axis

        wtcs, xwts = [], []

        for i in tqdm(range(number), desc='Performing wavelet coherence on surrogate pairs', total=number, disable=mute_pbar):
            coh_tmp = surr1.series_list[i].wavelet_coherence(surr2.series_list[i],
                                                             method  = self.wave_method,
                                                             settings = self.wave_args)
            wtcs.append(coh_tmp.wtc)
            xwts.append(coh_tmp.xwt)

        wtcs = np.array(wtcs)
        xwts = np.array(xwts)


        ne, nf, nt = np.shape(wtcs)

        # reshape because mquantiles only accepts inputs of at most 2D
        wtcs_r = np.reshape(wtcs, (ne, nf*nt))
        xwts_r = np.reshape(xwts, (ne, nf*nt))

        # define nd-arrays
        nq = len(qs)
        wtc_qs = np.ndarray(shape=(nq, nf, nt))
        xwt_qs = np.empty_like(wtc_qs)

        # extract quantiles and reshape
        wtc_qs = mquantiles(wtcs_r, qs, axis=0)
        wtc_qs = np.reshape(wtc_qs, (nq, nf, nt))
        xwt_qs = mquantiles(xwts_r, qs, axis=0)
        xwt_qs = np.reshape(xwt_qs, (nq, nf, nt))

        # put in Scalogram objects for export
        wtc_list, xwt_list = [],[]

        for i in range(nq):
            wtc_tmp = Scalogram(
                    frequency=self.frequency, time=self.time, amplitude=wtc_qs[i,:,:],
                    coi=self.coi, scale = self.scale,
                    freq_method=self.freq_method, freq_kwargs=self.freq_kwargs, label=f'{qs[i]*100:g}%',
                )
            wtc_list.append(wtc_tmp)
            xwt_tmp = Scalogram(
                    frequency=self.frequency, time=self.time, amplitude=xwt_qs[i,:,:],
                    coi=self.coi, scale = self.scale,
                    freq_method=self.freq_method, freq_kwargs=self.freq_kwargs, label=f'{qs[i]*100:g}%',
                )

            xwt_list.append(xwt_tmp)

        new.signif_qs = []
        new.signif_qs.append(MultipleScalogram(scalogram_list=wtc_list)) # Export WTC quantiles
        new.signif_qs.append(MultipleScalogram(scalogram_list=xwt_list)) # Export XWT quantiles
        new.signif_method = method
        new.qs = qs

        return new
    
    def phase_stats(self, scales, number=1000, level=0.05):
        ''' Estimate phase angle statistics of a Coherence object

        As per [1], the strength (consistency) of a phase relationship is assessed using:
            
        * sigma, the circular standard deviation   
        
        * kappa,  an estimate of the Von Mises distribution's concentration parameter. 
            It is a reciprocal measure of dispersion, so 1/kappa is analogous to the variance) [3].
        
        Because of inherent persistence of geophysical signals and of the 
        reproducing kernel of the continuous wavelet transform [3], phase statistics are 
        assessed relative to an AR(1) model fit to the angle deviations observed at the requested scale(s). 
    
        Specifically, if `number` is specified, the method simulates `number` 
        Monte Carlo realizations of an AR(1) process fit to fluctuations around 
        the mean angle. This ensemble is used to obtain the confidence limits: 
        `sigma_lo` (`level` quantile) and `kappa_hi` (1-`level` quantile). 
        These correspond to 1-tailed tests of the strength of the relationship.

        Parameters
        ----------
        scales : float
        
            scale at which to evaluate the phase angle
        
        number : int, optional
        
            number of AR(1) series to create for significance testing. The default is 1000.
       
        level : float, optional
        
           significance level against which to gauge sigma and kappa. default: 0.05
        

        Returns
        -------
        result : dict
        
            contains angle_mean (the mean angle for those scales), sigma (the 
            circular standard deviation), kappa, sigma_lo (alpha-level quantile 
            for sigma) and kappa_hi, the (1-alpha)-level quantile for kappa. 
                
        See also
        --------

        pyleoclim.core.series.Series.wavelet_coherence : Wavelet coherence

        pyleoclim.core.scalograms.Scalogram : Scalogram object

        pyleoclim.core.scalograms.MultipleScalogram : Multiple Scalogram object

        pyleoclim.core.coherence.Coherence.plot : plotting method for Coherence objects
        
        pyleoclime.utils.wavelet.angle_sig : significance of phase angle statistics
            
        pyleoclim.utils.wavelet.angle_stats: phase angle statistics
    
        
        References
        ----------
        
        [1] Grinsted, A., J. C. Moore, and S. Jevrejeva (2004), Application of the cross
        wavelet transform and wavelet coherence to geophysical time series, 
        Nonlinear Processes in Geophysics, 11, 561–566.
        
        [2] Huber, R., Dutra, L. V., & da Costa Freitas, C. (2001).
        SAR interferogram phase filtering based on the Von Mises distribution. 
        In IGARSS 2001. Scanning the Present and Resolving the Future. 
        Proceedings. IEEE 2001 International Geoscience and Remote Sensing Symposium 
        (Cat. No. 01CH37217) (Vol. 6, pp. 2816-2818). IEEE.  
        
        [3] Farge, M. and Schneider, K. (2006): Wavelets: application to turbulence
        Encyclopedia of Mathematical Physics (Eds. J.-P. Françoise, G. Naber and T.S. Tsun)
        pp 408-420. 

        Examples
        --------

        Calculate the phase angle between NINO3 and All India Rainfall at 5y scales:

        .. jupyter-execute::

            import pyleoclim as pyleo
            ts_air = pyleo.utils.load_dataset('AIR')
            ts_nino = pyleo.utils.load_dataset('NINO3')
            coh = ts_air.wavelet_coherence(ts_nino)
            coh.phase_stats(scales=5)
        
        One may also obtain phase angle statistics over an interval, like the 2-8y ENSO band: 
            
        .. jupyter-execute::
             
             phase = coh.phase_stats(scales=[2,8])
             print("The mean angle is {:4.2f}°".format(phase.mean_angle/np.pi*180))
             print(phase)
            
        From this example, one diagnoses a strong anti-phased relationship in the ENSO band,
        with high von Mises concentration (kappa ~ 3.35 >> kappa_hi) and low circular
        dispersion (sigma ~ 0.6 << sigma_lo). This would be strong evidence of a consistent
        anti-phasing between NINO3 and AIR at those scales.

        '''
        scales = np.array(scales)
        
        if scales.max() > self.scale.max():
            warnings.warn("Requested scale exceeds largest scale in object. Truncating to "+str(self.scale.max()))
        
        if scales.size == 1:
            scale_idx = np.argmin(np.abs(self.scale - scales))
            res = waveutils.angle_sig(self.phase[:,scale_idx],nMC=number,level=level)
        elif  scales.size == 2:
            idx_lo = np.argmin(np.abs(self.scale - scales.min()))
            idx_hi = np.argmin(np.abs(self.scale - scales.max()))
            if (idx_hi >= idx_lo):
                raise ValueError("Insufficiently spaced scales. Please pick a single one, or a wider interval")
            else: # average phase over those scales
                nt, ns = self.phase.shape
                phase = np.empty((nt)) 
                for i in range(nt):
                    phase[i], _, _ = waveutils.angle_stats(self.phase[i,idx_hi:idx_lo]) 
                res = waveutils.angle_sig(phase,nMC=number,level=level) # assess significance 
            
        return res