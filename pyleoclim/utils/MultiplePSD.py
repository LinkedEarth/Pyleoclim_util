"""
This module contains the MultiplePSD object.
"""

from ..utils import plotting 
from ..core import PSD

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import matplotlib as mpl
from scipy.stats.mstats import mquantiles


class MultiplePSD:
    
    def __init__(self, psd_list, beta_est_res=None):
        ''' Object for multiple PSD.

        This object stores several PSDs from different Series or ensemble members in an age model.         
       
        Parameters
        ----------
        
        beta_est_res : numpy.array
        
            Results of the beta estimation calculation
        
        See also
        --------
        
        pyleoclim.core.PSD.PSD.beta_est : Calculates the scaling exponent (i.e., the slope in a log-log plot) of the spectrum (beta)

        '''
        self.psd_list = psd_list
        if beta_est_res is None:
            self.beta_est_res = beta_est_res
        else:
            self.beta_est_res = np.array(beta_est_res)

    def copy(self):
        '''Copy object
        '''
        return deepcopy(self)

    def quantiles(self, qs=[0.05, 0.5, 0.95], lw=[0.5, 1.5, 0.5]):
        
        '''Calculate the quantiles of the significance testing

        Parameters
        ----------
        
        qs : list, optional
        
            List of quantiles to consider for the calculation. The default is [0.05, 0.5, 0.95].
            
        lw : list, optional
        
            Linewidth to use for plotting each level. Should be the same length as qs. The default is [0.5, 1.5, 0.5].

        Raises
        ------
        
        ValueError
        
            Frequency axis not consistent across the PSD list!

        Returns
        -------
        
        psds : pyleoclim.MultiplePSD

        '''
        if self.psd_list[0].timeseries is not None:
            period_unit = self.psd_list[0].timeseries.time_unit

        freq = np.copy(self.psd_list[0].frequency)
        amps = []
        for psd in self.psd_list:
            if not np.array_equal(psd.frequency, freq):
                raise ValueError('Frequency axis not consistent across the PSD list!')

            amps.append(psd.amplitude)

        amps = np.array(amps)
        amp_qs = mquantiles(amps, qs, axis=0)

        psd_list = []
        for i, amp in enumerate(amp_qs):
            psd_tmp = PSD.PSD(frequency=freq, amplitude=amp, label=f'{qs[i]*100:g}%', plot_kwargs={'color': 'gray', 'linewidth': lw[i]}, period_unit=period_unit)
            psd_list.append(psd_tmp)

        psds = MultiplePSD(psd_list=psd_list)
        return psds

    def beta_est(self, fmin=None, fmax=None, logf_binning_step='max', verbose=False):
        
        ''' Estimate the scaling factor beta of the each PSD. 
        
        This function calculates the scaling factor for each of the PSD stored in the object. The scaling factor represents the slope of the spectrum line if plot in log-log space. 

        Parameters
        ----------

        fmin : float
        
            the minimum frequency edge for beta estimation; the default is the minimum of the frequency vector of the PSD object

        fmax : float
        
            the maximum frequency edge for beta estimation; the default is the maximum of the frequency vector of the PSD object

        logf_binning_step : str; {'max', 'first'}
        
            if 'max', then the maximum spacing of log(f) will be used as the binning step.
            if 'first', then the 1st spacing of log(f) will be used as the binning step.

        verbose : bool
        
            If True, will print warning messages if there is any

        Returns
        -------

        new : pyleoclim.MultiplePSD
        
            New MultiplePSD object with the estimated scaling slope information, which is stored as a dictionary that includes:
            - beta: the scaling factor
            - std_err: the one standard deviation error of the scaling factor
            - f_binned: the binned frequency series, used as X for linear regression
            - psd_binned: the binned PSD series, used as Y for linear regression
            - Y_reg: the predicted Y from linear regression, used with f_binned for the slope curve plotting

        See also
        --------

        pyleoclim.core.PSD.PSD.beta_est : beta estimation for on a single PSD object

        '''

        res_dict = {}
        res_dict['beta'] = []
        res_dict['std_err'] = []
        res_dict['f_binned'] = []
        res_dict['psd_binned'] = []
        res_dict['Y_reg'] = []
        psd_beta_list = []
        for psd_obj in self.psd_list:
            psd_beta = psd_obj.beta_est(fmin=fmin, fmax=fmax, logf_binning_step=logf_binning_step, verbose=verbose)
            psd_beta_list.append(psd_beta)
            res = psd_beta.beta_est_res
            for k in res_dict.keys():
                res_dict[k].append(res[k])

        new = self.copy()
        new.beta_est_res = res_dict
        new.psd_list = psd_beta_list
        return new


    def plot(self, figsize=[10, 4], in_loglog=True, in_period=True, xlabel=None, ylabel='Amplitude', title=None,
             xlim=None, ylim=None, savefig_settings=None, ax=None, xticks=None, yticks=None, legend=True,
             colors=None, cmap=None, norm=None, plot_kwargs=None, lgd_kwargs=None):
        '''Plot multiple PSD on the same plot

        Parameters
        ----------
        
        figsize : list, optional
        
            Figure size. The default is [10, 4].
            
        in_loglog : bool, optional
        
            Whether to plot in loglog. The default is True.
            
        in_period : bool, {True, False} optional
        
            Plots against periods instead of frequencies. The default is True.
            
        xlabel : str, optional
        
            x-axis label. The default is None.
            
        ylabel : str, optional
        
            y-axis label. The default is 'Amplitude'.
            
        title : str, optional
        
            Title for the figure. The default is None.
            
        xlim : list, optional
        
            Limits for the x-axis. The default is None.
            
        ylim : list, optional
        
            limits for the y-axis. The default is None.
            
        colors : a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)
        
            Colors for plotting.
            If None, the plotting will cycle the 'tab10' colormap;
            if only one color is specified, then all curves will be plotted with that single color;
            if a list of colors are specified, then the plotting will cycle that color list.
            
        cmap : str
        
            The colormap to use when "colors" is None.
            
        norm : matplotlib.colors.Normalize like
        
            The nomorlization for the colormap.
            If None, a linear normalization will be used.
            
        savefig_settings : dict, optional
        
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}
            
        ax : matplotlib axis, optional
        
            The matplotlib axis object on which to retrun the figure. The default is None.
            
        xticks : list, optional
        
            x-ticks label. The default is None.
            
        yticks : list, optional
        
            y-ticks label. The default is None.
            
        legend : bool, optional
        
            Whether to plot the legend. The default is True.
            
        plot_kwargs : dictionary, optional
        
            Parameters for plot function. The default is None.
            
        lgd_kwargs : dictionary, optional
        
            Parameters for legend. The default is None.

        Returns
        -------
        fig : matplotlib.pyplot.figure
        
        ax : matplotlib.pyplot.axis

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for idx, psd in enumerate(self.psd_list):

            tmp_plot_kwargs = {}
            if psd.plot_kwargs is not None:
                tmp_plot_kwargs.update(psd.plot_kwargs)

            tmp_plot_kwargs.update(plot_kwargs)

            # get color for each psd curve
            use_clr = False

            if 'color' not in tmp_plot_kwargs and 'c' not in 'tmp_plot_kwargs':
                use_clr = True

            if 'color' in tmp_plot_kwargs and tmp_plot_kwargs['color'] is None:
                use_clr = True

            if 'c' in tmp_plot_kwargs and tmp_plot_kwargs['c'] is None:
                use_clr = True

            if colors is not None or cmap is not None:
                use_clr = True

            if use_clr:
                # use the color based on the argument 'colors' or 'cmap'
                if colors is None:
                    cmap = 'tab10' if cmap is None else cmap
                    cmap_obj = plt.get_cmap(cmap)
                    if hasattr(cmap_obj, 'colors'):
                        nc = len(cmap_obj.colors)
                    else:
                        nc = len(self.psd_list)

                    if norm is None:
                        norm = mpl.colors.Normalize(vmin=0, vmax=nc-1)

                    clr = cmap_obj(norm(idx%nc))
                elif type(colors) is str:
                    clr = colors
                elif type(colors) is list:
                    nc = len(colors)
                    clr = colors[idx%nc]
                else:
                    raise TypeError('"colors" should be a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)')

                tmp_plot_kwargs.update({'color': clr})

            ax = psd.plot(
                figsize=figsize, in_loglog=in_loglog, in_period=in_period, xlabel=xlabel, ylabel=ylabel,
                title=title, xlim=xlim, ylim=ylim, savefig_settings=savefig_settings, ax=ax,
                xticks=xticks, yticks=yticks, legend=legend, plot_kwargs=tmp_plot_kwargs, lgd_kwargs=lgd_kwargs,
            )

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax

    def plot_envelope(self, figsize=[10, 4], qs=[0.025, 0.5, 0.975],
             in_loglog=True, in_period=True, xlabel=None, ylabel='Amplitude', title=None,
             xlim=None, ylim=None, savefig_settings=None, ax=None, xticks=None, yticks=None, plot_legend=True,
             curve_clr=sns.xkcd_rgb['pale red'], curve_lw=3, shade_clr=sns.xkcd_rgb['pale red'], shade_alpha=0.3, shade_label=None,
             lgd_kwargs=None, members_plot_num=10, members_alpha=0.3, members_lw=1, seed=None):

        '''Plot an envelope statistics for mulitple PSD
        
        This function plots an envelope statistics from multiple PSD. This is especially useful when the PSD are coming from an ensemble of possible solutions (e.g., age ensembles)

        Parameters
        ----------
        
        figsize : list, optional
        
            The figure size. The default is [10, 4].
            
        qs : list, optional
        
            The significance levels to consider. The default is [0.025, 0.5, 0.975].
            
        in_loglog : bool, optional
        
            Plot in log space. The default is True.
            
        in_period : bool, optional
        
            Whether to plot periodicity instead of frequency. The default is True.
            
        xlabel : str, optional
        
            x-axis label. The default is None.
            
        ylabel : str, optional
        
            y-axis label. The default is 'Amplitude'.
            
        title : str, optional
        
            Plot title. The default is None.
            
        xlim : list, optional
        
            x-axis limits. The default is None.
            
        ylim : list, optional
        
            y-axis limits. The default is None.
            
        savefig_settings : dict, optional
        
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.
            
        ax : matplotlib.ax, optional
        
            Matplotlib axis on which to return the plot. The default is None.
            
        xticks : list, optional
        
            xticks label. The default is None.
            
        yticks : list, optional
        
            yticks label. The default is None.
            
        plot_legend : bool, optional
        
            Wether to plot the legend. The default is True.
            
        curve_clr : str, optional
        
            Color of the main PSD. The default is sns.xkcd_rgb['pale red'].
            
        curve_lw : str, optional
        
            Width of the main PSD line. The default is 3.
            
        shade_clr : str, optional
        
            Color of the shaded envelope. The default is sns.xkcd_rgb['pale red'].
            
        shade_alpha : float, optional
        
            Transparency on the envelope. The default is 0.3.
            
        shade_label : str, optional
        
            Label for the envelope. The default is None.
            
        lgd_kwargs : dict, optional
        
            Parameters for the legend. The default is None.
            
        members_plot_num : int, optional
        
            Number of individual members to plot. The default is 10.
            
        members_alpha : float, optional
        
            Transparency of the lines representing the multiple members. The default is 0.3.
            
        members_lw : float, optional
        
            With of the lines representing the multiple members. The default is 1.
            
        seed : int, optional
        
            Set the seed for random number generator. Useful for reproducibility. The default is None.

        Returns
        -------
        fig : matplotlib.pyplot.figure
        
        ax : matplotlib.pyplot.axis

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if members_plot_num > 0:
            if seed is not None:
                np.random.seed(seed)

            npsd = np.size(self.psd_list)
            random_draw_idx = np.random.choice(npsd, members_plot_num)

            for idx in random_draw_idx:
                self.psd_list[idx].plot(
                    in_loglog=in_loglog, in_period=in_period, xlabel=xlabel, ylabel=ylabel,
                    xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks, ax=ax, color='gray', alpha=members_alpha,
                    zorder=99, linewidth=members_lw,
                )
            ax.plot(np.nan, np.nan, color='gray', label=f'example members (n={members_plot_num})')

        psd_qs = self.quantiles(qs=qs)
        psd_qs.psd_list[1].plot(
            in_loglog=in_loglog, in_period=in_period, xlabel=xlabel, ylabel=ylabel, linewidth=curve_lw,
            xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks, ax=ax, color=curve_clr, zorder=100
        )


        if in_period:
            x_axis = 1/psd_qs.psd_list[0].frequency
        else:
            x_axis = psd_qs.psd_list[0].frequency

        if shade_label is None:
            shade_label = f'{psd_qs.psd_list[0].label}-{psd_qs.psd_list[-1].label}'

        ax.fill_between(
            x_axis, psd_qs.psd_list[0].amplitude, psd_qs.psd_list[-1].amplitude,
            color=shade_clr, alpha=shade_alpha, edgecolor=shade_clr, label=shade_label,
        )

        if title is not None:
            ax.set_title(title)

        if plot_legend:
            lgd_args = {'frameon': False}
            lgd_args.update(lgd_kwargs)
            ax.legend(**lgd_args)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax