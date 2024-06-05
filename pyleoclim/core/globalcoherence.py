#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The GlobalCoherence class stores the result of Series.global_coherence(), whether WWZ or CWT.
"""
from ..utils import plotting

import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from scipy.stats.mstats import mquantiles

class GlobalCoherence:
    '''Class to store the results of cross spectral analysis
    
    Attributes
    ----------
    
    global_coh: numpy array
        coherence values
        
    scale: numpy array
        scale values
        
    frequency: numpy array
        frequency values
        
    coi: numpy array
        cone of influence values

    coh: Coherence
        Original coherence object
    
        
    See Also
    --------
    
    pyleoclim.core.series.Series.global_coherence : method to compute the spectral coherence'''

    def __init__(self, global_coh, coh, signif_qs=None,signif_method=None,qs=None, label='Coherence'):
        self.global_coh = global_coh
        self.label = label
        self.coh = coh
        self.signif_qs = signif_qs
        self.signif_method = signif_method
        self.qs = qs

    def copy(self):
        '''Copy object
        '''
        return deepcopy(self)

    def signif_test(self,method='ar1sim',number=200,qs=[.95]):
        '''Perform a significance test on the coherence values

        Parameters
        ----------

        method: str; {'ar1sim','CN','phaseran'}
            method to use for the surrogate test. Default is 'ar1sim'.

        number: int
            number of surrogates to generate. Default is 200

        qs: list
            list of quantiles to compute. Default is [.95]
        
        Returns
        -------
        
        global_coh: pyleoclim.core.globalcoherence.GlobalCoherence
            Global coherence with significance field filled in'''
        
        from ..core.surrogateseries import SurrogateSeries

        new = self.copy()
        
        ts1 = self.coh.timeseries1
        ts2 = self.coh.timeseries2

        surr1 = SurrogateSeries(method=method,number=number)
        surr2 = SurrogateSeries(method=method,number=number)

        surr1.from_series(ts1)
        surr2.from_series(ts2)

        coh_array = np.empty((number,len(self.global_coh)))

        wavelet_kwargs = {
            'freq_method':self.coh.freq_method,
            'freq_kwargs':self.coh.freq_kwargs,
            'settings':self.coh.wave_args,
            'method':self.coh.wave_method,
        }

        for i in range(number):
            surr_series1 = surr1.series_list[i]
            surr_series2 = surr2.series_list[i]
            surr_coh = surr_series1.global_coherence(surr_series2,wavelet_kwargs=wavelet_kwargs)
            coh_array[i,:] = surr_coh.global_coh
        
        quantiles = mquantiles(coh_array,qs,axis=0)
        new.signif_qs = quantiles.data
        new.signif_method = method
        new.qs = qs

        return new

    def plot(self,figsize=(8,8),xlim=None,label=None,coh_y_label=None,coh_line_color=None,ax=None,coh_ylim=(.4,1),fill_alpha=.3,fill_color=None,coh_plot_kwargs=None,
             savefig_settings=None,spectral_kwargs=None,legend=True,legend_kwargs=None,spec1_plot_kwargs=None,spec2_plot_kwargs=None):
        '''Plot the coherence as a function of scale or frequency, alongside the spectrum of the two timeseries (using the same method used for the coherence).
        
        Parameters
        ----------

        figsize: tuple
            size of the figure. Default is (8,8). Only used if ax is None

        xlim: tuple
            x limits for the plot. Default is None

        label: str
            label of the plot
        
        coh_line_color: str
            color of the coherence line
        
        coh_ylim: tuple
            y limits for the coherence plot. Default is (.4,1)
        
        fill_alpha: float
            alpha value for the fill_between plot. Default is .3

        fill_color : str
            color of the fill_between plot
            
        coh_plot_kwargs: dict
            additional arguments to pass to the pyleoclim.utils.plotting.plot_xy

        savefig_settings: dict
            settings to pass to the pyleoclim.utils.plotting.savefig function
        
        spectral_kwargs: dict
            additional arguments to pass to the pyleo.Series.spectral method

        spec1_plot_kwargs: dict
            additional arguments to pass to the pyleo.Series.spectral method
        
        spec2_plot_kwargs: dict
            additional arguments to pass to the pyleo.Series.spectral method

        legend: bool
            whether to include a legend or not

        ax: matplotlib axis
            axis to plot on
            
        Returns
        -------
        
        ax: matplotlib axis
            axis with the plot'''

        coh_plot_kwargs = {} if coh_plot_kwargs is None else coh_plot_kwargs.copy()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        spectral_kwargs = {} if spectral_kwargs is None else spectral_kwargs.copy()
        legend_kwargs = {} if legend_kwargs is None else legend_kwargs.copy()
        spec1_plot_kwargs = {} if spec1_plot_kwargs is None else spec1_plot_kwargs.copy()
        spec2_plot_kwargs = {} if spec2_plot_kwargs is None else spec2_plot_kwargs.copy()

        if ax is None:
            fig,ax = plt.subplots(figsize=figsize)
        else:
            pass

        coh_dict = self.coh.__dict__

        if 'method' not in spectral_kwargs:
            spectral_kwargs.update({'method': coh_dict['wave_method']})
        if 'freq' not in spectral_kwargs:
            spectral_kwargs.update({'freq': coh_dict['freq_method']})
        if 'freq_kwargs' not in spectral_kwargs:
            spectral_kwargs.update({'freq_kwargs': coh_dict['freq_kwargs']})
        if spectral_kwargs['method'] == coh_dict['wave_method']:
            for key,value in coh_dict['wave_args'].items():
                if key not in spectral_kwargs:
                    spectral_kwargs.update({key: value})

        ts1 = coh_dict['timeseries1']
        ts2 = coh_dict['timeseries2']

        spec1 = ts1.spectral(label=ts1.label, **spectral_kwargs)
        spec2 = ts2.spectral(label=ts2.label, **spectral_kwargs)

        spec1.plot(ax=ax,**spec1_plot_kwargs)
        spec2.plot(ax=ax,**spec2_plot_kwargs)

        if xlim is not None:
            ax.set_xlim(xlim)

        ax2 = ax.twinx()

        if coh_line_color is not None:
            coh_plot_kwargs.update({'color':coh_line_color})
        if coh_y_label is not None:
            ax2.set_ylabel(coh_y_label)
        if coh_ylim is not None:
            ax2.set_ylim(coh_ylim)
        if label is None:
            label = self.label
        coh_plot_kwargs.update({'label': label})

        scale = coh_dict['scale']
        
        ax2.plot(scale,self.global_coh,**coh_plot_kwargs)
        ax2.fill_between(scale, 0, self.global_coh, color=fill_color, alpha=fill_alpha)
        ax2.grid(False)

        # plot significance levels if present
        if self.signif_qs is not None:
            signif_method_label = {
                'ar1sim': 'AR(1) simulations (MoM)',
                'phaseran': 'Phase Randomization',
                'CN': 'Colored Noise'
            }

            for i, q in enumerate(self.signif_qs):
                ax.plot(
                    scale, q,
                    label=f'{signif_method_label[self.signif_method]}, {self.qs[i]} threshold',
                    color='red',
                    linestyle='dashed',
                    linewidth=.8,
                )

        #formatting
        if legend:
            if len(legend_kwargs) == 0:
                ax.legend().set_visible(False)
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2)
            else:
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                if 'handles' not in legend_kwargs:
                    legend_kwargs.update({'handles': lines+lines2})
                if 'labels' not in legend_kwargs:
                    legend_kwargs.update({'labels': labels+labels2})
                ax.legend(**legend_kwargs)
                ax2.legend().set_visible(False)
        else:
            ax.legend().set_visible(False)
            ax2.legend().set_visible(False)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax
