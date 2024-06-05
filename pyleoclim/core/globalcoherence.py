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

    def plot(self,color=None,label=None,ax=None,coherence_ylim=(.4,1),fill_alpha=.3,fill_color=None,plot_kwargs=None,savefig_settings=None,spectral_kwargs=None,legend=True,
             spec1_plot_kwargs=None,spec2_plot_kwargs=None):
        '''Plot the coherence as a function of scale or frequency, alongside the spectrum of the two timeseries (using the same method used for the coherence).
        
        Parameters
        ----------
        
        color: str
            color of the plot
            
        label: str
            label of the plot
            
        ax: matplotlib axis
            axis to plot on
        
        coherence_ylim: tuple
            y limits for the coherence plot. Default is (.4,1)
        
        fill_alpha: float
            alpha value for the fill_between plot. Default is .3

        fill_color : str
            color of the fill_between plot
            
        plot_kwargs: dict
            additional arguments to pass to the pyleoclim.utils.plotting.plot_xy

        savefig_settings: dict
            settings to pass to the pyleoclim.utils.plotting.savefig function

        spectral_kwargs: dict
            additional arguments to pass to the pyleo.Series.spectral method

        legend: bool
            whether to include a legend or not
            
        Returns
        -------
        
        ax: matplotlib axis
            axis with the plot'''

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        spectral_kwargs = {} if spectral_kwargs is None else spectral_kwargs.copy()
        spec1_plot_kwargs = {} if spec1_plot_kwargs is None else spec1_plot_kwargs.copy()
        spec2_plot_kwargs = {} if spec2_plot_kwargs is None else spec2_plot_kwargs.copy()

        if ax is None:
            fig,ax = plt.subplots()
        else:
            pass

        if color is not None:
            plot_kwargs.update({'color':color})
        if label is None:
            label = self.label
        plot_kwargs.update({'label': label})

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

        ax2 = ax.twinx()

        scale = coh_dict['scale']
        coi = coh_dict['coi']
        mask = scale < np.max(coi)
        
        plotting.plot_xy(scale[mask],self.global_coh[mask],ax=ax2,plot_kwargs=plot_kwargs)
        ax2.fill_between(scale[mask], 0, self.global_coh[mask], color=fill_color, alpha=fill_alpha)
        ax2.axvline(np.max(coi))
        ax2.set_ylabel('Coherence')
        ax2.set_ylim(*coherence_ylim)

                # plot significance levels
        if self.signif_qs is not None:
            signif_method_label = {
                'ar1sim': 'AR(1) simulations (MoM)',
                'uar1': 'AR(1) simulations (MLE)',
                'ar1asym': 'AR(1) asymptotic solution',
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
            ax.legend().set_visible(False)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc=0)
            ax2.grid(False)
        else:
            ax.legend().set_visible(False)
            ax2.legend().set_visible(False)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
                return fig, ax
        else:
            return ax
