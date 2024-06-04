#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The SpectralCoherence class stores the result of Series.spectral_coherence(), whether WWZ or CWT.
"""
from ..utils import plotting
import matplotlib.pyplot as plt

class SpectralCoherence:
    '''Class to store the results of cross spectral analysis
    
    Attributes
    ----------
    
    coherence: numpy array
        coherence values
        
    scale: numpy array
        scale values
        
    frequency: numpy array
        frequency values
        
    coi: numpy array
        cone of influence values
        
    See Also
    --------
    
    pyleoclim.core.series.Series.spectral_coherence : method to compute the spectral coherence'''

    def __init__(self, coherence, scale, frequency,coi,method,timeseries1,timeseries2,label='Coherence'):
        self.coherence = coherence
        self.scale = scale
        self.frequency = frequency
        self.coi = coi
        self.method = method
        self.timeseries1 = timeseries1
        self.timeseries2 = timeseries2
        self.label = label

    def plot(self,color=None,label=None,ax=None,plot_kwargs=None,savefig_settings=None):
        '''Plot the coherence as a function of scale or frequency, alongside the spectrum of the two timeseries (using the same method used for the coherence).
        
        Parameters
        ----------
        
        color: str
            color of the plot
            
        label: str
            label of the plot
            
        ax: matplotlib axis
            axis to plot on
            
        plot_kwargs: dict
            additional arguments to pass to the pyleoclim.utils.plotting.plot_xy

        savefig_settings: dict
            settings to pass to the pyleoclim.utils.plotting.savefig function
            
        Returns
        -------
        
        ax: matplotlib axis
            axis with the plot'''

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        if ax is None:
            fig,ax = plt.subplots()
        else:
            pass

        if color is not None:
            plot_kwargs.update({'color':color})
        if label is None:
            label = self.label
        plot_kwargs.update({'label': label})

        spec1 = self.timeseries1.spectral(method=self.method)
        spec2 = self.timeseries2.spectral(method=self.method)

        spec1.plot(ax=ax)
        spec2.plot(ax=ax)

        ax2 = ax.twinx()
        mask = self.scale < max(self.coi)
        
        plotting.plot_xy(self.scale[mask],self.coherence[mask],ax=ax2,plot_kwargs=plot_kwargs)
        ax2.fill_between(self.scale[mask], 0, self.coherence[mask], color=color, alpha=0.3)

        #formatting
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax2.grid(False)
        ax2.set_ylabel('Coherence')

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
                return fig, ax
        else:
            return ax
