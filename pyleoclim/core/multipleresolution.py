#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultipleResolution objects are designed to contain, display, and analyze information on the resolution of multiple time axes from a MultipleSeries, EnsembleSeries, or SurrogateSeries object.
"""

from ..utils import tsutils, plotting, tsmodel, tsbase

import warnings

import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from matplotlib import gridspec

class MultipleResolution:
    ''' Resolution object
    
    Resolution objects store time axis resolution information derived from MultipleSeries objects.
    They are generated via the resolution method applied to a MultipleSeries object and contain methods relevant to the analysis of resolution information.

    See Also
    --------

    '''

    def __init__(self,resolution_list,time_unit):
        self.resolution_list = resolution_list
        self.time_unit = time_unit

    def plot(self,figsize=(10,8),xlabel=None,ylabel=None,legend=False,ax=None,
             boxplot_whis=[0,100], boxplot_width=.6, boxplot_dodge=False,boxplot_palette='viridis',
             stripplot_size=2,stripplot_color=".3",stripplot_alpha=.8,stripplot_dodge=False,
             boxplot_kwargs=None,stripplot_kwargs=None,savefig_settings=None):
        """Boxplot showing distribution of resolutions from each resolution object

        Parameters
        ----------

        figsize : tuple, list
            Size of the figure.

        xlabel : str
            Label for the x axis. "Resolution [{time_unit}]" will be used by default.

        ylabel : str
            Label for the y axis. Left empty by default.

        legend : bool
            Whether or not to plot the legend. Default is False.
        
        ax : matplotlib.ax
            The matplotlib axis onto which to return the figure. The default is None.

        boxplot_whis : float or pair of floats
            If scalar, whiskers are drawn to the farthest datapoint within whis * IQR from the nearest hinge. 
            If a tuple, it is interpreted as percentiles that whiskers represent. Default is [0,100]
        
        boxplot_width : float
            Width allotted to each element on the orient axis.
        
        boxplot_dodge : bool
            Whether boxplot elements should be narrowed and shifted along the orient axis to eliminate overlap. Default is False.

        boxplot_palette : palette name, list, or dict
            Colors to use for the different levels of the hue variable. 
            Should be something that can be interpreted by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
            Gets passed to seaborn.boxplot for details.

        stripplot_size : float
            Size of stripplot markers.

        stripplot_color : str
            Color for the stripplot markers.

        stripplot_alpha : float
            Alpha for the stripplot markers.

        stripplot_dodge : bool
            Whether stripplot elements should be narrowed and shifted along the orient axis to eliminate overlap. Default is False.

        boxplot_kwargs : dict
            Dictionary of arguments for seaborn.boxplot. Arguments that are passed here will overwrite explicit arguments (e.g. whis, width, etc.).
        
        stripplot_kwargs : dict
            Dictionary of argument for seaborn.stripplot. Arguments that are passed here will overwrite explicit arguments (e.g. size, color, etc.).

        savefig_settings : dictionary, optional
        
            the dictionary of arguments for pyleo.utils.plotting.savefig(); some notes below:
            - "path" must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.
        
        Examples
        --------
        .. jupyter-execute::
            import pyleoclim as pyleo

            co2ts = pyleo.utils.load_dataset('AACO2')
            edc = pyleo.utils.load_dataset('EDC-dD')
            ms = edc & co2ts # create MS object
            ms_resolution = ms.resolution()
            ms_resolution.plot()
            """
        
        boxplot_kwargs = {} if boxplot_kwargs is None else boxplot_kwargs.copy()
        stripplot_kwargs = {} if stripplot_kwargs is None else stripplot_kwargs.copy()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        if ax is None:
            fig,ax = plt.subplots(figsize=figsize)

        data = pd.DataFrame(columns=['Resolution','Label'])

        for resolution in self.resolution_list:
            for value in resolution.resolution:
                data.loc[len(data),['Resolution','Label']] = [value,resolution.label]

        if 'whis' in boxplot_kwargs:
            boxplot_whis = boxplot_kwargs.pop('whis')
        if 'width' in boxplot_kwargs:
            boxplot_width = boxplot_kwargs.pop('width')
        if 'dodge' in boxplot_kwargs:
            boxplot_dodge = boxplot_kwargs.pop('dodge')
        if 'palette' in boxplot_kwargs:
            boxplot_palette = boxplot_kwargs.pop('palette')

        if 'size' in stripplot_kwargs:
            stripplot_size = stripplot_kwargs.pop('size')
        if 'color' in stripplot_kwargs:
            stripplot_color = stripplot_kwargs.pop('color')
        if 'alpha' in stripplot_kwargs:
            stripplot_alpha = stripplot_kwargs.pop('alpha')
        if 'dodge' in stripplot_kwargs:
            stripplot_dodge = stripplot_kwargs.pop('dodge')

        sns.boxplot(data,x='Resolution',y='Label',hue='Label',whis=boxplot_whis, width=boxplot_width, dodge=boxplot_dodge, palette=boxplot_palette, ax=ax, **boxplot_kwargs)
        sns.stripplot(data, x="Resolution", y="Label",size=stripplot_size,color=stripplot_color,alpha=stripplot_alpha,dodge=stripplot_dodge,ax=ax,**stripplot_kwargs)

        if ylabel:
            ax.set(ylabel=ylabel)
        else:
            ax.set(ylabel="")

        if xlabel:
            ax.set(xlabel=xlabel)
        else:
            ax.set(xlabel=f"Resolution [{self.time_unit}]")

        if not legend:
            ax.legend().set_visible(False)

        # Tweak the visual presentation
        ax.xaxis.grid(True)
        sns.despine(trim=True, left=True)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax

    def describe(self):
        '''Describe the stats of the resolution list
        
        Returns
        -------
        
        resolution_dict : dict
            Dictionary of relevant stats produced by `scipy.stats.describe <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html>`_

        Examples
        --------

        ..jupyter-execute::
            import pyleoclim as pyleo

            co2ts = pyleo.utils.load_dataset('AACO2')
            edc = pyleo.utils.load_dataset('EDC-dD')
            ms = edc & co2ts # create MS object
            ms_resolution = ms.resolution()
            ms_resolution.plot()
        
        '''
        
        resolution_dict = {}

        for resolution in self.resolution_list:
            stats = st.describe(resolution.resolution)._asdict()
            
            median = np.median(resolution.resolution)
            stats['median'] = median

            resolution_dict[resolution.label] = stats

        return resolution_dict