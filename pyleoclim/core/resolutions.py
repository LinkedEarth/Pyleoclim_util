#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolution objects are designed to contain, display, and analyze information on the resolution of the time axis of a Series object.
"""

from ..utils import tsutils, plotting, tsmodel, tsbase

import warnings

import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from matplotlib import gridspec

class Resolution:
    ''' Resolution object
    
    Resolution objects store time axis resolution information derived from Series objects.
    They are generated via the resolution method applied to a Series object and contain methods relevant to the analysis of resolution information.

    Parameters
    ----------

    resolution : list or numpy.array
        An array containing information on the time step between subsequent values in the original time series.

    time : list or numpy.array
        An array containing the time axis from the original series.
        This should be same length as resolution, so removing one value from the original axis is required.

    resolution_unit : str
        Unit of resolution. This should be the same as the original time axis.

    label : str
        Label for the resolution object.

    timeseries : pyleoclim.Series
        Original pyleoclim timeseries object.


    '''

    def __init__(self,resolution,time=None,resolution_unit=None,label=None,timeseries=None,):
        resolution = np.array(resolution)
        self.resolution = resolution
        self.timeseries = timeseries

        if label is None:
            if self.timeseries.label is None:
                self.label = 'Resolution'
            else:
                self.label = self.timeseries.label
        else:
            self.label = label

        if time is None:
            self.time = self.timeseries.time[1:]
        else:
            time = np.array(time)
            self.time = time

        if resolution_unit is None:
            if timeseries is not None:
                self.resolution_unit = timeseries.time_unit
            else:
                self.resolution_unit = resolution_unit
        #Include time reasoning
        else:
            self.resolution_unit = resolution_unit
        

    def describe(self):
        '''Describe the stats of the time series
        
        Returns
        -------
        stats : dict
            Dictionary of relevant stats produced by `scipy.stats.describe <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html>`_

        Examples
        --------

        .. jupyter-execute::

            ts = pyleo.utils.load_dataset('EDC-dD')
            resolution = ts.resolution()
            resolution.describe()
        
        '''
        
        stats = st.describe(self.resolution)._asdict()
        
        median = np.median(self.resolution)
        stats['median'] = median

        return stats
    
    def plot(self, figsize=[10, 4],
              marker=None, markersize=None, color=None,
              linestyle=None, linewidth=None, xlim=None, ylim=None,
              label=None, xlabel=None, ylabel=None, title=None, zorder=None,
              legend=True, plot_kwargs=None, lgd_kwargs=None, alpha=None,
              savefig_settings=None, ax=None, invert_xaxis=False, invert_yaxis=False):
        ''' Plot the timeseries

        Parameters
        ----------

        figsize : list
            a list of two integers indicating the figure size

        marker : str
            e.g., 'o' for dots
            See [matplotlib.markers](https://matplotlib.org/stable/api/markers_api.html) for details

        markersize : float
            the size of the marker

        color : str, list
            the color for the line plot
            e.g., 'r' for red
            See [matplotlib colors](https://matplotlib.org/stable/gallery/color/color_demo.html) for details

        linestyle : str
            e.g., '--' for dashed line
            See [matplotlib.linestyles](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html) for details

        linewidth : float
            the width of the line

        label : str
            the label for the line

        xlabel : str
            the label for the x-axis

        ylabel : str
            the label for the y-axis

        title : str
            the title for the figure

        zorder : int
            The default drawing order for all lines on the plot

        legend : {True, False}
            plot legend or not

        invert_xaxis : bool, optional
            if True, the x-axis of the plot will be inverted

        invert_yaxis : bool, optional
            same for the y-axis

        plot_kwargs : dict
            the dictionary of keyword arguments for ax.plot()
            See [matplotlib.pyplot.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html) for details

        lgd_kwargs : dict
            the dictionary of keyword arguments for ax.legend()
            See [matplotlib.pyplot.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html) for details

        alpha : float
            Transparency setting

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
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/stable/api/figure_api.html) for details.

        ax : matplotlib.axis
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/stable/api/axes_api.html) for details.

        Notes
        -----

        When `ax` is passed, the return will be `ax` only; otherwise, both `fig` and `ax` will be returned.

        See also
        --------

        pyleoclim.utils.plotting.savefig : saving a figure in Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            ts = pyleo.utils.load_dataset('EDC-dD')
            resolution = ts.resolution()
            resolution.plot()

        '''

        time_label,value_label = self.make_labels()

        if xlabel is None:
            xlabel = time_label

        if ylabel is None:
            ylabel = value_label

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()

        if label is None:
            label = self.label

        if label is not None:
            plot_kwargs.update({'label': label})

        if marker is not None:
            plot_kwargs.update({'marker': marker})

        if markersize is not None:
            plot_kwargs.update({'markersize': markersize})

        if color is not None:
            plot_kwargs.update({'color': color})

        if linestyle is not None:
            plot_kwargs.update({'linestyle': linestyle})

        if linewidth is not None:
            plot_kwargs.update({'linewidth': linewidth})

        if alpha is not None:
            plot_kwargs.update({'alpha': alpha})

        if zorder is not None:
            plot_kwargs.update({'zorder': zorder})

        res = plotting.plot_xy(
            self.time, self.resolution,
            figsize=figsize, xlabel=xlabel, ylabel=ylabel,
            title=title, savefig_settings=savefig_settings,
            ax=ax, legend=legend, xlim=xlim, ylim=ylim,
            plot_kwargs=plot_kwargs, lgd_kwargs=lgd_kwargs,
            invert_xaxis=invert_xaxis, invert_yaxis=invert_yaxis
        )

        return res
    
    def histplot(self, figsize=[10, 4], title=None, savefig_settings=None,
                    ax=None, ylabel='KDE', vertical=False, edgecolor='w', **plot_kwargs):
        ''' Plot the distribution of the resolution values

        Parameters
        ----------

        figsize : list
            a list of two integers indicating the figure size

        title : str

            the title for the figure

        savefig_settings : dict

            the dictionary of arguments for plt.savefig(); some notes below:

            - "path" must be specified; it can be any existed or non-existed path,
                with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        ax : matplotlib.axis, optional

            A matplotlib axis

        ylabel : str

            Label for the count axis

        vertical : {True,False}

            Whether to flip the plot vertically

        edgecolor : matplotlib.color

            The color of the edges of the bar

        plot_kwargs : dict

            Plotting arguments for seaborn histplot: https://seaborn.pydata.org/generated/seaborn.histplot.html

        See also
        --------

        pyleoclim.utils.plotting.savefig : saving figure in Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            ts = pyleo.utils.load_dataset('EDC-dD')
            res = ts.resolution()
            res.histplot()

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        #make the data into a dataframe so we can flip the figure
        _,value_label = self.make_labels()
        
        if vertical == True:
            data=pd.DataFrame({'value':self.resolution})
            ax = sns.histplot(data=data, y="value", ax=ax, kde=True, edgecolor=edgecolor, **plot_kwargs)
            ax.set_ylabel(value_label)
            ax.set_xlabel(ylabel)
        else:
            ax = sns.histplot(self.resolution, ax=ax, kde=True, edgecolor=edgecolor, **plot_kwargs)
            ax.set_xlabel(value_label)
            ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax
            
    def dashboard(self, figsize=[11, 8], title=None, plot_kwargs=None, histplot_kwargs=None, savefig_settings=None):
        '''Resolution plot dashboard

        Parameters
        ----------
        
        figsize : list or tuple, optional

            Figure size. The default is [11,8].

        title : str

            Figure title

        plot_kwargs : dict

            the dictionary of keyword arguments for ax.plot()

        histplot_kwargs : dict, optional

            The dictionary of keyword arguments for ax.histplot()

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


        Examples
        --------

        .. jupyter-execute::

            ts = pyleo.utils.load_dataset('EDC-dD')
            resolution = ts.resolution()
            resolution.dashboard()

        '''

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        histplot_kwargs = {} if histplot_kwargs is None else histplot_kwargs.copy()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        fig = plt.figure(figsize=figsize)
        ax = {}

        gs = gridspec.GridSpec(1, 3, wspace=0)
        gs.update(left=0, right=1.2)

        ax['res'] = fig.add_subplot(gs[0, :-1])
        self.plot(ax=ax['res'],**plot_kwargs)

        ax['res_hist'] = fig.add_subplot(gs[0, -1:])
        self.histplot(ax=ax['res_hist'],ylabel='Counts',vertical=True,**histplot_kwargs)

        if 'ylabel' not in histplot_kwargs:  
            ax['res_hist'].set_yticklabels([])
            ax['res_hist'].set_ylabel('')

        if title:
            fig.suptitle(title)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)

        return fig, ax
            
    def make_labels(self):
        '''
        Initialization of plot labels based on Series metadata

        Returns
        -------
        time_header : str
            Label for the time axis
        value_header : str
            Label for the value axis

        '''
        if self.timeseries.time_name is not None:
            time_name_str = self.timeseries.time_name
        else:
            time_name_str = 'time'

        value_name_str = 'resolution'

        if self.resolution_unit is not None:
            value_header = f'{value_name_str} [{self.resolution_unit}]'
        else:
            value_header = f'{value_name_str}'

        if self.timeseries.time_unit is not None:
            time_header = f'{time_name_str} [{self.timeseries.time_unit}]'
        else:
            time_header = f'{time_name_str}'

        return time_header, value_header
    
class MultipleResolution:
    ''' MultipleResolution object
    
    MultipleResolution objects store time axis resolution information derived from MultipleSeries objects.
    They are generated via the resolution method applied to a MultipleSeries object and contain methods relevant to the analysis of resolution information.

    Parameters
    ----------

    resolution_list : list
        List of resolution objects.

    time_unit : str
        The unit of time for the resolution.

    See also
    --------

    Resolution : The base class from which MultipleResolution is derived.

    '''

    def __init__(self,resolution_list,time_unit):
        self.resolution_list = resolution_list
        self.time_unit = time_unit

    def summary_plot(self,figsize=(10,8),xlabel=None,ylabel=None,legend=False,ax=None,
             boxplot_whis=[0,100], boxplot_width=.6, boxplot_dodge=False,boxplot_palette='viridis',
             stripplot_size=2,stripplot_color=".3",stripplot_alpha=.8,stripplot_dodge=False,
             boxplot_kwargs=None,stripplot_kwargs=None,savefig_settings=None):
        """Summary plot showing distribution of resolutions from each resolution object as box plots with overlaid strip plots.

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
            ms_resolution = ms.resolution(statistic=None)
            ms_resolution.summary_plot()

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
        # if 'log_scale' in boxplot_kwargs:
        #     log_scale = boxplot_kwargs.pop('log_scale')

        if 'size' in stripplot_kwargs:
            stripplot_size = stripplot_kwargs.pop('size')
        if 'color' in stripplot_kwargs:
            stripplot_color = stripplot_kwargs.pop('color')
        if 'alpha' in stripplot_kwargs:
            stripplot_alpha = stripplot_kwargs.pop('alpha')
        if 'dodge' in stripplot_kwargs:
            stripplot_dodge = stripplot_kwargs.pop('dodge')
        # if 'log_scale' in stripplot_kwargs:
        #     log_scale = stripplot_kwargs.pop('log_scale')

        sns.boxplot(data,x='Resolution',y='Label',hue='Label',whis=boxplot_whis,width=boxplot_width,dodge=boxplot_dodge,palette=boxplot_palette,ax=ax,**boxplot_kwargs)
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
            ms_resolution = ms.resolution(statistic=None)
            ms_resolution.describe()
        
        '''
        
        resolution_dict = {}

        for resolution in self.resolution_list:
            stats = st.describe(resolution.resolution)._asdict()
            
            median = np.median(resolution.resolution)
            stats['median'] = median

            resolution_dict[resolution.label] = stats

        return resolution_dict