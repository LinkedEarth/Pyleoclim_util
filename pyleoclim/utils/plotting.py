#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities, leveraging Matplotlib.
"""

__all__ = ['set_style', 'closefig', 'savefig']

import matplotlib.pyplot as plt
import pathlib
import matplotlib as mpl
from matplotlib import cm
import numpy as np
import pandas as pd
import collections.abc

from ..utils import lipdutils


# import pandas as pd
# from matplotlib.patches import Rectangle
# from matplotlib.collections import PatchCollection
# from matplotlib.colors import ListedColormap
# import seaborn as sns

# this is here because it's only used to set labels in plots
def infer_period_unit_from_time_unit(time_unit):
    ''' infer a period unit based on the given time unit

    '''
    if time_unit is None:
        period_unit = None
    else:
        unit_group = lipdutils.timeUnitsCheck(time_unit)
        if unit_group != 'unknown':
            if unit_group == 'kage_units':
                period_unit = 'kyrs'
            else:
                period_unit = 'yrs'
        else:
            period_unit = f'{time_unit}'
            # if time_unit[-1] == 's':
            #     period_unit = time_unit
            # else:
            #     period_unit = f'{time_unit}s'

    return period_unit


def scatter_xy(x, y, c=None, figsize=None, xlabel=None, ylabel=None, title=None,
               xlim=None, ylim=None, savefig_settings=None, ax=None,
               legend=True, plot_kwargs=None, lgd_kwargs=None):
    """
    Make scatter plot. 

    Parameters
    ----------
    x : numpy.array
        x value
    y : numpy.array
        y value
    c : array-like or list of colors or color, optional
      The marker colors. Possible values:
      - A scalar or sequence of n numbers to be mapped to colors using cmap and norm.
      - A 2D array in which the rows are RGB or RGBA.
      - A sequence of colors of length n.
      - A single color format string.
      Note that c should not be a single numeric RGB or RGBA sequence because that is indistinguishable from an array of values to be colormapped. If you want to specify the same RGB or RGBA value for all points, use a 2D array with a single row. Otherwise, value-matching will have precedence in case of a size matching with x and y.
      If you wish to specify a single color for all points prefer the color keyword argument.
      Defaults to None. In that case the marker color is determined by the value of color, facecolor or facecolors. In case those are not specified or None, the marker color is determined by the next color of the Axes' current "shape and fill" color cycle. This cycle defaults to rcParams["axes.prop_cycle"] (default: cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])).
    figsize : list, optional
        A list of two integers indicating the dimension of the figure. The default is None.
    xlabel : str, optional
        x-axis label. The default is None.
    ylabel : str, optional
        y-axis label. The default is None.
    title : str, optional
        Title for the plot. The default is None.
    xlim : list, optional
        Limits for the x-axis. The default is None.
    ylim : list, optional
        Limits for the y-axis. The default is None.
    savefig_settings : dict, optional
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified; it can be any existing or non-existing path,
          with or without a suffix; if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}
       The default is None.
    ax : pyplot.axis, optional
        The axis object. The default is None.
    legend : bool, optional
        Whether to include a legend. The default is True.
    plot_kwargs : dict, optional
        the keyword arguments for ax.plot(). The default is None.
    lgd_kwargs : dict, optional
        the keyword arguments for ax.legend(). The default is None.

    Returns
    -------
    ax : the pyplot.axis object

    """
    savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
    lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, c=c, **plot_kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if len(lgd_kwargs)>0:
        ax.legend(**lgd_kwargs)
    # if legend:
    #     ax.legend(**lgd_kwargs)
    else:
        ax.legend().remove()

    if 'fig' in locals():
        if 'path' in savefig_settings:
            savefig(fig, settings=savefig_settings)
        return fig, ax
    else:
        return ax


def plot_scatter_xy(x1, y1, x2, y2, figsize=None, xlabel=None,
                    ylabel=None, title=None, xlim=None, ylim=None,
                    savefig_settings=None, ax=None, legend=True,
                    plot_kwargs=None, lgd_kwargs=None):
    ''' Plot a scatter on top of a line plot.
    
    Parameters
    ----------
    x1 : array
      x axis of timeseries1 - plotted as a line
    y1 : array
     values of timeseries1 - plotted as a line
    x2 : array
        x axis of scatter points
    y2 : array
        y of scatter points
    figsize : list
        a list of two integers indicating the figure size
    xlabel : str
        label for x-axis
    ylabel : str
        label for y-axis
    title : str
        the title for the figure
    xlim : list
        set the limits of the x axis
    ylim : list
        set the limits of the y axis
    ax : pyplot.axis
        the pyplot.axis object
    legend : bool
        plot legend or not
    lgd_kwargs : dict
        the keyword arguments for ax.legend()
    plot_kwargs : dict
        the keyword arguments for ax.plot()
    savefig_settings : dict
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified; it can be any existing or non-existing path,
          with or without a suffix; if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}

    Returns
    -------
    ax : the pyplot.axis object
    
    See also
    -------- 
    
    pyleoclim.utils.plotting.set_style : set different styles for the figures. Should be set before invoking the plotting functions
    pyleoclim.utils.plotting.savefig : save figures
    
    '''
    # handle dict defaults
    savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
    lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x1, y1, **plot_kwargs, color='green')
    ax.scatter(x2, y2, color='red')

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if (legend is True) & (len(lgd_kwargs)>0):
        ax.legend(**lgd_kwargs)
    # if legend:
    #     ax.legend(**lgd_kwargs)
    else:
        ax.legend().remove()

    if 'fig' in locals():
        if 'path' in savefig_settings:
            savefig(fig, settings=savefig_settings)
        return fig, ax
    else:
        return ax


def plot_xy(x, y, figsize=None, xlabel=None, ylabel=None, title=None,
            xlim=None, ylim=None, savefig_settings=None, ax=None,
            legend=True, plot_kwargs=None, lgd_kwargs=None,
            invert_xaxis=False, invert_yaxis=False):
    ''' Plot a timeseries
    
    Parameters
    ----------
    x : array
        The time axis for the timeseries
    y : array
        The values of the timeseries
    figsize : list
        a list of two integers indicating the figure size
    xlabel : str
        label for x-axis
    ylabel : str
        label for y-axis
    title : str
        the title for the figure
    xlim : list
        set the limits of the x axis
    ylim : list
        set the limits of the y axis
    ax : pyplot.axis
        the pyplot.axis object
    legend : bool
        plot legend or not
    lgd_kwargs : dict
        the keyword arguments for ax.legend()
    plot_kwargs : dict
        the keyword arguments for ax.plot()
    mute : bool
        if True, the plot will not show;
        recommend to turn on when more modifications are going to be made on ax
         (going to be deprecated)
    savefig_settings : dict
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified; it can be any existing or non-existing path,
          with or without a suffix; if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}
    invert_xaxis : bool, optional
        if True, the x-axis of the plot will be inverted
    invert_yaxis : bool, optional
        if True, the y-axis of the plot will be inverted
        
    Returns
    -------
    ax : the pyplot.axis object

    See Also
    --------
    
    pyleoclim.utils.plotting.set_style : set different styles for the figures. Should be set before invoking the plotting functions

    pyleoclim.utils.plotting.savefig : save figures


    '''
    # handle dict defaults
    savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
    lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, **plot_kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if (legend is True) & (len(lgd_kwargs)>0):
    # if legend:
        ax.legend(**lgd_kwargs)
    else:
        ax.legend().remove()

    if invert_xaxis:
        ax.invert_xaxis()

    if invert_yaxis:
        ax.invert_yaxis()

    if 'fig' in locals():
        if 'path' in savefig_settings:
            savefig(fig, settings=savefig_settings)
        return fig, ax
    else:
        return ax


def stripes_xy(x, y, cmap='coolwarm', figsize=None, ax=None,
               vmin=None, vmax=None, xlabel=None, ylabel=None,
               title=None, xlim=None, savefig_settings=None, label_color=None,
               x_offset=0.05, label_size=None, show_xaxis=False,
               invert_xaxis=False, top_label=None, bottom_label=None):
    '''
    Represent y = f(x) as an Ed Hawkins "warming stripes" pattern
    Uses Matplotlib's pcolormesh'
    Credit: https://esmvalgroup.github.io/ESMValTool_Tutorial/files/warming_stripes.py
    
    Parameters
    ----------
    x : array
        Independent variable
    y : array
        Dependent variable (asumees centered and normalized to unit standard deviation)
    cmap: str
        colormap name
    figsize : list
        a list of two integers indicating the figure size
    ax : pyplot.axis
        the pyplot.axis object, default is None
    vmin: float 
        lower bound for colormap normalization
    vmax: float 
        upper bound for colormap normalization    
    top_label : str
        the "title" label for the stripe. Set to '' if no label is wanted
    bottom_label : str
        the "ylabel" explaining which variable is being plotted. Set to '' if no label is wanted
    label_size : int
        size of the text in labels (in points). Default is the Matplotlib 'axes.labelsize'] rcParams
    xlim : list
        set the limits of the x axis
    x_offset : float (0-1)
        value controlling the horizontal offset between stripes and labels (default = 0.05)
    show_xaxis : bool
        flag indicating whether or not the x-axis should be shown (default = False)
    
    savefig_settings : dict
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified; it can be any existing or non-existing path,
          with or without a suffix; if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}
    invert_xaxis : bool, optional
        if True, the x-axis of the plot will be inverted

    Returns
    -------
    ax, or (fig, ax) if no axes were provided.

    See Also
    --------

    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html

    https://matplotlib.org/stable/tutorials/colors/colormapnorms.html


    '''
    # handle dict defaults
    savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if label_size is None:
        label_size = mpl.rcParams['axes.labelsize']

    ones = np.array([0, 1])
    # ax.set_axis_off()
    ax.pcolormesh(x, ones, np.vstack([y, y]), cmap=cmap,
                  vmin=vmin, vmax=vmax, shading='auto')
    # hide y axis
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    # manage x axis
    ax.spines['bottom'].set_visible(show_xaxis)
    ax.get_xaxis().set_visible(show_xaxis)
    if show_xaxis is True and xlabel is not None:
        ax.set_xlabel(xlabel)

    # parameters for label position
    thickness = ax.get_ybound()[1]
    xmax = ax.get_xbound()[1] * (1 + x_offset / 10)
    # xmax = x.max()*0.8*(1+x_offset)
    ax.text(xmax, 0.5 * thickness, top_label, color=label_color,
            fontsize=label_size, fontweight='bold')
    ax.text(xmax, 0 * thickness, bottom_label, color=label_color,
            fontsize=label_size)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if invert_xaxis:
        ax.invert_xaxis()

    if 'fig' in locals():
        # fig.tight_layout()
        if 'path' in savefig_settings:
            savefig(fig, settings=savefig_settings)
        return fig, ax
    else:
        return ax


def closefig(fig=None):
    '''Close the figure

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        The matplotlib figure object

    '''
    if fig is not None:
        plt.close(fig)
    else:
        plt.close()


def savefig(fig, path=None, dpi=300, settings={}, verbose=True):
    ''' Save a figure to a path

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        the figure to save
    path : str
        the path to save the figure, can be ignored and specify in "settings" instead
    dpi : int
        resolution in dot (pixels) per inch. Default: 300. 
    settings : dict
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified in settings if not assigned with the keyword argument;
          it can be any existing or non-existing path, with or without a suffix;
          if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}
    verbose : bool, {True,False}
        If True, print the path of the saved file.
        
    '''
    if path is None and 'path' not in settings:
        raise ValueError('"path" must be specified, either with the keyword argument or be specified in `settings`!')

    savefig_args = {'bbox_inches': 'tight', 'path': path, 'dpi': dpi}
    savefig_args.update(settings)

    path = pathlib.Path(savefig_args['path'])
    savefig_args.pop('path')

    dirpath = path.parent
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f'Directory created at: "{dirpath}"')

    path_str = str(path)
    if path.suffix not in ['.eps', '.pdf', '.png', '.ps']:
        path = pathlib.Path(f'{path_str}.pdf')

    fig.savefig(path_str, **savefig_args)
    plt.close(fig)

    if verbose:
        print(f'Figure saved at: "{str(path)}"')


def set_style(style='journal', font_scale=1.0, dpi=300):
    ''' Modify the visualization style
    
    This function is inspired by `Seaborn <https://github.com/mwaskom/seaborn>`_.
   
    
    Parameters
    ----------
    style : {journal,web,matplotlib,_spines, _nospines,_grid,_nogrid}

        set the styles for the figure:

            - journal (default): fonts appropriate for paper
            - web: web-like font (e.g. ggplot)
            - matplotlib: the original matplotlib style

            In addition, the following options are available:
            
            - _spines/_nospines: allow to show/hide spines
            - _grid/_nogrid: allow to show gridlines (default: _grid)
    
    font_scale : float

        Default is 1. Corresponding to 12 Font Size. 
    
    '''
    font_dict = {
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
    }

    style_dict = {}
    if 'journal' in style:
        style_dict.update({
            'axes.axisbelow': True,
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.grid': True,
            'grid.color': 'lightgrey',
            'grid.linestyle': '--',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],

            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': False,

            'axes.linewidth': 1,
            'grid.linewidth': 1,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 1,

            'xtick.major.width': 1.25,
            'ytick.major.width': 1.25,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,
        })
    elif 'web' in style:
        style_dict.update({
            'figure.facecolor': 'white',

            'axes.axisbelow': True,
            'axes.facecolor': 'whitesmoke',
            'axes.edgecolor': 'lightgrey',
            'axes.grid': True,
            'grid.color': 'white',
            'grid.linestyle': '-',
            'xtick.direction': 'out',
            'ytick.direction': 'out',

            'text.color': 'grey',
            'axes.labelcolor': 'grey',
            'xtick.color': 'grey',
            'ytick.color': 'grey',

            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],

            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': False,

            'axes.linewidth': 1,
            'grid.linewidth': 1,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 1,

            'xtick.major.width': 1.25,
            'ytick.major.width': 1.25,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,
        })

    elif 'matplotlib' in style:
        # mpl.rcParams.update(mpl.rcParamsDefault)
        style_dict.update({})


    else:
        raise ValueError(f'Style [{style}] not availabel!')

    if '_spines' in style:
        style_dict.update({
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.spines.top': True,
        })
    elif '_nospines' in style:
        style_dict.update({
            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.right': False,
            'axes.spines.top': False,
        })

    if '_grid' in style:
        style_dict.update({
            'axes.grid': True,
        })
    elif '_nogrid' in style:
        style_dict.update({
            'axes.grid': False,
        })

    figure_dict = {
        'savefig.dpi': dpi,
    }

    # modify font size based on font scale
    font_dict.update({k: v * font_scale for k, v in font_dict.items()})

    for d in [style_dict, font_dict, figure_dict]:
        mpl.rcParams.update(d)


def make_phantom_ax(ax):
    ''' Remove all visual annotation from ax object

    This function removes axis lines, axis labels, tick labels, tick marks and grid lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes object
        the axes object to clear

    Returns
    -------
    ax : matplotlib.axis
        the axis object from matplotlib
        See [matplotlib.axes](https://matplotlib.org/stable/api/axes_api.html) for details.

    '''

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    # _ax.set_xlim(xlim)
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax


def make_annotation_ax(fig, ax, loc='overlay',
                       ax_name='highlighted_intervals',
                       height=None, v_offset=0, b=None,
                       width=None, h_offset=0, l=None,
                       zorder=-1):
    ''' Makes a clean axis for adding annotation

    This function creates a new axes for adding annotation.
    If the bottom left corner is not specified, it is established based on the ax objects in ax.
    If there is only one ax object, this overkill, but is helpful to introduce annotations that span multiple data axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes object or dict
        If ax is a dict, assumes data axes are assigned to integer keys and
        supplemental axes have string keys

    loc : string
        if "overlay", annotation ax will attempt to cover the area with data axes
        if "above", annotation ax will be located directly above the top data ax
        if "below", annotation ax will be located below the bottom data ax

    ax_name : string
        name associated with new ax object

    height : float
        height of annotation ax
        if loc = "above" or "below", height=.025 if not specified
        if loc = "overlay", height=vertical span of data axes, if not specified

    v_offset : float
        vertical offset between data plot area and annotation ax
        a positive v_offset will place the bottom corner higher

    width : float
        width of annotation ax
        horizontal span of data axes, if not specified

    b : float
        location of bottom corner of annotation ax

    h_offset : float
        horizontal offset from left corner
        a positive h_offset will place the left corner farther to the right

    l : float
        location of left corner of annotation ax

    zorder : numeric
        index of annotation ax layer in fig
        zorder = -1 will place the layer behind other layers
        zorder = 1000 will place the layer in front of other layers

    Returns
    -------
    ax_d : dict
        ax_d contains the original ax object(s) and new annotation ax assigned to specified ax_name
        See [matplotlib.axes](https://matplotlib.org/stable/api/axes_api.html) for details.

    '''

    if type(ax) != dict:
        ax_d = {0: ax}
    else:
        ax_d = ax

    ll = []
    ur = []
    keys_list = [key for key in ax_d.keys() if type(key) == int]
    keys_list.sort()

    for ax_key in keys_list:
        bbox_coords = ax_d[ax_key].get_position()
        ll.append(bbox_coords._points[0].tolist())
        ur.append(bbox_coords._points[1].tolist())
        xlims = ax_d[ax_key].get_xlim()

    if l is None:
        l = min([_ll[0] for _ll in ll])

    u = max([_ur[1] for _ur in ur])
    r = max([_ur[0] for _ur in ur])

    if loc == 'overlay':
        if b is None:
            b = min([_ll[1] for _ll in ll])
        if height is None:
            height = u - b
    else:
        if height is None:
            height = .025
        if loc == 'above':
            if b is None:
                b = u
        if loc == 'below':
            if b is None:
                b = min([_ll[1] for _ll in ll]) - height

    if width is None:
        width = r - l
    b += v_offset
    l += h_offset

    ax_d[ax_name] = fig.add_axes([l, b, width, height],
                                 **{'zorder': zorder})

    ax_d[ax_name].set_xlim(xlims)
    ax_d[ax_name] = make_phantom_ax(ax_d[ax_name])
    ax_d[ax_name].set_facecolor((1, 1, 1, 0))

    return ax_d


import matplotlib.patches as mpatches


def hightlight_intervals(ax, intervals, labels=None, color='g', alpha=.3, legend=True):
    ''' Hightlights intervals

    This function highlights intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes object

    intervals : list
        list of intervals to be highlighted

    color : string or list
        If a string is passed, all intervals will be the specified color
        If a list is passed, the list is expected to be the same length as intervals

    alpha : float or list
        If a float is passed, all intervals will have the same specified alpha value
        If a list is passed, the list is expected to be the same length as intervals

    Returns
    -------
    ax : matplotlib.axis
        the axis object from matplotlib
        See [matplotlib.axes](https://matplotlib.org/stable/api/axes_api.html) for details.


    Examples
    --------

    .. jupyter-execute::

        import pyleoclim as pyleo

        ts_18 = pyleo.utils.load_dataset('cenogrid_d18O')
        ts_13 = pyleo.utils.load_dataset('cenogrid_d13C')
        ms = pyleo.MultipleSeries([ts_18, ts_13], label='Cenogrid', time_unit='ma BP')

        fig, ax = ms.stackplot(linewidth=0.5, fill_between_alpha=0)

        ax=pyleo.utils.plotting.make_annotation_ax(fig, ax, ax_name = 'highlighted_intervals', zorder=-1)
        intervals = [[3, 8], [12, 18], [30, 31], [40,43], [49, 60], [60, 65]]
        ax['highlighted_intervals'] = pyleo.utils.plotting.hightlight_intervals(ax['highlighted_intervals'], intervals,
            color='g', alpha=.1)

    '''

    if isinstance(intervals[0], list) is False:
        intervals = [intervals]

    handles = []

    new_labels = []
    new_colors = []
    new_alphas = []

    for ik, _ts in enumerate(intervals):
        if isinstance(color, list) is True:
            c = color[ik]
        else:
            c = color
        new_colors.append(c)

        if isinstance(alpha, list) is True:
            a = alpha[ik]
        else:
            a = alpha
        new_alphas.append(a)

        if isinstance(labels, list) is True:
            label = labels[ik]
        else:
            label = ''
        new_labels.append(label)

        ax.axvspan(_ts[0], _ts[1], facecolor=c, alpha=a)

    return ax


def get_label_width(ax, label, buffer=0., fontsize=10):
    """
    Helper function to find width of text when rendered in ax object
    """

    text = ax.text(0, 0, label, size=fontsize)
    width = text.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width
    text.remove()  # Remove the text used for measurement

    return width + buffer


def calculate_overlapping_sets(fig, ax, labels, x_locs, fontsize, buffer=.1):
    """
    Calculate overlapping sets of labels based on their positions and widths.

    This function identifies sets of labels that would overlap if rendered at the same height on a plot.
    It is used to determine how to place labels to avoid overlap in visualizations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object on which the labels will be plotted.

    labels : list of str
        A list of label strings.

    x_locs : list of float
        A list of x-coordinates where the labels are to be positioned.

    fontsize : int
        The font size used for the labels.

    buffer : float, optional
        Additional space to consider around each label to prevent overlap.
        Defaults to 0.1.

    Returns
    -------
    list of list of int : A list where each sublist contains the indices of overlapping labels.


    """

    # Calculate the horizontal span of each label
    intervals = []
    for i, label in enumerate(labels):
        w = get_label_width(ax, label, buffer=buffer, fontsize=fontsize)
        # ann = ax.text(x_locs[i], 0, label, size=fontsize)
        # box = ax.transData.inverted().transform(ann.get_tightbbox(fig.canvas.get_renderer()))
        # w = box[1][0] - box[0][0] + buffer
        # ann.remove()

        interval = pd.Interval(left=x_locs[i] - w / 2, right=x_locs[i] + w / 2)
        intervals.append(interval)

    # Group overlapping labels
    overlapping_sets = []
    for i, interval_i in enumerate(intervals):
        found = False
        for overlap_set in overlapping_sets:
            if any(interval_i.overlaps(intervals[j]) for j in overlap_set):
                overlap_set.add(i)
                found = True
                break
        if not found:
            overlapping_sets.append({i})

    # Convert sets to sorted lists
    return [sorted(list(s)) for s in overlapping_sets]


def label_intervals(fig, ax, labels, x_locs, orientation='north', overlapping_sets=None, baseline=0.5,
                    height=0.5, buffer=0.1, fontsize=10, linewidth=None, linestyle_kwargs=None,
                    text_kwargs=None
                    ):
    """
    Place labels on a plot with given orientations and style parameters, avoiding overlaps.

    This function positions labels at specified x-locations with adjustments to avoid overlaps.
    Labels can be oriented either above (north) or below (south) a baseline.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object where the labels are to be placed.

    labels : list of str
        A list of label strings.

    x_locs : list of float
        A list of x-coordinates for the labels.

    orientation : str, optional
        The vertical orientation of the labels, either 'north' or 'south'. Defaults to 'north'.

    overlapping_sets : list of list of int, optional
        Precomputed overlapping sets of labels. If None, the function will compute them. Defaults to None.

    baseline : float, optional
        The baseline height for the first label slot. Defaults to 0.5.

    height : float, optional
        The vertical spacing between slots. Defaults to 0.5.

    buffer : float, optional
        Horizontal buffer space around labels to prevent overlap. Defaults to 0.1.

    fontsize : int, optional
        Font size for labels. Defaults to 10.

    linewidth : float, optional
        Line width for connecting lines. If None, defaults to 1.

    linestyle_kwargs : dict, optional
        Additional keyword arguments for styling the connecting lines (per Matplotlib).

    text_kwargs : dict, optional
        Additional keyword arguments for styling the text labels (per Matplotlib).

    Returns
    -------
    matplotlib.axes.Axes: The modified Axes object with labels placed.

    Examples
    --------

    .. jupyter-execute::

        import pyleoclim as pyleo
        import numpy as np

        ts_18 = pyleo.utils.load_dataset('cenogrid_d18O')
        ts_13 = pyleo.utils.load_dataset('cenogrid_d13C')
        ms = pyleo.MultipleSeries([ts_18, ts_13], label='Cenogrid', time_unit='ma BP')

        fig, ax = ms.stackplot(linewidth=0.5, fill_between_alpha=0)

        ax=pyleo.utils.plotting.make_annotation_ax(fig, ax, ax_name = 'epochs', height=.03,
                                           loc='above', v_offset=.015,zorder=-2)
        ax['epochs'].set_facecolor((1, 1, 1, 0))

        ceno_intervals_pairs = [[0.0, 0.01], [0.01, 1.6], [1.6, 5.3], [5.3, 23.7], [23.7, 36.6], [36.6, 57.8], [57.8, 66.4]]
        ceno_epoch_labels = ['Holocene', 'Pleistocene', 'Pliocene', 'Miocene', 'Oligocene', 'Eocene', 'Paleocene']
        ax['epochs'].set_ylim([-1,0])

        colors = ['r', 'm', 'orange', 'blue', 'green', 'aqua', 'navy', 'pink']#['r', 'b']#'r' if ik%2 ==0 else 'b' for ik, _ts in enumerate(geo_ts)]
        ax['epochs'] = pyleo.utils.plotting.hightlight_intervals(ax['epochs'],
                                                    ceno_intervals_pairs, color=colors,
                                                                              alpha=.1)

        ### EPOCHS (labels)
        ax=pyleo.utils.plotting.make_annotation_ax(fig, ax['epochs'], ax_name = 'epoch_annotation',
                                                   zorder=1, v_offset=0.01,
                                                   height=.25, loc='above')

        x_locs = [np.mean(interval) for interval in ceno_intervals_pairs]
        ax['epoch_annotation'].set_ylim([0,3])
        ax['epoch_annotation'] = pyleo.utils.plotting.label_intervals(fig, ax['epoch_annotation'], ceno_epoch_labels, x_locs,
                                                               orientation='north', baseline=.45, height=0.35, buffer=0.1,
                                           linestyle_kwargs= {'color':'gray'}, text_kwargs={'fontsize':10, 'va':'bottom'}
                                          )


    """

    if linestyle_kwargs is None:
        linestyle_kwargs = {}

    linestyle_defaults = {'linestyle': '--', 'color': 'gray', 'linewidth': 1 if linewidth is None else linewidth}
    for key in linestyle_defaults:
        if key not in linestyle_kwargs:
            linestyle_kwargs[key] = linestyle_defaults[key]

    if text_kwargs is None:
        text_kwargs = {}

    text_defaults = {'fontsize': 10 if fontsize is None else fontsize, 'ha': 'center'}
    for key in text_defaults:
        if key not in text_kwargs:
            text_kwargs[key] = text_defaults[key]
    fontsize = text_kwargs['fontsize']

    # if overlapping sets aren't specified, calculate them
    if overlapping_sets is None:
        overlapping_sets = calculate_overlapping_sets(fig, ax, labels, x_locs, fontsize, buffer=buffer)

    label_alignments = ['center' for _ in labels]
    label_slots = [0 for _ in labels]

    for overlap_set in overlapping_sets:
        if len(overlap_set) > 1:
            sorted_set = sorted(overlap_set, key=lambda i: x_locs[i])
            peak = len(sorted_set) // 2
            for i, label_index in enumerate(sorted_set):
                label_slots[label_index] = i if i <= peak else peak - (i - peak)

            cluster_min, cluster_max = x_locs[sorted_set[0]], x_locs[sorted_set[-1]]
            for i, label_index in enumerate(sorted_set):
                if i == 0:
                    label_alignments[label_index] = 'right'
                else:
                    if len(sorted_set) == 2:
                        label_alignments[label_index] = 'center'
                    else:
                        if i == int((len(sorted_set) - 1) / 2):
                            label_alignments[label_index] = 'center'
                        elif i > int((len(sorted_set) - 1) / 2):
                            label_alignments[label_index] = 'left'
                        else:
                            label_width = get_label_width(ax, labels[label_index], buffer=buffer, fontsize=fontsize)
                            # label_width = get_label_width(labels[label_index])
                            if x_locs[label_index] - label_width / 2 < cluster_min:
                                label_alignments[label_index] = 'right'
                            elif x_locs[label_index] + label_width / 2 > cluster_max:
                                label_alignments[label_index] = 'left'
                            else:
                                label_alignments[label_index] = 'center'

        else:
            label_index = overlap_set[0]
            label_alignments[label_index] = 'center'

    for i, label in enumerate(labels):
        label_text_kwargs = text_kwargs.copy()
        slot_height = baseline + label_slots[i] * height if orientation == 'north' else -baseline - label_slots[
            i] * height

        label_text_kwargs['ha'] = label_alignments[i]
        if 'va' not in label_text_kwargs:
            label_text_kwargs['va'] = 'bottom' if orientation == 'north' else 'top'

        ax.text(x_locs[i], slot_height, label, **label_text_kwargs)
        ax.plot([x_locs[i], x_locs[i]], [0, slot_height], **linestyle_kwargs)

    return ax


def make_scalar_mappable(cmap=None, hue_vect=None, n=None, norm_kwargs=None):
    """
    Create a ScalarMappable object for mapping scalar data to colors.

    This function configures and returns a ScalarMappable object based on the provided colormap (`cmap`), the scalar values (`hue_vect`), the number of discrete colors (`n`), and normalization parameters (`norm_kwargs`). It supports dynamic selection of normalization and colormap based on the input parameters and the range of scalar values.

    Parameters
    ----------
    cmap : str, list, or None, optional
        The colormap to use for mapping scalar data to colors. Can be a name of a matplotlib colormap (str), a list of color names, or None. If None, defaults to 'vlag' if conditions for centered normalization are met, otherwise 'viridis'.
    hue_vect : np.ndarray, pd.Series, list, or None, optional
        An array-like object containing the scalar values to be mapped to colors. These values are used to determine the range and center for normalization.
    n : int or None, optional
        Specifies the number of discrete colors in the colormap if `cmap` is provided as a list. If None, the number of colors is not explicitly set.
    norm_kwargs : dict or None, optional
        A dictionary containing keyword arguments for the normalization process, specifically supporting 'vcenter' and 'clip'. Defaults to {'vcenter': 0, 'clip': False} if not provided or if provided keys are missing.

    Returns
    -------
    ax_sm : matplotlib.cm.ScalarMappable
        The configured ScalarMappable object, which can be used to map scalar data to colors based on the specified colormap and normalization settings.


    Examples
    --------

    .. jupyter-execute::

        import pyleoclim as pyleo
        import numpy as np

        scalar_values = np.random.randn(100)
        sm = pyleo.utils.plotting.make_scalar_mappable(cmap='viridis', hue_vect=scalar_values)
        # Now `sm` can be used with matplotlib plotting functions to map scalar values to colors.

        sm = pyleo.utils.plotting.make_scalar_mappable(cmap='viridis', hue_vect=scalar_values, n=10)
        # This creates a ScalarMappable a discrete color scale.

        sm = pyleo.utils.plotting.make_scalar_mappable(cmap=['blue', 'white', 'red'], hue_vect=scalar_values, norm_kwargs={'vcenter': 0})
        # This creates a ScalarMappable with a custom linear segmented colormap and centered normalization.

    """

    # if type(hue_vect) in [np.ndarray, pd.Series, list]:
    if isinstance(hue_vect, collections.abc.Iterable) and not isinstance(hue_vect, dict):
        ax_cmap = None
        ax_norm = None

        if type(norm_kwargs) != dict:
            norm_kwargs = {}
        if 'vcenter' not in norm_kwargs.keys():
            norm_kwargs['vcenter'] = 0
        if 'clip' not in norm_kwargs.keys():
            norm_kwargs['clip'] = False

        if all(isinstance(i, (int, float)) for i in hue_vect):
            if np.any((norm_kwargs['vcenter'] < max(hue_vect)) | (norm_kwargs['vcenter'] > min(hue_vect))) == True:
                if cmap is None:
                    cmap = 'vlag'
                # ax_cmap = keep_center_colormap(cmap, min(hue_vect), max(hue_vect), center=0)
                ax_norm = mpl.colors.CenteredNorm(
                    **norm_kwargs)  # vcenter=0, clip=False)#TwoSlopeNorm(0, vmin=min(hue_vect), vmax=max(hue_vect)) #
            else:
                ax_norm = mpl.colors.Normalize(vmin=min(hue_vect), vmax=max(hue_vect), clip=False)
                if cmap is None:
                    cmap = 'viridis'

        # if np.any((norm_kwargs['vcenter'] < max(hue_vect)) | (norm_kwargs['vcenter'] > min(hue_vect))) == True:
        #     if cmap is None:
        #         cmap = 'vlag'
        #     # ax_cmap = keep_center_colormap(cmap, min(hue_vect), max(hue_vect), center=0)
        #     ax_norm = mpl.colors.CenteredNorm(
        #         **norm_kwargs)  # vcenter=0, clip=False)#TwoSlopeNorm(0, vmin=min(hue_vect), vmax=max(hue_vect)) #
        # else:
        #     ax_norm = mpl.colors.Normalize(vmin=min(hue_vect), vmax=max(hue_vect), clip=False)
        #     if cmap is None:
        #         cmap = 'viridis'

    if ax_cmap is None:
        if type(cmap) == list:
            if n is None:
                ax_cmap = mpl.colors.LinearSegmentedColormap.from_list("MyCmapName", cmap)
            else:
                ax_cmap = mpl.colors.LinearSegmentedColormap.from_list("MyCmapName", cmap, N=n)
        elif type(cmap) == str:
            if n is None:
                ax_cmap = plt.get_cmap(name=cmap)
            else:
                ax_cmap = plt.get_cmap(name=cmap, lut=n)
        else:
            print('what madness is this?')
    ax_sm = cm.ScalarMappable(norm=ax_norm, cmap=ax_cmap)

    return ax_sm


import copy


def consolidate_legends(ax, split_btwn=True, hue='relation', style='exp_type', size=None, colorbar=False):
    break_pts = []
    hs, ls = [], []
    for ip, _ax in enumerate(ax):
        try:
            legend = ax[ip].get_legend()
            l2 = [_l._text for _l in legend.get_texts()]
            h2 = legend.legendHandles
        except:
            ax[ip].legend()
            h2, l2 = ax[ip].get_legend_handles_labels()
        hs.append(h2)
        ls.append(l2)

        break_pts2 = []
        for ib, _l in enumerate(l2):
            if _l in [hue, style, size]:
                break_pt2 = ib
                break_pts2.append(ib)
        break_pts2.append(len(l2))
        break_pts.append(break_pts2)

    labels = []
    handles = []

    print(break_pts, ls)
    for iq, bp_lst in enumerate(break_pts):  # [1:]):
        for ik, bp in enumerate(bp_lst):
            if ik > 0:
                if split_btwn is True:
                    labels.append('')
                    handles.append(copy.copy(hs[0][-1]))
                    handles[-1].set_alpha(0)
                for im, _l in enumerate(ls[iq][:bp]):
                    if _l not in labels:
                        labels.append(_l)
                        handles.append(hs[iq][im])

    if colorbar is True:
        start = 0
        end = 0
        looking = True
        for ib, _l in enumerate(labels):
            if looking is True:
                if _l == hue:
                    start = ib
                elif _l in [size, style]:
                    end = ib
                    looking = False

        labels = labels[0:start] + labels[end:]  # [start:]
        handles = handles[0:start] + handles[end:]

    start = 0
    looking = True
    for ib, _l in enumerate(labels):
        if looking is True:
            if _l == '':
                start = ib + 1
            else:
                looking = False

    labels = labels[start:]
    handles = handles[start:]

    return handles, labels


# def consolidate_legends(ax, split_btwn=True, hue='relation', style='exp_type', size=None):
#     break_pts = []
#     hs, ls = [], []
#     for ip, _ax in enumerate(ax):
#         try:
#             legend = ax[ip].get_legend()
#             l2 = [_l._text for _l in legend.get_texts()]
#             h2 = legend.legendHandles
#         except:
#             ax[ip].legend()
#             h2, l2 = ax[ip].get_legend_handles_labels()
#         hs.append(h2)
#         ls.append(l2)
#
#         break_pt2 = len(l2)
#         for ib, _l in enumerate(l2):
#             if _l in [hue, style, size]:
#                 break_pts.append(ib)
#
#     labels = []
#     handles = []
#
#     if break_pts[0] ==0:
#         break_pt_start = 1
#     else:
#         break_pt_start = 0
#
#     handles += hs[0][:break_pts[break_pt_start]]
#     labels += ls[0][:break_pts[break_pt_start]]
#
#     print(ls)
#     if len(break_pts)>1:
#         for ik, bp in enumerate(break_pts[break_pt_start:]):
#             if split_btwn is True:
#                 labels.append('')
#                 handles.append(copy.copy(handles[-1]))
#                 handles[-1].set_alpha(0)
#             for im, _l in enumerate(ls[0][:bp]):
#                 if _l not in labels:
#                     labels.append(_l)
#                     handles.append(hs[0][im])
#     print(labels)

#     labels.append('')
#     handles.append(copy.copy(handles[-1]))
#     handles[-1].set_alpha(0)
#
#     handles += hs[0][break_pts[break_pt_start]:]
#     labels += ls[0][break_pts[break_pt_start]:]
#
# for ik, bp in enumerate(break_pts[1:]):
#     ik += 1
#     if split_btwn is True:
#         labels.append('')
#         handles.append(copy.copy(handles[-1]))
#         handles[-1].set_alpha(0)
#     for im, _l in enumerate(ls[ik][bp:]):
#         if _l not in labels:
#             labels.append(_l)
#             handles.append(hs[ik][im])
#
# return handles, labels


def keep_center_colormap(cmap, vmin, vmax, center=0):
    """
    Adjust a colormap so that a specific value remains centered, and extend its limits symmetrically.

    This function modifies a given colormap such that the color representing the 'center' value
    remains at the center of the colormap. It does this by adjusting the minimum and maximum values
    symmetrically around the center and ensuring that the colormap covers a range that is at least
    20% larger than the absolute range from the center to either the original minimum or maximum value.
    This is particularly useful for visualizing data with a significant central value (e.g., zero in
    anomaly maps) to ensure that the colormap visually represents deviations from this center in a balanced manner.

    Parameters
    ----------
    cmap : str or Colormap
        The name of the colormap or a Colormap instance to be adjusted.
    vmin : float
        The original minimum value in the data range that the colormap should cover.
    vmax : float
        The original maximum value in the data range that the colormap should cover.
    center : float, optional
        The value that should be centered in the adjusted colormap. Default is 0.


    Returns
    -------
    newmap : matplotlib.colors.ListedColormap
        A new colormap instance adjusted so that 'center' is in the middle of the colormap,
        with its range symmetrically extended to ensure balanced representation of values around the center.


    Notes
    -----

    The adjustment involves shifting the original `vmin` and `vmax` values to be symmetric around the `center`,
    then expanding the range by at least 20% to ensure that the colormap's central part accurately represents
    the centered value across the data. This adjusted colormap can then be used for data visualization tasks
    where maintaining a perceptual 'zero' or central reference point is important.

    """

    vmin = vmin - center
    vmax = vmax - center

    vdelta = max([.2 * abs(vmin), .2 * abs(vmax)])
    vmax = vmax + .2 * vdelta
    vmin = vmin - .2 * vdelta

    dv = max(-vmin, vmax) * 2
    N = int(256 * dv / (vmax - vmin))
    cont_map = plt.get_cmap(cmap, N)
    newcolors = cont_map(np.linspace(0, 1, N))
    beg = int((dv / 2 + vmin) * N / dv)
    end = N - int((dv / 2 - vmax) * N / dv)
    newmap = mpl.colors.ListedColormap(newcolors[beg:end])

    return newmap
