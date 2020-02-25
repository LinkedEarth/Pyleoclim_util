#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 05:45:52 2020

@author: deborahkhider

Contains all relevant plotting functions
"""

import matplotlib.pyplot as plt
import pathlib
import matplotlib as mpl
import seaborn as sns


def plot(x,y,markersize=50,marker='ro',x_label=None ,y_label=None,
         title =None, legend=True, figsize =[10,4], ax = None, **plot_args):
    """ Make a 2-D plot
    
    Args
    ----
    
    x : numpy array
       a 1xn numpy array of values for the x-axis
    y : numpy array
       a 1xn numpy array for the y-axis
    markersize : int
                the size of the marker
    marker : string or list
            color and shape of the marker
    x_axis_label : str
                  the label for the x-axis
    y_axis_label : str
                  the label for the y-axis
    title : str
           the title for the plot
    figsize : list
              the size of the figure
    ax : object
        The axis to be specified (useful to integrate plot into a subplot)
            
    Returns
    -------
    
    ax : The axis 
    fig : The figure
    
    """
    
    if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, **plot_args)

    time_label, value_label = self.make_labels()

    ax.set_xlabel(time_label)
    ax.set_ylabel(value_label)

    if title is not None:
        ax.set_title(title)

    if 'fig' in locals():
        if 'path' in savefig_settings:
            savefig(fig, savefig_settings)
        else:
            showfig(fig)
        return fig, ax
    else:
        return ax
    
    
def plot_ens(ageEns, y, ens = None, color = 'r', alpha = 0.005, x_label = None,
            y_label = None, title = None, figsize = [10,4], ax = None):
    """Plot Ensemble Values
    
    This function allows to plot all or a subset of ensemble members of a 
    timeseries
    
    Args
    ----
    
    ageEns : numpy array
            Age ensemble data. Iterations should be stored in columns
    y : numpy array
       Ordinate values
    ens : int
         Number of ensemble to plots. If None, will choose either the number
         of ensembles stored in the ensemble matrix or 500, whichever is lower
    color : str
           Linecolor (default is red)
    alpha : float
           Transparency setting for each line (default is 0.005)
    x_label : str
             Label for the x-axis
    y_label : str
             Label for the y-axis
    title : str
           Title for the figure
    figsize : list
             Size of the figure. Default is [10,4]
    ax : object
        Return as axis instead of figure
    
    Returns
    -------
    ax : Axis for the figure
    fig : The figure
    
    TODO
    ----
    Enable paleoEnsemble       
    
    """

    #Make sure that the ensemble and paleo values are numpy arrays
    ageEns = np.array(ageEns)
    y = np.array(y)
    
    # Make sure that the length of y is the same as the number of rows in ensemble array
    if len(y) != np.shape(ageEns)[0]:
        raise ValueError("The length of the paleoData is different than number of rows in ensemble table!")

    # Figure out the number of ensembles to plot
    if not ens:
        if np.shape(ageEns)[1]<500:
            ens = np.shape(ageEns)[1]
        else:
            ens = 500
            print("Plotting 500 ensemble members")
    elif ens > np.shape(ageEns)[1]:
        ens = np.shape(ageEns)[1]
        print("Plotting all available ensemble members") 
        
    # Figure setting
    if not ax:
        fig, ax = plt.subplots(figsize = figsize)
        
    # Finally make the plot
    plt.style.use("ggplot")
    for i in np.arange(0,ens,1):
        plt.plot(ageEns[:,i],y,alpha=alpha,color=color)
    if x_label == None:
        x_label =''
    if y_label == None:
        y_label = ''
    if title == None:
        title = ''
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    return fig, ax    

def plot_hist(y, bins = None, hist = True, label = None, 
              kde = True, rug = False, fit = None, hist_kws = {"label":"Histogram"},
              kde_kws = {"label":"KDE fit"}, rug_kws = {"label":"rug"},
              fit_kws = {"label":"fit"}, color ='0.7' , vertical = False,
              norm_hist = True, figsize = [5,5], ax = None):
    """ Plot a univariate distribution of the PaleoData values
            
    This function is based on the seaborn displot function, which is
    itself a combination of the matplotlib hist function with the 
    seaborn kdeplot() and rugplot() functions. It can also fit 
    scipy.stats distributions and plot the estimated PDF over the data.
        
    Args
    ----
    
    y : array
       nx1 numpy array. No missing values allowed 
    bins : int
          Specification of hist bins following matplotlib(hist), 
          or None to use Freedman-Diaconis rule
    hist : bool
          Whether to plot a (normed) histogram 
    label : str
           The label for the axis
    kde : bool
         Whether to plot a gaussian kernel density estimate
    rug : bool
         Whether to draw a rugplot on the support axis
    fit : object
         Random variable object. An object with fit method, returning 
         a tuple that can be passed to a pdf method of positional 
         arguments following a grid of values to evaluate the pdf on.
    hist _kws : Dictionary
    kde_kws : Dictionary
    rug_kws : Dictionary
    fit_kws : Dictionary
             Keyword arguments for underlying plotting functions.
             If modifying the dictionary, make sure the labels "hist",
             "kde","rug","fit" are stall passed.        
    color : str
           matplotlib color. Color to plot everything but the
           fitted curve in.
    vertical : bool
              if True, oberved values are on y-axis.
    norm_hist : bool
               If True (default), the histrogram height shows
               a density rather than a count. This is implied if a KDE or 
               fitted density is plotted
    figsize : list
             the size of the figure
    ax : object
        Return as axis instead of figure (useful to integrate plot into a subplot)     
 
    Returns
    -------
    
    ax : The axis to the figure
    fig :  The figure
"""

    # make sure y is a numpy array
    y = np.array(y)
    
    # Check that these are vectors and not matrices
    # Check that these are vectors and not matrices
    if len(np.shape(y))>2:
        raise TypeError("x and y should be vectors and not matrices") 
     
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    sns.distplot(y,bins=bins, hist=hist, kde=kde, rug=rug,
                  fit=fit, hist_kws = hist_kws,
                  kde_kws = kde_kws,rug_kws = rug_kws,
                  axlabel = label, color = color,
                  vertical = vertical, norm_hist = norm_hist)         
                
       
        
    # Add a label to the PDF axis
    if label == None:
        label = ''
        
    if vertical == True:
        plt.xlabel('PDF')
        plt.ylabel(label)
    else:
        plt.ylabel('PDF')
        plt.xlabel(label)
            
    return ax 

#----------
# utilities
#---------- 
def in_notebook():
    ''' Check if the code is executed in a Jupyter notebook
    '''
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True

def showfig(fig):
    if in_notebook:
        try:
            from IPython.display import display
        except ImportError:
            pass

        plt.close()
        display(fig)

    else:
        plt.show()

def savefig(fig, settings={}, verbose=True):
    ''' Save a figure to a path

    Args
    ----

    fig : figure
        the figure to save
    settings : dict
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified; it can be any existed or non-existed path,
          with or without a suffix; if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}

    '''
    if 'path' not in settings:
        raise ValueError('"path" must be specified in `settings`!')

    savefig_args = {'bbox_inches': 'tight'}
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
    plt.close()

    if verbose:
        print(f'Figure saved at: "{str(path)}"')

def set_style(style='default', font_scale=1):
    ''' Modify the visualization style; inspired by [Seaborn](https://github.com/mwaskom/seaborn)
    '''
    mpl.rcParams.update(mpl.rcParamsDefault)

    font_dict = {
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
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
    elif 'default' in style:
        mpl.rcParams.update(mpl.rcParamsDefault)

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

    # modify font size based on font scale
    font_dict.update({k: v*font_scale for k, v in font_dict.items()})

    for d in [style_dict, font_dict]:
        mpl.rcParams.update(d)