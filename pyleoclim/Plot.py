# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:14:26 2016

@author: deborahkhider

Plot timeseries

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

def plot(x,y,markersize=50,marker='ro',x_label="",y_label="",\
         title ="", figsize =[10,4], ax = None):
    """ Make a 2-D plot
    
    Args:
        x (numpy array): a 1xn numpy array of values for the x-axis
        y (numpy array): a 1xn numpy array for the y-axis
        markersize (int): the size of the marker
        marker (string or list): color and shape of the marker
        x_axis_label (str): the label for the x-axis
        y_axis_label (str): the label for the y-axis
        title (str): the title for the plot
        figsize (list): the size of the figure
        ax: Return as axis instead of figure (useful to integrate plot into a subplot)
            
    Return:
        The figure       
    
    """
    # make sure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Check that these are vectors and not matrices
    if len(np.shape(x)) >2 or len(np.shape(y))>2:
        sys.exit("x and y should be vectors and not matrices") 

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
        
    plt.style.use("ggplot") # set the style
    # do a scatter plot of the original data
    plt.scatter(x,y,s=markersize,facecolor='none',edgecolor=marker[0],
                marker=marker[1],label='original')
    # plot a linear interpolation of the data
    plt.plot(x,y,color=marker[0],linewidth=1,label='interpolated')
    
    #Stylistic issues
    #plt.tight_layout()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc=3,scatterpoints=1,fancybox=True,shadow=True,fontsize=10)
    
    return ax
    
def plot_hist(y, bins = None, hist = True, label = "", \
              kde = True, rug = False, fit = None, hist_kws = {"label":"Histogram"},\
              kde_kws = {"label":"KDE fit"}, rug_kws = {"label":"rug"}, \
              fit_kws = {"label":"fit"}, color ='0.7' , vertical = False, \
              norm_hist = True, figsize = [5,5], ax = None):
    """ Plot a univariate distribution of the PaleoData values
            
    This function is based on the seaborn displot function, which is
    itself a combination of the matplotlib hist function with the 
    seaborn kdeplot() and rugplot() functions. It can also fit 
    scipy.stats distributions and plot the estimated PDF over the data.
        
    Args:
        y (array): nx1 numpy array. No missing values allowed 
        bins (int): Specification of hist bins following matplotlib(hist), 
            or None to use Freedman-Diaconis rule
        hist (bool): Whether to plot a (normed) histogram 
        label (str): The label for the axis
        kde (bool): Whether to plot a gaussian kernel density estimate
        rug (bool): Whether to draw a rugplot on the support axis
        fit: Random variable object. An object with fit method, returning 
            a tuple that can be passed to a pdf method of positional 
            arguments following a grid of values to evaluate the pdf on.
        {hist, kde, rug, fit}_kws: Dictionaries. Keyword arguments for 
            underlying plotting functions. If modifying the dictionary, make
            sure the labels "hist", "kde", "rug" and "fit" are still passed.
        color (str): matplotlib color. Color to plot everything but the
            fitted curve in.
        vertical (bool): if True, oberved values are on y-axis.
        norm_hist (bool): If True (default), the histrogram height shows
            a density rather than a count. This is implied if a KDE or 
            fitted density is plotted
        figsize (list): the size of the figure
        ax: Return as axis instead of figure (useful to integrate plot into a subplot)     
 
    Returns
       fig - The figure
"""

    # make sure y is a numpy array
    y = np.array(y)
    
    # Check that these are vectors and not matrices
    # Check that these are vectors and not matrices
    if len(np.shape(y))>2:
        sys.exit("x and y should be vectors and not matrices") 
     
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    sns.distplot(y,bins=bins, hist=hist, kde=kde, rug=rug,\
                  fit=fit, hist_kws = hist_kws,\
                  kde_kws = kde_kws,rug_kws = rug_kws,\
                  axlabel = label, color = color, \
                  vertical = vertical, norm_hist = norm_hist)         
                
       
        
    # Add a label to the PDF axis
    if vertical == True:
        plt.xlabel('PDF')
        plt.ylabel(label)
    else:
        plt.ylabel('PDF')
        plt.xlabel(label)
            
    return ax 
                
                           
                       

                         
                                     
            
            
            
