# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:42:34 2016

@author: deborahkhider

License agreement - GNU GENERAL PUBLIC LICENSE v3
https://github.com/LinkedEarth/Pyleoclim_util/blob/master/license

"""
#Import all the needed packages
import lipd as lpd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import gridspec
import seaborn as sns
import sys


# Import internal modules to pyleoclim
from pyleoclim import Map
from pyleoclim import LipdUtils
from pyleoclim import SummaryPlots
from pyleoclim import Plot
from pyleoclim import Spectral
from pyleoclim import Stats
from pyleoclim import Timeseries


"""
Open Lipd files and extract timeseries (set them as global variable)
 
"""

def readLipd(usr_path=""):
    """Read Lipd files into a dictionary
    
    This function is based on the function of the same name in the LiPD utilities.
    Sets the dictionary as global variable so that it doesn't have to be provided
    as an argument for every function.
    
    Args:
        usr_path (str): The path to a directory or a single file. (Optional argument)
        
    Returns:
        lipd_dict - a dictionary containing the LiPD library
    
    """
    global lipd_dict
    lipd_dict = lpd.readLipd(usr_path=usr_path)
    return lipd_dict

def extractTs(lipds=None):
    """Extract timeseries dictionary
    
    This function is based on the function of the same name in the LiPD utilities.
    Set the dictionary as a global variable so that it doesn't have to be
    provided as an argument for every function. 
    
    Args:
        lipds (dict): A dictionary of LiPD files obtained through the 
        readLipd function
    
    Returns:
        ts_list - A list of timeseries object
    
    """
    global ts_list
    if not lipds:
        if 'lipd_dict' not in globals():
            readLipd()
            
        ts_list = lpd.extractTs(lipd_dict)
        
    else:
        ts_list = lpd.extractTs(lipds)
    return ts_list
        

"""
Set a few global variables
"""
      
#Set the default palette for plots

plot_default = {'ice/rock': ['#FFD600','h'],
                'coral': ['#FF8B00','o'],
                'documents':['k','p'],
                'glacier ice':['#86CDFA', 'd'],
                'hybrid': ['#00BEFF','*'],
                'lake sediment': ['#4169E0','s'],
                'marine sediment': ['#8A4513', 's'],
                'sclerosponge' : ['r','o'],
                'speleothem' : ['#FF1492','d'],
                'wood' : ['#32CC32','^'],
                'molluskshells' : ['#FFD600','h'],
                'peat' : ['#2F4F4F','*']}

"""
Mapping
"""
def mapAllArchive(lipd_dict = "", markersize = 50, background = 'shadedrelief',\
                  saveFig = False, dir="", format='eps'):
    """Map all the available records loaded into the workspace by archiveType.

    Map of all the records into the workspace by archiveType.
        Uses the default color palette. Enter pyleoclim.plot_default for detail.

    Args:
        lipd_dict (dict): A dictionary of LiPD files. (Optional)
        markersize (int): The size of the markers. Default is 50
        background (str): Plots one of the following images on the map:
            bluemarble, etopo, shadedrelief, or none (filled continents).
            Default is shadedrelief.
        saveFig (bool): Default is to not save the figure
        dir (str): The absolute path of the directory in which to save the
            figure. If not provided, creates a default folder called 'figures'
            in the LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.

    Returns:
        The figure
    """
    
    # Get the dictionary of LiPD files
    if not lipd_dict and 'lipd_dict' not in globals():
        readLipd()
        
    # Initialize the various lists
    lat = []
    lon = []
    archiveType = []

    # Loop ang grab the metadata
    for idx, key in enumerate(lipd_dict):
        d = lipd_dict[key]
        lat.append(d['geo']['geometry']['coordinates'][1])
        lon.append(d['geo']['geometry']['coordinates'][0])
        archiveType.append(LipdUtils.LipdToOntology(d['archiveType']).lower())

    # append the default palette for other category
    plot_default.update({'other':['k','o']})

    # make sure criteria is in the plot_default list
    for idx,val in enumerate(archiveType):
        if val not in plot_default.keys():
            archiveType[idx] = 'other'


    # Make the map
    fig = Map.mapAll(lat,lon,archiveType,lat_0=0,lon_0=0,palette=plot_default,\
                     background = background, markersize = markersize)

    # Save the figure if asked
    if saveFig == True:
        LipdUtils.saveFigure('mapLipds_archive', format, dir)
    else:
        plt.show

    return fig

def mapLipd(timeseries="", countries = True, counties = False, \
        rivers = False, states = False, background = "shadedrelief",\
        scale = 0.5, markersize = 50, marker = "ro", \
        saveFig = False, dir = "", format="eps"):
    """ Create a Map for a single record

    Orthographic projection map of a single record.

    Args:
        timeseries: a LiPD timeseries object. Will prompt for one if not given
        countries (bool): Draws the country borders. Default is on (True).
        counties (bool): Draws the USA counties. Default is off (False).
        rivers (bool): Draws the rivers. Default is off (False).
        states (bool): Draws the American and Australian states borders.
            Default is off (False)
        background (str): Plots one of the following images on the map:
            bluemarble, etopo, shadedrelief, or none (filled continents).
            Default is shadedrelief
        scale (float): useful to downgrade the original image resolution to
            speed up the process. Default is 0.5.
        markersize (int): default is 50
        marker (str): a string (or list) containing the color and shape of the
            marker. Default is by archiveType. Type pyleo.plot_default to see
            the default palette.
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.

    Returns:
        The figure

    """
    # Make sure there are LiPD files to plot
    if not timeseries:
        if not 'ts_list' in globals():
            extractTs()
        timeseries = LipdUtils.getTs(ts_list)    

    # Get latitude/longitude

    lat = timeseries['geo_meanLat']
    lon = timeseries['geo_meanLon']

    # Get the marker color and shape
    archiveType = LipdUtils.LipdToOntology(timeseries['archiveType']).lower()

    # Make sure it's in the palette
    if marker == 'default':
        archiveType = LipdUtils.LipdToOntology(timeseries["archiveType"])
        marker = plot_default[archiveType]
    else:
        marker = 'ro'

    fig = Map.mapOne(lat,lon,marker=marker,markersize=markersize,\
                     countries = countries, counties = counties,rivers = rivers, \
                     states = states, background = background, scale =scale)

    # Save the figure if asked
    if saveFig == True:
        LipdUtils.saveFigure(timeseries['dataSetName']+'_map', format, dir)
    else:
        plt.show

    return fig

"""
Plotting
"""

def plotTs(timeseries = "", x_axis = "", markersize = 50,\
            marker = "default", saveFig = False, dir = "",\
            format="eps"):
    """Plot a single time series.

    Args:
        A timeseries: By default, will prompt the user for one.
        x_axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the
            system choose if only one available or prompt the user.
        markersize (int): default is 50.
        marker (str): a string (or list) containing the color and shape of the
            marker. Default is by archiveType. Type pyleo.plot_default to see
            the default palette.
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.

    Returns:
        The figure.

    """
    if not timeseries:
        if not 'ts_list' in globals():
            extractTs()
        timeseries = LipdUtils.getTs(ts_list) 

    y = np.array(timeseries['paleoData_values'], dtype = 'float64')
    x, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)

    # remove nans
    y_temp = np.copy(y)
    y = y[~np.isnan(y_temp)]
    x = x[~np.isnan(y_temp)]

    # get the markers
    if marker == "default":
        archiveType = LipdUtils.LipdToOntology(timeseries["archiveType"])
        marker = [plot_default[archiveType][0],plot_default[archiveType][1]]

    # Get the labels
    # title
    title = timeseries['dataSetName']
    # x_label
    if label+"Units" in timeseries.keys():
        x_label = label[0].upper()+label[1:]+ " ("+timeseries[label+"Units"]+")"
    else:
        x_label = label[0].upper()+label[1:]
    # ylabel
    if "paleoData_InferredVariableType" in timeseries.keys():
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_InferredVariableType"] + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = timeseries["paleoData_InferredVariableType"]
    elif "paleoData_ProxyObservationType" in timeseries.keys():
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_ProxyObservationType"] + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = timeseries["paleoData_ProxyObservationType"]
    else:
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_variableName"] + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = timeseries["paleoData_variableName"]

    # make the plot
    fig = Plot.plot(x,y,markersize=markersize,marker=marker,x_label=x_label,\
                    y_label=y_label, title=title)

    #Save the figure if asked
    if saveFig == True:
        name = 'plot_timeseries_'+timeseries["dataSetName"]+\
            "_"+y_label
        LipdUtils.saveFigure(name,format,dir)
    else:
        plt.show()

    return fig

def histTs(timeseries = "", bins = None, hist = True, \
             kde = True, rug = False, fit = None, hist_kws = {"label":"Histogram"},\
             kde_kws = {"label":"KDE fit"}, rug_kws = {"label":"Rug"}, \
             fit_kws = {"label":"Fit"}, color = "default", vertical = False, \
             norm_hist = True, saveFig = False, format ="eps",\
             dir = ""):
    """ Plot a univariate distribution of the PaleoData values

    This function is based on the seaborn displot function, which is
    itself a combination of the matplotlib hist function with the
    seaborn kdeplot() and rugplot() functions. It can also fit
    scipy.stats distributions and plot the estimated PDF over the data.

    Args:
        timeseries: A timeseries. By default, will prompt the user for one.
        bins (int): Specification of hist bins following matplotlib(hist),
            or None to use Freedman-Diaconis rule
        hist (bool): Whether to plot a (normed) histogram
        kde (bool): Whether to plot a gaussian kernel density estimate
        rug (bool): Whether to draw a rugplot on the support axis
        fit: Random variable object. An object with fit method, returning
            a tuple that can be passed to a pdf method of positional
            arguments following a grid of values to evaluate the pdf on.
        {hist, kde, rug, fit}_kws: Dictionaries. Keyword arguments for
            underlying plotting functions. If modifying the dictionary, make
            sure the labels "hist", "kde", "rug" and "fit" are still passed.
        color (str): matplotlib color. Color to plot everything but the
            fitted curve in. Default is to use the default paletter for each
            archive type.
        vertical (bool): if True, oberved values are on y-axis.
        norm_hist (bool): If True (default), the histrogram height shows
            a density rather than a count. This is implied if a KDE or
            fitted density is plotted
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.

    Returns
        fig - The figure

    """
    if not timeseries:
        if not 'ts_list' in globals():
            extractTs()
        timeseries = LipdUtils.getTs(ts_list) 

    # Get the values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')

    # Remove NaNs
    index = np.where(~np.isnan(y))[0]
    y = y[index]

    # Get the y_label
    if "paleoData_InferredVariableType" in timeseries.keys():
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_InferredVariableType"] + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = timeseries["paleoData_InferredVariableType"]
    elif "paleoData_ProxyObservationType" in timeseries.keys():
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_ProxyObservationType"] + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = timeseries["paleoData_ProxyObservationType"]
    else:
        if "paleoData_units" in timeseries.keys():
            y_label = timeseries["paleoData_variableName"] + \
                      " (" + timeseries["paleoData_units"]+")"
        else:
            y_label = timeseries["paleoData_variableName"]

    # Grab the color
    if color == "default":
       archiveType = LipdUtils.LipdToOntology(timeseries["archiveType"])
       color = plot_default[archiveType][0]

    # Make this histogram
    fig = Plot.plot_hist(y, bins = bins, hist = hist, \
             kde = kde, rug = rug, fit = fit, hist_kws = hist_kws,\
             kde_kws = kde_kws, rug_kws = rug_kws, \
             fit_kws = fit_kws, color = color, vertical = vertical, \
             norm_hist = norm_hist, label = y_label)

    #Save the figure if asked
    if saveFig == True:
        name = 'plot_timeseries_'+timeseries["dataSetName"]+\
            "_"+y_label
        LipdUtils.saveFigure(name,format,dir)
    else:
        plt.show()

    return fig

"""
SummaryPlots
"""

def summaryTs(timeseries = "", x_axis = "", saveFig = False, dir = "",
               format ="eps"):
    """Basic summary plot

    Plots the following information: the time series, a histogram of
    the PaleoData_values, location map, age-depth profile if both are
    available from the paleodata, metadata about the record.

    Args:
        timeseries: a timeseries object. By default, will prompt for one
        x_axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the
            system choose if only one available or prompt the user.
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.

    Returns:
        The figure

    """

    if not timeseries:
        if not 'ts_list' in globals():
            extractTs()
        timeseries = LipdUtils.getTs(ts_list) 

    # get the necessary metadata
    metadata = SummaryPlots.getMetadata(timeseries)

    # get the information about the timeseries
    x,y,archiveType,x_label,y_label = SummaryPlots.TsData(timeseries,
                                                          x_axis=x_axis)

    # get the age model information if any
    if "age" and "depth" in timeseries.keys() or "year" and "depth" in timeseries.keys():
        depth, age , depth_label, age_label,archiveType = SummaryPlots.agemodelData(timeseries)
        flag = ""
    else:
        flag = "no age or depth info"

    # Make the figure
    fig = plt.figure(figsize=(11,8))
    gs = gridspec.GridSpec(2, 5)
    gs.update(left=0, right =1.1)

    # Plot the timeseries
    ax1 = fig.add_subplot(gs[0,:-3])
    marker = [plot_default[archiveType][0],plot_default[archiveType][1]]
    markersize = 50

    ax1.scatter(x,y,s = markersize, facecolor = 'none', edgecolor = marker[0],
                marker = marker[1], label = 'original')
    ax1.plot(x,y, color = marker[0], linewidth = 1, label = 'interpolated')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    plt.title(timeseries['dataSetName'], fontsize = 14, fontweight = 'bold')
    plt.legend(loc=3,scatterpoints=1,fancybox=True,shadow=True,fontsize=10)

    # Plot the histogram and kernel density estimates
    ax2 = fig.add_subplot(gs[0,2])
    sns.distplot(y, vertical = True, color = marker[0], \
                hist_kws = {"label":"Histogram"},
                kde_kws = {"label":"KDE fit"})

    plt.xlabel('PDF')
    ax2.set_ylim([ymin,ymax])
    ax2.set_yticklabels([])

    # Plot the Map
    lat = timeseries["geo_meanLat"]
    lon = timeseries["geo_meanLon"]

    ax3 = fig.add_subplot(gs[1,0])
    map = Basemap(projection='ortho', lon_0=lon, lat_0=lat)
    map.drawcoastlines()
    map.shadedrelief(scale=0.5)
    map.drawcountries()
    X,Y = map(lon,lat)
    map.scatter(X,Y,
               s = 150,
               color = marker[0],
               marker = marker[1])

    # Plot Age model if any
    if not flag:
        ax4 = fig.add_subplot(gs[1,2])
        plt.style.use("ggplot")
        ax4.plot(depth,age,color = marker[0],linewidth = 1.0)
        plt.xlabel(depth_label)
        plt.ylabel(age_label)
    else:
        print("No age or depth information available, skipping age model plot")

    #Add the metadata
    textstr = "archiveType: " + metadata["archiveType"]+"\n"+"\n"+\
              "Authors: " + metadata["authors"]+"\n"+"\n"+\
              "Year: " + metadata["Year"]+"\n"+"\n"+\
              "DOI: " + metadata["DOI"]+"\n"+"\n"+\
              "Variable: " + metadata["Variable"]+"\n"+"\n"+\
              "units: " + metadata["units"]+"\n"+"\n"+\
              "Climate Interpretation: " +"\n"+\
              "    Climate Variable: " + metadata["Climate_Variable"] +"\n"+\
              "    Detail: " + metadata["Detail"]+"\n"+\
              "    Seasonality: " + metadata["Seasonality"]+"\n"+\
              "    Direction: " + metadata["Interpretation_Direction"]+"\n \n"+\
              "Calibration: \n" + \
              "    Equation: " + metadata["Calibration_equation"] + "\n" +\
              "    Notes: " + metadata["Calibration_notes"]
    plt.figtext(0.7, 0.4, textstr, fontsize = 12)

    #Save the figure if asked
    if saveFig == True:
        name = 'plot_timeseries_'+timeseries["dataSetName"]+\
            "_"+y_label
        LipdUtils.saveFigure(name,format,dir)
    else:
        plt.show()

    return fig

"""
Statistics
"""

def statsTs(timeseries=""):
    """ Calculate simple statistics of a timeseries

    Args:
        timeseries: sytem will prompt for one if not given

    Returns:
        the mean, median, min, max, standard deviation and the
        inter-quartile range (IQR) of a timeseries.

    Examples:
        >>> mean, median, min_, max_, std, IQR = pyleo.statsTs(timeseries)

    """
    if not timeseries:
        if not 'ts_list' in globals():
            extractTs()
        timeseries = LipdUtils.getTs(ts_list) 

    # get the values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')

    mean, median, min_, max_, std, IQR = Stats.simpleStats(y)

    return mean, median, min_, max_, std, IQR

def corrSigTs(timeseries1 = "", timeseries2 = "", x_axis = "", \
                 interp_step = "", start = "", end = "", nsim = 1000, \
                 method = 'isospectral', alpha = 0.5):
    """ Estimates the significance of correlations between non IID timeseries.

        Function written by. F. Zhu.

        Args:
            timeseries1, timeseries2: timeseries object. Default is blank.
            x-axis (str): The representation against which to express the
                paleo-data. Options are "age", "year", and "depth".
                Default is to let the system choose if only one available
                or prompt the user.
            interp_step (float): the step size. By default, will prompt the user.
            start (float): Start time/age/depth. Default is the maximum of
                the minima of the two timeseries
            end (float): End time/age/depth. Default is the minimum of the
                maxima of the two timeseries
            nsim (int): the number of simulations. Default is 1000
            method (str): method use to estimate the correlation and significance.
                Available methods include:
                    - 'ttest': T-test where the degrees of freedom are corrected for
                    the effect of serial correlation \n
                    - 'isopersistant': AR(1) modeling of the two timeseries \n
                    - 'isospectral' (default): phase randomization of original
                    inputs.
                The T-test is parametric test, hence cheap but usually wrong
                except in idyllic circumstances.
                The others are non-parametric, but their computational
                requirements scales with nsim.
            alpha (float): significance level for critical value estimation. Default is 0.05

        Returns:
            r (float) - correlation between the two timeseries \n
            sig (bool) -  Returns True if significant, False otherwise \n
            p (real) - the p-value

    """
    if not timeseries1:
        if not 'ts_list' in globals():
            extractTs()
        timeseries1 = LipdUtils.getTs(ts_list)
        
    if not timeseries2:
        if not 'ts_list' in globals():
            extractTs()
        timeseries2 = LipdUtils.getTs(ts_list)    

    # Get the first time and paleoData values
    y1 = np.array(timeseries1['paleoData_values'], dtype = 'float64')
    x1, label = LipdUtils.checkXaxis(timeseries1, x_axis=x_axis)

    # Get the second one
    y2 = np.array(timeseries2['paleoData_values'], dtype = 'float64')
    x2, label = LipdUtils.checkXaxis(timeseries2, x_axis=label)

    # Remove NaNs
    y1_temp = np.copy(y1)
    y1 = y1[~np.isnan(y1_temp)]
    x1 = x1[~np.isnan(y1_temp)]

    y2_temp = np.copy(y2)
    y2 = y2[~np.isnan(y2_temp)]
    x2 = x2[~np.isnan(y2_temp)]

    #Check that the two timeseries have the same lenght and if not interpolate
    if len(y1) != len(y2):
        print("The two series don't have the same length. Interpolating ...")
        xi, interp_values1, interp_values2 = Timeseries.onCommonAxis(x1,y1,x2,y2,
                                                                     interp_step = interp_step,
                                                                     start =start,
                                                                     end=end)
    elif min(x1) != min(x2) and max(x1) != max(x2):
        print("The two series don't have the same length. Interpolating ...")
        xi, interp_values1, interp_values2 = Timeseries.onCommonAxis(x1,y1,x2,y2,
                                                                     interp_step = interp_step,
                                                                     start =start,
                                                                     end=end)
    else:
        #xi = x1
        interp_values1 = y1
        interp_values2 = y2


    r, sig, p = Stats.corrsig(interp_values1,interp_values2,nsim=nsim,
                                 method=method,alpha=alpha)

    return r, sig, p


"""
Timeseries manipulation
"""

def binTs(timeseries="", x_axis = "", bin_size = "", start = "", end = ""):
    """Bin the paleoData values of the timeseries

    Args:
        timeseries. By default, will prompt the user for one.
        x-axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the
            system  choose if only one available or prompt the user.
        bin_size (float): the size of the bins to be used. By default,
            will prompt for one
        start (float): Start time/age/depth. Default is the minimum
        end (float): End time/age/depth. Default is the maximum

    Returns:
        binned_values- the binned output,\n
        bins-  the bins (centered on the median, i.e. the 100-200 bin is 150),\n
        n-  number of data points in each bin,\n
        error- the standard error on the mean in each bin\n

    """
    if not timeseries:
        if not 'ts_list' in globals():
            extractTs()
        timeseries = LipdUtils.getTs(ts_list)

    # Get the values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')
    x, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)

    #remove nans
    y_temp = np.copy(y)
    y = y[~np.isnan(y_temp)]
    x = x[~np.isnan(y_temp)]
    
    #Bin the timeseries:
    bins, binned_values, n, error = Timeseries.bin(x,y, bin_size = bin_size,\
                                                   start = start, end = end)

    return bins, binned_values, n, error

def interpTs(timeseries="", x_axis = "", interp_step = "", start = "", end = ""):
    """Simple linear interpolation

    Simple linear interpolation of the data using the numpy.interp method

    Args:
        timeseries. Default is blank, will prompt for it
        x-axis (str): The representation against which to plot the paleo-data.
            Options are "age", "year", and "depth". Default is to let the
            system choose if only one available or prompt the user.
        interp_step (float): the step size. By default, will prompt the user.
        start (float): Start year/age/depth. Default is the minimum
        end (float): End year/age/depth. Default is the maximum

    Returns:
        interp_age - the interpolated age/year/depth according to the end/start
        and time step, \n
        interp_values - the interpolated values

    """
    if not timeseries:
        if not 'ts_list' in globals():
            extractTs()
        timeseries = LipdUtils.getTs(ts_list)

    # Get the values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')
    x, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)

    #remove nans
    y_temp = np.copy(y)
    y = y[~np.isnan(y_temp)]
    x = x[~np.isnan(y_temp)]

    #Interpolate the timeseries
    interp_age, interp_values = Timeseries.interp(x,y,interp_step = interp_step,\
                                                  start= start, end=end)

    return interp_age, interp_values

def standardizeTs(timeseries = "", scale = 1, ddof = 0, eps = 1e-3):
    """ Centers and normalizes the paleoData values of a  given time series.

    Constant or nearly constant time series not rescaled.

    Args:
        x (array): vector of (real) numbers as a time series, NaNs allowed
        scale (real): a scale factor used to scale a record to a match a given variance
        axis (int or None): axis along which to operate, if None, compute over the whole array
        ddof (int): degress of freedom correction in the calculation of the standard deviation
        eps (real): a threshold to determine if the standard deviation is too close to zero

    Returns:
        - z (array): the standardized time series (z-score), Z = (X - E[X])/std(X)*scale, NaNs allowed \n
        - mu (real): the mean of the original time series, E[X] \n
        - sig (real): the standard deviation of the original time series, std[X] \n

    References:
        1. Tapio Schneider's MATLAB code: http://www.clidyn.ethz.ch/imputation/standardize.m
        2. The zscore function in SciPy: https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py

    @author: fzhu
    """
    if not timeseries:
        if not 'ts_list' in globals():
            extractTs()
        timeseries = LipdUtils.getTs(ts_list)

    # get the values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')

    # Remove NaNs
    y_temp = np.copy(y)
    y = y[~np.isnan(y_temp)]

    #Standardize
    z, mu, sig = Timeseries.standardize(y,scale=1,axis=None,ddof=0,eps=1e-3)

    return z, mu, sig


#"""
# Spectral Analysis
#"""

def wwzTs(timeseries = "", wwz = False, psd = True, wwz_default = True,
          psd_default = True, wwaplot_default = True, psdplot_default = True,
          fig = True, saveFig = False, dir = "", format = "eps"):
    """Weigthed wavelet Z-transform analysis
    
    Wavelet analysis for unevenly spaced data adapted from Foster et al. (1996)
    
    Args:
        timeseries (dict): A LiPD timeseries object (Optional, will prompt for one.)
        wwz (bool): If True, will perform wavelet analysis
        psd (bool): If True, will inform the power spectral density of the timeseries
        wwz_default: If True, will use the following default parameters:
            
            wwz_default = {'tau':None,'freqs':None,'c':1/(8*np.pi**2),'Neff':3,'nMC':200,
                               'nproc':8,'detrend':'no','method':'Kirchner_f2py'}.
                
            Modify the values for specific keys to change the default behavior.
                
        psd_default: If True, will use the following default parameters:
            
            psd_default = {'tau':None,
                          'freqs': None,
                          'c':1e-3,
                          'nproc':8,
                          'nMC':200,
                          'detrend':'no',
                          'Neff':3,
                          'anti_alias':False,
                          'avgs':2,
                          'method':'Kirchner_f2py'}
            
            Modify the values for specific keys to change the default behavior.
            
        wwaplot_default: If True, will use the following default parameters:
            
            wwaplot_default={'Neff':3,
                                 'AR1_q':AR1_q,
                                 'coi':coi,
                                 'levels':None,
                                 'tick_range':None,
                                 'yticks':None,
                                 'ylim':None,
                                 'xticks':None,
                                 'xlabels':None,
                                 'figsize':[20,8],
                                 'clr_map':'OrRd',
                                 'cbar_drawedges':False,
                                 'cone_alpha':0.5,
                                 'plot_signif':True,
                                 'signif_style':'contour',
                                 'plot_cone':True}
            
            Modify the values for specific keys to change the default behavior.
        psdplot_default: If True, will use the following default parameters:
            
            psdplot_default={'lmstyle':None,
                             'linewidth':None,
                             'xticks':None,
                             'xlim':None,
                             'ylim':None,
                             'figsize':[20,8],
                             'label':'PSD',
                             'plot_ar1':True,
                             'psd_ar1_q95':psd_ar1_q95,
                             'psd_ar1_color':sns.xkcd_rgb["pale red"]}
            
            Modify the values for specific keys to change the default behavior.
            
        fig (bool): If True, plots the figure
        saveFig (bool): default is to not save the figure
        dir (str): the full path of the directory in which to save the figure.
            If not provided, creates a default folder called 'figures' in the
            LiPD working directory (lipd.path).
        format (str): One of the file extensions supported by the active
            backend. Default is "eps". Most backend support png, pdf, ps, eps,
            and svg.
        
    Returns:
        dict_out (dict): A dictionary of outputs. 
            
            For wwz: 
            
            - wwa (array): The weights wavelet amplitude 
        
            - AR1_q (array): AR1 simulations 
        
            - coi (array): cone of influence 
        
            - freqs (array): vector for frequencies 
        
            - tau (array): the evenly-spaced time points, namely the time 
            shift for wavelet analysis. 
        
            - Neffs (array): The matrix of effective number of points in the
            time-scale coordinates.
        
            - coeff (array): The wavelet transform coefficients
        
            For psd: 
            
            - psd (array): power spectral density 
        
            - freqs (array): vector of frequency 
        
            - psd_ar1_q95 (array): the 95% quantile of the psds of AR1 processes 
        
        fig: The figure
         
        References:
            Foster, G. (1996). Wavelets for period analysis of unevenly 
            sampled time series. The Astronomical Journal, 112(4), 1709-1729.
        
        Examples:
            To run both wwz and psd: \n
            
            >>> dict_out, fig = pyleoclim.wwzTs(wwz=True)
            
            Note: This will return a single figure with wwa and psd \n
            
            To change a default behavior:\n
            
            >>> dict_out, fig = pyleoclim.wwzTs(psd_default = {'nMC':1000}) 
           
    """
    
    # Make sure there is something to compute
    if wwz is False and psd is False:
        sys.error("Set 'wwz' and/or 'psd' to True")
    
    # Get a timeseries
    if not timeseries: 
        if not 'ts_list' in globals():
            extractTs()
        timeseries = LipdUtils.getTs(ts_list)
    
     # Raise an error if age or year not in the keys
    if not 'age' in timeseries.keys() and not 'year' in timeseries.keys():
        sys.exit("No time information available")
    elif 'age' in timeseries.keys() and 'year' in timeseries.keys():
        print("Both age and year information are available.")
        x_axis = input("Which one would you like to use? ")
        while x_axis != "year" and x_axis != "age":
            x_axis = input("Only enter year or age: ")
    elif 'age' in timeseries.keys():
        x_axis = 'age'
    elif 'year' in timeseries.keys():
        x_axis = 'year'        
    
    # Get the values
    ys = np.array(timeseries['paleoData_values'], dtype = 'float64') 
    ts, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)
    
    # remove NaNs
    ys_temp = np.copy(ys)
    ys = ys[~np.isnan(ys_temp)]
    ts = ts[~np.isnan(ys_temp)]   
    
    # Perform the calculations
    if psd is True and wwz is False: # PSD only
            
        #Set default
        if psd_default is True:
           psd_default = {'tau':None,
                          'freqs': None,
                          'c':1e-3,
                          'nproc':8,
                          'nMC':200,
                          'detrend':'no',
                          'Neff':3,
                          'anti_alias':False,
                          'avgs':2,
                          'method':'Kirchner_f2py'}
         
        # Perform calculation
        psd, freqs, psd_ar1_q95 = Spectral.wwz_psd(ys, ts, **psd_default)
        
        # Wrap up the output dictionary
        dict_out = {'psd':psd,
               'freqs':freqs,
               'psd_ar1_q95':psd_ar1_q95}
        
        # Plot if asked
        if fig is True:
            #Set the plot default
            if psdplot_default is True:
                psdplot_default={'lmstyle':None,
                                 'linewidth':None,
                                 'xticks':None,
                                 'xlim':None,
                                 'ylim':None,
                                 'figsize':[20,8],
                                 'label':'PSD',
                                 'plot_ar1':True,
                                 'psd_ar1_q95':psd_ar1_q95,
                                 'psd_ar1_color':sns.xkcd_rgb["pale red"],
                                 'ax':None}
                
            fig = Spectral.plot_psd(psd,freqs,**psdplot_default)
            
            if saveFig is True:
                LipdUtils.saveFigure(timeseries['dataSetName']+'_PSDplot',format,dir)
            else:
                plt.show               
            
        else:
            fig = None
             
    elif psd is False and wwz is True: #WWZ only   
        # Set default 
        if wwz_default is True:
            wwz_default = {'tau':None,
                           'freqs':None,
                           'c':1/(8*np.pi**2),
                           'Neff':3,
                           'nMC':200,
                           'nproc':8,
                           'detrend':'no',
                           'method':'Kirchner_f2py'}
        
        #Perform the calculation
        wwa, phase, AR1_q, coi, freqs, tau, Neffs, coeff = Spectral.wwz(ys,ts, **wwz_default)
        
        #Wrap up the output dictionary
        dict_out = {'wwa':wwa,
                    'phase':phase,
                    'AR1_q':AR1_q,
                    'coi':coi,
                    'freqs':freqs,
                    'tau':tau,
                    'Neffs':Neffs,
                    'coeff':coeff}
        
        #PLot if asked
        if fig is True:
            # Set the plot default
            if wwaplot_default is True:
                wwaplot_default={'Neff':3,
                                 'AR1_q':AR1_q,
                                 'coi':coi,
                                 'levels':None,
                                 'tick_range':None,
                                 'yticks':None,
                                 'ylim':None,
                                 'xticks':None,
                                 'xlabels':None,
                                 'figsize':[20,8],
                                 'clr_map':'OrRd',
                                 'cbar_drawedges':False,
                                 'cone_alpha':0.5,
                                 'plot_signif':True,
                                 'signif_style':'contour',
                                 'plot_cone':True,
                                 'ax':None}
            
            fig = Spectral.plot_wwa(wwa, freqs, tau, **wwaplot_default)
            
            if saveFig is True:
                LipdUtils.saveFigure(timeseries['dataSetName']+'_PSDplot',format,dir)
            else:
                plt.show               
            
        else:
            fig = None
    
    elif psd is True and wwz is True: # perform both
    
        # Set the defaults
        
        if psd_default is True:
           psd_default = {'tau':None,
                          'freqs': None,
                          'c':1e-3,
                          'nproc':8,
                          'nMC':200,
                          'detrend':'no',
                          'Neff':3,
                          'anti_alias':False,
                          'avgs':2,
                          'method':'Kirchner_f2py'}
           
        if wwz_default is True:
            wwz_default = {'tau':None,
                           'freqs':None,
                           'c':1/(8*np.pi**2),
                           'Neff':3,
                           'nMC':200,
                           'nproc':8,
                           'detrend':'no',
                           'method':'Kirchner_f2py'}
            
        # Perform the calculations
        psd, freqs, psd_ar1_q95 = Spectral.wwz_psd(ys, ts, **psd_default)
        wwa, phase, AR1_q, coi, freqs, tau, Neffs, coeff = Spectral.wwz(ys,ts, **wwz_default)
          
        #Wrap up the output dictionary
        dict_out = {'wwa':wwa,
                    'phase':phase,
                    'AR1_q':AR1_q,
                    'coi':coi,
                    'freqs':freqs,
                    'tau':tau,
                    'Neffs':Neffs,
                    'coeff':coeff,
                    'psd':psd,
                    'psd_ar1_q95':psd_ar1_q95}
        
        # Make the plot if asked
        if fig is True:
            # Set the plot default
            if wwaplot_default is True:
                wwaplot_default={'Neff':3,
                                 'AR1_q':AR1_q,
                                 'coi':coi,
                                 'levels':None,
                                 'tick_range':None,
                                 'yticks':None,
                                 'ylim':None,
                                 'xticks':None,
                                 'xlabels':None,
                                 'figsize':[20,8],
                                 'clr_map':'OrRd',
                                 'cbar_drawedges':False,
                                 'cone_alpha':0.5,
                                 'plot_signif':True,
                                 'signif_style':'contour',
                                 'plot_cone':True}
                
            if psdplot_default is True:
                psdplot_default={'lmstyle':None,
                             'linewidth':None,
                             'xticks':None,
                             'xlim':None,
                             'ylim':None,
                             'figsize':[20,8],
                             'label':'PSD',
                             'plot_ar1':True,
                             'psd_ar1_q95':psd_ar1_q95,
                             'psd_ar1_color':sns.xkcd_rgb["pale red"]}
            
            if 'figsize' in wwaplot_default.keys():
                figsize = wwaplot_default['figsize']
            elif 'figsize' in psdplot_default.keys():
                figsize = psdplot_default['figsize']
            else:
                figsize = [20.8]
                
                
            fig = plt.figure(figsize = figsize)
            
            ax1 = plt.subplot2grid((1,3),(0,0), colspan =2)
            Spectral.plot_wwa(wwa, freqs, tau, **wwaplot_default,ax=ax1)
            
            ax2 = plt.subplot2grid((1,3),(0,2))
            Spectral.plot_psd(psd,freqs,**psdplot_default,ax=ax2)
            
            if saveFig is True:
                LipdUtils.saveFigure(timeseries['dataSetName']+'_PSDplot',format,dir)
            else:
                plt.show               
            
        else:
            fig = None
                         
    return dict_out, fig   
