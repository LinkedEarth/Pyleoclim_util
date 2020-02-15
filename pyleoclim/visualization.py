# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:00:07 2016

@author: deborahkhider

Mapping functions.


"""
import random
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pathlib

def setProj(projection='Robinson', proj_default = True): 
    """ Set the projection for Cartopy.
    
    Args
    ----
    
    projection : string
        the map projection. Available projections:
        'Robinson' (default), 'PlateCarree', 'AlbertsEqualArea',
        'AzimuthalEquidistant','EquidistantConic','LambertConformal',
        'LambertCylindrical','Mercator','Miller','Mollweide','Orthographic',
        'Sinusoidal','Stereographic','TransverseMercator','UTM',
        'InterruptedGoodeHomolosine','RotatedPole','OSGB','EuroPP',
        'Geostationary','NearsidePerspective','EckertI','EckertII',
        'EckertIII','EckertIV','EckertV','EckertVI','EqualEarth','Gnomonic',
        'LambertAzimuthalEqualArea','NorthPolarStereo','OSNI','SouthPolarStereo'
    proj_default : bool
        If True, uses the standard projection attributes.
        Enter new attributes in a dictionary to change them. Lists of attributes
        can be found in the Cartopy documentation: 
            https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv
    
    Returns
    -------
        proj : the Cartopy projection object
    """
    if proj_default is not True and type(proj_default) is not dict:
        raise TypeError('The default for the projections should either be provided'+
                 ' as a dictionary or set to True')
    
    # Set the projection
    if projection == 'Robinson':
        if proj_default is True:
            proj = ccrs.Robinson() 
        else: proj = ccrs.Robinson(**proj_default) 
    elif projection == 'PlateCarree':
        if proj_default is True:
            proj = ccrs.PlateCarree() 
        else: proj = ccrs.PlateCarree(**proj_default) 
    elif projection == 'AlbersEqualArea':
        if proj_default is True:
            proj = ccrs.AlbersEqualArea() 
        else: proj = ccrs.AlbersEqualArea(**proj_default) 
    elif projection == 'AzimuthalEquidistant':
        if proj_default is True:
            proj = ccrs.AzimuthalEquidistant() 
        else: proj = ccrs.AzimuthalEquidistant(**proj_default)
    elif projection == 'EquidistantConic':
        if proj_default is True:
            proj = ccrs.EquidistantConic() 
        else: proj = ccrs.EquidistantConic(**proj_default)
    elif projection == 'LambertConformal':
        if proj_default is True:
            proj = ccrs.LambertConformal() 
        else: proj = ccrs.LambertConformal(**proj_default)
    elif projection == 'LambertCylindrical':
        if proj_default is True:
            proj = ccrs.LambertCylindrical() 
        else: proj = ccrs.LambertCylindrical(**proj_default)
    elif projection == 'Mercator':
        if proj_default is True:
            proj = ccrs.Mercator() 
        else: proj = ccrs.Mercator(**proj_default)
    elif projection == 'Miller':
        if proj_default is True:
            proj = ccrs.Miller() 
        else: proj = ccrs.Miller(**proj_default)
    elif projection == 'Mollweide':
        if proj_default is True:
            proj = ccrs.Mollweide() 
        else: proj = ccrs.Mollweide(**proj_default)
    elif projection == 'Orthographic':
        if proj_default is True:
            proj = ccrs.Orthographic() 
        else: proj = ccrs.Orthographic(**proj_default)
    elif projection == 'Sinusoidal':
        if proj_default is True:
            proj = ccrs.Sinusoidal() 
        else: proj = ccrs.Sinusoidal(**proj_default)
    elif projection == 'Stereographic':
        if proj_default is True:
            proj = ccrs.Stereographic() 
        else: proj = ccrs.Stereographic(**proj_default)
    elif projection == 'TransverseMercator':
        if proj_default is True:
            proj = ccrs.TransverseMercator() 
        else: proj = ccrs.TransverseMercator(**proj_default)
    elif projection == 'TransverseMercator':
        if proj_default is True:
            proj = ccrs.TransverseMercator() 
        else: proj = ccrs.TransverseMercator(**proj_default)
    elif projection == 'UTM':
        if proj_default is True:
            proj = ccrs.UTM() 
        else: proj = ccrs.UTM(**proj_default)
    elif projection == 'UTM':
        if proj_default is True:
            proj = ccrs.UTM() 
        else: proj = ccrs.UTM(**proj_default)
    elif projection == 'InterruptedGoodeHomolosine':
        if proj_default is True:
            proj = ccrs.InterruptedGoodeHomolosine() 
        else: proj = ccrs.InterruptedGoodeHomolosine(**proj_default)
    elif projection == 'RotatedPole':
        if proj_default is True:
            proj = ccrs.RotatedPole() 
        else: proj = ccrs.RotatedPole(**proj_default)
    elif projection == 'OSGB':
        if proj_default is True:
            proj = ccrs.OSGB() 
        else: proj = ccrs.OSGB(**proj_default)
    elif projection == 'EuroPP':
        if proj_default is True:
            proj = ccrs.EuroPP() 
        else: proj = ccrs.EuroPP(**proj_default)
    elif projection == 'Geostationary':
        if proj_default is True:
            proj = ccrs.Geostationary() 
        else: proj = ccrs.Geostationary(**proj_default)
    elif projection == 'NearsidePerspective':
        if proj_default is True:
            proj = ccrs.NearsidePerspective() 
        else: proj = ccrs.NearsidePerspective(**proj_default)
    elif projection == 'EckertI':
        if proj_default is True:
            proj = ccrs.EckertI() 
        else: proj = ccrs.EckertI(**proj_default)
    elif projection == 'EckertII':
        if proj_default is True:
            proj = ccrs.EckertII() 
        else: proj = ccrs.EckertII(**proj_default)
    elif projection == 'EckertIII':
        if proj_default is True:
            proj = ccrs.EckertIII() 
        else: proj = ccrs.EckertIII(**proj_default)
    elif projection == 'EckertIV':
        if proj_default is True:
            proj = ccrs.EckertIV() 
        else: proj = ccrs.EckertIV(**proj_default)
    elif projection == 'EckertV':
        if proj_default is True:
            proj = ccrs.EckertV() 
        else: proj = ccrs.EckertV(**proj_default)
    elif projection == 'EckertVI':
        if proj_default is True:
            proj = ccrs.EckertVI() 
        else: proj = ccrs.EckertVI(**proj_default)
    elif projection == 'EqualEarth':
        if proj_default is True:
            proj = ccrs.EqualEarth() 
        else: proj = ccrs.EqualEarth(**proj_default)
    elif projection == 'Gnomonic':
        if proj_default is True:
            proj = ccrs.Gnomonic() 
        else: proj = ccrs.Gnomonic(**proj_default)
    elif projection == 'LambertAzimuthalEqualArea':
        if proj_default is True:
            proj = ccrs.LambertAzimuthalEqualArea() 
        else: proj = ccrs.LambertAzimuthalEqualArea(**proj_default)
    elif projection == 'NorthPolarStereo':
        if proj_default is True:
            proj = ccrs.NorthPolarStereo() 
        else: proj = ccrs.NorthPolarStereo(**proj_default)
    elif projection == 'OSNI':
        if proj_default is True:
            proj = ccrs.OSNI() 
        else: proj = ccrs.OSNI(**proj_default)
    elif projection == 'OSNI':
        if proj_default is True:
            proj = ccrs.SouthPolarStereo() 
        else: proj = ccrs.SouthPolarStereo(**proj_default)
    else:
        raise ValueError('Invalid projection type')
        
    return proj

def mapAll(lat, lon, criteria, projection = 'Robinson', proj_default = True,\
           background = True,borders = False, rivers = False, lakes = False,\
           figsize = [10,4], ax = None, palette=None, markersize = 50):
    """ Map the location of all lat/lon according to some criteria
    
    Map the location of all lat/lon according to some criteria. The choice of 
    plotting color/marker is passed through palette according to unique 
    criteria (e.g., record name, archive type, proxy observation type).
    
    Args
    ----
    
    lat : list
        a list of latitude.
    lon : list
        a list of longitude.
    criteria : list
        a list of criteria for plotting purposes. For instance,
        a map by the types of archive present in the dataset or proxy
        observations.
    projection : string
        the map projection. Available projections:
        'Robinson' (default), 'PlateCarree', 'AlbertsEqualArea',
        'AzimuthalEquidistant','EquidistantConic','LambertConformal',
        'LambertCylindrical','Mercator','Miller','Mollweide','Orthographic',
        'Sinusoidal','Stereographic','TransverseMercator','UTM',
        'InterruptedGoodeHomolosine','RotatedPole','OSGB','EuroPP',
        'Geostationary','NearsidePerspective','EckertI','EckertII',
        'EckertIII','EckertIV','EckertV','EckertVI','EqualEarth','Gnomonic',
        'LambertAzimuthalEqualArea','NorthPolarStereo','OSNI','SouthPolarStereo'
    proj_default : bool
        If True, uses the standard projection attributes.
        Enter new attributes in a dictionary to change them. Lists of attributes
        can be found in the Cartopy documentation: 
            https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv
    background : bool
        If True, uses a shaded relief background (only one 
        available in Cartopy)
    borders : bool
        Draws the countries border. Defaults is off (False). 
    rivers : bool
        Draws major rivers. Default is off (False).
    lakes : bool
        Draws major lakes. 
        Default is off (False).
    palette : dict
        A dictionary of plotting color/marker by criteria. The
        keys should correspond to ***unique*** criteria with a list of 
        associated values. The list should be in the format 
        ['color', 'marker'].
    markersize : int
        The size of the marker.
    figsize : list
        the size for the figure
    ax: axis,optional
        Return as axis instead of figure (useful to integrate plot into a subplot) 
        
    Returns
    -------
    
    ax: The figure, or axis if ax specified      
    """
    #Check that the lists have the same length and convert to numpy arrays
    if len(lat)!=len(lon) or len(lat)!=len(criteria) or len(lon)!=len(criteria):
        raise ValueError("Latitude, Longitude, and criteria list must be the same" +\
                 "length")
        
    # Check that the default is set to True or in dictionary format
    if proj_default is not True and type(proj_default) is not dict:
        raise TypeError('The default for the projections should either be provided'+
                 ' as a dictionary or set to True')
        
    # If palette is not given, then make a random one.
    if not palette:
        marker_list = ['o','v','^','<','>','8','s','p','*','h','D']
        color_list = ['#FFD600','#FF8B00','k','#86CDFA','#00BEFF','#4169E0',\
                 '#8A4513','r','#FF1492','#32CC32','#FFD600','#2F4F4F']
        # select at random for unique entries in criteria
        marker = [random.choice(marker_list) for _ in range(len(set(criteria)))]
        color = [random.choice(color_list) for _ in range(len(set(criteria)))]
        crit_unique = [crit for crit in set(criteria)]
        #initialize the palette
        palette = {crit_unique[0]:[color[0],marker[0]]}
        for i in range(len(crit_unique)):
            d1 = {crit_unique[i]:[color[i],marker[i]]}
            palette.update(d1)
    
    # get the projection:
    proj = setProj(projection=projection, proj_default=proj_default)        
    # Make the figure        
    if not ax:
        fig, ax = plt.subplots(figsize=figsize,subplot_kw=dict(projection=proj))     
    # draw the coastlines    
    ax.coastlines()
    
    # Background
    if background is True:
        ax.stock_img()
            
    #Other extra information
    if borders is True:
        ax.add_feature(cfeature.BORDERS)
    if lakes is True:
        ax.add_feature(cfeature.LAKES)
    if rivers is True:
        ax.add_feature(cfeature.RIVERS)
    
    # Get the indexes by criteria
    for crit in set(criteria):
        # Grab the indices with same criteria
        index = [i for i,x in enumerate(criteria) if x == crit]
        ax.scatter(np.array(lon)[index],np.array(lat)[index],
                    s= markersize,
                    facecolor = palette[crit][0],
                    marker = palette[crit][1],
                    zorder = 10,
                    label = crit,
                    transform=ccrs.PlateCarree())
    plt.legend(loc = 'center', bbox_to_anchor=(1.1,0.5),scatterpoints = 1,
               frameon = False, fontsize = 8, markerscale = 0.7)
    
    return ax    
        
def mapOne(lat, lon, projection = 'Orthographic', proj_default = True, label = None,\
           background = True,borders = False, rivers = False, lakes = False,\
           markersize = 50, marker = "ro", figsize = [4,4], \
           ax = None):
    """ Map one location on the globe
    
    Args
    ----
    
    lat : float
        a float number representing latitude
    lon : float
        a float number representing longitude
    projection : string
        the map projection. Available projections:
        'Robinson', 'PlateCarree', 'AlbertsEqualArea',
        'AzimuthalEquidistant','EquidistantConic','LambertConformal',
        'LambertCylindrical','Mercator','Miller','Mollweide','Orthographic' (Default),
        'Sinusoidal','Stereographic','TransverseMercator','UTM',
        'InterruptedGoodeHomolosine','RotatedPole','OSGB','EuroPP',
        'Geostationary','NearsidePerspective','EckertI','EckertII',
        'EckertIII','EckertIV','EckertV','EckertVI','EqualEarth','Gnomonic',
        'LambertAzimuthalEqualArea','NorthPolarStereo','OSNI','SouthPolarStereo'
    proj_default : bool
        If True, uses the standard projection attributes, including centering.
        Enter new attributes in a dictionary to change them. Lists of attributes
        can be found in the Cartopy documentation: 
            https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv
    background : bool
        If True, uses a shaded relief background (only one 
        available in Cartopy)
    label : string
        label for the point. Default is None. 
    borders : bool
        Draws the countries border. Defaults is off (False). 
    rivers : bool
        Draws major rivers. Default is off (False).
    lakes : bool
        Draws major lakes. 
        Default is off (False).
    markersize : int
        The size of the marker.
    marker : string or list
        color and type of marker. 
    figsize : list
        the size for the figure
    ax : optional, axis
        Return as axis instead of figure (useful to integrate plot into a subplot) 
    
    Returns
    -------
    
    ax: The figure, or axis if ax specified
    
    """
    # get the projection:
    if proj_default is True:
        proj_default = {'central_longitude':lon}
    proj = setProj(projection=projection, proj_default=proj_default)        
    # Make the figure        
    if not ax:
        fig, ax = plt.subplots(figsize=figsize,subplot_kw=dict(projection=proj))     
    # draw the coastlines    
    ax.coastlines()
    
    # Background
    if background is True:
        ax.stock_img()  
    
    #Other extra information
    if borders is True:
        ax.add_feature(cfeature.BORDERS)
    if lakes is True:
        ax.add_feature(cfeature.LAKES)
    if rivers is True:
        ax.add_feature(cfeature.RIVERS)
    
    # Draw the point
    ax.scatter(np.array(lon),np.array(lat),
               s= markersize,
               facecolor = marker[0],
               marker = marker[1],
               zorder = 10,
               transform=ccrs.PlateCarree())
    
    # Add a label if necessary
    if label is not None:
       assert type(label) is str, 'Label should be of type string'
       ax.annotate(label,(np.array(lon),np.array(lat)),fontweight='bold')
        
    return ax
def plot(x,y,markersize=50,marker='ro',x_label="",y_label="",\
         title ="", figsize =[10,4], ax = None):
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
        Return as axis instead of figure (useful to integrate plot into a subplot)
            
    Returns
    -------
    
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
    
def plotEns(ageEns, y, ens = None, color = 'r', alpha = 0.005, x_label = "",\
            y_label = "", title = "", figsize = [10,4], ax = None):
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
    The figure
    
    TODO
    ----
    Enable paleoEnsemble       
    
    """

    #Make sure that the ensemble and paleo values are numpy arrays
    ageEns = np.array(ageEns)
    y = np.array(y)
    
    # Make sure that the length of y is the same as the number of rows in ensemble array
    if len(y) != np.shape(ageEns)[0]:
        sys.exit("The length of the paleoData is different than number of rows in ensemble table!")

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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

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
    
    fig :  The figure
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
                
def getMetadata(timeseries):
    
    """ Get the necessary metadata to be printed out automatically
    
    Args
    ----
        
    timeseries : object
                a specific timeseries object. 
        
    Returns
    -------
    
    dict_out : dict
              A dictionary containing the following metadata:
                archiveType
                Authors (if more than 2, replace by et al)
                PublicationYear 
                Publication DOI 
                Variable Name
                Units
                Climate Interpretation
                Calibration Equation 
                Calibration References
                Calibration Notes
        
    """
    
    # Get all the necessary information
    # Top level information
    if "archiveType" in timeseries.keys():
        archiveType = timeseries["archiveType"]
    else:
        archiveType = "NA"
        
    if "pub1_author" in timeseries.keys():
        authors = timeseries["pub1_author"]
    else:
        authors = "NA"
    
    #Truncate if more than two authors
    idx = [pos for pos, char in enumerate(authors) if char == ";"]
    if  len(idx)>2:
        authors = authors[0:idx[1]+1] + "et al."
    
    if "pub1_pubYear" in timeseries.keys():
        Year = str(timeseries["pub1_pubYear"])
    else:
        Year = "NA"
    
    if "pub1_DOI" in timeseries.keys():
        DOI = timeseries["pub1_DOI"]  
    else:
        DOI = "NA"
    
    if "paleoData_InferredVariableType" in timeseries.keys():
        if type(timeseries["paleoData_InferredVariableType"]) is list:
            Variable = timeseries["paleoData_InferredVariableType"][0]
        else:
            Variable = timeseries["paleoData_InferredVariableType"]
    elif "paleoData_ProxyObservationType" in timeseries.keys():
        if type(timeseries["paleoData_ProxyObservationType"]) is list:
            Variable = timeseries["paleoData_ProxyObservationType"][0]
        else:
            Variable = timeseries["paleoData_ProxyObservationType"]
    else:
        Variable = timeseries["paleoData_variableName"]
    
    if "paleoData_units" in timeseries.keys():
        units = timeseries["paleoData_units"]
    else:
        units = "NA"
    
    #Climate interpretation information
    if "paleoData_interpretation" in timeseries.keys():
        interpretation = timeseries["paleoData_interpretation"][0]
        if "name" in interpretation.keys():
            ClimateVar = interpretation["name"]
        elif "variable" in interpretation.keys():
            ClimateVar = interpretation["variable"]
        else:
            ClimateVar = "NA"
        if "detail" in interpretation.keys(): 
            Detail = interpretation["detail"]
        elif "variableDetail" in interpretation.keys():
            Detail = interpretation['variableDetail']
        else:
            Detail = "NA"
        if "scope" in interpretation.keys():
            Scope = interpretation['scope']
        else:
            Scope = "NA"
        if "seasonality" in interpretation.keys():    
            Seasonality = interpretation["seasonality"]
        else:
            Seasonality = "NA"
        if "interpdirection" in interpretation.keys():    
            Direction = interpretation["interpdirection"]
        else:
            Direction = "NA"
    else:
        ClimateVar = "NA"
        Detail = "NA"
        Scope = "NA"
        Seasonality = "NA"
        Direction = "NA"
        
    # Calibration information
    if "paleoData_calibration" in timeseries.keys():
        calibration = timeseries['paleoData_calibration'][0]
        if "equation" in calibration.keys():
            Calibration_equation = calibration["equation"]
        else:
            Calibration_equation = "NA"
        if  "calibrationReferences" in calibration.keys():
            ref = calibration["calibrationReferences"]
            if "author" in ref.keys():
                ref_author = ref["author"][0] # get the first author
            else:
                ref_author = "NA"
            if  "publicationYear" in ref.keys():
                ref_year = str(ref["publicationYear"])
            else: ref_year="NA"
            Calibration_notes = ref_author +"."+ref_year
        elif "notes" in calibration.keys():
            Calibration_notes = calibration["notes"]
        else: Calibration_notes = "NA"    
    else:
        Calibration_equation = "NA"
        Calibration_notes = "NA"
    
    #Truncate the notes if too long
    charlim = 30;
    if len(Calibration_notes)>charlim:
        Calibration_notes = Calibration_notes[0:charlim] + " ..."
        
    dict_out = {"archiveType" : archiveType,
                "authors" : authors,
                "Year": Year,
                "DOI": DOI,
                "Variable": Variable,
                "units": units,
                "Climate_Variable" : ClimateVar,
                "Detail" : Detail,
                "Scope":Scope,
                "Seasonality" : Seasonality,
                "Interpretation_Direction" : Direction,
                "Calibration_equation" : Calibration_equation,
                "Calibration_notes" : Calibration_notes}
    
    return dict_out    

def TsData(timeseries, x_axis=""):
    """ Get the PaleoData with age/depth information
        
    Get the necessary information for the TS plots/necessary to allow for
    axes specification
    
    Args
    ----
    
    timeseries : object
                a single timeseries object. 
                By default, will prompt the user
    x-axis : str
            The representation against which to plot the 
            paleo-data. Options are "age", "year", and "depth". 
            Default is to let the system choose if only one available 
            or prompt the user.
    
    Returns
    -------
    
        x : list
           the x-values
        y : list
           the y-values 
        archiveType : str
                     the archiveType (for plot settings) \n
        x_label : str
                 the label for the x-axis \n
        y_label : str
                 the label for the y-axis \n
        label : str
               the results of the x-axis query. Either depth, year, or age
        
    """
    # Grab the x and y values
    y = np.array(timeseries['paleoData_values'], dtype = 'float64')   
    x, label = LipdUtils.checkXaxis(timeseries, x_axis=x_axis)

    # Remove NaNs
    y_temp = np.copy(y)
    y = y[~np.isnan(y_temp)]
    x = x[~np.isnan(y_temp)]

    # Grab the archiveType
    archiveType = LipdUtils.LipdToOntology(timeseries["archiveType"])

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

    return x,y,archiveType,x_label,y_label    

def agemodelData(timeseries):
    """Get the necessary information for the agemodel plot

    Args
    ----
    
    timeseries : object
                a single timeseries object. By default, will
                prompt the user

    Returns
    -------
    
        depth : float
               the depth values
        age : float
             the age values
        x_label : str
                 the label for the x-axis 
        y_label : str
                 the label for the y-axis \n
        archiveType : str
                     the archiveType (for default plot settings)

    """
    if not "age" in timeseries.keys() and not "year" in timeseries.keys():
        raise KeyError("No time information")
    elif not "depth" in timeseries.keys():
        raise KeyError("No depth information")
    else:
        if "age" in timeseries.keys() and "year" in timeseries.keys():
            print("Do you want to use age or year?")
            choice = int(input("Enter 0 for age and 1 for year: "))
            if choice == 0:
                age = timeseries['age']
                if "ageUnits" in timeseries.keys():
                    age_label = "Calendar Age (" +\
                                    timeseries["ageUnits"] +")"
                else:
                    age_label = "Calendar Age"
            elif choice == 1:
                age = timeseries['year']
                if "yearUnits" in timeseries.keys():
                    age_label = "Year (" +\
                                    timeseries["yearUnits"] +")"
                else:
                    age_label = "Year"
            else:
                raise ValueError("Enter 0 or 1")

        if "age" in timeseries.keys():
            age = timeseries['age']
            if "ageUnits" in timeseries.keys():
                age_label = "Calendar Age (" +\
                        timeseries["ageUnits"] +")"
            else:
                age_label = "Calendar Age"

        if "year" in timeseries.keys():
            age = timeseries['year']
            if "yearUnits" in timeseries.keys():
                age_label = "Year (" +\
                        timeseries["ageUnits"] +")"
            else:
                age_label = "Year"

        depth = timeseries['depth']
        if "depthUnits" in timeseries.keys():
            depth_label = "Depth (" + timeseries["depthUnits"] + ")"
        else:
            depth_label = "Depth"

    # Get the archiveType and make sure it aligns with the ontology
    archiveType = LipdUtils.LipdToOntology(timeseries["archiveType"])

    return depth, age, depth_label, age_label, archiveType


# utilities
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

    savefig_args = {'format': 'pdf', 'bbox_inches': 'tight'}
    savefig_args.update(settings)

    path = pathlib.Path(savefig_args['path'])
    savefig_args.pop('path')

    dirpath = path.parent
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f'Directory created at: "{dirpath}"')

    if path.suffix not in ['.eps', '.pdf', '.png', '.ps']:
        path_str = str(path)
        fmt = savefig_args['format']
        path = pathlib.Path(f'{path_str}.{fmt}')

    fig.savefig(str(path), **savefig_args)
    plt.close()

    if verbose:
        print(f'Figure saved at: "{str(path)}"')
