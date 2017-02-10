#Pyleoclim Documentation v0.1.3

**Index**
* [User Guide](#user_guide)
  * [Mapping](#mapping)
  * [Plotting](#plotting)
    * [Summary Plots](#summaryplt)
  * [Basic Functionalities](#basic)
* [Programmer's corner](#program_corner)
  * [LiPDutils](#lipdutils)
  * [Map](#map)
  * [TSPlot](#TSplot)
  * [Basic](#Basic)
  * [SummaryPlots](#sumplt)

## <a name = "user_guide"> User guide </a>
To help you get started with Pyleoclim, refer to the Jupyter Notebook on our <a href="https://github.com/LinkedEarth/Pyleoclim_util"> GitHub repository </a>.

### <a id="mapping">Mapping</a>

#### `MapAll()`
This function maps all the LiPD records in the working directory according to archiveType. It uses a default palette color accessible by typing `pyleo.plot_default`

Synthax: `fig = pyleoclim.MapAll(markersize = 50, saveFig = False, dir = "", format = "eps")`

Optional arguments:
* `markersize`: default is 50
* `saveFig`: if `True`, saves the map into the `dir` folder in the current working directory.
* `dir`: a folder in the current directory, where the various figures can be saved. If left blank, a folder named `figures` will be automatically created
* `format`: One of the file extensions supported by the active backend. Default is `eps`. Most backends support `png`, `pdf`, `ps`, and `svg`.

Returns: `fig`

#### `MapLiPD()`
This function maps one particular LiPD record stored in the working directory.

Synthax: `fig = pyleoclim.MapLiPD(name="", countries = True, counties = False, \
        rivers = False, states = False, background = "shadedrelief",\
        scale = 0.5, markersize = 50, marker = "default", \
        saveFig = False, dir = "", format="eps")`

Optional arguments:
* `name`: the name of the LiPD file. **WITH THE .LPD EXTENSION!**. If not provided, will prompt the user for one.
* `countries`: Draws the country borders. Default is on (True).
* `counties`: Draws the USA counties. Default is off (False).
* `states`: Draws the American and Australian states borders. Default is off (False)
* `background`: Plots one of the following images on the map: bluemarble, etopo, shadedrelief, or none (filled continents). Default is shadedrelief
* `scale`: useful to downgrade the original image resolution to speed up the process. Default is 0.5.
* `markersize`: default is 100
* `marker`: a string (or list) containing the color and shape of the marker. Default is by archiveType. Type pyleo.plot_default to see the default palette.
* `saveFig`: default is to not save the figure
* `dir`: the full path of the directory in which to save the figure. If not provided, creates a default folder called 'figures' in the LiPD working directory (lipd.path).  
* `format`: One of the file extensions supported by the active backend. Default is "eps". Most backend support png, pdf, ps, eps, and svg.

Returns: `fig`

### <a id="plotting">Plotting</a>

#### `plotTS()`

Plot a time series.

Synthax:
`fig = pyleoclim.PlotTS(timeseries = "", x_axis = "", markersize = 50, marker = "default", saveFig = False, dir = "figures", format="eps")`

Optional arguments:
* `timeseries`: A timeseries object as defined in LiPD. If left blank, you'll be prompted to choose the record from a list.
* `x-axis`: The representation against which to plot the paleo-data. Options are "age","year", and "depth". Default is to let the system choose if only one available or prompt the user.
* `markersize`: Default is 50
* `marker`: Shape and color. Default uses the Pyleoclim color palette. If you wish to change the default marker, enter the color and shape (**in this order**). For instance to use a red square, use `'rs'`.
*  `saveFig`: if `True`, saves the map into the `dir` folder in the current working directory.
* `dir`: a folder in the current directory, where the various figures can be saved. If left blank, a folder named `figures` will be automatically created
* `format`: One of the file extensions supported by the active backend. Default is `eps`. Most backends support `png`, `pdf`, `ps`, and `svg`.

Returns: `fig`

#### <a id="summaryplt">Summary Plots</a>

Summary Plots are special plots in Pyleoclim that allow to get basic information about a record.

##### `BasicSummary()`

This functions plots:
1. The time series
2. The location map
3. Age/depth profile if both are available in the paleoDataTable
4. Metadata information

Synthax: `fig  = pyleoclim.BasicSummary(timeseries = "", x_axis="", saveFig = False, format = "eps", dir = "figures")`

* `timeseries`: A timeseries object as defined in LiPD. If left blank, you'll be prompted to choose the record from a list.
* `x-axis`: The representation against which to plot the paleo-data. Options are "age","year", and "depth". Default is to let the system choose if only one available or prompt the user.  
* `saveFig`: if `True`, saves the map into the `dir` folder in the current working directory.
* `dir`: a folder in the current directory, where the various figures can be saved. If left blank, a folder named `figures` will be automatically created
* `format`: One of the file extensions supported by the active backend. Default is `eps`. Most backends support `png`, `pdf`, `ps`, and `svg`.

Returns: `fig`

### <a id="basic">Basic functionalities</a>
#### `TSstats()`

Returns the mean and standard deviation of the timeseries

synthax: `mean, std = pyleoclim.TSstats(timeseries="")`

Optional arguments:
* `timeseries`: If blank, will prompt for one.

#### `TSbin()`

Bins the values of the timeseries

synthax: `bins, binned_data, n, error = pyleoclim.TSbin(timeseries="", x_axis = "", bin_size = "", start = "", end = "")`

Optional arguments:
* `Timeseries`. Default is blank, will prompt for it
* `x-axis`: the time or depth index to use for binning. Valid keys inlude: depth, age, and year.
* `bin_size`: the size of the bins to be used. If not given, the function will prompt the user
* `start`: where the bins should start. Default is the minimum
* `end`: where the bins should end. Default is the maximum

Returns:
* `bins`: the bins centered on the median (i.e., the 100-200 yr bin is 150 yr)
* `binned_data`: the mean of the paleoData values in the particular bin
* `n`: the number of values used to obtain the average
* `error`: the standard error on the mean

#### `TSinterp()`

Bins the values of the timeseries

synthax: `interp_age, interp_values = pyleoclim.TSbin(timeseries="", x_axis = "", interp_step = "", start = "", end = "")`

Optional arguments:
* `Timeseries`. Default is blank, will prompt for it
* `x-axis`: the time or depth index to use for binning. Valid keys inlude: depth, age, and year.
* `bin_size`: the step size. If not given, the function will prompt the user
* `start`: where the bins should start. Default is the minimum
* `end`: where the bins should end. Default is the maximum

Returns:
* `interp_age`: the interpolated age according to the end/start and time step
* `interp_values`: the interpolated values

## <a id = "program_corner"> Programmer's corner </a>
Pyleoclim is meant to be developed by the Paleoclimate community. If you'd like to contribute codes, the content below explains the structure of the Pyleoclim package.

The easiest way to contribute is to create a class, regrouping all the necessary methods.

### `__init__`

When loading the Pyleoclim package, the LiPD utilities and all dependency are initialized. The package also loads the LiPD files into the workspace and automatically extracts the timeseries objects under the variable `time_series`.  
Finally, the default color palette `plot_default` is also initialized Both of these variables can be passed into the __init__ method of the various classes.

The `__init__` module also contains the top-level methods described in the [user's guide](#user_guide).

The package contains five modules, whose functions are described below.

### <a id = "lipdutils">`LiPDutils` </a>

Allows for basic manipulations of LiPD files (with no scientific purpose). The functions can be called from any other modules directly, therefore take care not to give your functions the same name.

* New directories and saving
  * `createdir(path, foldername)`: Creates a folder in the LiPD working directory (from lipd.path()).
  * `saveFigure(name, format="eps",dir="")`: Save the figure according to the name, format and directory.


* LiPD files:
  * `enumerateLipds()`: print the name of the LiPD file in the directory
  * `prompforLipd()`: Prompt the user to select a LiPD file from a list. Use this function in conjunction with `enumerateLipds()`


* Variable-level functions:
    * `promptforVariable()`: Ask the user to select the variable they are interested in. Use this function with `readHeaders()` or `getTSO()`
    * `valuesloc(dataframe, missing_value = "NaN", var_idx = 1)`: Look for the indixes where there are **no** missing values for the variable. Requires a panda Dataframe
    * `TSOxaxis(time_series)`: Look for "depth", or "year", or "age" in a timeseries object for time representation and prompt the user if there are several possibilities. Requires to single out **one** timeseries for use. See `getTSO()` for details.


* Timeseries objects
    * `enumerateTSO(time_series)`: enumerate the available timeseries objects from the `time_series` variable obtained upon intilization of the Pyleoclim package.
    * `getTSO(time_series)`: get a specific timeseries object
    * `TStoDF(time_series, x_axis="")`: Creates a dataframe with the time representation in one column and paleodata in the other.


* LinkedEarth ontology
    * `LiPDtoOntology(archiveType)`: harmonizes the archiveType from the Pages2k dataset with the <a href='http://linked.earth/ontology/archive/index-en.html'> LinkedEarth Ontology</a>.

### <a id="map">`Map`</a>

Contains class:
* `Map`, with methods:
  * `__init__(self, plot_default)`: Grabs the coordinates of all the LiPD files loaded in the workspace and passes the default color palette.
  * `map_all(self, markersize = int(50), saveFig = True, dir="", format='eps')`: map all LiPDs.
  * `map_one(self, name="",gridlines = False, borders = True, topo = True, markersize = int(100), marker = "default", saveFig = True, dir = "", format="eps")`: map one LiPD.

### <a id = "TSplot"> `TSPlot`</a>

Contains class:
* `Plot`, with methods:
  * `__init__(self, plot_default,time_series)`: Passes the default color palette and available timeseries objects.
  * `plotoneTSO(self, new_timeseries = "", x_axis = "", markersize = 50, marker = "default", saveFig = True, dir = "figures", format="eps")`: Plot one time_series
  * `agemodelplot(self, new_timeseries = "", markersize = 50, marker = "default", saveFig = True, dir = "figures", format="eps" )`: Plots the basic age-depth relationship inferred from the paleoDataTable

### <a id = "Basic"> `Basic`</a>

  Manipulation of LiPD files (timeseries objects) for scientific purposes

  Contains class:
  * `Basic`, with methods:
    * `__init__(self, time_series)`: Passes the `time_series` variable
    * `getValues(new_timeseries)`: staticmethod. Get the values for a specific timeseries objects. Need to use `getTSO()` to obtain `new_timeseries` from the `time_series` list.
    * `simplestats(self,new_timeseries="")`: calculates the mean and standard deviation
    * `bin_data(new_timeseries, x_axis = "", bin_size = "", start = "", end = "")`: staticmethod. Bins the data
    * `interp_data(new_timeseries, x_axis = "", interp_step="",start ="", end ="")`: staticmethod. Linear interpolation of the data.

### <a id = "sumplt"> `SummaryPlots` </a>

Contains class:
* `SummaryPlots`, with methods:
  * `__init__(self, time_series, plot_default)`: Passes the `time_series` and `plot_default` variables
  * `getMetadata(self, time_series)`: Returns a dictionary containing the necessary metadata to print on the figure
  * `TSdata(self,new_timeseries ="", x_axis = "")`: Get the necessary information for the timeseries plot
  * `agemodelData(self, new_timeseries ="")`: Get the necessary information for the age-depth relationship plot.
  * `basic(self,x_axis="", new_timeseries = "", saveFig = True, format = "eps", dir = "figures")`: Makes the basic summary plot.
