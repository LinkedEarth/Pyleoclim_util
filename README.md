[![PyPI](https://img.shields.io/pypi/dm/Pyleoclim.svg?maxAge=2592000)](https://pypi.python.org/pypi/Pyleoclim)
[![PyPI](https://img.shields.io/pypi/v/Pyleoclim.svg?maxAge=2592000)]()
[![PyPI](https://img.shields.io/badge/python-3.5-yellow.svg)]()
[![license](https://img.shields.io/github/license/linkedearth/Pyleoclim_util.svg?maxAge=2592000)]()

#Pyleoclim

**Table of contents**  

* [Description](#package)
    * [What is it?](#what)
    * [Installation](#install)
    * [Quickstart Guide](#quickstart)
    * [Requirements](#req)
    * [Further information](#further_info)
    * [Contact](#contact)
    * [License](#license)
    * [Disclaimer](#disclaimer)
* [Documentation](#doc)
    * [User's guide](#user_guide)
    * [Programmer's corner](#program_corner)

## <a name = "package">Description</a>
**Python Package for the Analysis of Paleoclimate Data**

Current Version: 0.1.8

### <a name = "what">What is it?</a>

Pyleoclim is a Python package for the analyses of paleoclimate data.

### <a name = "install"> Installation </a>

Python v3.5+ is required.

Pyleoclim is published through PyPi and easily installed via `pip`
```
pip install Pyleoclim
```

### <a name ="quickstart"> Quickstart guide </a>

1. Open your command line application (Terminal or Command Prompt).

2. Install with command: `pip install Pyleoclim`

3. Wait for installation to complete, then:

    3a. Import the package into your favorite Python environment (we recommend the use of Spyder, which comes standard with the Anaconda package)

    3b. Use Jupyter Notebook to go through the tutorial contained in the `PyleoclimQuickstart.ipynb` Notebook, which can be downloaded [here](https://github.com/LinkedEarth/Pyleoclim_util).

### <a name="req">Requirements</a>

- LiPD v0.1.8+
- pandas v0.19+
- numpy v1.12+
- matplotlib v2.0+
- Cartopy v0.13+

The installer will automatically check for the needed updates ***except*** for Cartopy.

Cartopy does not install properly through pip. The recommended method is through <a href="http://conda.pydata.org/miniconda.html"> Conda</a>. See the instructions on the <a href="http://scitools.org.uk/cartopy/docs/latest/installing.html"> developer website.

### <a name="further_info">Further information</a>

GitHub: https://github.com/LinkedEarth/Pyleoclim_util

LinkedEarth: http://linked.earth

Python and Anaconda: http://conda.pydata.org/docs/test-drive.html

Jupyter Notebook: http://jupyter.org

### <a name = "contact"> Contact </a>

Please report issues to <linkedearth@gmail.com>

### <a name ="license"> License </a>

The project is licensed under the GNU Public License. Please refer to the file call license.

### <a name = "disclaimer"> Disclaimer </a>

This material is based upon work supported by the National Science Foundation under Grant Number ICER-1541029. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the investigators and do not necessarily reflect the views of the National Science Foundation.

## <a name = "doc"> Documentation </a>

### <a name = "user_guide"> User guide </a>
To help you get started with Pyleoclim, refer to the Jupyter Notebook on our <a href="https://github.com/LinkedEarth/Pyleoclim_util"> GitHub repository </a>.

#### Mapping

##### `MapAll()`
This function maps all the LiPD records in the working directory according to archiveType. It uses a default palette color accessible by typing `pyleo.plot_default`

Synthax: `pyleoclim.MapAll(markersize = int(50), saveFig = True, dir = "", format = "eps")`

Optional arguments:
* `markersize`: default is 50
* `saveFig`: if `True`, saves the map into the `dir` folder in the current working directory.
* `dir`: a folder in the current directory, where the various figures can be saved. If left blank, a folder named `figures` will be automatically created
* `format`: One of the file extensions supported by the active backend. Default is `eps`. Most backends support `png`, `pdf`, `ps`, and `svg`.

##### `MapLiPD()`
This function maps one particular LiPD record stored in the working directory.

Synthax: `pyleoclim.MapLiPD(name="",gridlines = False, borders = True, topo = True, markersize = int(100), marker = "default", saveFig = True, dir = "", format = "eps")`

Optional arguments:
* `name`: Name of the LiPD record of interest, ***including the extension***! If you need a list of the records name you can either use `showLipds()` or leave the name blank. You'll be prompted to choose the record from a list.
* `gridlines`: Default is False.
* `borders`: Administrative borders. Default is `True`
* `topo`: Default cartopy topography. Default is `True`
* `markersize`: Default is 100
* `marker`: Pyleoclim comes with default marker shape and color for the various archives. If you wish to change the default marker, enter the color and shape (**in this order**). For instance to use a red square, use `'rs'`
*  `saveFig`: if `True`, saves the map into the `dir` folder in the current working directory.
* `dir`: a folder in the current directory, where the various figures can be saved. If left blank, a folder named `figures` will be automatically created
* `format`: One of the file extensions supported by the active backend. Default is `eps`. Most backends support `png`, `pdf`, `ps`, and `svg`.

#### Plotting

##### `plotTS()`

Plot a time series.

Synthax:
`pyleoclim.PlotTS(timeseries = "", x_axis = "", markersize = 50, marker = "default", saveFig = True, dir = "figures", format="eps")`

Optional arguments:
* `timeseries`: A timeseries object as defined in LiPD. If left blank, you'll be prompted to choose the record from a list.
* `x-axis`: The representation against which to plot the paleo-data. Options are "age","year", and "depth". Default is to let the system choose if only one available or prompt the user.
* `markersize`: Default is 50
* `marker`: Shape and color. Default uses the Pyleoclim color palette. If you wish to change the default marker, enter the color and shape (**in this order**). For instance to use a red square, use `'rs'`.
*  `saveFig`: if `True`, saves the map into the `dir` folder in the current working directory.
* `dir`: a folder in the current directory, where the various figures can be saved. If left blank, a folder named `figures` will be automatically created
* `format`: One of the file extensions supported by the active backend. Default is `eps`. Most backends support `png`, `pdf`, `ps`, and `svg`.

##### Summary Plots

Summary Plots are special plots in Pyleoclim that allow to get basic information about a record.

###### `BasicSummary()`

This functions plots:
1. The time series
2. The location map
3. Age/depth profile if both are available in the paleoDataTable
4. Metadata information

Synthax: `pyleoclim.BasicSummary(timeseries = "", x_axis="", saveFig = True, format = "eps", dir = "figures")`

* `timeseries`: A timeseries object as defined in LiPD. If left blank, you'll be prompted to choose the record from a list.
* `x-axis`: The representation against which to plot the paleo-data. Options are "age","year", and "depth". Default is to let the system choose if only one available or prompt the user.  
*  `saveFig`: if `True`, saves the map into the `dir` folder in the current working directory.
* `dir`: a folder in the current directory, where the various figures can be saved. If left blank, a folder named `figures` will be automatically created
* `format`: One of the file extensions supported by the active backend. Default is `eps`. Most backends support `png`, `pdf`, `ps`, and `svg`.

#### Basic functionalities
##### `TSstats()`

Returns the mean and standard deviation of the timeseries

synthax: `mean, std = pyleoclim.TSstats(timeseries="")`

Optional arguments:
* `timeseries`: If blank, will prompt for one.

##### `TSbin()`

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

##### `TSinterp()`

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

### <a name = "program_corner"> Programmer's corner </a>
Pyleoclim is meant to be developed by the Paleoclimate community. If you'd like to contribute codes, the content below explains the structure of the Pyleoclim package.

The easiest way to contribute is to create a class, regrouping all the necessary methods.

#### `__init__`

When loading the Pyleoclim package, the LiPD utilities and all dependency are initialized. The package also loads the LiPD files into the workspace and automatically extracts the timeseries objects under the variable `time_series`.  
Finally, the default color palette `plot_default` is also initialized Both of these variables can be passed into the __init__ method of the various classes.

The `__init__` module also contains the top-level methods described in the [user's guide](#user_guide).

The package contains five modules, whose functions are described below.

#### `LiPDutils`

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

#### `Map`

Contains class:
* `Map`, with methods:
  * `__init__(self, plot_default)`: Grabs the coordinates of all the LiPD files loaded in the workspace and passes the default color palette.
  * `map_all(self, markersize = int(50), saveFig = True, dir="", format='eps')`: map all LiPDs.
  * `map_one(self, name="",gridlines = False, borders = True, topo = True, markersize = int(100), marker = "default", saveFig = True, dir = "", format="eps")`: map one LiPD.

#### `TSPlot`

Contains class:
* `Plot`, with methods:
  * `__init__(self, plot_default,time_series)`: Passes the default color palette and available timeseries objects.
  * `plotoneTSO(self, new_timeseries = "", x_axis = "", markersize = 50, marker = "default", saveFig = True, dir = "figures", format="eps")`: Plot one time_series
  * `agemodelplot(self, new_timeseries = "", markersize = 50, marker = "default", saveFig = True, dir = "figures", format="eps" )`: Plots the basic age-depth relationship inferred from the paleoDataTable

#### `Basic`

  Manipulation of LiPD files (timeseries objects) for scientific purposes

  Contains class:
  * `Basic`, with methods:
    * `__init__(self, time_series)`: Passes the `time_series` variable
    * `getValues(new_timeseries)`: staticmethod. Get the values for a specific timeseries objects. Need to use `getTSO()` to obtain `new_timeseries` from the `time_series` list.
    * `simplestats(self,new_timeseries="")`: calculates the mean and standard deviation
    * `bin_data(new_timeseries, x_axis = "", bin_size = "", start = "", end = "")`: staticmethod. Bins the data
    * `interp_data(new_timeseries, x_axis = "", interp_step="",start ="", end ="")`: staticmethod. Linear interpolation of the data.

#### `SummaryPlots`

Contains class:
* `SummaryPlots`, with methods:
  * `__init__(self, time_series, plot_default)`: Passes the `time_series` and `plot_default` variables
  * `getMetadata(self, time_series)`: Returns a dictionary containing the necessary metadata to print on the figure
  * `TSdata(self,new_timeseries ="", x_axis = "")`: Get the necessary information for the timeseries plot
  * `agemodelData(self, new_timeseries ="")`: Get the necessary information for the age-depth relationship plot.
  * `basic(self,x_axis="", new_timeseries = "", saveFig = True, format = "eps", dir = "figures")`: Makes the basic summary plot.
