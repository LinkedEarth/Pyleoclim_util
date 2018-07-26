<!---[![PyPI](https://img.shields.io/pypi/dm/pyleoclim.svg)](https://pypi.python.org/pypi/Pyleoclim)-->
[![PyPI](https://img.shields.io/pypi/v/pyleoclim.svg)]()
[![PyPI](https://img.shields.io/badge/python-3.5-yellow.svg)]()
[![license](https://img.shields.io/github/license/linkedearth/Pyleoclim_util.svg)]()
[![DOI](https://zenodo.org/badge/59611213.svg)](https://zenodo.org/badge/latestdoi/59611213)

# Pyleoclim

**Python Package for the Analysis of Paleoclimate Data**

**Table of contents**

* [What is it?](#what)
* [Installation](#install)
* [Version Information](#version)
* [Quickstart Guide](#quickstart)
* [Requirements](#req)
* [Further information](#further_info)
* [Contact](#contact)
* [License](#license)
* [Disclaimer](#disclaimer)


Current Version: 0.4.8

### <a name = "what">What is it?</a>

Pyleoclim is a Python package primarily geared towards the analysis and visualization of paleoclimate data. Such data often come in the form of timeseries with missing values and age uncertainties, so the package includes several low-level methods to deal with these issues, as well as high-level methods that re-use those within scientific workflows.

High-level modules assume that data are stored in the Linked Paleo Data ([LiPD](http://www.clim-past.net/12/1093/2016/)) format and makes extensive use of the [LiPD utilities](http://nickmckay.github.io/LiPD-utilities/). Low-level modules are primarily based on [NumPy](http://www.numpy.org) arrays or [Pandas](https://pandas.pydata.org) dataframes, so Pyleoclim contains a lot of timeseries analysis code (e.g. spectral analysis, singular spectrum analysis, wavelet analysis, correlation analysis) that can apply to these more common types as well. [need a link to some notebook examples]

The package is aware of age ensembles stored via LiPD and uses them for time-uncertain analyses very much like [GeoChronR](http://nickmckay.github.io/GeoChronR/).

**Current capabilities**:
 - binning
 - interpolation
 - standardization
 - plotting maps, timeseries, and basic age model information
 - paleo-aware correlation analysis (isopersistent, isospectral and classical t-test)
 - weighted wavelet Z transform (WWZ)
 - age modelling through Bchron

**Future capabilities**:
 - paleo-aware singular spectrum analysis (AR(1) null eigenvalue identification, missing data)
 - spectral analysis (Multi-Taper Method, Lomb-Scargle)
 - cross-wavelet analysis
 - index reconstruction
 - climate reconstruction
 - ensemble methods for most of the above

 If you have specific requests, please contact linkedearth@gmail.com


### <a name = "version">Version Information</a>
0.4.8: Add support of f2py WWZ for Linux
0.4.7: Update to coherence function
0.4.6: Fix an issue when copying the .so files
0.4.5: Update to setup.py to include proper .so file according to version
0.4.4: New fix for .so issue
0.4.3: New fix for .so issue
0.4.2: Fix issue concerning download of .so files
0.4.1: Fix issues with tarball
0.4.0: New functionalities: map nearest records by archive type, plot ensemble time series, age modelling through Bchron
0.3.1: New functionalities: segment a timeseries using a gap detection criteria, update to summary plot to perform spectral analysis
0.3.0: Compatibility with LiPD 1.3 and Spectral module added
0.2.5: Fix error on loading (Looking for Spectral Module)
0.2.4: Fix load error from init
0.2.3: Freeze LiPD version to 1.2 to avoid conflicts with 1.3
0.2.2: Change progressbar to tqdm and add standardization function
0.2.1: Update package requirements
0.2.0: Restructure the package so that the main functions can be called without the use of a LiPD files and associated timeseries objects.
0.1.4: Rename function using camel case and consistency with LiPD utilities version 0.1.8.5
0.1.3: Compatible with LiPD utilities version 0.1.8.5.
Function openLiPD() renamed openLiPDs()
0.1.2: Compatible with LiPD utilities version 0.1.8.3. Uses basemap instead of cartopy
0.1.1: Freezes the package prior to version 0.1.8.2 of LiPD utilities
0.1.0: First release


### <a name = "install"> Installation </a>

Python v3.4+ is required. Tested with Python v3.5 and v3.6

Will not run on a Windows system

The **recommended way to install Pyleoclim** is via `conda`
```bash
conda install pyleoclim -c linkedearth
```

Besides, it can be installed via `pip`
```
pip install pyleoclim
```


### <a name ="quickstart"> Quickstart guide </a>

1. Open your command line application (Terminal or Command Prompt).

2. Install with command: `conda install pyleoclim -c linkedearth`

3. Wait for installation to complete, then:

    3a. Import the package into your favorite Python environment (we recommend the use of Spyder, which comes standard with the Anaconda package)

    3b. Use Jupyter Notebook to go through the tutorial contained in the `PyleoclimQuickstart.ipynb` Notebook, which can be downloaded [here](https://github.com/LinkedEarth/Pyleoclim_util/tree/master/Example).

4. Help with functionalities can be found in the Documentation folder on [here](http://linkedearth.github.io/Pyleoclim_util/).

### <a name="req">Requirements</a>

- LiPD 0.2.5+
- pandas v0.22+
- numpy v1.14+
- matplotlib v2.0+
- Basemap v1.0.7+
- scipy v0.19.0+
- statsmodel v0.8.0+
- seaborn 0.7.0+
- scikit-learn 0.17.1+
- tqdm 4.14.0+
- pathos 0.2.0+
- tqdm 4.14+
- rpy2 2.8.4+

The installer will automatically check for the needed updates

### <a name="further_info">Further information</a>

GitHub: https://github.com/LinkedEarth/Pyleoclim_util

LinkedEarth: http://linked.earth

Python and Anaconda: http://conda.pydata.org/docs/test-drive.html

Jupyter Notebook: http://jupyter.org

### <a name = "contact"> Contact </a>

Please report issues to <linkedearth@gmail.com>

### <a name ="license"> License </a>

The project is licensed under the GNU Public License. Please refer to the file call license.
If you use the code in publications, please credit the work using [this citation](https://zenodo.org/record/1212692#.WsaZ7maZNE4).


### <a name = "disclaimer"> Disclaimer </a>

This material is based upon work supported by the National Science Foundation under Grant Number ICER-1541029. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the investigators and do not necessarily reflect the views of the National Science Foundation.
