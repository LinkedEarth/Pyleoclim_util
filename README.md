[![PyPI](https://img.shields.io/pypi/dm/pyleoclim.svg)](https://pypi.python.org/pypi/Pyleoclim)
[![PyPI](https://img.shields.io/pypi/v/pyleoclim.svg)]()
[![PyPI](https://img.shields.io/badge/python-3.5-yellow.svg)]()
[![license](https://img.shields.io/github/license/linkedearth/Pyleoclim_util.svg)]()

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

Current Version: 0.3.1

### <a name = "what">What is it?</a>

Pyleoclim is a Python package primarily geared towards the analysis and visualization of paleoclimate data. Such data often come in the form of timeseries with missing values and age uncertainties, and the package includes several low-level methods to deal with these issues, as well as high-level methods that re-use those to perform scientific workflows.

The package assumes that data are stored in the Linked Paleo Data ([LiPD](http://www.clim-past.net/12/1093/2016/)) format and makes extensive use of the [LiPD utilities](http://nickmckay.github.io/LiPD-utilities/). The package is aware of age ensembles stored via LiPD and uses them for time-uncertain analyses very much like [GeoChronR](http://nickmckay.github.io/GeoChronR/).

**Current capabilities**:
 - binning
 - interpolation
 - standardization
 - plotting maps, timeseries, and basic age model information
 - paleo-aware correlation analysis (isopersistent, isospectral and classical t-test)
 - weighted wavelet Z transform (WWZ)

**Future capabilities**:
 - paleo-aware singular spectrum analysis (AR(1) null eigenvalue identification, missing data)
 - spectral analysis (Multi-Taper Method, Lomb-Scargle)
 - cross-wavelet analysis
 - index reconstruction
 - climate reconstruction
 - ensemble methods for most of the above

 If you have specific requests, please contact linkedearth@gmail.com

### <a name = "version">Version Information</a>
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

Python v3.4+ is required. Tested with Python v3.5

Pyleoclim is published through PyPi and easily installed via `pip`
```
pip install pyleoclim
```

**Warning**: Some dependencies require an installation through Conda

### <a name ="quickstart"> Quickstart guide </a>

1. Open your command line application (Terminal or Command Prompt).

2. Install denpendencies that `pip` cannot handle

    2a. Install `basemap` through Conda: `conda install -c conda-forge basemap`

    2b. Download and install `R` through [CRAN](https://cran.rstudio.com/) and then `pip install rpy2` to enable the usage of the Bchron module. Note that `conda install rpy2` or `conda install r-essentials` will not work properly.

3. Install the pyleoclim package with command: `pip install pyleoclim`

4. Wait for installation to complete, then:

    4a. Import the package into your favorite Python environment (we recommend the use of Spyder, which comes standard with the Anaconda package)

    4b. Use Jupyter Notebook to go through the tutorial contained in the `PyleoclimQuickstart.ipynb` Notebook, which can be downloaded [here](https://github.com/LinkedEarth/Pyleoclim_util/tree/master/Example).

5. Help with functionalities can be found in the Documentation folder on [here](http://linkedearth.github.io/Pyleoclim_util/).

### <a name="req">Requirements</a>

- LiPD 0.2.2+
- pandas v0.19+
- numpy v1.12+
- matplotlib v2.0+
- Basemap v1.0.7+
- scipy v0.19.0+
- statsmodel v0.8.0+
- seaborn 0.7.0+
- scikit-learn 0.17.1+
- tqdm 4.14.0+
- pathos 0.2.0+

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

### <a name = "disclaimer"> Disclaimer </a>

This material is based upon work supported by the National Science Foundation under Grant Number ICER-1541029. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the investigators and do not necessarily reflect the views of the National Science Foundation.
