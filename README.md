<!---[![PyPI](https://img.shields.io/pypi/dm/pyleoclim.svg)](https://pypi.python.org/pypi/Pyleoclim)-->
[![PyPI version](https://badge.fury.io/py/pyleoclim.svg)](https://badge.fury.io/py/pyleoclim)
[![PyPI](https://img.shields.io/badge/python-3.8-yellow.svg)]()
[![license](https://img.shields.io/github/license/linkedearth/Pyleoclim_util.svg)]()
[![DOI](https://zenodo.org/badge/59611213.svg)](https://zenodo.org/badge/latestdoi/59611213)
[![NSF-1541029](https://img.shields.io/badge/NSF-1541029-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1541029)
[![Build Status](https://travis-ci.org/LinkedEarth/Pyleoclim_util.svg?branch=master)](https://travis-ci.org/LinkedEarth/Pyleoclim_util)

# Pyleoclim

**Python Package for the Analysis of Paleoclimate Data**

Pyleoclim is a Python package primarily geared towards the analysis and visualization of paleoclimate data. Such data often come in the form of timeseries with missing values and age uncertainties, so the package includes several low-level methods to deal with these issues, as well as high-level methods that re-use those within scientific workflows.

High-level modules assume that data are stored in the Linked Paleo Data ([LiPD](http://www.clim-past.net/12/1093/2016/)) format and makes extensive use of the [LiPD utilities](http://nickmckay.github.io/LiPD-utilities/). Low-level modules are primarily based on [NumPy](http://www.numpy.org) arrays or [Pandas](https://pandas.pydata.org) dataframes, so Pyleoclim contains a lot of timeseries analysis code (e.g. spectral analysis, singular spectrum analysis, wavelet analysis, correlation analysis) that can apply to these more common types as well. See the example folder for details.

The package is aware of age ensembles stored via LiPD and uses them for time-uncertain analyses very much like [GeoChronR](http://nickmckay.github.io/GeoChronR/).

New in version 0.5.0:

- code design centered on timeseries objects
- [user interface](https://github.com/LinkedEarth/Pyleoclim_util/blob/master/example_notebooks/pyleoclim_ui_tutorial.ipynb) allowing succinct, expressive calls
- expanded repertoire of spectral and wavelet methods
- inclusion of singular spectral analysis, permitting missing data
- Sphinx [documentation](http://linkedearth.github.io/Pyleoclim_util/) for all functions
- clean, additive [plot styles](https://github.com/LinkedEarth/Pyleoclim_util/blob/master/example_notebooks/plot_styles.ipynb) inspired by Matplotlib [style sheets](https://matplotlib.org/3.3.1/gallery/style_sheets/style_sheets_reference.html).  


### Documentation

Online documentation is available through readthedocs:
- [Stable version](https://pyleoclim-util.readthedocs.io/en/stable/) (available through Pypi
- [Latest version](https://pyleoclim-util.readthedocs.io/en/latest/) (from the development branch)

### Dependencies

pyleoclim supports Python 3.8

### Installation

The latest stable release is available through Pypi. We recommend using Anaconda or Miniconda with a dedicated environment.
 
 `pip install pyleoclim`

You may also want to use the Development version from GitHub to access the latest functionalities
 
 `git+https://github.com/LinkedEarth/Pyleoclim_util.git@Development`


### Development

Pyleoclim development takes place on GitHub: https://github.com/LinkedEarth/Pyleoclim_util

Please submit any reproducible bugs you encounter to the [issue tracker](https://github.com/LinkedEarth/Pyleoclim_util/issues)

### License

The project is licensed under the GNU Public License. Please refer to the file call license.
If you use the code in publications, please credit the work using [this citation](https://zenodo.org/record/1212692#.WsaZ7maZNE4).


### Disclaimer

This material is based upon work supported by the National Science Foundation under Grant Number ICER-1541029. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the investigators and do not necessarily reflect the views of the National Science Foundation.

This research is funded in part by JP Morgan Chase & Co. Any views or opinions expressed herein are solely those of the authors listed, and may differ from the views and opinions expressed by JP Morgan Chase & Co. or its affilitates. This material is not a product of the Research Department of J.P. Morgan Securities LLC. This material should not be construed as an individual recommendation of for any particular client and is not intended as a recommendation of particular securities, financial instruments or strategies for a particular client. This material does not constitute a solicitation or offer in any jurisdiction.
