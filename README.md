[![PyPI](https://img.shields.io/pypi/dm/pyleoclim.svg?maxAge=2592000)](https://pypi.python.org/pypi/Pyleoclim)
[![PyPI](https://img.shields.io/pypi/v/pyleoclim.svg?maxAge=2592000)]()
[![PyPI](https://img.shields.io/badge/python-3.5-yellow.svg)]()
[![license](https://img.shields.io/github/license/linkedearth/Pyleoclim_util.svg?maxAge=2592000)]()

#Pyleoclim

**Python Package for the Analysis of Paleoclimate Data**

**Table of contents**  

* [What is it?](#what)
* [Installation](#install)
* [Quickstart Guide](#quickstart)
* [Requirements](#req)
* [Further information](#further_info)
* [Contact](#contact)
* [License](#license)
* [Disclaimer](#disclaimer)

Current Version: 0.1.0

### <a name = "what">What is it?</a>

Pyleoclim is a Python package primarily geared towards the analysis and visualization of paleoclimate data. Such data often come in the form of timeseries with missing values and age uncertainties, and the package includes several-low level methods to deal with these issues, as well as high-level methods that re-use those to perform scientific workflows.

The packages assumes that data are stored in the Linked Paleo Data ([LiPD](http://www.clim-past.net/12/1093/2016/)) format and makes extensive use of the [LiPD utilities](http://nickmckay.github.io/LiPD-utilities/). The package is aware of age ensembles stored via LiPD and uses them for time-uncertain analyses very much like [GeoChronR](http://nickmckay.github.io/GeoChronR/).

**Current capabilities**: 
 - binning
 - interpolation 
 - plotting maps, timeseries, and basic age model information
 - paleo-aware correlation analysis (isopersistent, isospectral and classical t-test)

**Future capabilities**: 
 - spectral analysis (Multi-Taper Method, Lomb-Scargle)
 - weighted wavelet Z transform (WWZ)
 - cross-wavelet analysis
 - index reconstruction
 - climate reconstruction
 - ensemble methods for all of the above
 
 If you have specific requests, please contact linkedearth@gmail.com

### <a name = "install"> Installation </a>

Python v3.5+ is required.

Pyleoclim is published through PyPi and easily installed via `pip`
```
pip install pyleoclim
```

### <a name ="quickstart"> Quickstart guide </a>

1. Open your command line application (Terminal or Command Prompt).

2. Install with command: `pip install pyleoclim`

3. Wait for installation to complete, then:

    3a. Import the package into your favorite Python environment (we recommend the use of Spyder, which comes standard with the Anaconda package)

    3b. Use Jupyter Notebook to go through the tutorial contained in the `PyleoclimQuickstart.ipynb` Notebook, which can be downloaded [here](https://github.com/LinkedEarth/Pyleoclim_util/tree/master/Example).

4. Help with functionalities can be found in the Documentation folder on our [GitHub repository](https://github.com/LinkedEarth/Pyleoclim_util) and on Pypi.     

### <a name="req">Requirements</a>

- LiPD v0.1.8+
- pandas v0.19+
- numpy v1.12+
- matplotlib v2.0+
- Cartopy v0.13+

The installer will automatically check for the needed updates ***except*** for Cartopy.

Cartopy doesn't install properly through pip. Use <a href="http://conda.pydata.org/miniconda.html"> Conda</a>. See the instructions on the <a href="http://scitools.org.uk/cartopy/docs/latest/installing.html"> developer website.

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
