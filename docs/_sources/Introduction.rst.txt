Pyleoclim
=========

What is it?
```````````

Pyleoclim is a Python package primarily geared towards the analysis and visualization of paleoclimate data.
Such data often come in the form of timeseries with missing values and age uncertainties, and the package
includes several low-level methods to deal with these issues, as well as high-level methods that re-use those
to perform scientific workflows.

The package assumes that the data are stored in the Linked Paleo Data (`LiPD <http://www.clim-past.net/12/1093/2016/>`_)
format and makes extensive use of the `LiPD utilities <http://nickmckay.github.io/LiPD-utilities/>`_. The package
is aware of age ensembles stored via LiPD and uses them for time-uncertain analyses very much like `GeoChronR <http://nickmckay.github.io/GeoChronR/>`_.

**Current Capabilities:**

* binning
* interpolation
* plotting maps, timeseries, and basic age model information
* paleo-aware correlation analysis (isopersistent, isospectral, and classical t-test)
* weighted wavelet Z transform (WWZ)
* age modeling through Bchron

**Future capabilities:**

* paleo-aware singular spectrum analysis (AR(1) null eigenvalue identification, missing data)
* spectral analysis (Multi-Taper Method, Lomb-Scargle)
* cross-wavelet analysis
* index reconstruction
* climate reconstruction

* ensemble methods for most of the above

Version Information
```````````````````
| 0.4.0: New functionalities: map nearest records by archive type, plot ensemble time series, age modelling through Bchron.
| 0.3.1: New functionalities: segment a timeseries using a gap detection criteria, update to summary plot to perform spectral analysis
| 0.3.0: Compatibility with LiPD 1.3 and Spectral module added
| 0.2.5: Fix error on loading (Looking for Spectral Module)
| 0.2.4: Fix load error from init
| 0.2.3: Freeze LiPD version to 1.2 to avoid conflicts with 1.3
| 0.2.2: Change progressbar to tqdm and add standardization function
| 0.2.1: Update package requirements
| 0.2.0: Restructure the package so that the main functions can be called without the use of a LiPD files and associated timeseries objects.
| 0.1.4: Rename functions using camel case convention and consistency with LiPD utilities version 0.1.8.5
| 0.1.3: Compatible with LiPD utilities version 0.1.8.5
|        Function openLiPD() renamed openLiPDs()
| 0.1.2: Compatible with LiPD utilities version 0.1.8.3
|        Uses Basemap instead of cartopy
| 0.1.1: Freezes the package prior to version 0.1.8.2 of LiPD utilities
| 0.1.0: First release

Installation
````````````
Python v3.4+ is required. Tested with Python v3.5

Will not run on a Windows system.

Pyleoclim is published through Pypi and easily installed via pip::

  pip install pyleoclim

Quickstart guide
````````````````

1. Open your command line application (Terminal or Command Prompt)
2. Install with command::

  pip install pyleoclim

3. Wait for installation to complete, then:

  a. Import the package into your favorite Python environment (we recommend the use of Spyder, which comes standard with the Anaconda build)
  b. Use Jupyter Notebook to go through the tutorial contained in the `PyleolimQuickstart.ipynb <https://github.com/LinkedEarth/Pyleoclim_util/tree/master/Example>`_

Requirements
````````````

* LiPD v0.2.5+
* pandas v0.22+
* numpy v1.14+
* matplotlib v2.0+
* Basemap v1.0.7+
* scipy v0.19.0+
* statsmodel v0.8.0+
* seaborn v0.7.0+
* scikit-learn v0.17.1+
* tqdm v4.14.0+
* pathos v0.2.0+
* tqdm 4.14+
* rpy2 2.8.4+

The installer will automatically check for the needed updates.

Further information
```````````````````
| GitHub: `https://github.com/LinkedEarth/Pyleoclim_util <https://github.com/LinkedEarth/Pyleoclim_util>`_
| LinkedEarth: `http://linked.earth <http://linked.earth>`_
| Python and Anaconda: `http://conda.pydata.org/docs/test-drive.html <http://conda.pydata.org/docs/test-drive.html>`_
| Jupyter Notebook: `http://jupyter.org/ <http://jupyter.org/>`_

Contact
```````
Please report issues to `linkedearth@gmail.com <linkedearth@gmail.com>`_

License
```````
The project is licensed under the `GNU Public License <https://github.com/LinkedEarth/Pyleoclim_util/blob/master/license>`_ .

Disclaimer
``````````
This material is based upon work supported by the U.S. National Science Foundation under Grant Number
ICER-1541029. Any opinions, findings, and conclusions or recommendations expressed in this material are those
of the investigators and do not necessarily reflect the views of the National Science Foundation.
