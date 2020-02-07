Pyleoclim
=========

What is it?
```````````

Pyleoclim is a Python package primarily geared towards the analysis and visualization of paleoclimate data.
Such data often come in the form of timeseries with missing values and age uncertainties,
so the package includes several low-level methods to deal with these issues,
as well as high-level methods that re-use those within scientific workflows.

High-level modules assume that data are stored in the Linked Paleo Data (`LiPD <http://www.clim-past.net/12/1093/2016/>`_) format
and makes extensive use of the `LiPD utilities <http://nickmckay.github.io/LiPD-utilities/>`_. Low-level modules are primarily
based on numpy arrays or Pandas dataframes, so Pyleoclim contains a lot of
timeseries analysis code (e.g. spectral analysis, singular spectrum analysis,
wavelet analysis, correlation analysis) that can apply to these more common types
as well. See the example folder for details.

The package is aware of age ensembles stored via LiPD and uses them for time-uncertain analyses very much like `GeoChronR <http://nickmckay.github.io/GeoChronR/>`_.

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
* causality
* ensemble methods for most of the above

Installation
````````````
Python v3.6 is **required**.

We recommend using `Anaconda <https://www.anaconda.com/distribution/>`_ and creating a new environment:

  conda create --name pyleoclim python=3.6

Activate the environment:

  conda activate pyleoclim

To install Pyleoclim, first install Cartopy and Numpy through Anaconda.

  conda install -c conda-forge cartopy
  conda install numpy

Pyleoclim is published through Pypi and easily installed via pip::

  pip install pyleoclim

Note that the pip command line above will trigger the installation of (most of) the dependencies, as well as the local compilation of the Fortran code for WWZ with the GNU Fortran compiler gfortran. If you have the Intel's Fortran compiler ifort installed, then further accerlation for WWZ could be achieved by compiling the Fortran code with ifort, and below are the steps:

- download the source code, either via git clone or just download the .zip file
- modify setup.py by commenting out the line of extra_f90_compile_args for gfortran, and use the line below for ifort
- run python setup.py build_ext --fcompiler=intelem && python setup.py install.

Some functionalities require R. See Installation for details.

Version Information
```````````````````
**Current Version**
0.4.10: Support local compilation of the Fortran code for WWZ; precompiled .so files have been removed.

**Past Version**
0.4.9: Major bug fixes; mapping module based on cartopy; compatibility with latest numpy package
0.4.8: Add support of f2py WWZ for Linux
0.4.7: Update to coherence function
0.4.0: New functionalities: map nearest records by archive type, plot ensemble time series, age modelling through Bchron.
0.3.1: New functionalities: segment a timeseries using a gap detection criteria, update to summary plot to perform spectral analysis
0.3.0: Compatibility with LiPD 1.3 and Spectral module added
0.2.5: Fix error on loading (Looking for Spectral Module)
0.2.4: Fix load error from init
0.2.3: Freeze LiPD version to 1.2 to avoid conflicts with 1.3
0.2.2: Change progressbar to tqdm and add standardization function
0.2.1: Update package requirements
0.2.0: Restructure the package so that the main functions can be called without the use of a LiPD files and associated timeseries objects.
0.1.4: Rename functions using camel case convention and consistency with LiPD utilities version 0.1.8.5
0.1.3: Compatible with LiPD utilities version 0.1.8.5
        Function openLiPD() renamed openLiPDs()
0.1.2: Compatible with LiPD utilities version 0.1.8.3
        Uses Basemap instead of cartopy
0.1.1: Freezes the package prior to version 0.1.8.2 of LiPD utilities
0.1.0: First release


Quickstart guide
````````````````

1. Install Pyleoclim

3. Wait for installation to complete, then:

  a. Import the package into your favorite Python environment (we recommend the use of Spyder, which comes standard with the Anaconda build)
  b. Use Jupyter Notebook to go through the tutorial contained in the `PyleolimQuickstart.ipynb <https://github.com/LinkedEarth/Pyleoclim_util/tree/master/Example>`_

Requirements
````````````

Tested with:

* LiPD 0.2.7
* pandas v0.25.0
* numpy v1.16.4
* matplotlib v3.1.0
* Cartopy v1.17.0
* scipy v1.3.1
* statsmodel v0.8.0
* seaborn 0.9.0
* scikit-learn 0.21.3
* tqdm 4.33.0
* pathos 0.2.4
* rpy2 3.0.5

The installer will automatically check for the needed updates.

Known issues
````````````
* Some of the packages supporting Pyleoclim do not have a build for Windows
* Known issues with proj4 v5.0-5.1, make sure your environment is setup with 5.2

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
If you use the code in publications, please credit the work using `this citation <https://zenodo.org/record/1212692#.WsaZ7maZNE4>`_.

Disclaimer
``````````
This material is based upon work supported by the U.S. National Science Foundation under Grant Number
ICER-1541029. Any opinions, findings, and conclusions or recommendations expressed in this material are those
of the investigators and do not necessarily reflect the views of the National Science Foundation.
