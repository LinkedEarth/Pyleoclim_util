.. Pyleoclim documentation master file, created by
   sphinx-quickstart on Fri Feb 10 13:56:30 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

Pyleoclim is a Python package designed for, but not limited to, the analysis of paleoclimate data.
Pyleoclim leverages various data science libraries (numpy, pandas, scikit-learn) for time series analysis, as well as Matplotlib and Cartopy for the creation of publication-quality graphics. Basic familiarity with Python is essential, and many good tutorials exist on the topic.

The package is designed around object-oriented :ref:`Series <core_api>`, which can be directly manipulated for plotting, spectral and wavelet analysis, and other time series-appropriate operations.
The main Series class is very flexible, applicable to virtually any timeseries, including instrumental or model-generated data. The package distinguishes itself by its handling of unevenly-spaced observations, which may come from Excel spreadsheets (csv), NumPy arrays, netCDF files, or pandas dataframes.
The GeoSeries class takes this one step further, enabling geolocation and associated functionalities (e.g. mapping).
These classes allow Pyleoclim to readily analyze datasets stored in the Linked Paleo Data (`LiPD <http://lipd.net>`_) format, through the `PyLiPD <https://pylipd.readthedocs.io/en/latest/>`_ package.
In particular, the package can make use of age ensembles and uses them for time-uncertain analysis. The age ensembles must however be generated externally, e.g. through the `GeoChronR <http://nickmckay.github.io/GeoChronR/>`_ package, which natively stores them as ensemble tables in LiPD.

However, Pyleoclim is by no means limited to LiPD-formatted data, and has been used in astronomy, finance, and robotics. Indeed, Pyleoclim is the workhorse supporting more general `machine-learning functionalities <https://github.com/KnowledgeCaptureAndDiscovery/autoTS>`_ for all manner of timeseries.

This documentation explains the basic usage of Pyleoclim functionalities. A progressive introduction to scientific uses of the package is available at `PyleoTutorials <http://linked.earth/PyleoTutorials/>`_. 
Examples of scientific use are given `this paper <https://doi.org/10.1029/2022PA004509>`_.  A growing collection of research-grade workflows using Pyleoclim and the LinkedEarth research ecosystem are available as `PaleoBooks <http://linked.earth/PaleoBooks/>`_, with video tutorials on the LinkedEarth `YouTube channel <https://www.youtube.com/watch?v=LJaQBFMK2-Q&list=PL93NbaRnKAuF4WpIQf-4y_U4lo-GqcrcW>`_. 
Python novices are encouraged to follow these `self-paced tutorials <http://linked.earth/LeapFROGS>`_ before trying Pyleoclim.

Getting Started
===============

.. toctree::
   :caption: Working with Pyleoclim
   :maxdepth: 1

   installation.rst
   core/api.rst
   tutorials.rst

The :ref:`Pyleoclim APIs <core_api>` make use of specialized routines which are described in details in advanced functionalities.

.. toctree::
   :caption: Advanced functionalities
   :maxdepth: 1

   utils/introduction.rst

Getting Involved
================

.. toctree::
   :Hidden:
   :caption: Getting Involved
   :maxdepth: 1

   citation.rst
   contribution_guide.rst

Pyleoclim has been made freely available under the terms of the `GNU Public License <https://github.com/LinkedEarth/Pyleoclim_util/blob/master/license>`_, and follows an open development model.
There are many ways to get :ref:`involved in the development of Pyleoclim <contributing_to_pyleoclim>`:

  * If you write a paper making use of Pyleoclim, please cite it :ref:`thus <citing_pyleoclim>`.
  * Report bugs and problems with the code or documentation to our `GitHub repository <https://github.com/LinkedEarth/Pyleoclim_util/issues>`_. Please make sure that there is not outstanding issues that cover the problem you're experiencing.
  * Contribute bug fixes
  * Contribute enhancements and new features
  * Contribute to the code documentation, and share your Pyleoclim-supported scientific workflow as a (`PaleoBook <http://linked.earth/PaleoBooks/>`_).

Search Pyleoclim
================

* :ref:`genindex`
* :ref:`search`
