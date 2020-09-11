.. Pyleoclim documentation master file, created by
   sphinx-quickstart on Fri Feb 10 13:56:30 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:
   installation.rst
   citation.rst
   core/introduction.rst

Introduction
============

Pyleoclim is a Python package designed for the analysis of paleoclimate data.

Pyleoclim makes use of various data science libraries (numpy, pandas, scikit-learn) for time series analysis and Matplotlib and Cartopy for the creation of publication quality figures.

Key features of Pyleoclim are its object-oriented :ref:`Series <ui_introduction>` which can be directly manipulated for plotting, spectral and wavelet analysis, and other time series-appropriate operations as well as its ability to manipulate `LiPD <http://lipd.net>`_ files directly, handling most of the data transformations internally.

The package is aware of age ensembles stored via LiPD and uses them for time-uncertain analyses. Age ensembles can be obtained through the `GeoChronR <http://nickmckay.github.io/GeoChronR/>`_ package.

Getting Started
===============

The :ref:`installation guide <installation>` provides information on getting up and running. The main Pyleoclim's function are available through its objected-oriented :ref:`interface <ui_introduction>` and documented in the form of a user guide, available online and in PDF format from GitHub.


Getting Involved
================

Pyleoclim was originally developed to allows scientists to analyze datasets in the LiPD format, including visualization, mapping, and time series analysis. Pyleoclim has been made freely available under the terms of the `GNU Public License <https://github.com/LinkedEarth/Pyleoclim_util/blob/master/license>`_.

There are many ways to get involved in the development of Pyleoclim:

  * If you write a paper making use of Pyleoclim, please consider :ref:`citing <citing_pyleoclim>`.
  * Report bugs and problems with the code or documentation to our `GitHub repository <https://github.com/LinkedEarth/Pyleoclim_util/issues>`_. Please make sure that there is not outstanding issues that cover the problem you're experiencing.
  * Contribute bug fixes
  * Contribute enhancements and new features

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
