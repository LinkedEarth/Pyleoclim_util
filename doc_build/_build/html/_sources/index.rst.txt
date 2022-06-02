.. Pyleoclim documentation master file, created by
   sphinx-quickstart on Fri Feb 10 13:56:30 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

Pyleoclim is a Python package designed for the analysis of paleoclimate data.
Pyleoclim leverages various data science libraries (numpy, pandas, scikit-learn) for time series analysis, as well as and Matplotlib and Cartopy for the creation of publication-quality figures. The package is designed around object-oriented :ref:`Series <core_ui>`, which can be directly manipulated for plotting, spectral and wavelet analysis, and other time series-appropriate operations. Basic familiarity with Python is essential, and many good tutorials exist on the topic.

.. image:: LiPD1p3.png
   :width: 191px
   :height: 248px
   :scale: 100 %
   :alt: The LiPD data model, version 1.3. Credit: Nick McKay
   :align: right

Pyleoclim natively "speaks" the language of Linked Paleo Data  (`LiPD <http://lipd.net>`_), which enables it to handle most of the data transformations internally, taking a good chunk of the pain out of analyzing paleoclimate data.
The package is aware of age ensembles stored via LiPD and uses them for time-uncertain analysis. Age ensembles can be generated through the `GeoChronR <http://nickmckay.github.io/GeoChronR/>`_ package, which natively stores them as ensemble tables in LiPD.

While convenient for the representation of paleoclimate observations, LiPD is not the only point of entry into Pyleoclim. The :ref:`Series <core_ui>` class is very flexible, and allows to apply Pyleoclim functionality to virtually any timeseries, including instrumental or model-generated data, as Excel spreadsheets, numpy arrays or pandas dataframes. Indeed, Pyleoclim is the workhorse supporting more general `machine-learning functionalities <https://github.com/KnowledgeCaptureAndDiscovery/autoTS>`_ for all manner of timeseries.

Getting Started
===============

.. toctree::
   :caption: Working with Pyleoclim
   :maxdepth: 1

   installation.rst
   core/api.rst
   tutorials.rst

The :ref:`Pyleoclim UI <core_ui>` makes use of specialized routines which are described in details in advanced funtionalities.

.. toctree::
   :caption: Advanced functionalities
   :maxdepth: 1

   utils/introduction.rst

Getting Involved
================

.. toctree::
   :Hidden:
   :caption: Getting Involved
   :maxdepth: 2

   citation.rst
   contribution_guide.rst

Pyleoclim was originally developed to allow scientists to analyze paleoclimate datasets, including visualization, mapping, and time series analysis. Pyleoclim has been made freely available under the terms of the `GNU Public License <https://github.com/LinkedEarth/Pyleoclim_util/blob/master/license>`_.

There are many ways to get :ref:`involved in the development of Pyleoclim <contributing_to_pyleoclim>`:

  * If you write a paper making use of Pyleoclim, please cite it :ref:`thus <citing_pyleoclim>`.
  * Report bugs and problems with the code or documentation to our `GitHub repository <https://github.com/LinkedEarth/Pyleoclim_util/issues>`_. Please make sure that there is not outstanding issues that cover the problem you're experiencing.
  * Contribute bug fixes
  * Contribute enhancements and new features
  * Contribute to the code documentation, and share your Pyleoclim-supported scientific workflows via our public repository (`LiPDBooks <https://github.com/LinkedEarth/LiPDbooks>`_).

Search Pyleoclim
================

* :ref:`genindex`
* :ref:`search`
