.. _core_ui:

.. currentmodules:: pyleoclim.core.ui

Pyleoclim User API
===================

Pyleoclim, like a lot of other Python packages, follows an object-oriented design. It sounds fancy, but it really is `quite simple <https://www.freecodecamp.org/news/object-oriented-programming-concepts-21bb035f7260/>`_. What this means for you is that we've gone through the trouble of coding up a lot of timeseries analysis methods that apply in various situations - so you don't have to worry about that.
These situations are described in classes, the beauty of which is called "inheritance" (see link above). Basically, it allows to define methods that will automatically apply to your dataset, as long as you put your data within one of those classes.
A major advantage of object-oriented design is that you, the user, can harness the power of Pyleoclim methods in very few lines of code through the user API without ever having to get your hands dirty with our code (unless you want to, of course).
The flipside is that any user would do well to understand Pyleoclim classes, what they are intended for, and what methods they can and cannot support.

.. image:: Pyleoclim_UI.png
   :scale: 80 %
   :alt: The Pyleoclim UI. Credit: Feng Zhu
   :align: center

The following describes the various classes that undergird the Pyleoclim edifice.

Series (pyleoclim.Series)
"""""""""""""""""""""""""

The Series class describes the most basic objects in Pyleoclim. A Series is a simple `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_ that contains 3 things:
- a series of real-valued numbers;
- a time axis at which those values were measured/simulated ;
- optionally, some metadata about both axes, like units, labels and the like.

How to create and manipulate such objects is described in a short example below, while `this notebook <https://nbviewer.jupyter.org/github/LinkedEarth/Pyleoclim_util/blob/master/example_notebooks/pyleoclim_ui_tutorial.ipynb>`_ demonstrates how to apply various Pyleoclim methods to Series objects.


.. autoclass:: pyleoclim.core.ui.Series
   :members:


MultipleSeries (pyleoclim.MultipleSeries)
"""""""""""""""""""""""""""""""""""""""""

As the name implies, a MultipleSeries object is a collection (more precisely, a `list <https://docs.python.org/3/tutorial/introduction.html#lists>`_) of multiple Series objects. This is handy in case you want to apply the same method to such a collection at once (e.g. process a bunch of series in a consistent fashion).

.. autoclass:: pyleoclim.core.ui.MultipleSeries
   :members:


EnsembleSeries (pyleoclim.EnsembleSeries)
"""""""""""""""""""""""""""""""""""""""""

The EnsembleSeries class is a child of MultipleSeries, designed for ensemble applications (e.g. draws from a posterior distribution of ages, model ensembles with randomized initial conditions, or some other stochastic ensemble).
Compared to a MultipleSeries object, an EnsembleSeries object has the following properties:  [TO BE COMPLETED]

.. autoclass:: pyleoclim.core.ui.EnsembleSeries
   :members:

Lipd (pyleoclim.Lipd)
"""""""""""""""""""""

This class allows to manipulate LiPD objects.

.. autoclass:: pyleoclim.core.ui.Lipd
   :members:


LipdSeries (pyleoclim.LipdSeries)
"""""""""""""""""""""""""""""""""

LipdSeries are (you guessed it), Series objects that are created from LiPD objects. As a subclass of Series, they inherit all its methods.
When created, LiPDSeries automatically instantiates the time, value and other parameters from what’s in the lipd file.

.. autoclass:: pyleoclim.core.ui.LipdSeries
   :members:


PSD (pyleoclim.PSD)
"""""""""""""""""""

The PSD (Power spectral density) class is intended for conveniently manipulating the result of spectral methods, including performing significance tests, estimating scaling coefficients, and plotting.
Available methods:

.. autoclass:: pyleoclim.core.ui.PSD
   :members:


Scalogram (pyleoclim.Scalogram)
"""""""""""""""""""""""""""""""
The Scalogram class is analogous to PSD, but for wavelet spectra (scalograms)

.. autoclass:: pyleoclim.core.ui.Scalogram
   :members:


Coherence (pyleoclim.Coherence)
"""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.Coherence
   :members:


SurrogateSeries (pyleoclim.SurrogateSeries)
"""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.SurrogateSeries
   :members:


MultiplePSD (pyleoclim.MultiplePSD)
"""""""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.MultiplePSD
   :members:


MultipleScalogram (pyleoclim.MultipleScalogram)
"""""""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.MultipleScalogram
   :members:

Corr (pyleoclim.Corr)
"""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.Corr
   :members:


CorrEns (pyleoclim.CorrEns)
"""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.CorrEns
   :members:

SsaRes (pyleoclim.SsaRes)
"""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.SsaRes
   :members:

gen_ts (pyleoclim.gen_ts)
"""""""""""""""""""""""""

.. autofunction:: pyleoclim.gen_ts
