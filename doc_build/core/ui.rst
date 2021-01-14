.. _core_ui:

.. currentmodules:: pyleoclim.core.ui

Demystifying the Pyleoclim User Interface
=========================================

Pyleoclim, like a lot of other Python packages, follows an object-oriented design. It sounds fancy, but it really is `quite simple <https://www.freecodecamp.org/news/object-oriented-programming-concepts-21bb035f7260/>`_. What this means for you is that we've gone through the trouble of coding up a lot of timeseries analysis methods that apply in various situations - so you don't have to worry about that.
These situations are described in classes, the beauty of which is called "inheritance" (see link above). Basically, it allows to define methods that will automatically apply to your dataset, as long as you put your data within one of those classes.
A major advantage of object-oriented design is that you, the user, can harness the power of Pyleoclim methods in very few lines of code.
The flipside is that it is important to understand the classes, what they are intended for, and what methods they can and cannot support.

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
   :inherited-members:

MultipleSeries (pyleoclim.MultipleSeries)
"""""""""""""""""""""""""""""""""""""""""

As the name implies, a MultipleSeries object is a collection (more precisely, a `list <https://docs.python.org/3/tutorial/introduction.html#lists>`_) of multiple Series objects. This is handy in case you want to apply the same method to such a collection at once (e.g. process a bunch of series in a consistent fashion).

.. autoclass:: pyleoclim.core.ui.MultipleSeries
   :members:
   :inherited-members:

EnsembleSeries (pyleoclim.EnsembleSeries)
"""""""""""""""""""""""""""""""""""""""""

The EnsembleSeries class is a child of MultipleSeries, designed for ensemble applications (e.g. draws from a posterior distribution of ages, model ensembles with randomized initial conditions, or some other stochastic ensemble).
Compared to a MultipleSeries object, an EnsembleSeries object has the following properties:  [TO BE COMPLETED]

.. autoclass:: pyleoclim.core.ui.EnsembleSeries
   :members:
   :inherited-members:

Lipd (pyleoclim.Lipd)
"""""""""""""""""""""

This class allows to manipulate LiPD objects.

.. autoclass:: pyleoclim.core.ui.Lipd
   :members:
   :inherited-members:

LipdSeries (pyleoclim.LipdSeries)
"""""""""""""""""""""""""""""""""

This class allows to manipulate LiPD timeseries objects. This is a subclass of Series. See applicable methods there.

.. autoclass:: pyleoclim.core.ui.LipdSeries
   :members:
   :inherited-members:

PSD (pyleoclim.PSD)
"""""""""""""""""""

The PSD (Power spectral density) class is intended for conveniently manipulating the result of spectral methods, including performing significance tests, estimating scaling coefficients, and plotting.
Available methods:

.. autoclass:: pyleoclim.core.ui.PSD
   :members:
   :inherited-members:

Scalogram (pyleoclim.Scalogram)
"""""""""""""""""""""""""""""""
The Scalogram class is analogous to PSD:

.. autoclass:: pyleoclim.core.ui.Scalogram
   :members:
   :inherited-members:

Coherence (pyleoclim.Coherence)
"""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.Coherence
   :members:
   :inherited-members:

SurrogateSeries (pyleoclim.SurrogateSeries)
"""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.SurrogateSeries
   :members:
   :inherited-members:

MultiplePSD (pyleoclim.MultiplePSD)
"""""""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.MultiplePSD
   :members:
   :inherited-members:

MultipleScalogram (pyleoclim.MultipleScalogram)
"""""""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.MultipleScalogram
   :members:
   :inherited-members:
