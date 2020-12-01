.. _core_ui:

.. currentmodules:: pyleoclim.core.ui

Using the Pyleoclim UI
======================

Series (pyleoclim.Series)
"""""""""""""""""""""""""

The Series class allows to manipulate Series objects, which are basic representations of timeseries, containing a time axis, values, and some basic metadata about both axes.

.. autoclass:: pyleoclim.core.ui.Series

MultipleSeries (pyleoclim.MultipleSeries)
"""""""""""""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.MultipleSeries

EnsembleSeries (pyleoclim.EnsembleSeries)
"""""""""""""""""""""""""""""""""""""""""

The EnsembleSeries class is a child of MultipleSeries, particularly designed for ensemble case.

.. autoclass:: pyleoclim.core.ui.EnsembleSeries
   :inherited-members:

Lipd (pyleoclim.Lipd)
"""""""""""""""""""""

This class allows to manipulate LiPD objects

.. autoclass:: pyleoclim.core.ui.Lipd

LipdSeries (pyleoclim.LipdSeries)
"""""""""""""""""""""""""""""""""

This class allows to manipulate LiPD timeseries objects. This is a subclass of Series. See applicable methods there.

.. autoclass:: pyleoclim.core.ui.LipdSeries

PSD (pyleoclim.PSD)
"""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.PSD

Scalogram (pyleoclim.Scalogram)
"""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.Scalogram

Coherence (pyleoclim.Coherence)
"""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.Coherence

SurrogateSeries (pyleoclim.SurrogateSeries)
"""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.SurrogateSeries

MultiplePSD (pyleoclim.MultiplePSD)
"""""""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.MultiplePSD

MultipleScalogram (pyleoclim.MultipleScalogram)
"""""""""""""""""""""""""""""""""""""""""""""""

.. autoclass:: pyleoclim.core.ui.MultipleScalogram
