Main Functions
==============

Getting started
```````````````

Pyleoclim relies heavily on the concept of timeseries objects introduced in
`LiPD <http://www.clim-past.net/12/1093/2016/>`_ and implemented in the
`LiPD utilities <http://nickmckay.github.io/LiPD-utilities/>`_.

Briefly, timeseries objects are dictionaries containing the ChronData values and
PaleoData values as well as the metadata associated with the record. If one record
has three ProxyObservations (e.g., Mg/Ca, d18O, d13C) then it will have three timeseries
objects, one for each of the observations.

The LiPD utilities function lipd.extractTs() returns a list of dictionaries for
the selected LiPD files, which need to be passed to Pyleoclim along with the path
to the directory containing the LiPD files.

This is done through the function pyleoclim.openLiPDs:

.. autofunction:: pyleoclim.openLipds

Mapping
```````
.. autofunction:: pyleoclim.mapAll

.. autofunction:: pyleoclim.mapLipd

Plotting
````````
.. autofunction:: pyleoclim.plotTs

Summary Plots
-------------

Summary plots are a special categories of plots enabled by Pyleoclim.
They allow to plot specific information about a timeseries but are not customizable.

.. autofunction:: pyleoclim.basicSummary

Statistics
``````````

.. autofunction:: pyleoclim.statsTs

.. autofunction:: pyleoclim.binTs

.. autofunction:: pyleoclim.interpTs
