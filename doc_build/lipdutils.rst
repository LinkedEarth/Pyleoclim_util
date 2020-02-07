LiPD Utilities
==============

This modules allow basic manipulation of LiPD files

Creating Directories and saving
```````````````````````````````

.. autofunction:: pyleoclim.lipdutils.createDir

.. autofunction:: pyleoclim.lipdutils.saveFigure

LiPD files
``````````

.. autofunction:: pyleoclim.lipdutils.enumerateLipds

.. autofunction:: pyleoclim.lipdutils.getLipd

Handling Variables
``````````````````

.. autofunction:: pyleoclim.lipdutils.promptForVariable

.. autofunction:: pyleoclim.lipdutils.xAxisTs

.. autofunction:: pyleoclim.lipdutils.checkXaxis

.. autofunction:: pyleoclim.lipdutils.checkTimeAxis

.. autofunction:: pyleoclim.lipdutils.searchVar

Handling timeseries objects
```````````````````````````

.. autofunction:: pyleoclim.lipdutils.enumerateTs

.. autofunction:: pyleoclim.lipdutils.getTs

Linking LiPDs to the LinkedEarth Ontology
`````````````````````````````````````````

.. autofunction:: pyleoclim.lipdutils.LipdToOntology

.. autofunction:: pyleoclim.lipdutils.timeUnitsCheck

Dealing with models
```````````````````

.. autofunction:: pyleoclim.lipdutils.isModel

.. autofunction:: pyleoclim.lipdutils.modelNumber

Extracting tables
`````````````````

.. autofunction:: pyleoclim.lipdutils.isMeasurement

.. autofunction:: pyleoclim.lipdutils.whichMeasurement

.. autofunction:: pyleoclim.lipdutils.getMeasurement

Dealing with ensembles
``````````````````````

.. autofunction:: pyleoclim.lipdutils.isEnsemble

.. autofunction:: pyleocli.lipdutils.getEnsembleValues

.. autofunction:: pyleoclim.lipdutils.mapAgeEnsembleToPaleoData
