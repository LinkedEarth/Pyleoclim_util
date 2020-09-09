.. _lipdutils_private:

.. toctree::
   :hidden:
   checkTimeAxis.rst
   checkXaxis.rst
   enumerateLipds.rst
   enumerateTs.rst
   gen_dict_extract.rst
   getEnsemble.rst
   getLipd.rst
   getMeasurement.rst
   getTs.rst
   isEnsemble.rst
   isMeasurement.rst
   isModel.rst
   LipdToOntology.rst
   pre_process_list.rst
   pre_process_str.rst
   promptForVariable.rst
   searchVar.rst
   similar_string.rst
   timeUnitsCheck.rst
   whichEnsemble.rst
   whichMeasurement.rst
   xAxisTs.rst
   
LiPD Utilities (pyleoclim.utils.lipdutils)
==========================================

Pyleoclim is meant to directly handle datasets in the Linked Paleo Data (`LiPD <http://lipd.net>`_) format. Pyleoclim works with timeseries objects and model tables present in these files.

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :ref:`enumerateLipds <lipdutils_enumerateLipds>`
     - enumerate the LiPD files loaded in the workspace
   * - :ref:`getLipd <lipdutils_getLipd>`
     - Prompt for a LiPD files
   * - :ref:`promptForVariable <lipdutils_promptForVariable>`
     - Prompt for a specific variable
   * - :ref:`xAxisTs <lipdutils_xAxisTs>`
     - Get the x-axis for the timeseries
   * - :ref:`checkXaxis <lipdutils_checkXaxis>`
     - Check that a x-axis is present for the timeseries
   * - :ref:`checkTimeAxis <lipdutils_checkTimeAxis>`
     - This function makes sure that time is available for the timeseries
   * - :ref:`searchVar <lipdutils_searchVar>`
     - This functions searches for keywords (exact match) for a variable.
   * - :ref:`enumerateTs <lipdutils_enumerateTs>`
     - Enumerate the available time series objects
   * - :ref:`getTs <lipdutils_getTs>`
     - Get a specific timeseries object from a dictionary of timeseries
   * - :ref:`LipdToOntology <lipdutils_LipdToOntology`
     - Standardize archiveType
   * - :ref:`timeUnitsCheck <lipdutils_timeUnitsCheck>`
     - This function attempts to make sense of the time units by checking for equivalence
   * - :ref:`pre_process_list <lipdutils_preprocesslist>`
     - Pre-process a series of strings for capitalized letters, space, and punctuation.
   * - :ref:`similar_string <lipdutils_similar_string>`
     - Returns a list of indices for strings with similar values
   * - :ref:`pre_process_str <lipdutils_pre_process_str>`
     - Pre-process a string for capitalized letters, space, and punctuation
   * - :ref:`isModel <lipdutils_isModel>`
     - Check for the presence of a model in the same object as the measurement table
   * - :ref:`modelNumber <lipdutils_modelNumber>`
     - Assign a new or existing model number
   * - :ref:`isMeasurement <lipdutils_isMeasurement>`
     - Check whether measurement tables are available
   * - :ref:`whichMeasurement <lipdutils_whichMeasurement>`
     - Select a measurement table from a list
   * - :ref:`getMeasurement <lipdutils_getMeasurement>`
     - Extract the dictionary corresponding to the measurement table
   * - :ref:`isEnsemble <lipdutils_isEnsemble>`
     - Check whether ensembles are available
   * - :ref:`whichEnsemble <lipdutils_whichEnsemble>`
     - Select an ensemble table from a list
   * - :ref:`getEnsemble <lipdutils_getEnsemble>`
     - Extracts the ensemble values and depth vector from the dictionary
   * - :ref:`mapAgeEnsembleToPaleoData <lipdutils_mapAgeEnsembleToPaleoData>`
     - Map the depth for the ensemble age values to the paleo depth
   * - :ref:`gen_dict_extract <lipdutils_gen_dict_extract>`
     - Recursively searches for all the values in nested dictionaries corresponding to a particular key
