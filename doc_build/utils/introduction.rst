.. _utils_introduction:

.. toctree::
   :hidden:
   spectral/welch.rst
   spectral/periodogram.rst
   spectral/mtm.rst
   spectral/lombscargle.rst
   spectral/wwz_psd.rst
   causality/granger.rst
   causality/liang.rst

Pyleoclim Utilities (pyleoclim.utils)
=====================================

Pyleoclim makes extensive use of functions from `Numpy <https://numpy.org>`_, `Pandas <https://pandas.pydata.org>`_, `Scipy <https://www.scipy.org>`_, and `scikit-learn <https://scikit-learn.org/stable/>`_. Please note that some default parameter values for these functions have been changed to more appropriate values for paleoclimate datasets.

Causality
---------
.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :ref:`granger <utils_granger>`
     - Estimate Granger causality
   * - :ref:`liang <utils_liang>`
     - Estimate Liang causality

Correlation
-----------

Decomposition
-------------

Filter
------

Mapping
-------

Plotting
--------

Spectral
--------
.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :ref:`welch <utils_welch>`
     - Estimate power spectral density using Welch's method
   * - :ref:`periodogram <utils_periodogram>`
     - Estimate power spectral density using periodogram method
   * - :ref:`mtm <utils_mtm>`
     - Estimate power spectral density using multi-taper method
   * - :ref:`lomb_scargle <utils_lombscargle>`
     - Estimate power spectral density using the Lomb-Scargle method
   * - :ref:`wwz_psd <utils_wwzpsd>`
     - Estimate power spectral density using the Weighted Z-Transform wavelet method






Tsmodel
-------

Wavelet
-------

Tsutils
-------

Lipdutils
---------
