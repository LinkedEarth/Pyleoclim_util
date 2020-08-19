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
   mapping/map_all.rst
   mapping/set_proj.rst
   filter/savitzky_golay.rst
   filter/ts_pad.rst
   filter/butterworth.rst
   decomposition/pca.rst
   decomposition/mssa.rst
   decomposition/ssa.rst
   correlation/corr_sig.rst
   correlation/fdr.rst

Pyleoclim Utilities (pyleoclim.utils)
=====================================

Pyleoclim makes extensive use of functions from `Numpy <https://numpy.org>`_, `Pandas <https://pandas.pydata.org>`_, `Scipy <https://www.scipy.org>`_, and `scikit-learn <https://scikit-learn.org/stable/>`_. Please note that some default parameter values for these functions have been changed to more appropriate values for paleoclimate datasets.

Causality
---------
.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :ref:`granger_causality <utils_granger>`
     - Estimate Granger causality
   * - :ref:`liang_causality <utils_liang>`
     - Estimate Liang causality

Correlation
-----------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :ref:`corr_sig <utils_corr_sig>`
     - Estimates the Pearson's correlation and associated significance between two non IID time series.
   * - :ref:`fdr <utils_fdr>`
     - False Discovery Rate


Decomposition
-------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :ref:`pca <utils_pca>`
     - Principal Component Analysis
   * - :ref:`ssa <utils_ssa>`
     - Singular Spectrum Analysis
   * - :ref:`MSSA <utils_mssa>`
     - Multi Channel Singular Spectrum Analysis.

Filter
------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :ref:`Savitzky-Golay filter <utils_savitzky_golay>`
     - Smooth (and optionally differentiate) data with a Savitzky-Golay filter
   * - :ref:`Butterworth filter <utils_butterworth>`
     - Applies a Butterworth filter with frequency fc, with optional padding

Mapping
-------
.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :ref:`map <utils_mapall>`
     - Maps records according to some criteria (e.g, proxy type, interpretation)

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
This module contains functions to manipulate LiPD files and automate data transformation whenever possible.
