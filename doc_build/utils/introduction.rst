.. _utils_introduction:

Pyleoclim Utilities API (pyleoclim.utils)
=========================================

Pyleoclim makes extensive use of functions from `numpy <https://numpy.org>`_, `Pandas <https://pandas.pydata.org>`_, `Scipy <https://www.scipy.org>`_, and `scikit-learn <https://scikit-learn.org/stable/>`_. Please note that some default parameter values for these functions have been changed to more appropriate values for paleoclimate datasets.

Causality
"""""""""
.. automodule:: pyleoclim.utils.causality
   :members: liang_causality, granger_causality

Correlation
"""""""""""

.. automodule:: pyleoclim.utils.correlation
   :members: fdr, corr_sig

Decomposition
"""""""""""""

.. automodule:: pyleoclim.utils.decomposition
   :members: ssa

Filter
""""""

.. automodule:: pyleoclim.utils.filter
   :members: butterworth, savitzky_golay, firwin, lanczos

Mapping
"""""""

.. automodule:: pyleoclim.utils.mapping
   :members: map, compute_distance

Plotting
""""""""

.. automodule:: pyleoclim.utils.plotting
   :members: set_style, closefig, savefig

Spectral
""""""""

.. automodule:: pyleoclim.utils.spectral
   :members: wwz_psd, cwt_psd, mtm, lomb_scargle, welch, periodogram

Tsmodel
"""""""

.. automodule:: pyleoclim.utils.tsmodel
   :members: ar1_sim, ar1_fit, colored_noise, colored_noise_2regimes, gen_ar1_evenly


Wavelet
"""""""

.. automodule:: pyleoclim.utils.wavelet
   :members: cwt, cwt_coherence, wwz, wwz_coherence


Tsutils
"""""""

.. automodule:: pyleoclim.utils.tsutils
   :members: simple_stats, bin, interp, gkernel, standardize, ts2segments, annualize, gaussianize, gaussianize_1d, detrend, remove_outliers


Tsbase
""""""

.. automodule:: pyleoclim.utils.tsbase
   :members: clean_ts, dropna, sort_ts, is_evenly_spaced, reduce_duplicated_timestampsers


Lipdutils
"""""""""

Utilities to manipulate LiPD files and automate data transformation whenever possible.
These functions are used throughout Pyleoclim but are not meant for direct interaction by users.
Also handles integration with the LinkedEarth wiki and the LinkedEarth Ontology.


jsonutils
"""""""""

.. automodule:: pyleoclim.utils.jsonutils
  :members: PyleoObj_to_json, json_to_PyleoObj, isPyleoclim
