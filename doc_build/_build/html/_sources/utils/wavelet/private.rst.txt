.. _wavelet_private:

.. toctree::
   :hidden:
   AliasFilter.rst
   assertPositiveInt.rst
   wwz_basic.rst
   wwz_nproc.rst
   kirchner_basic.rst
   kirchner_nproc.rst
   kirchner_numba.rst
   kirchner_f2py.rst
   make_coi.rst
   make_omega.rst
   wwa2psd.rst
   freq_vector_lomb_scargle.rst
   freq_vector_welch.rst
   freq_vector_nfft.rst
   freq_vector_scale.rst
   freq_vector_log.rst
   make_freq_vector.rst
   beta_estimation.rst
   beta2HurstIndex.rst
   psd_ar.rst
   fBMsim.rst
   psd_fBm.rst
   get_wwz_func.rst
   prepare_wwz.rst
   cross_wt.rst
   wavelet_coherence.rst
   reconstruct_ts.rst

Wavelet and spectral preprocessing functions (pyleoclim.utils.wavelet)
======================================================================
.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :ref:`AliasFilter <wavelet_AliasFilter>`
     - Performs anti-alis filter on a PSD. Experimental
   * - :ref:`assertPositiveInt <wavelet_assertPositiveInt>`
     - Asserts that the arguments are all positive integers
   * - :ref:`wwz_basic <wavelet_wwz_basic>`
     - Returns the weighted wavelet amplitude estimated from the original Foster method. No multiprocessing
   * - :ref:`wwz_nproc <wavelet_wwz_nproc>`
     - Returns the weighted wavelet amplitude using the original method from Kirchner. Supports multiprocessing
   * - :ref:`kirchner_basic <wavelet_kirchner_basic>`
     - Return the weighted wavelet amplitude (WWA) modified by Kirchner. No multiprocessing
   * - :ref:`kirchner_nproc <wavelet_kirchner_nproc>`
     - Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Supports multiprocessing
   * - :ref:`kirchner_numba <wavelet_kirchner_numba>`
     - Return the weighted wavelet amplitude (WWA) modified by Kirchner using Numba package.
   * - :ref:`kirchner_f2py <wavelet_kirchner_f2py>`
     - Returns the weighted wavelet amplitude (WWA) modified by Kirchner. Uses Fortran. Fastest method but requires a compiler.
   * - :ref:`make_coi <wavelet_make_coi>`
     - Return the cone of influence.
   * - :ref:`make_omega <wavelet_make_omega>`
     - Return the angular frequency based on the time axis and given frequency vector
   * - :ref:`wwa2psd <wavelet_wwa2psd>`
     - Return the power spectral density (PSD) using the weighted wavelet amplitude (WWA).
   * - :ref:`freq_vector_lomb_scargle <wavelet_freq_vector_lomb_scargle>`
     - Return the frequency vector based on the REDFIT recommendation.
   * - :ref:`freq_vector_welch <wavelet_freq_vector_welch>`
     - Return the frequency vector based on the Welch's method.
   * - :ref:`freq_vector_nfft <wavelet_freq_vector_nfft>`
     - Return the frequency vector based on NFFT
   * - :ref:`freq_vector_scale <wavelet_freq_vector_scale>`
     - Return the frequency vector based on scales
   * - :ref:`freq_vector_log <wavelet_freq_vector_log>`
     - Return the frequency vector based on logspace
   * - :ref:`make_freq_vector <wavelet_make_freq_vector>`
     - Make frequency vector
   * - :ref:`beta_estimation <wavelet_beta_estimation>`
     - Estimate the power slope of a 1/f^beta process.
   * - :ref:`beta2HurstIndex <wavelet_beta2HurstIndex>`
     - Translate psd slope to Hurst index
   * - :ref:`psd_ar <wavelet_psd_ar>`
     - Return the theoretical power spectral density (PSD) of an autoregressive model
   * - :ref:`fBMsim <wavelet_fBMsim>`
     - Select an ensemble table from a list
   * - :ref:`psd_fBM <wavelet_psd_fBM>`
     - Return the theoretical psd of a fBM
   * - :ref:`get_wwz_func <wavelet_get_wwz_func>`
     - Return the wwz function to use.
   * - :ref:`prepare_wwz <wavelet_prepare_wwz>`
     - Return the truncated time series with NaNs deleted and estimate frequency vector and tau
   * - :ref:`cross_wt <wavelet_cross_wt>`
     - Return the cross wavelet transform.
   * - :ref:`wavelet_coherence <wavelet_wavelet_coherence>`
     - Return the cross wavelet coherence.
   * - :ref:`reconstruct_ts <wavelet_reconstruct_ts>`
     - Reconstruct the normalized time series from the wavelet coefficients.
