"""Tests for pyleoclim.core.ui.MultiplePSD

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
"""

import numpy as np
import pytest

import pyleoclim as pyleo


# Tests below
class TestUiMultiplePsdBetaEst:
    """Tests for MultiplePSD.beta_est()"""

    def test_beta_est_t0(self, gen_ts, eps=0.3):
        """Test MultiplePSD.beta_est() of a list of colored noise"""
        alphas = np.arange(0.5, 1.5, 0.1)
        series_list = []
        for idx, alpha in enumerate(alphas):
            series_list.append(gen_ts(nt=1000, alpha=alpha))

        ts_surrs = pyleo.MultipleSeries(series_list=series_list)
        psds = ts_surrs.spectral(method="mtm")
        betas = psds.beta_est().beta_est_res["beta"]
        for idx, beta in enumerate(betas):
            assert np.abs(beta - alphas[idx]) < eps


class TestUiMultiplePsdPlot:
    """Tests for MultiplePSD.plot()"""

    def test_plot_t0(self, gen_ts):
        """Test MultiplePSD.plot() of a list of colored noise"""
        alphas = np.arange(0.5, 1.5, 0.1)
        series_list = []
        for idx, alpha in enumerate(alphas):
            series_list.append(gen_ts(nt=1000, alpha=alpha))

        ts_surrs = pyleo.MultipleSeries(series_list=series_list)
        psds = ts_surrs.spectral(method="mtm")
        fig, ax = psds.plot()
        pyleo.closefig(fig)


class TestUiMultiplePsdPlotEnvelope:
    """Tests for MultiplePSD.plot()"""

    def test_plot_envelope_t0(self, gen_ts):
        """Test MultiplePSD.plot() of a list of colored noise"""
        alphas = np.arange(0.5, 1.5, 0.1)
        series_list = []
        for idx, alpha in enumerate(alphas):
            series_list.append(gen_ts(nt=1000, alpha=alpha))

        ts_surrs = pyleo.MultipleSeries(series_list=series_list)
        psds = ts_surrs.spectral(method="mtm")
        fig, ax = psds.plot_envelope()
        pyleo.closefig(fig)


class TestUiMultiplePsdPlotTraces:
    """Tests for MultiplePSD.plot_traces()"""

    def test_plot_traces_t0(self, multipleseries_science):
        """Test MultiplePSD.plot_traces() of a list of colored noise"""
        ts_surrs = multipleseries_science
        psds = ts_surrs.spectral()
        fig, _ = psds.plot_traces()
        pyleo.closefig(fig)

    def test_plot_traces_t1(self, ensembleseries_science):
        """Test MultiplePSD.plot_traces() with different numbers of traces"""

        ts_surrs = ensembleseries_science
        psds = ts_surrs.spectral()
        fig, _ = psds.plot_traces(num_traces=5)
        pyleo.closefig(fig)
