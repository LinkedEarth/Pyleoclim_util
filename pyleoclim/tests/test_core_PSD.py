"""Tests for pyleoclim.core.ui.PSD

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

import pytest

import pyleoclim as pyleo


# Tests below
class TestUiPsdPlot:
    """Tests for PSD.plot()"""

    def test_plot_t0(self, gen_ts):
        """Test PSD.plot() with default parameters"""
        ts = gen_ts()
        psd = ts.spectral(method="mtm")
        fig, ax = psd.plot()
        pyleo.closefig(fig)


class TestUiPsdSignifTest:
    """Tests for PSD.signif_test()"""

    @pytest.mark.parametrize("method", ["ar1sim", "uar1", "ar1asym", "CN"])
    def test_signif_test_t0(self, method, gen_ts):
        """Test PSD.signif_test() with default parameters"""
        ts = gen_ts()
        psd = ts.spectral(method="mtm")
        psd_signif = psd.signif_test(number=10, method=method)
        fig, ax = psd_signif.plot()
        pyleo.closefig(fig)
