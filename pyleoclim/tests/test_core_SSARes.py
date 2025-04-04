"""Tests for pyleoclim.core.ui.SSARes

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

# import pandas as pd

import numpy as np
import pytest

import pyleoclim as pyleo

# a collection of useful functions

# Tests below


class TestUiSSAResScreeplot:
    """Tests for SSARes.screeplot()"""

    def test_plot_t0(self, gen_ts):
        """Test SSARes.screeplot with non-default parameters
        (default is already taken care of in TestUISeriesSsa)
        """
        nt = 500
        cn = gen_ts(model="colored_noise", nt=nt, alpha=1.0)

        cn_ssa = cn.ssa()
        fig, ax = cn_ssa.screeplot(title="Non default title")
        pyleo.closefig(fig)

    def test_plot_t1(self, gen_ts):
        """Test SSARes.screeplot with MC-SSA"""
        nt = 500
        cn = gen_ts(model="colored_noise", nt=nt, alpha=1.0)

        cn_ssa = cn.ssa(nMC=200)
        fig, ax = cn_ssa.screeplot(title="MC-SSA scree plot")
        pyleo.closefig(fig)


class TestUiSSAResModeplot:
    """Tests for SSARes.modeplot()"""

    @pytest.mark.parametrize("spec_method", ["mtm", "welch", "periodogram"])
    def test_plot_t0(self, gen_ts, spec_method):
        """Test SSARes.modeplot with 3 spectral methods"""
        nt = 500
        cn = gen_ts(model="colored_noise", nt=nt, alpha=1.0)

        cn_ssa = cn.ssa()
        fig, ax = cn_ssa.modeplot(spec_method=spec_method)
        pyleo.closefig(fig)

    def test_plot_t1(self, gen_ts):
        """Test SSARes.modeplot with nondefault mode index"""
        nt = 500
        cn = gen_ts(model="colored_noise", nt=nt, alpha=1.0)

        cn_ssa = cn.ssa()
        fig, ax = cn_ssa.modeplot(index=4)
        pyleo.closefig(fig)
