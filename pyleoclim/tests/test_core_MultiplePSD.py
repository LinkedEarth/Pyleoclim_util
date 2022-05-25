''' Tests for pyleoclim.core.ui.MultiplePSD

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
'''
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

import pytest

import pyleoclim as pyleo
from pyleoclim.utils.tsmodel import (
    ar1_sim,
    colored_noise,
)

# a collection of useful functions

def gen_normal(loc=0, scale=1, nt=100):
    ''' Generate random data with a Gaussian distribution
    '''
    t = np.arange(nt)
    v = np.random.normal(loc=loc, scale=scale, size=nt)
    return t, v

def gen_colored_noise(alpha=1, nt=100, f0=None, m=None, seed=None):
    ''' Generate colored noise
    '''
    t = np.arange(nt)
    v = colored_noise(alpha=alpha, t=t, f0=f0, m=m, seed=seed)
    return t, v


# Tests below
class TestUiMultiplePsdBetaEst:
    ''' Tests for MultiplePSD.beta_est()
    '''

    def test_beta_est_t0(self, eps=0.3):
        ''' Test MultiplePSD.beta_est() of a list of colored noise
        '''
        alphas = np.arange(0.5, 1.5, 0.1)
        t, v = {}, {}
        series_list = []
        for idx, alpha in enumerate(alphas):
            t[idx], v[idx] = gen_colored_noise(nt=1000, alpha=alpha)
            series_list.append(pyleo.Series(time=t[idx], value=v[idx]))

        ts_surrs = pyleo.MultipleSeries(series_list=series_list)
        psds = ts_surrs.spectral(method='mtm')
        betas = psds.beta_est().beta_est_res['beta']
        for idx, beta in enumerate(betas):
            assert np.abs(beta-alphas[idx]) < eps

class TestUiMultiplePsdPlot:
    ''' Tests for MultiplePSD.plot()
    '''

    def test_plot_t0(self):
        ''' Test MultiplePSD.plot() of a list of colored noise
        '''
        alphas = np.arange(0.5, 1.5, 0.1)
        t, v = {}, {}
        series_list = []
        for idx, alpha in enumerate(alphas):
            t[idx], v[idx] = gen_colored_noise(nt=1000, alpha=alpha)
            series_list.append(pyleo.Series(time=t[idx], value=v[idx]))

        ts_surrs = pyleo.MultipleSeries(series_list=series_list)
        psds = ts_surrs.spectral(method='mtm')
        fig, ax = psds.plot(mute=True)


class TestUiMultiplePsdPlotEnvelope:
    ''' Tests for MultiplePSD.plot()
    '''

    def test_plot_envelope_t0(self):
        ''' Test MultiplePSD.plot() of a list of colored noise
        '''
        alphas = np.arange(0.5, 1.5, 0.1)
        t, v = {}, {}
        series_list = []
        for idx, alpha in enumerate(alphas):
            t[idx], v[idx] = gen_colored_noise(nt=1000, alpha=alpha)
            series_list.append(pyleo.Series(time=t[idx], value=v[idx]))

        ts_surrs = pyleo.MultipleSeries(series_list=series_list)
        psds = ts_surrs.spectral(method='mtm')
        fig, ax = psds.plot_envelope(mute=True)