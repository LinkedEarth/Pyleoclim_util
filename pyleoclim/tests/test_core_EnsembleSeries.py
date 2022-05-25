''' Tests for pyleoclim.core.ui.EnsembleSeries

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
from pyleoclim.tests.examples import load_dataset

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
class TestUIEnsembleSeriesCorrelation():
    def test_correlation_t0(self):
        '''Test for EnsembleSeries.correlation() when the target is a Series
        '''
        nt = 100
        t0, v0 = gen_colored_noise(nt=nt)
        t0, noise = gen_normal(nt=nt)

        ts0 = pyleo.Series(time=t0, value=v0)
        ts1 = pyleo.Series(time=t0, value=v0+noise)
        ts2 = pyleo.Series(time=t0, value=v0+2*noise)

        ts_list = [ts1, ts2]

        ts_ens = pyleo.EnsembleSeries(ts_list)

        corr_res = ts_ens.correlation(ts0)
        signif_list = corr_res.signif
        for signif in signif_list:
            assert signif is True


    def test_correlation_t1(self):
        '''Test for EnsembleSeries.correlation() when the target is an EnsembleSeries with same number of Series
        '''
        nt = 100
        t0, v0 = gen_colored_noise(nt=nt)
        t0, noise = gen_normal(nt=nt)

        ts0 = pyleo.Series(time=t0, value=v0)
        ts1 = pyleo.Series(time=t0, value=v0+noise)
        ts2 = pyleo.Series(time=t0, value=v0+2*noise)
        ts3 = pyleo.Series(time=t0, value=v0+1/2*noise)

        ts_list1 = [ts0, ts1]
        ts_list2 = [ts2, ts3]

        ts_ens = pyleo.EnsembleSeries(ts_list1)
        ts_target = pyleo.EnsembleSeries(ts_list2)

        corr_res = ts_ens.correlation(ts_target)
        signif_list = corr_res.signif
        for signif in signif_list:
            assert signif is True

        assert np.size(corr_res.p) == np.size(ts_list1)


    def test_correlation_t2(self):
        '''Test for EnsembleSeries.correlation() when the target is an EnsembleSeries with fewer Series
        '''
        nt = 100
        t0, v0 = gen_colored_noise(nt=nt)
        t0, noise = gen_normal(nt=nt)

        ts0 = pyleo.Series(time=t0, value=v0)
        ts1 = pyleo.Series(time=t0, value=v0+noise)
        ts2 = pyleo.Series(time=t0, value=v0+2*noise)
        ts3 = pyleo.Series(time=t0, value=v0+1/2*noise)
        ts4 = pyleo.Series(time=t0, value=v0+3/2*noise)

        ts_list1 = [ts0, ts1, ts4]
        ts_list2 = [ts2, ts3]

        ts_ens = pyleo.EnsembleSeries(ts_list1)
        ts_target = pyleo.EnsembleSeries(ts_list2)

        corr_res = ts_ens.correlation(ts_target)
        signif_list = corr_res.signif
        for signif in signif_list:
            assert signif is True

        assert np.size(corr_res.p) == np.size(ts_list1)


    def test_correlation_t3(self):
        '''Test for EnsembleSeries.correlation() when the target is an EnsembleSeries with more Series
        '''
        nt = 100
        t0, v0 = gen_colored_noise(nt=nt)
        t0, noise = gen_normal(nt=nt)

        ts0 = pyleo.Series(time=t0, value=v0)
        ts1 = pyleo.Series(time=t0, value=v0+noise)
        ts2 = pyleo.Series(time=t0, value=v0+2*noise)
        ts3 = pyleo.Series(time=t0, value=v0+1/2*noise)
        ts4 = pyleo.Series(time=t0, value=v0+3/2*noise)

        ts_list1 = [ts0, ts1]
        ts_list2 = [ts2, ts3, ts4]

        ts_ens = pyleo.EnsembleSeries(ts_list1)
        ts_target = pyleo.EnsembleSeries(ts_list2)

        corr_res = ts_ens.correlation(ts_target)
        signif_list = corr_res.signif
        for signif in signif_list:
            assert signif is True

        assert np.size(corr_res.p) == np.size(ts_list1)

    def test_plot_envelope_t0(self):
        ''' Test EnsembleSeries.plot_envelope() on a list of colored noise
        '''
        nn = 30 # number of noise realizations
        nt = 500
        series_list = []

        signal = pyleo.gen_ts(model='colored_noise',nt=nt,alpha=1.0).standardize() 
        noise = np.random.randn(nt,nn)

        for idx in range(nn):  # noise
            ts = pyleo.Series(time=signal.time, value=signal.value+noise[:,idx])
            series_list.append(ts)

        ts_ens = pyleo.EnsembleSeries(series_list)

        fig, ax = ts_ens.plot_envelope(curve_lw=1.5, mute=True) 

    def test_plot_traces_t0(self):
        ''' Test EnsembleSeries.plot_traces() on a list of colored noise
        '''
        nn = 30 # number of noise realizations
        nt = 500
        series_list = []

        signal = pyleo.gen_ts(model='colored_noise',nt=nt,alpha=1.0).standardize() 
        noise = np.random.randn(nt,nn)

        for idx in range(nn):  # noise
            ts = pyleo.Series(time=signal.time, value=signal.value+noise[:,idx])
            series_list.append(ts)

        ts_ens = pyleo.EnsembleSeries(series_list)

        fig, ax = ts_ens.plot_traces(alpha=0.2,num_traces=8, mute=True) 
