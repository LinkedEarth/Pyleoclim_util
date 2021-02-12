''' Tests for pyleoclim.core.ui.MultipleSeries

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
from statsmodels.multivariate.pca import PCA

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
class TestUIMultipleSeriesDetrend():
    @pytest.mark.parametrize('detrend_method',['linear','constant','savitzky-golay','emd'])
    def test_detrend_t1(self, detrend_method):
        alpha=1
        t, v = gen_colored_noise(nt=550, alpha=alpha)
        #Trends
        slope = 1e-5
        slope1= 2e-5
        intercept = -1
        nonlinear_trend = slope*t**2 + intercept
        nonlinear_trend1 = slope1*t**2 + intercept
        v_trend = v + nonlinear_trend
        v_trend1 = v + nonlinear_trend1

        #create series object
        ts=pyleo.Series(time=t,value=v_trend)
        ts1=pyleo.Series(time=t,value=v_trend1)

        # Create a multiple series object
        ts_all= pyleo.MultipleSeries([ts,ts1])
        ts_detrend=ts_all.detrend(method=detrend_method)

class TestMultipleSeriesPlot:
    '''Test for MultipleSeries.plot()

    MultipleSeries.plot outputs a matplotlib figure and axis object with two datasets,
    so we will compare the time axis of the axis object to the time arrays we generate,
    and the value axis with the value arrays we generate'''

    def test_plot(self):

        #Generate time and value arrays
        t_0, v_0 = gen_normal()
        t_1, v_1 = gen_normal()

        #Create series objects
        ts_0 = pyleo.Series(time = t_0, value = v_0)
        ts_1 = pyleo.Series(time = t_1, value = v_1)

        #Create a list of series objects
        serieslist = [ts_0, ts_1]

        #Turn this list into a multiple series object
        ts_M = pyleo.MultipleSeries(serieslist)

        fig, ax = ts_M.plot()

        lines_0 = ax.lines[0]
        lines_1 = ax.lines[1]

        x_plot_0 = lines_0.get_xdata()
        y_plot_0 = lines_0.get_ydata()

        x_plot_1 = lines_1.get_xdata()
        y_plot_1 = lines_1.get_ydata()

        assert_array_equal(t_0, x_plot_0)
        assert_array_equal(t_1, x_plot_1)
        assert_array_equal(v_0, y_plot_0)
        assert_array_equal(v_1, y_plot_1)


class TestMultipleSeriesStandardize:
    '''Test for MultipleSeries.standardize()

    Standardize normalizes the multiple series object, so we'll simply test maximum and minimum values,
    only now we are running the test on series in a MultipleSeries object'''

    def test_standardize(self):
        t_0, v_0 = gen_colored_noise()
        t_1, v_1 = gen_colored_noise()

        ts_0 = pyleo.Series(time = t_0, value = v_0)
        ts_1 = pyleo.Series(time = t_1, value = v_1)

        serieslist = [ts_0, ts_1]

        ts_M = pyleo.MultipleSeries(serieslist)

        ts_M_std = ts_M.standardize()

        x_axis_0 = ts_M_std.series_list[0].__dict__['time']
        x_axis_1 = ts_M_std.series_list[1].__dict__['time']

        y_axis_0 = ts_M_std.series_list[0].__dict__['value']
        y_axis_1 = ts_M_std.series_list[1].__dict__['value']

        assert_array_equal(x_axis_0, t_0)
        assert_array_equal(x_axis_1, t_1)

        assert max(v_0) > max(y_axis_0)
        assert max(v_1) > max(y_axis_1)

class TestMultipleSeriesBin:
    '''Test for MultipleSeries.bin()

    Testing if the bin function will place the series on the same time axis
    '''

    def test_bin(self):
        t_0, v_0 = gen_colored_noise()
        t_1, v_1 = gen_colored_noise()

        ts_0 = pyleo.Series(time = t_0, value = v_0)
        ts_1 = pyleo.Series(time = t_1, value = v_1)

        serieslist = [ts_0, ts_1]

        ts_M = pyleo.MultipleSeries(serieslist)

        ts_M_bin = ts_M.bin()

        x_axis_0 = ts_M_bin.series_list[0].__dict__['time']
        x_axis_1 = ts_M_bin.series_list[1].__dict__['time']

        assert_array_equal(x_axis_0, x_axis_1)

class TestMultipleSeriesInterp:
    '''Test for MultipleSeries.interp()

    Testing if the interp function will place the series on the same time axis
    '''

    def test_interp(self):
        t_0, v_0 = gen_colored_noise()
        t_1, v_1 = gen_colored_noise()

        ts_0 = pyleo.Series(time = t_0, value = v_0)
        ts_1 = pyleo.Series(time = t_1, value = v_1)

        serieslist = [ts_0, ts_1]

        ts_M = pyleo.MultipleSeries(serieslist)

        ts_M_interp = ts_M.interp()

        x_axis_0 = ts_M_interp.series_list[0].__dict__['time']
        x_axis_1 = ts_M_interp.series_list[1].__dict__['time']

        assert_array_equal(x_axis_0, x_axis_1)
    
class TestMultipleSeriesGkernel:
    '''Test for MultipleSeries.gkernel()

    Testing the gkernel function will place the series on the same time axis
    '''

    def test_gkernel(self):
        t_0, v_0 = gen_colored_noise()
        t_1, v_1 = gen_colored_noise()

        ts_0 = pyleo.Series(time = t_0, value = v_0)
        ts_1 = pyleo.Series(time = t_1, value = v_1)

        serieslist = [ts_0, ts_1]

        ts_M = pyleo.MultipleSeries(serieslist)

        ts_M_gkernel = ts_M.gkernel()

        x_axis_0 = ts_M_gkernel.series_list[0].__dict__['time']
        x_axis_1 = ts_M_gkernel.series_list[1].__dict__['time']

        assert_array_equal(x_axis_0, x_axis_1)    

class TestMultipleSeriesPca:
    '''Test for MultipleSeries.pca()

    Testing the PCA function 
    '''

    def test_pca(self):
        p = 20
        x = np.random.randn(100)[:, None]
        x = x + np.random.randn(100, p)
        pc = PCA(x, ncomp=p, missing= None)
    
        t = np.arange(100)
    
        mslist = []
        for i in range(p):
            mslist.append(pyleo.Series(time = t, value = x[:,i]))
        ms = pyleo.MultipleSeries(mslist)

        #res = ms.pca(nMC=20, missing='fill-em', standardize=False)
        res = ms.pca(nMC=20)
    
        # assert what?
        assert_array_equal(pc.eigenvals, res['eigvals'])    

    
class TestMultipleSeriesCommonTime:
    '''Test for MultipleSeries.common_time()
    '''
    @pytest.mark.parametrize('method', ['bin', 'interp', 'gkernel'])
    def test_common_time(self, method):
        t_0, v_0 = gen_colored_noise()
        t_1, v_1 = gen_colored_noise()

        ts_0 = pyleo.Series(time = t_0, value = v_0)
        ts_1 = pyleo.Series(time = t_1, value = v_1)

        serieslist = [ts_0, ts_1]

        ts_M = pyleo.MultipleSeries(serieslist)

        ts_M_ct = ts_M.common_time(method=method)

        x_axis_0 = ts_M_ct.series_list[0].time
        x_axis_1 = ts_M_ct.series_list[1].time

        assert_array_equal(x_axis_0, x_axis_1)