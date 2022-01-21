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

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

import pytest
from urllib.request import urlopen
import json

import pyleoclim as pyleo
from pyleoclim.utils.tsmodel import (
    ar1_sim,
    colored_noise,
)
from pyleoclim.utils.decomposition import mcpca

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
    
def load_data():
    #Loads stott MD982176 record
    try:
        d = pyleo.Lipd(usr_path='http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004')
    except:
        d = pyleo.Lipd('./example_data/MD982176.Stott.2004.lpd')
    return d

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
    '''Tests for MultipleSeries.pca()

    Testing the PCA function 
    '''

    def test_pca_t0(self):
        '''
        Test with synthetic data, no missing values, screeplot()

        Returns
        -------
        None.

        '''
        p = 10; n = 100
        signal = pyleo.gen_ts(model='colored_noise',nt=n,alpha=1.0).standardize() 
        X = signal.value[:,None] + np.random.randn(n,p)
        t = np.arange(n)
    
        mslist = []
        for i in range(p):
            mslist.append(pyleo.Series(time = t, value = X[:,i]))
        ms = pyleo.MultipleSeries(mslist)

        res = ms.pca()
        
        # check that all variance was recovered
        assert abs(res.pctvar.sum() - 100)<0.1 
        
    
    def test_pca_t1(self):
        '''
        Test with synthetic data, with missing values

        '''
        p = 10; n = 100
        signal = pyleo.gen_ts(model='colored_noise',nt=n,alpha=1.0).standardize() 
        X = signal.value[:,None] + np.random.randn(n,p)
        t = np.arange(n)
        
        # poke some holes at random in the array
        Xflat = X.flatten()
        Xflat[np.random.randint(n*p, size=p-1)]=np.nan  # note: at most ncomp missing vals
        X = np.reshape(Xflat, (n,p))
    
        #X[-1,0] = np.nan
    
        mslist = []
        for i in range(p):
            mslist.append(pyleo.Series(time = t, value = X[:,i],clean_ts=False))
        ms = pyleo.MultipleSeries(mslist)

        res = ms.pca(ncomp=4,gls=True)  
                
        fig, ax = res.screeplot(mute=True) 
        
    def test_pca_t2(self):
        '''
        Test with real data, same time axis
    
        ''' 
        d=load_data()
        tslist = d.to_LipdSeriesList()
        tslist = tslist[2:]
        ms = pyleo.MultipleSeries(tslist)
        msl = ms.common_time()  # put on common time
    
        res = msl.pca()
        
        res.screeplot(mute=True)
        res.modeplot(mute=True)
        
    def test_pca_t3(self):
        '''
        Test with synthetic data, no missing values, kwargs

        Returns
        -------
        None.

        '''
        p = 10; n = 100
        signal = pyleo.gen_ts(model='colored_noise',nt=n,alpha=1.0)
        X = signal.value[:,None] + np.random.randn(n,p)
        t = np.arange(n)
    
        mslist = []
        for i in range(p):
            mslist.append(pyleo.Series(time = t, value = X[:,i]))
        ms = pyleo.MultipleSeries(mslist)

        res = ms.pca(method='eig',standardize=True,demean=False,normalize=True)
        # check that all variance was recovered
        assert abs(res.pctvar.sum() - 100)<0.001 
        
 
class TestMultipleSeriesGridProperties:
    '''Test for MultipleSeries.grid_properties()
    
    '''
    @pytest.mark.parametrize('step_style', ['min', 'max', 'mean', 'median'])
    def test_grid_properties(self, step_style):
        p = 10; n = 100
        signal = pyleo.gen_ts(model='colored_noise',nt=n,alpha=1.0).standardize() 
        X = signal.value[:,None] + np.random.randn(n,p)
        t = np.arange(n)
    
        mslist = []
        for i in range(p):
            mslist.append(pyleo.Series(time = t, value = X[:,i]))
        ms = pyleo.MultipleSeries(mslist)
        
        gp = ms.grid_properties(step_style=step_style)
        
        assert (gp[0,:] == np.array((t.min(), t.max(), 1.))).all()
               
# class TestMultipleSeriesMcPca:
#     '''Test for MultipleSeries.mcpca()

#     Testing the MC-PCA function 
#     '''    
#     def test_mcpca_t0(self):
#         url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
#         data = pyleo.Lipd(usr_path = url)
#         tslist = data.to_LipdSeriesList()
#         tslist = tslist[2:] # drop the first two series which only concerns age and depth
#         ms = pyleo.MultipleSeries(tslist)
    
#         # TO DO !!!!
    
#         # msc = ms.common_time()
    
#         # res = msc.pca(nMC=20)   

    
class TestMultipleSeriesCommonTime:
    '''Test for MultipleSeries.common_time()
    '''
    @pytest.mark.parametrize('method', ['bin', 'interp', 'gkernel'])
    def test_common_time_t0(self, method):
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
        
    def test_common_time_t1(self):
        time = np.arange(1900, 2020, step=1/12)
        ndel = 200
        seriesList = []
        for j in range(4):
            v = pyleo.gen_ts(model='colored_noise',alpha=1, t=time)
            deleted_idx = np.random.choice(range(np.size(time)), ndel, replace=False)
            tu =  np.delete(time.copy(), deleted_idx)
            vu =  np.delete(v.value, deleted_idx)
            ts = pyleo.Series(time=tu, value=vu,  value_name='Series_'+str(j+1))
            seriesList.append(ts)
    
        ms = pyleo.MultipleSeries(seriesList)

        ms1 = ms.common_time(method='interp', start=1910, stop=2010, step=1/12)
        
        assert (np.diff(ms1.series_list[0].time)[0] - 1/12) < 1e-3
        
class TestMultipleSeriesStackPlot():
    ''' Test for MultipleSeries.Stackplot
    '''
    
    @pytest.mark.parametrize('labels', [None, 'auto', ['sst','d18Osw']])
    def test_StackPlot_t0(self, labels):
    
        d=load_data()
        sst = d.to_LipdSeries(number=5)
        d18Osw = d.to_LipdSeries(number=3)
        ms = pyleo.MultipleSeries([sst,d18Osw])
        ms.stackplot(labels=labels, mute=True)
    
    @pytest.mark.parametrize('plot_kwargs', [{'marker':'o'},[{'marker':'o'},{'marker':'^'}]])
    def test_StackPlot_t1(self, plot_kwargs):
    
        d=load_data()
        sst = d.to_LipdSeries(number=5)
        d18Osw = d.to_LipdSeries(number=3)
        ms = pyleo.MultipleSeries([sst,d18Osw])
        ms.stackplot(plot_kwargs=plot_kwargs, mute=True)
        
class TestMultipleSeriesSpectral():
    ''' Test for MultipleSeries.spectral
    '''
    
    def test_spectral_t0(self):
        '''Test the spectral function with pre-generated scalogram objects
        '''
        
        d=load_data()
        sst = d.to_LipdSeries(number=5)
        d18Osw = d.to_LipdSeries(number=3)
        ms = pyleo.MultipleSeries([sst,d18Osw])
        scals = ms.wavelet()
        psds = ms.spectral(method='wwz',scalogram_list=scals)
        
        
        
        