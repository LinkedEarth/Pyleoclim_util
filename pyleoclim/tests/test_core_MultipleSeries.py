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
import datetime as dt
import numpy as np
import pandas as pd
import os

from numpy.testing import assert_array_equal, assert_allclose
from pandas.testing import assert_frame_equal

import pytest
from urllib.request import urlopen
import json

import pyleoclim as pyleo
from pyleoclim.utils.tsmodel import (
    ar1_sim,
    colored_noise,
)
# from pyleoclim.utils.decomposition import mcpca

# a collection of useful functions

def gen_ts(model, nt, alpha, t=None):
    'wrapper for gen_ts in pyleoclim'
    t, v = pyleo.utils.gen_ts(model=model, nt=nt, alpha=alpha, t=t)
    ts = pyleo.Series(t, v)
    return ts

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
        ts = pyleo.Series(time=t,value=v_trend)
        ts1 = pyleo.Series(time=t,value=v_trend1)

        # Create a multiple series object
        ts_all= pyleo.MultipleSeries([ts,ts1])
        ts_detrend=ts_all.detrend(method=detrend_method)     
        detrend_0 = ts_detrend.series_list[0]
        detrend_1 = ts_detrend.series_list[1]

        assert len(detrend_0.value)==len(detrend_0.time)
        assert len(detrend_1.value)==len(detrend_1.time)

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

        assert_allclose(t_0, x_plot_0)
        assert_allclose(t_1, x_plot_1)
        assert_allclose(v_0, y_plot_0)
        assert_allclose(v_1, y_plot_1)
        pyleo.closefig(fig)

class TestMultipleSeriesStripes:
    '''Test for MultipleSeries.stripes()

    '''

    def test_stripes(self):

        #Generate time and value arrays
        t_0, v_0 = gen_normal()
        t_1, v_1 = gen_normal()

        #Create series objects
        ts_0 = pyleo.Series(time = t_0, value = v_0, label='series 1')
        ts_1 = pyleo.Series(time = t_1, value = v_1, label='series 2')

        #Turn this list into a multiple series object
        ts_M = ts_0 &  ts_1

        fig, ax = ts_M.stripes(sat=0.9, label_color = 'red', cmap='cividis_r')
        pyleo.closefig(fig)

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
        signal = gen_ts(model='colored_noise', nt=n, alpha=1.0).standardize()
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
        signal = gen_ts(model='colored_noise', nt=n, alpha=1.0).standardize()
        X = signal.value[:,None] + np.random.randn(n,p)
        t = np.arange(n)
        
        # poke some holes at random in the array
        Xflat = X.flatten()
        Xflat[np.random.randint(n*p, size=p-1)]=np.nan  # note: at most ncomp missing vals
        X = np.reshape(Xflat, (n,p))
    
        #X[-1,0] = np.nan
    
        mslist = []
        for i in range(p):
            mslist.append(pyleo.Series(time = t, value = X[:,i],dropna=False))
        ms = pyleo.MultipleSeries(mslist)

        res = ms.pca(ncomp=4, gls=True)
                
        fig, ax = res.screeplot()
        pyleo.closefig(fig)
        
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
        
        fig, ax = res.screeplot()
        pyleo.closefig(fig)
        fig, ax = res.modeplot()
        pyleo.closefig(fig)
        
    def test_pca_t3(self):
        '''
        Test with synthetic data, no missing values, kwargs

        Returns
        -------
        None.

        '''
        p = 10; n = 100
        signal = gen_ts(model='colored_noise', nt=n, alpha=1.0)
        X = signal.value[:,None] + np.random.randn(n,p)
        t = np.arange(n)
    
        mslist = []
        for i in range(p):
            mslist.append(pyleo.Series(time = t, value = X[:,i]))
        ms = pyleo.MultipleSeries(mslist)

        res = ms.pca(method='eig',standardize=True,demean=False,normalize=True)
        # check that all variance was recovered
        assert abs(res.pctvar.sum() - 100)<0.001 
        
 
class TestMultipleSeriesIncrements:
    '''Test for MultipleSeries.increments()
    
    '''
    @pytest.mark.parametrize('step_style', ['min', 'max', 'mean', 'median'])
    def test_increments(self, step_style):
        p = 2; n = 100
        signal = gen_ts(model='colored_noise', nt=n, alpha=1.0).standardize()
        X = signal.value[:,None] + np.random.randn(n,p)
        t = np.arange(n)
    
        mslist = []
        for i in range(p):
            mslist.append(pyleo.Series(time = t, value = X[:,i]))
        ms = pyleo.MultipleSeries(mslist)
        
        gp = ms.increments(step_style=step_style)
        
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
        n = 100
        for j in range(4):
            v = gen_ts(model='colored_noise', nt=n, alpha=1, t=time)
            deleted_idx = np.random.choice(range(np.size(time)), ndel, replace=False)
            tu =  np.delete(time.copy(), deleted_idx)
            vu =  np.delete(v.value, deleted_idx)
            ts = pyleo.Series(time=tu, value=vu,  value_name='Series_'+str(j+1))
            seriesList.append(ts)
    
        ms = pyleo.MultipleSeries(seriesList)

        ms1 = ms.common_time(method='interp', start=1910, stop=2010, step=1/12)
        
        assert (np.diff(ms1.series_list[0].time)[0] - 1/12) < 1e-3

    def test_common_time_t2(self):
        time = np.arange(1900, 2020, step=1/12)
        ndel = 200
        seriesList = []
        n = 100
        for j in range(4):
            v = gen_ts(model='colored_noise', nt=n, alpha=1, t=time)
            deleted_idx = np.random.choice(range(np.size(time)), ndel, replace=False)
            tu =  np.delete(time.copy(), deleted_idx)
            vu =  np.delete(v.value, deleted_idx)
            ts = pyleo.Series(time=tu, value=vu,  value_name='Series_'+str(j+1))
            seriesList.append(ts)
    
        ms = pyleo.MultipleSeries(seriesList)

        new_time = np.arange(1950,2000,1)

        ms1 = ms.common_time(method='interp', time_axis = new_time)
        
        assert_array_equal(new_time,ms1.series_list[0].time)
        
class TestMultipleSeriesStackPlot():
    ''' Test for MultipleSeries.Stackplot
    '''
    
    @pytest.mark.parametrize('labels', [None, 'auto', ['sst','d18Osw']])
    def test_StackPlot_t0(self, labels):
    
        d=load_data()
        sst = d.to_LipdSeries(number=5)
        d18Osw = d.to_LipdSeries(number=3)
        ms = pyleo.MultipleSeries([sst,d18Osw])
        fig, ax = ms.stackplot(labels=labels)
        pyleo.closefig(fig)
    
    @pytest.mark.parametrize('plot_kwargs', [{'marker':'o'},[{'marker':'o'},{'marker':'^'}]])
    def test_StackPlot_t1(self, plot_kwargs):
    
        d=load_data()
        sst = d.to_LipdSeries(number=5)
        d18Osw = d.to_LipdSeries(number=3)
        ms = pyleo.MultipleSeries([sst,d18Osw])

        fig, ax = ms.stackplot(plot_kwargs=plot_kwargs)
        pyleo.closefig(fig)
        
class TestMultipleSeriesSpectral():
    ''' Test for MultipleSeries.spectral
    '''
    @pytest.mark.parametrize('spec_method',['wwz','cwt'])
    def test_spectral_t0(self,spec_method):
        '''Test the spectral function with pre-generated scalogram objects
        '''
        
        d = load_data()
        sst = d.to_LipdSeries(number=5)
        d18Osw = d.to_LipdSeries(number=3)
        if spec_method == 'cwt':
            sst = sst.interp()
            d18Osw = d18Osw.interp()
        ms = pyleo.MultipleSeries([sst,d18Osw])
        scals = ms.wavelet(method=spec_method)
        ms.spectral(method=spec_method,scalogram_list=scals)
 
class TestToCSV:
    def test_to_csv_default(self):
        soi = pyleo.utils.load_dataset('SOI')
        nino = pyleo.utils.load_dataset('NINO3')
        ms = soi & nino
        ms.to_csv()
        os.unlink('MultipleSeries.csv')
    def test_to_csv_label(self):
        soi = pyleo.utils.load_dataset('SOI')
        nino = pyleo.utils.load_dataset('NINO3')
        ms = soi & nino
        ms.label='enso series'
        ms.to_csv()
        os.unlink('enso_series.csv') # this check that the file does exist
    def test_to_csv_label_path(self):
        soi = pyleo.utils.load_dataset('SOI')
        nino = pyleo.utils.load_dataset('NINO3')
        ms = soi & nino
        ms.label='enso wah wah'
        ms.to_csv(path='./enso.csv')
        os.unlink('enso.csv') # this check that the file does exist
        
class TestRemove:
    def test_remove(self):
        ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='foo')
        ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='bar')
        ms = pyleo.MultipleSeries([ts1, ts2])
        ms.remove('bar')
        assert len(ms.series_list) == 1
        assert ms.series_list[0].equals(ts1) == (True, True)


class TestToPandas:
    def test_to_pandas_with_common_time(self): 
        ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='foo')
        ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='bar')
        ms = pyleo.MultipleSeries([ts1, ts2])
        result = ms.to_pandas(use_common_time=True)
        expected_index = pd.DatetimeIndex(
            np.array(['0000-12-31 05:48:45', '0002-07-02 02:31:54','0003-12-31 23:15:03'], dtype='datetime64[s]'),
            name='datetime',
        )
        expected = pd.DataFrame({'foo': [7, 5.25,9.00], 'bar': [7, 7.75,1.00]}, index=expected_index)
        pd.testing.assert_frame_equal(result, expected)

    def test_to_pandas_defau(self): 
        ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='foo')
        ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='bar')
        ms = pyleo.MultipleSeries([ts1, ts2])
        result = ms.to_pandas()
        expected_index = pd.DatetimeIndex(
            np.array(['0000-12-31 05:48:45', '0001-12-31 11:37:31', '0002-12-31 17:26:17', '0003-12-31 23:15:03'],
                     dtype='datetime64[s]'),
            name='datetime',
            freq='31556926S',
        )
        expected = pd.DataFrame({'foo': [7, 4, np.nan, 9], 'bar': [7, np.nan, 8, 1]}, index=expected_index)
        pd.testing.assert_frame_equal(result, expected)
        
    @pytest.mark.parametrize('paleo_style',[True,False])
    def test_to_pandas_args_kwargs(self, paleo_style):
        ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='foo',verbose=False)
        ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='bar',verbose=False)
        ms = pyleo.MultipleSeries([ts1, ts2])
        result = ms.to_pandas(paleo_style=paleo_style,method='bin', use_common_time=True, start=2)
        if paleo_style:
            expected_index = pd.Index([3.0], dtype='float64', name='time')
        else:
            expected_index = pd.DatetimeIndex(
                np.array(['0002-12-31 17:26:17'], dtype='datetime64[s]'),
                name='datetime',
            )
        expected = pd.DataFrame({'foo': [6.5], 'bar': [4.5]}, index=expected_index)
        pd.testing.assert_frame_equal(result, expected)

class TestOverloads:
    def test_add(self):
        ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='foo')
        ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='bar')
        ts3 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='baz')
        ms = pyleo.MultipleSeries([ts1, ts2])
        ms = ms + ts3
        assert ms.series_list[0].equals(ts1) == (True, True)
        assert ms.series_list[1].equals(ts2) == (True, True)
        assert ms.series_list[2].equals(ts3) == (True, True)

    def test_sub(self):
        ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='foo')
        ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='bar')
        ms = pyleo.MultipleSeries([ts1, ts2])
        ms - 'bar'
        assert len(ms.series_list) == 1
        assert ms.series_list[0].equals(ts1) == (True, True)


    def test_create_from_series(self):
        ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='foo')
        ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='bar')
        ts3 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='baz')
        ms_from_overloads = ts1 & ts2 & ts3
        ms_from_constructor = pyleo.MultipleSeries([ts1, ts2, ts3])
        for i, _ in enumerate(ms_from_constructor.series_list):
            assert ms_from_constructor.series_list[i].equals(ms_from_overloads.series_list[i]) == (True, True)

    def test_add_other_multiple_series(self):
        ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='sound')
        ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='the')
        ts3 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='alarm')
        ts4 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='climate')
        ts5 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='emergency')
        ms1 = pyleo.MultipleSeries([ts1, ts2, ts3])
        ms2 = pyleo.MultipleSeries([ts4, ts5])
        ms = ms1 + ms2
        assert ms.series_list[0].equals(ts1) == (True, True)
        assert ms.series_list[1].equals(ts2) == (True, True)
        assert ms.series_list[2].equals(ts3) == (True, True)
        assert ms.series_list[3].equals(ts4) == (True, True)
        assert ms.series_list[4].equals(ts5) == (True, True)
        
    def test_add_identical_series(self):
        ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='sound')
        ms = pyleo.MultipleSeries([ts1])
        with pytest.raises(ValueError, match='Given series is identical to existing series'):
            ms + ts1

class TestSel:
    
    @pytest.mark.parametrize(
        ('value', 'expected_time', 'expected_value', 'tolerance'),
        [
            (1, np.array([3]), np.array([1]), 0),
            (1, np.array([1, 3]), np.array([4, 1]), 3),
            (slice(1, 4), np.array([1, 3]), np.array([4, 1]), 0),
            (slice(1, 4), np.array([1, 2, 3]), np.array([4, 6, 1]), 2),
            (slice(1, None), np.array([1, 2, 3]), np.array([4, 6, 1]), 0),
            (slice(None, 1), np.array([3]), np.array([1]), 0),
        ])
    
    def test_value(self, value, expected_time, expected_value, tolerance):
    
        ts1 = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years BP', label='ts1')
        ts2 = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years BP', label='ts2')
        ts3 = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years BP', label='ts3')
        ms = pyleo.MultipleSeries([ts1, ts2, ts3])
        result = ms.sel(value=value, tolerance=tolerance)
        #check
        expected = pyleo.Series(time=expected_time, value=expected_value, time_unit='years BP')
        for item in result.series_list:
            values_match, _ = item.equals(expected)
            assert values_match
    
    @pytest.mark.parametrize(
        ('time', 'expected_time', 'expected_value', 'tolerance'),
        [
            (1, np.array([1]), np.array([4]), 0),
            (1, np.array([1, 2]), np.array([4, 6]), 1),
            (dt.datetime(1948, 1, 1), np.array([2, 3]), np.array([6, 1]), dt.timedelta(days=365)),
            ('1948', np.array([2, 3]), np.array([6, 1]), dt.timedelta(days=365)),
            (slice(1, 2), np.array([1, 2]), np.array([4, 6]), 0),
            (slice(1, 2), np.array([1, 2, 3]), np.array([4, 6, 1]), 1),
            (slice(1, None), np.array([1, 2, 3]), np.array([4, 6, 1]), 0),
            (slice(None, 1), np.array([1]), np.array([4]), 0),
            (slice('1948', '1949'), np.array([1, 2]), np.array([4, 6]), 0),
            (slice('1947', None), np.array([1, 2, 3]), np.array([4, 6, 1]), 0),
            (slice(None, '1948'), np.array([3]), np.array([1]), 0),
            (slice(dt.datetime(1948, 1, 1), dt.datetime(1949, 1, 1)), np.array([1, 2]), np.array([4, 6]), 0),
            (slice(dt.datetime(1947, 1, 1), None), np.array([1, 2, 3]), np.array([4, 6, 1]), 0),
            (slice(None, dt.datetime(1948, 1, 1)), np.array([3]), np.array([1]), 0),
            (slice(dt.datetime(1948, 1, 1), dt.datetime(1949, 1, 1)), np.array([1, 2, 3]), np.array([4, 6, 1]), dt.timedelta(days=365)),
            (slice(dt.datetime(1947, 1, 1), None), np.array([1, 2, 3]), np.array([4, 6, 1]), dt.timedelta(days=365)),
            (slice(None, dt.datetime(1948, 1, 1)), np.array([2, 3]), np.array([6, 1]), dt.timedelta(days=365)),
            (slice('1948', '1949'), np.array([1, 2, 3]), np.array([4, 6, 1]), dt.timedelta(days=365)),
            (slice('1947', None), np.array([1, 2, 3]), np.array([4, 6, 1]), dt.timedelta(days=365)),
            (slice(None, '1948'), np.array([2, 3]), np.array([6, 1]), dt.timedelta(days=365)),
        ]
    )
    def test_time(self, time, expected_time, expected_value, tolerance):
        ts1 = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years BP', label='ts1')
        ts2 = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years BP', label='ts2')
        ts3 = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years BP', label='ts3')
        ms = pyleo.MultipleSeries([ts1, ts2, ts3])
        result = ms.sel(time=time, tolerance=tolerance)
        expected = pyleo.Series(time=expected_time, value=expected_value, time_unit='years BP')
        for item in result.series_list:
            values_match, _ = item.equals(expected)
            assert values_match
    
    def test_invalid(self):
        ts1 = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years BP', label='ts1')
        ts2 = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years BP', label='ts2')
        ts3 = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years BP', label='ts3')
        ms = pyleo.MultipleSeries([ts1, ts2, ts3])
        with pytest.raises(TypeError, match="Cannot pass both `value` and `time`"):
            ms.sel(time=1, value=1)

    
