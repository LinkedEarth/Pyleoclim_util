''' Tests for pyleoclim.core.ui.Series

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

import pytest
#import scipy.io as sio
import sys#, json
#import os
import pathlib
test_dirpath = pathlib.Path(__file__).parent.absolute()

#from urllib.request import urlopen

import pyleoclim as pyleo
from pyleoclim.utils.tsmodel import (
    ar1_sim,
    ar1_fit)


from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt

# a collection of useful functions

def gen_ts(model='colored_noise',alpha=1, nt=100, f0=None, m=None, seed=None):
    'wrapper for gen_ts in pyleoclim'
    
    t,v = pyleo.utils.gen_ts(model=model,alpha=alpha, nt=nt, f0=f0, m=m, seed=seed)
    ts=pyleo.Series(t,v)
    return ts

def gen_normal(loc=0, scale=1, nt=100):
    ''' Generate random data with a Gaussian distribution
    '''
    t = np.arange(nt)
    v = np.random.normal(loc=loc, scale=scale, size=nt)
    ts = pyleo.Series(t,v)
    return ts
    
# def load_data():
#     try:
#         url = 'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/scal_signif_benthic.json'
#         d = pyleo.utils.jsonutils.json_to_Scalogram(url)
#     except:
#         d = pyleo.utils.jsonutils.json_to_Scalogram('../../example_data/scal_signif_benthic.json')
#     return d

# Tests below

class TestUiSeriesMakeLabels:
    ''' Tests for Series.make_labels()

    Since Series.make_labels() has several `if` statements,
    multiple tests will be performed to test different possibilities of the `if` statements.
    '''
    def test_make_labels_t0(self):
        ''' Test Series.make_labels() with default metadata for Series()
        '''
        # generate test data
        ts = gen_normal()

        # call the target function for testing
        time_header, value_header = ts.make_labels()

        assert time_header == 'time'
        assert value_header == 'value'


    def test_make_labels_t1(self):
        ''' Test Series.make_labels() with fully specified metadata for Series()
        '''
        # generate test data
        ts = gen_normal()

        # define a Series() obj with meta data
        ts1 = pyleo.Series(
            time=ts.time, value=ts.value,
            time_name='Year (CE)', time_unit='yr',
            value_name='Temperature', value_unit='K',
            label='Gaussian Noise', clean_ts=False
        )

        time_header, value_header = ts1.make_labels()

        assert time_header == 'Year (CE) [yr]'
        assert value_header == 'Temperature [K]'

    def test_make_labels_t2(self):
        ''' Test Series.make_labels() with partially specified metadata for Series()
        '''
        # generate test data
        ts = gen_normal()

        # define a Series() obj with meta data
        ts1 = pyleo.Series(
            time=ts.time, value=ts.value,
            time_name='Year (CE)',
            value_name='Temperature', value_unit='K',
            label='Gaussian Noise', clean_ts=False
        )

        time_header, value_header = ts1.make_labels()

        assert time_header == 'Year (CE)'
        assert value_header == 'Temperature [K]'


class TestUiSeriesSpectral:
    ''' Tests for Series.spectral()

    Since Series.spectral() has several `method` options along with different keyword arguments,
    multiple tests will be performed to test each method with different keyword arguments combinations.
    Note the usage of the decorator `@pytest.mark.parametrize()` for convenient
    [parametrizing tests](https://docs.pytest.org/en/stable/example/parametrize.html).
    '''

    @pytest.mark.parametrize('spec_method', ['wwz', 'mtm', 'lomb_scargle', 'welch', 'periodogram','cwt'])
    def test_spectral_t0(self, spec_method, eps=0.5):
        ''' Test Series.spectral() with available methods using default arguments

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha=1
        ts = gen_ts(nt=1000, alpha=alpha, seed=2333)
        psd = ts.spectral(method=spec_method)
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-alpha) < eps

    @pytest.mark.parametrize('freq_method', ['log', 'scale', 'nfft', 'lomb_scargle', 'welch'])
    def test_spectral_t1(self, freq_method, eps=0.3):
        ''' Test Series.spectral() with MTM using available `freq_method` options with other arguments being default

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        ts = gen_ts(nt=500, alpha=alpha)
        psd = ts.spectral(method='mtm', freq_method=freq_method)
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-alpha) < eps

    @pytest.mark.parametrize('nfreq', [10, 20, 30])
    def test_spectral_t2(self, nfreq, eps=0.3):
        ''' Test Series.spectral() with MTM using `freq_method='log'` with different values for its keyword argument `nfreq`

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        ts = gen_ts(nt=500, alpha=alpha)
        psd = ts.spectral(method='mtm', freq_method='log', freq_kwargs={'nfreq': nfreq})
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-alpha) < eps

    @pytest.mark.parametrize('dj', [0.25, 0.5, 1])
    def test_spectral_t3(self, dj, eps=0.3):
        ''' Test Series.spectral() with MTM using `freq_method='scale'` with different values for its keyword argument `nv`

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        ts = gen_ts(nt=500, alpha=alpha)
        psd = ts.spectral(method='mtm', freq_method='scale', freq_kwargs={'dj': dj})
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-alpha) < eps

    @pytest.mark.parametrize('dt, nf, ofac, hifac', [(None, 20, 1, 1), (None, None, 2, 0.5)])
    def test_spectral_t4(self, dt, nf, ofac, hifac, eps=0.5):
        ''' Test Series.spectral() with Lomb_Scargle using `freq_method=lomb_scargle` with different values for its keyword arguments

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        ts = gen_ts(nt=500, alpha=alpha)
        psd = ts.spectral(method='mtm', freq_method='lomb_scargle', freq_kwargs={'dt': dt, 'nf': nf, 'ofac': ofac, 'hifac': hifac})
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-alpha) < eps

    def test_spectral_t5(self, eps=0.6):
        ''' Test Series.spectral() with WWZ with specified frequency vector passed via `settings`

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        Note `asser_array_equal(psd.frequency, freq)` is used to make sure the specified frequency vector is really working.
        Also, we give `label` a test.
        '''
        alpha = 1
        ts = gen_ts(nt=500, alpha=alpha)
        freq = np.linspace(1/500, 1/2, 100)
        psd = ts.spectral(method='wwz', settings={'freq': freq}, label='WWZ')
        beta = psd.beta_est(fmin=1/200, fmax=1/10).beta_est_res['beta']
        assert_array_equal(psd.frequency, freq)
        assert np.abs(beta-alpha) < eps

    @pytest.mark.parametrize('spec_method', ['wwz', 'lomb_scargle'])
    def test_spectral_t6(self, spec_method, eps=0.3):
        ''' Test Series.spectral() with WWZ and Lomb Scargle on unevenly-spaced data with default arguments

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        ts = gen_ts(nt=500, alpha=alpha)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        psd = ts2.spectral(method=spec_method)
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-alpha) < eps
    
    @pytest.mark.parametrize('spec_method', ['wwz','cwt'])
    def test_spectral_t7(self, spec_method,):
        '''Test the spectral significance testing with pre-generated scalogram objects
        '''
        
        ts = gen_ts()
        scal = ts.wavelet(method=spec_method)
        signif = scal.signif_test(number=2,export_scal = True)
        sig_psd = ts.spectral(method=spec_method,scalogram=scal)
        sig_psd.signif_test(number=2,scalogram=signif).plot()

class TestUiSeriesBin:
    ''' Tests for Series.bin()

    Functions to test the various kwargs arguments for binning a timeseries
    '''

    def test_bin_t1(self):
        ''' Test the bin function with default parameter values'''
        alpha = 1
        ts = gen_ts(nt=500, alpha=alpha)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts2_bin=ts2.bin()

    def test_bin_t2(self):
        ''' Test the bin function by passing arguments'''
        alpha = 1
        ts = gen_ts(nt=500, alpha=alpha)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)
        start_date= np.min(t_unevenly)
        end_date = np.max(t_unevenly)
        bin_size=np.mean(np.diff(t_unevenly))

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts2_bin=ts2.bin(start=start_date,bin_size=bin_size,stop=end_date)

class TestUiSeriesStats:
    '''Test for Series.stats()

    Since Series.stats() is a numpy wrapper we will test it against known values'''

    def test_stats(self):
        '''Run test_stats against known dataset'''

        #Generate data
        t = np.arange(10)
        v = np.arange(10)

        #Create time series object
        ts = pyleo.Series(time=t,value=v)

        #Call target function for testing
        stats = ts.stats()

        #Generate answer key
        key = {'mean': 4.5,'median': 4.5,'min': 0.0,'max': 9.0,'std': np.std(t),'IQR': 4.5}

        assert stats == key
        
class TestUiSeriesCenter:
    '''Test for Series.center()

    Center removes the mean, so we'll simply test maximum and minimum values'''

    def test_center(self):
        #Generate sample data
        alpha = 1
        ts = gen_ts(nt=500, alpha=alpha)

        #Call function to be tested
        tsc = ts.center()

        assert np.abs(tsc.mean) <= np.sqrt(sys.float_info.epsilon) 

class TestUiSeriesStandardize:
    '''Test for Series.standardize()

    Standardize normalizes the series object, so we'll simply test maximum and minimum values'''

    def test_standardize(self):
        alpha = 1
        ts = gen_ts(nt=500, alpha=alpha)

        #Call function to be tested
        ts_std = ts.standardize()

        #Compare maximum and minimum values
        value = ts.__dict__['value']
        value_std = ts_std.__dict__['value']

        assert max(value) > max(value_std)
        assert min(value) < min(value_std)

class TestUiSeriesClean:
    '''Test for Series.clean()

    Since Series.clean() is intended to order the time axis,
    we will test the first and last values on the time axis and ensure that the length
    of the time and value sections are equivalent'''

    def test_clean(self):

        #Generate data
        ts = gen_normal()

        #Call function for testing
        ts_clean = ts.clean()

        #Isolate time and value components
        time = ts_clean.__dict__['time']
        value = ts_clean.__dict__['value']

        assert time[len(time) - 1] >= time[0]
        assert len(time) == len(value)

class TestUiSeriesGaussianize:
    '''Test for Series.gaussianize()

    Gaussianize normalizes the series object, so we'll simply test maximum and minimum values'''
    def test_gaussianize(self):
        ts = gen_ts()
        ts_gauss = ts.gaussianize()

        value = ts.__dict__['value']
        value_std = ts_gauss.__dict__['value']

        assert max(value) > max(value_std)
        assert min(value) < min(value_std)

class TestUiSeriesSegment:
    '''Tests for Series.segment()

    Segment has an if and elif statement, so we will test each in turn'''

    def test_segment_t0(self):
        '''Test that in the case of segmentation, segment returns a Multiple Series object'''
        t = (1,2,3000)
        v = (1,2,3)

        ts = pyleo.Series(time = t, value = v)

        ts_seg = ts.segment()

        assert str(type(ts_seg)) == "<class 'pyleoclim.core.multipleseries.MultipleSeries'>"

    def test_segment_t1(self):
        '''Test that in the case of no segmentation, segment and original time series
        are the some object type'''
        ts = gen_normal()
        ts_seg = ts.segment()

        assert type(ts_seg) == type(ts)

class TestUiSeriesSlice:
    '''Test for Series.slice()

    We commit slices at known time intervals and check minimum and maximum values'''

    def test_slice(self):
        ts = gen_normal()
        ts_slice = ts.slice(timespan = (10, 50, 80, 90))
        times = ts_slice.__dict__['time']

        assert min(times) == 10
        assert max(times) == 90

class TestUiSeriesSurrogates:
    ''' Test Series.surrogates()
    '''
    def test_surrogates_t0(self, eps=0.2):
        ''' Generate AR(1) surrogates based on a AR(1) series with certain parameters,
        and then evaluate and assert the parameters of the surrogates are correct
        '''
        g = 0.5  # persistence
        ar = [1, -g]
        ma = [1, 0]
        n = 1000
        ar1 = arma_generate_sample(ar, ma, nsample=n, scale=1)
        ts = pyleo.Series(time=np.arange(1000), value=ar1)

        ts_surrs = ts.surrogates(number=1)
        for ts_surr in ts_surrs.series_list:
            g_surr = ar1_fit(ts_surr.value)
            assert np.abs(g_surr-g) < eps

class TestUiSeriesSummaryPlot:
    ''' Test Series.summary_plot()
    '''
    def test_summary_plot_t0(self):
        '''Testing that labels are being passed and that psd and scalogram objects dont fail when passed. 
        Also testing that we can specify fewer significance tests than those stored in the scalogram object
        Note that we should avoid pyleo.showfig() in tests.
        
        Passing pre generated scalogram and psd.
        '''
        ts = gen_ts()
        scal = ts.wavelet(method='cwt')
        psd = ts.spectral(method='cwt')
        period_label='Period'
        psd_label='PSD'
        time_label='Time'
        value_label='Value'
        fig, ax = ts.summary_plot(
            psd = psd, scalogram=scal, figsize=[4, 5], title='Test',
            period_label=period_label, psd_label=psd_label,
            value_label=value_label, time_label=time_label
        )
        
        assert ax['scal'].properties()['ylabel'] == period_label, 'Period label is not being passed properly'
        assert ax['psd'].properties()['xlabel'] == psd_label, 'PSD label is not being passed properly'
        assert ax['scal'].properties()['xlabel'] == time_label, 'Time label is not being passed properly'
        assert ax['ts'].properties()['ylabel'] == value_label, 'Value label is not being passed properly'

        plt.close(fig)

    def test_summary_plot_t1(self):
        '''Testing that the bare function works
        Note that we should avoid pyleo.showfig() in tests.
    
        Passing just a pre generated psd.
        '''
        ts = gen_ts()
        scal = ts.wavelet(method='cwt')
        psd = ts.spectral(method='cwt')
        fig, ax = ts.summary_plot(psd=psd,scalogram=scal)
    
        plt.close(fig)  
    

class TestUiSeriesCorrelation:
    ''' Test Series.correlation()
    '''
    @pytest.mark.parametrize('corr_method', ['ttest', 'isopersistent', 'isospectral'])
    def test_correlation_t0(self, corr_method, eps=1):
        ''' Generate two series from a same basic series and calculate their correlation
        '''
        alpha = 1
        nt = 100
        ts = gen_ts(nt=nt,alpha=alpha)
        v1 = ts.value + np.random.normal(loc=0, scale=1, size=nt)
        v2 = ts.value + np.random.normal(loc=0, scale=2, size=nt)

        ts1 = pyleo.Series(time=ts.time, value=v1)
        ts2 = pyleo.Series(time=ts.time, value=v2)

        corr_res = ts1.correlation(ts2, settings={'method': corr_method})
        r = corr_res.r
        assert np.abs(r-1) < eps

    @pytest.mark.parametrize('corr_method', ['ttest', 'isopersistent', 'isospectral'])
    def test_correlation_t1(self, corr_method, eps=1):
        ''' Generate two colored noise series calculate their correlation
        '''
        alpha = 1
        nt = 1000
        ts = gen_ts(nt=nt,alpha=alpha)
        v1 = ts.value + np.random.normal(loc=0, scale=1, size=nt)
        v2 = ts.value + np.random.normal(loc=0, scale=2, size=nt)

        ts1 = pyleo.Series(time=ts.time, value=v1)
        ts2 = pyleo.Series(time=ts.time, value=v2)

        corr_res = ts1.correlation(ts2, settings={'method': corr_method})
        r = corr_res.r
        assert np.abs(r-0) < eps

    @pytest.mark.parametrize('corr_method', ['ttest', 'isopersistent', 'isospectral'])
    def test_correlation_t2(self, corr_method, eps=1):
        ''' Test correlation between two series with inconsistent time axis
        '''
        data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/wtc_test_data_nino.csv')
        nino = np.array(data['nino'])
        air  = np.array(data['air'])
        t = np.array(data['t'])

        # randomly delete 500 data pts
        n_del = 500
        deleted_idx_air = np.random.choice(range(np.size(t)), n_del, replace=False)
        deleted_idx_nino = np.random.choice(range(np.size(t)), n_del, replace=False)
        air_value_unevenly =  np.delete(air, deleted_idx_air)
        air_time_unevenly =  np.delete(t, deleted_idx_air)
        nino_value_unevenly =  np.delete(nino, deleted_idx_nino)
        nino_time_unevenly =  np.delete(t, deleted_idx_nino)

        ts1_evenly = pyleo.Series(time=t, value=air)
        ts2_evenly = pyleo.Series(time=t, value=nino)
        corr_res_evenly = ts1_evenly.correlation(ts2_evenly, settings={'method': corr_method})
        r_evenly = corr_res_evenly.r

        ts1 = pyleo.Series(time=air_time_unevenly, value=air_value_unevenly)
        ts2 = pyleo.Series(time=nino_time_unevenly, value=nino_value_unevenly)

        corr_res = ts1.correlation(ts2, settings={'method': corr_method}, common_time_kwargs={'method': 'interp'})
        r = corr_res.r
        assert np.abs(r-r_evenly) < eps

class TestUiSeriesCausality:
    ''' Test Series.causality()
    '''
    @pytest.mark.parametrize('method', ['liang', 'granger'])
    def test_causality_t0(self, method, eps=1):
        ''' Generate two series from a same basic series and calculate their correlation

        Note: NO assert statements for this test yet
        '''
        alpha = 1
        nt = 100
        ts = gen_ts(nt=nt,alpha=alpha)
        v1 = ts.value + np.random.normal(loc=0, scale=1, size=nt)
        v2 = ts.value + np.random.normal(loc=0, scale=2, size=nt)

        ts1 = pyleo.Series(time=ts.time, value=v1)
        ts2 = pyleo.Series(time=ts.time, value=v2)

        causal_res = ts1.causality(ts2, method=method)
        
class TestUISeriesOutliers:
    ''' Tests for Series.outliers()

    Remove outliers from a timeseries. Note that for CI purposes only, the automated version can be tested
    '''
    @pytest.mark.parametrize('remove_outliers', [True,False])
    def test_outliers_t1(self,remove_outliers):

        #Generate data
        ts = gen_ts()
        #Add outliers
        outliers_start = np.mean(ts.value)+5*np.std(ts.value)
        outliers_end = np.mean(ts.value)+7*np.std(ts.value)
        outlier_values = np.arange(outliers_start,outliers_end,0.1)
        index = np.random.randint(0,len(ts.value),6)
        v_out = ts.value
        for i,ind in enumerate(index):
            v_out[ind] = outlier_values[i]
        # Get a series object
        ts2 = pyleo.Series(time = ts.time, value = v_out)
        # Remove outliers
        ts_out, res = ts.outliers(remove=remove_outliers)
    
    @pytest.mark.parametrize('method', ['kmeans','DBSCAN'])
    def test_outliers_t2(self,method):

        #Generate data
        ts = gen_ts()
        #Add outliers
        outliers_start = np.mean(ts.value)+5*np.std(ts.value)
        outliers_end = np.mean(ts.value)+7*np.std(ts.value)
        outlier_values = np.arange(outliers_start,outliers_end,0.1)
        index = np.random.randint(0,len(ts.value),6)
        v_out = ts.value
        for i,ind in enumerate(index):
            v_out[ind] = outlier_values[i]
        # Get a series object
        ts2 = pyleo.Series(time = ts.time, value = v_out)
        # Remove outliers
        ts_out, res = ts.outliers(method=method)

class TestUISeriesGkernel:
    ''' Unit tests for the TestUISeriesGkernel function
    '''
    def test_interp_t1(self):
        ''' Test the gkernel function with default parameter values'''

        ts = gen_ts(nt=550, alpha=1.0)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts_interp=ts2.gkernel()

    def test_interp_t2(self):
        ''' Test the gkernel function with specified bandwidth'''

        ts = gen_ts(nt=550, alpha=1.0)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts_interp=ts2.gkernel(h=15)


class TestUISeriesInterp():
    ''' Unit tests for the interpolation function
    '''

    def test_interp_t1(self):
        ''' Test the interp function with default parameter values'''
        ts = gen_ts(nt=550, alpha=1.0)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts_interp=ts.interp()

    def test_interp_t2(self):
        ''' Test the bin function by passing arguments'''
        ts = gen_ts(nt=550, alpha=1.0)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)
        start_date= np.min(t_unevenly)
        end_date = np.max(t_unevenly)
        bin_size=np.mean(np.diff(t_unevenly))

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts_interp=ts2.interp(start=start_date, step=bin_size, stop=end_date)

    @pytest.mark.parametrize('interp_method', ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'])
    def test_interp_t3(self,interp_method):
        ''' Test the interp function with default parameter values'''
        ts = gen_ts(nt=550, alpha=1.0)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts_interp=ts2.interp(method=interp_method)


class TestUISeriesDetrend():
    ''' Unit tests for the detrending function
    '''

    @pytest.mark.parametrize('detrend_method',['linear','constant','savitzky-golay','emd'])
    def test_detrend_t1(self,detrend_method):
        #Generate data
        ts = gen_ts(nt=550, alpha=1.0)
        #Add a trend
        slope = 1e-5
        intercept = -1
        nonlinear_trend = slope*ts.time**2 + intercept
        v_trend = ts.value + nonlinear_trend
        #create a timeseries object
        ts2 = pyleo.Series(time=ts.time, value=v_trend)
        ts_detrend=ts2.detrend(method=detrend_method)

class TestUISeriesWaveletCoherence():
    ''' Test the wavelet coherence
    '''
    @pytest.mark.parametrize('xwave_method',['wwz','cwt'])
    def test_xwave_t0(self, xwave_method):
        ''' Test Series.wavelet_coherence() with available methods using default arguments
        '''
        nt = 200
        ts1 = gen_ts(model='colored_noise', nt=nt)
        ts2 = gen_ts(model='colored_noise', nt=nt)
        _ = ts2.wavelet_coherence(ts1,method=xwave_method)

    def test_xwave_t1(self):
        ''' Test Series.wavelet_coherence() with WWZ with specified frequency vector passed via `settings`
        '''
        nt = 200
        ts1 = gen_ts(model='colored_noise', nt=nt)
        ts2 = gen_ts(model='colored_noise', nt=nt)
        freq = np.linspace(1/500, 1/2, 20)
        _ = ts1.wavelet_coherence(ts2,method='wwz',settings={'freq':freq})
        
    @pytest.mark.parametrize('mother',['MORLET', 'PAUL', 'DOG'])
    def test_xwave_t2(self,mother):
        ''' Test Series.wavelet_coherence() with CWT with mother wavelet specified  via `settings`
        '''
        nt = 500
        ts1 = gen_ts(model='colored_noise', nt=nt)
        ts2 = gen_ts(model='colored_noise', nt=nt)
        _ = ts1.wavelet_coherence(ts2,method='cwt',settings={'mother':mother})

    def test_xwave_t3(self):
        ''' Test Series.wavelet_coherence() with WWZ on unevenly spaced data
        '''
        
        ts1 = gen_ts(nt=220, alpha=1)
        ts2 = gen_ts(nt=220, alpha=1)
        #remove points
        n_del = 20
        deleted_idx = np.random.choice(range(np.size(ts1.time)), n_del, replace=False)
        deleted_idx1 = np.random.choice(range(np.size(ts2.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts1.time, deleted_idx)
        v_unevenly =  np.delete(ts1.value, deleted_idx)
        t1_unevenly =  np.delete(ts2.time, deleted_idx1)
        v1_unevenly =  np.delete(ts2.value, deleted_idx1)
        ts3 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts4 = pyleo.Series(time=t1_unevenly, value=v1_unevenly)
        scal = ts3.wavelet_coherence(ts4,method='wwz')

class TestUISeriesWavelet():
    ''' Test the wavelet functionalities
    '''

    @pytest.mark.parametrize('wave_method',['wwz','cwt'])
    def test_wave_t0(self, wave_method):
        ''' Test Series.wavelet() with available methods using default arguments
        '''
        ts = gen_ts(model='colored_noise',nt=100)
        _ = ts.wavelet(method=wave_method)

    @pytest.mark.parametrize('wave_method',['wwz','cwt'])
    def test_wave_t1(self,wave_method):
        '''Test Series.spectral() with WWZ/cwt with specified frequency vector passed via `settings`
        '''
        n = 100
        ts = gen_ts(model='colored_noise',nt=n)
        freq = np.linspace(1/n, 1/2, 20)
        _ = ts.wavelet(method=wave_method, settings={'freq': freq})
        
    def test_wave_t2(self):
       ''' Test Series.wavelet() ntau option and plot functionality
       '''
       ts = gen_ts(model='colored_noise',nt=200)
       _ = ts.wavelet(method='wwz',settings={'ntau':10})
 
    @pytest.mark.parametrize('mother',['MORLET', 'PAUL', 'DOG'])
    def test_wave_t3(self,mother):
       ''' Test Series.wavelet() with different mother wavelets
       '''
       ts = gen_ts(model='colored_noise',nt=200)
       _ = ts.wavelet(method='cwt',settings={'mother':mother})

class TestUISeriesSsa():
    ''' Test the SSA functionalities
    '''

    def test_ssa_t0(self):
        ''' Test Series.ssa() with available methods using default arguments
        '''

        cn = gen_ts(model = 'colored_noise', nt= 500, alpha=1.0)

        res = cn.ssa()
        assert abs(res.pctvar.sum() - 100.0)<0.01
        

    def test_ssa_t1(self):
        '''Test Series.ssa() with var truncation
        '''
        ts = gen_ts(model = 'colored_noise', nt=500, alpha=1.0)
        res = ts.ssa(trunc='var')

    def test_ssa_t2(self):
        '''Test Series.ssa() with Monte-Carlo truncation
        '''

        ts = gen_ts(model = 'colored_noise', nt=500, alpha=1.0)

        res = ts.ssa(M=60, nMC=10, trunc='mcssa')
        fig, ax = res.screeplot()
        pyleo.closefig(fig)

    def test_ssa_t3(self):
        '''Test Series.ssa() with Kaiser truncation
        '''
        ts = gen_ts(model = 'colored_noise', nt=500, alpha=1.0)
        res = ts.ssa(trunc='kaiser')
        
    def test_ssa_t4(self):
        '''Test Series.ssa() on Allen&Smith dataset
        '''
        df = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/mratest.txt',delim_whitespace=True,names=['Total','Signal','Noise'])
        mra = pyleo.Series(time=df.index, value=df['Total'], value_name='Allen&Smith test data', time_name='Time', time_unit='yr')
        mraSsa = mra.ssa(nMC=10)
        fig, ax = mraSsa.screeplot()
        pyleo.closefig(fig)

class TestUiSeriesPlot:
    '''Test for Series.plot()

    Series.plot outputs a matplotlib figure and axis object, so we will compare the time axis
    of the axis object to the time array.'''

    def test_plot(self):

        ts = gen_normal()

        fig, ax = ts.plot()

        line = ax.lines[0]

        x_plot = line.get_xdata()
        y_plot = line.get_ydata()

        
        pyleo.closefig(fig)

class TestUiSeriesDistplot:
    '''Test for Series.distplot()'''

    def test_distplot_t0(self, max_axis = 5):
        ts = gen_normal()
        
        fig, ax = ts.distplot()

        line = ax.lines[0]

        x_plot = line.get_xdata()
        y_plot = line.get_ydata()

        assert max(x_plot) < max_axis
        
        pyleo.closefig(fig)

    def test_distplot_t1(self, vertical = True):
        ts = gen_normal()
        

        fig, ax = ts.distplot(vertical=vertical)
        
        pyleo.closefig(fig)

class TestUiSeriesFilter:
    '''Test for Series.filter()'''

    @pytest.mark.parametrize('method', ['butterworth', 'firwin','lanczos','savitzky-golay'])
    def test_filter_t0(self, method):
        ''' Low-pass filtering with Butterworth or FIR with window
        '''
        t = np.linspace(0, 1, 1000)
        sig1 = np.sin(2*np.pi*10*t)
        sig2 = np.sin(2*np.pi*20*t)
        sig = sig1 + sig2
        ts1 = pyleo.Series(time=t, value=sig1)
        ts = pyleo.Series(time=t, value=sig)
        ts_lp = ts.filter(cutoff_freq=15, method=method)
        val_diff = ts_lp.value - ts1.value
        assert np.mean(val_diff**2) < 0.2


    @pytest.mark.parametrize('method', ['butterworth', 'firwin'])
    def test_filter_t1(self, method):
        ''' Band-pass filtering with Butterworth or FIR with window
        '''
        t = np.linspace(0, 1, 1000)
        sig1 = np.sin(2*np.pi*10*t)
        sig2 = np.sin(2*np.pi*20*t)
        sig = sig1 + sig2
        ts2 = pyleo.Series(time=t, value=sig2)
        ts = pyleo.Series(time=t, value=sig)
        ts_bp = ts.filter(cutoff_freq=[15, 25], method=method)
        val_diff = ts_bp.value - ts2.value
        assert np.mean(val_diff**2) < 0.1
    