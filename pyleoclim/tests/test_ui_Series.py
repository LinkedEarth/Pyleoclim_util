''' Tests for pyleoclim.core.ui.Series

Naming rules:
1. classe: Test{filename}{Class}{method} with appropriate camel case
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

class TestUiSeriesMakeLabels:
    ''' Tests for Series.make_labels()

    Since Series.make_labels() has several `if` statements,
    multiple tests will be performed to test different possibilities of the `if` statements.
    '''
    def test_make_labels_t0(self):
        ''' Test Series.make_labels() with default metadata for Series()
        '''
        # generate test data
        t, v = gen_normal()

        # define a Series() obj without meta data
        ts = pyleo.Series(time=t, value=v)

        # call the target function for testing
        time_header, value_header = ts.make_labels()

        assert time_header == 'time'
        assert value_header == 'value'


    def test_make_labels_t1(self):
        ''' Test Series.make_labels() with fully specified metadata for Series()
        '''
        # generate test data
        t, v = gen_normal()

        # define a Series() obj with meta data
        ts = pyleo.Series(
            time=t, value=v,
            time_name='Year (CE)', time_unit='yr',
            value_name='Temperature', value_unit='K',
            label='Gaussian Noise', clean_ts=False
        )

        time_header, value_header = ts.make_labels()

        assert time_header == 'Year (CE) [yr]'
        assert value_header == 'Temperature [K]'

    def test_make_labels_t2(self):
        ''' Test Series.make_labels() with partially specified metadata for Series()
        '''
        # generate test data
        t, v = gen_normal()

        # define a Series() obj with meta data
        ts = pyleo.Series(
            time=t, value=v,
            time_name='Year (CE)',
            value_name='Temperature', value_unit='K',
            label='Gaussian Noise', clean_ts=False
        )

        time_header, value_header = ts.make_labels()

        assert time_header == 'Year (CE)'
        assert value_header == 'Temperature [K]'


class TestUiSeriesSpectral:
    ''' Tests for Series.spectral()

    Since Series.spectral() has several `method` options along with different keyword arguments,
    multiple tests will be performed to test each method with different keyword arguments combinations.
    Note the usage of the decorator `@pytest.mark.parametrize()` for convenient
    [parametrizing tests](https://docs.pytest.org/en/stable/example/parametrize.html).
    '''

    @pytest.mark.parametrize('spec_method', ['wwz', 'mtm', 'lomb_scargle', 'welch', 'periodogram'])
    def test_spectral_t0(self, spec_method, eps=0.5):
        ''' Test Series.spectral() with available methods using default arguments

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        t, v = gen_colored_noise(nt=500, alpha=alpha)
        ts = pyleo.Series(time=t, value=v)
        psd = ts.spectral(method=spec_method)
        beta = psd.beta_est()['beta']
        assert np.abs(beta-alpha) < eps

    @pytest.mark.parametrize('freq_method', ['log', 'scale', 'nfft', 'lomb_scargle', 'welch'])
    def test_spectral_t1(self, freq_method, eps=0.3):
        ''' Test Series.spectral() with MTM using available `freq_method` options with other arguments being default

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        t, v = gen_colored_noise(nt=500, alpha=alpha)
        ts = pyleo.Series(time=t, value=v)
        psd = ts.spectral(method='mtm', freq_method=freq_method)
        beta = psd.beta_est()['beta']
        assert np.abs(beta-alpha) < eps

    @pytest.mark.parametrize('nfreq', [10, 20, 30])
    def test_spectral_t2(self, nfreq, eps=0.3):
        ''' Test Series.spectral() with MTM using `freq_method='log'` with different values for its keyword argument `nfreq`

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        t, v = gen_colored_noise(nt=500, alpha=alpha)
        ts = pyleo.Series(time=t, value=v)
        psd = ts.spectral(method='mtm', freq_method='log', freq_kwargs={'nfreq': nfreq})
        beta = psd.beta_est()['beta']
        assert np.abs(beta-alpha) < eps

    @pytest.mark.parametrize('nv', [10, 20, 30])
    def test_spectral_t3(self, nv, eps=0.3):
        ''' Test Series.spectral() with MTM using `freq_method='scale'` with different values for its keyword argument `nv`

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        t, v = gen_colored_noise(nt=500, alpha=alpha)
        ts = pyleo.Series(time=t, value=v)
        psd = ts.spectral(method='mtm', freq_method='scale', freq_kwargs={'nv': nv})
        beta = psd.beta_est()['beta']
        assert np.abs(beta-alpha) < eps

    @pytest.mark.parametrize('dt, nf, ofac, hifac', [(None, 20, 1, 1), (None, None, 2, 0.5)])
    def test_spectral_t4(self, dt, nf, ofac, hifac, eps=0.5):
        ''' Test Series.spectral() with MTM using `freq_method=lomb_scargle` with different values for its keyword arguments

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        t, v = gen_colored_noise(nt=500, alpha=alpha)
        ts = pyleo.Series(time=t, value=v)
        psd = ts.spectral(method='mtm', freq_method='lomb_scargle', freq_kwargs={'dt': dt, 'nf': nf, 'ofac': ofac, 'hifac': hifac})
        beta = psd.beta_est()['beta']
        assert np.abs(beta-alpha) < eps

    def test_spectral_t5(self, eps=0.3):
        ''' Test Series.spectral() with WWZ with specified frequency vector passed via `settings`

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        Note `asser_array_equal(psd.frequency, freq)` is used to make sure the specified frequency vector is really working.
        Also, we give `label` a test.
        '''
        alpha = 1
        t, v = gen_colored_noise(nt=500, alpha=alpha)
        ts = pyleo.Series(time=t, value=v)
        freq = np.linspace(1/500, 1/2, 20)
        psd = ts.spectral(method='wwz', settings={'freq': freq}, label='WWZ')
        beta = psd.beta_est()['beta']
        assert_array_equal(psd.frequency, freq)
        assert np.abs(beta-alpha) < eps

    @pytest.mark.parametrize('spec_method', ['wwz', 'lomb_scargle'])
    def test_spectral_t6(self, spec_method, eps=0.3):
        ''' Test Series.spectral() with WWZ and Lomb Scargle on unevenly-spaced data with default arguments

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        alpha = 1
        t, v = gen_colored_noise(nt=550, alpha=alpha)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(t)), n_del, replace=False)
        t_unevenly =  np.delete(t, deleted_idx)
        v_unevenly =  np.delete(v, deleted_idx)

        ts = pyleo.Series(time=t_unevenly, value=v_unevenly)
        psd = ts.spectral(method=spec_method)
        beta = psd.beta_est()['beta']
        assert np.abs(beta-alpha) < eps

class TestUiSeriesBin:
    ''' Tests for Series.bin()

    Functions to test the various kwargs arguments for binning a timeseries
    '''
    
    def test_bin_t1(self):
        ''' Test the bin function with default parameter values'''
        alpha = 1
        t, v = gen_colored_noise(nt=550, alpha=alpha)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(t)), n_del, replace=False)
        t_unevenly =  np.delete(t, deleted_idx)
        v_unevenly =  np.delete(v, deleted_idx)
        
        ts = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts_bin=ts.bin()
    
    def test_bin_t2(self):
        ''' Test the bin function by passing arguments'''
        alpha = 1
        t, v = gen_colored_noise(nt=550, alpha=alpha)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(t)), n_del, replace=False)
        t_unevenly =  np.delete(t, deleted_idx)
        v_unevenly =  np.delete(v, deleted_idx)
        start_date= np.min(t_unevenly)
        end_date = np.max(t_unevenly)
        bin_size=np.mean(np.diff(t_unevenly))
        
        ts = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts_bin=ts.bin(start=start_date,bin_size=bin_size,end=end_date)
        
class TestUiSeriesStats:
    '''Test for Series.stats()

    Since Series.stats() is a numpy wrapper we will test it against known values,
    and ensure that it is returning the appropriate data format (dict).'''

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

        assert type(stats) == dict
        assert stats == key

class TestUiSeriesStandardize:
    '''Test for Series.standardize()

    Standardize normalizes the series object, so we'll simply test maximum and minimum values'''

    def test_standardize(self):
        #Generate sample data
        t, v = gen_colored_noise()

        #Create time series with sample data
        ts = pyleo.Series(time = t, value = v)

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
        t, v = gen_normal()

        #Create time series object
        ts = pyleo.Series(time=t,value=v)

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
        t, v = gen_colored_noise()

        ts = pyleo.Series(time = t, value = v)

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

        assert str(type(ts_seg)) == "<class 'pyleoclim.core.ui.MultipleSeries'>"

    def test_segment_t1(self):
        '''Test that in the case of no segmentation, segment and original time series
        are the some object type'''
        t, v = gen_normal()

        ts = pyleo.Series(time = t, value = v)

        ts_seg = ts.segment()

        assert type(ts_seg) == type(ts)

class TestUiSeriesSlice:
    '''Test for Series.slice()

    We commit slices at known time intervals and check minimum and maximum values'''

    def test_slice(self):
        t, v = gen_normal()

        ts = pyleo.Series(time = t, value = v)

        ts_slice = ts.slice(timespan = (10, 50, 80, 90))

        times = ts_slice.__dict__['time']

        assert min(times) == 10
        assert max(times) == 90

class TestUISeriesOutliers:
    ''' Tests for Series.outliers()
    
    Remove outliers from a timeseries. Note that for CI purposes only, the automated version can be tested
    '''
    @pytest.mark.parametrize('remove_outliers', [True,False])
    def test_outliers(self,remove_outliers):
        
        #Generate data
        t, v = gen_colored_noise()
        #Add outliers
        outliers_start = np.mean(v)+5*np.std(v)
        outliers_end = np.mean(v)+7*np.std(v)
        outlier_values = np.arange(outliers_start,outliers_end,0.1)
        index = np.random.randint(0,len(v),6)
        v_out = v
        for i,ind in enumerate(index):
            v_out[ind] = outlier_values[i]
        # Get a series object
        ts = pyleo.Series(time = t, value = v_out) 
        # Remove outliers
        ts_out = ts.outliers(remove=remove_outliers)
        
    
    