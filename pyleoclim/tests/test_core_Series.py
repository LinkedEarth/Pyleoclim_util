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
import datetime as dt
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal

import pytest
#import scipy.io as sio
import sys#, json
import os
import pathlib
test_dirpath = pathlib.Path(__file__).parent.absolute()


import pyleoclim as pyleo
import pyleoclim.utils.tsbase as tsbase



# a collection of useful functions

def gen_ts(model='colored_noise',alpha=1, nt=50, f0=None, m=None, seed=None):
    'wrapper for gen_ts in pyleoclim'

    t,v = pyleo.utils.gen_ts(model=model,alpha=alpha, nt=nt, f0=f0, m=m, seed=seed)
    ts=pyleo.Series(t,v, verbose=False, auto_time_params=True)
    return ts

def gen_normal(loc=0, scale=1, nt=20):
    ''' Generate random data with a Gaussian distribution
    '''
    t = np.arange(nt)
    v = np.random.normal(loc=loc, scale=scale, size=nt)
    ts = pyleo.Series(t,v, verbose=False, auto_time_params=True)
    return ts

# def load_data():
#     try:
#         url = 'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/scal_signif_benthic.json'
#         d = pyleo.utils.jsonutils.json_to_Scalogram(url)
#     except:
#         d = pyleo.utils.jsonutils.json_to_Scalogram('../../example_data/scal_signif_benthic.json')
#     return d

# Tests below

class TestUISeriesInit:
     ''' Test for Series instantiation '''
    
     def test_init_no_dropna(self, evenly_spaced_series):
         ts = evenly_spaced_series
         t = ts.time
         v = ts.value
         v[0] = np.nan
         ts2 = pyleo.Series(time=t,value=v,dropna=False, verbose=False)
         assert np.isnan(ts2.value[0])
         
     def test_init_no_sorting(self, evenly_spaced_series):
         ts = evenly_spaced_series
         t = ts.time[::-1]
         v = ts.value[::-1]
         ts2 = pyleo.Series(time=t,value=v,sort_ts='Nein', verbose=False)
         res, _, sign = pyleo.utils.tsbase.resolution(ts2.time) 
         assert sign == 'negative'  
         
     def test_init_clean_ts(self, evenly_spaced_series):
         ts = evenly_spaced_series
         t = ts.time[::-1]
         v = ts.value[::-1]
         v[0] = np.nan
         ts2 = pyleo.Series(time=t,value=v, dropna=False, clean_ts=True, verbose=False)
         res, _, sign = pyleo.utils.tsbase.resolution(ts2.time) 
         assert np.isnan(ts2.value[-1])
     
     @pytest.mark.parametrize('units',[None, 'C.E.'])    
     def test_init_year_time_name_CE(self, evenly_spaced_series, units):
         ts = evenly_spaced_series
         t = ts.time
         v = ts.value
         ts2 = pyleo.Series(time=t, value=v, verbose=False,
                            time_name='year', time_unit=units)
         assert ts2.time_name == 'Time'
         assert ts2.time_unit == 'years CE'
         (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(ts2.time_unit)
         assert datum == 0
         assert direction == 'prograde'
     
     @pytest.mark.parametrize('units',['BP', 'B.P.'])    
     def test_init_year_time_name_BP(self, evenly_spaced_series, units):
         ts = evenly_spaced_series
         t = ts.time
         v = ts.value
         ts2 = pyleo.Series(time=t, value=v, verbose=False,
                            time_name='year', time_unit=units)
         
         assert ts2.time_name == 'Age'
         assert ts2.time_unit == units
         
         (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(ts2.time_unit)
         
         assert datum == 1950
         assert direction == 'retrograde'
    
     @pytest.mark.parametrize('archiveType', ['FluvialSediment', 'creek'])
     def test_init_archiveType(self, archiveType):
         ts =pyleo.Series(time=[2,3,5], value =[4,5,6], archiveType=archiveType, control_archiveType=True)
        
         assert ts.archiveType=='FluvialSediment'
          

class TestSeriesIO:
    ''' Test Series import from and export to other formats
    '''
    @pytest.mark.parametrize('ds_name',['NINO3','LR04'])
    def test_csv_roundtrip(self, ds_name):
        ts1 = pyleo.utils.load_dataset(ds_name)
        ts1.to_csv()
        filename = ts1.label.replace(" ", "_") + '.csv'
        ts2 = pyleo.Series.from_csv(path=filename)
        assert ts1.equals(ts2)
        #clean up file
        os.unlink(filename)

class TestUISeriesMakeLabels:
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

        assert time_header == 'Time [years CE]'
        assert value_header == 'value'


    def test_make_labels_t1(self):
        ''' Test Series.make_labels() with fully specified metadata for Series()
        '''
        # generate test data
        ts = gen_normal()

        # define a Series() obj with meta data
        ts1 = pyleo.Series(
            time=ts.time, value=ts.value,
            time_name='Time', time_unit='yr CE',
            value_name='Temperature', value_unit='K',
            label='Gaussian Noise', clean_ts=False
        )

        time_header, value_header = ts1.make_labels()

        assert time_header == 'Time [yr CE]'
        assert value_header == 'Temperature [K]'

    def test_make_labels_t2(self):
        ''' Test Series.make_labels() with partially specified metadata for Series()
        '''
        # generate test data
        ts = gen_normal()

        # define a Series() obj with meta data
        ts1 = pyleo.Series(
            time=ts.time, value=ts.value,
            time_name='time',
            value_name='Temperature', value_unit='K',
            label='Gaussian Noise', clean_ts=False
        )

        time_header, value_header = ts1.make_labels()

        assert time_header == 'time [years CE]'
        assert value_header == 'Temperature [K]'


class TestUISeriesSpectral:
    ''' Tests for Series.spectral()

    Since Series.spectral() has several `method` options along with different keyword arguments,
    multiple tests will be performed to test each method with different keyword arguments combinations.
    Note the usage of the decorator `@pytest.mark.parametrize()` for convenient
    [parametrizing tests](https://docs.pytest.org/en/stable/example/parametrize.html).
    '''

    @pytest.mark.parametrize('spec_method', ['wwz', 'mtm', 'lomb_scargle', 'welch', 'periodogram','cwt'])
    def test_spectral_t0(self, pinkseries, spec_method, eps=0.5):
        ''' Test Series.spectral() with available methods using default arguments

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        ts = pinkseries # has slope 1/f
        psd = ts.spectral(method=spec_method)
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-1.0) < eps

    @pytest.mark.parametrize('freq', ['log', 'scale', 'nfft', 'lomb_scargle', 'welch'])
    def test_spectral_t1(self, pinkseries, freq, eps=0.3):
        ''' Test Series.spectral() with MTM using available `freq` options with other arguments being default

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        ts = pinkseries
        psd = ts.spectral(method='mtm', freq=freq)
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-1.0) < eps

    @pytest.mark.parametrize('nf', [10, 20, 30])
    def test_spectral_t2(self, pinkseries, nf, eps=0.3):
        ''' Test Series.spectral() with MTM using `freq='log'` with different values for its keyword argument `nfreq`

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        ts = pinkseries
        psd = ts.spectral(method='mtm', freq='log', freq_kwargs={'nf': nf})
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-1.0) < eps

    @pytest.mark.parametrize('dj', [0.25, 0.5, 1])
    def test_spectral_t3(self, pinkseries, dj, eps=0.3):
        ''' Test Series.spectral() with MTM using `freq='scale'` with different values for its keyword argument `nv`

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        ts = pinkseries
        psd = ts.spectral(method='mtm', freq='scale', freq_kwargs={'dj': dj})
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-1.0) < eps

    @pytest.mark.parametrize('dt, nf, ofac, hifac', [(None, 20, 1, 1), (None, None, 2, 0.5)])
    def test_spectral_t4(self, pinkseries, dt, nf, ofac, hifac, eps=0.5):
        ''' Test Series.spectral() with Lomb_Scargle using `freq=lomb_scargle` with different values for its keyword arguments

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        ts = pinkseries
        psd = ts.spectral(method='mtm', freq='lomb_scargle', freq_kwargs={'dt': dt, 'nf': nf, 'ofac': ofac, 'hifac': hifac})
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-1.0) < eps

    def test_spectral_t5(self, pinkseries, eps=0.6):
        ''' Test Series.spectral() with LS with specified frequency vector passed via `settings`

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        Note `asser_array_equal(psd.frequency, freq)` is used to make sure the specified frequency vector is really working.
        Also, we give `label` a test.
        '''
        ts = pinkseries
        freq = np.linspace(1/500, 1/2, 100)
        psd = ts.spectral(method='lomb_scargle', freq=freq)
        beta = psd.beta_est(fmin=1/200, fmax=1/10).beta_est_res['beta']
        assert_array_equal(psd.frequency, freq)
        assert np.abs(beta-1.0) < eps

    @pytest.mark.parametrize('spec_method', ['wwz', 'lomb_scargle'])
    def test_spectral_t6(self, pinkseries, spec_method, eps=0.3):
        ''' Test Series.spectral() with WWZ and Lomb Scargle on unevenly-spaced data with default arguments

        We will estimate the scaling slope of an ideal colored noise to make sure the result is reasonable.
        '''
        ts = pinkseries
        # randomly remove some data pts
        n_del = 3
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly,auto_time_params=True)
        psd = ts2.spectral(method=spec_method)
        beta = psd.beta_est().beta_est_res['beta']
        assert np.abs(beta-1.0) < eps

    @pytest.mark.parametrize('spec_method', ['wwz','cwt'])
    def test_spectral_t7(self, pinkseries, spec_method):
        '''Test the spectral significance testing with pre-generated scalogram objects
        '''

        ts = pinkseries
        scal = ts.wavelet(method=spec_method)
        signif = scal.signif_test(number=2,export_scal = True)
        sig_psd = ts.spectral(method=spec_method,scalogram=scal)
        sig_psd.signif_test(number=2,scalogram=signif).plot()

class TestUISeriesBin:
    ''' Tests for Series.bin()

    Functions to test the various kwargs arguments for binning a timeseries
    '''

    def test_bin_t1(self, pinkseries):
        ''' Test the bin function with default parameter values'''
        ts = pinkseries
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts2_bin=ts2.bin(keep_log=True)
        print(ts2_bin.log[-1])

    def test_bin_t2(self, pinkseries):
        ''' Test the bin function by passing arguments'''
        ts = pinkseries
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

class TestUISeriesStats:
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

class TestUISeriesCenter:
    '''Test for Series.center()

    Center removes the mean, so we'll simply test maximum and minimum values'''

    def test_center(self, pinkseries):
        # Generate sample data
        ts = pinkseries

        # Call function to be tested
        tsc = ts.center(keep_log=True)

        assert np.abs(tsc.value.mean()) <= np.sqrt(sys.float_info.epsilon)

        #assert ts.value.mean() == tsc.log[1]['previous_mean']

class TestUISeriesStandardize:
    '''Test for Series.standardize()

    Standardize normalizes the series object, so we'll simply test maximum and minimum values'''

    def test_standardize(self, pinkseries, eps=0.1):     
        ts = pinkseries
        ts_std = ts.standardize(keep_log=False)
        assert np.abs(ts_std.value.std()-1) < eps # current

class TestUISeriesClean:
    '''Test for Series.clean()

    Since Series.clean() is intended to order the time axis,
    we will test the first and last values on the time axis and ensure that the length
    of the time and value sections are equivalent'''

    def test_clean(self):

        #Generate data
        ts = gen_normal()

        #Call function for testing
        ts_clean = ts.clean(keep_log=True)

        #Isolate time and value components
        time = ts_clean.__dict__['time']
        value = ts_clean.__dict__['value']

        assert time[len(time) - 1] >= time[0]
        assert len(time) == len(value)

class TestUISeriesGaussianize:
    '''Test for Series.gaussianize()

    Gaussianize normalizes the series object, so we'll simply test maximum and minimum values'''
    def test_gaussianize(self, pinkseries):
        ts = pinkseries
        ts_gauss = ts.gaussianize(keep_log=True)

        value = ts.__dict__['value']
        value_std = ts_gauss.__dict__['value']

        assert max(value) > max(value_std)
        assert min(value) < min(value_std)

class TestUISeriesSegment:
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
        
    def test_segment_t2(self):
        '''Test that in the case of segmentation, segment returns a Multiple Series object'''
        t = (1,2,3000)
        v = (1,2,3)

        ts = pyleo.Series(time = t, value = v, label = 'series')

        ts_seg = ts.segment()

        assert ts_seg.series_list[0].label == 'series segment 1'

class TestUISeriesSlice:
    '''Test for Series.slice()

    We commit slices at known time intervals and check minimum and maximum values'''

    def test_slice(self):
        ts = gen_normal(nt=100)
        ts_slice = ts.slice(timespan = (10, 50, 80, 90))
        times = ts_slice.__dict__['time']

        assert min(times) == 10
        assert max(times) == 90

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
        ]
    )
    def test_value(self, value, expected_time, expected_value, tolerance):
        ts = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years', verbose=False)
        result = ts.sel(value=value, tolerance=tolerance)
        expected = pyleo.Series(time=expected_time, value=expected_value, time_unit='years', verbose=False)
        values_match, _ = result.equals(expected)
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
    @pytest.mark.xfail  # ask MARCO
    def test_time(self, time, expected_time, expected_value, tolerance):
        ts = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years CE')
        result = ts.sel(time=time, tolerance=tolerance)
        expected = pyleo.Series(time=expected_time, value=expected_value, time_unit='years CE')
        values_match, _ = result.equals(expected)
        assert values_match
    
    def test_invalid(self):
        ts = pyleo.Series(time=np.array([1, 2, 3]), value=np.array([4, 6, 1]), time_unit='years')
        with pytest.raises(TypeError, match="Cannot pass both `value` and `time`"):
            ts.sel(time=1, value=1)



            

class TestUISeriesSummaryPlot:
    ''' Test Series.summary_plot()
    '''
    def test_summary_plot_t0(self, pinkseries):
        '''Testing that labels are being passed and that psd and scalogram objects dont fail when passed.
        Also testing that we can specify fewer significance tests than those stored in the scalogram object
        Note that we should avoid pyleo.showfig() in tests.

        Passing pre generated scalogram and psd.
        '''
        ts = pinkseries
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

        pyleo.closefig(fig)

    def test_summary_plot_t1(self, pinkseries):
        '''Testing that the bare function works
        Note that we should avoid pyleo.showfig() in tests.

        Passing just a pre generated psd.
        '''
        ts = pinkseries
        scal = ts.wavelet(method='cwt')
        psd = ts.spectral(method='cwt')
        fig, ax = ts.summary_plot(psd=psd,scalogram=scal)
        pyleo.closefig(fig)


class TestUISeriesCorrelation:
    ''' Test Series.correlation()
    '''
    @pytest.mark.parametrize('sig_method', ['ttest','built-in','ar1sim','phaseran','CN'])
    @pytest.mark.parametrize('number', [2,5])
    def test_correlation_t0a(self, sig_method, number, eps=0.1):
        ''' Test the various significance methods
        '''
        nt = 100
        rho = 0.4 # target correlation
        ts1 = gen_ts(nt=nt,alpha=1,seed=333).standardize()
        # generate series whose correlation with ts1 should be close to rho:
        v = rho*ts1.value + np.sqrt(1-rho**2)*np.random.normal(loc=0, scale=1, size=nt)
        ts2 = pyleo.Series(time=ts1.time, value=v, verbose=False, auto_time_params=True)

        corr_res = ts1.correlation(ts2, method= sig_method, number=number)
        assert np.abs(rho-corr_res.r) < eps
        
    @pytest.mark.parametrize('sig_method', ['isopersistent', 'isospectral'])
    def test_correlation_t0b(self, sig_method,eps=0.1):
        ''' Test that deprecated method names get a proper warning
        '''
        nt = 100
        rho = 0.4 # target correlation
        ts1 = gen_ts(nt=nt,alpha=1,seed=333).standardize()
        v = rho*ts1.value + np.sqrt(1-rho**2)*np.random.normal(loc=0, scale=1, size=nt)
        ts2 = pyleo.Series(time=ts1.time, value=v, verbose=False, auto_time_params=True)
        with pytest.deprecated_call():
            corr_res = ts1.correlation(ts2, method= sig_method, number=2)
        assert np.abs(rho-corr_res.r) < eps

    @pytest.mark.parametrize('sig_method', ['ttest','built-in','ar1sim','phaseran'])
    def test_correlation_t1(self, sig_method, eps=0.5):
        ''' Test correlation between two series with inconsistent time axis
        '''
        
        nino_ts = pyleo.utils.load_dataset('NINO3') 
        air_ts = pyleo.utils.load_dataset('AIR')
        t = nino_ts.time
        nino = nino_ts.value
        air = air_ts.value
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
        corr_res_evenly = ts1_evenly.correlation(ts2_evenly, method=sig_method,number=2)
        r_evenly = corr_res_evenly.r

        ts1 = pyleo.Series(time=air_time_unevenly, value=air_value_unevenly, verbose=False, auto_time_params=True)
        ts2 = pyleo.Series(time=nino_time_unevenly, value=nino_value_unevenly, verbose=False, auto_time_params=True)

        corr_res = ts1.correlation(ts2, method=sig_method, number=2, common_time_kwargs={'method': 'interp'})
        r = corr_res.r
        assert np.abs(r-r_evenly) < eps
    
    @pytest.mark.parametrize('stat', ['linregress','pearsonr','spearmanr','pointbiserialr','kendalltau','weightedtau'])
    def test_correlation_t2(self, stat, eps=0.2):
        ''' Test that various statistics can be used
        https://docs.scipy.org/doc/scipy/reference/stats.html#association-correlation-tests
        '''
        nt = 100
        rho = 0.6 # target correlation
        ts1 = gen_ts(nt=nt,alpha=1,seed=333).standardize()
        # generate series whose correlation with ts1 should be close to rho:
        v = rho*ts1.value + np.sqrt(1-rho**2)*np.random.normal(loc=0, scale=1, size=nt)
        ts2 = pyleo.Series(time=ts1.time, value=v, verbose=False, auto_time_params=True)
        
        if stat == 'weightedtau':
            corr_res = ts1.correlation(ts2, statistic=stat,number = 2)
        else:
            corr_res = ts1.correlation(ts2, statistic=stat, method='built-in')
        assert np.abs(rho-corr_res.r) < eps
        
    def test_correlation_t3(self, eps=0.1, nt=10):
        ''' Test that t-test screams at the user if a non-pearson statistic is requested
        '''
        rho = 0.4 # target correlation
        ts1 = gen_ts(nt=nt, alpha=1,seed=333).standardize()
        # generate series whose correlation with ts1 should be close to rho:
        v = rho*ts1.value + np.sqrt(1-rho**2)*np.random.normal(loc=0, scale=1, size=nt)
        ts2 = pyleo.Series(time=ts1.time, value=v, verbose=False, auto_time_params=True)
        with pytest.raises(ValueError):  
            corr_res = ts1.correlation(ts2, statistic='kendalltau', method= 'ttest')
                                       
        
        

class TestUISeriesCausality:
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

        _ = ts1.causality(ts2, method=method)

    @pytest.mark.parametrize('method', ['liang', 'granger'])
    def test_causality_t1(self, method, eps=1):
        ''' Generate two series from a same basic series and calculate their correlation
            on a specified timespan
        Note: NO assert statements for this test yet
        '''
        alpha = 1
        nt = 100
        ts = gen_ts(nt=nt,alpha=alpha)
        v1 = ts.value + np.random.normal(loc=0, scale=1, size=nt)
        v2 = ts.value + np.random.normal(loc=0, scale=2, size=nt)

        ts1 = pyleo.Series(time=ts.time, value=v1)
        ts2 = pyleo.Series(time=ts.time, value=v2)

        _ = ts1.causality(ts2, method=method, timespan=(0, 67))

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
        ts_out = ts2.outliers(remove=remove_outliers)

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
        ts_out = ts2.outliers(method=method)

    @pytest.mark.parametrize('keep_log', [True,False])
    def test_outliers_t3(self,keep_log):

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
        ts_out = ts2.outliers(keep_log=keep_log)
    
    def test_outliers_t4(self):
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
        ts_out = ts2.outliers(method = 'kmeans', settings={'nbr_clusters':2, 'threshold':2})
    
    def test_outliers_t5(self):
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
        ts_out = ts2.outliers(method = 'DBSCAN', settings={'nbr_clusters':2})
        
    @pytest.mark.parametrize('LOF_param', [True,False])
    def test_outliers_t6(self,LOF_param):
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
        ts_out = ts2.outliers(method = 'kmeans', settings={'LOF':LOF_param})
        
    def test_outliers_t7(self):
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
        ts_out = ts2.outliers(method = 'kmeans', settings={'LOF':True, 'n_frac':0.8, 'contamination':0.1})
        

class TestUISeriesGkernel:
    ''' Unit tests for the TestUISeriesGkernel function
    '''
    def test_gkernel_t1(self):
        ''' Test the gkernel function with default parameter values'''

        ts = gen_ts(nt=550, alpha=1.0)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts_interp=ts2.gkernel(keep_log=True)

    def test_gkernel_t2(self):
        ''' Test the gkernel function with specified bandwidth'''

        ts = gen_ts(nt=550, alpha=1.0)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts_interp=ts2.gkernel(h=15)
        
    def test_gkernel_t3(self):
        ''' Test the gkernel function with specified step_style'''

        ts = gen_ts(nt=550, alpha=1.0)
        # randomly remove some data pts
        n_del = 50
        deleted_idx = np.random.choice(range(np.size(ts.time)), n_del, replace=False)
        t_unevenly =  np.delete(ts.time, deleted_idx)
        v_unevenly =  np.delete(ts.value, deleted_idx)

        ts2 = pyleo.Series(time=t_unevenly, value=v_unevenly)
        ts_interp=ts2.gkernel(step_style='median')

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
        ts_interp=ts.interp(keep_log=True)

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
        ts_detrend=ts2.detrend(method=detrend_method, keep_log=True)

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
        ts1 = gen_ts(model='colored_noise')
        ts2 = gen_ts(model='colored_noise')
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
        _ = ts3.wavelet_coherence(ts4,method='wwz')
  
    def test_xwave_t4(self):
       ''' Test Series.wavelet_coherence() with specified frequency parameters
       '''
       ts1 = gen_ts(model='colored_noise')
       ts2 = gen_ts(model='colored_noise')
       nf = 10
       fmin = 1/(len(ts1.time)//2)
       fmax = 10*fmin
       scal = ts1.wavelet_coherence(ts2,method='cwt',freq_kwargs={'fmin':fmin,'fmax':fmax,'nf':nf})  
       freq = pyleo.utils.wavelet.freq_vector_log(ts1.time, fmin=fmin, fmax=fmax, nf=nf)
       
       assert all(scal.frequency == freq)
            
    def test_xwave_t5(self):
       ''' Test Series.wavelet_coherence() with WWZ with specified ntau
       '''
       ts1 = gen_ts(model='colored_noise')
       ts2 = gen_ts(model='colored_noise')
       _ = ts1.wavelet_coherence(ts2,method='wwz',settings={'ntau':10})  
       
    def test_xwave_t6(self):
       ''' Test Series.wavelet_coherence() with WWZ with specified tau
       '''
       ts1 = gen_ts(model='colored_noise')
       ts2 = gen_ts(model='colored_noise')
       tau = ts1.time[::10]
       _ = ts1.wavelet_coherence(ts2,method='wwz',settings={'tau':tau})    


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
       
    @pytest.mark.parametrize('freq_meth', ['log', 'scale', 'nfft', 'welch'])
    def test_wave_t4(self,freq_meth):
       ''' Test Series.wavelet() with different mother wavelets
       '''
       ts = gen_ts(model='colored_noise',nt=200)
       _ = ts.wavelet(method='cwt',freq_method=freq_meth)

class TestUISeriesSsa():
    ''' Test the SSA functionalities
    '''

    def test_ssa_t0(self):
        ''' Test Series.ssa() with available methods using default arguments
        '''

        cn = gen_ts(model = 'colored_noise', nt= 500, alpha=1.0)

        res = cn.ssa()
        assert abs(res.pctvar.sum() - 100.0)<0.01

    @pytest.mark.parametrize('trunc',['var', 'kaiser'])
    def test_ssa_t1(self, trunc):
        '''Test Series.ssa() with various truncations
        '''
        ts = gen_ts(model = 'colored_noise', nt=100, alpha=1.0)
        ssa = ts.ssa(trunc=trunc)
        assert ssa.RCseries.label == 'SSA reconstruction (' + trunc + ')'

    def test_ssa_t2(self):
        '''Test Series.ssa() with Monte-Carlo truncation
        '''
        ts = gen_ts(model = 'colored_noise', nt=100, alpha=1.0)
        mc_ssa = ts.ssa(M=60, nMC=10, trunc='mcssa')
        assert mc_ssa.RCseries.label == 'SSA reconstruction (mcssa)'

    def test_ssa_t3(self):
        '''Test Series.ssa() with Knee truncation'''
        ts = pyleo.utils.load_dataset('SOI')
        ssa = ts.ssa(trunc='knee')
        knee = 12
        assert_array_equal(ssa.mode_idx, np.arange(knee+1))

    def test_ssa_t4(self):
        '''Test Series.ssa() with missing values
        '''
        soi = pyleo.utils.load_dataset('SOI')
        # erase 20% of values
        n = len(soi.value)
        missing = np.random.choice(n,np.floor(0.2*n).astype('int'),replace=False)
        soi_m = soi.copy()
        soi_m.value[missing] = np.nan  # put NaNs at the randomly chosen locations
        miss_ssa = soi_m.ssa()
        assert all(miss_ssa.eigvals >= 0)
        assert np.square(miss_ssa.RCseries.value - soi.value).mean() < 0.3

class TestUISeriesPlot:
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

class TestUISeriesStripes:
    '''Test for Series.stripes()'''
    @pytest.mark.parametrize('show_xaxis', [True, False])
    def test_stripes(self, show_xaxis):

        ts = gen_normal()

        fig, ax = ts.stripes(ref_period=[61,90], show_xaxis=show_xaxis)
        assert ax.xaxis.get_ticklabels() != []
        pyleo.closefig(fig)


class TestUISeriesHistplot:
    '''Test for Series.histplot()'''

    def test_histplot_t0(self, max_axis = 5):
        ts = gen_normal()

        fig, ax = ts.histplot()

        line = ax.lines[0]

        x_plot = line.get_xdata()
        y_plot = line.get_ydata()

        assert max(x_plot) < max_axis

        pyleo.closefig(fig)

    def test_histplot_t1(self, vertical = True):
        ts = gen_normal()

        fig, ax = ts.histplot(vertical=vertical)

        pyleo.closefig(fig)

class TestUISeriesFilter:
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
        ts_lp = ts.filter(cutoff_freq=15, method=method, keep_log=True)
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

class TestUISeriesConvertTimeUnit:
    '''Tests for Series.convert_time_unit'''

    @pytest.mark.parametrize('keep_log',[False,True])
    def test_convert_time_unit_t0(self,keep_log):
        ts = gen_ts(nt=100, alpha=1.0)
        ts.time_unit = 'kyr BP'
        ts_converted = ts.convert_time_unit('yr BP',keep_log)
        np.testing.assert_allclose(ts.time*1000,ts_converted.time,atol=1)
        
    def test_convert_time_unit_t1(self):
        ts = gen_ts(nt=100, alpha=1.0)
        ts.time_unit = 'Ma'
        ts_converted = ts.convert_time_unit('yr BP')
        np.testing.assert_allclose(ts.time*1e6,ts_converted.time,atol=1)

    def test_convert_time_unit_t2(self):
        ts = gen_ts(nt=100, alpha=1.0)
        ts.time_unit = 'year'
        ts.time += 1950
        tsBP = ts.convert_time_unit('yr BP')
        assert tsBP.time_name == 'Age' # should infer time_name correctly   

class TestUISeriesFillNA:
    '''Tests for Series.fill_na'''

    @pytest.mark.parametrize('timespan,nt,ts_dt,dt', [(None,500,8,5),(None,500,1,2),([100,400],500,4,2)])
    def test_fill_na_t0(self,timespan,nt,ts_dt,dt):
        t = np.arange(0,nt,ts_dt)
        v = np.ones(len(t))
        ts = pyleo.Series(t,v)
        ts.fill_na(timespan=timespan,dt=dt)

    @pytest.mark.parametrize('keep_log', [True,False])
    def test_fill_na_t1(self,keep_log):
        t = np.arange(0,500,10)
        v = np.ones(len(t))
        ts = pyleo.Series(t,v)
        ts.fill_na(dt=5,keep_log=keep_log)

class TestUISeriesSort:
    '''Tests for Series.sort'''

    @pytest.mark.parametrize('keep_log',[True,False])
    def test_sort_t0(self,keep_log):
        ts = gen_ts(nt=50,alpha=1.0)
        ts = ts.sort()
        np.all(np.diff(ts.time) >= 0)

    def test_sort_t1(self):
        t = np.arange(50,0,-1)
        v = np.ones(len(t))
        ts = pyleo.Series(t,v)
        ts.sort()
        assert np.all(np.diff(ts.time) >= 0)

#@pytest.mark.xfail

class TestResample:
    
    @pytest.mark.parametrize('rule', pyleo.utils.tsbase.MATCH_A)
    def test_resample_simple(self, rule, dataframe_dt, metadata):
        ser = dataframe_dt.loc[:, 0]
        ts = pyleo.Series.from_pandas(ser, metadata)
        result = ts.resample(rule).mean()
        result_ser = result.to_pandas()
        expected_values = np.array([0., 1., 2., 3., 4.])
        expected_idx = pd.DatetimeIndex(
            ['2018-07-02T12:00:00', '2019-07-02T12:00:00', '2020-07-01T12:00:00', '2021-07-02T12:00:00', '2022-07-02T12:00:00'],
            name='datetime',
        ).as_unit('s')
        expected_ser = pd.Series(expected_values, expected_idx, name='SOI')
        expected_metadata = {
            'time_unit': 'years CE',
            'time_name': 'Time',
            'value_unit': 'mb',
            'value_name': 'SOI',
            'label': f'Southern Oscillation Index ({rule} resampling)',
            'archiveType': 'Instrumental',
            'importedFrom': None,
            'log': (
                    {0: 'dropna', 'applied': True, 'verbose': True},
                    {1: 'sort_ts', 'direction': 'ascending'}
                )
        }
        pd.testing.assert_series_equal(result_ser, expected_ser)
        assert result.metadata == expected_metadata

    
    # @pytest.mark.parametrize(
    #     ('rule', 'expected_idx', 'expected_values'),
    #     [
    #         (
    #             '1ga',
    #             pd.date_range(start = np.datetime64('500000000-01-01', 's'), end = np.datetime64('1500000000-01-01', 's'), freq='1000000000YS-JAN', unit='s'),
    #             np.array([0., 1.]),
    #         ),
    #         (
    #             '1ma',
    #             pd.date_range(np.datetime64('500000-01-01', 's'), np.datetime64('1000500000-01-01', 's'), freq='1000000YS-JAN', unit='s'),
    #             np.array([0.]+[np.nan]*999 + [1.]),
    #         ),
    #     ]
    # )
    # @pytest.mark.skip(reason="Known Pandas Bug")
    # def test_resample_long_periods(self, rule, expected_idx, expected_values, metadata):
    #     ser_index = pd.DatetimeIndex([
    #         np.datetime64('0000-01-01', 's'),
    #         np.datetime64('1000000000-01-01', 's'),
    #     ])
    #     ser = pd.Series(range(2), index=ser_index)
    #     ts = pyleo.Series.from_pandas(ser, metadata)
    #     result =ts.resample(rule).mean()
    #     result_ser = result.to_pandas()
    #     expected_idx = pd.DatetimeIndex(expected_idx, freq=None, name='datetime')
    #     expected_ser = pd.Series(expected_values, index=expected_idx, name='SOI')
    #     expected_metadata = {
    #         'time_unit': 'years CE',
    #         'time_name': 'Time',
    #         'value_unit': 'mb',
    #         'value_name': 'SOI',
    #         'label': f'Southern Oscillation Index ({rule} resampling)',
    #         'archiveType': 'Instrumental',
    #         'importedFrom': None,
    #         'log': (
    #                 {0: 'dropna', 'applied': True, 'verbose': True},
    #                 {1: 'sort_ts', 'direction': 'ascending'}
    #             )
    #     }
    #     # check indexes match to within 10 seconds
    #     assert np.abs(result_ser.index.to_numpy() - expected_ser.index.to_numpy()).max() <= 10
    #     np.testing.assert_array_equal(result_ser.to_numpy(), expected_ser.to_numpy())
    #     assert result.metadata == expected_metadata
 
 
    def test_resample_invalid(self, dataframe_dt, metadata):
        ser = dataframe_dt.loc[:, 0]
        ts = pyleo.Series.from_pandas(ser, metadata)
        with pytest.raises(ValueError, match='Invalid frequency: foo'):
            ts.resample('foo').sum()
        with pytest.raises(ValueError, match='Invalid rule provided, got: 412'):
            ts.resample('412').sum()
    

    # def test_resample_interpolate(self, metadata):
    #     ser_index = pd.DatetimeIndex([
    #         np.datetime64('0000-01-01', 's'),
    #         np.datetime64('2000-01-01', 's'),
    #     ])
    #     ser = pd.Series(range(2), index=ser_index)
    #     ts = pyleo.Series.from_pandas(ser, metadata)
    #     result_ser = ts.resample('ka').interpolate().to_pandas()
    #     expected_idx = pd.DatetimeIndex(
    #         [
    #             np.datetime64('499-12-31 12:00:00', 's'),
    #             np.datetime64('1500-01-01 12:00:00', 's'),
    #             np.datetime64('2499-12-31 12:00:00', 's')
    #         ],
    #         name='datetime'
    #     )
    #     expected_ser = pd.Series([0, 0.5, 1], name='SOI', index=expected_idx)
    #     pd.testing.assert_series_equal(result_ser, expected_ser)


    # @pytest.mark.parametrize(
    #     ['rule', 'expected_idx', 'expected_values'],
    #     (
    #         (
    #             'MS',
    #             [0.9596372 , 1.04451238, 1.12938757, 1.20604903],
    #             [8., 0., 3., 5.],
    #         ),
    #         (
    #             'SMS',
    #             [0.97880256, 1.02534702, 1.06367775, 1.11022221, 1.14855294, 1.18688367],
    #             [8., 0., 0., 3., 0., 5.],
    #         ),
    #     )
    # )
    # def test_resample_non_pyleo_unit(self, rule, expected_idx, expected_values):
    #     ts1 = pyleo.Series(time=np.array([1, 1.1, 1.2]), value=np.array([8, 3, 5]), time_unit='yr CE')
    #     result= ts1.resample(rule).sum()
    #     expected = pyleo.Series(
    #         time=np.array(expected_idx),
    #         value=np.array(expected_values),
    #         time_unit='yr CE',
    #     )
    #     assert result.equals(expected) == (True, True)
        
    # def test_resample_log(self, metadata):
    #     ser_index = pd.DatetimeIndex([
    #         np.datetime64('0000-01-01', 's'),
    #         np.datetime64('2000-01-01', 's'),
    #     ])
    #     ser = pd.Series(range(2), index=ser_index)
    #     ts = pyleo.Series.from_pandas(ser, metadata)
    #     result_ser = ts.resample('ka',keep_log=True).interpolate()
    #     expected_log = ({0: 'dropna', 'applied': True, 'verbose': True},
    #                     {1: 'sort_ts', 'direction': 'ascending'},
    #                     {2: 'resample', 'rule': '1000AS'})
    #     assert result_ser.log == expected_log
    

    def test_resample_retrograde(self):
        ts1 = pyleo.Series(
            time=np.array([-3, -2, -1]),
            value=np.array([8, 3, 5]),
            time_unit='yrs BP',
        )
        result = ts1.resample('Y').mean().to_pandas()
        expected = pd.Series(
            [5.5, 5],
            index=pd.DatetimeIndex(['1952-07-01 12:00:00', '1951-07-02 12:00:00'], name='datetime').as_unit('s')
        )
        pd.testing.assert_series_equal(result, expected)


class TestUISeriesEquals():
    ''' Test for equals() method '''
    @pytest.mark.parametrize('ds_name',['SOI','NINO3'])
    def test_equals_t0(self, ds_name):
        # test equality of data when true
        ts1 = pyleo.utils.load_dataset('SOI')
        ts2 = pyleo.utils.load_dataset(ds_name)

        same_data, _ = ts1.equals(ts2)
        if ds_name == 'SOI':
            assert same_data
        else:
            assert not same_data

    def test_equals_t1(self):
        # test equality of metadata
        ts1 = pyleo.utils.load_dataset('SOI')
        ts2 = ts1.copy()
        ts2.label = 'Counterfeit SOI'
        same_data, same_metadata = ts1.equals(ts2)
        assert not same_metadata

    def test_equals_t2(self):
        # test value tolerance
        tol = 1e-3
        ts1 = pyleo.utils.load_dataset('SOI')
        ts2 = ts1.copy()
        ts2.value[0] = ts1.value[0]+tol
        same_data, _ = ts1.equals(ts2, value_tol= 2*tol)
        assert same_data

    def test_equals_t3(self):
        # test index tolerance
        soi = pyleo.utils.load_dataset('SOI')
        soi_pd = soi.to_pandas()
        soi_pd.index = soi_pd.index + pd.DateOffset(1)
        soi2 = pyleo.Series.from_pandas(soi_pd, soi.metadata)
        same_data, _ = soi.equals(soi2, index_tol= 1.1*86400)
        assert same_data
        
