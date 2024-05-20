''' Tests for pyleoclim.core.ui.SurrogateSeries

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
import pytest
import pyleoclim as pyleo
import pyleoclim.utils.tsmodel as tsmodel
from numpy.testing import assert_array_equal
from scipy import stats

class TestSurrogateSeriesFromSeries:
    '''Tests for SurrogateSeries.from_series() method'''
    @pytest.mark.parametrize('method', ['uar1','ar1sim','phaseran'])
    @pytest.mark.parametrize('number', [1,10])
    @pytest.mark.parametrize('seed', [None,42])
    def test_fromseries_t0(self,gen_ts,method,number,seed):
        '''Test from_series method'''
        series = gen_ts
        surr = pyleo.SurrogateSeries(method=method,number=number,seed=seed)
        surr.from_series(target_series=series)

    @pytest.mark.parametrize('method',['ar1sim','uar1','phaseran','CN'])          
    @pytest.mark.parametrize('number',[1,5])     
    def test_from_series_match(self, method, number):
        ''' Test that from_series() works with all methods, matches the original
            time axis AND that number can be varied
        '''
        t, v = pyleo.utils.gen_ts(model='colored_noise', nt=100)
        ts = pyleo.Series(time = t, value = v, verbose=False, auto_time_params=True)
        surr = pyleo.SurrogateSeries(method=method, number=number) 
        surr.from_series(ts)
        assert len(surr.series_list) == number # test right number 
        for s in surr.series_list:
            assert_array_equal(s.time, ts.time) # test time axis match

class TestSurrogateSeriesFromParam:
    '''Tests for SurrogateSeries.from_param() method'''
    @pytest.mark.parametrize('method, noise_param', [('uar1',[1,5]),
                                                     ('CN',[1]),
                                                     ('ar1sim',[1,5]),])
    @pytest.mark.parametrize('number', [1,10])
    @pytest.mark.parametrize('seed', [None,42])
    @pytest.mark.parametrize('time_pattern,settings',[('even',None),
                                                       ('random',None),
                                                       ('specified',{'time':np.arange(50)})])
    def test_from_param_t0(self,method,noise_param,number,seed,time_pattern,settings):
        '''Test from_series method'''
        surr = pyleo.SurrogateSeries(method=method,number=number,seed=seed)
        surr.from_param(param=noise_param,time_pattern=time_pattern,settings=settings)

    @pytest.mark.parametrize('delta_t',[1,3]) 
    @pytest.mark.parametrize('length',[10,20]) 
    def test_from_param_even(self, delta_t, length, number=2):
        ''' Test from_param() with even time axes with varying resolution
        '''
        surr = pyleo.SurrogateSeries(method='ar1sim', number=number) 
        surr.from_param(param = [5,1], time_pattern='even', length= length,
                         settings={"delta_t" :delta_t})
        for ts in surr.series_list:
            assert(np.allclose(tsmodel.inverse_cumsum(ts.time),delta_t))
            assert len(ts.time) == length
    
    @pytest.mark.parametrize('method',['ar1sim','uar1']) 
    @pytest.mark.parametrize('param',[[5,1],[2,2]])  # stacking pytests is legal!
    def test_from_param_ar1(self, method, param, nsim=10, eps=1, seed = 108):
        ''' Generate AR(1) surrogates based on a AR(1) series with certain parameters,
        estimate the parameters from the surrogates series and verify accuracy
        '''
        
        ar1 = pyleo.SurrogateSeries(method=method, number=nsim, seed=seed) 
        ar1.from_param(param = param)
        tau = np.empty((nsim))
        sigma_2 = np.empty_like(tau)
        for i, ts in enumerate(ar1.series_list):
            tau[i], sigma_2[i] = tsmodel.uar1_fit(ts.value, ts.time)
        assert np.abs(tau.mean()-param[0]) < eps
        assert np.abs(sigma_2.mean()-param[1]) < eps

    @pytest.mark.parametrize('param',[1.0,[1.0]])    
    def test_from_param_CN_t0(self, param, nsim=10, eps=.1, seed = 108):
        ''' Generate colored noise surrogates based known parameters,
        estimate the parameters from the surrogates series and verify accuracy
        '''
        CN = pyleo.SurrogateSeries(method='CN', number=nsim, seed=seed) 
        CN.from_param(param = param, length=200)
        beta = np.empty((nsim))
        for i, ts in enumerate(CN.series_list):
            beta[i] = ts.interp().spectral(method='cwt').beta_est().beta_est_res['beta']
        assert np.abs(beta.mean()-1.0) < eps
        
    def test_from_param_CN_t1(self, nsim=10):
        ''' Generate colored noise surrogates with too many params; check error msg
        '''
        CN = pyleo.SurrogateSeries(method='CN', number=nsim) 
        with pytest.raises(ValueError):  
            CN.from_param(param = [2.0, 3.0], length=200) # check that this returns an error 
   
    @pytest.mark.parametrize('length', [10, 100])
    def test_from_param_specified_time(self, length, number=2):
        ''' Test AR(1) model with specified time axis
        '''
        ti = tsmodel.random_time_axis(n=length)
        surr = pyleo.SurrogateSeries(method='ar1sim', number=number, seed=108) 
        surr.from_param(param = [5,1], time_pattern='specified',
                         settings={"time":ti})
                
        for ts in surr.series_list:
            assert_array_equal(ts.time, ti) # test time axis match

    @pytest.mark.parametrize('dist', ["exponential", "poisson"])
    def test_from_param_random_time_t0(self, dist, number=2, tol = 3, param=[1]):
        ''' Test from_param() with random time axes with two 1-parameter distributions
        '''
        surr = pyleo.SurrogateSeries(method='ar1sim', number=number, seed=108) 
        surr.from_param(param = [5,1], length=200, time_pattern='random',
                         settings={"delta_t_dist" :dist ,"param":param})
        
        if dist == "exponential":
            scipy_dist = stats.expon
        else:
            scipy_dist = stats.poisson
            
        for i in range(number):
            delta_t = tsmodel.inverse_cumsum(surr.series_list[i].time)
            # Compute the empirical cumulative distribution function (CDF) of the generated data
            empirical_cdf, bins = np.histogram(delta_t, bins=100, density=True)
            empirical_cdf = np.cumsum(empirical_cdf) * np.diff(bins)
            # Compute the theoretical CDF of the Exponential distribution
            theoretical_cdf = scipy_dist.cdf(bins[1:],param[0])
            # Trim theoretical_cdf to match the size of empirical_cdf
            theoretical_cdf = theoretical_cdf[:len(empirical_cdf)]
            # Compute the L2 norm (Euclidean distance) between empirical and theoretical CDFs
            l2_norm = np.linalg.norm(empirical_cdf - theoretical_cdf)
            assert(l2_norm<tol)
            
    def test_from_param_random_time_t1(self, number=2, tol = 4, param=[4.2,2.5]):
        ''' Test random time axes with Pareto distribution
        '''
        surr = pyleo.SurrogateSeries(method='ar1sim', number=number, seed=108) 
        surr.from_param(param = [5,1], time_pattern='random',
                         settings={"delta_t_dist" :"pareto" ,"param":param})
        
        for i in range(number):
            delta_t = tsmodel.inverse_cumsum(surr.series_list[i].time)
            # Compute the empirical cumulative distribution function (CDF) of the generated data
            empirical_cdf, bins = np.histogram(delta_t, bins=100, density=True)
            empirical_cdf = np.cumsum(empirical_cdf) * np.diff(bins)
            # Compute the theoretical CDF of the Exponential distribution
            theoretical_cdf = stats.pareto.cdf(bins[1:],*param)
            # Trim theoretical_cdf to match the size of empirical_cdf
            theoretical_cdf = theoretical_cdf[:len(empirical_cdf)]
            # Compute the L2 norm (Euclidean distance) between empirical and theoretical CDFs
            l2_norm = np.linalg.norm(empirical_cdf - theoretical_cdf)
            assert(l2_norm<tol)

    def test_from_param_random_time_t2(self, number=2, tol = 3, param=[[1,2],[.80,.20]]):
        ''' Test AR(1) model w/ random time axes (uniform choice)
        '''
        surr = pyleo.SurrogateSeries(method='ar1sim', number=number, seed=108) 
        surr.from_param(param = [5,1], time_pattern='random',
                         settings={"delta_t_dist" :"random_choice" ,"param":param})
        
        for ts in surr.series_list:
            delta_t = tsmodel.inverse_cumsum(ts.time)
            assert all(np.isin(delta_t, [1., 2.]))
            
    def test_from_param_forbidden(self, number=1):
        ''' Test that phase randomization is rejected in from_param()
        '''
        surr = pyleo.SurrogateSeries(method='phaseran', number=number) 
        with pytest.raises(ValueError):  
            surr.from_param(param = [5,1]) # check that this returns an error    