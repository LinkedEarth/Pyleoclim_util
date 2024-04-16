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


class TestUISurrogatesSeries:
    ''' Tests for pyleoclim.core.ui.SurrogateSeries
    '''
    @pytest.mark.parametrize('method',['ar1sim','uar1']) 
    def test_surrogates_ar1(self, method, p=5, eps=0.2, seed = 108):
        ''' Generate AR(1) surrogates based on a AR(1) series with certain parameters,
        estimate the parameters from the surrogates series and verify accuracy
        '''
        g = 0.5  # persistence
        ar = [1, -g]
        ma = [1, 0]
        n = 1000
        ar1 = arma_generate_sample(ar, ma, nsample=n, scale=1)
        ts = pyleo.Series(time=np.arange(1000), value=ar1)

        ts_surrs = ts.surrogates(number=p)
        for ts_surr in ts_surrs.series_list:
            g_surr = tsmodel.ar1_fit(ts_surr.value)
            assert np.abs(g_surr-g) < eps
            
    @pytest.mark.parametrize('number',[1,5])     
    def test_surrogates_uar1_match(self, number):
        ts = gen_ts(nt=550, alpha=1.0)
        # generate surrogates
        surr = ts.surrogates(method = 'uar1', number = number, time_pattern ="match")
        for i in range(number):
            assert(np.allclose(surr.series_list[i].time, ts.time))
            
    def test_surrogates_uar1_even(self, p=5):
        ts = gen_ts(nt=550, alpha=1.0)
        time_incr = np.median(np.diff(ts.time))
        # generate surrogates
        surr = ts.surrogates(method = 'uar1', number = p, time_pattern ="even", settings={"time_increment" :time_incr})
        for i in range(p):
            assert(np.allclose(tsmodel.inverse_cumsum(surr.series_list[i].time),time_incr))

    def test_surrogates_uar1_random(self, p=5, tol = 0.5):
        tau = 2
        sigma_2 = 1
        n = 500
        # generate time index
        t = np.arange(1,(n+1))
        # create time series
        ys = tsmodel.uar1_sim(t, tau_0=tau, sigma_2_0=sigma_2)
        ts = pyleo.Series(time = t, value=ys, auto_time_params=True,verbose=False)
        # generate surrogates default is exponential with parameter value 1
        surr = ts.surrogates(method = 'uar1', number = p, time_pattern ="random")
        #surr = ts.surrogates(method = 'uar1', number = p, time_pattern ="uneven",settings={"delta_t_dist" :"poisson","param":[1]} )
    
        for i in range(p):
            delta_t = tsmodel.inverse_cumsum(surr.series_list[i].time)
            # Compute the empirical cumulative distribution function (CDF) of the generated data
            empirical_cdf, bins = np.histogram(delta_t, bins=100, density=True)
            empirical_cdf = np.cumsum(empirical_cdf) * np.diff(bins)
            # Compute the theoretical CDF of the Exponential distribution
            theoretical_cdf = expon.cdf(bins[1:], scale=1)
            # Trim theoretical_cdf to match the size of empirical_cdf
            theoretical_cdf = theoretical_cdf[:len(empirical_cdf)]
            # Compute the L2 norm (Euclidean distance) between empirical and theoretical CDFs
            l2_norm = np.linalg.norm(empirical_cdf - theoretical_cdf)
            assert(l2_norm<tol)