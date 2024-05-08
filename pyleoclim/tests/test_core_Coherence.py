''' Tests for pyleoclim.core.ui.Coherence

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
#import pandas as pd

import pytest
import pyleoclim as pyleo

# Tests below
      
class TestUiCoherencePlot:
    ''' Tests for Coherence.plot()
    '''

    def test_plot_t0(self, gen_ts):
        ''' Test Coherence.plot with default parameters
        '''
        ts1 = gen_ts
        ts2 = gen_ts
        coh = ts2.wavelet_coherence(ts1)
        fig,ax = coh.plot()
        pyleo.closefig(fig)
        
    @pytest.mark.parametrize('method', ['ar1sim','phaseran','uar1','CN'])
    def test_plot_t1(self, gen_ts, method):
        ''' Test Coherence.plot WTC with significance
        '''
        ts1 = gen_ts
        ts2 = gen_ts
        coh = ts2.wavelet_coherence(ts1)
        
        coh_signif = coh.signif_test(number=10, method=method, qs = [0.8, 0.9, .95])
        fig,ax = coh_signif.plot(signif_thresh=0.99)
        pyleo.closefig(fig)
        
    def test_plot_t2(self, gen_ts):
        ''' Test Coherence.plot XWT with significance
        '''
        ts1 = gen_ts
        ts2 = gen_ts
        coh = ts2.wavelet_coherence(ts1)
        
        coh_signif = coh.signif_test(number=10)
        fig,ax = coh_signif.plot(var='xwt')
        pyleo.closefig(fig)
        
class TestUiCoherenceDashboard:
    ''' Tests for Coherence.dashboard()
    '''        
    def test_dashboard_t0(self, gen_ts):
        ''' Test Coherence.dashboard() with default parameters
        '''
        ts1 = gen_ts
        ts2 = gen_ts
        coh = ts2.wavelet_coherence(ts1)
        fig,ax  = coh.dashboard()
        pyleo.closefig(fig)
        
    def test_dashboard_t1(self, gen_ts):
        ''' Test Coherence.dashboard() with optional parameter
        '''
        ts1 = gen_ts
        ts2 = gen_ts
        coh = ts2.wavelet_coherence(ts1)
        fig, ax = coh.dashboard(wavelet_plot_kwargs={'contourf_style':{'cmap': 'cividis'}})
        pyleo.closefig(fig)
        
class TestUiCoherencePhaseStats:
    ''' Tests for Coherence.phase_stats()
    '''        
    def test_phasestats_t0(self, gen_ts):
        ''' Test Coherence.phase_stats() with default parameters
        '''
        ts1 = gen_ts
        ts2 = gen_ts
        coh = ts2.wavelet_coherence(ts1)
        _ = coh.phase_stats(scales=[2,8])