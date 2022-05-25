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


def gen_ts(model,nt):
    'wrapper for gen_ts in pyleoclim'
    
    t,v = pyleo.utils.gen_ts(model=model,nt=nt)
    ts=pyleo.Series(t,v)
    return ts

# Tests below
      
class TestUiCoherencePlot:
    ''' Tests for Coherence.plot()
    '''

    def test_plot_t0(self):
        ''' Test Coherence.plot with default parameters
        '''
        nt = 200
        ts1 = gen_ts(model='colored_noise', nt=nt)
        ts2 = gen_ts(model='colored_noise', nt=nt)
        coh = ts2.wavelet_coherence(ts1)
        fig,ax = coh.plot()
        pyleo.closefig(fig)
    
    def test_plot_t1(self):
        ''' Test Coherence.plot WTC with significance
        '''
        nt = 200
        ts1 = gen_ts(model='colored_noise', nt=nt)
        ts2 = gen_ts(model='colored_noise', nt=nt)
        coh = ts2.wavelet_coherence(ts1)
        
        coh_signif = coh.signif_test(number=10,qs = [0.8, 0.9, .95])
        fig,ax = coh_signif.plot(signif_thresh=0.99)
        pyleo.closefig(fig)
        
    def test_plot_t2(self):
        ''' Test Coherence.plot XWT with significance
        '''
        nt = 200
        ts1 = gen_ts(model='colored_noise', nt=nt)
        ts2 = gen_ts(model='colored_noise', nt=nt)
        coh = ts2.wavelet_coherence(ts1)
        
        coh_signif = coh.signif_test(number=10)
        fig,ax = coh_signif.plot(var='xwt')
        pyleo.closefig(fig)
        
class TestUiCoherenceDashboard:
    ''' Tests for Coherence.dashboard()
    '''        
    def test_dashboard_t0(self):
        ''' Test Coherence.dashboard() with default parameters
        '''
        nt = 200
        ts1 = gen_ts(model='colored_noise', nt=nt)
        ts2 = gen_ts(model='colored_noise', nt=nt)
        coh = ts2.wavelet_coherence(ts1)
        fig,ax  = coh.dashboard()
        pyleo.closefig(fig)
        
    def test_dashboard_t1(self):
        ''' Test Coherence.dashboard() with optional parameter
        '''
        nt = 200
        ts1 = gen_ts(model='colored_noise', nt=nt)
        ts2 = gen_ts(model='colored_noise', nt=nt)
        coh = ts2.wavelet_coherence(ts1)
        fig, ax = coh.dashboard(wavelet_plot_kwargs={'contourf_style':{'cmap': 'cividis'}})
        pyleo.closefig(fig)
        
        