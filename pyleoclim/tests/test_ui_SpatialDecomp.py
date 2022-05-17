''' Tests for pyleoclim.core.ui.SSARes

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
import numpy as np


# Tests below

         
class TestUiSpatialDecompScreeplot:
    ''' Tests for SpatialDecomp.screeplot()
    '''

    def test_plot_t0(self):
        ''' Test SpatialDecomp.screeplot() with default parameters

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
        
        fig, ax = res.screeplot(mute=True)
        
        
class TestUipatialDecompModeplot:
    ''' Tests for SSARes.modeplot()
    '''
    @pytest.mark.parametrize('spec_method', ['mtm', 'welch', 'periodogram','wwz'])
    def test_plot_t0(self,spec_method):
       '''
       Test with synthetic data, non missing values, all allowable spectral methods

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
       fig, ax = res.modeplot(mute=True,spec_method=spec_method)
         
        
    def test_plot_t1(self):
       '''
       Test with synthetic data, non missing values, modeplot()

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
       fig, ax = res.modeplot(index=2,mute=True)
       
       
       # TODO: add test for maps, including different projections
        