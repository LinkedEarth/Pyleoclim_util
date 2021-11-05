''' Tests for pyleoclim.core.ui.CorrEns class

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
import matplotlib.pyplot as plt

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

import pytest

import pyleoclim as pyleo


# Tests below
class TestUiCorrEns():
    def test_plot_t0(self):
        ''' Test CorrEns.plot() for multiple plots
            (qui peut le plus peut le moins)
        '''
        nn = 20 # number of noise realizations
        nt = 200
        
        signal = pyleo.gen_ts(model='colored_noise',nt=nt,alpha=1.0).standardize() 
        noise = np.random.randn(nt,nn)

        list1 = []
        list2 = []
        nhlf = int(nn/2)
        for idx in range(nhlf):  
            ts1 = pyleo.Series(time=signal.time, value=signal.value+noise[:,idx])  
            list1.append(ts1)
            ts2 = pyleo.Series(time=signal.time, value=signal.value+noise[:,idx+nhlf])
            list2.append(ts2)

        ts_ens1 = pyleo.EnsembleSeries(list1)
        ts_ens2 = pyleo.EnsembleSeries(list2)

        fig, axs = plt.subplots(1,2)
        corr1 = ts_ens1.correlation(signal,settings={'nsim':100})
        corr1.plot(ax=axs[0],mute=True)
        corr2 = ts_ens2.correlation(signal,settings={'nsim':100})
        corr2.plot(ax=axs[1],mute=True)
        #pyleo.showfig(fig)  # debug only
        

