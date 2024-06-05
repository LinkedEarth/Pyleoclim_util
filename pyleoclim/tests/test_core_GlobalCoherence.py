''' Tests for pyleoclim.core.globalcoherence.GlobalCoherence

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

import pytest
import pyleoclim as pyleo

class TestUiGlobalCoherencePlot:
    ''' Tests for GlobalCoherence.plot()
    '''

    def test_plot_t0(self, gen_ts):
        ''' Test GlobalCoherence.plot with various parameters
        '''
        ts1 = gen_ts
        ts2 = gen_ts
        coh = ts1.global_coherence(ts2)
        fig,ax = coh.plot()
        pyleo.closefig(fig)

    def test_plot_t1(self, gen_ts):
        ''' Test GlobalCoherence.plot with signif tests
        '''
        ts1 = gen_ts
        ts2 = gen_ts
        coh = ts1.global_coherence(ts2).signif_test(number=1)
        fig,ax = coh.plot()
        pyleo.closefig(fig)

class TestUiGlobalCoherenceSignifTest:
    ''' Tests for GlobalCoherence.signif_test()
    '''

    @pytest.mark.parametrize('method',['ar1sim','phaseran','CN'])
    @pytest.mark.parametrize('number',[1,10])
    @pytest.mark.parametrize('qs',[[.95],[.05,.95]])
    def test_signiftest_t0(self,method,number, qs,gen_ts):
        ''' Test GlobalCoherence.signif_test
        '''
        ts1 = gen_ts
        ts2 = gen_ts
        _ = ts1.global_coherence(ts2).signif_test(method=method,number=number,qs=qs)