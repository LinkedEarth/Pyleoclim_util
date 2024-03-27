''' Tests for pyleoclim.core.ui.MultipleResolution

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

class TestUIMultipleResolutionSummaryPlot:
    @pytest.mark.parametrize('ms_fixture', ['multipleseries_basic','multipleseries_nans'])
    def test_plot_t0(self,ms_fixture,request):
        '''
        test resolution plot
        '''
        ms = request.getfixturevalue(ms_fixture)
        msr = ms.resolution(statistic=None)
        fig,ax=msr.summary_plot()
        pyleo.closefig(fig)
    def test_plot_t1(self):
        '''
        test resolution plot with time unit
        '''
        co2ts = pyleo.utils.load_dataset('AACO2')
        lr04 = pyleo.utils.load_dataset('LR04')
        edc = pyleo.utils.load_dataset('EDC-dD')
        ms = lr04.flip() & edc & co2ts # create MS object
        msr = ms.resolution(time_unit='kyr BP',statistic=None)
        fig,ax=msr.summary_plot()
        pyleo.closefig(fig)

class TestUIMultipleResolutionDescribe:
    @pytest.mark.parametrize('ms_fixture', ['multipleseries_basic','multipleseries_nans'])
    def test_describe_t0(self,ms_fixture,request):
        '''
        test resolution describe
        '''
        ms = request.getfixturevalue(ms_fixture)
        msr = ms.resolution(statistic=None)
        msr.describe()
    def test_describe_t2(self):
        '''
        test resolution plot with time unit
        '''
        co2ts = pyleo.utils.load_dataset('AACO2')
        lr04 = pyleo.utils.load_dataset('LR04')
        edc = pyleo.utils.load_dataset('EDC-dD')
        ms = lr04.flip() & edc & co2ts # create MS object
        msr = ms.resolution(time_unit='kyr BP',statistic=None)
        msr.describe()