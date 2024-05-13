''' Tests for pyleoclim.core.ui.EnsembleGeoSeries

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test locally:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import pytest
import numpy as np
import pyleoclim as pyleo

class TestUIEnsembleGeoSeriesDashboard():
    '''Tests for the EnsembleGeoSeries.dashboard function'''
    def test_dashboard_t0(self,ensemblegeoseries_basic):
        ens = ensemblegeoseries_basic
        fig,_ = ens.dashboard()
        pyleo.closefig(fig)

    def test_dashboard_t1(self,ensemblegeoseries_nans):
        ens = ensemblegeoseries_nans
        fig,_ = ens.dashboard()
        pyleo.closefig(fig)

class TestUIEnsembleGeoSeriesfromAgeEnsembleArray():
    def test_fromAgeEnsembleArray_t0(self,pinkgeoseries):
        series = pinkgeoseries
        length = len(series.time)
        num = 3
        age_array = np.array([np.arange(length) for _ in range(num)]).T
        _ = pyleo.EnsembleGeoSeries.from_AgeEnsembleArray(geo_series=series, age_array=age_array)

    def test_fromAgeEnsembleArray_t1(self,pinkgeoseries):
        series = pinkgeoseries
        length = len(series.time)
        value_depth = np.arange(length)
        age_depth = np.arange(length)
        num = 3
        age_array = np.array([np.arange(length) for _ in range(num)]).T
        _ = pyleo.EnsembleGeoSeries.from_AgeEnsembleArray(geo_series=series, age_array=age_array,value_depth=value_depth,age_depth=age_depth)

    def test_fromAgeEnsembleArray_t1(self,pinkgeoseries):
        series = pinkgeoseries
        length = len(series.time)
        series.depth = np.arange(length)
        age_depth = np.arange(length)
        num = 3
        age_array = np.array([np.arange(length) for _ in range(num)]).T
        _ = pyleo.EnsembleGeoSeries.from_AgeEnsembleArray(geo_series=series, age_array=age_array,age_depth=age_depth)