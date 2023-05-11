#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:28:24 2021

@author: deborahkhider

Tests for pyleoclim.core.ui.LipdSeries

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
"""
import pytest
import pyleoclim as pyleo
import numpy as np
import pandas as pd
from urllib.request import urlopen
import json

class TestUIGeoSeriesResample():
    ''' test GeoSeries.Resample()
    '''
    def test_resample_edc(self):
        EDC = pyleo.utils.load_dataset('EDC-dD')
        EDC5k = EDC.resample('5ka').mean()
        res = EDC5k.resolution()
        assert np.isclose(res.describe()['mean'], 5000)
        assert res.describe()['variance'] < 1e-3
    

class TestUiGeoSeriesMap():
    ''' test GeoSeries.map()
    '''
    
    def test_map_t0(self, pinkgeoseries):
        ts = pinkgeoseries
        fig, ax = ts.map()
        pyleo.closefig(fig)

class TestUiGeoSeriesDashboard():
    ''' test GeoSeries.Dashboard
    '''
    
    def test_dashboard_t0(self):
        ts = pyleo.utils.datasets.load_dataset('EDC-dD')
        ts_interp =ts.convert_time_unit('kyr BP').interp(step=10)
        fig, ax = ts_interp.dashboard(spectralsignif_kwargs={'number':2})
        pyleo.closefig(fig)
    def test_dashboard_t1(self, pinkgeoseries):
        ts = pinkgeoseries
        fig, ax = ts.dashboard(spectralsignif_kwargs={'number':2})
        pyleo.closefig(fig)

#class TestUiLipdSeriesMapNearRecord():
    # '''Test LipdSeries.MapNear Record
    
    # Requires a dictionary of LiPDs and selection
    # '''
    
    # def test_mapNearRecord_t0(self):
    #     D=importLiPD()
    #     d=pyleo.Lipd(lipd_dict=D)
    #     ts = d.to_LipdSeries(number=6)
    #     res=ts.mapNearRecord(d)
    #     pyleo.closefig(res[0])
    
#     def test_mapNearRecord_t1(self):
#         D=importLiPD()
#         d=pyleo.Lipd(lipd_dict=D)
#         ts = d.to_LipdSeries(number=6)
#         res=ts.mapNearRecord(d,n=6)
#         pyleo.closefig(res[0])
    
#     def test_mapNearRecord_t2(self):
#         D=importLiPD()
#         d=pyleo.Lipd(lipd_dict=D)
#         ts = d.to_LipdSeries(number=6)
#         res=ts.mapNearRecord(d,radius=1000)
#         pyleo.closefig(res[0])
     
        
