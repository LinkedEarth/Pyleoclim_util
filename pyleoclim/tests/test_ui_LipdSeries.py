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

def get_ts():
    d=pyleo.Lipd('http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD98-2170.Stott.2004')
    ts=d.to_LipdSeries(number=5)
    return ts

class TestUiLipdSeriesMap():
    ''' test LipdSeries.map()
    '''
    
    def test_map_t0(self):
        ts=get_ts()
        res=ts.map(mute=True)

class TestUiLipdSeriesgetMetadata():
    ''' test LipdSeries.getMetadata
    '''
    
    def test_getMetadata_t0(self):
        ts=get_ts()
        res=ts.getMetadata()

class TestUiLipdSeriesDashboard():
    ''' test LipdSeries.Dashboard
    '''
    
    def test_dashboard_t0(self):
        ts=get_ts()
        res = ts.dashboard(mute=True)
    
