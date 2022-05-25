#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:02:02 2021

@author: deborahkhider


Tests for pyleoclim.core.ui.Lipd

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
"""

import pytest
import pyleoclim as pyleo
from urllib.request import urlopen
import json

from pyleoclim.tests.examples import load_dataset

# For some of the testing importa JSON file with a dictionary of possible LiDPs

def importLiPD():
    url = 'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/lipds.json'
    response = urlopen(url)
    d = json.loads(response.read())
    return d
    
def load_data():
    #Loads stott MD982176 record
    try:
        d = pyleo.Lipd(usr_path='http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004')
    except:
        d = pyleo.Lipd('./example_data/MD982176.Stott.2004.lpd')
    return d

class TestUiLipdTo_tso():
    ''' Test Lipd.to_tso()
    '''
    def test_to_tso_t0(self):
        d=load_data()
        ts=d.to_tso()
        assert len(ts)>0

class TestUiLipdExtract():
    '''Test Lipd.extract()
    '''
    
    def test_extract_t0(self):
        D = importLiPD()
        d = pyleo.Lipd(lipd_dict=D)
        name = 'Eur-SpannagelCave.Mangini.2005'
        d2=d.extract(name)
        
        assert d2.__dict__['lipd']['dataSetName'] == name

class TestUiLipdTo_LipdSeriesList():
    ''' Test Lipd.to_LipdSeriesList
    '''
    def test_to_LipdSeriesList_t0(self):
        d=load_data()
        ts=d.to_LipdSeriesList()

class TestUiLipdTo_LipdSeries():
    ''' Test Lipd.to_LipdSeries
    '''
    def test_to_LipdSeries_t0(self):
        d=load_data()
        ts=d.to_LipdSeries(number=5)

class TestUiLipdMapAllArchive():
    ''' Test Lipd.mapAllArchive
    '''
    def test_mapAllArchive_t0(self):
        d=load_data()
        res = d.mapAllArchive(mute=True)