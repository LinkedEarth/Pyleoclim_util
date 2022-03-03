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
from urllib.request import urlopen
import json

def importLiPD():
    url = 'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/lipds.json'
    response = urlopen(url)
    d = json.loads(response.read())
    return d

def importEnsLiPD():
    url = 'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/crystalcave_ens.json'
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
    
def get_ts():
    d=load_data()
    ts=d.to_LipdSeries(number=3)
    return ts

class TestUiLipdSeriesMap():
    ''' test LipdSeries.map()
    '''
    
    def test_map_t0(self):
        ts=get_ts()
        fig,ax=ts.map(mute=True)
        pyleo.closefig(fig)

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
        fig,ax = ts.dashboard(mute=True)
        pyleo.closefig(fig)

class TestUiLipdSeriesMapNearRecord():
    '''Test LipdSeries.MapNear Record
    
    Requires a dictionary of LiPDs and selection
    '''
    
    def test_mapNearRecord_t0(self):
        D=importLiPD()
        d=pyleo.Lipd(lipd_dict=D)
        ts = d.to_LipdSeries(number=6)
        res=ts.mapNearRecord(d,mute=True)
    
    def test_mapNearRecord_t1(self):
        D=importLiPD()
        d=pyleo.Lipd(lipd_dict=D)
        ts = d.to_LipdSeries(number=6)
        res=ts.mapNearRecord(d,n=6,mute=True)
    
    def test_mapNearRecord_t2(self):
        D=importLiPD()
        d=pyleo.Lipd(lipd_dict=D)
        ts = d.to_LipdSeries(number=6)
        res=ts.mapNearRecord(d,radius=1000,mute=True)

class TestUiLipdSeriesChronEnsembleToPaleo():
    ''' Test the ability to get the chron ensemble tables
    '''

    def test_chronEnsembletoPaleo_t0(self):
        D=importEnsLiPD()
        d=pyleo.Lipd(lipd_dict=D)
        ts = d.to_LipdSeries(number=2)
        ens = ts.chronEnsembleToPaleo(d)
        
        assert type(ens)==pyleo.core.ui.EnsembleSeries
        
        
class TestUiLipdSeriesPlotAgeDepth():
    '''test LipdSeries.plot_age_depth
    '''

    def test_plot_age_depth_t0(self):
        D=importEnsLiPD()
        d=pyleo.Lipd(lipd_dict=D)
        ts = d.to_LipdSeries(number=2)
        fig,ax = ts.plot_age_depth()
        pyleo.closefig(fig)
    
    @pytest.mark.parametrize('traces', [10,0])
    def test_plot_age_depth_t1(self,traces):
        D=importEnsLiPD()
        d=pyleo.Lipd(lipd_dict=D)
        ts = d.to_LipdSeries(number=2)
        fig,ax = ts.plot_age_depth(ensemble=True, D=d, num_traces=traces)
        pyleo.closefig(fig)
        