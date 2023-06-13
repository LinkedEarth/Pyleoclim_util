#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:28:24 2021

@author: deborahkhider

Tests for pyleoclim.core.geoseries.GeoSeries

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

# here's the "fixture". Can't figure out how to make an actual pytest fixture take arguments
def multiple_pinkgeoseries(nrecs = 20, seed = 108, geobox=[-85.0,85.0,-180,180]):
    """Set of Multiple geoseries with 1/f (pink) temporal structure """
    nt = 200
    lats = np.random.default_rng(seed=seed).uniform(geobox[0],geobox[1],nrecs)
    lons = np.random.default_rng(seed=seed+1).uniform(geobox[2],geobox[3],nrecs)
    elevs = np.random.default_rng(seed=seed+2).uniform(0,4000,nrecs)
    unknowns = np.random.randint(0,len(elevs)-1, size=2)
    for ik in unknowns:
        elevs[ik]=None
    
    archives = np.random.default_rng(seed=seed).choice(list(pyleo.utils.PLOT_DEFAULT.keys())+[None],size=nrecs)
    obsTypes = np.random.default_rng(seed=seed).choice(['MXD', 'd18O', 'Sr/Ca', None],size=nrecs)
    
    ts_list = []
    for i in range(nrecs):
        t,v = pyleo.utils.gen_ts(model='colored_noise',alpha=1.0, nt=nt)
        ts = pyleo.GeoSeries(t,v, verbose=False, label = f'pink series {i}',
                             archiveType=archives[i], observationType=obsTypes[i],
                             lat=lats[i], lon = lons[i], elevation=elevs[i]).standardize()
        ts_list.append(ts)
        
    return pyleo.MultipleGeoSeries(ts_list, label='Multiple Pink GeoSeries')

class TestUIGeoSeriesResample():
    ''' test GeoSeries.Resample()
    '''
    def test_resample_edc(self):
        EDC = pyleo.utils.load_dataset('EDC-dD')
        EDC5k = EDC.resample('5ka').mean()
        res = EDC5k.resolution()
        assert np.isclose(res.describe()['mean'], 5000)
        assert res.describe()['variance'] < 1e-3

class TestUIGeoSeriesMapNeighbors():
    ''' test GeoSeries.map_neighbors()
    '''    
    def test_map_neighbors_t0(self, pinkgeoseries):
        ts = pinkgeoseries
        mgs = multiple_pinkgeoseries()
        fig, ax = ts.map_neighbors(mgs)
        pyleo.closefig(fig)
        
    def test_map_neighbors_t1(self, pinkgeoseries):
        ts = pinkgeoseries
        mgs = multiple_pinkgeoseries()
        fig, ax = ts.map_neighbors(mgs, radius=5000)
        pyleo.closefig(fig)

class TestUiGeoSeriesMap():
    ''' test GeoSeries.map()
    '''
    
    def test_map_t0(self, pinkgeoseries):
        ts = pinkgeoseries
        fig, ax = ts.map()
        pyleo.closefig(fig)
        
        
def test_segment():
    '''
        test GeoSeries.segment
    '''
    import numpy as np
    gs = pyleo.utils.datasets.load_dataset('EDC-dD')
    gs.value[4000:5000] = np.nan # cut a large gap in the middle
    mgs = gs.segment()
    assert np.array_equal(mgs.series_list[0].value,gs.value[:4000]) 
    assert np.array_equal(mgs.series_list[1].value,gs.value[5000:]) 
    
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
     
        
