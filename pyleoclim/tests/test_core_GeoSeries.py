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

class TestUIGeoSeriesInit:
    ''' Test for GeoSeries instantiation '''
    
    def test_init_no_dropna_depth(self, evenly_spaced_series):
            ts = evenly_spaced_series
            t = ts.time
            v = ts.value
            d = np.arange(len(t))
            v[0] = np.nan
            ts2 = pyleo.GeoSeries(time=t,value=v,depth=d,dropna=False, verbose=False,lat=0,lon=0)
            assert np.isnan(ts2.value[0])
            assert ts2.depth[0] == d[0]
        
    def test_init_dropna_depth(self, evenly_spaced_series):
            ts = evenly_spaced_series
            t = ts.time
            v = ts.value
            d = np.arange(len(t))
            v[0] = np.nan
            ts2 = pyleo.GeoSeries(time=t,value=v,depth=d,dropna=True, verbose=False,lat=0,lon=0)
            print(ts2.value)
            assert ~np.isnan(ts2.value[0])
            assert ts2.depth[0] == d[1]

    def test_init_no_dropna(self, evenly_spaced_series):
            ts = evenly_spaced_series
            t = ts.time
            v = ts.value
            v[0] = np.nan
            ts2 = pyleo.GeoSeries(time=t,value=v,dropna=False, verbose=False,lat=0,lon=0)
            assert np.isnan(ts2.value[0])
        
    def test_init_dropna(self, evenly_spaced_series):
            ts = evenly_spaced_series
            t = ts.time
            v = ts.value
            v[0] = np.nan
            ts2 = pyleo.GeoSeries(time=t,value=v,dropna=True, verbose=False,lat=0,lon=0)
            print(ts2.value)
            assert ~np.isnan(ts2.value[0])

#@pytest.mark.xfail   # will fail until pandas is fixed
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
        
    @pytest.mark.parametrize('title',[None, False, True, 'Untitled'])    
    def test_map_neighbors_t2(self, title):
        PLOT_DEFAULT = pyleo.utils.lipdutils.PLOT_DEFAULT
        ntypes = len(PLOT_DEFAULT)

        lat = np.random.uniform(20,70,ntypes)
        lon = np.random.uniform(-20,60,ntypes)

        dummy = [1, 2, 3]

        ts = pyleo.GeoSeries(time = dummy, value=dummy, lat=lat.mean(), lon=lon.mean(),
                             auto_time_params=True, verbose=False, archiveType='Wood',
                             label='Random Tree')
        series_list = []
        for i, key in enumerate(PLOT_DEFAULT.keys()):
            ser = ts.copy()
            ser.lat=lat[i]
            ser.lon=lon[i]
            ser.archiveType=key
            ser.label=key
            series_list.append(ser)
            
        mgs = pyleo.MultipleGeoSeries(series_list,time_unit='Years CE', label = 'multi-archive maelstrom')

        fig, ax = ts.map_neighbors(mgs,radius=5000, title = title)
        
        if title is None or title == False:
            assert ax['map'].get_title() == ''
        elif title == True:
            assert ax['map'].get_title() == 'multi-archive maelstrom neighbors for Random Tree within 5000 km'
        else:
            ax['map'].get_title() == 'Untitled'
        pyleo.closefig(fig)

class TestUIGeoSeriesMap():
    ''' test GeoSeries.map()
    '''
    
    def test_map_t0(self, pinkgeoseries):
        ts = pinkgeoseries
        fig, ax = ts.map()
        pyleo.closefig(fig)
        
    @pytest.mark.parametrize('title',[None, False, True, 'Untitled'])    
    def test_map_t1(self, pinkgeoseries, title):
        ts = pinkgeoseries
        fig, ax = ts.map(title=title)
        if title is None or title == False:
            assert ax['map'].get_title() == ''
        elif title == True:
            assert ax['map'].get_title() == 'pink noise geoseries location'
        else:
            ax['map'].get_title() == 'Untitled'
        pyleo.closefig(fig)
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
    
class TestUIGeoSeriesDashboard():
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
     
        
