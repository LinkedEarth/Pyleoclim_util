#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:47:08 2023

@author: julieneg
"""
import pyleoclim as pyleo
import numpy as np
# from bs4 import BeautifulSoup
# import requests
import pytest

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

class TestUIMultipleGeoSeriesMap:
    def test_map_archives(self):
        '''
        test mapping semantics
        '''
        mgs = multiple_pinkgeoseries()
        fig, ax = mgs.map(marker = 'archiveType')
        pyleo.closefig(fig)
        # assert something?

    def test_map_obs(self):
        '''
        test mapping semantics
        '''
        mgs = multiple_pinkgeoseries()
        fig, ax = mgs.map(hue = 'observationType')
        pyleo.closefig(fig)
        # assert something?

    def test_map_elevation(self):
        '''
        test mapping semantics
        '''
        mgs = multiple_pinkgeoseries()
        fig, ax = mgs.map(hue = 'elevation')
        pyleo.closefig(fig)
        # assert something?
    
    @pytest.mark.parametrize('crit_dist',[5000,30000])
    def test_map_proj_pick_global(self, crit_dist):
        '''
        test automated projection picker
        '''
        mgs = multiple_pinkgeoseries()
        lat = np.array([ts.lat for ts in mgs.series_list])
        lon = np.array([ts.lon for ts in mgs.series_list])
        clat, clon = pyleo.utils.mapping.centroid_coords(lat, lon)
        #d =  pyleo.utils.mapping.compute_dist(clat, clon, lat, lon) # computes distances to centroid

        fig, ax_d = mgs.map(crit_dist=crit_dist)
        if crit_dist == 30000:
            assert ax_d['map'].projection.proj4_params['proj'] == 'ortho'
            assert ax_d['map'].projection.proj4_params['lon_0'] == clon
            assert ax_d['map'].projection.proj4_params['lat_0'] == clat
        else:
            assert ax_d['map'].projection.proj4_params['proj'] == 'robin'
            assert ax_d['map'].projection.proj4_params['lon_0'] == clon
        
    def test_map_proj_pick_regional(self):
        '''
        test automated projection picker
        '''
        mgs = multiple_pinkgeoseries(geobox=[20, 70, -10, 40])
        lat = [ts.lat for ts in mgs.series_list]
        lon = [ts.lon for ts in mgs.series_list]
        clat, clon = pyleo.utils.mapping.centroid_coords(lat, lon)
        fig, ax_d = mgs.map()
        assert ax_d['map'].projection.proj4_params['proj'] == 'ortho'
        assert ax_d['map'].projection.proj4_params['lon_0'] == clon
        assert ax_d['map'].projection.proj4_params['lat_0'] == clat
            
        
class TestUIMultipleGeoSeriesPCA:
    def test_pca_t0(self):
        '''
        test PCA output
        '''
        mgs = multiple_pinkgeoseries()
        pca = mgs.pca()
        assert pca.name == 'Multiple Pink GeoSeries PCA'
        
    def test_pca_t1(self):
        '''
        test PCA screeplot
        '''
        mgs = multiple_pinkgeoseries()
        pca = mgs.pca()
        fig, ax = pca.screeplot()
        
    def test_pca_t2(self):
        '''
        test PCA modeplot
        '''
        mgs = multiple_pinkgeoseries()
        pca = mgs.pca()
        fig, ax = pca.modeplot(index=1)

class TestUIMultipleGeoSeriesTimeGeoPlot:
    def test_time_geo_plot_t0(self):
        '''
        test PCA output
        '''
        mgs = multiple_pinkgeoseries()
        fig,ax = mgs.time_geo_plot()
        pyleo.closefig(fig)
        

# def create_Euro2k():
#     '''
#     Tests the creation of Euro2k MultipleGeoSeries
#     '''


#     url = 'https://github.com/LinkedEarth/Pyleoclim_util/tree/master/example_data/Euro2k_json'
#     ext = 'json'

#     def listFD(url, ext=''):
#         page = requests.get(url).text
#         soup = BeautifulSoup(page, 'html.parser')
#         return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

#     files = []
#     github_url = 'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/Euro2k_json/'
#     for file in listFD(url, ext):
#         if file.split('.')[-1] == 'json':
#             filename = github_url+file.split('/')[-1]
#             files.append(filename)

#     ts_list = []
#     for item in files:
#         ts_list.append(pyleo.GeoSeries.from_json(item))


#     Euro2k = pyleo.MultipleGeoSeries(ts_list)
