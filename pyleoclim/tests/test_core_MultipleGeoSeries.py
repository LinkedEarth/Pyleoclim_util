#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:47:08 2023

@author: julieneg
"""
import numpy as np

# from bs4 import BeautifulSoup
# import requests
import pytest

import pyleoclim as pyleo


class TestUIMultipleGeoSeriesMap:
    def test_map_archives(self, multiple_pinkgeoseries):
        """
        test mapping semantics
        """
        mgs = multiple_pinkgeoseries()
        fig, ax = mgs.map(marker="archiveType")
        pyleo.closefig(fig)
        # assert something?

    def test_map_obs(self, multiple_pinkgeoseries):
        """
        test mapping semantics
        """
        mgs = multiple_pinkgeoseries()
        fig, ax = mgs.map(hue="observationType")
        pyleo.closefig(fig)
        # assert something?

    def test_map_elevation(self, multiple_pinkgeoseries):
        """
        test mapping semantics
        """
        mgs = multiple_pinkgeoseries()
        fig, ax = mgs.map(hue="elevation")
        pyleo.closefig(fig)
        # assert something?

    @pytest.mark.parametrize("crit_dist", [5000, 30000])
    def test_map_proj_pick_global(self, crit_dist, multiple_pinkgeoseries):
        """
        test automated projection picker
        """
        mgs = multiple_pinkgeoseries()
        lat = np.array([ts.lat for ts in mgs.series_list])
        lon = np.array([ts.lon for ts in mgs.series_list])
        clat, clon = pyleo.utils.mapping.centroid_coords(lat, lon)
        # d =  pyleo.utils.mapping.compute_dist(clat, clon, lat, lon) # computes distances to centroid

        fig, ax_d = mgs.map(crit_dist=crit_dist)
        if crit_dist == 30000:
            assert ax_d["map"].projection.proj4_params["proj"] == "ortho"
            assert ax_d["map"].projection.proj4_params["lon_0"] == clon
            assert ax_d["map"].projection.proj4_params["lat_0"] == clat
        else:
            assert ax_d["map"].projection.proj4_params["proj"] == "robin"
            assert ax_d["map"].projection.proj4_params["lon_0"] == clon

    def test_map_proj_pick_regional(self, multiple_pinkgeoseries):
        """
        test automated projection picker
        """
        mgs = multiple_pinkgeoseries(geobox=[20, 70, -10, 40])
        lat = [ts.lat for ts in mgs.series_list]
        lon = [ts.lon for ts in mgs.series_list]
        clat, clon = pyleo.utils.mapping.centroid_coords(lat, lon)
        fig, ax_d = mgs.map()
        assert ax_d["map"].projection.proj4_params["proj"] == "ortho"
        assert ax_d["map"].projection.proj4_params["lon_0"] == clon
        assert ax_d["map"].projection.proj4_params["lat_0"] == clat


class TestUIMultipleGeoSeriesPCA:
    def test_pca_t0(self, multiple_pinkgeoseries):
        """
        test PCA output
        """
        mgs = multiple_pinkgeoseries()
        pca = mgs.pca()
        assert pca.name == "Multiple Pink GeoSeries PCA"

    def test_pca_t1(self, multiple_pinkgeoseries):
        """
        test PCA screeplot
        """
        mgs = multiple_pinkgeoseries()
        pca = mgs.pca()
        fig, ax = pca.screeplot()

    def test_pca_t2(self, multiple_pinkgeoseries):
        """
        test PCA modeplot
        """
        mgs = multiple_pinkgeoseries()
        pca = mgs.pca()
        fig, ax = pca.modeplot(index=1)


class TestUIMultipleGeoSeriesTimeGeoPlot:
    def test_time_geo_plot_t0(self, multiple_pinkgeoseries):
        """
        test PCA output
        """
        mgs = multiple_pinkgeoseries()
        fig, ax = mgs.time_geo_plot()
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
