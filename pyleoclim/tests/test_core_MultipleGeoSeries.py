#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:47:08 2023

@author: julieneg
"""
import pyleoclim as pyleo
# from bs4 import BeautifulSoup
# import requests
import pytest


class TestUIMultipleGeoSeriesMap:
    def test_map_archives(multiple_pinkgeoseries):
        '''
        test mapping semantics
        '''
        mgs = multiple_pinkgeoseries
        fig, ax = mgs.map(hue = 'archiveType')
        pyleo.closefig(fig)
        # assert something?

    def test_map_obs(multiple_pinkgeoseries):
        '''
        test mapping semantics
        '''
        mgs = multiple_pinkgeoseries
        fig, ax = mgs.map(hue = 'observationType')
        pyleo.closefig(fig)
        # assert something?

    def test_map_elevation(multiple_pinkgeoseries):
        '''
        test mapping semantics
        '''
        mgs = multiple_pinkgeoseries
        fig, ax = mgs.map(hue = 'elevation')
        pyleo.closefig(fig)
        # assert something?

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
