#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:47:08 2023

@author: julieneg
"""


def create_MultipleGeoSeries():
    '''
    Tests the creation of Euro2k MultipleGeoSeries
    '''
    from bs4 import BeautifulSoup
    import requests
    import pyleoclim as pyleo

    url = 'https://github.com/LinkedEarth/Pyleoclim_util/tree/master/example_data/Euro2k_json'
    ext = 'json'

    def listFD(url, ext=''):
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

    files = []
    github_url = 'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/Euro2k_json/'
    for file in listFD(url, ext):
        if file.split('.')[-1] == 'json':
            filename = github_url+file.split('/')[-1]
            files.append(filename)

    ts_list = []
    for item in files:
        ts_list.append(pyleo.GeoSeries.from_json(item))
        
        
    Euro2k = pyleo.MultipleGeoSeries(ts_list)
    # check lat/lon
    lon = [ts.lon for ts in Euro2k.series_list]  # fail
    