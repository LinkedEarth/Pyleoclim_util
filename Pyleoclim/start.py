# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:51:35 2016

@author: deborahkhider

"""
# Import the needed modules
from lipd.start import *
import pyleoclim.pkg_resources.MapLipd as MapLipd
import pyleoclim.pkg_resources.PlotLipd as PlotLipd


# load the LiPD files
loadLipds()

# MAKE A MAP

def MapAll(markersize = int(50), saveFig = True):
    """
    Map all the Lipd records by proxies
    """
    map1 = MapLipd.MakeMap()
    map1.map_all(markersize, saveFig)

def MapaLipd(name="",gridlines = False, borders = True, \
        topo = True, markersize = int(50), marker = "Default", \
        saveFig = True):
    """
    Map one particular LiPD record
    """
    map1 = MapLipd.MakeMap()
    map1.map_aLiPD(name,gridlines, borders, topo, markersize, marker, \
        saveFig)

# PLOT TIME SERIES

def PlotaLipd(filename = "", \
        plot_style = 'seaborn-ticks', x_min = [], x_max = [], \
        saveFig = True):
    """
    Plot one particular LiPD record
    """

    plot1 = PlotLipd.MakePlot()
    plot1.plot_aLiPD(filename, plot_style, x_min, x_max, saveFig)
