#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:28:31 2020

@author: deborahkhider

Example for welch
"""

from pyleoclim import utils
import matplotlib.pyplot as plt
import numpy as np

def welch():
#create a periodic signal
    time = np.arange(2001)
    f = 1/50
    signal = np.cos(2*np.pi*f*time)
    #analysis
    res=utils.welch(signal,time)
    #plot
    fig = plt.loglog(res['freq'],res['psd'])
    plt.xlabel('Frequency')
    plt.ylabel('PSD')
    plt.show()
