#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:49:05 2022

@author: julieneg
"""

import pyleoclim as pyleo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino.csv')
time = data['t'].values
air = data['air'].values
nino = data['nino'].values
ts_air = pyleo.Series(time=time, value=air, time_name='Year (CE)',label='AIR (mm/month)')
ts_nino = pyleo.Series(time=time, value=nino, time_name='Year (CE)',label='NINO3 (K)')

# without any arguments, the `tau` will be determined automatically
coh = ts_air.wavelet_coherence(ts_nino)
fig, ax = coh.plot()
#pyleo.closefig()


# or it can be passed
ntau=50
tau = np.linspace(np.min(ts_nino.time), np.max(ts_nino.time), ntau)
coh = ts_air.wavelet_coherence(ts_nino,settings={'tau':tau})
fig, ax = coh.plot()
#settings = {'ntau':50,'tau':tau}


import pyleoclim as pyleo
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
time = data.iloc[:,1]
value = data.iloc[:,2]
ts = pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')

scal1 = ts.wavelet(method='cwt') 
scal_signif = scal1.signif_test(number=200)  # for research-grade work, use number=200 or even larger
scal_signif.plot(title='CWT scalogram')
#@savefig scal_cwt.png


# if you wanted to invoke the WWZ instead 
scal2 = ts.wavelet(method='wwz')  
scal2.plot(title='WWZ scalogram')

# notice that the two scalograms have different units, which are arbitrary

