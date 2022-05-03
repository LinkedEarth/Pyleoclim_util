#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:49:05 2022

@author: julieneg
"""

import pyleoclim as pyleo
import pandas as pd
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
import numpy as np
ntau=50
tau = np.linspace(np.min(ts_nino.time), np.max(ts_nino.time), ntau)
coh = ts_air.wavelet_coherence(ts_nino,settings={'tau':tau})
fig, ax = coh.plot()
#settings = {'ntau':50,'tau':tau}


import pyleoclim as pyleo

ts = pyleo.gen_ts(model='colored_noise',nt=100)
scal = ts.wavelet(method='cwt') 