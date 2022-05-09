#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:49:05 2022

@author: julieneg
"""

import pyleoclim as pyleo
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


#data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino_even.csv')
data = pd.read_csv('/Users/julieneg/Documents/GitHub/Pyleoclim_util/example_data/wtc_test_data_nino_even.csv')
time = data['t'].values
air = data['air'].values
nino = data['nino'].values
ts_air = pyleo.Series(time=time, value=air, time_name='Year (CE)',label='AIR (mm/month)')
ts_nino = pyleo.Series(time=time, value=nino, time_name='Year (CE)',label='NINO3 (K)')

coh_cwt = ts_air.wavelet_coherence(ts_nino) # by default, the method applied is CWT 
coh_wwz = ts_air.wavelet_coherence(ts_nino, method = 'wwz') # it's also easy (but slow) to apply WWZ to an evenly-spaced series

fig, ax = coh_cwt.plot()
#pyleo.closefig()

cwt1 =  pyleo.utils.wavelet.cwt(nino, time)
cwt2 =  pyleo.utils.wavelet.cwt(air, time)

wt_coeff1 = cwt1.coeff.T # transpose so that scale is second axis, as for wwz
wt_coeff2 = cwt2.coeff.T 

scale = cwt1.scale
tau = time
smooth_factor = 0.25

# compute XWT and CWT
xw_coherence, xw_phase = pyleo.utils.wavelet.wtc(wt_coeff1, wt_coeff2, scale, tau, smooth_factor=smooth_factor)
xw_t, xw_amplitude, _ = pyleo.utils.wavelet.xwt(wt_coeff1, wt_coeff2)

#  COMPARE TO WWZ

res_wwz1 = pyleo.utils.wavelet.wwz(nino, time)
res_wwz2 = pyleo.utils.wavelet.wwz(air, time)

wt_coeff1 = res_wwz1.coeff[1] - res_wwz1.coeff[2]*1j
wt_coeff2 = res_wwz2.coeff[1] - res_wwz2.coeff[2]*1j

scale = 1/res_wwz1.freq  # `scales` here is the `Period` axis in the wavelet plot

xw_coherence, xw_phase = pyleo.utils.wavelet.wtc(wt_coeff1, wt_coeff2, scale, tau, 
                                                 smooth_factor=smooth_factor)


# or it can be passed
ntau=50
tau = np.linspace(np.min(ts_nino.time), np.max(ts_nino.time), ntau)
coh = ts_air.wavelet_coherence(ts_nino,settings={'tau':tau})
fig, ax = coh.plot()
#settings = {'ntau':50,'tau':tau}


# ========== Coherence summary plot ======

 # if ax is None:
 #     if xwt is False:
 #         fig, ax = plt.subplots(figsize=figsize)
 #         ax = np.ndarray([ax]) # ensures that ax is iterable
 #     else:
 #         fig, ax = plt.subplots(2,1,figsize=(figsize[0],1.8*figsize[1]),sharex=True)
