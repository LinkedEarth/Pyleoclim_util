#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:49:05 2022

@author: julieneg
"""

import pyleoclim as pyleo
import pandas as pd
import numpy as np


#data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino_even.csv')
data = pd.read_csv('/Users/julieneg/Documents/GitHub/Pyleoclim_util/example_data/wtc_test_data_nino_even.csv')
time = data['t'].values
air = data['air'].values
nino = data['nino'].values
ts_air = pyleo.Series(time=time, value=air, time_name='Year (CE)', label='AIR (mm/month)')
ts_nino = pyleo.Series(time=time, value=nino, time_name='Year (CE)',label='NINO3 (K)')

# by default, the method applied is CWT 
coh_cwt = ts_air.wavelet_coherence(ts_nino)
@savefig coh_cwt.png 
fig, ax = coh_cwt.plot() 
# the high frequencies are not very meaningful, so one can truncate
@savefig coh_trunc.png 
fig, ax = coh_cwt.plot(ylim=[1,50]) 

# To pass method-specific parameters, use the `settings` dictionary. For instance, 
# you may change the default mother wavelet ('MORLET') to a 
#  derivative of a Gaussian (DOG), with degree 2 by default ("Mexican Hat wavelet"):

coh_mex = ts_air.wavelet_coherence(ts_nino,settings = {'mother':'DOG'})
@savefig coh_mex.png 
coh_mex.plot(ylim=[1,50]) 

# for more information on such parameters, look up the underlying method (cwt (NEED LINK))

# Note that in this example both timeseries area already on a common, 
# evenly-spaced time axis. If they are not (either because the data are unevenly spaced, 
# or because the time axes are different in some other way), an error will be raised.
# To circumvent this error, you can either put the series 
# on a common time axis (e.g. using common_time()) prior to applying CWT, or you
# can use the Weighte Wavelet Z-transform (WWZ) instead, as it is designed for 
# unevenly-spaced data. However, it is usually far slower:

coh_wwz = ts_air.wavelet_coherence(ts_nino, method = 'wwz') 
@savefig coh_wwz.png
fig, ax = coh_wwz.plot() 

# note that, for computational efficiency, the time axis for WWZ is coarse-grained
# by default to 50 time points, which explains in part the diffence with the CWT coherency. 

# if you need a custom axis, it (and other method-specific  parameters) can also be 
# passed via the `settings` dictionary:
ntau = 40
tau = np.linspace(np.min(ts_nino.time), np.max(ts_nino.time), ntau)
coh_wwz2 = ts_air.wavelet_coherence(ts_nino, method = 'wwz', settings={'tau':tau})
fig, ax = coh_wwz2 .plot()


# Significance 
cwt_sig = coh_cwt.signif_test()
cwt_sig.plot(signif_thresh = 0.97)





 



# ========== Coherence summary plot ======

 # if ax is None:
 #     if xwt is False:
 #         fig, ax = plt.subplots(figsize=figsize)
 #         ax = np.ndarray([ax]) # ensures that ax is iterable
 #     else:
 #         fig, ax = plt.subplots(2,1,figsize=(figsize[0],1.8*figsize[1]),sharex=True)
