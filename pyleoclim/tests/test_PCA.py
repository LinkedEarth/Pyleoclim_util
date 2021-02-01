#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scratchpad to test the PCA method

Created on Fri Jan 29 19:37:27 2021

@author: julieneg
"""
import pyleoclim as pyleo
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.multivariate.pca import PCA


# Test 1: using MD76
url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
data = pyleo.Lipd(usr_path = url)
tslist = data.to_tso()
mslist = []
for item in tslist:
    mslist.append(pyleo.Series(time = item['age'], value = item['paleoData_values']))
ms = pyleo.MultipleSeries(mslist[2:5],name='MD76')

msc = ms.common_time(method='interp')


res = msc.pca(nMC=20, missing='fill-em')
# fails because too many NaNs are left through, whether with 'binning', 'interp' or 'gkernel'


# Test 2: Use synthetic data
p = 20
x = np.random.randn(100)[:, None]
x = x + np.random.randn(100, p)
pc = PCA(x)

t = np.arange(100)

mslist = []
for i in range(p):
    mslist.append(pyleo.Series(time = t, value = x[:,i]))
ms = pyleo.MultipleSeries(mslist,name='signal')
        
res = ms.pca(nMC=1000)

# 
v = res['eigval']
n = res["pcs"].shape[0] # sample size
dv =  v*np.sqrt(2/(n-1))
idx = np.arange(len(v))
plt.plot(idx,res['eigval95'],color='silver',ls='--',label='AR(1) 95% benchmark')
plt.errorbar(x=idx,y=v,yerr = dv, color='crimson',marker='o',ls='',alpha=0.8,label=ms.name)
plt.title('Scree plot',fontweight='bold')
plt.legend()
plt.xlabel(r'Mode index $i$'); plt.ylabel(r'$\lambda_i$')
plt.show()
#res = decomposition.mcpca(ys, nMC)