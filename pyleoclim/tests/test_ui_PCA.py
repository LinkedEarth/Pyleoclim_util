#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:37:27 2021

@author: julieneg
"""
import pyleoclim as pyleo
import numpy as np
from statsmodels.multivariate.pca import PCA


# Test 1: using MD76
url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
data = pyleo.Lipd(usr_path = url)
tslist = data.to_tso()
mslist = []
for item in tslist:
    mslist.append(pyleo.Series(time = item['age'], value = item['paleoData_values']))
ms = pyleo.MultipleSeries(mslist)

msc = ms.common_time()

res = msc.pca(nMC=20)
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
ms = pyleo.MultipleSeries(mslist)
        
res = ms.pca(nMC=20)

#res = decomposition.mcpca(ys, nMC)