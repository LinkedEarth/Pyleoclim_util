#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:57:48 2021

@author: julieneg
"""

import pyleoclim as pyleo
import numpy as np
    

def gen_colored_noise(alpha=1, nt=100, f0=None, m=None, seed=None):
    ''' Generate colored noise
    '''
    t = np.arange(nt)
    v = pyleo.utils.tsmodel.colored_noise(alpha=alpha, t=t, f0=f0, m=m, seed=seed)
    return t, v

t, v = gen_colored_noise(nt=200, alpha=1.0)

tc = t[1::2]

vc = pyleo.utils.tsutils.gkernel(t,v,tc, h = 11)

ts = pyleo.Series(time=t,value = v,label='original series')
tsc = pyleo.Series(time=tc,value = vc,label=r'coarsened series, $h=11$')

fig, ax = ts.plot()
tsc.plot(ax=ax)
#pyleo.showfig(fig)

# MultiplSeries