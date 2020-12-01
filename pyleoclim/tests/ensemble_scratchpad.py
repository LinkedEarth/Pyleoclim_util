#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Tests for pyleoclim.align

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import pyleoclim as pyleo
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
#from sklearn import metrics
pyleo.set_style('journal')

# 0. load the data
data = sio.loadmat('./example_data/wtc_test_data_nino.mat')
nino = data['nino'][:, 0]
t = data['datayear'][:, 0]


# generate perturbed age models using a variant of bam_simul from https://github.com/sylvia-dee/PRYSM/blob/master/psm/agemodels/banded.py

n = len(t)
p = 1  #just one series here
ns = 10 # generate 10 simulations
param = np.array([0.01,0.01]) # probability of missing/double-counted layers
delta = np.ones((n,p,ns))*dt  # modified time matrix
dt = 1.0/12

for nn in range(ns):
    num_event_mis = np.random.poisson(param[0]*n,size=(p,1)) # poisson model for missing bands
    num_event_dbl = np.random.poisson(param[1]*n,size=(p,1)) # poisson model for layers counted multiple times

    for ii in range(p):
        jumps = np.random.choice(n-1,num_event_mis[ii][0])+1            # place events uniformly on {2,...,n}
        delta[jumps,ii,nn] = delta[jumps,ii,nn]-dt                 # remove 1 at jump locations
        jumps = np.random.choice(n-1,num_event_dbl[ii][0])+1
        delta[jumps,ii,nn] = delta[jumps,ii,nn]+dt                 # add 1 at jump locations

time = min(t) + np.cumsum(delta,axis=0)



# create case of year BP
trev = 1950-tm 


#class Ensemble(MultipleSeries)
#lipdutils.CheckTimeAxis?

# randomnly decimate the time axis
#tmin = min(t) + np.random.randint(1,10,2)
#tmax = max(t) - np.random.randint(1,10,2)

# timeseries select 
ts = pyleo.Series(time=t,value=nino)

ts.select(tmin=,tmax=)

#ind0 = np.where(t>=tmin[0]) and np.where(t<=tmax[0])
#ind1 = np.where(t>=tmin[1]) and np.where(t<=tmax[1])

#create series object
ts0=pyleo.Series(time=t[ind0],value=nino[ind0])
ts1=pyleo.Series(time=t[ind1],value=nino[ind1])

# Create a multiple series object
ts_all= pyleo.MultipleSeries([ts0,ts1])



# mockup of "commonAxis" function


# write in ts_utils very broad  on numpy array or pandas df
#  max(min), min(max), median spacing.

# expose in ui.py
#  create method in MultipleSeries that returns common time
# use within binTs, interpTs, Corr_sig

# write test at MultipleSeries level

