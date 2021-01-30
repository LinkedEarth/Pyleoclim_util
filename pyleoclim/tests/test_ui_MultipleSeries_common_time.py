#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Tests for pyleoclim.common_time

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
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
test_dirpath = pathlib.Path(__file__).parent.absolute()
#from sklearn import metrics
pyleo.set_style('journal')

# 0. load the data
data = sio.loadmat(os.path.join(test_dirpath, '../../example_data/wtc_test_data_nino.mat'))
nino = data['nino'][:, 0]
air  = data['air'][:, 0]
t = data['datayear'][:, 0]


# generate perturbed age models using a variant of bam_simul from https://github.com/sylvia-dee/PRYSM/blob/master/psm/agemodels/banded.py

n = len(t)
p = 1  #just one series here
ns = 10 # generate 10 simulations
param = np.array([0.1,0.1]) # probability of missing/double-counted layers
dt = 1.0/12
delta = np.ones((n,p,ns))*dt  # modified time matrix
ts = pyleo.Series(time=t,value=nino)

for nn in range(ns):
    num_event_mis = np.random.poisson(param[0]*n,size=(p,1)) # poisson model for missing bands
    num_event_dbl = np.random.poisson(param[1]*n,size=(p,1)) # poisson model for layers counted multiple times

    for ii in range(p):
        jumps = np.random.choice(n-1,num_event_mis[ii][0])+1            # place events uniformly on {2,...,n}
        delta[jumps,ii,nn] = delta[jumps,ii,nn]-dt                 # remove 1 at jump locations
        jumps = np.random.choice(n-1,num_event_dbl[ii][0])+1
        delta[jumps,ii,nn] = delta[jumps,ii,nn]+dt                 # add 1 at jump locations

yearCE = min(t) + np.cumsum(delta,axis=0)
yearCE = yearCE.reshape((n,ns))
# create case of year BP
yearBP = 1950-yearCE

# creat BP timeseries object

tsBP = ts.copy()
tsBP.time = 1950-t

#create multiple series object and append all these perturbed time axes
mtsCE = pyleo.MultipleSeries([ts])
mtsBP = pyleo.MultipleSeries([tsBP])

for nn in range(ns):
    ts_a = pyleo.Series(time=yearCE[:,nn],value=nino,label='ensemble member '+str(nn))
    mtsCE = mtsCE.append(ts_a)
    ts_a = pyleo.Series(time=yearBP[:,nn],value=nino,label='ensemble member '+str(nn))
    mtsBP = mtsBP.append(ts_a)
    
# first work with yearCE: extract grid properties and define common axis
#mts = mtsCE.copy()
#gp = np.empty((len(mts.series_list),3))
#for idx,item in enumerate(mts.series_list):
#    gp[idx,:]  = pyleo.tsutils.grid_properties(item.time)  

#delta_t  = gp[:,2] 



# test the binning method
ms_bin = mtsCE.common_time(method = 'binning')   # VERY SLOW
#ms_bin.plot()
#plt.show()

# test the interpolation method
ms_interp = mtsCE.common_time(method = 'interp')   # ACCEPTABLY FAST
#ms_interp.plot()
#plt.show()

ms_gkernel = mtsCE.common_time(method = 'gkernel')  # VERY FAST
#ms_gkernel.plot()
#plt.show()

# combine CE and BP
mts_joint = pyleo.MultipleSeries([*mtsCE.series_list, *mtsBP.series_list])
#mts_joint.plot(color='blue',plot_kwargs={'alpha': 0.1})
#plt.show()

# test the joint interpolation method
ms_interp2 = mts_joint.common_time(method = 'interp')   
ms_interp2.plot(color='gray',plot_kwargs={'alpha': 0.1})
# plt.show()   # FAIL


#class Ensemble(MultipleSeries)
#lipdutils.CheckTimeAxis?

# timeseries select 


#  something of the form ts.select(tmin=,tmax=)

#ind0 = np.where(t>=tmin[0]) and np.where(t<=tmax[0])
#ind1 = np.where(t>=tmin[1]) and np.where(t<=tmax[1])

#create series object
#ts0=pyleo.Series(time=t[ind0],value=nino[ind0])
#ts1=pyleo.Series(time=t[ind1],value=nino[ind1])
# Create a multiple series object
#ts_all= pyleo.MultipleSeries([ts0,ts1])



