#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyleoclim import utils
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import numpy as np

# Create a signal
time = np.arange(2001)
f = 1/50  # the period is then 1/f = 50
signal = np.cos(2*np.pi*f*time)

# Wavelet Analysis
res = utils.wwz(signal, time)

# Visualization
fig, ax = plt.subplots()
contourf_args = {'cmap': 'magma', 'origin': 'lower', 'levels': 11}
cbar_args = {'drawedges': False, 'orientation': 'vertical', 'fraction': 0.15, 'pad': 0.05}
cont = ax.contourf(res.time, 1/res.freq, res.amplitude.T, **contourf_args)
ax.plot(res.time, res.coi, 'k--')  # plot the cone of influence
ax.set_yscale('log')
ax.set_yticks([2, 5, 10, 20, 50, 100, 200, 500, 1000])
ax.set_ylim([2, 1000])
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.set_xlabel('Time (yr)')
ax.set_ylabel('Period (yrs)')
cb = plt.colorbar(cont, **cbar_args)


# In[2]:


import numpy as np
import pyleoclim as pyleo

x = np.array([1,2,3,5,8,12,20])
y = np.ones(len(x))
xc,yc = pyleo.utils.tsutils.gkernel(x,y,bin_edges=[1,4,8,12,16,20])
xc


# In[3]:


x = np.array([1,2,3,5,8,12,20])
y = np.ones(len(x))
xc,yc = pyleo.utils.tsutils.gkernel(x,y,time_axis=[1,4,8,12,16,20])
xc


# In[4]:


x = np.array([1,2,3,5,8,12,20])
y = np.ones(len(x))
xc,yc = pyleo.utils.tsutils.gkernel(x,y,step=2)
xc


# In[5]:


x = np.array([1,2,3,5,8,12,20])
y = np.ones(len(x))
xc,yc = pyleo.utils.tsutils.gkernel(x,y,step_style='max')
xc


# In[6]:


x = np.array([1,2,3,5,8,12,20])
y = np.ones(len(x))
xc,yc = pyleo.utils.tsutils.gkernel(x,y)
xc


# In[7]:


import numpy as np
import pyleoclim as pyleo

x = np.array([1,2,3,5,8,12,20])
y = np.ones(len(x))
xi,yi = pyleo.utils.tsutils.interp(x,y,time_axis=[1,4,8,12,16])
xi


# In[8]:


x = np.array([1,2,3,5,8,12,20])
y = np.ones(len(x))
xi,yi = pyleo.utils.tsutils.interp(x,y,step=2)
xi


# In[9]:


x = np.array([1,2,3,5,8,12,20])
y = np.ones(len(x))
xi,yi = pyleo.utils.tsutils.interp(x,y,step_style='max')
xi


# In[10]:


x = np.array([1,2,3,5,8,12,20])
y = np.ones(len(x))
xi,yi = pyleo.utils.tsutils.interp(x,y)
xi


# In[11]:


from pyleoclim.utils.tsbase import convert_datetime_index_to_time
import pandas as pd
import numpy as np

time_unit = 'ga'
time_name = None
dti = pd.date_range("2018-01-01", periods=5, freq="Y", unit='s')
df = pd.DataFrame(np.array(range(5)), index=dti)
time = convert_datetime_index_to_time(
            df.index,
            time_unit,
            time_name=time_name,
            )
print(np.array(time))


# In[12]:


from pyleoclim.utils.tsbase import time_unit_to_datum_exp_dir

(datum, exponent, direction) = time_unit_to_datum_exp_dir(time_unit)
(datum, exponent, direction)

