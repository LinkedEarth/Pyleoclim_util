#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import pandas as pd
import numpy as np


# In[2]:


import pyleoclim as pyleo
pyleo.set_style('journal')


# In[3]:


from pyleoclim import tests


# ## Series

# In[4]:


soi_data = tests.load_dataset('soi', skiprows=1)
soi_data['Date'] = pd.to_datetime(soi_data['Date'], format='%Y%m')
nt = len(soi_data)
to = np.linspace(1951, 2020-1/12, nt)
Xo = soi_data['Value'].values


# In[5]:


ts = pyleo.Series(time=to, value=Xo, time_name='Year', time_unit='AD', value_name='SOI', value_unit='K')
# print(ts)

fig, ax = ts.plot()


# In[6]:


fig, ax = ts.plot(savefig_settings={'path': './figs/soi.pdf'})


# ##  Spectral analysis on the Series

# In[6]:


get_ipython().run_cell_magic('time', '', "psd_wwz = ts.spectral(settings={'nMC': 0})")


# In[7]:


fig, ax = psd_wwz.plot(label='WWZ')


# In[8]:


psd_mtm = ts.spectral(method='mtm')
ax = psd_mtm.plot(ax=ax, label='MTM')

pyleo.showfig(fig)


# ## Wavelet analysis on the Series

# In[10]:


get_ipython().run_cell_magic('time', '', "scal = ts.wavelet(settings={'nMC': 0})")


# In[11]:


fig, ax = scal.plot()


# ## Wavelet coherence analysis on two Series objs

# In[12]:


import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('../../example_data/wtc_test_data_nino.mat')
air = data['air'][:, 0]
nino = data['nino'][:, 0]
t = data['datayear'][:, 0]


# In[13]:


ts_air = pyleo.Series(time=t, value=air)
ts_nino = pyleo.Series(time=t, value=nino)

fig, ax = ts_air.plot(title='Deasonalized All Indian Rainfall Index')
fig, ax = ts_nino.plot(title='El Nino Region 3 -- SST Anomalies')


# In[17]:


coh = ts_air.wavelet_coherence(ts_nino, settings={'nMC': 0})


# In[18]:


fig, ax = coh.plot(phase_style={'skip_x': 50})


# ##  Correlation analysis

# In[20]:


corr_res = ts_air.correlation(ts_nino)
print(corr_res)


# ## Causality analysis

# In[22]:


causal_res = ts_air.causality(ts_nino)
print(causal_res)


# In[ ]:




