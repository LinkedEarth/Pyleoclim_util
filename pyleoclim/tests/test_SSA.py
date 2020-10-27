''' Tests for pyleoclim.ssa

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
from sklearn import metrics
#pyleo.set_style('journal')

# 0. load the data
data = sio.loadmat('./example_data/wtc_test_data_nino.mat')
nino = data['nino'][:, 0]
t = data['datayear'][:, 0]
ts_nino = pyleo.Series(time=t, value=nino)
ts_n    = ts_nino.standardize()
#fig, ax = ts_n.plot(title='NINO3 region SST Anomalies',mute=True)
#ax.set_ylabel(r'NINO3 ($\sigma$ units)')
#pyleo.showfig(fig)  #debug only

# 1. Test basic SSA
nino_ssa = ts_n.ssa(M = 60)
# check that first 15 modes capture > 95% of the variance.
d  = nino_ssa['eig_val']
var_pct = d**2/np.sum(d**2)*100
assert np.abs(var_pct[15:].sum()*100-4.825612144779388) < 1e-6

# DEBUGGING ONLY:
# fig = plt.figure(figsize=(8,4))
# plt.errorbar(rk,d[:r],yerr=de[:r],label='SSA eigenvalues w/ 95% CI')
# plt.title('Scree plot of SSA eigenvalues')
# plt.xlabel('Rank $i$'); plt.ylabel(r'$\lambda_i$')
# plt.legend(loc='upper right')
# pyleo.showfig(fig)

# 2. test Monte Carlo SSA
nino_mcssa = ts_n.ssa(M = 60, nMC=200)
# for now, just testing that it works

# 3. test  SSA with missing values
n = len(nino)
fm = 0.1  #fraction of missing values
missing = np.random.choice(n,np.floor(fm*n).astype('int'),replace=False)
nino_miss = np.copy(ts_n.value)
nino_miss[missing] = np.nan  # put NaNs at the randomly chosen locations
ts_miss = pyleo.Series(time=t,value=nino_miss,clean_ts=False)
miss_ssa = ts_miss.ssa(M = 60)
RCmiss = miss_ssa['RC'][:,:19].sum(axis=1)

# check that it approximates ts_n quite well in a least squares sense.
MSE = metrics.mean_squared_error(nino, RCmiss)
assert MSE < 0.3

## DEBUG only
#fig, ax = ts_nino.plot(title=r'SSA reconstruction with '+ str(fm*100) +'% missing values',mute=True,label='monthly NINO3')
#ax.plot(t,RCmiss,label='SSA recon, $k=20$',color='orange')
#ax.legend()
#pyleo.showfig(fig)
