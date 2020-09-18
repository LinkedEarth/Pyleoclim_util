from pyleoclim import utils
import matplotlib.pyplot as plt
import numpy as np
# Create a signal
time = np.arange(2001)
f = 1/50
signal = np.cos(2*np.pi*f*time)
# Spectral Analysis
res = utils.lomb_scargle(signal, time)
# plot
fig = plt.loglog(
          res['freq'],
          res['psd'])
plt.xlabel('Frequency')
plt.ylabel('PSD')
plt.show()
