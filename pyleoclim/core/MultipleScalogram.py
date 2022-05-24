#from ..utils import tsutils, plotting, mapping, lipdutils, tsmodel, tsbase
#from ..utils import wavelet as waveutils
#from ..utils import spectral as specutils
#from ..utils import correlation as corrutils
#from ..utils import causality as causalutils
#from ..utils import decomposition
#from ..utils import filter as filterutils

#from textwrap import dedent

# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# from tabulate import tabulate
# from collections import namedtuple
from copy import deepcopy

# from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MaxNLocator
# import matplotlib.transforms as transforms
# from matplotlib import cm
# from matplotlib import gridspec
# import matplotlib as mpl
#from matplotlib.colors import BoundaryNorm, Normalize

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# from tqdm import tqdm
from scipy.stats.mstats import mquantiles
# from scipy import stats
# from statsmodels.multivariate.pca import PCA
# import warnings
# import os

# import lipd as lpd

# import collections

from ..core import Scalogram

class MultipleScalogram:
    ''' Multiple Scalogram objects
    '''
    def __init__(self, scalogram_list):
        self.scalogram_list = scalogram_list

    def copy(self):
        ''' Copy the object
        '''
        return deepcopy(self)

    def quantiles(self, qs=[0.05, 0.5, 0.95]):
        '''Calculate quantiles

        Parameters
        ----------
        qs : list, optional
            List of quantiles to consider for the calculation. The default is [0.05, 0.5, 0.95].

        Raises
        ------
        ValueError
            Frequency axis not consistent across the PSD list!

        Value Error
            Time axis not consistent across the scalogram list!

        Returns
        -------
        scals : pyleoclim.MultipleScalogram
        '''
        freq = np.copy(self.scalogram_list[0].frequency)
        scale = np.copy(self.scalogram_list[0].scale)
        time = np.copy(self.scalogram_list[0].time)
        coi = np.copy(self.scalogram_list[0].coi)
        amps = []
        for scal in self.scalogram_list:
            if not np.array_equal(scal.frequency, freq):
                raise ValueError('Frequency axis not consistent across the scalogram list!')

            if not np.array_equal(scal.time, time):
                raise ValueError('Time axis not consistent across the scalogram list!')

            amps.append(scal.amplitude)

        amps = np.array(amps)
        ne, nf, nt = np.shape(amps)
        amp_qs = np.ndarray(shape=(np.size(qs), nf, nt))

        for i in range(nf):
            for j in range(nt):
                amp_qs[:,i,j] = mquantiles(amps[:,i,j], qs)

        scal_list = []
        for i, amp in enumerate(amp_qs):
            scal_tmp = Scalogram(frequency=freq, time=time, amplitude=amp,
                                 scale = scale, coi=coi, label=f'{qs[i]*100:g}%')
            scal_list.append(scal_tmp)

        scals = MultipleScalogram(scalogram_list=scal_list)
        return scals