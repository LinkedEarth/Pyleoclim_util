#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurrogateSeries is a child of MultipleSeries, designed for Monte Carlo tests 
"""

# from ..utils import tsutils, plotting, mapping, lipdutils, tsmodel, tsbase
# from ..utils import wavelet as waveutils
# from ..utils import spectral as specutils
# from ..utils import correlation as corrutils
# from ..utils import causality as causalutils
# from ..utils import decomposition
# from ..utils import filter as filterutils

from ..core import MultipleSeries

#from textwrap import dedent

# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from tabulate import tabulate
# from collections import namedtuple
# from copy import deepcopy

# from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MaxNLocator
# import matplotlib.transforms as transforms
# from matplotlib import cm
# from matplotlib import gridspec
# import matplotlib as mpl
#from matplotlib.colors import BoundaryNorm, Normalize

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# from tqdm import tqdm
# from scipy.stats.mstats import mquantiles
# from scipy import stats
# from statsmodels.multivariate.pca import PCA
# import warnings
# import os
# import lipd as lpd
# import collections

class SurrogateSeries(MultipleSeries):
    ''' Object containing surrogate timeseries, usually obtained through recursive modeling (e.g., AR1)

    Surrogate Series is a child of MultipleSeries. All methods available for MultipleSeries are available for surrogate series.
    '''
    def __init__(self, series_list, surrogate_method=None, surrogate_args=None):
        self.series_list = series_list
        self.surrogate_method = surrogate_method
        self.surrogate_args = surrogate_args