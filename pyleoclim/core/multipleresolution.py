#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultipleResolution objects are designed to contain, display, and analyze information on the resolution of multiple time axes from a MultipleSeries, EnsembleSeries, or SurrogateSeries object.
"""

from ..utils import tsutils, plotting, tsmodel, tsbase

import warnings

import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from matplotlib import gridspec

class MultipleResolution:
    ''' Resolution object
    
    Resolution objects store time axis resolution information derived from MultipleSeries objects.
    They are generated via the resolution method applied to a MultipleSeries object and contain methods relevant to the analysis of resolution information.

    See Also
    --------

    '''

    def __init__(self,resolution_list,time_unit):
        self.resolution_list = resolution_list
        self.time_unit = time_unit

    def plot(self,ax=None,figsize=(10,8)):
        """Boxplot showing distribution of resolutions from each resolution object"""

        if ax is None:
            fig,ax = plt.subplots(figsize=figsize)

        df = pd.DataFrame(columns=['Resolution','Label'])
        