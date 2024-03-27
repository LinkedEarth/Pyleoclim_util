#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurrogateSeries is a child of MultipleSeries, designed for Monte Carlo tests 
"""

from ..core.multipleseries import MultipleSeries

supported_surrogates = frozenset(['ar1sim','phaseran']) # broadcast all supported surrogates as global variable, for exception handling

class SurrogateSeries(MultipleSeries):
    ''' Object containing surrogate timeseries, usually obtained through recursive modeling (e.g., AR(1))

    Surrogate Series is a child of MultipleSeries. All methods available for MultipleSeries are available for surrogate series.
    EnsembleSeries would be a more logical choice, but it creates circular imports that break the package. 
    '''
    def __init__(self, series_list, label, surrogate_method=None, surrogate_args=None): 
        self.series_list = series_list
        self.surrogate_method = surrogate_method
        self.surrogate_args = surrogate_args
        
        # refine the display name
        if surrogate_method == 'ar1sim':
            self.label = str(label or "series") + " surrogates [AR(1)]"
        elif surrogate_method == 'phaseran':
            self.label = str(label or "series") + " surrogates [phase-randomized]"
        else:
            raise ValueError('Surrogate method should either be "ar1sim" or "phaseran"')
        
         
       
        
