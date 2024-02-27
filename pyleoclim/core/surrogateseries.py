#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurrogateSeries is a child of MultipleSeries, designed for Monte Carlo tests 
"""

from ..core.multipleseries import MultipleSeries
#from ..core.ensembleseries import EnsembleSeries

class SurrogateSeries(MultipleSeries):
    ''' Object containing surrogate timeseries, usually obtained through recursive modeling (e.g., AR(1))

    Surrogate Series is a child of MultipleSeries. All methods available for MultipleSeries are available for surrogate series.
    EnsembleSeries would be a more logical choice
    '''
    def __init__(self, series_list, label, surrogate_method=None, surrogate_args=None): 
        self.series_list = series_list
        self.surrogate_method = surrogate_method
        self.surrogate_args = surrogate_args
        
        # refine the display name
        if surrogate_method == 'ar1sim':
            self.label = self.label + " surrogates [AR(1)]"
        elif surrogate_method == 'phaseran':
            self.label = self.label + " surrogates [phase-randomized]"
        else:
            raise ValueError('Surrogate method should either be "ar1sim" or "phaseran"')
        
         
       
        
