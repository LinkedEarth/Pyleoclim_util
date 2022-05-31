#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurrogateSeries is a child of MultipleSeries, designed for Monte Carlo tests 
"""

from ..core.multipleseries import MultipleSeries

class SurrogateSeries(MultipleSeries):
    ''' Object containing surrogate timeseries, usually obtained through recursive modeling (e.g., AR1)

    Surrogate Series is a child of MultipleSeries. All methods available for MultipleSeries are available for surrogate series.
    '''
    def __init__(self, series_list, surrogate_method=None, surrogate_args=None): 
        self.series_list = series_list
        self.surrogate_method = surrogate_method
        self.surrogate_args = surrogate_args