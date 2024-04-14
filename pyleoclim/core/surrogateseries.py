#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurrogateSeries is a child of EnsembleSeries, designed for parametric and non-parametric Monte Carlo tests 
"""

from ..core.ensembleseries import EnsembleSeries
from ..core.series import Series
from ..utils import tsutils, tsmodel

import numpy as np
import warnings

supported_surrogates = frozenset(['ar1sim','phaseran', 'uar1']) # broadcast all supported surrogates as global variable, for exception handling

class SurrogateSeries(EnsembleSeries):
    ''' Object containing surrogate timeseries, obtained by emulating an existing series, or more a statistical model

    Surrogate Series is a child of EnsembleSeries. All methods available for EnsembleSeries are available for surrogate series.
    
    '''
    def __init__(self, series_list=[], number=1, method=None, label=None): 
        '''
        Initiatlize a SurrogateSeries object. The parameters below are then used for series generation and labeling. 

        Parameters
        ----------
        series_list : list        
            a list of pyleoclim.Series objects. The default is [].
        
        number : int
            The number of surrogates to generate. THe default is 1. 
            
        method : str {ar1sim, phaseran, uar1}
            The name of the method used to generate surrogates of the timeseries
            
        label : str
            label of the collection of timeseries (e.g. 'SOI surrogates [AR(1) MLE]')
            If not provided, defaults to "series surrogates [{method}]"

        Raises
        ------
        ValueError
            errors out if an unknown method is specified

        '''
        self.series_list = series_list
        self.method = method
        self.number = number
        
        # refine the display name
        if method == 'ar1sim':
            self.label = str(label or "series") + " surrogates [AR(1) MoM]"
        elif method == 'phaseran':
            self.label = str(label or "series") + " surrogates [phase-randomized]"
        elif method == 'uar1':
            self.label = str(label or "series") + " surrogates [AR(1) MLE]"
        else:
            raise ValueError(f"Unknown method: {self.method}. Please use one of {supported_surrogates}")
            
    def from_series(self, target_series, seed=None):
        '''
        Fashion the SurrogateSeries object after a target series

        Parameters
        ----------
        target_series : Series object
            target Series used to infer surrogate properties
            
        seed : int
            Control random seed option for reproducibility

        Returns
        -------
        surr : SurrogateSeries

        See also
        --------

        pyleoclim.utils.tsmodel.ar1_sim : AR(1) simulator
        pyleoclim.utils.tsmodel.uar1_sim : maximum likelihood AR(1) simulator
        pyleoclim.utils.tsutils.phaseran2 : phase randomization
        
        Examples
        --------
        
        SOI = pyleo.utils.load_dataset('SOI')
        SOI_surr = pyleo.SurrogateSeries(method='phaseran', number=10) 
        SOI_surr.from_series(SOI)

        '''    
        #settings = {} if settings is None else settings.copy()
        
        if not isinstance(target_series, Series):
            raise TypeError(f"Expected pyleo.Series, got: {type(target_series)}")
            
        if seed is not None:
            np.random.seed(seed)
        
        # generate time axes according to provided pattern
        times = np.tile(target_series.time, (self.number, 1)).T  
        
        # apply surrogate method
        if self.method == 'ar1sim':
                y_surr = tsmodel.ar1_sim(target_series.value, self.number, target_series.time)  

        elif self.method == 'phaseran':
            if target_series.is_evenly_spaced():
                y_surr = tsutils.phaseran2(target_series.value, self.number)
            else:
                raise ValueError("Phase-randomization presently requires evenly-spaced series.")

        elif self.method == 'uar1':
            # estimate theta with MLE
            theta_hat = tsmodel.uar1_fit(target_series.value, target_series.time)
            # generate surrogates
            y_surr = np.empty_like(times)
            for j in range(self.number):
                y_surr[:,j] = tsmodel.uar1_sim(t = times[:,j],
                                               tau_0=theta_hat[0],sigma_2_0=theta_hat[1])         
            
        s_list = []
        for i, (t, y) in enumerate(zip(times.T,y_surr.T)):
            ts = Series(time=t, value=y,  
                           time_name=target_series.time_name,
                           time_unit=target_series.time_unit,
                           value_name=target_series.value_name,
                           value_unit=target_series.value_unit,
                           label = str(target_series.label or '') + " surr #" + str(i+1),
                           verbose=False, auto_time_params=False)
            s_list.append(ts)

        self.series_list = s_list
                  
    def from_params(self, params, length=50, time_pattern = 'even', seed=None, settings=None):
        '''
        Simulate the SurrogateSeries from a given probability model

        Parameters
        ----------
        params : list
            model parameters (e.g. [tau, sigma0] for an AR(1) model)
        
        length : int
            Length of the series. Default: 50
            
        time_pattern : str {match, even, random}
            The pattern used to generate the surrogate time axes
            'even' uses an evenly-spaced time with spacing `delta_t` specified in settings (if not specified, defaults to 1)
            'random' uses random_time_index() with specified distribution and parameters (default: 'exponential' with parameter 1)
            'custom' 
        seed : int
            Control random seed option for reproducibility

        settings : dict
            Parameters for surrogate generator. See individual methods for details.    

        Returns
        -------
        surr : SurrogateSeries

        See also
        --------

        pyleoclim.utils.tsmodel.ar1_sim : AR(1) simulator
        pyleoclim.utils.tsmodel.uar1_sim : maximum likelihood AR(1) simulator
        
        Examples
        --------
        ar1 = pyleo.SurrogateSeries(method='ar1sim', number=10) 
        ar1.from_process(length=100)

        '''    
        params = list(params) # coerce params into a list, no matter the original format
        settings = {} if settings is None else settings.copy()
        
        if seed is not None:
            np.random.seed(seed)
        
        # generate time axes according to provided pattern
        if time_pattern == "even":
            time_increment = settings["time_increment"] if "time_increment" in settings else 1
            t = np.cumsum([time_increment]*length)
            times = np.tile(t, (self.number, 1)).T     
        elif time_pattern == "random":
            times = np.zeros((length, self.number))
            for i in range(self.number):
                dist_name = settings['delta_t_dist'] if "delta_t_dist" in settings else "exponential"
                dist_param = settings['param'] if "param" in settings else [1]
                times[:, i] = tsmodel.random_time_index(length, dist_name,dist_param) 
        elif time_pattern == 'specified':
            if "time" not in settings:
                raise ValueError("'time' not found in settings")
            else:
                times =  np.tile(settings["time"], (self.number, 1)).T  
        else:
            raise ValueError(f"Unknown time pattern: {time_pattern}")
       
        # apply surrogate method
        if self.method == 'ar1sim' or 'uar1':
            y_surr = np.empty_like(times)
            if len(params)<2:
                raise ValueError('The AR(1) model needs 2 paramaters, tau and sigma2')
            # generate surrogates
            for j in range(self.number):
                y_surr[:,j] = tsmodel.uar1_sim(t = times[:,j],
                                               tau=params[0],sigma_2=params[1])       
        elif self.method == 'phaseran':
            raise ValueError("Phase-randomization is only available in from_series().")

        # create the series_list    
        s_list = []
        for i, (t, y) in enumerate(zip(times.T,y_surr.T)):
            ts = Series(time=t, value=y,  
                           label = str(self.label or '') + " surr #" + str(i+1),
                           verbose=False, auto_time_params=False)
            s_list.append(ts)

        self.series_list = s_list
                  
        
