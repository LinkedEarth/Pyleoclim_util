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

supported_surrogates = frozenset(['ar1sim','phaseran','uar1','CN']) # broadcast all supported surrogates as global variable, for exception handling

class SurrogateSeries(EnsembleSeries):
    ''' Object containing surrogate timeseries, obtained either by emulating an 
    existing series (see `from_series()`) or from a parametric model (see `from_param()`)

    Surrogate Series is a child of EnsembleSeries. All methods available for EnsembleSeries are available for surrogate series.
    
    '''
    def __init__(self, series_list=[], number=1, method=None, label=None, param=None, seed=None): 
        '''
        Initialize a SurrogateSeries object. The parameters below are then used for series generation and labeling. 

        Parameters
        ----------
        series_list : list        
            a list of pyleoclim.Series objects. The default is [].
        
        number : int
            The number of surrogates to generate. The default is 1. 
            
        method : str {ar1sim, phaseran, uar1, CN}
            The name of the method used to generate surrogates of the timeseries (no default)
            - 'ar1sim' fits an AR(1) model using the method of moments (MoM) as per [1]
            - 'phaseran' applies phase randomization as per [2, 3]
            - 'CN' fits a "colored noise" model (power-law spectrum) as per [4] (Eq 15)
            
        param : list 
            a list of parameters for the method in question. Ignored if the method does not require parameters. 
            
        label : str
            label of the collection of timeseries (e.g. 'SOI surrogates [AR(1) MoM]')
            If not provided, defaults to "series surrogates [{method}]"

        Raises
        ------
        ValueError
            errors out if an unknown method is specified

        References
        ----------

        _ [1] Mudelsee, M. (2002), TAUEST: a computer program for estimating persistence in unevenly spaced weather/climate time series, Computers and Geosciences, 28, 69–72, doi:10.1016/S0098- 3004(01)00041-3.
        _ [2] Ebisuzaki, W. (1997), A method to estimate the statistical significance of a correlation when the data are serially correlated, Journal of Climate, 10(9), 2147–2153, doi:10.1175/1520- 0442(1997)010¡2147:AMTETS¿2.0.CO;2.
        _ [3] Prichard, D., and J. Theiler (1994), Generating surrogate data for time series with several simultaneously measured variables, Phys. Rev. Lett., 73, 951–954, doi: 10.1103/PhysRevLett.73.951.
        _ [4] Kirchner, J. W., Aliasing in 1/f(alpha) noise spectra: origins, consequences, and remedies. Phys Rev E Stat Nonlin Soft Matter Phys 71, 066110 (2005).

        '''
        self.series_list = series_list
        self.method = method
        self.param = param
        self.number = number
        self.seed = seed
        
        if param is not None:
            nparam = len(param)
            if 'ar1' in method and nparam !=2:
                raise ValueError("2 parameters are needed for this model")
            elif method == 'phaseran':
                warnings.warn("Phase-randomization is a non parametric method; the provided parameters will be ignored")
            elif method == 'CN' and nparam >1:
                raise ValueError(f"1 parameter is needed for this model; only the first of the provided {nparam} will be used")
       
        # refine the display name
        if label is None:
            if method == 'ar1sim':
                self.label =  "AR(1) surrogates (MoM)"
            elif method == 'phaseran':
                self.label = "Phase-randomized surrogates"
            elif method == 'uar1':
                self.label = "AR(1) surrogates (MLE)"
            elif method == 'CN':
                self.label = r'Power-law surrogates ($S(f) \propto f^{-\beta}$)'
            else:
                raise ValueError(f"Unknown method: {self.method}. Please use one of {supported_surrogates}")
                
    def from_series(self, target_series):
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
        SOI_surr = pyleo.SurrogateSeries(method='phaseran', number=4) 
        SOI_surr.from_series(SOI)
        fig, ax = SOI_surr.plot_traces()
        SOI.plot(ax=ax,color='black',linewidth=1, ylim=[-6,3])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2)
        ax.set_title(SOI_surr.label)

        '''    
        #settings = {} if settings is None else settings.copy()
        
        if not isinstance(target_series, Series):
            raise TypeError(f"Expected pyleo.Series, got: {type(target_series)}")
            
        if self.seed is not None:
            np.random.default_rng(self.seed)
        
        # apply surrogate method
        if self.method == 'ar1sim':
            tsi = target_series if target_series.is_evenly_spaced() else target_series.interp()
            mu = tsi.value.mean()
            y_surr = tsmodel.ar1_sim(target_series.value, self.number, target_series.time) + mu  

        elif self.method == 'phaseran':
            if target_series.is_evenly_spaced():
                y_surr = tsutils.phaseran2(target_series.value, self.number)
            else:
                raise ValueError("Phase-randomization presently requires evenly-spaced series.")

        elif self.method == 'uar1':
            # estimate theta with MLE
            tau, sigma_2 = tsmodel.uar1_fit(target_series.value, target_series.time)
            tsi = target_series if target_series.is_evenly_spaced() else target_series.interp()
            mu = tsi.value.mean()
            self.param = [tau, sigma_2] # assign parameters for future use
            # generate time axes according to provided pattern
            times = np.squeeze(np.tile(target_series.time, (self.number, 1)).T)  
            # generate matrix and add  the mean
            y_surr = tsmodel.uar1_sim(t = times, tau=tau, sigma_2=sigma_2) + mu
            
        elif self.method == 'CN':
            tsi = target_series if target_series.is_evenly_spaced() else target_series.interp()
            mu = tsi.value.mean()
            sigma = tsi.value.std()
            alpha = tsi.spectral(method='cwt').beta_est().beta_est_res['beta'] # fit the parameter using the smoothest spectral method
            self.param = [alpha]
            y_surr = np.empty((len(target_series.time),self.number))
            for i in range(self.number):
                y_surr[:,i] = tsmodel.colored_noise(alpha=alpha,t=target_series.time, std = sigma) + mu
        
        if self.number > 1:
            s_list = []
            for i, y in enumerate(y_surr.T):
                ts = target_series.copy() # copy Series
                ts.value = y 
                ts.label = str(target_series.label or '') + " #" + str(i+1)
                s_list.append(ts)
        else:
            ts = target_series.copy() # copy Series
            ts.value = y_surr
            ts.label = str(target_series.label or '') 
            s_list = [ts]

        self.series_list = s_list
                  
    def from_param(self, param, length=50, time_pattern = 'even', settings=None):
        '''
        Simulate the SurrogateSeries from a parametric model

        Parameters
        ----------
        param : list
            model parameters (e.g. [tau, sigma0] for an AR(1) model, [beta] for colored noise)
        
        length : int
            Length of the series. Default: 50
            
        time_pattern : str {even, random, specified}
            The pattern used to generate the surrogate time axes
            * 'even' uses an evenly-spaced time with spacing `delta_t` specified in settings (if not specified, defaults to 1.0)
            * 'random' uses random_time_axis() with specified distribution and parameters 
               Relevant settings are `delta_t_dist` and `param`. (default: 'exponential' with parameter = 1.0)
            * 'specified': uses time axis `time` from `settings`.  
            
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

        pyleoclim.utils.tsmodel.colored_noise: simulating from a power law spectrum, $S(f) \propto f^{-\beta}$

        pyleoclim.utils.tsmodel.random_time_axis : Generate time increment vector according to a specific probability model
        
        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo

            ar1 = pyleo.SurrogateSeries(method='ar1sim', number=10)
            ar1.from_param(length=100, param = [2,2])
            ar1.plot_envelope(title=rf'AR(1) synthetic series ($\tau={2},\sigma^2={2}$)')

        '''    
        param = param if isinstance(param, list) else [param] # coerce param into a list, no matter the original format
        nparam = len(param)
        
        settings = {} if settings is None else settings.copy()
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # generate time axes according to provided pattern
        if time_pattern == "even":
            delta_t = settings["delta_t"] if "delta_t" in settings else 1.0
            t = np.cumsum([float(delta_t)]*length)
            times = np.tile(t, (self.number, 1)).T     
        elif time_pattern == "random":
            times = np.zeros((length, self.number))
            for i in range(self.number):
                dist_name = settings['delta_t_dist'] if "delta_t_dist" in settings else "exponential"
                dist_param = settings['param'] if "param" in settings else [1]
                times[:, i] = tsmodel.random_time_axis(length, dist_name,dist_param) 
        elif time_pattern == 'specified':
            if "time" not in settings:
                raise ValueError("'time' not found in settings")
            else:
                if self.number > 1:
                    times =  np.tile(settings["time"], (self.number, 1)).T 
                else:
                    times = settings["time"]
        else:
            raise ValueError(f"Unknown time pattern: {time_pattern}")
         
        # apply surrogate method
        y_surr = np.empty_like(times)                 
        
        if self.method in ['uar1','ar1sim']: # AR(1) models
            if nparam != 2:
                warnings.warn(f'The AR(1) model needs 2 parameters, tau and sigma2 (in that order); {nparam} provided. default values used, tau=5, sigma=0.5',UserWarning, stacklevel=2)
                param = [5,0.5]
            y_surr = tsmodel.uar1_sim(t = times, tau=param[0], sigma_2=param[1])
            
        elif self.method == 'CN': # colored noise
            if nparam >1:
                raise ValueError(f"1 parameter is needed for this model; only the first of the provided {nparam} will be used")
            y_surr = np.empty((length,self.number))
            if self.number > 1:
                for i in range(self.number):
                    y_surr[:,i] = tsmodel.colored_noise(alpha=param[0],t=times[:,i])
            else:
                y_surr = tsmodel.colored_noise(alpha=param[0],t=times)
                 
        elif self.method == 'phaseran':  
            raise ValueError("Phase-randomization is only available in from_series().")
        
        # create the series_list    
        s_list = []
        if self.number > 1:
            for i, (t, y) in enumerate(zip(times.T,y_surr.T)):
                ts = Series(time=t, value=y,  
                               label = str(self.label or '') + " #" + str(i+1),
                               verbose=False, auto_time_params=True)
                s_list.append(ts)
        else:
            ts = Series(time=np.squeeze(times), value=np.squeeze(y_surr),  
                        label = str(self.label or '') + " #`",
                        verbose=False, auto_time_params=True)
            s_list.append(ts)

        self.series_list = s_list
                  
        
