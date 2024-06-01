"""
The EnsembleSeries class is a child of MultipleSeries, designed for ensemble applications (e.g. draws from a posterior distribution of ages, model ensembles with randomized initial conditions, or some other stochastic ensemble).
In addition to a MultipleSeries object, an EnsembleSeries object has the following properties:
- All series members are assumed to share the same units and other metadata.
- The class enables ensemble-oriented methods for computation (e.g., quantiles) and visualization (e.g., envelope plot).    

"""

from ..utils import plotting, lipdutils
from ..utils import correlation as corrutils
from ..core.series import Series
from ..core.correns import CorrEns
from ..core.multipleseries import MultipleSeries

import warnings

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy

from matplotlib.ticker import FormatStrFormatter
import matplotlib.transforms as transforms
import matplotlib as mpl
from tqdm import tqdm

class EnsembleSeries(MultipleSeries):
    ''' EnsembleSeries object

    The EnsembleSeries object is a child of the MultipleSeries object, that is, a special case of MultipleSeries, aiming for ensembles of similar series.
    Ensembles usually arise from age modeling or Bayesian calibrations. All members of an EnsembleSeries object are assumed to share identical labels and units.

    All methods available for MultipleSeries are available for EnsembleSeries. Some functions were modified for the special case of ensembles.
    The class enables ensemble-oriented methods for computation (e.g., quantiles) 
    and visualization (e.g., envelope plot) that are unavailable to other classes.

    '''
    def __init__(self, series_list, label=None):
        self.series_list = series_list
        self.label = label

    @classmethod
    def from_AgeEnsembleArray(self, series, age_array, value_depth = None, age_depth = None, extrapolate=True,verbose=True):
        '''Function to create an EnsembleSeries object

        Function assumes that the input series and the age array share the same units.
        If depth vectors are passed, these are also assumed to share the same units

        Parameters
        ----------
        series : pyleoclim.core.series.Series
            A Series object with the values to be mapped

        age_array : np.array
            An array of ages to map the values to

        value_depth : vector
            An array of depths corresponding to the series values

        age_depth : vector
            An array of depths corresponding to the age array

        extrapolate : bool
            Whether to extrapolate the age array to the value depth. Default is True

        verbose : bool
            Whether to print warnings. Default is True

        Returns
        -------
        EnsembleSeries : pyleoclim.core.ensembleseries.EnsembleSeries
            The ensemble created using the time axes from age_array and the values from series.

        Examples
        --------

        .. jupyter-execute::

            #Create an ensemble of 100 series with random time axes of length 1000
            length = 1000
            age_array = np.array([pyleo.utils.tsmodel.random_time_axis(length) for i in range(100)]).T

            #Create a random series
            value = np.random.randn(length)
            time = pyleo.utils.tsmodel.random_time_axis(length)
            series = pyleo.Series(time=time,value=value, verbose=False)

            #Create an ensemble using these objects
            #Note that the time axis of the series object and the number of rows in the age array must match when depth is not passed
            ens = pyleo.EnsembleSeries.from_AgeEnsembleArray(series = series,age_array=age_array,verbose=False)

        .. jupyter-execute::

            #If we have depth vectors for our series and age array, we can pass them to the function
            age_length = 1000
            age_array = np.array([pyleo.utils.tsmodel.random_time_axis(age_length) for i in range(100)]).T
            age_depth = np.arange(age_length)

            value_length = 800
            value = np.random.randn(value_length)
            time = pyleo.utils.tsmodel.random_time_axis(value_length)
            series = pyleo.Series(time=time, value=value, verbose=False)
            value_depth = np.arange(value_length)

            #Note that the length of the depth vectors must match the length of the corresponding object (number of values or number of rows in age array)
            ens = pyleo.EnsembleSeries.from_AgeEnsembleArray(series = series,age_array=age_array, value_depth=value_depth, age_depth=age_depth,verbose=False)
        '''

        if not isinstance(series, Series):
            raise ValueError('series must be a GeoSeries object')

        #squeeze paleoValues into a vector
        values = np.squeeze(np.array(series.value))
        
        if age_depth is None:
            if value_depth is None:
                if len(values) != age_array.shape[0]:
                    raise ValueError("Age array and series need to have the same length when age_depth is not passed.")
                else:
                    mapped_age = age_array
                pass
            else:
                raise ValueError('Age_depth not found. Please pass both a value depth array and age depth array if value and age are not already aligned. Otherwise, pass neither.')
        else:
            #Check that both arrays were passed
            if value_depth is None:
                raise ValueError('Value_depth not found. Please pass both a value depth array and age depth array if value and age are not already aligned. Otherwise, pass neither.')
            
            #Make sure that numpy arrays were given and try to coerce them into vectors if possible
            age_depth=np.squeeze(np.array(age_depth))
            value_depth = np.squeeze(np.array(value_depth))

            #Check that arrays are vectors for np.interp
            if age_depth.ndim > 1:
                raise ValueError('chronDepth has more than one dimension, please pass it as a vector')
            if value_depth.ndim > 1:
                raise ValueError('paleoDepth has more than one dimension, please pass it as a vector')
            #Check that the shape of the depth arrays matches up with the age array and value vector (separately)
            if len(age_depth)!=age_array.shape[0]:
                raise ValueError("Age depth and age array need to have the same length")
            if len(value_depth)!=len(values):
                raise ValueError("Paleo depth and series time need to have the same length")
            
            #Interpolate the age array to the value depth
            mapped_age = lipdutils.mapAgeEnsembleToPaleoData(
                ensembleValues=age_array, 
                depthEnsemble=age_depth, 
                depthPaleo=value_depth,
                extrapolate=extrapolate
            )
        
        series_list = []

        #check that mapped_age and the original time vector are similar
        if verbose:
            if (np.mean(mapped_age[-1,:]) > 10*series.time[-1]) or (np.mean(mapped_age[-1,:]) < 0.1*series.time[-1]):
                warnings.warn('The mapped age array is significantly different from the original time vector. You may want to check that the units are appropriate.')
            elif (np.mean(mapped_age[0,:]) > 10*series.time[0]) or (np.mean(mapped_age[0,:]) < 0.1*series.time[0]):
                warnings.warn('The mapped age array is significantly different from the original time vector. You may want to check that the units are appropriate.')
        
        for s in mapped_age.T:
            series_tmp = series.copy()
            series_tmp.time = s
            series_list.append(series_tmp)

        return EnsembleSeries(series_list)

    def make_labels(self):
        '''Initialization of labels

        Returns
        -------
        time_header : str

            Label for the time axis

        value_header : str

            Label for the value axis

        '''
        ts_list = self.series_list

        if ts_list[0].time_name is not None:
            time_name_str = ts_list[0].time_name
        else:
            time_name_str = 'time'

        if ts_list[0].value_name is not None:
            value_name_str = ts_list[0].value_name
        else:
            value_name_str = 'value'

        if ts_list[0].value_unit is not None:
            value_header = f'{value_name_str} [{ts_list[0].value_unit}]'
        else:
            value_header = f'{value_name_str}'

        if ts_list[0].time_unit is not None:
            time_header = f'{time_name_str} [{ts_list[0].time_unit}]'
        else:
            time_header = f'{time_name_str}'

        return time_header, value_header
    
    def slice(self, timespan):
        ''' Selects a limited time span from the object 

        Parameters
        ----------
        timespan : tuple or list
            The list of time points for slicing, whose length must be even.
            When there are n time points, the output Series includes n/2 segments.
            For example, if timespan = [a, b], then the sliced output includes one segment [a, b];
            if timespan = [a, b, c, d], then the sliced output includes segment [a, b] and segment [c, d].

        Returns
        -------
        new : EnsembleSeries
            The sliced EnsembleSeries object.

        Examples
        --------

        Select part of an object

        .. jupyter-execute::

            nn = 20 # number of noise realizations
            nt = 200
            series_list = []

            time, signal = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=2.0)
            
            ts = pyleo.Series(time=time, value = signal, verbose=False).standardize()
            noise = np.random.randn(nt,nn)

            for idx in range(nn):  # noise
                ts = pyleo.Series(time=time, value=ts.value+5*noise[:,idx], verbose=False)
                series_list.append(ts)

            ts_ens = pyleo.EnsembleSeries(series_list)
            
            fig, ax = ts_ens.plot_envelope(curve_lw=1.5)
            fig, ax = ts_ens.slice([100, 199]).plot_envelope(curve_lw=1.5)

        '''
        new = self.copy()
        
        for idx, ts in enumerate(self.series_list):
            tsc = ts.slice(timespan)
            new.series_list[idx] = tsc
            
        return new

    

    def quantiles(self, qs=[0.05, 0.5, 0.95], axis = 'value'):
        '''Calculate quantiles of an EnsembleSeries object. If axis is 'value', the calculation requires for the time axis to be the same. You can use the common_time method to do so. In essence, it transforms the time uncertainty into a y-axis uncertainty. If axis is 'time', the values should be the same for all members of the emsemble. 

        Reuses [scipy.stats.mstats.mquantiles](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html) function.

        Parameters
        ----------
        qs : list, optional

            List of quantiles to consider for the calculation. The default is [0.05, 0.5, 0.95].
        
        axis : ['time', 'value']
            
            Whether to calculate the quantiles over the values or time. Default is 'value'. 

        Returns
        -------
        ens_qs : EnsembleSeries

            EnsembleSeries object containing empirical quantiles of original 
        
        See also
        --------
        
        pyleoclim.core.multipleseries.MultipleSeries.common_time : A method to align axes

        Examples
        --------

        .. jupyter-execute::

            nn = 30 # number of noise realizations
            nt = 500
            series_list = []

            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            signal = pyleo.Series(t,v)

            for idx in range(nn):  # noise
                noise = np.random.randn(nt,nn)*100
                ts = pyleo.Series(time=signal.time, value=signal.value+noise[:,idx], verbose=False)
                series_list.append(ts)

            ts_ens = pyleo.EnsembleSeries(series_list)

            ens_qs = ts_ens.quantiles()
            
        To calculate in the time dimension:
            
        .. jupyter-execute::
            
            nn = 30 #number of age models
            time = np.arange(1,20000,100) #create a time vector
            std_dev = 20 # Noise to be considered
            
            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=len(time),alpha=1.0)
            
            series_list = []
            
            for i in range(nn):
                noise = np.random.normal(0,std_dev,len(time))
                ts=pyleo.Series(time=np.sort(time+noise),value=v,verbose=False)
                series_list.append(ts)
            
            time_ens = pyleo.EnsembleSeries(series_list)
            
            ens_qs = time_ens.quantiles(axis='time')            

        '''
        if axis == 'value':
            time = np.copy(self.series_list[0].time)
            vals = []
            for ts in self.series_list:
                if not np.array_equal(ts.time, time):
                    raise ValueError('Time axis not consistent across the ensemble!')
    
                vals.append(ts.value)
    
            vals = np.array(vals)
            # ens_qs = mquantiles(vals, qs, axis=0)
            ens_qs = np.nanquantile(vals, qs, axis=0)
    
            ts_list = []
            for i, quant in enumerate(ens_qs):
                ts = Series(time=time, value=quant, label=f'{qs[i]*100:g}%', verbose=False)
                ts_list.append(ts)
        
        elif axis == 'time':
            
            value = np.copy(self.series_list[0].value)
            vals = []
            for ts in self.series_list:
                if not np.array_equal(ts.value, value):
                    raise ValueError('Value axis not consistent across the ensemble!')
                vals.append(ts.time)        
            
            vals = np.array(vals)
                    
            # ens_qs = mquantiles(vals, qs, axis=0)
            ens_qs = np.nanquantile(vals, qs, axis=0)
    
            ts_list = []
            for i, quant in enumerate(ens_qs):
                ts = Series(time=quant, value=value, label=f'{qs[i]*100:g}%', verbose=False)
                ts_list.append(ts)
        
        else:
            raise ValueError("Axis should be either 'value' or 'time'")

        ens_qs = EnsembleSeries(series_list=ts_list)

        return ens_qs

    def correlation(self, target=None, timespan=None, alpha=0.05, method = 'ttest', statistic = 'pearsonr',number=1000,
                    settings=None, fdr_kwargs=None, common_time_kwargs=None, mute_pbar=False, seed=None):
        ''' Calculate the correlation between an EnsembleSeries object to a target.

        If the target is not specified, then the 1st member of the ensemble will be the target
        Note that the FDR approach is applied by default to determine the significance of the p-values (more information in See Also below).

        Parameters
        ----------
        target : Series or EnsembleSeries

            A pyleoclim Series object or EnsembleSeries object.
            When the target is also an EnsembleSeries object, then the calculation of correlation is performed in a one-to-one sense,
            and the ourput list of correlation values and p-values will be the size of the series_list of the self object.
            That is, if the self object contains n Series, and the target contains n+m Series,
            then only the first n Series from the object will be used for the calculation;
            otherwise, if the target contains only n-m Series, then the first m Series in the target will be used twice in sequence.

        timespan : tuple

            The time interval over which to perform the calculation

        alpha : float

            The significance level (0.05 by default)
       
        method : str, {'ttest','built-in','ar1sim','phaseran'}
            method for significance testing. Default is 'ttest'
            
        statistic : str
            The name of the statistic used to measure the association, to be chosen from a subset of
            https://docs.scipy.org/doc/scipy/reference/stats.html#association-correlation-tests
            Currently supported: ['pearsonr','spearmanr','pointbiserialr','kendalltau','weightedtau']
            The default is 'pearsonr'.

        settings : dict

            Parameters for the correlation function (per scipy)

        number : int
                the number of simulations (default: 1000)

        fdr_kwargs : dict

            Parameters for the FDR function

        common_time_kwargs : dict

            Parameters for the method MultipleSeries.common_time()

        mute_pbar : bool; {True,False}

            If True, the progressbar will be muted. Default is False.

        seed : float or int

            random seed for isopersistent and isospectral methods

        Returns
        -------
        corr_ens : CorrEns

            The resulting object, see pyleoclim.CorrEns

        See also
        --------

        pyleoclim.utils.correlation.corr_sig : Correlation function

        pyleoclim.utils.correlation.fdr : False Discovery Rate

        pyleoclim.core.correns.CorrEns : The correlation ensemble object

        Examples
        --------

        .. jupyter-execute::

            nn = 50 # number of noise realizations
            nt = 100
            series_list = []

            time, signal = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=2.0)
            
            ts = pyleo.Series(time=time, value = signal, verbose=False).standardize()
            noise = np.random.randn(nt,nn)

            for idx in range(nn):  # noise
                ts = pyleo.Series(time=time, value=ts.value+5*noise[:,idx], verbose=False)
                series_list.append(ts)

            ts_ens = pyleo.EnsembleSeries(series_list)
              
            # to set an arbitrary random seed to fix the result
            corr_res = ts_ens.correlation(ts, seed=2333)
            print(corr_res)
            
            # to change the statistic: 
            corr_res = ts_ens.correlation(ts, statistic='kendalltau', method='phaseran', number=20)
            print(corr_res)
            
        The `print` function tabulates the output, and conveys the p-value according
        to the correlation test applied ("isospec", by default). To plot the result:
        
        .. jupyter-execute::
            
            corr_res.plot()

        '''
        if target is None:
            target = self.series_list[0]

        r_list = []
        p_list = []
        signif_list = []
        print("Looping over "+ str(len(self.series_list)) +" Series in the ensemble")
        for idx, ts1 in tqdm(enumerate(self.series_list), total=len(self.series_list), disable=mute_pbar):
            if hasattr(target, 'series_list'):
                nEns = np.size(target.series_list)
                if idx < nEns:
                    value2 = target.series_list[idx].value
                    time2 = target.series_list[idx].time
                else:
                    value2 = target.series_list[idx-nEns].value
                    time2 = target.series_list[idx-nEns].time
            else:
                value2 = target.value
                time2 = target.time

            ts2 = Series(time=time2, value=value2, verbose=idx==0, auto_time_params=False)
            corr_res = ts1.correlation(ts2, timespan=timespan, method=method,number=number,
                                       statistic=statistic,
                                       settings=settings, mute_pbar=True,
                                       common_time_kwargs=common_time_kwargs, seed=seed)
            r_list.append(corr_res.r)
            signif_list.append(corr_res.signif)
            p_list.append(corr_res.p)

        r_list = np.array(r_list)
        p_list = np.array(p_list)

        signif_fdr_list = []
        fdr_kwargs = {} if fdr_kwargs is None else fdr_kwargs.copy()
        args = {}
        args.update(fdr_kwargs)
        for i in range(np.size(signif_list)):
            signif_fdr_list.append(False)

        fdr_res = corrutils.fdr(p_list, **fdr_kwargs)
        if fdr_res is not None:
            for i in fdr_res:
                signif_fdr_list[i] = True

        corr_ens = CorrEns(r_list, p_list, signif_list, signif_fdr_list, alpha)
        return corr_ens

    def plot_traces(self, figsize=[10, 4], xlabel=None, ylabel=None, title=None, num_traces=10, seed=None,
             xlim=None, ylim=None, linestyle='-', savefig_settings=None, ax=None, plot_legend=True,
             color=sns.xkcd_rgb['pale red'], lw=0.5, alpha=0.3, lgd_kwargs=None):
        '''Plot EnsembleSeries as a subset of traces.

        Parameters
        ----------
        figsize : list, optional

            The figure size. The default is [10, 4].

        xlabel : str, optional

            x-axis label. The default is None.

        ylabel : str, optional

            y-axis label. The default is None.

        title : str, optional

            Plot title. The default is None.

        xlim : list, optional

            x-axis limits. The default is None.

        ylim : list, optional

            y-axis limits. The default is None.

        color : str, optional

            Color of the traces. The default is sns.xkcd_rgb['pale red'].

        alpha : float, optional

            Transparency of the lines representing the multiple members. The default is 0.3.

        linestyle : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}

            Set the linestyle of the line

        lw : float, optional

            Width of the lines representing the multiple members. The default is 0.5.

        num_traces : int, optional

            Number of traces to plot. The default is 10.

        savefig_settings : dict, optional

            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
                with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.

        ax : matplotlib.ax, optional

            Matplotlib axis on which to return the plot. The default is None.

        plot_legend : bool; {True,False}, optional

            Whether to plot the legend. The default is True.

        lgd_kwargs : dict, optional

            Parameters for the legend. The default is None.

        seed : int, optional

            Set the seed for the random number generator. Useful for reproducibility. The default is None.

        Returns
        -------
        fig : matplotlib.figure
        
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html) for details.

        ax : matplotlib.axis
        
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        See also
        --------

        pyleoclim.utils.plotting.savefig : Saving figure in Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            nn = 30 # number of noise realizations
            nt = 500
            series_list = []

            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            signal = pyleo.Series(time=t,value=v, verbose=False)

            for idx in range(nn):  # noise
                noise = np.random.randn(nt,nn)*100
                ts = pyleo.Series(time=signal.time, value=signal.value+noise[:,idx], verbose=False)
                series_list.append(ts)

            ts_ens = pyleo.EnsembleSeries(series_list)

            fig, ax = ts_ens.plot_traces(alpha=0.2,num_traces=8)

            '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()
        
        num_traces = min(num_traces, len(self.series_list)) # restrict to the smaller of the two

        # generate default axis labels
        time_label, value_label = self.make_labels()

        if xlabel is None:
            xlabel = time_label

        if ylabel is None:
            ylabel = value_label

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if num_traces > 0:
            if seed is not None:
                np.random.seed(seed)

            nts = np.size(self.series_list)
            random_draw_idx = np.random.choice(nts, num_traces, replace=False)

            for idx in random_draw_idx:
                self.series_list[idx].plot(xlabel=xlabel, ylabel=ylabel, zorder=99, linewidth=lw,
                    xlim=xlim, ylim=ylim, ax=ax, color=color, alpha=alpha,linestyle='-')
            ax.plot(np.nan, np.nan, color=color, label=f'example members (n={num_traces})',linestyle='-')

        if title is not None:
            ax.set_title(title)
        else:
            if self.label is not None:
                ax.set_title(self.label)
            
        if plot_legend:
            lgd_args = {'frameon': False}
            lgd_args.update(lgd_kwargs)
            ax.legend(**lgd_args)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax

    def plot_envelope(self, figsize=[10, 4], qs=[0.025, 0.25, 0.5, 0.75, 0.975],
                      xlabel=None, ylabel=None, title=None,
                      xlim=None, ylim=None, savefig_settings=None, ax=None, plot_legend=True,
                      curve_clr=sns.xkcd_rgb['pale red'], curve_lw=2, shade_clr=sns.xkcd_rgb['pale red'], shade_alpha=0.2,
                      inner_shade_label='IQR', outer_shade_label='95% CI', lgd_kwargs=None):
        ''' Plot EnsembleSeries as an envelope.

        Parameters
        ----------
        figsize : list, optional

            The figure size. The default is [10, 4].

        qs : list, optional

            The significance levels to consider. The default is [0.025, 0.25, 0.5, 0.75, 0.975] (median, interquartile range, and central 95% region)

        xlabel : str, optional

            x-axis label. The default is None.

        ylabel : str, optional

            y-axis label. The default is None.

        title : str, optional

            Plot title. The default is None.

        xlim : list, optional

            x-axis limits. The default is None.

        ylim : list, optional

            y-axis limits. The default is None.

        savefig_settings : dict, optional

            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.

        ax : matplotlib.ax, optional

            Matplotlib axis on which to return the plot. The default is None.

        plot_legend : bool; {True,False}, optional

            Wether to plot the legend. The default is True.

        curve_clr : str, optional

            Color of the main line (median). The default is sns.xkcd_rgb['pale red'].

        curve_lw : str, optional

            Width of the main line (median). The default is 2.

        shade_clr : str, optional

            Color of the shaded envelope. The default is sns.xkcd_rgb['pale red'].

        shade_alpha : float, optional

            Transparency on the envelope. The default is 0.2.

        inner_shade_label : str, optional

            Label for the envelope. The default is 'IQR'.

        outer_shade_label : str, optional

            Label for the envelope. The default is '95\% CI'.

        lgd_kwargs : dict, optional

            Parameters for the legend. The default is None.

        Returns
        -------
        fig : matplotlib.figure
        
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html) for details.

        ax : matplotlib.axis
        
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        See also
        --------

        pyleoclim.utils.plotting.savefig : Saving figure in Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            nn = 30 # number of noise realizations
            nt = 500
            series_list = []

            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            signal = pyleo.Series(time=t,value=v, verbose=False)

            for idx in range(nn):  # noise
                noise = np.random.randn(nt,nn)*100
                ts = pyleo.Series(time=signal.time, value=signal.value+noise[:,idx], verbose=False)
                series_list.append(ts)

            ts_ens = pyleo.EnsembleSeries(series_list)

            fig, ax = ts_ens.plot_envelope(curve_lw=1.5)

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        # generate default axis labels
        time_label, value_label = self.make_labels()

        if xlabel is None:
            xlabel = time_label

        if ylabel is None:
            ylabel = value_label

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ts_qs = self.quantiles(qs=qs)

        if inner_shade_label is None:
            inner_shade_label = f'{ts_qs.series_list[1].label}-{ts_qs.series_list[-2].label}'

        if outer_shade_label is None:
            outer_shade_label = f'{ts_qs.series_list[0].label}-{ts_qs.series_list[-1].label}'

        time = ts_qs.series_list[0].time
        # plot outer envelope
        ax.fill_between(
            time, ts_qs.series_list[0].value, ts_qs.series_list[4].value,
            color=shade_clr, alpha=shade_alpha, edgecolor=shade_clr, label=outer_shade_label
        )
        # plot inner envelope on top
        ax.fill_between(
            time, ts_qs.series_list[1].value, ts_qs.series_list[3].value,
            color=shade_clr, alpha=2*shade_alpha, edgecolor=shade_clr, label=inner_shade_label
        )

        # plot the median
        ts_qs.series_list[2].plot(xlabel=xlabel, ylabel=ylabel, linewidth=curve_lw, color=curve_clr,
            xlim=xlim, ylim=ylim, ax=ax,  zorder=100, label = 'median'
        )

        if title is not None:
            ax.set_title(title)

        if plot_legend:
            lgd_args = {'frameon': False}
            lgd_args.update(lgd_kwargs)
            ax.legend(**lgd_args)
        else:
            ax.legend().set_visible(False)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax


    def stackplot(self, figsize=[5, 15], savefig_settings=None,  xlim=None, fill_between_alpha=0.2, colors=None, cmap='tab10', norm=None,
                  spine_lw=1.5, grid_lw=0.5, font_scale=0.8, label_x_loc=-0.15, v_shift_factor=3/4, linewidth=1.5):
        ''' Stack plot of multiple series

        Note that the plotting style is uniquely designed for this one and cannot be properly reset with `pyleoclim.set_style()`.
        
        Parameters
        ----------
        figsize : list

            Size of the figure.

        colors : list

            Colors for plotting.
            If None, the plotting will cycle the 'tab10' colormap;
            if only one color is specified, then all curves will be plotted with that single color;
            if a list of colors are specified, then the plotting will cycle that color list.

        cmap : str

            The colormap to use when "colors" is None.

        norm : matplotlib.colors.Normalize like

            The nomorlization for the colormap.
            If None, a linear normalization will be used.

        savefig_settings : dictionary

            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.

        xlim : list

            The x-axis limit.

        fill_between_alpha : float

            The transparency for the fill_between shades.

        spine_lw : float

            The linewidth for the spines of the axes.

        grid_lw : float

            The linewidth for the gridlines.

        linewidth : float

            The linewidth for the curves.

        font_scale : float

            The scale for the font sizes. Default is 0.8.

        label_x_loc : float

            The x location for the label of each curve.

        v_shift_factor : float

            The factor for the vertical shift of each axis.
            The default value 3/4 means the top of the next axis will be located at 3/4 of the height of the previous one.

        Returns
        -------
        fig : matplotlib.figure
        
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html) for details.

        ax : matplotlib.axis
        
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        See also
        --------

        pyleoclim.utils.plotting.savefig : Saving figure in Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            nn = 10 # number of noise realizations
            nt = 200
            series_list = []

            t, v = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            signal, _, _ = pyleo.utils.standardize(v)
            noise = np.random.randn(nt,nn)

            for idx in range(nn):  # noise
                ts = pyleo.Series(time=t, value=signal+noise[:,idx], label='trace #'+str(idx+1), verbose=False)
                series_list.append(ts)

            ts_ens = pyleo.EnsembleSeries(series_list)

            fig, ax = ts_ens.stackplot()

        '''
        current_style = deepcopy(mpl.rcParams)
        plotting.set_style('journal', font_scale=font_scale)
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        n_ts = len(self.series_list)

        fig = plt.figure(figsize=figsize)

        if xlim is None:
            time_min = np.inf
            time_max = -np.inf
            for ts in self.series_list:
                if np.min(ts.time) <= time_min:
                    time_min = np.min(ts.time)
                if np.max(ts.time) >= time_max:
                    time_max = np.max(ts.time)
            xlim = [time_min, time_max]

        ax = {}
        left = 0
        width = 1
        height = 1/n_ts
        bottom = 1
        for idx, ts in enumerate(self.series_list):
            if colors is None:
                cmap_obj = plt.get_cmap(cmap)
                if hasattr(cmap_obj, 'colors'):
                    nc = len(cmap_obj.colors)
                else:
                    nc = len(self.series_list)

                if norm is None:
                    norm = mpl.colors.Normalize(vmin=0, vmax=nc-1)

                clr = cmap_obj(norm(idx%nc))
            elif type(colors) is str:
                clr = colors
            elif type(colors) is list:
                nc = len(colors)
                clr = colors[idx%nc]
            else:
                raise TypeError('"colors" should be a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)')

            bottom -= height*v_shift_factor
            ax[idx] = fig.add_axes([left, bottom, width, height])
            ax[idx].plot(ts.time, ts.value, color=clr, lw=linewidth)
            ax[idx].patch.set_alpha(0)
            ax[idx].set_xlim(xlim)
            time_label, value_label = ts.make_labels()
            ax[idx].set_ylabel(value_label, weight='bold')

            mu = np.mean(ts.value)
            std = np.std(ts.value)
            ylim = [mu-4*std, mu+4*std]
            ax[idx].fill_between(ts.time, ts.value, y2=mu, alpha=fill_between_alpha, color=clr)
            trans = transforms.blended_transform_factory(ax[idx].transAxes, ax[idx].transData)
            if ts.label is not None:
                ax[idx].text(label_x_loc, mu, ts.label, horizontalalignment='right', transform=trans, color=clr, weight='bold')
            ax[idx].set_ylim(ylim)
            ax[idx].set_yticks(ylim)
            ax[idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax[idx].grid(False)
            if idx % 2 == 0:
                ax[idx].spines['left'].set_visible(True)
                ax[idx].spines['left'].set_linewidth(spine_lw)
                ax[idx].spines['left'].set_color(clr)
                ax[idx].spines['right'].set_visible(False)
                ax[idx].yaxis.set_label_position('left')
                ax[idx].yaxis.tick_left()
            else:
                ax[idx].spines['left'].set_visible(False)
                ax[idx].spines['right'].set_visible(True)
                ax[idx].spines['right'].set_linewidth(spine_lw)
                ax[idx].spines['right'].set_color(clr)
                ax[idx].yaxis.set_label_position('right')
                ax[idx].yaxis.tick_right()

            ax[idx].yaxis.label.set_color(clr)
            ax[idx].tick_params(axis='y', colors=clr)
            ax[idx].spines['top'].set_visible(False)
            ax[idx].spines['bottom'].set_visible(False)
            ax[idx].tick_params(axis='x', which='both', length=0)
            ax[idx].set_xlabel('')
            ax[idx].set_xticklabels([])
            xt = ax[idx].get_xticks()[1:-1]
            for x in xt:
                ax[idx].axvline(x=x, color='lightgray', linewidth=grid_lw, ls='-', zorder=-1)
            ax[idx].axhline(y=mu, color='lightgray', linewidth=grid_lw, ls='-', zorder=-1)

        bottom -= height*(1-v_shift_factor)
        ax[n_ts] = fig.add_axes([left, bottom, width, height])
        ax[n_ts].set_xlabel(time_label)
        ax[n_ts].spines['left'].set_visible(False)
        ax[n_ts].spines['right'].set_visible(False)
        ax[n_ts].spines['bottom'].set_visible(True)
        ax[n_ts].spines['bottom'].set_linewidth(spine_lw)
        ax[n_ts].set_yticks([])
        ax[n_ts].patch.set_alpha(0)
        ax[n_ts].set_xlim(xlim)
        ax[n_ts].grid(False)
        ax[n_ts].tick_params(axis='x', which='both', length=3.5)
        xt = ax[n_ts].get_xticks()[1:-1]
        for x in xt:
            ax[n_ts].axvline(x=x, color='lightgray', linewidth=grid_lw, ls='-', zorder=-1)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            mpl.rcParams.update(current_style)
            return fig, ax
        else:
            # reset the plotting style
            mpl.rcParams.update(current_style)
            return ax


    def histplot(self, figsize=[10, 4], title=None, savefig_settings=None,
                 ax=None, ylabel='KDE', vertical=False, edgecolor='w', **plot_kwargs):
        """ Plots the distribution of the timeseries across ensembles

        Reuses the seaborn [histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html) function.

        Parameters
        ----------
        figsize : list, optional

            The size of the figure. The default is [10, 4].

        title : str, optional

            Title for the figure. The default is None.

        savefig_settings : dict, optional

            the dictionary of arguments for plt.savefig(); some notes below:
              - "path" must be specified; it can be any existed or non-existed path,
                with or without a suffix; if the suffix is not given in "path", it will follow "format"
              - "format" can be one of {"pdf", "eps", "png", "ps"}.
            The default is None.

        ax : matplotlib.axis, optional

            A matplotlib axis. The default is None.

        ylabel : str, optional

            Label for the count axis. The default is 'KDE'.

        vertical : bool; {True,False}, optional

            Whether to flip the plot vertically. The default is False.

        edgecolor : matplotlib.color, optional

            The color of the edges of the bar. The default is 'w'.

        plot_kwargs : dict

            Plotting arguments for seaborn histplot: https://seaborn.pydata.org/generated/seaborn.histplot.html.

        See also
        --------

        pyleoclim.utils.plotting.savefig : Saving figure in Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            nn = 30 # number of noise realizations
            nt = 500
            series_list = []

            time, signal = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            
            ts = pyleo.Series(time=time, value = signal, verbose=False).standardize()
            noise = np.random.randn(nt,nn)

            for idx in range(nn):  # noise
                ts = pyleo.Series(time=time, value=signal+noise[:,idx], verbose=False)
                series_list.append(ts)

            ts_ens = pyleo.EnsembleSeries(series_list)

            fig, ax = ts_ens.histplot()

        """
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        #make the data into a dataframe so we can flip the figure
        time_label, value_label = self.make_labels()

        #append all the values together for the plot
        val = self.series_list[0].value
        for i in range(1,len(self.series_list)):
            val=np.append(val,self.series_list[i].value)

        if vertical == True:
            data=pd.DataFrame({'value':val})
            ax = sns.histplot(data=data, y="value", ax=ax, kde=True, edgecolor=edgecolor, **plot_kwargs)
            ax.set_ylabel(value_label)
            ax.set_xlabel(ylabel)
        else:
            ax = sns.histplot(val, ax=ax, kde=True, edgecolor=edgecolor, **plot_kwargs)
            ax.set_xlabel(value_label)
            ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax

    def to_dataframe(self, axis = 'value'):
        '''
        Export the ensemble as a Pandas DataFrame, with members of the ensemble as columns. The columns are labeled according to the label in the individual series or numbered if 'label' is None.
        

        Parameters
        ----------
        axis : str, ['time', 'value']
            Whether the return the ensemble from value or time. each The default is 'value'.

        Raises
        ------
        ValueError
            Axis should be either 'time' or 'value'

        Returns
        -------
        df : pandas.DataFrame
            A Pandas DataFrame containing members of the ensemble as columns. 
        
        .. jupyter-execute::
            
            nn = 30 #number of age models
            time = np.arange(1,20000,100) #create a time vector
            std_dev = 20 # Noise to be considered
            
            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=len(time),alpha=1.0)
            
            series_list = []
            
            for i in range(nn):
                noise = np.random.normal(0,std_dev,len(time))
                ts=pyleo.Series(time=np.sort(time+noise),value=v,verbose=False)
                series_list.append(ts)
            
            time_ens = pyleo.EnsembleSeries(series_list)
            ens_qs = time_ens.quantiles(axis='time')
            
            df=ens_qs.to_dataframe(axis='time')

        '''
        
        df_dict = {}
        idx = 0
        
        if axis == 'value':
            for ts in self.series_list:
                if ts.label is None:
                    df_dict[idx]=ts.value
                else:
                    df_dict[ts.label]=ts.value
                idx+=1
        
        elif axis == 'time':
            for ts in self.series_list:
                if ts.label is None:
                    df_dict[idx]=ts.time
                else:
                    df_dict[ts.label]=ts.time
                idx+=1
        
        else:
            raise ValueError('Axis should be either "time" or "value"')
            
        df = pd.DataFrame(df_dict)
        
        return df
    
    def to_array(self, axis='value', labels=True):
        '''
        Returns an ensemble as a numpy array with an optional list for labels. Each column in the array corresponds to an ensemble member.

        Parameters
        ----------
        axis : str, ['time', 'value'], optional
            Whether the return the ensemble from value or time. The default is 'value'.
        labels : bool, [True,False], optional
            Whether to retrun a separate list with the timseries labels. The default is True.

        Raises
        ------
        ValueError
            Axis should be either 'time' or 'value'

        Returns
        -------
        vals: numpy.array
            An array where each column corresponds to an ensemble member
            
        headers: list
            A list of corresponding labels for each columm
            
        Examples
        --------
            
        .. jupyter-execute::
            
            nn = 30 #number of age models
            time = np.arange(1,20000,100) #create a time vector
            std_dev = 20 # Noise to be considered
            
            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=len(time),alpha=1.0)
            
            series_list = []
            
            for i in range(nn):
                noise = np.random.normal(0,std_dev,len(time))
                ts=pyleo.Series(time=np.sort(time+noise),value=v,verbose=False)
                series_list.append(ts)
            
            time_ens = pyleo.EnsembleSeries(series_list)
            ens_qs = time_ens.quantiles(axis='time')
            
            vals,headers=ens_qs.to_array(axis='time')    

        '''
        
        vals=np.empty((len(self.series_list[0].value),len(self.series_list)))
        headers=[]
        
        if axis == 'value':
            for i, ts in enumerate(self.series_list):
                headers.append(ts.label)
                vals[:,i]=ts.value
        
        elif axis == 'time':
            for i, ts in enumerate(self.series_list):
                headers.append(ts.label)
                vals[:,i]=ts.time
        
        else:
            raise ValueError('Axis should be either "time" or "value"')
        
        if labels == True:
            return vals, headers
        else:
            return vals
            
        