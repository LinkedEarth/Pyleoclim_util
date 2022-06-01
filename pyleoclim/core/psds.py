from ..utils import plotting, lipdutils
from ..utils import wavelet as waveutils
from ..utils import spectral as specutils

#from ..core.multiplepsd import *
#import ..core.multiplepsd


import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from copy import deepcopy
import warnings

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter


def infer_period_unit_from_time_unit(time_unit):
    ''' infer a period unit based on the given time unit

    '''
    if time_unit is None:
        period_unit = None
    else:
        unit_group = lipdutils.timeUnitsCheck(time_unit)
        if unit_group != 'unknown':
            if unit_group == 'kage_units':
                period_unit = 'kyrs'
            else:
                period_unit = 'yrs'
        else:
            if time_unit[-1] == 's':
                period_unit = time_unit
            else:
                period_unit = f'{time_unit}s'

    return period_unit

class PSD:
    '''The PSD (Power spectral density) class is intended for conveniently manipulating
    the result of spectral methods, including performing significance tests,
    estimating scaling coefficients, and plotting. 

    See examples in pyleoclim.core.series.Series.spectral to see how to create and manipulate these objects

    Parameters
    ----------

    frequency : numpy.array, list, or float

        One or more frequencies in power spectrum

    amplitude : numpy.array, list, or float

        The amplitude at each (frequency, time) point;
        note the dimension is assumed to be (frequency, time)

    label : str, optional

        Descriptor of the PSD.
        Default is None

    timeseries : pyleoclim.Series, optional

        Default is None

    plot_kwargs : dict, optional

        Plotting arguments for seaborn histplot: https://seaborn.pydata.org/generated/seaborn.histplot.html.
        Default is None

    spec_method : str, optional

        The name of the spectral method to be applied on the timeseries
        Default is None

    spec_args : dict, optional

        Arguments for wavelet analysis ('freq', 'scale', 'mother', 'param')
        Default is None

    signif_qs : pyleoclim.MultipleScalogram, optional

        Pyleoclim MultipleScalogram object containing the quantiles qs of the surrogate scalogram distribution.
        Default is None

    signif_method : str, optional

        The method used to obtain the significance level.
        Default is None

    period_unit : str, optional

        Unit of time.
        Default is None

    beta_est_res : list or numpy.array, optional

        Results of the beta estimation calculation.
        Default is None.

    See also
    --------

    pyleoclim.core.series.Series.spectral : Spectral analysis

    pyleoclim.core.scalograms.Scalogram :  Scalogram object

    pyleoclim.core.scalograms.MultipleScalogram : Object storing multiple scalogram objects

    pyleoclim.core.psds.MultiplePSD : Object storing several PSDs from different Series or ensemble members in an age model



    '''
    def __init__(self, frequency, amplitude, label=None, timeseries=None, plot_kwargs=None,
                 spec_method=None, spec_args=None, signif_qs=None, signif_method=None, period_unit=None,
                 beta_est_res=None):

        self.frequency = np.array(frequency)
        self.amplitude = np.array(amplitude)
        self.label = label
        self.timeseries = timeseries
        self.spec_method = spec_method
        if spec_args is not None:
            if 'freq' in spec_args.keys():
                spec_args['freq'] = np.array(spec_args['freq'])
        self.spec_args = spec_args
        self.signif_qs = signif_qs
        self.signif_method = signif_method
        self.plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        if beta_est_res is None:
            self.beta_est_res = beta_est_res
        else:
            self.beta_est_res = np.array(beta_est_res)
        if period_unit is not None:
            self.period_unit = period_unit
        elif timeseries is not None:
            self.period_unit = infer_period_unit_from_time_unit(timeseries.time_unit)
        else:
            self.period_unit = None

    def copy(self):
        '''Copy object
        '''
        return deepcopy(self)

    def __str__(self):
        table = {
            'Frequency': self.frequency,
            'Amplitude': self.amplitude,
        }

        msg = print(tabulate(table, headers='keys'))
        return f'Length: {np.size(self.frequency)}'

    def signif_test(self, method='ar1sim', number=None, seed=None, qs=[0.95],
                    settings=None, scalogram = None):
        '''


        Parameters
        ----------

        number : int, optional

            Number of surrogate series to generate for significance testing. The default is None.

        method : str; {'ar1asym','ar1sim'}

            Method to generate surrogates. AR1sim uses simulated timeseries with similar persistence. AR1asymp represents the closed form solution. The default is AR1sim

        seed : int, optional

            Option to set the seed for reproducibility. The default is None.

        qs : list, optional

            Significance levels to return. The default is [0.95].

        settings : dict, optional

            Parameters for the specific significance test. The default is None. Note that the default value for the asymptotic solution is `time-average`

        scalogram : pyleoclim.Scalogram object, optional

            Scalogram containing signif_scals exported during significance testing of scalogram.
            If number is None and signif_scals are present, will use length of scalogram list as number of significance tests

        Returns
        -------

        new : pyleoclim.PSD

            New PSD object with appropriate significance test

        Examples
        --------
        
        Compute the spectrum of the Southern Oscillation Index and assess significance against an AR(1) benchmark:

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pandas as pd
            csv = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/soi_data.csv',skiprows = 1)
            soi = pyleo.Series(time = csv['Year'],value = csv['Value'], time_name = 'Years', time_unit = 'AD')
            psd = soi.standardize().spectral('mtm',settings={'NW':2})
            psd_sim = psd.signif_test(number=20)
            @savefig psd_sim.png
            fig, ax = psd_sim.plot()
            pyleo.closefig(fig)
            
        By default, this method uses 200 Monte Carlo simulations of an AR(1) process. 
        For a smoother benchmark, up the number of simulations. 
        Also, you may obtain and visualize several quantiles at once, e.g. 90% and 95%:
        
        .. ipython:: python
            :okwarning:
            :okexcept:
                
            psd_1000 = psd.signif_test(number=100, qs=[0.90, 0.95])
            @savefig psd_1000.png
            fig, ax = psd_1000.plot()
            pyleo.closefig(fig)
        
        Another option is to use a closed-form, asymptotic solution for the AR(1) spectrum:
            
        .. ipython:: python
            :okwarning:
            :okexcept:
            
            psd_asym = psd.signif_test(method='ar1asym',qs=[0.90, 0.95])
            @savefig psd_asym.png
            fig, ax = psd_asym.plot()
            pyleo.closefig(fig)
            
        If significance tests from a comparable scalogram have been saved, they can be passed here to speed up the generation of noise realizations for significance testing.
        Setting export_scal to True saves the noise realizations generated during significance testing for future use:
            
        .. ipython:: python
            :okwarning:
            :okexcept:
                
            scalogram = soi.standardize().wavelet().signif_test(number=20, export_scal=True)

        The psd can be calculated by using the previously generated scalogram
        
        .. ipython:: python
            :okwarning:
            :okexcept:
                
            psd_scal = soi.standardize().spectral(scalogram=scalogram)

        The same scalogram can then be passed to do significance testing. 
        Pyleoclim will dig through the scalogram object to find the saved 
        noise realizations and reuse them flexibly.
        
        .. ipython:: python
            :okwarning:
            :okexcept:
                
            @savefig psd_scal.png
            fig, ax = psd.signif_test(scalogram=scalogram).plot()
            pyleo.closefig(fig)

        See also
        --------

        pyleoclim.utils.wavelet.tc_wave_signif : asymptotic significance calculation

        pyleoclim.core.psds.MultiplePSD : Object storing several PSDs from different Series or ensemble members in an age model

        pyleoclim.core.scalograms.Scalogram :  Scalogram object

        pyleoclim.core.series.Series.surrogates : Generate surrogates with increasing time axis

        pyleoclim.core.series.Series.spectral : Performs spectral analysis on Pyleoclim Series

        pyleoclim.core.series.Series.wavelet : Performs wavelet analysis on Pyleoclim Series

        '''

        if self.spec_method == 'wwz' and method == 'ar1asym':
            raise ValueError('Asymptotic solution is not supported for the wwz method')

        if self.spec_method == 'lomb_scargle' and method == 'ar1asym':
            raise ValueError('Asymptotic solution is not supported for the Lomb-Scargle method')

        if method not in ['ar1sim', 'ar1asym']:
                raise ValueError("The available methods are 'ar1sim' and 'ar1asym'")

        if method == 'ar1sim':
            signif_scals = None
            if scalogram:
                try:
                    signif_scals = scalogram.signif_scals
                except:
                    return ValueError('Could not find signif_scals in passed object, make sure this is a scalogram with signif_scals that were saved during significance testing')


            if number is None and signif_scals:
                number = len(signif_scals.scalogram_list)
            elif number is None and signif_scals is None:
                number = 200
            elif number == 0:
                return self

            new = self.copy()
            surr = self.timeseries.surrogates(
                number=number, seed=seed, method=method, settings=settings
            )

            if signif_scals:
                surr_psd = surr.spectral(
                    method=self.spec_method, settings=self.spec_args, scalogram_list=signif_scals
                )
            else:
                surr_psd = surr.spectral(method=self.spec_method, settings=self.spec_args)
            new.signif_qs = surr_psd.quantiles(qs=qs)
            new.signif_method = method

        elif method == 'ar1asym':
            std = self.timeseries.stats()['std'] # assess standard deviation 
            if np.abs(std-1) > 0.1: 
                warnings.warn("Asymptoics are only defined for a standard deviation of unity. Please apply to a standardized series only")
            
            new=self.copy()

            if type(qs) is not list:
                raise TypeError('qs should be a list')

            settings = {'sigtest':'time-average'} if settings is None else settings.copy()

            if self.spec_method=='cwt':
                if 'dof' not in settings.keys():
                    dof = len(self.timeseries.value) - self.spec_args['scale']
                    settings.update({'dof':dof})
                signif_levels=waveutils.tc_wave_signif(self.timeseries.value,
                                                       self.timeseries.time,
                                                       self.spec_args['scale'],
                                                       self.spec_args['mother'],
                                                       self.spec_args['param'],
                                                       qs=qs, **settings)
            else:
                # hard code Mortlet values to obtain the spectrum
                param = 6
                fourier_factor = 4 * np.pi / (param + np.sqrt(2 + param**2))
                scale = 1/(fourier_factor*self.frequency)
                if 'dof' not in settings.keys():
                    dof = len(self.timeseries.value) - scale
                    settings.update({'dof':dof})
                signif_levels=waveutils.tc_wave_signif(self.timeseries.value,
                                                       self.timeseries.time,
                                                       scale,
                                                       'MORLET',
                                                       param,
                                                       qs=qs, **settings)

            # get it back into the object
            new.signif_method = method

            ms_base = []
            for idx, item in enumerate(signif_levels):
                label = str(int(qs[idx]*100))+'%'
                s = PSD(frequency=self.frequency, amplitude = item, label=label)
                ms_base.append(s)
                new.signif_qs = MultiplePSD(ms_base)

        return new

    def beta_est(self, fmin=None, fmax=None, logf_binning_step='max', verbose=False):
        ''' Estimate the scaling exponent (beta) of the PSD 

        For a power law S(f) ~ f^beta in log-log space, beta is simply the slope.
        
        Parameters
        ----------

        fmin : float, optional
            the minimum frequency edge for beta estimation; the default is the minimum of the frequency vector of the PSD obj

        fmax : float, optional
            the maximum frequency edge for beta estimation; the default is the maximum of the frequency vector of the PSD obj

        logf_binning_step : str, {'max', 'first'}
            if 'max', then the maximum spacing of log(f) will be used as the binning step
            if 'first', then the 1st spacing of log(f) will be used as the binning step

        verbose : bool; {True, False}
            If True, will print warning messages if there is any

        Returns
        -------

        new : pyleoclim.PSD
            New PSD object with the estimated scaling slope information, which is stored as a dictionary that includes:
            - beta: the scaling factor
            - std_err: the one standard deviation error of the scaling factor
            - f_binned: the binned frequency series, used as X for linear regression
            - psd_binned: the binned PSD series, used as Y for linear regression
            - Y_reg: the predicted Y from linear regression, used with f_binned for the slope curve plotting

        Examples
        --------

        Generate fractal noise and verify that its scaling exponent is close to unity

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            t, v = pyleo.utils.tsmodel.gen_ts(model='colored_noise')
            ts = pyleo.Series(time=t, value= v, label = 'fractal noise, unit slope')
            psd = ts.detrend().spectral(method='cwt')

            # estimate the scaling slope
            psd_beta = psd.beta_est(fmin=1/50, fmax=1/2)

            @savefig color_noise_beta.png
            fig, ax = psd_beta.plot(color='tab:blue',beta_kwargs={'color':'tab:red','linewidth':2})
            pyleo.closefig(fig)

        See also
        --------
        pyleoclim.core.series.Series.spectral : spectral analysis

        pyleoclim.utils.spectral.beta_estimation : Estimate the scaling exponent of a power spectral density
        
        pyleoclim.core.psds.PSD.plot : plotting method for PSD objects

        '''
        if fmin is None:
            fmin = np.min(self.frequency)

        if fmax is None:
            fmax = np.max(self.frequency)

        res = specutils.beta_estimation(self.amplitude, self.frequency, fmin=fmin, fmax=fmax, logf_binning_step=logf_binning_step, verbose=verbose)
        res_dict = {
            'beta': res.beta,
            'std_err': res.std_err,
            'f_binned': res.f_binned,
            'psd_binned': res.psd_binned,
            'Y_reg': res.Y_reg,
        }
        new = self.copy()
        new.beta_est_res = res_dict
        return new

    def plot(self, in_loglog=True, in_period=True, label=None, xlabel=None, ylabel='PSD', title=None,
             marker=None, markersize=None, color=None, linestyle=None, linewidth=None, transpose=False,
             xlim=None, ylim=None, figsize=[10, 4], savefig_settings=None, ax=None,
             legend=True, lgd_kwargs=None, xticks=None, yticks=None, alpha=None, zorder=None,
             plot_kwargs=None, signif_clr='red', signif_linestyles=['--', '-.', ':'], signif_linewidth=1,
             plot_beta=True, beta_kwargs=None):

        '''Plots the PSD estimates and signif level if included


        Parameters
        ----------
        in_loglog : bool; {True, False}, optional

            Plot on loglog axis. The default is True.

        in_period : bool; {True, False}, optional

            Plot the x-axis as periodicity rather than frequency. The default is True.

        label : str, optional

            label for the series. The default is None.

        xlabel : str, optional

            Label for the x-axis. The default is None. Will guess based on Series

        ylabel : str, optional

            Label for the y-axis. The default is 'PSD'.

        title : str, optional

            Plot title. The default is None.

        marker : str, optional

            marker to use. The default is None.

        markersize : int, optional

            size of the marker. The default is None.

        color : str, optional

            Line color. The default is None.

        linestyle : str, optional

            linestyle. The default is None.

        linewidth : float, optional

            Width of the line. The default is None.

        transpose : bool; {True, False}, optional

            Plot periodicity on y-. The default is False.

        xlim : list, optional

            x-axis limits. The default is None.

        ylim : list, optional

            y-axis limits. The default is None.

        figsize : list, optional

            Figure size. The default is [10, 4].

        savefig_settings : dict, optional

            save settings options. The default is None.
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        ax : ax, optional

            The matplotlib.Axes object onto which to return the plot.
            The default is None.

        legend : bool; {True, False}, optional

            whether to plot the legend. The default is True.

        lgd_kwargs : dict, optional

            Arguments for the legend. The default is None.

        xticks : list, optional

            xticks to use. The default is None.

        yticks : list, optional

            yticks to use. The default is None.

        alpha : float, optional

            Transparency setting. The default is None.

        zorder : int, optional

            Order for the plot. The default is None.

        plot_kwargs : dict, optional

            Other plotting argument. The default is None.

        signif_clr : str, optional

            Color for the significance line. The default is 'red'.

        signif_linestyles : list of str, optional

            Linestyles for significance. The default is ['--', '-.', ':'].

        signif_linewidth : float, optional

            width of the significance line. The default is 1.

        plot_beta : bool; {True, False}, optional

            If True and self.beta_est_res is not None, then the scaling slope line will be plotted

        beta_kwargs : dict, optional

            The visualization keyword arguments for the scaling slope

        Returns
        -------
        fig, ax

        Examples
        --------

        Generate fractal noise, assess significance against an AR(1) benchmark, and plot:

        .. ipython:: python
            :okwarning:
            :okexcept:

            import matplotlib.pyplot as plt
            
            t, v = pyleo.utils.tsmodel.gen_ts(model='colored_noise')
            ts = pyleo.Series(time = t, value = v, label = 'fractal noise')
            tsn = ts.standardize()
    
            psd_sim = tsn.spectral(method='mtm').signif_test(number=20)
            @savefig mtm_sim.png
            psd_sim.plot()
            pyleo.closefig(fig)
            
        If you add the estimate of the scaling exponent, the line of best fit 
        will be added to the plot, and the estimated exponent to its legend. For instance:
            
        .. ipython:: python
            :okwarning:
            :okexcept:    
            
            psd_beta = psd_sim.beta_est(fmin=1/100, fmax=1/2)

            @savefig mtm_sig_beta.png
            fig, ax = psd_beta.plot()
            pyleo.closefig(fig)

        See also
        --------

        pyleoclim.core.series.Series.spectral : spectral analysis
        
        pyleoclim.core.psds.PSD.signif_test : significance testing for PSD objects
        
        pyleoclim.core.psds.PSD.beta_est : scaling exponent estimation for PSD objects

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = self.plot_kwargs if plot_kwargs is None else plot_kwargs.copy()
        beta_kwargs = {} if beta_kwargs is None else beta_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if label is None:
            if plot_beta and self.beta_est_res is not None:
                label = fr'{self.label} ($\hat \beta=${self.beta_est_res["beta"]:.2f}$\pm${self.beta_est_res["std_err"]:.2f})'
            else:
                label = self.label

        if label is not None:
            plot_kwargs.update({'label': label})

        if marker is not None:
            plot_kwargs.update({'marker': marker})

        if markersize is not None:
            plot_kwargs.update({'markersize': markersize})

        if color is not None:
            plot_kwargs.update({'color': color})

        if linestyle is not None:
            plot_kwargs.update({'linestyle': linestyle})

        if linewidth is not None:
            plot_kwargs.update({'linewidth': linewidth})

        if alpha is not None:
            plot_kwargs.update({'alpha': alpha})

        if zorder is not None:
            plot_kwargs.update({'zorder': zorder})

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if in_period:
            idx = np.argwhere(self.frequency==0)
            x_axis = 1/np.delete(self.frequency, idx)
            y_axis = np.delete(self.amplitude, idx)
            if xlabel is None:
                xlabel = f'Period [{self.period_unit}]' if self.period_unit is not None else 'Period'

            if xticks is None:
                xticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])
                mask = (xticks_default >= np.nanmin(x_axis)) & (xticks_default <= np.nanmax(x_axis))
                xticks = xticks_default[mask]

            if xlim is None:
                xlim = [np.max(xticks), np.min(xticks)]

        else:
            idx = np.argwhere(self.frequency==0)
            x_axis = np.delete(self.frequency, idx)
            y_axis = np.delete(self.amplitude, idx)
            if xlabel is None:
                xlabel = f'Frequency [1/{self.period_unit}]' if self.period_unit is not None else 'Frequency'

            if xlim is None:
                xlim = ax.get_xlim()
                xlim = [np.min(xlim), np.max(xlim)]

        if transpose:
            x_axis, y_axis = y_axis, x_axis
            xlim, ylim = ylim, xlim
            xticks, yticks = yticks, xticks
            xlabel, ylabel = ylabel, xlabel
            ax.set_ylim(ylim[::-1])
        else:
            ax.set_xlim(xlim)

        ax.plot(x_axis, y_axis, **plot_kwargs)


        # plot significance levels
        if self.signif_qs is not None:
            signif_method_label = {
                'ar1sim': 'AR(1) simulations',
                'ar1asym': 'AR(1) asymptotic solution'
            }
            nqs = np.size(self.signif_qs.psd_list)

            for i, q in enumerate(self.signif_qs.psd_list):
                idx = np.argwhere(q.frequency==0)
                signif_x_axis = 1/np.delete(q.frequency, idx) if in_period else np.delete(q.frequency, idx)
                signif_y_axis = np.delete(q.amplitude, idx)
                if transpose:
                    signif_x_axis, signif_y_axis = signif_y_axis, signif_x_axis

                ax.plot(
                    signif_x_axis, signif_y_axis,
                    label=f'{signif_method_label[self.signif_method]}, {q.label} threshold',
                    color=signif_clr,
                    linestyle=signif_linestyles[i%3],
                    linewidth=signif_linewidth,
                )

        if in_loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')

        if xticks is not None:
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

        if yticks is not None:
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if plot_beta and self.beta_est_res is not None:
            plot_beta_kwargs = {
                'linestyle': '--',
                'color': 'k',
                'linewidth': 1,
                'zorder': 99,
            }
            plot_beta_kwargs.update(beta_kwargs)
            beta_x_axis = 1/self.beta_est_res['f_binned']
            beta_y_axis = self.beta_est_res['Y_reg']
            if transpose:
                beta_x_axis, beta_y_axis = beta_y_axis, beta_x_axis
            ax.plot(beta_x_axis, beta_y_axis , **plot_beta_kwargs)

        if legend:
            lgd_args = {'frameon': False}
            lgd_args.update(lgd_kwargs)
            ax.legend(**lgd_args)

        if title is not None:
            ax.set_title(title)

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax
        

import seaborn as sns
import matplotlib as mpl
from scipy.stats.mstats import mquantiles

class MultiplePSD:
    '''MultiplePSD objects store several PSDs from different Series or ensemble members from 
    a posterior distribution (e.g. age model, Bayesian climate reconstruction, etc). 
    This is used extensively for Monte Carlo significance tests. 
    '''
    def __init__(self, psd_list, beta_est_res=None):
        ''' Object for multiple PSD.

        This object stores several PSDs from different Series or ensemble members in an age model.         
       
        Parameters
        ----------
        
        beta_est_res : numpy.array
        
            Results of the beta estimation calculation
        
        See also
        --------
        
        pyleoclim.core.psds.PSD.beta_est : Calculates the scaling exponent (i.e., the slope in a log-log plot) of the spectrum (beta)

        '''
        self.psd_list = psd_list
        if beta_est_res is None:
            self.beta_est_res = beta_est_res
        else:
            self.beta_est_res = np.array(beta_est_res)

    def copy(self):
        '''Copy object
        '''
        return deepcopy(self)

    def quantiles(self, qs=[0.05, 0.5, 0.95], lw=[0.5, 1.5, 0.5]):
        
        '''Calculate the quantiles of the significance testing

        Parameters
        ----------
        
        qs : list, optional
        
            List of quantiles to consider for the calculation. The default is [0.05, 0.5, 0.95].
            
        lw : list, optional
        
            Linewidth to use for plotting each level. Should be the same length as qs. The default is [0.5, 1.5, 0.5].

        Raises
        ------
        
        ValueError
        
            Frequency axis not consistent across the PSD list!

        Returns
        -------
        
        psds : pyleoclim.core.psds.MultiplePSD

        '''
        if self.psd_list[0].timeseries is not None:
            period_unit = self.psd_list[0].timeseries.time_unit

        freq = np.copy(self.psd_list[0].frequency)
        amps = []
        for psd in self.psd_list:
            if not np.array_equal(psd.frequency, freq):
                raise ValueError('Frequency axis not consistent across the PSD list!')

            amps.append(psd.amplitude)

        amps = np.array(amps)
        amp_qs = mquantiles(amps, qs, axis=0)

        psd_list = []
        for i, amp in enumerate(amp_qs):
            psd_tmp = PSD(frequency=freq, amplitude=amp, label=f'{qs[i]*100:g}%', plot_kwargs={'color': 'gray', 'linewidth': lw[i]}, period_unit=period_unit)
            psd_list.append(psd_tmp)

        psds = MultiplePSD(psd_list=psd_list)
        return psds

    def beta_est(self, fmin=None, fmax=None, logf_binning_step='max', verbose=False):
        
        ''' Estimate the scaling exponent of each constituent PSD 
        
        This function calculates the scaling exponent (beta) for each of the PSDs stored in the object. 
        The scaling exponent represents the slope of the spectrum in log-log space. 

        Parameters
        ----------

        fmin : float
        
            the minimum frequency edge for beta estimation; the default is the minimum of the frequency vector of the PSD object

        fmax : float
        
            the maximum frequency edge for beta estimation; the default is the maximum of the frequency vector of the PSD object

        logf_binning_step : str; {'max', 'first'}
        
            if 'max', then the maximum spacing of log(f) will be used as the binning step.
            if 'first', then the 1st spacing of log(f) will be used as the binning step.

        verbose : bool
        
            If True, will print warning messages if there is any

        Returns
        -------

        new : pyleoclim.MultiplePSD
        
            New MultiplePSD object with the estimated scaling slope information, which is stored as a dictionary that includes:
            - beta: the scaling factor
            - std_err: the one standard deviation error of the scaling factor
            - f_binned: the binned frequency series, used as X for linear regression
            - psd_binned: the binned PSD series, used as Y for linear regression
            - Y_reg: the predicted Y from linear regression, used with f_binned for the slope curve plotting

        See also
        --------

        pyleoclim.core.psds.PSD.beta_est : scaling exponent estimation for a single PSD object

        '''

        res_dict = {}
        res_dict['beta'] = []
        res_dict['std_err'] = []
        res_dict['f_binned'] = []
        res_dict['psd_binned'] = []
        res_dict['Y_reg'] = []
        psd_beta_list = []
        for psd_obj in self.psd_list:
            psd_beta = psd_obj.beta_est(fmin=fmin, fmax=fmax, logf_binning_step=logf_binning_step, verbose=verbose)
            psd_beta_list.append(psd_beta)
            res = psd_beta.beta_est_res
            for k in res_dict.keys():
                res_dict[k].append(res[k])

        new = self.copy()
        new.beta_est_res = res_dict
        new.psd_list = psd_beta_list
        return new


    def plot(self, figsize=[10, 4], in_loglog=True, in_period=True, xlabel=None, ylabel='Amplitude', title=None,
             xlim=None, ylim=None, savefig_settings=None, ax=None, xticks=None, yticks=None, legend=True,
             colors=None, cmap=None, norm=None, plot_kwargs=None, lgd_kwargs=None):
        '''Plot multiple PSDs on the same plot

        Parameters
        ----------
        
        figsize : list, optional
        
            Figure size. The default is [10, 4].
            
        in_loglog : bool, optional
        
            Whether to plot in loglog. The default is True.
            
        in_period : bool, {True, False} optional
        
            Plots against periods instead of frequencies. The default is True.
            
        xlabel : str, optional
        
            x-axis label. The default is None.
            
        ylabel : str, optional
        
            y-axis label. The default is 'Amplitude'.
            
        title : str, optional
        
            Title for the figure. The default is None.
            
        xlim : list, optional
        
            Limits for the x-axis. The default is None.
            
        ylim : list, optional
        
            limits for the y-axis. The default is None.
            
        colors : a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)
        
            Colors for plotting.
            If None, the plotting will cycle the 'tab10' colormap;
            if only one color is specified, then all curves will be plotted with that single color;
            if a list of colors are specified, then the plotting will cycle that color list.
            
        cmap : str
        
            The colormap to use when "colors" is None.
            
        norm : matplotlib.colors.Normalize like
        
            The nomorlization for the colormap.
            If None, a linear normalization will be used.
            
        savefig_settings : dict, optional
        
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}
            
        ax : matplotlib axis, optional
        
            The matplotlib axis object on which to retrun the figure. The default is None.
            
        xticks : list, optional
        
            x-ticks label. The default is None.
            
        yticks : list, optional
        
            y-ticks label. The default is None.
            
        legend : bool, optional
        
            Whether to plot the legend. The default is True.
            
        plot_kwargs : dictionary, optional
        
            Parameters for plot function. The default is None.
            
        lgd_kwargs : dictionary, optional
        
            Parameters for legend. The default is None.

        Returns
        -------
        fig : matplotlib.pyplot.figure
        
        ax : matplotlib.pyplot.axis
        
        See also
        --------
        
        pyleoclim.core.psds.PSD.plot : plotting method for PSD objects

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for idx, psd in enumerate(self.psd_list):

            tmp_plot_kwargs = {}
            if psd.plot_kwargs is not None:
                tmp_plot_kwargs.update(psd.plot_kwargs)

            tmp_plot_kwargs.update(plot_kwargs)

            # get color for each psd curve
            use_clr = False

            if 'color' not in tmp_plot_kwargs and 'c' not in 'tmp_plot_kwargs':
                use_clr = True

            if 'color' in tmp_plot_kwargs and tmp_plot_kwargs['color'] is None:
                use_clr = True

            if 'c' in tmp_plot_kwargs and tmp_plot_kwargs['c'] is None:
                use_clr = True

            if colors is not None or cmap is not None:
                use_clr = True

            if use_clr:
                # use the color based on the argument 'colors' or 'cmap'
                if colors is None:
                    cmap = 'tab10' if cmap is None else cmap
                    cmap_obj = plt.get_cmap(cmap)
                    if hasattr(cmap_obj, 'colors'):
                        nc = len(cmap_obj.colors)
                    else:
                        nc = len(self.psd_list)

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

                tmp_plot_kwargs.update({'color': clr})

            ax = psd.plot(
                figsize=figsize, in_loglog=in_loglog, in_period=in_period, xlabel=xlabel, ylabel=ylabel,
                title=title, xlim=xlim, ylim=ylim, savefig_settings=savefig_settings, ax=ax,
                xticks=xticks, yticks=yticks, legend=legend, plot_kwargs=tmp_plot_kwargs, lgd_kwargs=lgd_kwargs,
            )

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax

    def plot_envelope(self, figsize=[10, 4], qs=[0.025, 0.5, 0.975],
             in_loglog=True, in_period=True, xlabel=None, ylabel='Amplitude', title=None,
             xlim=None, ylim=None, savefig_settings=None, ax=None, xticks=None, yticks=None, plot_legend=True,
             curve_clr=sns.xkcd_rgb['pale red'], curve_lw=3, shade_clr=sns.xkcd_rgb['pale red'], shade_alpha=0.3, shade_label=None,
             lgd_kwargs=None, members_plot_num=10, members_alpha=0.3, members_lw=1, seed=None):

        '''Plot an envelope statistics for mulitple PSD
        
        This function plots an envelope statistics from multiple PSD. This is especially useful when the PSD are coming from an ensemble of possible solutions (e.g., age ensembles)

        Parameters
        ----------
        
        figsize : list, optional
        
            The figure size. The default is [10, 4].
            
        qs : list, optional
        
            The significance levels to consider. The default is [0.025, 0.5, 0.975].
            
        in_loglog : bool, optional
        
            Plot in log space. The default is True.
            
        in_period : bool, optional
        
            Whether to plot periodicity instead of frequency. The default is True.
            
        xlabel : str, optional
        
            x-axis label. The default is None.
            
        ylabel : str, optional
        
            y-axis label. The default is 'Amplitude'.
            
        title : str, optional
        
            Plot title. The default is None.
            
        xlim : list, optional
        
            x-axis limits. The default is None.
            
        ylim : list, optional
        
            y-axis limits. The default is None.
            
        savefig_settings : dict, optional
        
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.
            
        ax : matplotlib.ax, optional
        
            Matplotlib axis on which to return the plot. The default is None.
            
        xticks : list, optional
        
            xticks label. The default is None.
            
        yticks : list, optional
        
            yticks label. The default is None.
            
        plot_legend : bool, optional
        
            Wether to plot the legend. The default is True.
            
        curve_clr : str, optional
        
            Color of the main PSD. The default is sns.xkcd_rgb['pale red'].
            
        curve_lw : str, optional
        
            Width of the main PSD line. The default is 3.
            
        shade_clr : str, optional
        
            Color of the shaded envelope. The default is sns.xkcd_rgb['pale red'].
            
        shade_alpha : float, optional
        
            Transparency on the envelope. The default is 0.3.
            
        shade_label : str, optional
        
            Label for the envelope. The default is None.
            
        lgd_kwargs : dict, optional
        
            Parameters for the legend. The default is None.
            
        members_plot_num : int, optional
        
            Number of individual members to plot. The default is 10.
            
        members_alpha : float, optional
        
            Transparency of the lines representing the multiple members. The default is 0.3.
            
        members_lw : float, optional
        
            With of the lines representing the multiple members. The default is 1.
            
        seed : int, optional
        
            Set the seed for random number generator. Useful for reproducibility. The default is None.

        Returns
        -------
        fig : matplotlib.pyplot.figure
        
        ax : matplotlib.pyplot.axis
        
        See also
        --------
        
        pyleoclim.core.psds.PSD.plot : plotting method for PSD objects
        
        pyleoclim.core.ensembleseries.EnsembleSeries.plot_envelope : envelope plot for ensembles
        
        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:
            
            import pyleoclim as pyleo
            import numpy as np
            nn = 30 # number of noise realizations
            nt = 500 # timeseries length
            psds = []

            time, signal = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            
            ts = pyleo.Series(time=time, value = signal).standardize()
            noise = np.random.randn(nt,nn)

            for idx in range(nn):  # noise
                ts = pyleo.Series(time=time, value=signal+10*noise[:,idx])
                psd = ts.spectral()
                psds.append(psd)

            mPSD = pyleo.MultiplePSD(psds)
            
            @savefig ens_specplot.png
            fig, ax = mPSD.plot_envelope()
            pyleo.closefig(fig) 

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if members_plot_num > 0:
            if seed is not None:
                np.random.seed(seed)

            npsd = np.size(self.psd_list)
            random_draw_idx = np.random.choice(npsd, members_plot_num)

            for idx in random_draw_idx:
                self.psd_list[idx].plot(
                    in_loglog=in_loglog, in_period=in_period, xlabel=xlabel, ylabel=ylabel,
                    xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks, ax=ax, color='gray', alpha=members_alpha,
                    zorder=99, linewidth=members_lw,
                )
            ax.plot(np.nan, np.nan, color='gray', label=f'example members (n={members_plot_num})')

        psd_qs = self.quantiles(qs=qs)
        psd_qs.psd_list[1].plot(
            in_loglog=in_loglog, in_period=in_period, xlabel=xlabel, ylabel=ylabel, linewidth=curve_lw,
            xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks, ax=ax, color=curve_clr, zorder=100
        )


        if in_period:
            x_axis = 1/psd_qs.psd_list[0].frequency
        else:
            x_axis = psd_qs.psd_list[0].frequency

        if shade_label is None:
            shade_label = f'{psd_qs.psd_list[0].label}-{psd_qs.psd_list[-1].label}'

        ax.fill_between(
            x_axis, psd_qs.psd_list[0].amplitude, psd_qs.psd_list[-1].amplitude,
            color=shade_clr, alpha=shade_alpha, edgecolor=shade_clr, label=shade_label,
        )

        if title is not None:
            ax.set_title(title)

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
    
