'''
The PSD (Power spectral density) class is intended for conveniently manipulating 
the result of spectral methods, including performing significance tests, 
estimating scaling coefficients, and plotting.
'''

from ..utils import plotting, lipdutils 
from ..utils import wavelet as waveutils
from ..utils import spectral as specutils

from ..core.MultiplePSD import MultiplePSD


import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from copy import deepcopy

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
    '''PSD object obtained from spectral analysis.

    See examples in pyleoclim.core.Series.spectral to see how to create and manipulate these objects

    See also
    --------

    pyleoclim.core.Series.spectral : spectral analysis

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
        method : {ar1asym,'ar1sim'}
            Method to generate surrogates. AR1sim uses simulated timeseries with similar persistence. AR1asymp represents the closed form solution. The default is AR1sim
        seed : int, optional
            Option to set the seed for reproducibility. The default is None.
        qs : list, optional
            Singificance levels to return. The default is [0.95].
        settings : dict, optional
            Parameters for the specific significance test. The default is None. Note that the default value for the asymptotic solution is `time-average`
        scalogram : Pyleoclim Scalogram object, optional
            Scalogram containing signif_scals exported during significance testing of scalogram.
            If number is None and signif_scals are present, will use length of scalogram list as number of significance tests

        Returns
        -------
        new : pyleoclim.PSD
            New PSD object with appropriate significance test

        Examples
        --------

        If significance tests from a comparable scalogram have been saved, they can be passed here to speed up the generation of noise realizations for significance testing

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            ts=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/soi_data.csv',skiprows = 1)
            series = pyleo.Series(time = ts['Year'],value = ts['Value'], time_name = 'Years', time_unit = 'AD')

            #Setting export_scal to True saves the noise realizations generated during significance testing for future use
            scalogram = series.wavelet().signif_test(number=2,export_scal=True)

            #The psd can be calculated by using the previously generated scalogram
            psd = series.spectral(scalogram=scalogram)

            #The same scalogram can then be passed to do significance testing. Pyleoclim will dig through the scalogram to find the saved noise realizations and reuse them flexibly.
            fig, ax = psd.signif_test(scalogram=scalogram).plot()


        See also
        --------

        pyleoclim.utils.wavelet.tc_wave_signif : asymptotic significance calculation

        '''

        if self.spec_method == 'wwz' and method == 'ar1asym':
            raise ValueError('Asymptotic solution is not supported for the wwz method')

        if self.spec_method == 'lomb_scargle' and method == 'ar1asym':
            raise ValueError('Asymptotic solution is not supported for the Lomb-Scargle method')

        if method not in ['ar1sim', 'ar1asym']:
                raise ValueError("The available methods are ar1sim'and 'ar1asym'")

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
        ''' Estimate the scaling factor beta of the PSD in a log-log space

        Parameters
        ----------

        fmin : float
            the minimum frequency edge for beta estimation; the default is the minimum of the frequency vector of the PSD obj

        fmax : float
            the maximum frequency edge for beta estimation; the default is the maximum of the frequency vector of the PSD obj

        logf_binning_step : str, {'max', 'first'}
            if 'max', then the maximum spacing of log(f) will be used as the binning step
            if 'first', then the 1st spacing of log(f) will be used as the binning step

        verbose : bool
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

        .. ipython:: python
            :okwarning:
            :okexcept:

            # generate colored noise with default scaling slope 'alpha' equals to 1
            ts = pyleo.gen_ts(model='colored_noise')
            ts.label = 'colored noise'
            psd = ts.spectral()

            # estimate the scaling slope
            psd_beta = psd.beta_est(fmin=1/50, fmax=1/2)

            @savefig color_noise_beta.png
            fig, ax = psd_beta.plot()
            pyleo.closefig(fig)

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
        in_loglog : bool, optional
            Plot on loglog axis. The default is True.
        in_period : bool, optional
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
        transpose : bool, optional
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
            The matplotlib.Axes object onto which to return the plot. The default is None.
        legend : bool, optional
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
        plot_beta : boll, optional
            If True and self.beta_est_res is not None, then the scaling slope line will be plotted
        beta_kwargs : dict, optional
            The visualization keyword arguments for the scaling slope

        Returns
        -------
        fig, ax

        See also
        --------

        pyleoclim.core.Series.spectral : spectral analysis

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = self.plot_kwargs if plot_kwargs is None else plot_kwargs.copy()
        beta_kwargs = {} if beta_kwargs is None else beta_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if label is None:
            if plot_beta and self.beta_est_res is not None:
                label = fr'{self.label} ($\beta=${self.beta_est_res["beta"]:.2f}$\pm${self.beta_est_res["std_err"]:.2f})'
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