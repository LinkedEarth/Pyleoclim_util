''' The application interface for the users

@author: fengzhu

Created on Jan 31, 2020
'''
from ..utils import tsutils, plotting, mapping, lipdutils, tsmodel
from ..utils import wavelet as waveutils
from ..utils import spectral as specutils
from ..utils import correlation as corrutils
from ..utils import causality as causalutils

#from textwrap import dedent

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import namedtuple
from copy import deepcopy

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib import cm

from tqdm import tqdm
from scipy.stats.mstats import mquantiles
import warnings


def dict2namedtuple(d):
    tupletype = namedtuple('tupletype', sorted(d))
    return tupletype(**d)

class Series:
    def __init__(self, time, value, time_name=None, time_unit=None, value_name=None, value_unit=None, label=None):
        self.time = np.array(time)
        self.value = np.array(value)
        self.time_name = time_name
        self.time_unit = time_unit
        self.value_name = value_name
        self.value_unit = value_unit
        self.label = label

    def make_labels(self):
        if self.time_name is not None:
            time_name_str = self.time_name
        else:
            time_name_str = 'time'

        if self.value_name is not None:
            value_name_str = self.value_name
        else:
            value_name_str = 'value'

        if self.value_unit is not None:
            value_header = f'{value_name_str} [{self.value_unit}]'
        else:
            value_header = f'{value_name_str}'

        if self.time_unit is not None:
            time_header = f'{time_name_str} [{self.time_unit}]'
        else:
            time_header = f'{time_name_str}'

        return time_header, value_header

    def __str__(self):
        time_label, value_label = self.make_labels()

        table = {
            time_label: self.time,
            value_label: self.value,
        }

        msg = print(tabulate(table, headers='keys'))
        return f'Length: {np.size(self.time)}'

    def plot(self, figsize=[10, 4],
             marker=None, markersize=None, color=None,
             linestyle=None, linewidth=None, xlim=None, ylim=None,
             label=None, xlabel=None, ylabel=None, title=None, zorder=None,
             legend=True, plot_kwargs=None, lgd_kwargs=None, alpha=None,
             savefig_settings=None, ax=None, mute=False):
        ''' Plot the timeseries

        Args
        ----

        figsize : list
            a list of two integers indicating the figure size

        marker : str
            e.g., 'o' for dots
            See [matplotlib.markers](https://matplotlib.org/3.1.3/api/markers_api.html) for details

        markersize : float
            the size of the marker

        linestyle : str
            e.g., '--' for dashed line
            See [matplotlib.linestyles](https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html) for details

        linewidth : float
            the width of the line

        label : str
            the label for the line

        xlabel : str
            the label for the x-axis

        ylabel : str
            the label for the y-axis

        title : str
            the title for the figure

        legend : {True, False}
            plot legend or not

        plot_kwargs : dict
            the dictionary of keyword arguments for ax.plot()
            See [matplotlib.pyplot.plot](https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.plot.html) for details

        lgd_kwargs : dict
            the dictionary of keyword arguments for ax.legend()
            See [matplotlib.pyplot.legend](https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.legend.html) for details

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        ax : matplotlib.axis, optional
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        Returns
        -------

        fig : matplotlib.figure
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html) for details.

        ax : matplotlib.axis
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        Notes
        -----

        When `ax` is passed, the return will be `ax` only; otherwise, both `fig` and `ax` will be returned.

        '''
        # generate default axis labels
        time_label, value_label = self.make_labels()

        if xlabel is None:
            xlabel = time_label

        if ylabel is None:
            ylabel = value_label

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()

        if label is None:
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

        res = plotting.plot_xy(
            self.time, self.value,
            figsize=figsize, xlabel=xlabel, ylabel=ylabel,
            title=title, savefig_settings=savefig_settings,
            ax=ax, legend=legend, xlim=xlim, ylim=ylim,
            plot_kwargs=plot_kwargs, lgd_kwargs=lgd_kwargs,
            mute=mute,
        )

        return res

    def distplot(self, figsize=[10, 4], title=None, savefig_settings={}, ax=None, ylabel='KDE', mute=False, **plot_kwargs):
        ''' Plot the distribution of the timeseries values

        Args
        ----

        figsize : list
            a list of two integers indicating the figure size

        title : str
            the title for the figure

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax = sns.distplot(self.value, ax=ax, **plot_kwargs)

        time_label, value_label = self.make_labels()

        ax.set_xlabel(value_label)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def copy(self):
        return deepcopy(self)

    def clean(self):
        ''' Clean up the timeseries by removing NaNs and sort with increasing time points
        '''
        new = self.copy()
        v_mod, t_mod = tsutils.clean_ts(self.value, self.time)
        new.time = t_mod
        new.value = v_mod
        return new

    def gaussianize(self):
        new = self.copy()
        v_mod = tsutils.gaussianize(self.value)
        new.value = v_mod
        return new

    def standardize(self):
        new = self.copy()
        v_mod = tsutils.standardize(self.value)[0]
        new.value = v_mod
        return new

    def detrend(self, method='emd', **kwargs):
        new = self.copy()
        v_mod = tsutils.detrend(self.value, x=self.time, method=method, **kwargs)
        new.value = v_mod
        return new

    def spectral(self, method='wwz', settings=None, label=None, verbose=False):
        ''' Perform spectral analysis on the timeseries
        '''
        if not verbose:
            warnings.simplefilter('ignore')

        settings = {} if settings is None else settings.copy()
        spec_func = {
            'wwz': specutils.wwz_psd,
            'mtm': specutils.mtm,
            'lomb_scargle': specutils.lomb_scargle,
            'welch': specutils.welch,
            'periodogram': specutils.periodogram
        }
        args = {}
        args['wwz'] = {}
        args['mtm'] = {}
        args['lomb_scargle'] = {}
        args['welch'] = {}
        args['periodogram'] = {}
        args[method].update(settings)
        spec_res = spec_func[method](self.value, self.time, **args[method])
        if type(spec_res) is dict:
            spec_res = dict2namedtuple(spec_res)

        if label is None:
            label = self.label

        psd = PSD(
            frequency=spec_res.freq,
            amplitude=spec_res.psd,
            label=label,
            timeseries=self,
            spec_method=method,
            spec_args=args[method]
        )

        return psd

    def wavelet(self, method='wwz', nv=12, settings=None, verbose=False):
        ''' Perform wavelet analysis on the timeseries
        '''
        if not verbose:
            warnings.simplefilter('ignore')

        settings = {} if settings is None else settings.copy()
        wave_func = {
            'wwz': waveutils.wwz,
        }
        # generate default freq
        s0 = 2*np.median(np.diff(self.time))
        a0 = 2**(1/nv)
        noct = np.floor(np.log2(np.size(self.time)))-1
        scale = s0*a0**(np.arange(noct*nv+1))
        freq = 1/scale[::-1]

        args = {}
        args['wwz'] = {'tau': self.time, 'freq': freq}
        args[method].update(settings)
        wave_res = wave_func[method](self.value, self.time, **args[method])
        scal = Scalogram(
            frequency=wave_res.freq,
            time=wave_res.time,
            amplitude=wave_res.amplitude,
            coi=wave_res.coi,
            label=self.label,
            timeseries=self,
            wave_method=method,
            wave_args=args[method],
        )

        return scal

    def wavelet_coherence(self, target_series, nv=12, method='wwz', settings=None, verbose=False):
        ''' Perform wavelet coherence analysis with the target timeseries
        '''
        if not verbose:
            warnings.simplefilter('ignore')

        settings = {} if settings is None else settings.copy()
        xwc_func = {
            'wwz': waveutils.xwc,
        }
        # generate default freq
        s0 = 2*np.median(np.diff(self.time))
        a0 = 2**(1/nv)
        noct = np.floor(np.log2(np.size(self.time)))-1
        scale = s0*a0**(np.arange(noct*nv+1))
        freq = 1/scale[::-1]

        t1 = np.copy(self.time)
        t2 = np.copy(target_series.time)
        dt1 = np.median(np.diff(t1))
        dt2 = np.median(np.diff(t2))
        overlap = np.arange(np.max([t1[0], t2[0]]), np.min([t1[-1], t2[-1]]), np.max([dt1, dt2]))

        args = {}
        args['wwz'] = {'tau': overlap, 'freq': freq}
        args[method].update(settings)
        xwc_res = xwc_func[method](self.value, self.time, target_series.value, target_series.time, **args[method])

        coh = Coherence(
            frequency=xwc_res.freq,
            time=xwc_res.time,
            coherence=xwc_res.xw_coherence,
            phase=xwc_res.xw_phase,
            coi=xwc_res.coi,
            timeseries1=self,
            timeseries2=target_series,
        )

        return coh

    def correlation(self, target_series, settings=None):
        ''' Perform correlation analysis with the target timeseries
        '''
        settings = {} if settings is None else settings.copy()
        args = {}
        args.update(settings)
        r, signif, p = corrutils.corr_sig(self.value, target_series.value, **args)
        corr_res = {
            'r': r,
            'signif': signif,
            'pvalue': p,
        }
        return corr_res

    def causality(self, target_series, method='liang', settings=None):
        ''' Perform causality analysis with the target timeseries
        '''
        settings = {} if settings is None else settings.copy()
        args = {}
        args['liang'] = {}
        args[method].update(settings)
        causal_res = causalutils.causality_est(self.value, target_series.value, method=method, **args[method])
        return causal_res

    def surrogates(self, method='ar1', number=1, length=None, seed=None, settings=None):
        settings = {} if settings is None else settings.copy()
        surrogate_func = {
            'ar1': tsmodel.ar1_sim,
        }
        args = {}
        args['ar1'] = {'ts': self.time}
        args[method].update(settings)

        if length is None:
            length = np.size(self.value)

        if seed is not None:
            np.random.seed(seed)

        surr_res = surrogate_func[method](self.value, length, number, **args[method])
        if len(np.shape(surr_res)) == 1:
            surr_res = surr_res[:, np.newaxis]

        s_list = []
        for s in surr_res.T:
            s_tmp = Series(time=self.time, value=s)
            s_list.append(s_tmp)

        surr = MultipleSeries(series_list=s_list, surrogate_method=method, surrogate_args=args[method])

        return surr

class PSD:
    def __init__(self, frequency, amplitude, label=None, timeseries=None,
                 spec_method=None, spec_args=None, signif_qs=None, signif_method=None):
        self.frequency = np.array(frequency)
        self.amplitude = np.array(amplitude)
        self.label = label
        self.timeseries = timeseries
        self.spec_method = spec_method
        self.spec_args = spec_args
        self.signif_qs = signif_qs
        self.signif_method = signif_method

    def copy(self):
        return deepcopy(self)

    def __str__(self):
        table = {
            'Frequency': self.freq,
            'Amplitude': self.amplitude,
        }

        msg = print(tabulate(table, headers='keys'))
        return f'Length: {np.size(self.freq)}'

    def signif_test(self, number=200, method='ar1', seed=None, qs=[0.95],
                    settings=None):
        new = self.copy()
        surr = self.timeseries.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )
        surr_psd = surr.spectral(method=self.spec_method, settings=self.spec_args)
        new.signif_qs = surr_psd.quantiles(qs=qs)
        new.signif_method = method

        return new

    def plot(self, in_loglog=True, in_period=True, label=None, xlabel=None, ylabel='Amplitude', title=None,
             marker=None, markersize=None, color=None, linestyle=None, linewidth=None,
             xlim=None, ylim=None, figsize=[10, 4], savefig_settings=None, ax=None, mute=False,
             plot_legend=True, lgd_kwargs=None, xticks=None, yticks=None, alpha=None, zorder=None,
             plot_kwargs=None, signif_clr='red', signif_linestyles=['--', '-.', ':'], signif_linewidth=1):
        ''' Plot the power sepctral density (PSD)

        Args
        ----

        figsize : list
            a list of two integers indicating the figure size

        title : str
            the title for the figure

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}
        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if label is None:
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
                xlabel = 'Period'

            if xticks is None:
                xticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
                mask = (xticks_default >= np.nanmin(x_axis)) & (xticks_default <= np.nanmax(x_axis))
                xticks = xticks_default[mask]

            if xlim is None:
                xlim = [np.max(xticks), np.min(xticks)]

        else:
            idx = np.argwhere(self.frequency==0)
            x_axis = np.delete(self.frequency, idx)
            y_axis = np.delete(self.amplitude, idx)
            if xlabel is None:
                xlabel = 'Frequency'

            if xlim is None:
                xlim = ax.get_xlim()
                xlim = [np.min(xlim), np.max(xlim)]

        ax.set_xlim(xlim)
        ax.plot(x_axis, y_axis, **plot_kwargs)

        # plot significance levels
        if self.signif_qs is not None:
            signif_method_label = {
                'ar1': 'AR(1)',
            }
            nqs = np.size(self.signif_qs.psd_list)

            for i, q in enumerate(self.signif_qs.psd_list):
                idx = np.argwhere(q.frequency==0)
                signif_x_axis = 1/np.delete(q.frequency, idx) if in_period else np.delete(q.frequency, idx)
                signif_y_axis = np.delete(q.amplitude, idx)
                ax.plot(
                    signif_x_axis, signif_y_axis,
                    label=f'{signif_method_label[self.signif_method]}, {q.label} threshold',
                    color=signif_clr,
                    linestyle=signif_linestyles[i%3],
                    linewidth=signif_linewidth,
                )

        if in_loglog:
            ax.set_xscale('log', nonposx='clip')
            ax.set_yscale('log', nonposy='clip')

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

        if plot_legend:
            lgd_args = {'frameon': False}
            lgd_args.update(lgd_kwargs)
            ax.legend(**lgd_args)

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
            return fig, ax
        else:
            return ax

class Scalogram:
    def __init__(self, frequency, time, amplitude, coi=None, label=None, Neff=3, timeseries=None,
                 wave_method=None, wave_args=None, signif_qs=None, signif_method=None):
        '''
        Args
        ----
            frequency : array
                the frequency axis
            time : array
                the time axis
            amplitude : array
                the amplitude at each (frequency, time) point;
                note the dimension is assumed to be (frequency, time)
        '''
        self.frequency = np.array(frequency)
        self.time = np.array(time)
        self.amplitude = np.array(amplitude)
        if coi is not None:
            self.coi = np.array(coi)
        else:
            self.coi = waveutils.make_coi(self.time, Neff=Neff)
        self.label = label
        self.timeseries = timeseries
        self.wave_method = wave_method
        self.wave_args = wave_args
        self.signif_qs = signif_qs
        self.signif_method = signif_method

    def copy(self):
        return deepcopy(self)

    def __str__(self):
        table = {
            'Frequency': self.frequency,
            'Time': self.time,
            'Amplitude': self.amplitude,
        }

        msg = print(tabulate(table, headers='keys'))
        return f'Dimension: {np.size(self.frequency)} x {np.size(self.time)}'

    def plot(self, in_period=True, xlabel='Time', ylabel=None, title=None,
             ylim=None, xlim=None, yticks=None, figsize=[10, 8], mute=False,
             signif_clr='white', signif_linestyles='-', signif_linewidths=1,
             contourf_style={}, cbar_style={}, savefig_settings={}, ax=None):
        ''' Plot the scalogram from wavelet analysis

        Args
        ----

        figsize : list
            a list of two integers indicating the figure size

        title : str
            the title for the figure

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}
        '''
        contourf_args = {'cmap': 'magma', 'origin': 'lower', 'levels': 11}
        contourf_args.update(contourf_style)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if in_period:
            y_axis = 1/self.frequency
            if ylabel is None:
                ylabel = 'Period'

            if yticks is None:
                yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
                mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
                yticks = yticks_default[mask]
        else:
            y_axis = self.frequency
            if ylabel is None:
                ylabel = 'Frequency'

        cont = ax.contourf(self.time, y_axis, self.amplitude.T, **contourf_args)
        ax.set_yscale('log', nonposy='clip')

        # plot colorbar
        cbar_args = {'drawedges': False, 'orientation': 'vertical', 'fraction': 0.15, 'pad': 0.05}
        cbar_args.update(cbar_style)

        cb = plt.colorbar(cont, **cbar_args)

        # plot cone of influence
        if self.coi is not None:
            ax.plot(self.time, self.coi, 'k--')

        if yticks is not None:
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

        if title is not None:
            ax.set_title(title)

        if ylim is None:
            ylim = [np.min(y_axis), np.min([np.max(y_axis), np.max(self.coi)])]

        ax.fill_between(self.time, self.coi, np.max(self.coi), color='white', alpha=0.5)

        # plot significance levels
        if self.signif_qs is not None:
            signif_method_label = {
                'ar1': 'AR(1)',
            }
            signif_scal = self.signif_qs.scalogram_list[0]
            signif_boundary = self.amplitude.T / signif_scal.amplitude.T
            ax.contour(
                self.time, y_axis, signif_boundary, [-99, 1],
                colors=signif_clr,
                linestyles=signif_linestyles,
                linewidths=signif_linewidths,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        ax.set_ylim(ylim)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def signif_test(self, number=200, method='ar1', seed=None, qs=[0.95],
                    settings=None):
        new = self.copy()
        surr = self.timeseries.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )
        surr_scal = surr.wavelet(method=self.wave_method, settings=self.wave_args)

        if len(qs) > 1:
            raise ValueError('qs should be a list with size 1!')

        new.signif_qs = surr_scal.quantiles(qs=qs)
        new.signif_method = method

        return new


class Coherence:
    def __init__(self, frequency, time, coherence, phase, coi=None, timeseries1=None, timeseries2=None, signif_qs=None, signif_method=None):
        self.frequency = np.array(frequency)
        self.time = np.array(time)
        self.coherence = np.array(coherence)
        if coi is not None:
            self.coi = np.array(coi)
        else:
            self.coi = waveutils.make_coi(self.time, Neff=Neff)
        self.phase = np.array(phase)
        self.timeseries1 = timeseries1
        self.timeseries2 = timeseries2
        self.signif_qs = signif_qs
        self.signif_method = signif_method

    def copy(self):
        return deepcopy(self)

    def plot(self, xlabel='Time', ylabel='Period', title=None, figsize=[10, 8],
             ylim=None, xlim=None, in_period=True, yticks=None, mute=False,
             contourf_style={}, phase_style={}, cbar_style={}, savefig_settings={}, ax=None,
             signif_clr='white', signif_linestyles='-', signif_linewidths=1,
             under_clr='ivory', over_clr='black', bad_clr='dimgray'):
        ''' Plot the wavelet coherence result

        Args
        ----

        figsize : list
            a list of two integers indicating the figure size

        title : str
            the title for the figure

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if in_period:
            y_axis = 1/self.frequency
            if ylabel is None:
                ylabel = 'Period'

            if yticks is None:
                yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
                mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
                yticks = yticks_default[mask]
        else:
            y_axis = self.frequency
            if ylabel is None:
                ylabel = 'Frequency'

        # plot coherence amplitude
        contourf_args = {
            'cmap': 'magma',
            'origin': 'lower',
            'levels': np.linspace(0, 1, 11),
        }
        contourf_args.update(contourf_style)

        cmap = cm.get_cmap(contourf_args['cmap'])
        cmap.set_under(under_clr)
        cmap.set_over(over_clr)
        cmap.set_bad(bad_clr)
        contourf_args['cmap'] = cmap

        cont = ax.contourf(self.time, y_axis, self.coherence.T, **contourf_args)

        # plot significance levels
        if self.signif_qs is not None:
            signif_method_label = {
                'ar1': 'AR(1)',
            }
            signif_coh = self.signif_qs.scalogram_list[0]
            signif_boundary = self.coherence.T / signif_coh.amplitude.T
            ax.contour(
                self.time, y_axis, signif_boundary, [-99, 1],
                colors=signif_clr,
                linestyles=signif_linestyles,
                linewidths=signif_linewidths,
            )

        # plot colorbar
        cbar_args = {
            'drawedges': False,
            'orientation': 'vertical',
            'fraction': 0.15,
            'pad': 0.05,
            'ticks': np.linspace(0, 1, 11)
        }
        cbar_args.update(cbar_style)

        cb = plt.colorbar(cont, **cbar_args)

        # plot cone of influence
        ax.set_yscale('log', nonposy='clip')
        ax.plot(self.time, self.coi, 'k--')

        if ylim is None:
            ylim = [np.min(y_axis), np.min([np.max(y_axis), np.max(self.coi)])]

        ax.fill_between(self.time, self.coi, np.max(self.coi), color='white', alpha=0.5)

        if yticks is not None:
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # plot phase
        dt = np.max([int(np.median(np.diff(self.time))), 1])
        phase_args = {'pt': 0.5, 'skip_x': 10*dt, 'skip_y': 5*dt, 'scale': 30, 'width': 0.004}
        phase_args.update(phase_style)

        pt = phase_args['pt']
        skip_x = phase_args['skip_x']
        skip_y = phase_args['skip_y']
        scale = phase_args['scale']
        width = phase_args['width']

        phase = np.copy(self.phase)

        if self.signif_qs is None:
            phase[self.coherence < pt] = np.nan
        else:
            phase[signif_boundary.T < 1] = np.nan

        X, Y = np.meshgrid(self.time, 1/self.frequency)
        U, V = np.cos(phase).T, np.sin(phase).T

        ax.quiver(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x],
                  U[::skip_y, ::skip_x], V[::skip_y, ::skip_x],
                  scale=scale, width=width, zorder=99)

        ax.set_ylim(ylim)

        if xlim is not None:
            ax.set_xlim(xlim)

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def signif_test(self, number=200, method='ar1', seed=None, qs=[0.95], settings=None):
        new = self.copy()
        surr1 = self.timeseries1.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )
        surr2 = self.timeseries2.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )

        cohs = []
        for i in tqdm(range(number), desc='Performing wavelet coherence on surrogate pairs'):
            coh_tmp = surr1.series_list[i].wavelet_coherence(surr2.series_list[i])
            cohs.append(coh_tmp.coherence)

        cohs = np.array(cohs)

        ne, nf, nt = np.shape(cohs)

        coh_qs = np.ndarray(shape=(np.size(qs), nf, nt))
        for i in range(nf):
            for j in range(nt):
                coh_qs[:,i,j] = mquantiles(cohs[:,i,j], qs)

        scal_list = []
        for i, amp in enumerate(coh_qs):
            scal_tmp = Scalogram(frequency=self.frequency, time=self.time, amplitude=amp, coi=self.coi, label=f'{qs[i]*100:g}%')
            scal_list.append(scal_tmp)

        new.signif_qs = MultipleScalogram(scalogram_list=scal_list)
        new.signif_method = method

        return new

class MultipleSeries:
    def __init__(self, series_list, surrogate_method=None, surrogate_args=None):
        self.series_list = series_list
        self.surrogate_method = surrogate_method
        self.surrogate_args = surrogate_args

    def copy(self):
        return deepcopy(self)

    def spectral(self, method='wwz', settings={}):
        settings = {} if settings is None else settings.copy()

        psd_list = []
        for s in tqdm(self.series_list, desc='Performing spectral analysis on surrogates'):
            psd_tmp = s.spectral(method=method, settings=settings)
            psd_list.append(psd_tmp)

        psds = MultiplePSD(psd_list=psd_list)

        return psds

    def wavelet(self, method='wwz', nv=12, settings={}):
        settings = {} if settings is None else settings.copy()

        scal_list = []
        for s in tqdm(self.series_list, desc='Performing wavelet analysis on surrogates'):
            scal_tmp = s.wavelet(method=method, nv=nv, settings=settings)
            scal_list.append(scal_tmp)

        scals = MultipleScalogram(scalogram_list=scal_list)

        return scals

    def plot(self, figsize=[10, 4],
             marker=None, markersize=None, color=None,
             linestyle=None, linewidth=None,
             label=None, xlabel=None, ylabel=None, title=None,
             legend=True, plot_kwargs=None, lgd_kwargs=None,
             savefig_settings=None, ax=None, mute=False):

        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for s in self.series_list:
            ax = s.plot(
                figsize=figsize, marker=marker, markersize=markersize, color=color, linestyle=linestyle,
                linewidth=linewidth, label=label, xlabel=xlabel, ylabel=ylabel, title=title,
                legend=False, plot_kwargs=plot_kwargs, ax=ax,
            )

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
            return fig, ax
        else:
            return ax

class MultiplePSD:
    def __init__(self, psd_list):
        self.psd_list = psd_list

    def copy(self):
        return deepcopy(self)

    def quantiles(self, qs=[0.05, 0.5, 0.95]):
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
            psd_tmp = PSD(frequency=freq, amplitude=amp, label=f'{qs[i]*100:g}%')
            psd_list.append(psd_tmp)

        psds = MultiplePSD(psd_list=psd_list)
        return psds

    def plot(self, figsize=[10, 4], in_loglog=True, in_period=True, xlabel=None, ylabel='Amplitude', title=None,
             xlim=None, ylim=None, savefig_settings=None, ax=None, xticks=None, yticks=None, plot_legend=True,
             plot_kwargs=None, lgd_kwargs=None, mute=False):

        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for psd in self.psd_list:
            ax = psd.plot(
                figsize=figsize, in_loglog=in_loglog, in_period=in_period, label=psd.label, xlabel=xlabel, ylabel=ylabel,
                title=title, xlim=xlim, ylim=ylim, savefig_settings=savefig_settings, ax=ax,
                xticks=xticks, yticks=yticks, plot_legend=plot_legend, plot_kwargs=plot_kwargs, lgd_kwargs=lgd_kwargs,
            )

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
            return fig, ax
        else:
            return ax

class MultipleScalogram:
    def __init__(self, scalogram_list):
        self.scalogram_list = scalogram_list

    def copy(self):
        return deepcopy(self)

    def quantiles(self, qs=[0.05, 0.5, 0.95]):
        freq = np.copy(self.scalogram_list[0].frequency)
        time = np.copy(self.scalogram_list[0].time)
        coi = np.copy(self.scalogram_list[0].coi)
        amps = []
        for scal in self.scalogram_list:
            if not np.array_equal(scal.frequency, freq):
                raise ValueError('Frequency axis not consistent across the scalogram list!')

            if not np.array_equal(scal.time, time):
                raise ValueError('Time axis not consistent across the scalogram list!')

            amps.append(scal.amplitude)

        amps = np.array(amps)
        ne, nf, nt = np.shape(amps)
        amp_qs = np.ndarray(shape=(np.size(qs), nf, nt))

        for i in range(nf):
            for j in range(nt):
                amp_qs[:,i,j] = mquantiles(amps[:,i,j], qs)

        scal_list = []
        for i, amp in enumerate(amp_qs):
            scal_tmp = Scalogram(frequency=freq, time=time, amplitude=amp, coi=coi, label=f'{qs[i]*100:g}%')
            scal_list.append(scal_tmp)

        scals = MultipleScalogram(scalogram_list=scal_list)
        return scals


class Lipd:
    def __init__(self, lipd_list):
        self.plot_default = {'ice/rock': ['#FFD600','h'],
                'coral': ['#FF8B00','o'],
                'documents':['k','p'],
                'glacier ice':['#86CDFA', 'd'],
                'hybrid': ['#00BEFF','*'],
                'lake sediment': ['#4169E0','s'],
                'marine sediment': ['#8A4513', 's'],
                'sclerosponge' : ['r','o'],
                'speleothem' : ['#FF1492','d'],
                'wood' : ['#32CC32','^'],
                'molluskshells' : ['#FFD600','h'],
                'peat' : ['#2F4F4F','*'],
                'other':['k','o']}
    
    def getLipd(self, usr_path=None):
        """Read Lipd files into a dictionary

        Sets the dictionary as global variable so that it doesn't have to be provided
        as an argument for every function.
    
        Args
        ----
    
        usr_path : str
                  The path to a directory or a single file. (Optional argument)
    
        Returns
        -------
    
        lipd_dict : dict
                   a dictionary containing the LiPD library
    
        """
        global lipd_dict
        lipd_dict = lpd.readLipd(usr_path=usr_path)
        return lipd_dict
        

class LipdSeries:
    def __init__(self, ts):
        self.ts = ts
        self.plot_default = {'ice/rock': ['#FFD600','h'],
                'coral': ['#FF8B00','o'],
                'documents':['k','p'],
                'glacier ice':['#86CDFA', 'd'],
                'hybrid': ['#00BEFF','*'],
                'lake sediment': ['#4169E0','s'],
                'marine sediment': ['#8A4513', 's'],
                'sclerosponge' : ['r','o'],
                'speleothem' : ['#FF1492','d'],
                'wood' : ['#32CC32','^'],
                'molluskshells' : ['#FFD600','h'],
                'peat' : ['#2F4F4F','*'],
                'other':['k','o']}

    def mapone(self, projection='Orthographic', proj_default=True,
               background=True, label='default', borders=False,
               rivers=False, lakes=False, markersize=50, marker="default",
               figsize=[4,4], ax=None, savefig_settings={}):
        """ Create a Map for a single record

        Orthographic projection map of a single record.

        Args
        ----

        timeseries : object
                    a LiPD timeseries object. Will prompt for one if not given
        projection : string
                    the map projection. Available projections:
                    'Robinson', 'PlateCarree', 'AlbertsEqualArea',
                    'AzimuthalEquidistant','EquidistantConic','LambertConformal',
                    'LambertCylindrical','Mercator','Miller','Mollweide','Orthographic' (Default),
                    'Sinusoidal','Stereographic','TransverseMercator','UTM',
                    'InterruptedGoodeHomolosine','RotatedPole','OSGB','EuroPP',
                    'Geostationary','NearsidePerspective','EckertI','EckertII',
                    'EckertIII','EckertIV','EckertV','EckertVI','EqualEarth','Gnomonic',
                    'LambertAzimuthalEqualArea','NorthPolarStereo','OSNI','SouthPolarStereo'
        proj_default : bool
                      If True, uses the standard projection attributes, including centering.
                      Enter new attributes in a dictionary to change them. Lists of attributes
            can be found in the Cartopy documentation:
                https://scitools.org.uk/cartopy/docs/latest/crs/projections.html#eckertiv
        background : bool
                    If True, uses a shaded relief background (only one
                    available in Cartopy)
        label : str
               label for archive marker. Default is to use the name of the
               physical sample. If no archive name is available, default to
               None. None returns no label.
        borders : bool
                 Draws the countries border. Defaults is off (False).
        rivers : bool
                Draws major rivers. Default is off (False).
        lakes : bool
               Draws major lakes. Default is off (False).
        markersize : int
                    The size of the marker.
        marker : str or list
                color and type of marker. Default will use the
                default color palette for archives
        figsize : list
                 the size for the figure

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        Returns
        -------
        The figure

        """

        # Get latitude/longitude

        lat = self.ts['geo_meanLat']
        lon = self.ts['geo_meanLon']

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Make sure it's in the palette
        if marker == 'default':
            archiveType = lipdutils.LipdToOntology(self.ts['archiveType']).lower()
            if archiveType not in self.plot_default.keys():
                archiveType = 'other'
            marker = self.plot_default[archiveType]

        # Get the label
        if label == 'default':
            for i in self.ts.keys():
                if 'physicalSample_name' in i:
                    label = self.ts[i]
                elif 'measuredOn_name' in i:
                    label = self.ts[i]
            if label == 'default':
                label = None
        elif label is None:
            label = None
        else:
            raise TypeError('the argument label should be of type str')

        fig, ax = mapping.mapOne(lat, lon, projection = projection, proj_default = proj_default,
               background = background, label = label, borders = borders, rivers = rivers, lakes = lakes,
               markersize = markersize, marker = marker, figsize = figsize, ax = ax)

        # Save the figure if "path" is specified in savefig_settings
        if 'path' in savefig_settings:
            plotting.savefig(fig, savefig_settings)
        else:
            plooting.showfig(fig)

        return fig
