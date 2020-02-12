''' The application interface for the users

@author: fengzhu

Created on Jan 31, 2020
'''
from . import spectral
from . import timeseries
from textwrap import dedent

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

class Series:
    def __init__(self, time, value, time_name=None, time_unit=None, value_name=None, value_unit=None):
        self.time = np.array(time)
        self.value = np.array(value)
        self.time_name = time_name
        self.time_unit = time_unit
        self.value_name = value_name
        self.value_unit = value_unit

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

    def plot(self, figsize=[10, 4], title=None):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.time, self.value)

        xlabel, ylabel = self.make_labels()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        return fig, ax

    def spectral(self, method='wwz', args={}):
        ''' Perform spectral analysis on the timeseries
        '''
        spec_func = {
            'wwz': spectral.wwz_psd,
        }
        spec_res = spec_func[method](self.value, self.time, **args)
        psd = PSD(freq=spec_res.freq, amplitude=spec_res.psd)
        return psd

    def wavelet(self, method='wwz', args={}):
        ''' Perform wavelet analysis on the timeseries
        '''
        wave_func = {
            'wwz': spectral.wwz,
        }
        wave_res = wave_func[method](self.value, self.time, **args)
        scal = Scalogram(freq=wave_res.freq, time=wave_res.time, amplitude=wave_res.amplitude, coi=wave_res.coi)

        return scal
    def corr_with(self, target_series):
        ''' Perform correlation analysis with the target timeseries
        '''
        pass

    def wavelet_coherence(self, target_series, method='wwz', args={}):
        ''' Perform wavelet coherence analysis with the target timeseries
        '''
        xwc_func = {
            'wwz': spectral.xwc,
        }
        xwc_res = xwc_func[method](self.value, self.time, target_series.value, target_series.time, **args)

        coh = Coherence(freq=xwc_res.freq, time=xwc_res.time, coherence=xwc_res.xw_coherence, phase=xwc_res.xw_phase, coi=xwc_res.coi)

        return coh


class PSD:
    def __init__(self, freq, amplitude):
        self.freq = np.array(freq)
        self.amplitude = np.array(amplitude)
    def __str__(self):
        table = {
            'Frequency': self.freq,
            'Amplitude': self.amplitude,
        }

        msg = print(tabulate(table, headers='keys'))
        return f'Length: {np.size(self.freq)}'

    def plot(self, in_loglog=True, in_period=True, xlabel=None, ylabel='Amplitude',
             xlim=None, ylim=None, figsize=[10, 4]):

        fig, ax = plt.subplots(figsize=figsize)

        if in_period:
            x_axis = 1/self.freq
            if xlabel is None:
                xlabel = 'Period'
        else:
            x_axis = self.freq
            if xlabel is None:
                xlabel = 'Frequency'

        if in_loglog:
            ax.loglog(x_axis, self.amplitude)
        else:
            ax.plot(x_axis, self.amplitude)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if in_period:
            if xlim is None:
                xlim = ax.get_xlim()
                xlim = [np.max(xlim), np.min(xlim)]

            ax.set_xlim(xlim)

        return fig, ax

class Scalogram:
    def __init__(self, freq, time, amplitude, coi):
        self.freq = np.array(freq)
        self.time = np.array(time)
        self.amplitude = np.array(amplitude)
        self.coi = np.array(coi)

    def __str__(self):
        table = {
            'Frequency': self.freq,
            'Time': self.time,
            'Amplitude': self.amplitude,
        }

        msg = print(tabulate(table, headers='keys'))
        return f'Dimension: {np.size(self.freq)} x {np.size(self.time)}'

    def plot(self, xlabel='Time', ylabel='Period',
             contourf_args={}, cbar_args={}, figsize=[8, 8], ylim=None, xlim=None):

        if contourf_args == {}:
            contourf_args = {'cmap': 'RdBu_r', 'origin': 'lower', 'levels': 11}

        fig, ax = plt.subplots(figsize=figsize)

        cont = ax.contourf(self.time, 1/self.freq, self.amplitude.T, **contourf_args)
        # plot colorbar
        if cbar_args == {}:
            cbar_args = {'drawedges': False, 'orientation': 'vertical', 'fraction': 0.15, 'pad': 0.05}

        cb = plt.colorbar(cont, **cbar_args)

        # plot cone of influence
        ax.plot(self.time, self.coi, 'k--')
        if ylim is None:
            ylim = [np.min(1/self.freq), np.max(1/self.freq)]

        ax.set_ylim(ylim)

        if xlim is not None:
            ax.set_xlim(xlim)

        ax.fill_between(self.time, self.coi, ylim[1], color='white', alpha=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale('log', nonposy='clip')

        return fig, ax


class Coherence:
    def __init__(self, freq, time, coherence, phase, coi):
        self.freq = np.array(freq)
        self.time = np.array(time)
        self.coherence = np.array(coherence)
        self.coi = np.array(coi)
        self.phase = np.array(phase)

    def plot(self, xlabel='Time', ylabel='Period', figsize=[10, 8], contourf_args={}, ylim=None, xlim=None,
             phase_args={}, cbar_args={}):

        if contourf_args == {}:
            contourf_args = {'cmap': 'RdBu_r', 'origin': 'lower', 'levels': 11}

        fig, ax = plt.subplots(figsize=figsize)

        # plot coherence amplitude
        cont = ax.contourf(self.time, 1/self.freq, self.coherence.T, **contourf_args)

        # plot colorbar
        if cbar_args == {}:
            cbar_args = {'drawedges': False, 'orientation': 'vertical', 'fraction': 0.15, 'pad': 0.05}

        cb = plt.colorbar(cont, **cbar_args)

        # plot cone of influence
        ax.plot(self.time, self.coi, 'k--')

        # set xlim and ylim
        if ylim is None:
            ylim = [np.min(1/self.freq), np.max(1/self.freq)]

        ax.set_ylim(ylim)

        if xlim is not None:
            ax.set_xlim(xlim)

        ax.fill_between(self.time, self.coi, ylim[1], color='white', alpha=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale('log', nonposy='clip')

        # plot phase
        if phase_args == {}:
            phase_args = {'pt': 0.5, 'skip_x': 5, 'skip_y': 5, 'scale': 30, 'width': 0.004}

        pt = phase_args['pt']
        skip_x = phase_args['skip_x']
        skip_y = phase_args['skip_y']
        scale = phase_args['scale']
        width = phase_args['width']

        phase = np.copy(self.phase)
        phase[self.coherence < pt] = np.nan

        X, Y = np.meshgrid(self.time, 1/self.freq)
        U, V = np.cos(phase).T, np.sin(phase).T

        ax.quiver(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x],
                  U[::skip_y, ::skip_x], V[::skip_y, ::skip_x],
                  scale=scale, width=width)

        return fig, ax
