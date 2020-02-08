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

    def plot(self, sns_args={}, figsize=[10, 4]):
        if sns_args == {}:
            sns_args={'style': 'darkgrid', 'font_scale': 1.5}

        sns.set(**sns_args)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.time, self.value)

        xlabel, ylabel = self.make_labels()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax

    def spectral(self, method='wwz', args={}):
        ''' Perform spectral analysis on the timeseries
        '''
        spec_func = {
            'wwz': spectral.wwz_psd,
        }
        spec_res = spec_func[method](self.value, self.time, **args)
        psd = PSD(freq=spec_res.freqs, amplitude=spec_res.psd)
        return psd

    def wavelet(self, method='wwz', args={}):
        ''' Perform wavelet analysis on the timeseries
        '''
        wave_func = {
            'wwz': spectral.wwz,
        }
        wave_res = wave_func[method](self.value, self.time, **args)
        scal = Scalogram(freq=wave_res.freq, time=wave_res.time, amplitude=wave_res.amplitude)

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

        coh = Coherence()

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

    def plot(self, loglog=True, xlabel='Frequency', ylabel='Amplitude', sns_args={}, figsize=[10, 4]):
        if sns_args == {}:
            sns_args={'style': 'darkgrid', 'font_scale': 1.5}

        sns.set(**sns_args)

        fig, ax = plt.subplots(figsize=figsize)
        if loglog:
            ax.loglog(self.freq, self.amplitude)
        else:
            ax.plot(self.freq, self.amplitude)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax

class Scalogram:
    def __init__(self, freq, time, amplitude):
        self.freq = np.array(freq)
        self.time = np.array(time)
        self.amplitude = np.array(amplitude)

    def __str__(self):
        table = {
            'Frequency': self.freq,
            'Time': self.time,
            'Amplitude': self.amplitude,
        }

        msg = print(tabulate(table, headers='keys'))
        return f'Dimension: {np.size(self.freq)} x {np.size(self.time)}'

    def plot(self, xlabel='Time', ylabel='Period',
             sns_args={}, contourf_args={},  figsize=[5, 5]):

        if sns_args == {}:
            sns_args = {'style': 'ticks', 'font_scale': 1.5}

        if contourf_args == {}:
            contourf_args = {'cmap': 'OrRd', 'origin': 'lower', 'levels': 21}

        sns.set(**sns_args)
        fig, ax = plt.subplots(figsize=figsize)

        ax.contourf(self.time, 1/self.freq, self.amplitude.T, **contourf_args)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax


class Coherence:
    def __init__(self, coherence, phase, freqs, tau, AR1_q, coi):
        self.freq = np.array(freq)
        self.time = np.array(time)
        self.amplitude = np.array(amplitude)

    def plot(self, xlabel='Time', ylabel='Frequency', sns_args={}, figsize=[10, 4]):
        if sns_args == {}:
            sns_args={'style': 'darkgrid', 'font_scale': 1.5}

        sns.set(**sns_args)
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax
