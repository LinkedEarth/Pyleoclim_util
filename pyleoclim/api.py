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
    def __init__(self, time, value,
                 time_name=None, time_unit=None,
                 value_name=None, value_unit=None):
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


    def wwz(self, wwz_args={}):
        wwz_res = Spectral.wwz_psd(self.value, self.time, **wwz_args)
        psd = PSD(freq=wwz_res.freqs, amplitude=wwz_res.psd)
        return psd

    def wavelet_ana(self):
        pass

    def corr_with(self, another_series):
        pass


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
