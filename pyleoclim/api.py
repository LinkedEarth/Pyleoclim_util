''' The application interface for the users

@author: fengzhu

Created on Jan 31, 2020
'''
from . import analysis
from . import visualization
from . import lipdutils

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
        ''' Plot the timeseries
        '''
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.time, self.value)

        xlabel, ylabel = self.make_labels()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        return fig, ax

    def clean(self):
        ''' Clean up the timeseries by removing NaNs and sort with increasing time points
        '''
        y = self.value
        t = self.time
        y_cleaned, t_cleaned = analysis.clean_ts(y, t)
        self.time = t_cleaned
        self.value = y_cleaned

    def spectral(self, method='wwz', args={}):
        ''' Perform spectral analysis on the timeseries
        '''
        spec_func = {
            'wwz': analysis.wwz_psd,
        }
        spec_res = spec_func[method](self.value, self.time, **args)
        psd = PSD(freq=spec_res.freq, amplitude=spec_res.psd)
        return psd

    def wavelet(self, method='wwz', args={}):
        ''' Perform wavelet analysis on the timeseries
        '''
        wave_func = {
            'wwz': analysis.wwz,
        }
        wave_res = wave_func[method](self.value, self.time, **args)
        scal = Scalogram(freq=wave_res.freq, time=wave_res.time, amplitude=wave_res.amplitude, coi=wave_res.coi)

        return scal

    def wavelet_coherence(self, target_series, method='wwz', args={}):
        ''' Perform wavelet coherence analysis with the target timeseries
        '''
        xwc_func = {
            'wwz': analysis.xwc,
        }
        xwc_res = xwc_func[method](self.value, self.time, target_series.value, target_series.time, **args)

        coh = Coherence(freq=xwc_res.freq, time=xwc_res.time, coherence=xwc_res.xw_coherence, phase=xwc_res.xw_phase, coi=xwc_res.coi)

        return coh

    def correlation(self, target_series, args={}):
        ''' Perform correlation analysis with the target timeseries
        '''
        r, signif, p = analysis.corrsig(self.value, target_series.value, **args)
        corr_res = {
            'r': r,
            'signif': signif,
            'pvalue': p,
        }
        return corr_res

    def causality(self, target_series, args={}):
        ''' Perform causality analysis with the target timeseries
        '''
        causal_res = analysis.causality_est(self.value, target_series.value, **args)
        return causal_res


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
        ''' Plot the power sepctral density (PSD)
        '''

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
             contourf_args={}, cbar_args={}, figsize=[10, 8], ylim=None, xlim=None):
        ''' Plot the scalogram from wavelet analysis
        '''
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
        ''' Plot the wavelet coherence result
        '''
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

class Lipd:
    def __init__(self, lipd_list):
        self.lipd_list = lipd_list
    
    #def mapallarchive(self,markersize = 50, projection = 'Robinson',\
                  #proj_default = True, background = True,borders = False,\
                  #rivers = False, lakes = False, figsize = [10,4],\
                  #saveFig = False, dir=None, format='eps'):
        
        
    

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
    
    def mapone(self,projection = 'Orthographic', proj_default = True,\
           background = True, label = 'default', borders = False, \
           rivers = False, lakes = False,markersize = 50, marker = "default",\
           figsize = [4,4], savefig = False, dir = None, format="eps"):
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
        savefig : bool
                 default is to not save the figure
        dir : str
             the full path of the directory in which to save the figure.
             If not provided, creates a default folder called 'figures' in the
             LiPD working directory (lipd.path).
        format : str
                One of the file extensions supported by the active
                backend. Default is "eps". Most backend support png, pdf, ps, eps,
                and svg.
    
        Returns
        -------
        The figure
    
        """
        
        # Get latitude/longitude
    
        lat = self.ts['geo_meanLat']
        lon = self.ts['geo_meanLon']
    
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
            assert type(label) is str, 'the argument label should be of type str'
    
        fig = visualization.mapOne(lat, lon, projection = projection, proj_default = proj_default,\
               background = background, label = label, borders = borders, \
               rivers = rivers, lakes = lakes,\
               markersize = markersize, marker = marker, figsize = figsize, \
               ax = None)
    
        # Save the figure if asked
        if savefig == True:
            lipdutils.saveFigure(self.ts['dataSetName']+'_map', format, dir)
        else:
            plt.show()
    
        return fig
    
        
