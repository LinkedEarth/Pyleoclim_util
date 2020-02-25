''' The application interface for the users

@author: fengzhu

Created on Jan 31, 2020
'''
from ..utils import tsutils, plotting, mapping, lipdutils
from ..utils import wavelet as waveutils
from ..utils import spectral as specutils
from ..utils import correlation as corrutils
from ..utils import causality as causalutils

from textwrap import dedent

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import namedtuple

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib import cm

def dict2namedtuple(d):
    tupletype = namedtuple('tupletype', sorted(d))
    return tupletype(**d)

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

    def plot(self, figsize=[10, 4], title=None, savefig_settings={}, ax=None,
             **plot_args):
        ''' Plot the timeseries

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

        ax.plot(self.time, self.value, **plot_args)

        time_label, value_label = self.make_labels()

        ax.set_xlabel(time_label)
        ax.set_ylabel(value_label)

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def distplot(self, figsize=[10, 4], title=None, savefig_settings={}, ax=None, ylabel='KDE', **plot_args):
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

        ax = sns.distplot(self.value, ax=ax, **plot_args)

        time_label, value_label = self.make_labels()

        ax.set_xlabel(value_label)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def clean(self):
        ''' Clean up the timeseries by removing NaNs and sort with increasing time points
        '''
        y = self.value
        t = self.time
        y_cleaned, t_cleaned = tsutils.clean_ts(y, t)
        self.time = t_cleaned
        self.value = y_cleaned

    def spectral(self, method='wwz', settings={}):
        ''' Perform spectral analysis on the timeseries
        '''
        spec_func = {
            'wwz': specutils.wwz_psd,
            'mtm': specutils.mtm,
        }
        args = {}
        args.update(settings)
        spec_res = spec_func[method](self.value, self.time, **args)
        if type(spec_res) is dict:
            spec_res = dict2namedtuple(spec_res)

        psd = PSD(freq=spec_res.freq, amplitude=spec_res.psd)
        return psd

    def wavelet(self, method='wwz', nv=12, settings={}):
        ''' Perform wavelet analysis on the timeseries
        '''
        wave_func = {
            'wwz': waveutils.wwz,
        }
        # generate default freq
        s0 = 2*np.median(np.diff(self.time))
        a0 = 2**(1/nv)
        noct = np.floor(np.log2(np.size(self.time)))-1
        scale = s0*a0**(np.arange(noct*nv+1))
        freq = 1/scale[::-1]

        args = {'tau': self.time, 'freq': freq}
        args.update(settings)
        wave_res = wave_func[method](self.value, self.time, **args)
        scal = Scalogram(freq=wave_res.freq, time=wave_res.time, amplitude=wave_res.amplitude, coi=wave_res.coi)

        return scal

    def wavelet_coherence(self, target_series, nv=12, method='wwz', settings={}):
        ''' Perform wavelet coherence analysis with the target timeseries
        '''
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

        args = {'tau': overlap, 'freq': freq}
        args.update(settings)
        xwc_res = xwc_func[method](self.value, self.time, target_series.value, target_series.time, **args)

        coh = Coherence(freq=xwc_res.freq, time=xwc_res.time, coherence=xwc_res.xw_coherence, phase=xwc_res.xw_phase, coi=xwc_res.coi)

        return coh

    def correlation(self, target_series, settings={}):
        ''' Perform correlation analysis with the target timeseries
        '''
        args = {}
        args.update(settings)
        r, signif, p = corrutils.corr_sig(self.value, target_series.value, **args)
        corr_res = {
            'r': r,
            'signif': signif,
            'pvalue': p,
        }
        return corr_res

    def causality(self, target_series, settings={}):
        ''' Perform causality analysis with the target timeseries
        '''
        args = {}
        args.update(settings)
        causal_res = causalutils.causality_est(self.value, target_series.value, **args)
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

    def plot(self, in_loglog=True, in_period=True, xlabel=None, ylabel='Amplitude', title=None,
             xlim=None, ylim=None, figsize=[10, 4], savefig_settings={}, ax=None,
             plot_legend=True, lgd_settings={}, xticks=None, yticks=None, **plot_args):
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

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if in_period:
            idx = np.argwhere(self.freq==0)
            x_axis = 1/np.delete(self.freq, idx)
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
            idx = np.argwhere(self.freq==0)
            x_axis = np.delete(self.freq, idx)
            y_axis = np.delete(self.amplitude, idx)
            if xlabel is None:
                xlabel = 'Frequency'

            if xlim is None:
                xlim = ax.get_xlim()
                xlim = [np.min(xlim), np.max(xlim)]

        ax.set_xlim(xlim)
        ax.plot(x_axis, y_axis, **plot_args)

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
            lgd_args.update(lgd_settings)
            ax.legend(**lgd_args)

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                plotting.showfig(fig)
            return fig, ax
        else:
            return ax

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

    def plot(self, in_period=True, xlabel='Time', ylabel=None, title=None,
             ylim=None, xlim=None, yticks=None, figsize=[10, 8],
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
            y_axis = 1/self.freq
            if ylabel is None:
                ylabel = 'Period'

            if yticks is None:
                yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
                mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
                yticks = yticks_default[mask]
        else:
            y_axis = self.freq
            if ylabel is None:
                ylabel = 'Frequency'

        cont = ax.contourf(self.time, y_axis, self.amplitude.T, **contourf_args)
        ax.set_yscale('log', nonposy='clip')

        # plot colorbar
        cbar_args = {'drawedges': False, 'orientation': 'vertical', 'fraction': 0.15, 'pad': 0.05}
        cbar_args.update(cbar_style)

        cb = plt.colorbar(cont, **cbar_args)

        # plot cone of influence
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

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        ax.set_ylim(ylim)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                plotting.showfig(fig)
            return fig, ax
        else:
            return ax


class Coherence:
    def __init__(self, freq, time, coherence, phase, coi):
        self.freq = np.array(freq)
        self.time = np.array(time)
        self.coherence = np.array(coherence)
        self.coi = np.array(coi)
        self.phase = np.array(phase)

    def plot(self, xlabel='Time', ylabel='Period', title=None, figsize=[10, 8],
             ylim=None, xlim=None, in_period=True, yticks=None,
             contourf_style={}, phase_style={}, cbar_style={}, savefig_settings={}, ax=None,
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
            y_axis = 1/self.freq
            if ylabel is None:
                ylabel = 'Period'

            if yticks is None:
                yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
                mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
                yticks = yticks_default[mask]
        else:
            y_axis = self.freq
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
        phase[self.coherence < pt] = np.nan

        X, Y = np.meshgrid(self.time, 1/self.freq)
        U, V = np.cos(phase).T, np.sin(phase).T

        ax.quiver(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x],
                  U[::skip_y, ::skip_x], V[::skip_y, ::skip_x],
                  scale=scale, width=width)

        ax.set_ylim(ylim)

        if xlim is not None:
            ax.set_xlim(xlim)


        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, savefig_settings)
            else:
                plotting.showfig(fig)
            return fig, ax
        else:
            return ax

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
