''' The application programming interface for Pyleoclim

@author: fengzhu

Created on Jan 31, 2020
'''
from ..utils import tsutils, plotting, mapping, lipdutils, tsmodel, tsbase
from ..utils import wavelet as waveutils
from ..utils import spectral as specutils
from ..utils import correlation as corrutils
from ..utils import causality as causalutils
from ..utils import decomposition
from ..utils import filter as filterutils

#from textwrap import dedent

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import namedtuple
from copy import deepcopy

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MaxNLocator
import matplotlib.transforms as transforms
from matplotlib import cm
from matplotlib import gridspec
import matplotlib as mpl
#from matplotlib.colors import BoundaryNorm, Normalize

import cartopy.crs as ccrs
import cartopy.feature as cfeature


from tqdm import tqdm
from scipy.stats.mstats import mquantiles
from scipy import stats
from statsmodels.multivariate.pca import PCA
import warnings
import os

import lipd as lpd

def pval_format(p, threshold=0.01, style='exp'):
    ''' Print p-value with proper format when p is close to 0
    '''
    if p < threshold:
        if p == 0:
            if style == 'float':
                s = '< 0.000001'
            elif style == 'exp':
                s = '< 1e-6'
            else:
                raise ValueError('Wrong style.')
        else:
            n = int(np.ceil(np.log10(p)))
            if style == 'float':
                s = f'< {10**n}'
            elif style == 'exp':
                s = f'< 1e{n}'
            else:
                raise ValueError('Wrong style.')
    else:
        s = f'{p:.2f}'

    return s

def dict2namedtuple(d):
    ''' Convert a dictionary to a namedtuple
    '''
    tupletype = namedtuple('tupletype', sorted(d))
    return tupletype(**d)

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

def gen_ts(model, t=None, nt=1000, **kwargs):
    ''' Generate pyleoclim.Series with timeseries models

    Parameters
    ----------

    model : str, {'colored_noise', 'colored_noise_2regimes', 'ar1'}
        the timeseries model to use
        - colored_noise : colored noise with one scaling slope
        - colored_noise_2regimes : colored noise with two regimes of two different scaling slopes
        - ar1 : AR(1) series

    t : array
        the time axis

    nt : number of time points
        only works if 't' is None, and it will use an evenly-spaced vector with nt points 

    kwargs : dict
        the keyward arguments for the specified timeseries model

    Returns
    -------

    ts : `pyleoclim.Series`

    See also
    --------

    pyleoclim.utils.tsmodel.colored_noise : generate a colored noise timeseries
    pyleoclim.utils.tsmodel.colored_noise_2regimes : generate a colored noise timeseries with two regimes
    pyleoclim.utils.tsmodel.gen_ar1_evenly : generate an AR(1) series


    Examples
    --------

    - AR(1) series

    .. ipython:: python
        :okwarning:
        :okexcept:    

        import pyleoclim as pyleo

        # default length nt=1000; default persistence parameter g=0.5
        ts = pyleo.gen_ts(model='ar1')
        g = pyleo.utils.tsmodel.ar1_fit(ts.value)
        @savefig gen_ar1_t0.png
        fig, ax = ts.plot(label=f'g={g:.2f}')
        pyleo.closefig(fig)

        # use 'nt' to modify the data length
        ts = pyleo.gen_ts(model='ar1', nt=100)
        g = pyleo.utils.tsmodel.ar1_fit(ts.value)
        @savefig gen_ar1_t1.png
        fig, ax = ts.plot(label=f'g={g:.2f}')
        pyleo.closefig(fig)

        # use 'settings' to modify the persistence parameter 'g'
        ts = pyleo.gen_ts(model='ar1', g=0.9)
        g = pyleo.utils.tsmodel.ar1_fit(ts.value)
        @savefig gen_ar1_t2.png
        fig, ax = ts.plot(label=f'g={g:.2f}')
        pyleo.closefig(fig)

    - Colored noise with 1 regime

    .. ipython:: python
        :okwarning:
        :okexcept:    

        # default scaling slope 'alpha' is 1
        ts = pyleo.gen_ts(model='colored_noise')
        psd = ts.spectral()

        # estimate the scaling slope
        psd_beta = psd.beta_est(fmin=1/50, fmax=1/2)
        print(psd_beta.beta_est_res['beta'])

        # visualize
        @savefig gen_cn_t0.png
        fig, ax = psd.plot()
        pyleo.closefig(fig)

        # modify 'alpha' with 'settings'
        ts = pyleo.gen_ts(model='colored_noise', alpha=2)
        psd = ts.spectral()

        # estimate the scaling slope
        psd_beta = psd.beta_est(fmin=1/50, fmax=1/2)
        print(psd_beta.beta_est_res['beta'])

        # visualize
        @savefig gen_cn_t1.png
        fig, ax = psd.plot()
        pyleo.closefig(fig)

    - Colored noise with 2 regimes

    .. ipython:: python
        :okwarning
        :okexcept:

        # default scaling slopes 'alpha1' is 0.5 and 'alpha2' is 2, with break at 1/20
        ts = pyleo.gen_ts(model='colored_noise_2regimes')
        psd = ts.spectral()

        # estimate the scaling slope
        psd_beta_lf = psd.beta_est(fmin=1/50, fmax=1/20)
        psd_beta_hf = psd.beta_est(fmin=1/20, fmax=1/2)
        print(psd_beta_lf.beta_est_res['beta'])
        print(psd_beta_hf.beta_est_res['beta'])

        # visualize
        @savefig gen_cn2_t0.png
        fig, ax = psd.plot()
        pyleo.closefig(fig)

        # modify the scaling slopes and scaling break with 'settings'
        ts = pyleo.gen_ts(model='colored_noise_2regimes', alpha1=2, alpha2=1, f_break=1/10)
        psd = ts.spectral()

        # estimate the scaling slope
        psd_beta_lf = psd.beta_est(fmin=1/50, fmax=1/10)
        psd_beta_hf = psd.beta_est(fmin=1/10, fmax=1/2)
        print(psd_beta_lf.beta_est_res['beta'])
        print(psd_beta_hf.beta_est_res['beta'])

        # visualize
        @savefig gen_cn2_t1.png
        fig, ax = psd.plot()
        pyleo.closefig(fig)

    '''
    if t is None:
        t = np.arange(nt)

    tsm = {
        'colored_noise': tsmodel.colored_noise,
        'colored_noise_2regimes': tsmodel.colored_noise_2regimes,
        'ar1': tsmodel.gen_ar1_evenly,
    }

    tsm_args = {}
    tsm_args['colored_noise'] = {'alpha': 1}
    tsm_args['colored_noise_2regimes'] = {'alpha1': 1/2, 'alpha2': 2, 'f_break': 1/20}
    tsm_args['ar1'] = {'g': 0.5}
    tsm_args[model].update(kwargs)

    v = tsm[model](t=t, **tsm_args[model])
    ts = Series(time=t, value=v)
    return ts


class Series:
    ''' pyleoSeries object

    The Series class is, at its heart, a simple structure containing two arrays y and t of equal length, and some
    metadata allowing to interpret and plot the series. It is similar to a pandas Series, but the concept
    was extended because pandas does not yet support geologic time.

    Parameters
    ----------

    time : list or numpy.array
        independent variable (t)

    value : list of numpy.array
        values of the dependent variable (y)

    time_unit : string
        Units for the time vector (e.g., 'years').
        Default is 'years'

    time_name : string
        Name of the time vector (e.g., 'Time','Age').
        Default is None. This is used to label the time axis on plots

    value_name : string
        Name of the value vector (e.g., 'temperature')
        Default is None

    value_unit : string
        Units for the value vector (e.g., 'deg C')
        Default is None

    label : string
        Name of the time series (e.g., 'Nino 3.4')
        Default is None

    clean_ts : boolean flag
        set to True to remove the NaNs and make time axis strictly prograde with duplicated timestamps reduced by averaging the values
        Default is True

    verbose : bool
        If True, will print warning messages if there is any

    Examples
    --------

    In this example, we import the Southern Oscillation Index (SOI) into a pandas dataframe and create a pyleoSeries object.

    .. ipython:: python
        :okwarning:
        :okexcept:    

        import pyleoclim as pyleo
        import pandas as pd
        data=pd.read_csv(
            'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
            skiprows=0, header=1
        )
        time=data.iloc[:,1]
        value=data.iloc[:,2]
        ts=pyleo.Series(
            time=time, value=value,
            time_name='Year (CE)', value_name='SOI', label='Southern Oscillation Index'
        )
        ts
        ts.__dict__.keys()
    '''

    def __init__(self, time, value, time_name=None, time_unit=None, value_name=None, value_unit=None, label=None, clean_ts=True, verbose=False):

        if clean_ts==True:
            value, time = tsbase.clean_ts(np.array(value), np.array(time), verbose=verbose)

        self.time = time
        self.value = value
        self.time_name = time_name
        self.time_unit = time_unit
        self.value_name = value_name
        self.value_unit = value_unit
        self.label = label
        self.clean_ts=clean_ts
        self.verbose=verbose

    def convert_time_unit(self, time_unit='years'):
        ''' Convert the time unit of the Series object

        Parameters
        ----------

        time_unit : str
            the target time unit, possible input:
            {
                'year', 'years', 'yr', 'yrs',
                'y BP', 'yr BP', 'yrs BP', 'year BP', 'years BP',
                'ky BP', 'kyr BP', 'kyrs BP', 'ka BP', 'ka',
                'my BP', 'myr BP', 'myrs BP', 'ma BP', 'ma',
            }

        Examples
        --------
        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv(
                'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
                skiprows=0, header=1
            )
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts = pyleo.Series(time=time, value=value, time_unit='years')
            new_ts = ts.convert_time_unit(time_unit='yrs BP')
            print('Original timeseries:')
            print('time unit:', ts.time_unit)
            print('time:', ts.time)
            print()
            print('Converted timeseries:')
            print('time unit:', new_ts.time_unit)
            print('time:', new_ts.time)
        '''

        new_ts = self.copy()
        if time_unit is not None:
            tu = time_unit.lower()
            if tu.find('ky')>=0 or tu.find('ka')>=0:
                time_unit_label = 'ky BP'
            elif tu.find('my')>=0 or tu.find('ma')>=0:
                time_unit_label = 'my BP'
            elif tu.find('y bp')>=0 or tu.find('yr bp')>=0 or tu.find('yrs bp')>=0 or tu.find('year bp')>=0 or tu.find('years bp')>=0:
                time_unit_label = 'yrs BP'
            elif tu.find('yr')>=0 or tu.find('year')>=0 or tu.find('yrs')>=0 or tu.find('years')>=0:
                time_unit_label = 'yrs'
            else:
                raise ValueError(f"Input time_unit={time_unit} is not supported. Supported input: 'year', 'years', 'yr', 'yrs', 'y BP', 'yr BP', 'yrs BP', 'year BP', 'years BP', 'ky BP', 'kyr BP', 'kyrs BP', 'ka BP', 'my BP', 'myr BP', 'myrs BP', 'ma BP'.")
        else:
            return new_ts

        def convert_to_years():
            def prograde_time(time, time_datum, time_exponent):
                new_time = (time_datum + time)*10**(time_exponent)
                return new_time

            def retrograde_time(time, time_datum, time_exponent):
                new_time = (time_datum - time)*10**(time_exponent)
                return new_time

            convert_func = {
                'prograde': prograde_time,
                'retrograde': retrograde_time,
            }
            if self.time_unit is not None:
                tu = self.time_unit.lower()
                if tu.find('ky')>=0 or tu.find('ka')>=0:
                    time_dir = 'retrograde'
                    time_datum = 1950/1e3
                    time_exponent = 3
                    time_unit_label = 'ky BP'
                elif tu.find('my')>=0 or tu.find('ma')>=0:
                    time_dir = 'retrograde'
                    time_datum = 1950/1e6
                    time_exponent = 6
                elif tu.find('y bp')>=0 or tu.find('yr bp')>=0 or tu.find('yrs bp')>=0 or tu.find('year bp')>=0 or tu.find('years bp')>=0:
                    time_dir ='retrograde'
                    time_datum = 1950
                    time_exponent = 0
                else:
                    time_dir ='prograde'
                    time_datum = 0
                    time_exponent = 0

                new_time = convert_func[time_dir](self.time, time_datum, time_exponent)
            else:
                new_time = None

            return new_time

        def convert_to_bp():
            time_yrs = convert_to_years()
            time_bp = 1950 - time_yrs
            return time_bp

        def convert_to_ka():
            time_bp = convert_to_bp()
            time_ka = time_bp / 1e3
            return time_ka

        def convert_to_ma():
            time_bp = convert_to_bp()
            time_ma = time_bp / 1e6
            return time_ma

        convert_to = {
            'yrs': convert_to_years(),
            'yrs BP': convert_to_bp(),
            'ky BP': convert_to_ka(),
            'my BP': convert_to_ma(),
        }

        new_time = convert_to[time_unit_label]

        dt = np.diff(new_time)
        if any(dt<=0):
            new_value, new_time = tsbase.sort_ts(self.value, new_time)
        else:
            new_value = self.copy().value

        new_ts.time = new_time
        new_ts.value = new_value
        new_ts.time_unit = time_unit

        return new_ts

    def make_labels(self):
        '''
        Initialization of labels

        Returns
        -------
        time_header : str
            Label for the time axis
        value_header : str
            Label for the value axis

        '''
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
        '''
        Prints out the series in a table format and length of the series

        Returns
        -------
        str
            length of the timeseries.

        '''
        time_label, value_label = self.make_labels()

        table = {
            time_label: self.time,
            value_label: self.value,
        }

        msg = print(tabulate(table, headers='keys'))
        return f'Length: {np.size(self.time)}'

    def stats(self):
        """ Compute basic statistics for the time series

        Computes the mean, median, min, max, standard deviation, and interquartile range of a numpy array y, ignoring NaNs.

        Returns
        -------

        res : dictionary
            Contains the mean, median, minimum value, maximum value, standard
            deviation, and interquartile range for the Series.

        Examples
        --------

        Compute basic statistics for the SOI series

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import pandas as pd
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time=data.iloc[:,1]
            value=data.iloc[:,2]
            ts=pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')
            ts.stats()
        """
        mean, median, min_, max_, std, IQR = tsutils.simple_stats(self.value)
        res={'mean':mean,
             'median':median,
             'min':min_,
             'max':max_,
             'std':std,
             'IQR': IQR}
        return res

    def plot(self, figsize=[10, 4],
             marker=None, markersize=None, color=None,
             linestyle=None, linewidth=None, xlim=None, ylim=None,
             label=None, xlabel=None, ylabel=None, title=None, zorder=None,
             legend=True, plot_kwargs=None, lgd_kwargs=None, alpha=None,
             savefig_settings=None, ax=None, mute=False, invert_xaxis=False):
        ''' Plot the timeseries

        Parameters
        ----------

        figsize : list
            a list of two integers indicating the figure size

        marker : str
            e.g., 'o' for dots
            See [matplotlib.markers](https://matplotlib.org/3.1.3/api/markers_api.html) for details

        markersize : float
            the size of the marker

        color : str, list
            the color for the line plot
            e.g., 'r' for red
            See [matplotlib colors] (https://matplotlib.org/3.2.1/tutorials/colors/colors.html) for details

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

        zorder : int
            The default drawing order for all lines on the plot

        legend : {True, False}
            plot legend or not

        invert_xaxis : bool, optional
            if True, the x-axis of the plot will be inverted

        plot_kwargs : dict
            the dictionary of keyword arguments for ax.plot()
            See [matplotlib.pyplot.plot](https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.plot.html) for details

        lgd_kwargs : dict
            the dictionary of keyword arguments for ax.legend()
            See [matplotlib.pyplot.legend](https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.legend.html) for details

        alpha : float
            Transparency setting

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        ax : matplotlib.axis, optional
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        mute : {True,False}
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)

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

        See also
        --------

        pyleoclim.utils.plotting.savefig : saving figure in Pyleoclim

        Examples
        --------

        Plot the SOI record

            .. ipython:: python
                :okwarning:
                :okexcept:    

                import pyleoclim as pyleo
                import pandas as pd
                data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
                time = data.iloc[:,1]
                value = data.iloc[:,2]
                ts = pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')
                @savefig ts_plot.png
                fig, ax = ts.plot()
                pyleo.closefig(fig)

        Change the line color

            .. ipython:: python
                :okwarning:
                :okexcept:    

                @savefig ts_plot2.png
                fig, ax = ts.plot(color='r')
                pyleo.closefig(fig)

        Save the figure. Two options available:
            * Within the plotting command
            * After the figure has been generated

            .. ipython:: python
                :okwarning:
                :okexcept:    

                fig, ax = ts.plot(color='k', savefig_settings={'path': 'ts_plot3.png'})
                pyleo.savefig(fig,path='ts_plot3.png')
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
            mute=mute, invert_xaxis=invert_xaxis,
        )

        return res

    def ssa(self, M=None, nMC=0, f=0.3, trunc = None, var_thresh=80):
        ''' Singular Spectrum Analysis
        
        Nonparametric, orthogonal decomposition of timeseries into constituent oscillations.
        This implementation  uses the method of [1], with applications presented in [2].
        Optionally (MC>0), the significance of eigenvalues is assessed by Monte-Carlo simulations of an AR(1) model fit to X, using [3].
        The method expects regular spacing, but is tolerant to missing values, up to a fraction 0<f<1 (see [4]).

        Parameters
        ----------
        M : int, optional
            window size. The default is None (10% of the length of the series).
        MC : int, optional
            Number of iteration in the Monte-Carlo process. The default is 0.
        f : float, optional
            maximum allowable fraction of missing values. The default is 0.3.
        trunc : str
            if present, truncates the expansion to a level K < M owing to one of 3 criteria:
                (1) 'kaiser': variant of the Kaiser-Guttman rule, retaining eigenvalues larger than the median
                (2) 'mc-ssa': Monte-Carlo SSA (use modes above the 95% threshold)
                (3) 'var': first K modes that explain at least var_thresh % of the variance.
            Default is None, which bypasses truncation (K = M)

        var_thresh : float
            variance threshold for reconstruction (only impactful if trunc is set to 'var')

        Returns
        -------
        res : object of the SsaRes class containing:

        - eigvals : (M, ) array of eigenvalues

        - eigvecs : (M, M) Matrix of temporal eigenvectors (T-EOFs)

        - PC : (N - M + 1, M) array of principal components (T-PCs)

        - RCmat : (N,  M) array of reconstructed components
        
        - RCseries : (N,) reconstructed series, with mean and variance restored

        - pctvar: (M, ) array of the fraction of variance (%) associated with each mode

        - eigvals_q : (M, 2) array contaitning the 5% and 95% quantiles of the Monte-Carlo eigenvalue spectrum [ if nMC >0 ]
        
            
        Examples
        --------

        SSA with SOI

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts = pyleo.Series(time=time, value=value, time_name='Year C.E', value_name='SOI', label='SOI')
            # plot
            @savefig ts_plot4.png
            fig, ax = ts.plot()
            pyleo.closefig(fig)

            # SSA
            nino_ssa = ts.ssa(M=60)

        Let us now see how to make use of all these arrays. The first step is too inspect the eigenvalue spectrum ("scree plot") to identify remarkable modes. Let us restrict ourselves to the first 40, so we can see something:

        .. ipython:: python
            :okwarning:
            :okexcept:    
            var_pct = nino_ssa['pctvar'] # extract the fraction of variance attributable to each mode

            # plot eigenvalues
           
            nino_ssa.screeplot()
            @savefig ts_eigen.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)

        This highlights a few common phenomena with SSA:
            * the eigenvalues are in descending order
            * their uncertainties are proportional to the eigenvalues themselves
            * the eigenvalues tend to come in pairs : (1,2) (3,4), are all clustered within uncertainties . (5,6) looks like another doublet
            * around i=15, the eigenvalues appear to reach a floor, and all subsequent eigenvalues explain a very small amount of variance.

        So, summing the variance of all modes higher than 19, we get:

        .. ipython:: python
            :okwarning:
            :okexcept:    

            print(nino_ssa.pctvar[15:].sum()*100)

        That is, over 95% of the variance is in the first 15 modes. That is a typical result for a (paleo)climate timeseries; a few modes do the vast majority of the work. That means we can focus our attention on these modes and capture most of the interesting behavior. To see this, let's use the reconstructed components (RCs), and sum the RC matrix over the first 15 columns:

        .. ipython:: python
            :okwarning:
            :okexcept:    

            RCk = nino_ssa.RCmat[:,:14].sum(axis=1)
            fig, ax = ts.plot(title='ONI') # we mute the first call to only get the plot with 2 lines
            ax.plot(time,RCk,label='SSA reconstruction, 14 modes',color='orange')
            ax.legend()
            @savefig ssa_recon.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)

        Indeed, these first few modes capture the vast majority of the low-frequency behavior, including all the El Niño/La Niña events. What is left (the blue wiggles not captured in the orange curve) are high-frequency oscillations that might be considered "noise" from the standpoint of ENSO dynamics. This illustrates how SSA might be used for filtering a timeseries. One must be careful however:
            * there was not much rhyme or reason for picking 15 modes. Why not 5, or 39? All we have seen so far is that they gather >95% of the variance, which is by no means a magic number.
            * there is no guarantee that the first few modes will filter out high-frequency behavior, or at what frequency cutoff they will do so. If you need to cut out specific frequencies, you are better off doing it with a classical filter, like the butterworth filter implemented in Pyleoclim. However, in many instances the choice of a cutoff frequency is itself rather arbitrary. In such cases, SSA provides a principled alternative for generating a version of a timeseries that preserves features and excludes others (i.e, a filter).
            * as with all orthgonal decompositions, summing over all RCs will recover the original signal within numerical precision.

        Monte-Carlo SSA

        Selecting meaningful modes in eigenproblems (e.g. EOF analysis) is more art than science. However, one technique stands out: Monte Carlo SSA, introduced by Allen & Smith, (1996) to identiy SSA modes that rise above what one would expect from "red noise", specifically an AR(1) process_process). To run it, simply provide the parameter MC, ideally with a number of iterations sufficient to get decent statistics. Here's let's use MC = 1000. The result will be stored in the eigval_q array, which has the same length as eigval, and its two columns contain the 5% and 95% quantiles of the ensemble of MC-SSA eigenvalues.

        .. ipython:: python
            :okwarning:
            :okexcept:    

            nino_mcssa = ts.ssa(M = 60, nMC=1000)

        Now let's look at the result:

        .. ipython:: python
            :okwarning:
            :okexcept:    

            nino_mcssa.screeplot()
            @savefig scree_nmc.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)

        This suggests that modes 1-5 fall above the red noise benchmark.

        '''

        res = decomposition.ssa(self.value, M=M, nMC=nMC, f=f, trunc = trunc, var_thresh=var_thresh)
        
                
        resc = SsaRes(name=self.value_name, original=self.value, time = self.time, eigvals = res['eigvals'], eigvecs = res['eigvecs'],
                        pctvar = res['pctvar'], PC = res['PC'], RCmat = res['RCmat'], 
                        RCseries=res['RCseries'], mode_idx=res['mode_idx'])
        if nMC >= 0:
           resc.eigvals_q=res['eigvals_q'] # assign eigenvalue quantiles if Monte-Carlo SSA was called
        
        return resc

    def is_evenly_spaced(self, tol=1e-3):
        ''' Check if the Series time axis is evenly-spaced, within tolerance

        Returns
        ------

        res : bool
        '''

        res = tsbase.is_evenly_spaced(self.time, tol)
        return res

    def filter(self, cutoff_freq=None, cutoff_scale=None, method='butterworth', **kwargs):
        ''' Filtering methods for Series objects using four possible methods:
            - `Butterworth <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_ 
            - `Lanczos <http://scitools.org.uk/iris/docs/v1.2/examples/graphics/SOI_filtering.html>`_  
            - `Finite Impulse Response <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html>`_  
            - `Savitzky-Golay filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_
                
        By default, this method implements a lowpass filter, though it can easily be turned into a bandpass or high-pass filter (see examples below).

        Parameters
        ----------

        method : str, {'savitzky-golay', 'butterworth', 'firwin', 'lanczos'}

            the filtering method
            - 'butterworth': a Butterworth filter (default = 3rd order)
            - 'savitzky-golay': Savitzky-Golay filter
            - 'firwin': finite impulse response filter design using the window method, with default window as Hamming
            - 'lanczos': Lanczos zero-phase filter 

        cutoff_freq : float or list
            The cutoff frequency only works with the Butterworth method.
            If a float, it is interpreted as a low-frequency cutoff (lowpass).
            If a list,  it is interpreted as a frequency band (f1, f2), with f1 < f2 (bandpass). 
            Note that only the Butterworth option (default) currently supports bandpass filtering. 

        cutoff_scale : float or list
            cutoff_freq = 1 / cutoff_scale
            The cutoff scale only works with the Butterworth method and when cutoff_freq is None.
            If a float, it is interpreted as a low-frequency (high-scale) cutoff (lowpass).
            If a list,  it is interpreted as a frequency band (f1, f2), with f1 < f2 (bandpass).

        kwargs : dict
            a dictionary of the keyword arguments for the filtering method,
            see `pyleoclim.utils.filter.savitzky_golay`, `pyleoclim.utils.filter.butterworth`, `pyleoclim.utils.filter.lanczos` and `pyleoclim.utils.filter.firwin` for the details

        Returns
        -------

        new : pyleoclim.Series

        See also
        --------

        pyleoclim.utils.filter.butterworth : Butterworth method
        pyleoclim.utils.filter.savitzky_golay : Savitzky-Golay method
        pyleoclim.utils.filter.firwin : FIR filter design using the window method
        pyleoclim.utils.filter.lanczos : lowpass filter via Lanczos resampling


        Examples
        --------

        In the example below, we generate a signal as the sum of two signals with frequency 10 Hz and 20 Hz, respectively.
        Then we apply a low-pass filter with a cutoff frequency at 15 Hz, and compare the output to the signal of 10 Hz.
        After that, we apply a band-pass filter with the band 15-25 Hz, and compare the outcome to the signal of 20 Hz.

        - Generating the test data

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import numpy as np

            t = np.linspace(0, 1, 1000)
            sig1 = np.sin(2*np.pi*10*t)
            sig2 = np.sin(2*np.pi*20*t)
            sig = sig1 + sig2
            ts1 = pyleo.Series(time=t, value=sig1)
            ts2 = pyleo.Series(time=t, value=sig2)
            ts = pyleo.Series(time=t, value=sig)
            fig, ax = ts.plot(label='mix')
            ts1.plot(ax=ax, label='10 Hz')
            ts2.plot(ax=ax, label='20 Hz')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=3)
            @savefig ts_filter1.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)

        - Applying a low-pass filter

        .. ipython:: python
            :okwarning:
            :okexcept:    

            fig, ax = ts.plot(label='mix')
            ts.filter(cutoff_freq=15).plot(ax=ax, label='After 15 Hz low-pass filter')
            ts1.plot(ax=ax, label='10 Hz')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=3)
            @savefig ts_filter2.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)

        - Applying a band-pass filter

        .. ipython:: python
            :okwarning:
            :okexcept:    

            fig, ax = ts.plot(label='mix')
            ts.filter(cutoff_freq=[15, 25]).plot(ax=ax, label='After 15-25 Hz band-pass filter')
            ts2.plot(ax=ax, label='20 Hz')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=3)
            @savefig ts_filter3.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)

        Above is using the default Butterworth filtering. To use FIR filtering with a window like Hanning is also simple:
            
        .. ipython:: python
            :okwarning:
            :okexcept:    

            fig, ax = ts.plot(label='mix')
            ts.filter(cutoff_freq=[15, 25], method='firwin', window='hanning').plot(ax=ax, label='After 15-25 Hz band-pass filter')
            ts2.plot(ax=ax, label='20 Hz')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=3)
            @savefig ts_filter4.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)
            
        - Applying a high-pass filter

        .. ipython:: python
            :okwarning:
            :okexcept:    

            fig, ax = ts.plot(label='mix')
            ts_low  = ts.filter(cutoff_freq=15)
            ts_high = ts.copy()
            ts_high.value = ts.value - ts_low.value # subtract low-pass filtered series from original one
            ts_high.plot(label='High-pass filter @ 15Hz',ax=ax)
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=3)
            @savefig ts_filter5.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)

        '''
        if not self.is_evenly_spaced():
            raise ValueError('This  method assumes evenly-spaced timeseries, while the input is not. Use the ".interp()", ".bin()" or ".gkernel()" methods prior to ".filter()".')

        new = self.copy()
        
        mu = np.mean(self.value) # extract the mean
        y = self.value - mu
        
        fs = 1/np.mean(np.diff(self.time))

        method_func = {
            'savitzky-golay': filterutils.savitzky_golay,
            'butterworth': filterutils.butterworth,
            'firwin': filterutils.firwin,
            'lanczos': filterutils.lanczos,
        }

        args = {}

        if method in ['butterworth', 'firwin', 'lanczos']:
            if cutoff_freq is None:
                if cutoff_scale is None:
                    raise ValueError('Please set the cutoff frequency or scale argument: "cutoff_freq" or "cutoff_scale".')
                else:
                    if np.isscalar(cutoff_scale):
                        cutoff_freq = 1 / cutoff_scale
                    elif len(cutoff_scale) == 2 and method in ['butterworth', 'firwin']:
                        cutoff_scale = np.array(cutoff_scale)
                        cutoff_freq = np.sort(1 / cutoff_scale)
                        cutoff_freq = list(cutoff_freq)
                    elif len(cutoff_scale) > 1 and method == 'lanczos':
                        raise ValueError('Lanczos filter requires a scalar input as cutoff scale/frequency')
                    else:
                        raise ValueError('Wrong cutoff_scale; should be either one float value (lowpass) or a list two float values (bandpass).')
            # assign optional arguments            
            args['butterworth'] = {'fc': cutoff_freq, 'fs': fs}
            args['firwin'] = {'fc': cutoff_freq, 'fs': fs}
            args['lanczos'] = {'fc': cutoff_freq, 'fs': fs}
        
        else: # for Savitzky-Golay only
            if cutoff_scale and cutoff_freq is None:
                raise ValueError('No cutoff_scale or cutoff_freq argument provided')
            elif cutoff_freq is not None:
                cutoff_scale = 1 / cutoff_freq
            
            window_length = int(cutoff_scale*fs)
            if window_length % 2 == 0:
                window_length += 1   # window length needs to be an odd integer
            args['savitzky-golay'] = {'window_length': window_length}
            args[method].update(kwargs)

        new_val = method_func[method](y, **args[method])
        new.value = new_val + mu # restore the mean

        return new



    def distplot(self, figsize=[10, 4], title=None, savefig_settings=None,
                 ax=None, ylabel='KDE', vertical=False, edgecolor='w',mute=False, **plot_kwargs):
        ''' Plot the distribution of the timeseries values

        Parameters
        ----------

        figsize : list
            a list of two integers indicating the figure size

        title : str
            the title for the figure

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
              - "path" must be specified; it can be any existed or non-existed path,
                with or without a suffix; if the suffix is not given in "path", it will follow "format"
              - "format" can be one of {"pdf", "eps", "png", "ps"}

        ax : matplotlib.axis, optional
            A matplotlib axis

        ylabel : str
            Label for the count axis

        vertical : {True,False}
            Whether to flip the plot vertically

        edgecolor : matplotlib.color
            The color of the edges of the bar

        mute : {True,False}
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)

        plot_kwargs : dict
            Plotting arguments for seaborn histplot: https://seaborn.pydata.org/generated/seaborn.histplot.html

        See also
        --------

        pyleoclim.utils.plotting.savefig : saving figure in Pyleoclim

        Examples
        --------

        Distribution of the SOI record

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import pandas as pd
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time=data.iloc[:,1]
            value=data.iloc[:,2]
            ts=pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')

            @savefig ts_plot5.png
            fig, ax = ts.plot()
            pyleo.closefig(fig)

            @savefig ts_dist.png
            fig, ax = ts.distplot()
            pyleo.closefig(fig)

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        #make the data into a dataframe so we can flip the figure
        time_label, value_label = self.make_labels()
        if vertical == True:
            data=pd.DataFrame({'value':self.value})
            ax = sns.histplot(data=data, y="value", ax=ax, kde=True, edgecolor=edgecolor, **plot_kwargs)
            ax.set_ylabel(value_label)
            ax.set_xlabel(ylabel)
        else:
            ax = sns.histplot(self.value, ax=ax, kde=True, edgecolor=edgecolor, **plot_kwargs)
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

    def summary_plot(self, psd=None, scalogram=None, figsize=[8, 10], title=None,
                    time_lim=None, value_lim=None, period_lim=None, psd_lim=None, n_signif_test=None,
                    time_label=None, value_label=None, period_label=None, psd_label='PSD', wavelet_method = 'wwz', 
                    wavelet_kwargs = None, psd_method = 'wwz', psd_kwargs = None, ts_plot_kwargs = None, wavelet_plot_kwargs = None, 
                    psd_plot_kwargs = None, trunc_series = None, preprocess = True, y_label_loc = -.15, savefig_settings=None, 
                    mute=False):
        ''' Generate a plot of the timeseries and its frequency content through spectral and wavelet analyses.


        Parameters
        ----------

        psd : PSD
            the PSD object of a Series. If None, and psd_kwargs is empty, the PSD from the calculated Scalogram will be used. 
            Otherwise it will be calculated based on specifications in psd_kwargs.

        scalogram : Scalogram
            the Scalogram object of a Series. If None, will be calculated. This process can be slow as it will be using the WWZ method.
            If the passed scalogram object contains stored signif_scals (see pyleo.Scalogram.signif_test() for details) these will
            be flexibly reused as a function of the value of n_signif_test in the summary plot. 

        figsize : list
            a list of two integers indicating the figure size

        title : str
            the title for the figure

        time_lim : list or tuple
            the limitation of the time axis. This is for display purposes only, the scalogram and psd will still be calculated using the full time series.

        value_lim : list or tuple
            the limitation of the value axis of the timeseries. This is for display purposes only, the scalogram and psd will still be calculated using the full time series.

        period_lim : list or tuple
            the limitation of the period axis

        psd_lim : list or tuple
            the limitation of the psd axis

        n_signif_test=None : int
            Number of Monte-Carlo simulations to perform for significance testing. Default is None. If a scalogram is passed it will be parsed for significance testing purposes.

        time_label : str
            the label for the time axis

        value_label : str
            the label for the value axis of the timeseries

        period_label : str
            the label for the period axis

        psd_label : str
            the label for the amplitude axis of PDS
            
        wavelet_method : str
            the method for the calculation of the scalogram, see pyleoclim.core.ui.Series.wavelet for details
            
        wavelet_kwargs : dict
            arguments to be passed to the wavelet function, see pyleoclim.core.ui.Series.wavelet for details
        
        psd_method : str
            the method for the calculation of the psd, see pyleoclim.core.ui.Series.spectral for details
            
        psd_kwargs : dict
            arguments to be passed to the spectral function, see pyleoclim.core.ui.Series.spectral for details
            
        ts_plot_kwargs : dict
            arguments to be passed to the timeseries subplot, see pyleoclim.core.ui.Series.plot for details
        
        wavelet_plot_kwargs : dict
            arguments to be passed to the scalogram plot, see pyleoclim.core.ui.Scalogram.plot for details
        
        psd_plot_kwargs : dict
            arguments to be passed to the psd plot, see pyleoclim.core.ui.PSD.plot for details
                Certain psd plot settings are required by summary plot formatting. These include:
                    - ylabel
                    - legend
                    - tick parameters
                These will be overriden by summary plot to prevent formatting errors
                
        y_label_loc : float
            Plot parameter to adjust horizontal location of y labels to avoid conflict with axis labels, default value is -0.15
                
        trunc_series : list or tuple
            the limitation of the time axis. This will slice the actual time series into one contained within the 
            passed boundaries and as such effect the resulting scalogram and psd objects 
            (assuming said objects are to be generated by summary_plot).
        
        preprocess : bool
            if True, the series will be standardized and detrended using pyleoclim defaults 
            prior to the calculation of the scalogram and psd. The unedited series will be used in the plot,
            while the edited series will be used to calculate the psd and scalogram.

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        mute : bool
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)

        See also
        --------

        pyleoclim.core.ui.Series.spectral : Spectral analysis for a timeseries

        pyleoclim.core.ui.Series.wavelet : Wavelet analysis for a timeseries

        pyleoclim.utils.plotting.savefig : saving figure in Pyleoclim

        pyleoclim.core.ui.PSD : PSD object

        pyleoclim.core.ui.MultiplePSD : Multiple PSD object
        
        Examples
        --------

        Simple summary_plot with n_signif_test = 1 for computational ease, defaults otherwise.

        .. ipython:: python
            :okwarning:
            :okexcept:    
            
            import pyleoclim as pyleo
            import pandas as pd
            ts=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/soi_data.csv',skiprows = 1)
            series = pyleo.Series(time = ts['Year'],value = ts['Value'], time_name = 'Years', time_unit = 'AD')
            fig, ax = series.summary_plot(n_signif_test=1)

            pyleo.showfig(fig)

            pyleo.closefig(fig)
            
        Summary_plot with pre-generated psd and scalogram objects. Note that if the scalogram contains saved noise realizations these will be flexibly reused. See pyleo.Scalogram.signif_test() for details
        
        .. ipython:: python
            :okwarning:
            :okexcept:    
            
            import pyleoclim as pyleo
            import pandas as pd
            ts=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/soi_data.csv',skiprows = 1)
            series = pyleo.Series(time = ts['Year'],value = ts['Value'], time_name = 'Years', time_unit = 'AD')
            psd = series.spectral(freq_method = 'welch')
            scalogram = series.wavelet(freq_method = 'welch')
            fig, ax = series.summary_plot(psd = psd,scalogram = scalogram,n_signif_test=2)

            pyleo.showfig(fig)

            pyleo.closefig(fig)
        
        Summary_plot with pre-generated psd and scalogram objects from before and some plot modification arguments passed. Note that if the scalogram contains saved noise realizations these will be flexibly reused. See pyleo.Scalogram.signif_test() for details
        
        .. ipython:: python
            :okwarning:
            :okexcept:    
            
            import pyleoclim as pyleo
            import pandas as pd
            ts=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/soi_data.csv',skiprows = 1)
            series = pyleo.Series(time = ts['Year'],value = ts['Value'], time_name = 'Years', time_unit = 'AD')
            psd = series.spectral(freq_method = 'welch')
            scalogram = series.wavelet(freq_method = 'welch')
            fig, ax = series.summary_plot(psd = psd,scalogram = scalogram, n_signif_test=2, period_lim = [5,0], ts_plot_kwargs = {'color':'red','linewidth':.5}, psd_plot_kwargs = {'color':'red','linewidth':.5})
            
            pyleo.showfig(fig)
            
            pyleo.closefig(fig)

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(6, 12)
        gs.update(wspace=0, hspace=0)

        wavelet_kwargs={} if wavelet_kwargs is None else wavelet_kwargs.copy()
        wavelet_plot_kwargs={} if wavelet_plot_kwargs is None else wavelet_plot_kwargs.copy()
        psd_kwargs={} if psd_kwargs is None else psd_kwargs.copy()
        psd_plot_kwargs={} if psd_plot_kwargs is None else psd_plot_kwargs.copy()
        ts_plot_kwargs={} if ts_plot_kwargs is None else ts_plot_kwargs.copy()
        
        if trunc_series is not None:
            sub_time = []
            if trunc_series[0] <= self.time[0] and trunc_series[1] >= self.time[-1]:
                print('Truncation period encapsulates entire series, continuing with defaults.')
            else:
                for i in self.time:
                    if i >= trunc_series[0] and i <= trunc_series[1]:
                        sub_time.append(i)
                try:
                    self = self.slice(sub_time)
                except:
                    print('Number of time points in given truncation period is not even. Removing last time point and continuing.')
                    sub_time.pop(-1)
                    self = self.slice(sub_time)
        
        ax = {}
        ax['ts'] = plt.subplot(gs[0:1, :-3])
        ax['ts'] = self.plot(ax=ax['ts'], **ts_plot_kwargs)
        ax['ts'].xaxis.set_visible(False)
        ax['ts'].get_yaxis().set_label_coords(y_label_loc,0.5)
        
        if preprocess:
            self = self.standardize().detrend()
        
        if time_lim is not None:
            ax['ts'].set_xlim(time_lim)
            if 'xlim' in ts_plot_kwargs:
                print('Xlim passed to time series plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument.')
            
        if value_lim is not None:
            ax['ts'].set_ylim(value_lim)
            if 'ylim' in ts_plot_kwargs:
                print('Ylim passed to time series plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument.')

        ax['scal'] = plt.subplot(gs[1:5, :-3], sharex=ax['ts'])
        
        if 'method' in list(wavelet_kwargs.keys()):
            del wavelet_kwargs['method']
            print('Please pass method via exposed wavelet_method argument, exposed argument overrides key word argument')
        
        if n_signif_test is None:
            #If significance testing isn't specified then we either use what we can find or don't do significance testing
            if scalogram is not None:
                if getattr(scalogram,'signif_scals',None) is not None:
                    n_signif_test = len(scalogram.signif_scals.scalogram_list)
                else:
                    n_signif_test = 0
            else:
                n_signif_test = 0
    
        if n_signif_test > 0:
            #If a scalogram is not passed and significance tests are requested
            if scalogram is None:
                scalogram = self.wavelet(method=wavelet_method, **wavelet_kwargs).signif_test(number=n_signif_test, export_scal=True)
            #If a scalogram is passed, significance tests are requested, and the passed scalogram has some significance tests stored in it
            elif scalogram is not None:
                scalogram = scalogram.signif_test(number=n_signif_test, export_scal = True)
        elif n_signif_test == 0:
            #If specifically no significance tests are requested we just need to generate a scalogram (unless one has been passed)
            if scalogram is None:
                scalogram = self.wavelet(method=wavelet_method, **wavelet_kwargs)
        
        if 'cbar_style' not in wavelet_plot_kwargs:
            wavelet_plot_kwargs.update({'cbar_style':{'orientation': 'horizontal', 'pad': 0.12}})

        ax['scal'] = scalogram.plot(ax=ax['scal'], **wavelet_plot_kwargs)
        ax['scal'].invert_yaxis()
        ax['scal'].get_yaxis().set_label_coords(y_label_loc,0.5)

        ax['psd'] = plt.subplot(gs[1:4, -3:], sharey=ax['scal'])
            
        if 'method' in list(psd_kwargs.keys()):
            del psd_kwargs['method']
            print('Please pass method via exposed psd_method argument, exposed argument overrides key word argument')
        
        #Doing effectively the same thing we did for scalogram but now for the psd object
        if n_signif_test > 0:
            if psd is None:
                if psd_method == scalogram.wave_method:
                    psd = self.spectral(method=psd_method,scalogram=scalogram,**psd_kwargs).signif_test(number=n_signif_test,scalogram=scalogram)
                elif psd_method != scalogram.wave_method:
                    psd = self.spectral(method=psd_method,**psd_kwargs).signif_test(number=n_signif_test)
            elif psd is not None:
                if psd_method == scalogram.wave_method:
                    psd = psd.signif_test(number=n_signif_test,scalogram=scalogram)
                elif psd_method != scalogram.wave_method:
                    psd = psd.signif_test(number=n_signif_test)
        #At this point n_signif_test will be set to zero or some positive number
        elif n_signif_test == 0:
            if psd is None:
                if psd_method == scalogram.wave_method:
                    psd = self.spectral(method=psd_method,scalogram=scalogram,**psd_kwargs)
                elif psd_method != scalogram.wave_method:
                    psd = self.spectral(method=psd_method,**psd_kwargs)

        ax['psd'] = psd.plot(ax=ax['psd'], transpose=True, **psd_plot_kwargs)
        
        if period_lim is not None:
            ax['psd'].set_ylim(period_lim)
            if 'ylim' in psd_plot_kwargs:
               print('Ylim passed to psd plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument.')
        
        ax['psd'].yaxis.set_visible(False)
        ax['psd'].invert_yaxis()
        ax['psd'].set_ylabel(None)
        ax['psd'].tick_params(axis='y', direction='in', labelleft=False)
        ax['psd'].legend().remove()

        if psd_lim is not None:
            ax['psd'].set_xlim(psd_lim)
            if 'xlim' in psd_plot_kwargs:
                print('Xlim passed to psd plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument')

        if title is not None:
            ax['ts'].set_title(title)
            if 'title' in ts_plot_kwargs:
                print('Title passed to time series plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument.')

        if value_label is not None:
            #time_label, value_label = self.make_labels()
            ax['ts'].set_ylabel(value_label)
            if 'ylabel' in ts_plot_kwargs:
                print('Ylabel passed to time series plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument.')

        if time_label is not None:
            #time_label, value_label = self.make_labels()
            ax['scal'].set_xlabel(time_label)
            if  'xlabel' in wavelet_plot_kwargs:
                print('Xlabel passed to scalogram plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument.')

        if period_label is not None:
            #period_unit = infer_period_unit_from_time_unit(self.time_unit)
            #period_label = f'Period [{period_unit}]' if period_unit is not None else 'Period'
            ax['scal'].set_ylabel(period_label)
            if 'ylabel' in wavelet_plot_kwargs:
                print('Ylabel passed to scalogram plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument.')
                

        if psd_label is not None:
            ax['psd'].set_xlabel(psd_label)
            if 'xlabel' in psd_plot_kwargs:
                print('Xlabel passed to psd plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument.')

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)
        # else:
        #     if not mute:
        #         plotting.showfig(fig)
        return fig, ax

    def copy(self):
        '''Make a copy of the Series object

        Returns
        -------
        Series
            A copy of the Series object

        '''
        return deepcopy(self)

    def clean(self, verbose=False):
        ''' Clean up the timeseries by removing NaNs and sort with increasing time points

        Parameters
        ----------
        verbose : bool
            If True, will print warning messages if there is any

        Returns
        -------
        Series
            Series object with removed NaNs and sorting

        '''
        new = self.copy()
        v_mod, t_mod = tsbase.clean_ts(self.value, self.time, verbose=verbose)
        new.time = t_mod
        new.value = v_mod
        return new
    
    def sort(self, verbose=False):
        ''' Ensure timeseries is aligned to a prograde axis.
            If the time axis is prograde to begin with, no transformation is applied.
            
        Parameters
        ----------
        verbose : bool
            If True, will print warning messages if there is any
    
        Returns
        -------
        Series
            Series object with removed NaNs and sorting

        '''
        new = self.copy()
        v_mod, t_mod = tsbase.sort_ts(self.value, self.time, verbose=verbose)
        new.time = t_mod
        new.value = v_mod
        return new
    
    def gaussianize(self):
        ''' Gaussianizes the timeseries

        Returns
        -------
        new : pyleoclim.Series
            The Gaussianized series object

        '''
        new = self.copy()
        v_mod = tsutils.gaussianize(self.value)
        new.value = v_mod
        return new

    def standardize(self):
        '''Standardizes the series ((i.e. renove its estimated mean and divides
            by its estimated standard deviation)

        Returns
        -------
        new : pyleoclim.Series
            The standardized series object

        '''
        new = self.copy()
        v_mod = tsutils.standardize(self.value)[0]
        new.value = v_mod
        return new

    def center(self, timespan=None):
        ''' Centers the series (i.e. renove its estimated mean)

        Parameters
        ----------
        timespan : tuple or list
            The timespan over which the mean must be estimated.
            In the form [a, b], where a, b are two points along the series' time axis.

        Returns
        -------
        tsc : pyleoclim.Series
            The centered series object
        ts_mean : estimated mean of the original series, in case it needs to be restored later   

        '''
        tsc = self.copy()
        if timespan is not None:
            ts_mean  = np.nanmean(self.slice(timespan).value)
            vc = self.value - ts_mean
        else:
            ts_mean  = np.nanmean(self.value)
            vc = self.value - ts_mean
        tsc.value = vc
        return tsc, ts_mean

    def segment(self, factor=10):
        """Gap detection

        This function segments a timeseries into n number of parts following a gap
            detection algorithm. The rule of gap detection is very simple:
            we define the intervals between time points as dts, then if dts[i] is larger than factor * dts[i-1],
            we think that the change of dts (or the gradient) is too large, and we regard it as a breaking point
            and divide the time series into two segments here

        Parameters
        ----------

        ts : pyleoclim Series

        factor : float
            The factor that adjusts the threshold for gap detection

        Returns
        -------

        res : pyleoclim MultipleSeries Object or pyleoclim Series Object
            If gaps were detected, returns the segments in a MultipleSeries object,
            else, returns the original timeseries.

        """
        seg_y, seg_t, n_segs = tsutils.ts2segments(self.value,self.time,factor=factor)
        if len(seg_y)>1:
            s_list=[]
            for idx,s in enumerate(seg_y):
                s_tmp=Series(time=seg_t[idx],value=s,time_name=self.time_name,
                             time_unit=self.time_unit, value_name=self.value_name,
                             value_unit=self.value_unit,label=self.label)
                s_list.append(s_tmp)
            res=MultipleSeries(series_list=s_list)
        elif len(seg_y)==1:
            res=self.copy()
        else:
            raise ValueError('No timeseries detected')
        return res

    def slice(self, timespan):
        ''' Slicing the timeseries with a timespan (tuple or list)

        Parameters
        ----------

        timespan : tuple or list
            The list of time points for slicing, whose length must be even.
            When there are n time points, the output Series includes n/2 segments.
            For example, if timespan = [a, b], then the sliced output includes one segment [a, b];
            if timespan = [a, b, c, d], then the sliced output includes segment [a, b] and segment [c, d].

        Returns
        -------

        new : Series
            The sliced Series object.

        '''
        new = self.copy()
        n_elements = len(timespan)
        if n_elements % 2 == 1:
            raise ValueError('The number of elements in timespan must be even!')

        n_segments = int(n_elements / 2)
        mask = [False for i in range(np.size(self.time))]
        for i in range(n_segments):
            mask |= (self.time >= timespan[i*2]) & (self.time <= timespan[i*2+1])

        new.time = self.time[mask]
        new.value = self.value[mask]
        return new

    def fill_na(self, timespan=None, dt=1):
        ''' Fill NaNs into the timespan

        Parameters
        ----------

        timespan : tuple or list
            The list of time points for slicing, whose length must be 2.
            For example, if timespan = [a, b], then the sliced output includes one segment [a, b].
            If None, will use the start point and end point of the original timeseries

        dt : float
            The time spacing to fill the NaNs; default is 1.

        Returns
        -------

        new : Series
            The sliced Series object.

        '''
        new = self.copy()
        if timespan is None:
            start = np.min(self.time)
            end = np.max(self.time)
        else:
            start = timespan[0]
            end = timespan[-1]

        new_time = np.arange(start, end+dt, dt)
        new_value = np.empty(np.size(new_time))

        for i, t in enumerate(new_time):
            if t in self.time:
                loc = list(self.time).index(t)
                new_value[i] = self.value[loc]
            else:
                new_value[i] = np.nan

        new.time = new_time
        new.value = new_value

        return new


    def detrend(self, method='emd', **kwargs):
        '''Detrend Series object

        Parameters
        ----------
        method : str, optional
            The method for detrending. The default is 'emd'.
            Options include:
                * "linear": the result of a n ordinary least-squares stright line fit to y is subtracted.
                * "constant": only the mean of data is subtracted.
                * "savitzky-golay", y is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
                * "emd" (default): Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
        **kwargs : dict
            Relevant arguments for each of the methods.

        Returns
        -------
        new : pyleoclim.Series
            Detrended Series object

        See also
        --------
        pyleoclim.utils.tsutils.detrend : detrending wrapper functions

        Examples
        --------

        We will generate a random signal with a nonlinear trend and use two  detrending options to recover the original signal.

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import numpy as np

            # Generate a mixed harmonic signal with known frequencies
            freqs=[1/20,1/80]
            time=np.arange(2001)
            signals=[]
            for freq in freqs:
                signals.append(np.cos(2*np.pi*freq*time))
            signal=sum(signals)
            
            # Add a non-linear trend
            slope = 1e-5;  intercept = -1
            nonlinear_trend = slope*time**2 + intercept
            
            # Add a modicum of white noise
            np.random.seed(2333)
            sig_var = np.var(signal)
            noise_var = sig_var / 2 #signal is twice the size of noise
            white_noise = np.random.normal(0, np.sqrt(noise_var), size=np.size(signal))
            signal_noise = signal + white_noise
            
            # Place it all in a series object and plot it:
            ts = pyleo.Series(time=time,value=signal_noise + nonlinear_trend)
            @savefig random_series.png
            fig, ax = ts.plot(title='Timeseries with nonlinear trend')
            pyleo.closefig(fig)
            
            # Detrending with default parameters (using EMD method with 1 mode)
            ts_emd1 = ts.detrend()
            ts_emd1.label = 'default detrending (EMD, last mode)' 
            @savefig ts_emd1.png      
            fig, ax = ts_emd1.plot(title='Detrended with EMD method')
            ax.plot(time,signal_noise,label='target signal')
            ax.legend()
            pyleo.showfig(fig)
            pyleo.closefig(fig)
            
            # We see that the default function call results in a "Hockey Stick" at the end, which is undesirable. 
            # There is no automated way to do this, but with a little trial and error, we find that removing the 2 smoothest modes performs reasonably:
                
            ts_emd2 = ts.detrend(method='emd', n=2)
            ts_emd2.label = 'EMD detrending, last 2 modes' 
            @savefig ts_emd_n2.png
            fig, ax = ts_emd2.plot(title='Detrended with EMD (n=2)')
            ax.plot(time,signal_noise,label='target signal')
            ax.legend()
            pyleo.showfig(fig)
            pyleo.closefig(fig)
            
            # Another option for removing a nonlinear trend is a Savitzky-Golay filter:
            ts_sg = ts.detrend(method='savitzky-golay')
            ts_sg.label = 'savitzky-golay detrending, default parameters'
            @savefig ts_sg.png
            fig, ax = ts_sg.plot(title='Detrended with Savitzky-Golay filter')
            ax.plot(time,signal_noise,label='target signal')
            ax.legend()
            pyleo.showfig(fig)
            pyleo.closefig(fig)
            
            # As we can see, the result is even worse than with EMD (default). Here it pays to look into the underlying method, which comes from SciPy.
            # It turns out that by default, the Savitzky-Golay filter fits a polynomial to the last "window_length" values of the edges. 
            # By default, this value is close to the length of the series. Choosing a value 10x smaller fixes the problem here, though you will have to tinker with that parameter until you get the result you seek.
            
            ts_sg2 = ts.detrend(method='savitzky-golay',sg_kwargs={'window_length':201})
            ts_sg2.label = 'savitzky-golay detrending, window_length = 201'
            @savefig ts_sg2.png
            fig, ax = ts_sg2.plot(title='Detrended with Savitzky-Golay filter')
            ax.plot(time,signal_noise,label='target signal')
            ax.legend()
            pyleo.showfig(fig)
            pyleo.closefig(fig)

        '''
        new = self.copy()
        v_mod = tsutils.detrend(self.value, x=self.time, method=method, **kwargs)
        new.value = v_mod
        return new

    def spectral(self, method='lomb_scargle', freq_method='log', freq_kwargs=None, settings=None, label=None, scalogram=None, verbose=False):
        ''' Perform spectral analysis on the timeseries

        Parameters
        ----------

        method : str
            {'wwz', 'mtm', 'lomb_scargle', 'welch', 'periodogram'}

        freq_method : str
            {'log','scale', 'nfft', 'lomb_scargle', 'welch'}

        freq_kwargs : dict
            Arguments for frequency vector

        settings : dict
            Arguments for the specific spectral method

        label : str
            Label for the PSD object

        scalogram : pyleoclim.core.ui.Series.Scalogram
            The return of the wavelet analysis; effective only when the method is 'wwz'

        verbose : bool
            If True, will print warning messages if there is any

        Returns
        -------

        psd : pyleoclim.PSD
            A :mod:`pyleoclim.PSD` object

        See also
        --------
        pyleoclim.utils.spectral.mtm : Spectral analysis using the Multitaper approach

        pyleoclim.utils.spectral.lomb_scargle : Spectral analysis using the Lomb-Scargle method

        pyleoclim.utils.spectral.welch: Spectral analysis using the Welch segement approach

        pyleoclim.utils.spectral.periodogram: Spectral anaysis using the basic Fourier transform

        pyleoclim.utils.spectral.wwz_psd : Spectral analysis using the Wavelet Weighted Z transform

        pyleoclim.utils.wavelet.make_freq_vector : Functions to create the frequency vector

        pyleoclim.utils.tsutils.detrend : Detrending function

        pyleoclim.core.ui.PSD : PSD object

        pyleoclim.core.ui.MultiplePSD : Multiple PSD object


        Examples
        --------

        Calculate the spectrum of SOI using the various methods and compute significance

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts = pyleo.Series(time=time, value=value, time_name='Year C.E', value_name='SOI', label='SOI')
            # Standardize the time series
            ts_std = ts.standardize()

        - Lomb-Scargle

        .. ipython:: python
            :okwarning:
            :okexcept:    

            psd_ls = ts_std.spectral(method='lomb_scargle')
            psd_ls_signif = psd_ls.signif_test(number=20) #in practice, need more AR1 simulations
            @savefig spec_ls.png
            fig, ax = psd_ls_signif.plot(title='PSD using Lomb-Scargle method')
            pyleo.closefig(fig)

        We may pass in method-specific arguments via "settings", which is a dictionary.
        For instance, to adjust the number of overlapping segment for Lomb-Scargle, we may specify the method-specific argument "n50";
        to adjust the frequency vector, we may modify the "freq_method" or modify the method-specific argument "freq".

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import numpy as np
            psd_LS_n50 = ts_std.spectral(method='lomb_scargle', settings={'n50': 4})  # c=1e-2 yields lower frequency resolution
            psd_LS_freq = ts_std.spectral(method='lomb_scargle', settings={'freq': np.linspace(1/20, 1/0.2, 51)})
            psd_LS_LS = ts_std.spectral(method='lomb_scargle', freq_method='lomb_scargle')  # with frequency vector generated using REDFIT method
            fig, ax = psd_LS_n50.plot(
                title='PSD using Lomb-Scargle method with 4 overlapping segments',
                label='settings={"n50": 4}')
            psd_ls.plot(ax=ax, label='settings={"n50": 3}', marker='o')
            @savefig spec_ls_n50.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)

            fig, ax = psd_LS_freq.plot(
                title='PSD using Lomb-Scargle method with differnt frequency vectors',
                label='freq=np.linspace(1/20, 1/0.2, 51)', marker='o')
            psd_ls.plot(ax=ax, label='freq_method="log"', marker='o')
            @savefig spec_ls_freq.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)

        You may notice the differences in the PSD curves regarding smoothness and the locations of the analyzed period points.

        For other method-specific arguments, please look up the specific methods in the "See also" section.

        - WWZ

        .. ipython:: python
            :okwarning:
            :okexcept:    

            psd_wwz = ts_std.spectral(method='wwz')  # wwz is the default method
            psd_wwz_signif = psd_wwz.signif_test(number=1)  # significance test; for real work, should use number=200 or even larger
            @savefig spec_wwz.png
            fig, ax = psd_wwz_signif.plot(title='PSD using WWZ method')
            pyleo.closefig(fig)

        We may take advantage of a pre-calculated scalogram using WWZ to accelerate the spectral analysis
        (although note that the default parameters for spectral and wavelet analysis using WWZ are different):

        .. ipython:: python
            :okwarning:
            :okexcept:    

            scal_wwz = ts_std.wavelet(method='wwz')  # wwz is the default method
            psd_wwz_fast = ts_std.spectral(method='wwz', scalogram=scal_wwz)
            @savefig spec_wwz_fast.png
            fig, ax = psd_wwz_fast.plot(title='PSD using WWZ method w/ pre-calculated scalogram')
            pyleo.closefig(fig)

        - Periodogram

        .. ipython:: python
            :okwarning:
            :okexcept:    

            ts_interp = ts_std.interp()
            psd_perio = ts_interp.spectral(method='periodogram')
            psd_perio_signif = psd_perio.signif_test(number=20) #in practice, need more AR1 simulations
            @savefig spec_perio.png
            fig, ax = psd_perio_signif.plot(title='PSD using Periodogram method')
            pyleo.closefig(fig)

        - Welch

        .. ipython:: python
            :okwarning:
            :okexcept:    

            ts_interp = ts_std.interp()
            psd_welch = ts_interp.spectral(method='welch')
            psd_welch_signif = psd_welch.signif_test(number=20) #in practice, need more AR1 simulations
            @savefig spec_welch.png
            fig, ax = psd_welch_signif.plot(title='PSD using Welch method')
            pyleo.closefig(fig)

        - MTM

        .. ipython:: python
            :okwarning:
            :okexcept:    

            ts_interp = ts_std.interp()
            psd_mtm = ts_interp.spectral(method='mtm')
            psd_mtm_signif = psd_mtm.signif_test(number=20) #in practice, need more AR1 simulations
            @savefig spec_mtm.png
            fig, ax = psd_mtm_signif.plot(title='PSD using MTM method')
            pyleo.closefig(fig)

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
        freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
        freq = waveutils.make_freq_vector(self.time, method=freq_method, **freq_kwargs)

        args['wwz'] = {'freq': freq}
        args['mtm'] = {}
        args['lomb_scargle'] = {'freq': freq}
        args['welch'] = {}
        args['periodogram'] = {}
        args[method].update(settings)

        if method == 'wwz' and scalogram is not None:
            args['wwz'].update(
                {
                    'wwa': scalogram.amplitude,
                    'wwz_Neffs': scalogram.wwz_Neffs,
                    'wwz_freq': scalogram.frequency,
                }
            )

        spec_res = spec_func[method](self.value, self.time, **args[method])
        if type(spec_res) is dict:
            spec_res = dict2namedtuple(spec_res)

        if label is None:
            label = self.label

        if method == 'wwz' and scalogram is not None:
            args['wwz'].pop('wwa')
            args['wwz'].pop('wwz_Neffs')
            args['wwz'].pop('wwz_freq')

        psd = PSD(
            frequency=spec_res.freq,
            amplitude=spec_res.psd,
            label=label,
            timeseries=self,
            spec_method=method,
            spec_args=args[method]
        )

        return psd

    def wavelet(self, method='wwz', settings=None, freq_method='log', ntau=None, freq_kwargs=None, verbose=False):
        ''' Perform wavelet analysis on the timeseries

        cwt wavelets documented on https://pywavelets.readthedocs.io/en/latest/ref/cwt.html

        Parameters
        ----------

        method : {wwz, cwt}
            Whether to use the wwz method for unevenly spaced timeseries or traditional cwt (from pywavelets)

        freq_method : str
            {'log', 'scale', 'nfft', 'lomb_scargle', 'welch'}

        freq_kwargs : dict
            Arguments for frequency vector

        ntau : int
            The length of the time shift points that determins the temporal resolution of the result.
            If None, it will be either the length of the input time axis, or at most 50.

        settings : dict
            Arguments for the specific spectral method

        verbose : bool
            If True, will print warning messages if there is any

        Returns
        -------

        scal : Series.Scalogram

        See also
        --------

        pyleoclim.utils.wavelet.wwz : wwz function

        pyleoclim.utils.wavelet.cwt : cwt function

        pyleoclim.utils.wavelet.make_freq_vector : Functions to create the frequency vector

        pyleoclim.utils.tsutils.detrend : Detrending function

        pyleoclim.core.ui.Scalogram : Scalogram object

        pyleoclim.core.ui.MultipleScalogram : Multiple Scalogram object

        Examples
        --------

        Wavelet analysis on the SOI record.

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts = pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')
            # WWZ
            scal = ts.wavelet()
            scal_signif = scal.signif_test(number=1)  # for real work, should use number=200 or even larger
            @savefig spec_mtm.png
            fig, ax = scal_signif.plot()
            pyleo.closefig(fig)

        '''
        if not verbose:
            warnings.simplefilter('ignore')

        settings = {} if settings is None else settings.copy()
        wave_func = {
            'wwz': waveutils.wwz
            # 'cwt': waveutils.cwt,
        }

        if method == 'cwt' and 'freq' in settings.keys():
            scales=1/np.array(settings['freq'])
            settings.update({'scales':scales})
            del settings['freq']

        freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
        freq = waveutils.make_freq_vector(self.time, method=freq_method, **freq_kwargs)

        args = {}

        if ntau is None:
            ntau = np.min([np.size(self.time), 50])

        tau = np.linspace(np.min(self.time), np.max(self.time), ntau)

        args['wwz'] = {'tau': tau, 'freq': freq}
        args['cwt'] = {'wavelet' : 'morl', 'scales':1/freq}


        args[method].update(settings)
        wave_res = wave_func[method](self.value, self.time, **args[method])
        if method == 'wwz':
            wwz_Neffs = wave_res.Neffs
        else:
            wwz_Neffs = None

        scal = Scalogram(
            frequency=wave_res.freq,
            time=wave_res.time,
            amplitude=wave_res.amplitude,
            coi=wave_res.coi,
            label=self.label,
            timeseries=self,
            wave_method=method,
            freq_method=freq_method,
            freq_kwargs=freq_kwargs,
            wave_args=args[method],
            wwz_Neffs=wwz_Neffs,
        )

        return scal

    def wavelet_coherence(self, target_series, method='wwz', settings=None, freq_method='log', ntau=None, tau=None, freq_kwargs=None, verbose=False):
        ''' Perform wavelet coherence analysis with the target timeseries

        Parameters
        ----------

        target_series : pyleoclim.Series
            A pyleoclim Series object on which to perform the coherence analysis

        method : {'wwz'}

        freq_method : str
            {'log','scale', 'nfft', 'lomb_scargle', 'welch'}

        freq_kwargs : dict
            Arguments for frequency vector

        tau : array
            The time shift points that determins the temporal resolution of the result.
            If None, it will be calculated using ntau.

        ntau : int
            The length of the time shift points that determins the temporal resolution of the result.
            If None, it will be either the length of the input time axis, or at most 50.

        settings : dict
            Arguments for the specific spectral method

        verbose : bool
            If True, will print warning messages if there is any

        Returns
        -------

        coh : pyleoclim.Coherence

        See also
        --------

        pyleoclim.utils.wavelet.xwt : Cross-wavelet analysis based on WWZ method

        pyleoclim.utils.wavelet.make_freq_vector : Functions to create the frequency vector

        pyleoclim.utils.tsutils.detrend : Detrending function

        pyleoclim.core.ui.Coherence : Coherence object

        Examples
        --------

        Wavelet coherence with the default arguments:

        .. ipython:: python
            :okwarning:
            :okexcept:                   

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino.csv')
            time = data['t'].values
            air = data['air'].values
            nino = data['nino'].values
            ts_air = pyleo.Series(time=time, value=air, time_name='Year (CE)')
            ts_nino = pyleo.Series(time=time, value=nino, time_name='Year (CE)')

            # without any arguments, the `tau` will be determined automatically
            coh = ts_air.wavelet_coherence(ts_nino)

            @savefig coh.png
            fig, ax = coh.plot()
            pyleo.closefig()

        We may specify `ntau` to adjust the temporal resolution of the scalogram, which will affect the time consumption of calculation and the result itself:

        .. ipython:: python
            :okwarning:
            :okexcept:

            coh_ntau = ts_air.wavelet_coherence(ts_nino, ntau=30)

            @savefig coh_ntau.png
            fig, ax = coh_ntau.plot()
            pyleo.closefig()

        We may also specify the `tau` vector explicitly:

        .. ipython:: python
            :okwarning:
            :okexcept:    

            coh_tau = ts_air.wavelet_coherence(ts_nino, tau=np.arange(1880, 2001))

            @savefig coh_tau.png
            fig, ax = coh_tau.plot()
            pyleo.closefig()

        '''
        if not verbose:
            warnings.simplefilter('ignore')

        settings = {} if settings is None else settings.copy()
        xwc_func = {
            'wwz': waveutils.xwc,
        }

        freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
        freq = waveutils.make_freq_vector(self.time, method=freq_method, **freq_kwargs)

        t1 = np.copy(self.time)
        t2 = np.copy(target_series.time)
        dt1 = np.median(np.diff(t1))
        dt2 = np.median(np.diff(t2))
        overlap = np.arange(np.max([t1[0], t2[0]]), np.min([t1[-1], t2[-1]]), np.max([dt1, dt2]))

        if ntau is None:
            ntau = np.min([np.size(overlap), 50])

        if tau is None:
            tau = np.linspace(np.min(overlap), np.max(overlap), ntau)

        args = {}
        args['wwz'] = {'tau': tau, 'freq': freq, 'verbose': verbose}
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
            freq_method=freq_method,
            freq_kwargs=freq_kwargs,
        )

        return coh

    def correlation(self, target_series, timespan=None, alpha=0.05, settings=None, common_time_kwargs=None, seed=None):
        ''' Estimates the Pearson's correlation and associated significance between two non IID time series

        The significance of the correlation is assessed using one of the following methods:

        1) 'ttest': T-test adjusted for effective sample size.
        2) 'isopersistent': AR(1) modeling of x and y.
        3) 'isospectral': phase randomization of original inputs. (default)

        The T-test is a parametric test, hence computationally cheap but can only be performed in ideal circumstances.
        The others are non-parametric, but their computational requirements scale with the number of simulations.

        The choise of significance test and associated number of Monte-Carlo simulations are passed through the settings parameter.

        Parameters
        ----------

        target_series : pyleoclim.Series
            A pyleoclim Series object

        timespan : tuple
            The time interval over which to perform the calculation

        alpha : float
            The significance level (default: 0.05)

        settings : dict
            Parameters for the correlation function, including:

            nsim : int
                the number of simulations (default: 1000)
            method : str, {'ttest','isopersistent','isospectral' (default)}
                method for significance testing

        common_time_kwargs : dict
            Parameters for the method `MultipleSeries.common_time()`. Will use interpolation by default.

        seed : float or int
            random seed for isopersistent and isospectral methods

        Returns
        -------

        corr : pyleoclim.ui.Corr
            the result object, containing

            - r : float
                correlation coefficient
            - p : float
                the p-value
            - signif : bool
                true if significant; false otherwise
                Note that signif = True if and only if p <= alpha.
            - alpha : float
                the significance level

        See also
        --------

        pyleoclim.utils.correlation.corr_sig : Correlation function

        Examples
        --------

        Correlation between the Nino3.4 index and the Deasonalized All Indian Rainfall Index

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import pandas as pd

            data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino.csv')
            t = data.iloc[:, 0]
            air = data.iloc[:, 1]
            nino = data.iloc[:, 2]
            ts_nino = pyleo.Series(time=t, value=nino)
            ts_air = pyleo.Series(time=t, value=air)

            # with `nsim=20` and default `method='isospectral'`
            # set an arbitrary randome seed to fix the result
            corr_res = ts_nino.correlation(ts_air, settings={'nsim': 20}, seed=2333)
            print(corr_res)

            # using a simple t-test
            # set an arbitrary randome seed to fix the result
            corr_res = ts_nino.correlation(ts_air, settings={'nsim': 20, 'method': 'ttest'}, seed=2333)
            print(corr_res)

            # using the method "isopersistent"
            # set an arbitrary random seed to fix the result
            corr_res = ts_nino.correlation(ts_air, settings={'nsim': 20, 'method': 'isopersistent'}, seed=2333)
            print(corr_res)
        '''

        settings = {} if settings is None else settings.copy()
        corr_args = {'alpha': alpha}
        corr_args.update(settings)

        ms = MultipleSeries([self, target_series])
        if list(self.time) != list(target_series.time):
            common_time_kwargs = {} if common_time_kwargs is None else common_time_kwargs.copy()
            ct_args = {'method': 'interp'}
            ct_args.update(common_time_kwargs)
            ms = ms.common_time(**ct_args)

        if timespan is None:
            value1 = ms.series_list[0].value
            value2 = ms.series_list[1].value
        else:
            value1 = ms.series_list[0].slice(timespan).value
            value2 = ms.series_list[1].slice(timespan).value


        if seed is not None:
            np.random.seed(seed)

        corr_res = corrutils.corr_sig(value1, value2, **corr_args)
        signif = True if corr_res['signif'] == 1 else False
        corr = Corr(corr_res['r'], corr_res['p'], signif, alpha)

        return corr

    def causality(self, target_series, method='liang', settings=None):
        ''' Perform causality analysis with the target timeseries
            The timeseries are first sorted in ascending order.         

        Parameters
        ----------

        target_series : pyleoclim.Series
            A pyleoclim Series object on which to compute causality

        method : {'liang', 'granger'}
            The causality method to use.

        settings : dict
            Parameters associated with the causality methods. Note that each method has different parameters. See individual methods for details

        Returns
        -------

        res : dict
            Dictionary containing the results of the the causality analysis. See indivudal methods for details

        See also
        --------

        pyleoclim.utils.causality.liang_causality : Liang causality

        pyleoclim.utils.causality.granger_causality : Granger causality

        Examples
        --------

        Liang causality

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import pandas as pd
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino.csv')
            t=data.iloc[:,0]
            air=data.iloc[:,1]
            nino=data.iloc[:,2]
            ts_nino=pyleo.Series(time=t,value=nino)
            ts_air=pyleo.Series(time=t,value=air)

            # plot the two timeseries
            @savefig ts_nino.png
            fig, ax = ts_nino.plot(title='NINO3 -- SST Anomalies')
            pyleo.closefig(fig)

            @savefig ts_air.png
            fig, ax = ts_air.plot(title='Deasonalized All Indian Rainfall Index')
            pyleo.closefig(fig)

            # we use the specific params below in ts_nino.causality() just to make the example less heavier;
            # please drop the `settings` for real work
            caus_res = ts_nino.causality(ts_air, settings={'nsim': 2, 'signif_test': 'isopersist'})
            print(caus_res)

        Granger causality

        .. ipython:: python
            :okwarning:
            :okexcept:    

            caus_res = ts_nino.causality(ts_air, method='granger')
            print(caus_res)

        '''
        
        # Sort both timeseries 
    
        sorted_self   = self.sort(verbose=True) 
        sorted_target = target_series.sort(verbose=True)        
        
        settings = {} if settings is None else settings.copy()
        spec_func={
            'liang':causalutils.liang_causality,
            'granger':causalutils.granger_causality}
        args = {}
        args['liang'] = {}
        args['granger'] = {}
        args[method].update(settings)
        causal_res = spec_func[method](sorted_self.value, sorted_target.value, **args[method])
        return causal_res

    def surrogates(self, method='ar1', number=1, length=None, seed=None, settings=None):
        ''' Generate surrogates with increasing time axis

        Parameters
        ----------

        method : {ar1}
            Uses an AR1 model to generate surrogates of the timeseries

        number : int
            The number of surrogates to generate

        length : int
            Lenght of the series

        seed : int
            Control seed option for reproducibility

        settings : dict
            Parameters for surogate generator. See individual methods for details.

        Returns
        -------
        surr : pyleoclim SurrogateSeries

        See also
        --------

        pyleoclim.utils.tsmodel.ar1_sim : AR1 simulator
        '''
        settings = {} if settings is None else settings.copy()
        surrogate_func = {
            'ar1': tsmodel.ar1_sim,
        }
        args = {}
        args['ar1'] = {'t': self.time}
        args[method].update(settings)

        if seed is not None:
            np.random.seed(seed)

        surr_res = surrogate_func[method](self.value, number, **args[method])
        if len(np.shape(surr_res)) == 1:
            surr_res = surr_res[:, np.newaxis]

        s_list = []
        for s in surr_res.T:
            s_tmp = Series(time=self.time, value=s, time_name=self.time_name, time_unit=self.time_unit, value_name=self.value_name, value_unit=self.value_unit)
            s_list.append(s_tmp)

        surr = SurrogateSeries(series_list=s_list, surrogate_method=method, surrogate_args=args[method])

        return surr

    def outliers(self, auto=True, remove=True, fig_outliers=True,fig_knee=True,
                 plot_outliers_kwargs=None,plot_knee_kwargs=None,figsize=[10,4],
                 saveknee_settings=None,saveoutliers_settings=None, mute=False):
        '''
        Detects outliers in a timeseries and removes if specified

        Parameters
        ----------

        auto : boolean
            True by default, detects knee in the plot automatically
        remove : boolean
            True by default, removes all outlier points if detected
        fig_knee  : boolean
            True by default, plots knee plot if true
        fig_outliers : boolean
            True by degault, plots outliers if true
        save_knee : dict
            default parameters from matplotlib savefig None by default
        save_outliers : dict
            default parameters from matplotlib savefig None by default
        plot_knee_kwargs : dict
            arguments for the knee plot
        plot_outliers_kwargs : dict
            arguments for the outliers plot
        figsize : list
            by default [10,4]
        mute : {True,False}
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)

        Returns
        -------
        new : Series
            Time series with outliers removed if they exist

        See also
        --------

        pyleoclim.utils.tsutils.remove_outliers : remove outliers function

        pyleoclim.utils.plotting.plot_xy : basic x-y plot

        pyleoclim.utils.plotting.plot_scatter_xy : Scatter plot on top of a line plot

        '''
        new = self.copy()

        #outlier_indices,fig1,ax1,fig2,ax2 = tsutils.detect_outliers(self.time, self.value, auto=auto, plot_knee=fig_knee,plot_outliers=fig_outliers,\
        #                                                   figsize=figsize,save_knee=save_knee,save_outliers=save_outliers,plot_outliers_kwargs=plot_outliers_kwargs,plot_knee_kwargs=plot_knee_kwargs)
        outlier_indices = tsutils.detect_outliers(
            self.time, self.value, auto=auto, plot_knee=fig_knee,plot_outliers=fig_outliers,
            figsize=figsize,saveknee_settings=saveknee_settings,saveoutliers_settings=saveoutliers_settings,
            plot_outliers_kwargs=plot_outliers_kwargs,plot_knee_kwargs=plot_knee_kwargs, mute=mute
        )
        outlier_indices = np.asarray(outlier_indices)
        if remove == True:
            new = self.copy()
            ys = np.delete(self.value, outlier_indices)
            t = np.delete(self.time, outlier_indices)
            new.value = ys
            new.time = t

        return new

    def interp(self, method='linear', **kwargs):
        '''Interpolate a Series object onto a new time axis

        Parameters
        ----------

        method : {‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’}
            where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. Default is ‘linear’.

        kwargs :
            Arguments specific to each interpolation function. See pyleoclim.utils.tsutils.interp for details

        Returns
        -------

        new : pyleoclim.Series
            An interpolated Series object

        See also
        --------

        pyleoclim.utils.tsutils.interp : interpolation function

        '''
        new = self.copy()
        ti, vi = tsutils.interp(self.time,self.value,interp_type=method,**kwargs)
        new.time = ti
        new.value = vi
        return new

    def gkernel(self, step_type='median', **kwargs):
        ''' Coarse-grain a Series object via a Gaussian kernel.

        Parameters
        ----------
        step_type : str
            type of timestep: 'mean', 'median', or 'max' of the time increments
        kwargs :
            Arguments for kernel function. See pyleoclim.utils.tsutils.gkernel for details
        Returns
        -------
        new : pyleoclim.Series
            The coarse-grained Series object
        See also
        --------
        pyleoclim.utils.tsutils.gkernel : application of a Gaussian kernel
        '''

        new=self.copy()

        ti, vi = tsutils.gkernel(self.time, self.value, **kwargs) # apply kernel
        new.time = ti
        new.value = vi
        return new

    def bin(self,**kwargs):
        '''Bin values in a time series

        Parameters
        ----------

        kwargs :
            Arguments for binning function. See pyleoclim.utils.tsutils.bin for details

        Returns
        -------

        new : pyleoclim.Series
            An binned Series object

        See also
        --------

        pyleoclim.utils.tsutils.bin : bin the time series into evenly-spaced bins

        '''
        new=self.copy()
        res_dict = tsutils.bin(self.time,self.value,**kwargs)
        new.time = res_dict['bins']
        new.value = res_dict['binned_values']
        return new

class PSD:
    '''PSD object obtained from spectral analysis.

    See examples in pyleoclim.core.ui.Series.spectral to see how to create and manipulate these objects

    See also
    --------

    pyleoclim.core.ui.Series.spectral : spectral analysis

    '''
    def __init__(self, frequency, amplitude, label=None, timeseries=None, plot_kwargs=None,
                 spec_method=None, spec_args=None, signif_qs=None, signif_method=None, period_unit=None,
                 beta_est_res=None):
        self.frequency = np.array(frequency)
        self.amplitude = np.array(amplitude)
        self.label = label
        self.timeseries = timeseries
        self.spec_method = spec_method
        self.spec_args = spec_args
        self.signif_qs = signif_qs
        self.signif_method = signif_method
        self.plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        self.beta_est_res = beta_est_res

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

    def signif_test(self, number=None, method='ar1', seed=None, qs=[0.95],
                    settings=None, scalogram = None):
        '''


        Parameters
        ----------
        number : int, optional
            Number of surrogate series to generate for significance testing. The default is None.
        method : {ar1}, optional
            Method to generate surrogates. The default is 'ar1'.
        seed : int, optional
            Option to set the seed for reproducibility. The default is None.
        qs : list, optional
            Singificance levels to return. The default is [0.95].
        settings : dict, optional
            Parameters. The default is None.
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

            pyleo.showfig(fig)

            pyleo.closefig(fig)

        '''
        
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

        res = waveutils.beta_estimation(self.amplitude, self.frequency, fmin=fmin, fmax=fmax, logf_binning_step=logf_binning_step, verbose=verbose)
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
             xlim=None, ylim=None, figsize=[10, 4], savefig_settings=None, ax=None, mute=False,
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
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax The default is False.
            (going to be deprecated)
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

        pyleoclim.core.ui.Series.spectral : spectral analysis

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
                'ar1': 'AR(1)',
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
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            return fig, ax
        else:
            return ax

class Scalogram:
    def __init__(self, frequency, time, amplitude, coi=None, label=None, Neff=3, wwz_Neffs=None, timeseries=None,
                 wave_method=None, wave_args=None, signif_qs=None, signif_method=None, freq_method=None, freq_kwargs=None,
                 period_unit=None, time_label=None, signif_scals=None):
        '''
        Parameters
        ----------
            frequency : array
                the frequency axis
            time : array
                the time axis
            amplitude : array
                the amplitude at each (frequency, time) point;
                note the dimension is assumed to be (frequency, time)
            coi : array
                Cone of influence
            label : str
                Label for the Series
            Neff : int
                the threshold of the number of effective samples
            wwz_Neffs : array
                the matrix of effective number of points in the time-scale coordinates obtained from wwz
            timeseries : pyleoclim.Series
                A copy of the timeseries for which the scalogram was obtained
            wave_method: str
                The method used to obtain the scalogram
            wave_args: dict
                The parameters values of the wavelet method
            signif_qs : dict
                The significance limits
            signif_method: str
                The method used to obtain the significance level
            freq_method: str
                The method used to obtain the frequency vector
            freq_kwargs: dict
                Arguments for the frequency vector
            period_unit: str
                Units for the period axis
            time_label: str
                Label for the time axis
            signif_scals: pyleoclim.MultipleScalogram
                A list of the scalogram from the AR1 MC significance testing. Useful when obtaining a PSD. 
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
        self.freq_method = freq_method
        self.freq_kwargs = freq_kwargs
        self.signif_scals = signif_scals
        #if wave_method == 'wwz':
        self.wwz_Neffs = wwz_Neffs

        if period_unit is not None:
            self.period_unit = period_unit
        elif timeseries is not None:
            self.period_unit = infer_period_unit_from_time_unit(timeseries.time_unit)
        else:
            self.period_unit = None

        if time_label is not None:
            self.time_label = time_label
        elif timeseries is not None:
            if timeseries.time_unit is not None:
                self.time_label = f'{timeseries.time_name} [{timeseries.time_unit}]'
            else:
                self.time_label = f'{timeseries.time_name}'
        else:
            self.time_label = None


    def copy(self):
        '''Copy object
        '''
        return deepcopy(self)

    def __str__(self):
        table = {
            'Frequency': self.frequency,
            'Time': self.time,
            'Amplitude': self.amplitude,
        }

        msg = print(tabulate(table, headers='keys'))
        return f'Dimension: {np.size(self.frequency)} x {np.size(self.time)}'

    def plot(self, in_period=True, xlabel=None, ylabel=None, title=None,
             ylim=None, xlim=None, yticks=None, figsize=[10, 8], mute=False,
             signif_clr='white', signif_linestyles='-', signif_linewidths=1,
             contourf_style={}, cbar_style={}, savefig_settings={}, ax=None):
        '''Plot the scalogram

        Parameters
        ----------
        in_period : bool, optional
            Plot the in period instead of frequency space. The default is True.
        xlabel : str, optional
            Label for the x-axis. The default is None.
        ylabel : str, optional
            Label for the y-axis. The default is None.
        title : str, optional
            Title for the figure. The default is None.
        ylim : list, optional
            Limits for the y-axis. The default is None.
        xlim : list, optional
            Limits for the x-axis. The default is None.
        yticks : list, optional
            yticks label. The default is None.
        figsize : list, optional
            Figure size The default is [10, 8].
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax The default is False.
            (going to be deprecated)
        signif_clr : str, optional
            Color of the singificance line. The default is 'white'.
        signif_linestyles : str, optional
            Linestyle of the significance line. The default is '-'.
        signif_linewidths : float, optional
            Width for the significance line. The default is 1.
        contourf_style : dict, optional
            Arguments for the contour plot. The default is {}.
        cbar_style : dict, optional
            Arguments for the colarbar. The default is {}.
        savefig_settings : dict, optional
            saving options for the figure. The default is {}.
        ax : ax, optional
            Matplotlib Axis on which to return the figure. The default is None.

        Returns
        -------
        fig, ax

        See also
        --------

        pyleoclim.core.ui.Series.wavelet : Wavelet analysis


        '''
        contourf_args = {'cmap': 'magma', 'origin': 'lower', 'levels': 11}
        contourf_args.update(contourf_style)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if in_period:
            y_axis = 1/self.frequency
            if ylabel is None:
                ylabel = f'Period [{self.period_unit}]' if self.period_unit is not None else 'Period'

            if yticks is None:
                yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])
                mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
                yticks = yticks_default[mask]
        else:
            y_axis = self.frequency
            if ylabel is None:
                ylabel = f'Frequency [1/{self.period_unit}]' if self.period_unit is not None else 'Frequency'

        cont = ax.contourf(self.time, y_axis, self.amplitude.T, **contourf_args)
        ax.set_yscale('log')

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

        if xlabel is None:
            xlabel = self.time_label

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        ax.set_ylim(ylim)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def signif_test(self, number=None, method='ar1', seed=None, qs=[0.95],
                    settings=None, export_scal = False):
        '''Significance test for wavelet analysis

        Parameters
        ----------
        number : int, optional
            Number of surrogates to generate for significance analysis. The default is 200.
        method : {'ar1'}, optional
            Method to use to generate the surrogates. The default is 'ar1'.
        seed : int, optional
            Set the seed for the random number generator. Useful for reproducibility The default is None.
        qs : list, optional
            Significane level to consider. The default is [0.95].
        settings : dict, optional
            Parameters for the model. The default is None.
        export_scal : bool
            Whether or not to export the scalograms used in the noise realizations. Note: The scalograms used for wavelet analysis are slightly different
            than those used for spectral analysis (different decay constant). As such, this functionality should be used only to expedite exploratory analysis.

        Raises
        ------
        ValueError
            qs should be a list with at least one value.

        Returns
        -------
        new : pyleoclim.Scalogram
            A new Scalogram object with the significance level

        Examples
        --------

        Generating scalogram, running significance tests, and saving the output for future use in generating psd objects or in summary_plot()

        .. ipython:: python
            :okwarning:
            :okexcept:    
            
            import pyleoclim as pyleo
            import pandas as pd
            ts=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/soi_data.csv',skiprows = 1)
            series = pyleo.Series(time = ts['Year'],value = ts['Value'], time_name = 'Years', time_unit = 'AD')

            #By setting export_scal to True, the noise realizations used to generate the significance test will be saved. These come in handy for generating summary plots and for running significance tests on spectral objects.
            scalogram = series.wavelet().signif_test(number=2, export_scal=True)

        See also
        --------

        pyleoclim.core.ui.Series.wavelet : wavelet analysis

        '''

        if hasattr(self,'signif_scals'):
            signif_scals = self.signif_scals
        
        #Allow for a few different configurations of passed number of signif tests, default behavior is to set number = 200
        if number is None and signif_scals is not None:
            number = len(signif_scals.scalogram_list)
        elif number is None and signif_scals is None:
            number = 200
        elif number == 0:
            return self

        new = self.copy()

        if signif_scals:
            scalogram_list = signif_scals.scalogram_list
            #If signif_scals already in scalogram object are more than those requested for significance testing, use as many of them as required
            if len(scalogram_list) > number:
                surr_scal = MultipleScalogram(scalogram_list=scalogram_list[:number])
            #If number is the same as the length of signif_scals, just use signif_scals
            elif len(scalogram_list) == number:
                surr_scal = signif_scals
            #If the number is more than the length of signif_scals, reuse what is available and calculate the rest
            elif len(scalogram_list) < number:
                number -= len(scalogram_list)
                surr_scal_tmp = []
                surr_scal_tmp.extend(scalogram_list)
                surr = self.timeseries.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )
                surr_scal_tmp.extend(surr.wavelet(method=self.wave_method, settings=self.wave_args,).scalogram_list)
                surr_scal = MultipleScalogram(scalogram_list=surr_scal_tmp)
        else:
            surr = self.timeseries.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )
            surr_scal = surr.wavelet(method=self.wave_method, settings=self.wave_args,)

        if len(qs) > 1:
            raise ValueError('qs should be a list with size 1!')

        new.signif_qs = surr_scal.quantiles(qs=qs)
        new.signif_method = method
        
        if export_scal == True:
            new.signif_scals = surr_scal

        return new


class Coherence:
    '''Coherence object

    See also
    --------

    pyleoclim.core.ui.Series.wavelet_coherence : Wavelet coherence

    '''
    def __init__(self, frequency, time, coherence, phase, coi=None,
                 timeseries1=None, timeseries2=None, signif_qs=None, signif_method=None,
                 freq_method=None, freq_kwargs=None, Neff=3, period_unit=None, time_label=None):
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
        self.freq_method = freq_method
        self.freq_kwargs = freq_kwargs

        if period_unit is not None:
            self.period_unit = period_unit
        elif timeseries1 is not None:
            self.period_unit = infer_period_unit_from_time_unit(timeseries1.time_unit)
        elif timeseries2 is not None:
            self.period_unit = infer_period_unit_from_time_unit(timeseries2.time_unit)
        else:
            self.period_unit = None

        if time_label is not None:
            self.time_label = time_label
        elif timeseries1 is not None:
            if timeseries1.time_unit is not None:
                self.time_label = f'{timeseries1.time_name} [{timeseries1.time_unit}]'
            else:
                self.time_label = f'{timeseries1.time_name}'
        elif timeseries2 is not None:
            if timeseries2.time_unit is not None:
                self.time_label = f'{timeseries2.time_name} [{timeseries2.time_unit}]'
            else:
                self.time_label = f'{timeseries2.time_name}'
        else:
            self.time_label = None

    def copy(self):
        '''Copy object
        '''
        return deepcopy(self)

    def plot(self, xlabel=None, ylabel=None, title=None, figsize=[10, 8],
             ylim=None, xlim=None, in_period=True, yticks=None, mute=False,
             contourf_style={}, phase_style={}, cbar_style={}, savefig_settings={}, ax=None,
             signif_clr='white', signif_linestyles='-', signif_linewidths=1,
             under_clr='ivory', over_clr='black', bad_clr='dimgray'):
        '''Plot the cross-wavelet results

        Parameters
        ----------
        xlabel : str, optional
            x-axis label. The default is None.
        ylabel : str, optional
            y-axis label. The default is None.
        title : str, optional
            Title of the plot. The default is None.
        figsize : list, optional
            Figure size. The default is [10, 8].
        ylim : list, optional
            y-axis limits. The default is None.
        xlim : list, optional
            x-axis limits. The default is None.
        in_period : bool, optional
            Plots periods instead of frequencies The default is True.
        yticks : list, optional
            y-ticks label. The default is None.
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax The default is False. The default is False.
            (going to be deprecated)
        contourf_style : dict, optional
            Arguments for the contour plot. The default is {}.
        phase_style : dict, optional
            Arguments for the phase arrows. The default is {}. It includes:
            - 'pt': the default threshold above which phase arrows will be plotted
            - 'skip_x': the number of points to skip between phase arrows along the x-axis
            - 'skip_y':  the number of points to skip between phase arrows along the y-axis
            - 'scale': number of data units per arrow length unit (see matplotlib.pyplot.quiver)
            - 'width': shaft width in arrow units (see matplotlib.pyplot.quiver)
        cbar_style : dict, optional
            Arguments for the color bar. The default is {}.
        savefig_settings : dict, optional
            The default is {}.
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}
        ax : ax, optional
            Matplotlib axis on which to return the figure. The default is None.
        signif_clr : str, optional
            Color of the singificance line. The default is 'white'.
        signif_linestyles : str, optional
            Style of the significance line. The default is '-'.
        signif_linewidths : float, optional
            Width of the significance line. The default is 1.
        under_clr : str, optional
            Color for under 0. The default is 'ivory'.
        over_clr : str, optional
            Color for over 1. The default is 'black'.
        bad_clr : str, optional
            Color for missing values. The default is 'dimgray'.

        Returns
        -------
        fig, ax

        See also
        --------

        pyleoclim.core.ui.Series.wavelet_coherence
        matplotlib.pyplot.quiver

        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # handling NaNs
        mask_freq = []
        for i in range(np.size(self.frequency)):
            if all(np.isnan(self.coherence[:, i])):
                mask_freq.append(False)
            else:
                mask_freq.append(True)

        if in_period:
            y_axis = 1/self.frequency[mask_freq]
            if ylabel is None:
                ylabel = f'Period [{self.period_unit}]' if self.period_unit is not None else 'Period'

            if yticks is None:
                yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])
                mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
                yticks = yticks_default[mask]
        else:
            y_axis = self.frequency[mask_freq]
            if ylabel is None:
                ylabel = f'Frequency [1/{self.period_unit}]' if self.period_unit is not None else 'Frequency'

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

        cont = ax.contourf(self.time, y_axis, self.coherence[:, mask_freq].T, **contourf_args)

        # plot significance levels
        if self.signif_qs is not None:
            signif_method_label = {
                'ar1': 'AR(1)',
            }
            signif_coh = self.signif_qs.scalogram_list[0]
            signif_boundary = self.coherence[:, mask_freq].T / signif_coh.amplitude[:, mask_freq].T
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
        ax.set_yscale('log')
        ax.plot(self.time, self.coi, 'k--')

        if ylim is None:
            ylim = [np.min(y_axis), np.min([np.max(y_axis), np.max(self.coi)])]

        ax.fill_between(self.time, self.coi, np.max(self.coi), color='white', alpha=0.5)

        if yticks is not None:
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

        if xlabel is None:
            xlabel = self.time_label

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # plot phase
        skip_x = np.max([int(np.size(self.time)//20), 1])
        skip_y = np.max([int(np.size(y_axis)//20), 1])
        phase_args = {'pt': 0.5, 'skip_x': skip_x, 'skip_y': skip_y, 'scale': 30, 'width': 0.004}
        phase_args.update(phase_style)

        pt = phase_args['pt']
        skip_x = phase_args['skip_x']
        skip_y = phase_args['skip_y']
        scale = phase_args['scale']
        width = phase_args['width']

        phase = np.copy(self.phase)[:, mask_freq]

        if self.signif_qs is None:
            phase[self.coherence[:, mask_freq] < pt] = np.nan
        else:
            phase[signif_boundary.T < 1] = np.nan

        X, Y = np.meshgrid(self.time, 1/self.frequency[mask_freq])
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
                plotting.savefig(fig, settings=savefig_settings)
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def signif_test(self, number=200, method='ar1', seed=None, qs=[0.95], settings=None, mute_pbar=False):
        '''Significance testing

        Parameters
        ----------
        number : int, optional
            Number of surrogate series to create for significance testing. The default is 200.
        method : {'ar1'}, optional
            Method through which to generate the surrogate series. The default is 'ar1'.
        seed : int, optional
            Fixes the seed for the random number generator. Useful for reproducibility. The default is None.
        qs : list, optional
            Significanc level to return. The default is [0.95].
        settings : dict, optional
            Parameters for surrogate model. The default is None.
        mute_pbar : bool, optional
            Mute the progress bar. The default is False.

        Returns
        -------
        new : pyleoclim.Coherence
            Coherence with significance level

        See also
        --------

        pyleoclim.core.ui.Series.wavelet_coherence : Wavelet coherence
        '''

        if number == 0:
            return self

        new = self.copy()
        surr1 = self.timeseries1.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )
        surr2 = self.timeseries2.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )

        cohs = []
        for i in tqdm(range(number), desc='Performing wavelet coherence on surrogate pairs', total=number, disable=mute_pbar):
            coh_tmp = surr1.series_list[i].wavelet_coherence(surr2.series_list[i], settings={'tau': self.time, 'freq': self.frequency})
            cohs.append(coh_tmp.coherence)

        cohs = np.array(cohs)

        ne, nf, nt = np.shape(cohs)

        coh_qs = np.ndarray(shape=(np.size(qs), nf, nt))
        for i in range(nf):
            for j in range(nt):
                coh_qs[:,i,j] = mquantiles(cohs[:,i,j], qs)

        scal_list = []
        for i, amp in enumerate(coh_qs):
            scal_tmp = Scalogram(
                    frequency=self.frequency, time=self.time, amplitude=amp, coi=self.coi,
                    freq_method=self.freq_method, freq_kwargs=self.freq_kwargs, label=f'{qs[i]*100:g}%',
                )
            scal_list.append(scal_tmp)

        new.signif_qs = MultipleScalogram(scalogram_list=scal_list)
        new.signif_method = method

        return new

class MultipleSeries:
    '''MultipleSeries object.

    This object handles a collection of the type Series and can be created from a list of such objects.
    MultipleSeries should be used when the need to run analysis on multiple records arises, such as running principal component analysis.
    Some of the methods automatically refocus the time axis prior to analysis to ensure that the analysis is run over the same time period.

    Parameters
    ----------

    series_list : list
        a list of pyleoclim.Series objects

    time_unit : str
        The target time unit for every series in the list.
        If None, then no conversion will be applied;
        Otherwise, the time unit of every series in the list will be converted to the target.

   name : str
        name of the collection of timeseries (e.g. 'PAGES 2k ice cores')

    Examples
    --------
    .. ipython:: python
        :okwarning:
        :okexcept:    

        import pyleoclim as pyleo
        import pandas as pd
        data = pd.read_csv(
            'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
            skiprows=0, header=1
        )
        time = data.iloc[:,1]
        value = data.iloc[:,2]
        ts1 = pyleo.Series(time=time, value=value, time_unit='years')
        ts2 = pyleo.Series(time=time, value=value, time_unit='years')
        ms = pyleo.MultipleSeries([ts1, ts2], name = 'SOI x2')
    '''
    def __init__(self, series_list, time_unit=None, name=None):
        self.series_list = series_list
        self.time_unit = time_unit
        self.name = name

        if self.time_unit is not None:
            new_ts_list = []
            for ts in self.series_list:
                new_ts = ts.convert_time_unit(time_unit=self.time_unit)
                new_ts_list.append(new_ts)

            self.series_list = new_ts_list

    def convert_time_unit(self, time_unit='years'):
        ''' Convert the time unit of the timeseries

        Parameters
        ----------

        time_unit : str
            the target time unit, possible input:
            {
                'year', 'years', 'yr', 'yrs',
                'y BP', 'yr BP', 'yrs BP', 'year BP', 'years BP',
                'ky BP', 'kyr BP', 'kyrs BP', 'ka BP', 'ka',
                'my BP', 'myr BP', 'myrs BP', 'ma BP', 'ma',
            }

        Examples
        --------
        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv(
                'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
                skiprows=0, header=1
            )
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts1 = pyleo.Series(time=time, value=value, time_unit='years')
            ts2 = pyleo.Series(time=time, value=value, time_unit='years')
            ms = pyleo.MultipleSeries([ts1, ts2])
            new_ms = ms.convert_time_unit('yr BP')
            print('Original timeseries:')
            print('time unit:', ms.time_unit)
            print()
            print('Converted timeseries:')
            print('time unit:', new_ms.time_unit)
        '''

        new_ms = self.copy()
        new_ts_list = []
        for ts in self.series_list:
            new_ts = ts.convert_time_unit(time_unit=time_unit)
            new_ts_list.append(new_ts)

        new_ms.time_unit = time_unit
        new_ms.series_list = new_ts_list
        return new_ms

    def filter(self, cutoff_freq=None, cutoff_scale=None, method='butterworth', **kwargs):
        ''' Filtering the timeseries in the MultipleSeries object

        Parameters
        ----------

        method : str, {'savitzky-golay', 'butterworth', 'firwin'}
            the filtering method
            - 'butterworth': the Butterworth method (default)
            - 'savitzky-golay': the Savitzky-Golay method
            - 'firwin': FIR filter design using the window method, with default window as Hamming
            - 'lanczos': lowpass filter via Lanczos resampling

        cutoff_freq : float or list
            The cutoff frequency only works with the Butterworth method.
            If a float, it is interpreted as a low-frequency cutoff (lowpass).
            If a list,  it is interpreted as a frequency band (f1, f2), with f1 < f2 (bandpass).

        cutoff_scale : float or list
            cutoff_freq = 1 / cutoff_scale
            The cutoff scale only works with the Butterworth method and when cutoff_freq is None.
            If a float, it is interpreted as a low-frequency (high-scale) cutoff (lowpass).
            If a list,  it is interpreted as a frequency band (f1, f2), with f1 < f2 (bandpass).

        kwargs : dict
            a dictionary of the keyword arguments for the filtering method,
            see `pyleoclim.utils.filter.savitzky_golay`, `pyleoclim.utils.filter.butterworth`, and `pyleoclim.utils.filter.firwin` for the details

        Returns
        -------

        ms : pyleoclim.MultipleSeries

        See also
        --------

        pyleoclim.utils.filter.butterworth : Butterworth method
        pyleoclim.utils.filter.savitzky_golay : Savitzky-Golay method
        pyleoclim.utils.filter.firwin : FIR filter design using the window method
        pyleoclim.utils.filter.lanczos : lowpass filter via Lanczos resampling

        '''

        ms = self.copy()

        new_tslist = []
        for ts in self.series_list:
            new_tslist.append(ts.filter(cutoff_freq=cutoff_freq, cutoff_scale=cutoff_scale, method=method, **kwargs))

        ms.series_list = new_tslist

        return ms



    def append(self,ts):
        '''Append timeseries ts to MultipleSeries object

        Returns
        -------
        ms : pyleoclim.MultipleSeries
            The augmented object, comprising the old one plus `ts`

        '''
        ms = self.copy()
        ts_list = deepcopy(ms.series_list)
        ts_list.append(ts)
        ms = MultipleSeries(ts_list)
        return ms

    def copy(self):
        '''Copy the object
        '''
        return deepcopy(self)

    def standardize(self):
        '''Standardize each series object in a collection 

        Returns
        -------
        ms : pyleoclim.MultipleSeries
            The standardized Series

        '''
        ms=self.copy()
        for idx,item in enumerate(ms.series_list):
            s=item.copy()
            v_mod=tsutils.standardize(item.value)[0]
            s.value=v_mod
            ms.series_list[idx]=s
        return ms
    
    def grid_properties(self, step_style='median'):
        '''
        Extract grid properties (start, stop, step) of all the Series objects in 
        a collection.
        
        Parameters
        ----------
        step_style : str
            Method to obtain a representative step if x is not evenly spaced.
            Valid entries: 'median' [default], 'mean', 'mode' or 'max'
            The mode is the most frequent entry in a dataset, and may be a good choice if the timeseries
            is nearly equally spaced but for a few gaps. 
            
            Max is a conservative choice, appropriate for binning methods and Gaussian kernel coarse-graining

        Returns
        -------
        
        grid_properties : numpy array
            n x 3 array, where n is the number of series
            
        '''
        gp = np.empty((len(self.series_list),3)) # obtain grid parameters
        for idx,item in enumerate(self.series_list):
            item      = item.clean(verbose=idx==0)
            gp[idx,:] = tsutils.grid_properties(item.time, step_style=step_style)
            
        return gp

    def common_time(self, method='interp', common_step = 'max', start=None, stop = None, step=None, step_style = None, **kwargs):
        ''' Aligns the time axes of a MultipleSeries object, via binning
        interpolation., or Gaussian kernel. Alignmentis critical for workflows
        that need to assume a common time axis for the group of series under consideration.


        The common time axis is characterized by the following parameters:

        start : the latest start date of the bunch (maximun of the minima)
        stop  : the earliest stop date of the bunch (minimum of the maxima)
        step  : The representative spacing between consecutive values

        Optional arguments for binning or interpolation are those of the underling functions.

        If the time axis are retrograde, this step makes them prograde.

        Parameters
        ----------
        method :  string
            either 'bin', 'interp' [default] or 'gkernel'
        common_step : string
            Method to obtain a representative step among all Series
            Valid entries: 'median' [default], 'mean', 'mode' or 'max'
        start : float
            starting point of the common time axis [default = None]
        stop : float
            end point of the common time axis [default = None]
        step : float
            increment the common time axis  [default = None]

        if not provided, `pyleoclim` will use `grid_properties()` to determine these parameters

        step_style : 'string'
            step style to be applied from `grid_properties` [default = None]


        kwargs: dict
            keyword arguments (dictionary) of the various methods

        Returns
        -------
        ms : pyleoclim.MultipleSeries
            The MultipleSeries objects with all series aligned to the same time axis.


        See also
        --------

        pyleoclim.utils.tsutils.bin : put timeseries values into bins of equal size (possibly leaving NaNs in).
        pyleoclim.utils.tsutils.gkernel : coarse-graining using a Gaussian kernel
        pyleoclim.utils.tsutils.interp : interpolation onto a regular grid (default = linear interpolation)
        pyleoclim.utils.tsutils.grid_properties : infer grid properties

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import numpy as np
            import pyleoclim as pyleo
            import matplotlib.pyplot as plt
            from pyleoclim.utils.tsmodel import colored_noise

            # create 2 incompletely sampled series
            ns = 2 ; nt = 200; n_del = 20
            serieslist = []


            for j in range(ns):
                t = np.arange(nt)
                v = colored_noise(alpha=1, t=t)
                deleted_idx = np.random.choice(range(np.size(t)), n_del, replace=False)
                tu =  np.delete(t, deleted_idx)
                vu =  np.delete(v, deleted_idx)
                ts = pyleo.Series(time = tu, value = vu, label = 'series ' + str(j+1))
                serieslist.append(ts)

            # create MS object from the list
            ms = pyleo.MultipleSeries(serieslist)

            fig, ax = plt.subplots(2,2)
            ax = ax.flatten()
            # apply common_time with default parameters
            msc = ms.common_time()
            msc.plot(title='linear interpolation',ax=ax[0])

            # apply common_time with binning
            msc = ms.common_time(method='bin')
            msc.plot(title='Binning',ax=ax[1], legend=False)

            # apply common_time with gkernel
            msc = ms.common_time(method='gkernel')
            msc.plot(title=r'Gaussian kernel ($h=3$)',ax=ax[2],legend=False)

            # apply common_time with gkernel and a large bandwidth
            msc = ms.common_time(method='gkernel', h=11)
            msc.plot(title=r'Gaussian kernel ($h=11$)',ax=ax[3],legend=False)

            # display, save and close figure
            fig.tight_layout()
            @savefig ms_ct.png
            pyleo.showfig(fig)
            pyleo.closefig(fig)

        '''

        if step_style == None:
            if method == 'bin' or method == 'gkernel':
               step_style = 'max'
            elif  method == 'interp':
               step_style = 'mean'

        gp = self.grid_properties(step_style=step_style)        

        # define parameters for common time axis
        start = gp[:,0].max()
        stop  = gp[:,1].min()
        if start > stop:
            raise ValueError('At least one series has no common time interval with others. Please check the time axis of the series.')

        if step is None:
            if common_step == 'mean':
                step = gp[:,2].mean()
            elif common_step == 'max':
                step = gp[:,2].max()
            elif common_step == 'mode':
                step = stats.mode(gp[:,2])[0][0]
            else:
                step = np.median(gp[:,2])

        ms = self.copy()

        if method == 'bin':
            for idx,item in enumerate(self.series_list):
                ts = item.copy()
                d = tsutils.bin(ts.time, ts.value, bin_size=step, start=start, stop=stop, evenly_spaced = False, **kwargs)
                ts.time  = d['bins']
                ts.value = d['binned_values']
                ms.series_list[idx] = ts

        elif method == 'interp':
            for idx,item in enumerate(self.series_list):
                ts = item.copy()
                ti, vi = tsutils.interp(ts.time, ts.value, step=step, start=start, stop=stop,**kwargs)
                ts.time  = ti
                ts.value = vi
                ms.series_list[idx] = ts

        elif method == 'gkernel':
            for idx,item in enumerate(self.series_list):
                ts = item.copy()
                ti, vi = tsutils.gkernel(ts.time,ts.value,step=step, start=start, stop=stop, **kwargs)
                ts.time  = ti
                ts.value = vi
                ms.series_list[idx] = ts

        else:
            raise NameError('Unknown methods; no action taken')

        return ms

    def correlation(self, target=None, timespan=None, alpha=0.05, settings=None, fdr_kwargs=None, common_time_kwargs=None, mute_pbar=False, seed=None):
        ''' Calculate the correlation between a MultipleSeries and a target Series

        If the target Series is not specified, then the 1st member of MultipleSeries will be the target

        Parameters
        ----------
        target : pyleoclim.Series, optional
            A pyleoclim Series object.

        timespan : tuple
            The time interval over which to perform the calculation

        alpha : float
            The significance level (0.05 by default)

        fdr_kwargs : dict
            Parameters for the FDR function

        settings : dict
            Parameters for the correlation function, including:

            nsim : int
                the number of simulations (default: 1000)
            method : str, {'ttest','isopersistent','isospectral' (default)}
                method for significance testing

        common_time_kwargs : dict
            Parameters for the method MultipleSeries.common_time()

        seed : float or int
            random seed for isopersistent and isospectral methods

        mute_pbar : bool
            If True, the progressbar will be muted. Default is False.

        Returns
        -------

        corr : pyleoclim.ui.CorrEns
            the result object, see `pyleoclim.ui.CorrEns`

        See also
        --------

        pyleoclim.utils.correlation.corr_sig : Correlation function
        pyleoclim.utils.correlation.fdr : FDR function
        pyleoclim.ui.CorrEns : the correlation ensemble object

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            from pyleoclim.utils.tsmodel import colored_noise
            import numpy as np

            nt = 100
            t0 = np.arange(nt)
            v0 = colored_noise(alpha=1, t=t0)
            noise = np.random.normal(loc=0, scale=1, size=nt)

            ts0 = pyleo.Series(time=t0, value=v0)
            ts1 = pyleo.Series(time=t0, value=v0+noise)
            ts2 = pyleo.Series(time=t0, value=v0+2*noise)
            ts3 = pyleo.Series(time=t0, value=v0+1/2*noise)

            ts_list = [ts1, ts2, ts3]

            ms = pyleo.MultipleSeries(ts_list)
            ts_target = ts0

            # set an arbitrary randome seed to fix the result
            corr_res = ms.correlation(ts_target, settings={'nsim': 20}, seed=2333)
            print(corr_res)

            # set an arbitrary randome seed to fix the result
            corr_res = ms.correlation(settings={'nsim': 20}, seed=2333)
            print(corr_res)

        '''
        r_list = []
        signif_list = []
        p_list = []

        if target is None:
            target = self.series_list[0]

        for idx, ts in tqdm(enumerate(self.series_list),  total=len(self.series_list), disable=mute_pbar):
            corr_res = ts.correlation(target, timespan=timespan, alpha=alpha, settings=settings, common_time_kwargs=common_time_kwargs, seed=seed)
            r_list.append(corr_res.r)
            signif_list.append(corr_res.signif)
            p_list.append(corr_res.p)

        r_list = np.array(r_list)
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

    # def mssa(self, M, MC=0, f=0.5):
    #     data = []
    #     for val in self.series_list:
    #         data.append(val.value)
    #     data = np.transpose(np.asarray(data))


    #     res = decomposition.mssa(data, M=M, MC=MC, f=f)
    #     return res

    def equal_lengths(self):
        ''' Test whether all series in object have equal length

        Parameters
        ----------
        None

        Returns
        -------
        flag : boolean
        lengths : list containing the lengths of the series in object
        '''

        lengths = []
        for ts in self.series_list:
            lengths.append(len(ts.value))

        L = lengths[0]
        r = lengths[1:]
        flag = all (l==L for l in r)

        return flag, lengths
    
    def pca(self,weights=None,missing='fill-em',tol_em=5e-03, max_em_iter=100,**pca_kwargs):
        '''Principal Component Analysis (Empirical Orthogonal Functions)

        Decomposition of dataset ys in terms of orthogonal basis functions.
        Tolerant to missing values, infilled by an EM algorithm. 
        Requires ncomp to be less than the number of missing values.
        
        Do make sure the time axes are aligned, however! (e.g. use `common_time()`)
        
        Algorithm from statsmodels: https://www.statsmodels.org/stable/generated/statsmodels.multivariate.pca.PCA.html
        
        
        Parameters
        ----------

        
        weights : ndarray, optional
            Series weights to use after transforming data according to standardize
            or demean when computing the principal components.
        
        missing : {str, None}
            Method for missing data.  Choices are:
        
            * 'drop-row' - drop rows with missing values.
            * 'drop-col' - drop columns with missing values.
            * 'drop-min' - drop either rows or columns, choosing by data retention.
            * 'fill-em' - use EM algorithm to fill missing value.  ncomp should be
              set to the number of factors required.
            * `None` raises if data contains NaN values.
        
        tol_em : float
            Tolerance to use when checking for convergence of the EM algorithm.
        max_em_iter : int
            Maximum iterations for the EM algorithm.
        
        Attributes
        ----------
        res: pyleoclim.ui.SpatialDecomp
            the result object, see `pyleoclim.ui.SpatialDecomp`
            

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist).common_time()
        
            res = ms.pca() # carry out PCA
            
            res.screeplot() # plot the eigenvalue spectrum
            res.modeplot() # plot the first mode
        '''
        flag, lengths = self.equal_lengths()

        if flag==False:
            print('All Time Series should be of same length. Apply common_time() first')
        else: # if all series have equal length
            p = len(lengths)
            n = lengths[0]
            ys = np.empty((n,p))
            for j in range(p):
                ys[:,j] = self.series_list[j].value  # fill in data matrix
        
        nc = min(ys.shape) # number of components to return
        
        out  = PCA(ys,weights=weights,missing=missing,tol_em=tol_em, max_em_iter=max_em_iter,**pca_kwargs)
        
        # compute effective sample size
        PC1  = out.factors[:,0]
        neff = tsutils.eff_sample_size(PC1) 
        
        # compute percent variance
        pctvar = out.eigenvals**2/np.sum(out.eigenvals**2)*100
        
        # assign result to SpatiamDecomp class
        # Note: need to grab coordinates from Series or LiPDSeries        
        res = SpatialDecomp(name='PCA', time = self.series_list[0].time, neff= neff,
                            pcs = out.scores, pctvar = pctvar,  locs = None,
                            eigvals = out.eigenvals, eigvecs = out.eigenvecs)
        return res    

    # def mcpca(self,nMC=200,**pca_kwargs):
    #     ''' Monte Carlo Principal Component Analysis
        
    #     (UNDER REPAIR)

    #     Parameters
    #     ----------

    #     nMC : int
    #         number of Monte Carlo simulations

    #     pca_kwargs : tuple


    #     Returns
    #     -------
    #     res : dictionary containing:

    #         - eigval : eigenvalues (nrec,)
    #         - eig_ar1 : eigenvalues of the AR(1) ensemble (nrec, nMC)
    #         - pcs  : PC series of all components (nrec, nt)
    #         - eofs : EOFs of all components (nrec, nrec)

    #     References:
    #     ----------
    #     Deininger, M., McDermott, F., Mudelsee, M. et al. (2017): Coherency of late Holocene
    #     European speleothem δ18O records linked to North Atlantic Ocean circulation.
    #     Climate Dynamics, 49, 595–618. https://doi.org/10.1007/s00382-016-3360-8

    #     See also
    #     --------

    #     pyleoclim.utils.decomposition.mcpca: Monte Carlo PCA

    #     Examples
    #     --------

    #     .. ipython:: python
    #         :okwarning:

    #         import pyleoclim as pyleo
    #         url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
    #         data = pyleo.Lipd(usr_path = url)
    #         tslist = data.to_LipdSeriesList()
    #         tslist = tslist[2:] # drop the first two series which only concerns age and depth
    #         ms = pyleo.MultipleSeries(tslist)

    #         # msc = ms.common_time()

    #         # res = msc.pca(nMC=20)

    #     '''
    #     flag, lengths = self.equal_lengths()

    #     if flag==False:
    #         print('All Time Series should be of same length. Apply common_time() first')
    #     else: # if all series have equal length
    #         p = len(lengths)
    #         n = lengths[0]
    #         ys = np.empty((n,p))
    #         for j in range(p):
    #             ys[:,j] = self.series_list[j].value

    #     res = decomposition.mcpca(ys, nMC, **pca_kwargs)
    #     return res

    def bin(self, **kwargs):
        '''Aligns the time axes of a MultipleSeries object, via binning.
        
        This is critical for workflows that need to assume a common time axis
        for the group of series under consideration.


        The common time axis is characterized by the following parameters:

        start : the latest start date of the bunch (maximin of the minima)
        stop  : the earliest stop date of the bunch (minimum of the maxima)
        step  : The representative spacing between consecutive values (mean of the median spacings)

        This is a special case of the common_time function.

        Parameters
        ----------

        kwargs : dict
            Arguments for the binning function. See pyleoclim.utils.tsutils.bin

        Returns
        -------
        ms : pyleoclim.MultipleSeries
            The MultipleSeries objects with all series aligned to the same time axis.

        See also
        --------

        pyleoclim.core.ui.MultipleSeries.common_time: Base function on which this operates

        pyleoclim.utils.tsutils.bin: Underlying binning function

        pyleoclim.core.ui.Series.bin: Bin function for Series object

        Examples
        --------
        
        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist)
            msbin = ms.bin()

        '''

        ms = self.copy()

        ms = ms.common_time(method = 'bin', **kwargs)

        return ms

    def gkernel(self, **kwargs):
        ''' Aligns the time axes of a MultipleSeries object, via Gaussian kernel.
        This is critical for workflows that need to assume a common time axis
        for the group of series under consideration.


        The common time axis is characterized by the following parameters:

        start : the latest start date of the bunch (maximin of the minima)
        stop  : the earliest stop date of the bunch (minimum of the maxima)
        step  : The representative spacing between consecutive values (mean of the median spacings)

        This is a special case of the common_time function.

        Parameters
        ----------

        kwargs : dict
            Arguments for gkernel. See pyleoclim.utils.tsutils.gkernel for details.

        Returns
        -------
        ms : pyleoclim.MultipleSeries
            The MultipleSeries objects with all series aligned to the same time axis.

        See also
        --------

        pyleoclim.core.ui.MultipleSeries.common_time: Base function on which this operates

        pyleoclim.utils.tsutils.gkernel: Underlying kernel module


        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist)
            msk = ms.gkernel()

        '''

        ms = self.copy()

        ms = ms.common_time(method = 'gkernel', **kwargs)

        return ms

    def interp(self, **kwargs):
        ''' Aligns the time axes of a MultipleSeries object, via interpolation.
        This is critical for workflows that need to assume a common time axis
        for the group of series under consideration.


        The common time axis is characterized by the following parameters:

        start : the latest start date of the bunch (maximin of the minima)
        stop  : the earliest stop date of the bunch (minimum of the maxima)
        step  : The representative spacing between consecutive values (mean of the median spacings)

        This is a special case of the common_time function.

        Parameters
        ----------

        kwargs: keyword arguments (dictionary) for the interpolation method

        Returns
        -------
        ms : pyleoclim.MultipleSeries
            The MultipleSeries objects with all series aligned to the same time axis.

        See also
        --------

        pyleoclim.core.ui.MultipleSeries.common_time: Base function on which this operates

        pyleoclim.utils.tsutils.interp: Underlying interpolation function

        pyleoclim.core.ui.Series.interp: Interpolation function for Series object

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist)
            msinterp = ms.interp()

        '''
        ms = self.copy()

        ms = ms.common_time(method='interp', **kwargs)

        return ms

    def detrend(self,method='emd',**kwargs):
        '''Detrend timeseries

        Parameters
        ----------
        method : str, optional
            The method for detrending. The default is 'emd'.
            Options include:
                * linear: the result of a linear least-squares fit to y is subtracted from y.
                * constant: only the mean of data is subtrated.
                * "savitzky-golay", y is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
                * "emd" (default): Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
        **kwargs : dict
            Relevant arguments for each of the methods.

        Returns
        -------
        ms : pyleoclim.MultipleSeries
            The detrended timeseries

        See also
        --------

        pyleoclim.core.ui.Series.detrend : Detrending for a single series
        pyleoclim.utils.tsutils.detrend : Detrending function

        '''
        ms=self.copy()
        for idx,item in enumerate(ms.series_list):
            s=item.copy()
            v_mod=tsutils.detrend(item.value,x=item.time,method=method,**kwargs)
            s.value=v_mod
            ms.series_list[idx]=s
        return ms

    def spectral(self, method='lomb_scargle', settings=None, mute_pbar=False, freq_method='log', freq_kwargs=None, label=None, verbose=False, scalogram_list=None):
        ''' Perform spectral analysis on the timeseries

        Parameters
        ----------

        method : str
            {'wwz', 'mtm', 'lomb_scargle', 'welch', 'periodogram'}

        freq_method : str
            {'log','scale', 'nfft', 'lomb_scargle', 'welch'}

        freq_kwargs : dict
            Arguments for frequency vector

        settings : dict
            Arguments for the specific spectral method

        label : str
            Label for the PSD object

        verbose : bool
            If True, will print warning messages if there is any

        mute_pbar : {True, False}
            Mute the progress bar. Default is False.
        
        scalogram_list : pyleoclim.MultipleScalogram object, optional
            Multiple scalogram object containing pre-computed scalograms to use when calculating spectra, only works with wwz

        Returns
        -------

        psd : pyleoclim.MultiplePSD
            A Multiple PSD object

        See also
        --------
        pyleoclim.utils.spectral.mtm : Spectral analysis using the Multitaper approach

        pyleoclim.utils.spectral.lomb_scargle : Spectral analysis using the Lomb-Scargle method

        pyleoclim.utils.spectral.welch: Spectral analysis using the Welch segement approach

        pyleoclim.utils.spectral.periodogram: Spectral anaysis using the basic Fourier transform

        pyleoclim.utils.spectral.wwz_psd : Spectral analysis using the Wavelet Weighted Z transform

        pyleoclim.utils.wavelet.make_freq_vector : Functions to create the frequency vector

        pyleoclim.utils.tsutils.detrend : Detrending function

        pyleoclim.core.ui.Series.spectral : Spectral analysis for a single timeseries

        pyleoclim.core.ui.PSD : PSD object

        pyleoclim.core.ui.MultiplePSD : Multiple PSD object
        '''
        settings = {} if settings is None else settings.copy()

        psd_list = []
        if method == 'wwz' and scalogram_list:
            scalogram_list_len = len(scalogram_list.scalogram_list)
            series_len = len(self.series_list)

            #In the case where the scalogram list and series list are the same we can re-use scalograms in a one to one fashion
            #OR if the scalogram list is longer than the series list we use as many scalograms from the scalogram list as we need
            if scalogram_list_len >= series_len:
                for idx, s in enumerate(tqdm(self.series_list, desc='Performing spectral analysis on individual series', position=0, leave=True, disable=mute_pbar)):
                    psd_tmp = s.spectral(method=method, settings=settings, freq_method=freq_method, freq_kwargs=freq_kwargs, label=label, verbose=verbose,scalogram = scalogram_list.scalogram_list[idx])
                    psd_list.append(psd_tmp)
            #If the scalogram list isn't as long as the series list, we re-use all the scalograms we can and then recalculate the rest
            elif scalogram_list_len < series_len:
                for idx, s in enumerate(tqdm(self.series_list, desc='Performing spectral analysis on individual series', position=0, leave=True, disable=mute_pbar)):
                    if idx < scalogram_list_len:
                        psd_tmp = s.spectral(method=method, settings=settings, freq_method=freq_method, freq_kwargs=freq_kwargs, label=label, verbose=verbose,scalogram = scalogram_list.scalogram_list[idx])
                        psd_list.append(psd_tmp)
                    else:
                        psd_tmp = s.spectral(method=method, settings=settings, freq_method=freq_method, freq_kwargs=freq_kwargs, label=label, verbose=verbose)
                        psd_list.append(psd_tmp)
        else: 
            for s in tqdm(self.series_list, desc='Performing spectral analysis on individual series', position=0, leave=True, disable=mute_pbar):
                psd_tmp = s.spectral(method=method, settings=settings, freq_method=freq_method, freq_kwargs=freq_kwargs, label=label, verbose=verbose)
                psd_list.append(psd_tmp)

        psds = MultiplePSD(psd_list=psd_list)

        return psds

    def wavelet(self, method='wwz', settings={}, freq_method='log', ntau=None, freq_kwargs=None, verbose=False, mute_pbar=False):
        '''Wavelet analysis

        Parameters
        ----------
        method : {wwz, cwt}
            Whether to use the wwz method for unevenly spaced timeseries or traditional cwt (from pywavelets)

        settings : dict, optional
            Settings for the particular method. The default is {}.

        freq_method : str
            {'log', 'scale', 'nfft', 'lomb_scargle', 'welch'}

        freq_kwargs : dict
            Arguments for frequency vector

        ntau : int
            The length of the time shift points that determins the temporal resolution of the result.
            If None, it will be either the length of the input time axis, or at most 100.

        settings : dict
            Arguments for the specific spectral method

        verbose : bool
            If True, will print warning messages if there is any


        mute_pbar : bool, optional
            Whether to mute the progress bar. The default is False.

        Returns
        -------
        scals : pyleoclim.MultipleScalograms

        See also
        --------
        pyleoclim.utils.wavelet.wwz : wwz function

        pyleoclim.utils.wavelet.make_freq_vector : Functions to create the frequency vector

        pyleoclim.utils.tsutils.detrend : Detrending function

        pyleoclim.core.ui.Series.wavelet : wavelet analysis on single object

        pyleoclim.core.ui.MultipleScalogram : Multiple Scalogram object

        '''
        settings = {} if settings is None else settings.copy()

        scal_list = []
        for s in tqdm(self.series_list, desc='Performing wavelet analysis on individual series', position=0, leave=True, disable=mute_pbar):
            scal_tmp = s.wavelet(method=method, settings=settings, freq_method=freq_method, freq_kwargs=freq_kwargs, verbose=verbose, ntau=ntau)
            scal_list.append(scal_tmp)

        scals = MultipleScalogram(scalogram_list=scal_list)

        return scals

    def plot(self, figsize=[10, 4],
             marker=None, markersize=None,
             linestyle=None, linewidth=None, colors=None, cmap='tab10', norm=None,
             xlabel=None, ylabel=None, title=None,
             legend=True, plot_kwargs=None, lgd_kwargs=None,
             savefig_settings=None, ax=None, mute=False, invert_xaxis=False):
        '''Plot multiple timeseries on the same axis

        Parameters
        ----------
        figsize : list, optional
            Size of the figure. The default is [10, 4].
        marker : str, optional
            marker type. The default is None.
        markersize : float, optional
            marker size. The default is None.
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
        linestyle : str, optional
            Line style. The default is None.
        linewidth : float, optional
            The width of the line. The default is None.
        xlabel : str, optional
            x-axis label. The default is None.
        ylabel : str, optional
            y-axis label. The default is None.
        title : str, optional
            Title. The default is None.
        legend : bool, optional
            Wether the show the legend. The default is True.
        plot_kwargs : dict, optional
            Plot parameters. The default is None.
        lgd_kwargs : dict, optional
            Legend parameters. The default is None.
        savefig_settings : dictionary, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.
        ax : matplotlib.ax, optional
            The matplotlib axis onto which to return the figure. The default is None.
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)
        invert_xaxis : bool, optional
            if True, the x-axis of the plot will be inverted

        Returns
        -------
        fig, ax

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if ylabel is None:
            consistent_ylabels = True
            time_label, value_label = self.series_list[0].make_labels()
            for s in self.series_list[1:]:
                time_label_tmp, value_label_tmp = s.make_labels()
                if value_label_tmp != value_label:
                    consistent_ylabels = False

            if consistent_ylabels:
                ylabel = value_label
            else:
                ylabel = 'value'

        for idx, s in enumerate(self.series_list):
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

            ax = s.plot(
                figsize=figsize, marker=marker, markersize=markersize, color=clr, linestyle=linestyle,
                linewidth=linewidth, label=s.label, xlabel=xlabel, ylabel=ylabel, title=title,
                legend=legend, lgd_kwargs=lgd_kwargs, plot_kwargs=plot_kwargs, ax=ax,
            )

        if invert_xaxis:
            ax.invert_xaxis()

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def stackplot(self, figsize=None, savefig_settings=None,  xlim=None, fill_between_alpha=0.2, colors=None, cmap='tab10', norm=None, labels='auto',
                  spine_lw=1.5, grid_lw=0.5, font_scale=0.8, label_x_loc=-0.15, v_shift_factor=3/4, linewidth=1.5, plot_kwargs=None, mute=False):
        ''' Stack plot of multiple series

        Note that the plotting style is uniquely designed for this one and cannot be properly reset with `pyleoclim.set_style()`.

        Parameters
        ----------
        figsize : list
            Size of the figure.
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
        labels: None, 'auto' or list
            If None, doesn't add labels to the subplots
            If 'auto', uses the labels passed during the creation of pyleoclim.Series
            If list, pass a list of strings for each labels. 
            Default is 'auto'
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
        plot_kwargs: dict or list of dict
            Arguments to further customize the plot from matplotlib.pyplot.plot.
            Dictionary: Arguments will be applied to all lines in the stackplots
            List of dictionary: Allows to customize one line at a time. 
        mute : {True,False}
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)
        
        Returns
        -------
        fig, ax

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist)
            @savefig mts_stackplot.png
            fig, ax = ms.stackplot()
            pyleo.closefig(fig)
        
        Let's change the labels on the left
        
        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            sst = d.to_LipdSeries(number=5)
            d18Osw = d.to_LipdSeries(number=3)
            ms = pyleo.MultipleSeries([sst,d18Osw])
            @savefig mts_stackplot_customlabels.png
            fig, ax = ms.stackplot(labels=['sst','d18Osw'])
            pyleo.closefig(fig)
            
        And let's remove them completely
        
        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            sst = d.to_LipdSeries(number=5)
            d18Osw = d.to_LipdSeries(number=3)
            ms = pyleo.MultipleSeries([sst,d18Osw])
            @savefig mts_stackplot_nolabels.png
            fig, ax = ms.stackplot(labels=None)
            pyleo.closefig(fig)
        
        Now, let's add markers to the timeseries.
        
        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            sst = d.to_LipdSeries(number=5)
            d18Osw = d.to_LipdSeries(number=3)
            ms = pyleo.MultipleSeries([sst,d18Osw])
            @savefig mts_stackplot_samemarkers.png
            fig, ax = ms.stackplot(labels=None, plot_kwargs={'marker':'o'})
            pyleo.closefig(fig)
        
        But I really want to use different markers
        
        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            sst = d.to_LipdSeries(number=5)
            d18Osw = d.to_LipdSeries(number=3)
            ms = pyleo.MultipleSeries([sst,d18Osw])
            @savefig mts_stackplot_differentmarkers.png
            fig, ax = ms.stackplot(labels=None, plot_kwargs=[{'marker':'o'},{'marker':'^'}])
            pyleo.closefig(fig)

        '''
        current_style = deepcopy(mpl.rcParams)
        plotting.set_style('journal', font_scale=font_scale)
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        n_ts = len(self.series_list)
        
        if type(labels)==list:
            if len(labels) != n_ts:
                raise ValueError("The length of the label list should match the number of timeseries to be plotted")
        
        # Deal with plotting arguments
        if type(plot_kwargs)==dict:
            plot_kwargs = [plot_kwargs]*n_ts
        
        if plot_kwargs is not None and len(plot_kwargs) != n_ts:
            raise ValueError("When passing a list of dictionaries for kwargs arguments, the number of items should be the same as the number of timeseries")
            
                
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
            #deal with other plotting arguments
            if plot_kwargs is None:
                p_kwargs = {}
            else:
                p_kwargs = plot_kwargs[idx]
            
            bottom -= height*v_shift_factor
            ax[idx] = fig.add_axes([left, bottom, width, height])
            ax[idx].plot(ts.time, ts.value, color=clr, lw=linewidth,**p_kwargs)
            ax[idx].patch.set_alpha(0)
            ax[idx].set_xlim(xlim)
            time_label, value_label = ts.make_labels()
            ax[idx].set_ylabel(value_label, weight='bold')

            mu = np.mean(ts.value)
            std = np.std(ts.value)
            ylim = [mu-4*std, mu+4*std]
            ax[idx].fill_between(ts.time, ts.value, y2=mu, alpha=fill_between_alpha, color=clr)
            trans = transforms.blended_transform_factory(ax[idx].transAxes, ax[idx].transData)
            if labels == 'auto':
                if ts.label is not None:
                    ax[idx].text(label_x_loc, mu, ts.label, horizontalalignment='right', transform=trans, color=clr, weight='bold')
            elif type(labels) ==list:
                ax[idx].text(label_x_loc, mu, labels[idx], horizontalalignment='right', transform=trans, color=clr, weight='bold')
            elif labels==None:
                pass
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
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            # reset the plotting style
            mpl.rcParams.update(current_style)
            return fig, ax
        else:
            plotting.showfig(fig)
            # reset the plotting style
            mpl.rcParams.update(current_style)
            return ax


class SurrogateSeries(MultipleSeries):
    ''' Object containing surrogate timeseries, usually obtained through recursive modeling (e.g., AR1)

    Surrogate Series is a child of MultipleSeries. All methods available for MultipleSeries are available for surrogate series.
    '''
    def __init__(self, series_list, surrogate_method=None, surrogate_args=None):
        self.series_list = series_list
        self.surrogate_method = surrogate_method
        self.surrogate_args = surrogate_args

class EnsembleSeries(MultipleSeries):
    ''' EnsembleSeries object

    The EnsembleSeries object is a child of the MultipleSeries object, that is, a special case of MultipleSeries, aiming for ensembles of similar series.
    Ensembles usually arise from age modeling or Bayesian calibrations. All members of an EnsembleSeries object are assumed to share identical labels and units.

    All methods available for MultipleSeries are available for EnsembleSeries. Some functions were modified for the special case of ensembles.

    '''
    def __init__(self, series_list):
        self.series_list = series_list

    def make_labels(self):
        '''
        Initialization of labels

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

    def quantiles(self, qs=[0.05, 0.5, 0.95]):
        '''Calculate quantiles of an EnsembleSeries object

        Parameters
        ----------
        qs : list, optional
            List of quantiles to consider for the calculation. The default is [0.05, 0.5, 0.95].

        Returns
        -------
        ens_qs : pyleoclim.EnsembleSeries

        '''
        time = np.copy(self.series_list[0].time)
        vals = []
        for ts in self.series_list:
            if not np.array_equal(ts.time, time):
                raise ValueError('Time axis not consistent across the ensemble!')

            vals.append(ts.value)

        vals = np.array(vals)
        ens_qs = mquantiles(vals, qs, axis=0)

        ts_list = []
        for i, quant in enumerate(ens_qs):
            ts = Series(time=time, value=quant, label=f'{qs[i]*100:g}%')
            ts_list.append(ts)

        ens_qs = EnsembleSeries(series_list=ts_list)

        return ens_qs

    def correlation(self, target=None, timespan=None, alpha=0.05, settings=None, fdr_kwargs=None, common_time_kwargs=None, mute_pbar=False, seed=None):
        ''' Calculate the correlation between an EnsembleSeries object to a target.

        If the target is not specified, then the 1st member of the ensemble will be the target
        Note that the FDR approach is applied by default to determine the significance of the p-values (more information in See Also below).

        Parameters
        ----------
        target : pyleoclim.Series or pyleoclim.EnsembleSeries, optional
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

        settings : dict
            Parameters for the correlation function, including:

            nsim : int
                the number of simulations (default: 1000)
            method : str, {'ttest','isopersistent','isospectral' (default)}
                method for significance testing

        fdr_kwargs : dict
            Parameters for the FDR function

        common_time_kwargs : dict
            Parameters for the method MultipleSeries.common_time()

        mute_pbar : bool
            If True, the progressbar will be muted. Default is False.

        seed : float or int
            random seed for isopersistent and isospectral methods

        Returns
        -------

        corr : pyleoclim.ui.CorrEns
            the result object, see `pyleoclim.ui.CorrEns`

        See also
        --------

        pyleoclim.utils.correlation.corr_sig : Correlation function
        pyleoclim.utils.correlation.fdr : FDR function
        pyleoclim.ui.CorrEns : the correlation ensemble object

        Examples
        --------

        .. ipython:: python
            :okwarning:

            import pyleoclim as pyleo
            import numpy as np
            from pyleoclim.utils.tsmodel import colored_noise

            nt = 100
            t0 = np.arange(nt)
            v0 = colored_noise(alpha=1, t=t0)
            noise = np.random.normal(loc=0, scale=1, size=nt)

            ts0 = pyleo.Series(time=t0, value=v0)
            ts1 = pyleo.Series(time=t0, value=v0+noise)
            ts2 = pyleo.Series(time=t0, value=v0+2*noise)
            ts3 = pyleo.Series(time=t0, value=v0+1/2*noise)

            ts_list1 = [ts0, ts1]
            ts_list2 = [ts2, ts3]

            ts_ens = pyleo.EnsembleSeries(ts_list1)
            ts_target = pyleo.EnsembleSeries(ts_list2)

            # set an arbitrary randome seed to fix the result
            corr_res = ts_ens.correlation(ts_target, seed=2333)
            print(corr_res)

        '''
        if target is None:
            target = self.series_list[0]

        r_list = []
        p_list = []
        signif_list = []
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

            ts2 = Series(time=time2, value=value2, verbose=idx==0)
            corr_res = ts1.correlation(ts2, timespan=timespan, settings=settings, common_time_kwargs=common_time_kwargs, seed=seed)
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
             color=sns.xkcd_rgb['pale red'], lw=0.5, alpha=0.3, lgd_kwargs=None, mute=False):
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
            plot_legend : bool, optional
                Whether to plot the legend. The default is True.
            lgd_kwargs : dict, optional
                Parameters for the legend. The default is None.
            mute : bool, optional
                if True, the plot will not show;
                recommend to turn on when more modifications are going to be made on ax. The default is False.
                (going to be deprecated)
            seed : int, optional
                Set the seed for the random number generator. Useful for reproducibility. The default is None.

            Returns
            -------
            fig, ax
            
            Examples
            --------

            .. ipython:: python
                :okwarning:
                :okexcept:    
                
                nn = 30 # number of noise realizations
                nt = 500
                series_list = []
        
                signal = pyleo.gen_ts(model='colored_noise',nt=nt,alpha=1.0).standardize() 
                noise = np.random.randn(nt,nn)
        
                for idx in range(nn):  # noise
                    ts = pyleo.Series(time=signal.time, value=signal.value+noise[:,idx])
                    series_list.append(ts)
        
                ts_ens = pyleo.EnsembleSeries(series_list)
        
                fig, ax = ts_ens.plot_traces(alpha=0.2,num_traces=8) 
                pyleo.closefig(fig)

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

            if num_traces > 0:
                if seed is not None:
                    np.random.seed(seed)

                nts = np.size(self.series_list)
                random_draw_idx = np.random.choice(nts, num_traces)

                for idx in random_draw_idx:
                    self.series_list[idx].plot(xlabel=xlabel, ylabel=ylabel, zorder=99, linewidth=lw,
                        xlim=xlim, ylim=ylim, ax=ax, color=color, alpha=alpha,linestyle='-')
                ax.plot(np.nan, np.nan, color=color, label=f'example members (n={num_traces})',linestyle='-')

            if title is not None:
                ax.set_title(title)

            if plot_legend:
                lgd_args = {'frameon': False}
                lgd_args.update(lgd_kwargs)
                ax.legend(**lgd_args)

            if 'fig' in locals():
                if 'path' in savefig_settings:
                    plotting.savefig(fig, settings=savefig_settings)
                # else:
                #     if not mute:
                #         plotting.showfig(fig)
                return fig, ax
            else:
                return ax

    def plot_envelope(self, figsize=[10, 4], qs=[0.025, 0.25, 0.5, 0.75, 0.975],
                      xlabel=None, ylabel=None, title=None,
                      xlim=None, ylim=None, savefig_settings=None, ax=None, plot_legend=True,
                      curve_clr=sns.xkcd_rgb['pale red'], curve_lw=2, shade_clr=sns.xkcd_rgb['pale red'], shade_alpha=0.2,
                      inner_shade_label='IQR', outer_shade_label='95% CI', lgd_kwargs=None, mute=False):
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
        plot_legend : bool, optional
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
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax. The default is False.
            (going to be deprecated)

        Returns
        -------
        fig, ax
        
        Examples
        --------
        .. ipython:: python
            :okwarning:
            :okexcept:    
            
            nn = 30 # number of noise realizations
            nt = 500
            series_list = []
    
            signal = pyleo.gen_ts(model='colored_noise',nt=nt,alpha=1.0).standardize() 
            noise = np.random.randn(nt,nn)
    
            for idx in range(nn):  # noise
                ts = pyleo.Series(time=signal.time, value=signal.value+noise[:,idx])
                series_list.append(ts)
    
            ts_ens = pyleo.EnsembleSeries(series_list)  
            fig, ax = ts_ens.plot_envelope(curve_lw=1.5) 
            pyleo.closefig(fig)
 
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
            color=shade_clr, alpha=shade_alpha, edgecolor=shade_clr, label=outer_shade_label,
        )
        # plot inner envelope on top
        ax.fill_between(
            time, ts_qs.series_list[1].value, ts_qs.series_list[3].value,
            color=shade_clr, alpha=2*shade_alpha, edgecolor=shade_clr, label=inner_shade_label,
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

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            return fig, ax
        else:
            return ax


    def stackplot(self, figsize=[5, 15], savefig_settings=None,  xlim=None, fill_between_alpha=0.2, colors=None, cmap='tab10', norm=None,
                  spine_lw=1.5, grid_lw=0.5, font_scale=0.8, label_x_loc=-0.15, v_shift_factor=3/4, linewidth=1.5, mute=False):
        ''' Stack plot of multiple series

        Note that the plotting style is uniquely designed for this one and cannot be properly reset with `pyleoclim.set_style()`.
                Parameters
        ----------
        figsize : list
            Size of the figure.
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
        mute : bool
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
             (going to be deprecated)

        Returns
        -------
        fig, ax
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
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            # reset the plotting style
            mpl.rcParams.update(current_style)
            return fig, ax
        else:
            # reset the plotting style
            mpl.rcParams.update(current_style)
            return ax

    
    def distplot(self, figsize=[10, 4], title=None, savefig_settings=None,
                 ax=None, ylabel='KDE', vertical=False, edgecolor='w',mute=False, **plot_kwargs):
        """
        Plots the distribution of the timeseries across ensembles

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
        vertical : {True,False}, optional
            Whether to flip the plot vertically. The default is False.
        edgecolor : matplotlib.color, optional
            The color of the edges of the bar. The default is 'w'.
        mute : {True,False}, optional
           if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax. The default is False.
            (going to be deprecated)
        **plot_kwargs : dict
            Plotting arguments for seaborn histplot: https://seaborn.pydata.org/generated/seaborn.histplot.html.
            
        See also
        --------

        pyleoclim.utils.plotting.savefig : saving figure in Pyleoclim

        """
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        #make the data into a dataframe so we can flip the figure
        time_label, value_label = self.make_labels()
        
        #append all the values together for the plot
        for item in self.series_list:
            try:
                val=np.append(val,item.value)
            except:
                val=item.value
        
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
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            return fig, ax
        else:
            return ax        
        

class MultiplePSD:
    ''' Object for multiple PSD.

    Used for significance level
    '''
    def __init__(self, psd_list, beta_est_res=None):
        self.psd_list = psd_list
        self.beta_est_res = beta_est_res

    def copy(self):
        '''Copy object
        '''
        return deepcopy(self)

    def quantiles(self, qs=[0.05, 0.5, 0.95], lw=[0.5, 1.5, 0.5]):
        '''Calculate quantiles

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
        psds : pyleoclim.MultiplePSD

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
        ''' Estimate the scaling factor beta of the each PSD from the psd_list in a log-log space

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

        new : pyleoclim.MultiplePSD
            New MultiplePSD object with the estimated scaling slope information, which is stored as a dictionary that includes:
            - beta: the scaling factor
            - std_err: the one standard deviation error of the scaling factor
            - f_binned: the binned frequency series, used as X for linear regression
            - psd_binned: the binned PSD series, used as Y for linear regression
            - Y_reg: the predicted Y from linear regression, used with f_binned for the slope curve plotting

        See also
        --------

        pyleoclim.core.ui.PSD.beta_est : beta estimation for on a single PSD object

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
             colors=None, cmap=None, norm=None, plot_kwargs=None, lgd_kwargs=None, mute=False):
        '''Plot multiple PSD on the same plot

        Parameters
        ----------
        figsize : list, optional
            Figure size. The default is [10, 4].
        in_loglog : bool, optional
            Whether to plot in loglog. The default is True.
        in_period : bool, optional
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
            - "path" must be specified; it can be any existed or non-existed path,
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
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            The default is False.
            (going to be deprecated)

        Returns
        -------
        fig, ax

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
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def plot_envelope(self, figsize=[10, 4], qs=[0.025, 0.5, 0.975],
             in_loglog=True, in_period=True, xlabel=None, ylabel='Amplitude', title=None,
             xlim=None, ylim=None, savefig_settings=None, ax=None, xticks=None, yticks=None, plot_legend=True,
             curve_clr=sns.xkcd_rgb['pale red'], curve_lw=3, shade_clr=sns.xkcd_rgb['pale red'], shade_alpha=0.3, shade_label=None,
             lgd_kwargs=None, mute=False, members_plot_num=10, members_alpha=0.3, members_lw=1, seed=None):

        '''Plot mutiple PSD as an envelope.

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
            - "path" must be specified; it can be any existed or non-existed path,
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
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax. The default is False.
            (going to be deprecated)
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
        fig, ax

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
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            return fig, ax
        else:
            return ax


class MultipleScalogram:
    ''' Multiple Scalogram objects
    '''
    def __init__(self, scalogram_list):
        self.scalogram_list = scalogram_list

    def copy(self):
        ''' Copy the object
        '''
        return deepcopy(self)

    def quantiles(self, qs=[0.05, 0.5, 0.95]):
        '''Calculate quantiles

        Parameters
        ----------
        qs : list, optional
            List of quantiles to consider for the calculation. The default is [0.05, 0.5, 0.95].

        Raises
        ------
        ValueError
            Frequency axis not consistent across the PSD list!

        Value Error
            Time axis not consistent across the scalogram list!

        Returns
        -------
        scals : pyleoclim.MultipleScalogram
        '''
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

 
class Corr:
    ''' The object for correlation result in order to format the print message

    Parameters
    ----------

    r: float
        the correlation coefficient

    p: float
        the p-value

    p_fmt_td: float
        the threshold for p-value formating (0.01 by default, i.e., if p<0.01, will print "< 0.01" instead of "0")

    p_fmt_style: str
        the style for p-value formating (exponential notation by default)

    signif: bool
        the significance

    alpha : float
        The significance level (0.05 by default)

    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Correlation function
    pyleoclim.utils.correlation.fdr : FDR function
    '''
    def __init__(self, r, p, signif, alpha, p_fmt_td=0.01, p_fmt_style='exp'):
        self.r = r
        self.p = p
        self.p_fmt_td = p_fmt_td
        self.p_fmt_style = p_fmt_style
        self.signif = signif
        self.alpha = alpha

    def __str__(self):
        '''
        Prints out the correlation results
        '''
        formatted_p = pval_format(self.p, threshold=self.p_fmt_td, style=self.p_fmt_style)

        table = {
            'correlation': [self.r],
            'p-value': [formatted_p],
            f'signif. (α: {self.alpha})': [self.signif],
        }

        msg = print(tabulate(table, headers='keys'))

        return ''

class CorrEns:
    ''' Correlation Ensemble

    Parameters
    ----------

    r: list
        the list of correlation coefficients

    p: list
        the list of p-values

    p_fmt_td: float
        the threshold for p-value formating (0.01 by default, i.e., if p<0.01, will print "< 0.01" instead of "0")

    p_fmt_style: str
        the style for p-value formating (exponential notation by default)

    signif: list
        the list of significance without FDR

    signif_fdr: list
        the list of significance with FDR

    signif_fdr: list
        the list of significance with FDR

    alpha : float
        The significance level

    See also
    --------

    pyleoclim.utils.correlation.corr_sig : Correlation function
    pyleoclim.utils.correlation.fdr : FDR function
    '''
    def __init__(self, r, p, signif, signif_fdr, alpha, p_fmt_td=0.01, p_fmt_style='exp'):
        self.r = r
        self.p = p
        self.p_fmt_td = p_fmt_td
        self.p_fmt_style = p_fmt_style
        self.signif = signif
        self.signif_fdr = signif_fdr
        self.alpha = alpha

    def __str__(self):
        '''
        Prints out the correlation results
        '''

        pi_list = []
        for pi in self.p:
            pi_list.append(pval_format(pi, threshold=self.p_fmt_td, style=self.p_fmt_style))

        table = {
            'correlation': self.r,
            'p-value': pi_list,
            f'signif. w/o FDR (α: {self.alpha})': self.signif,
            f'signif. w/ FDR (α: {self.alpha})': self.signif_fdr,
        }

        msg = print(tabulate(table, headers='keys'))

        return f'Ensemble size: {len(self.r)}'


    def plot(self, figsize=[4, 4], title=None, ax=None, savefig_settings=None, hist_kwargs=None, title_kwargs=None, xlim=None,
             clr_insignif=sns.xkcd_rgb['grey'], clr_signif=sns.xkcd_rgb['teal'], clr_signif_fdr=sns.xkcd_rgb['pale orange'],
             clr_percentile=sns.xkcd_rgb['salmon'], rwidth=0.8, bins=None, vrange=None, mute=False):
        ''' Plot the correlation ensembles

        Parameters
        ----------
        figsize : list, optional
            The figure size. The default is [4, 4].

        title : str, optional
            Plot title. The default is None.

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        hist_kwargs : dict
            the keyword arguments for ax.hist()

        title_kwargs : dict
            the keyword arguments for ax.set_title()

        ax : matplotlib.axis, optional
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        mute : {True,False}
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)

        xlim : list, optional
            x-axis limits. The default is None.

        See Also
        --------

        matplotlib.pyplot.hist: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.hist.html
        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        hist_kwargs = {} if hist_kwargs is None else hist_kwargs.copy()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if vrange is None:
            vrange = [np.min(self.r), np.max(self.r)]

        clr_list = [clr_insignif, clr_signif, clr_signif_fdr]
        args = {'rwidth': rwidth, 'bins': bins, 'range': vrange, 'color': clr_list}
        args.update(hist_kwargs)
        # insignif_args.update(hist_kwargs)

        r_insignif = np.array(self.r)[~np.array(self.signif)]
        r_signif = np.array(self.r)[self.signif]
        r_signif_fdr = np.array(self.r)[self.signif_fdr]
        r_stack = [r_insignif, r_signif, r_signif_fdr]
        ax.hist(r_stack, stacked=True, **args)
        ax.legend([f'p ≥ {self.alpha}', f'p < {self.alpha} (w/o FDR)', f'p < {self.alpha} (w/ FDR)'], loc='upper left', bbox_to_anchor=(1.1, 1), ncol=1)

        frac_signif = np.size(r_signif) / np.size(self.r)
        frac_signif_fdr = np.size(r_signif_fdr) / np.size(self.r)
        ax.text(x=1.1, y=0.5, s=f'Fraction significant: {frac_signif*100:.1f}%', transform=ax.transAxes, fontsize=10, color=clr_signif)
        ax.text(x=1.1, y=0.4, s=f'Fraction significant: {frac_signif_fdr*100:.1f}%', transform=ax.transAxes, fontsize=10, color=clr_signif_fdr)

        r_pcts = np.percentile(self.r, [2.5, 25, 50, 75, 97.5])
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for r_pct, pt, ls in zip(r_pcts, np.array([2.5, 25, 50, 75, 97.5])/100, [':', '--', '-', '--', ':']):
            ax.axvline(x=r_pct, linestyle=ls, color=clr_percentile)
            ax.text(x=r_pct, y=1.02, s=pt, color=clr_percentile, transform=trans, ha='center', fontsize=10)

        ax.set_xlabel(r'$r$')
        ax.set_ylabel('Count')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if xlim is not None:
            ax.set_xlim(xlim)


        if title is not None:
            title_kwargs = {} if title_kwargs is None else title_kwargs.copy()
            t_args = {'y': 1.1, 'weight': 'bold'}
            t_args.update(title_kwargs)
            ax.set_title(title, **t_args)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            return fig, ax
        else:
            return ax

        # if 'path' in savefig_settings:
        #     plotting.savefig(fig, settings=savefig_settings)
        # else:
        #     if not mute:
        #         plotting.showfig(fig)
        # return fig, ax

class SpatialDecomp:
    ''' Class to hold the results of spatial decompositions
        applies to : `pca()`, `mcpca()`, `mssa()` 
        
        Attributes
        ----------

        time: float
            the common time axis
            
        locs: float (p, 2)
            a p x 2 array of coordinates (latitude, longitude) for mapping the spatial patterns ("EOFs")
            
        name: str
            name of the dataset/analysis to use in plots
            
        eigvals: float
            vector of eigenvalues from the decomposition
            
        eigvecs: float
            array of eigenvectors from the decomposition  
         
        pctvar: float    
            array of pct variance accounted for by each mode
            
        neff: float
            scalar representing the effective sample size of the leading mode
    
    '''
    def __init__(self, time, locs, name, eigvals, eigvecs, pctvar, pcs, neff):
        self.time       = time
        self.name       = name
        self.locs       = locs 
        self.eigvals    = eigvals
        self.eigvecs    = eigvecs
        self.pctvar     = pctvar
        self.pcs        = pcs
        self.neff       = neff
        
    def screeplot(self, figsize=[6, 4], uq='N82' ,title='scree plot', ax=None, savefig_settings=None, 
                  title_kwargs=None, xlim=[0,10], clr_eig='C0',  mute=False):
        ''' Plot the eigenvalue spectrum with uncertainties

        Parameters
        ----------
        figsize : list, optional
            The figure size. The default is [6, 4].

        title : str, optional
            Plot title. The default is 'scree plot'.

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        title_kwargs : dict, optional
            the keyword arguments for ax.set_title()

        ax : matplotlib.axis, optional
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        mute : {True,False}
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)

        xlim : list, optional
            x-axis limits. The default is [0, 10] (first 10 eigenvalues)
            
        uq : str, optional
            Method used for uncertainty quantification of the eigenvalues.
            'N82' uses the North et al "rule of thumb" [1] with effective sample size 
            computed as in [2]. 
            'MC' uses Monte-Carlo simulations (e.g. MC-EOF). Returns an error if no ensemble is found.
            
        clr_eig : str, optional
            color to be used for plotting eigenvalues
        
            
        References
        ----------
        [1] North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng (1982), 
            Sampling errors in the estimation of empirical orthogonal functions,
            Mon. Weather Rev., 110, 699–706.
        [2] Hannachi, A., I. T. Jolliffe, and D. B. Stephenson (2007), Empirical
            orthogonal functions and related techniques in atmospheric science: 
            A review, International Journal of Climatology, 27(9), 
            1119–1152, doi:10.1002/joc.1499.

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        
        
        if self.neff < 2:
            self.neff = 2
        
        # compute 95% CI    
        if uq == 'N82':
            eb_lbl =  r'95% CI ($n_\mathrm{eff} = $'+ '{:.1f}'.format(self.neff) +')' # declare method
            Lc = self.eigvals # central estimate 
            Lerr  = np.tile(Lc,(2,1)) # declare array
            Lerr[0,:]  = Lc*np.sqrt(1-np.sqrt(2/self.neff))
            Lerr[1,:]  = Lc*np.sqrt(1+np.sqrt(2/self.neff))
        elif uq =='MC':
            eb_lbl =  '95% CI (Monte Carlo)' # declare method
            try:
                Lq = np.quantile(self.eigvals,[0.025,0.5,0.975],axis = 1)
                Lc = Lq[1,:]
                Lerr  = np.tile(Lc,(2,1)) # declare array
                Lerr[0,:]  = Lq[0,:]
                Lerr[1,:]  = Lq[2,:]
        
            except ValueError:
                print("Eigenvalue array must have more than 1 non-singleton dimension.")             
        else:
            raise NameError("unknown UQ method. No action taken")
           
            
        idx = np.arange(len(Lc)) + 1
        
        ax.errorbar(x=idx,y=Lc,yerr = Lerr, color=clr_eig,marker='o',ls='',
                    alpha=1.0,label=eb_lbl)
        
        ax.set_title(title,fontweight='bold'); ax.legend(); 
        ax.set_xlabel(r'Mode index $i$'); ax.set_ylabel(r'$\lambda_i$')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # enforce integer values


        if xlim is not None:
            ax.set_xlim(0.5,min(max(xlim),len(Lc)))

        if title is not None:
            title_kwargs = {} if title_kwargs is None else title_kwargs.copy()
            t_args = {'y': 1.1, 'weight': 'bold'}
            t_args.update(title_kwargs)
            ax.set_title(title, **t_args)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)
        # else:
        #     if not mute:
        #         plotting.showfig(fig)
        return fig, ax

    def modeplot(self, index=0, figsize=[10, 5], ax=None, savefig_settings=None, 
              title_kwargs=None, mute=False, spec_method = 'mtm'):
        ''' Dashboard visualizing the properties of a given mode, including:
            1. The temporal coefficient (PC or similar)
            2. its spectrum
            3. The spatial loadings (EOF or similar)

        Parameters
        ----------
        index : int
            the (0-based) index of the mode to visualize. 
            Default is 0, corresponding to the first mode. 
        
        figsize : list, optional
            The figure size. The default is [10, 5].
        
        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        title_kwargs : dict
            the keyword arguments for ax.set_title()

        gs : matplotlib.gridspec object, optional
            the axis object from matplotlib
            See [matplotlib.gridspec.GridSpec](https://matplotlib.org/stable/tutorials/intermediate/gridspec.html) for details.

        mute : {True,False}
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)

        spec_method: str, optional
            The name of the spectral method to be applied on the PC. Default: MTM
            Note that the data are evenly-spaced, so any spectral method that
            assumes even spacing is applicable here:  'mtm', 'welch', 'periodogram'
            'wwz' is relevant if scaling exponents need to be estimated, but ill-advised otherwise, as it is very slow. 
      
        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        PC = self.pcs[:,index]    
        ts = Series(time=self.time, value=PC) # define timeseries object for the PC

        fig = plt.figure(tight_layout=True,figsize=figsize)
        gs = gridspec.GridSpec(2, 2) # define grid for subplots
        ax1 = fig.add_subplot(gs[0, :])
        ts.plot(ax=ax1)
        ax1.set_ylabel('PC '+str(index+1))
        ax1.set_title('Mode '+str(index+1)+', '+ '{:3.2f}'.format(self.pctvar[index]) + '% variance explained',weight='bold')
        
        # plot spectrum
        ax2 = fig.add_subplot(gs[1, 0])
        psd_mtm_rc = ts.interp().spectral(method=spec_method)    
        _ = psd_mtm_rc.plot(ax=ax2)
        ax2.set_xlabel('Period')
        ax2.set_title('Power spectrum ('+spec_method+')',weight='bold')
        
        # plot T-EOF
        ax3 = fig.add_subplot(gs[1, 1])
        #EOF = self.eigvecs[:,mode]
        ax3.set_title('Spatial loadings \n (under construction)',weight='bold')

        # if title is not None:
        #     title_kwargs = {} if title_kwargs is None else title_kwargs.copy()
        #     t_args = {'y': 1.1, 'weight': 'bold'}
        #     t_args.update(title_kwargs)
        #     ax.set_title(title, **t_args)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)
        # else:
        #     if not mute:
        #         plotting.showfig(fig)
        return fig, gs



class SsaRes:
    ''' Class to hold the results of SSA method

    Parameters
    ----------

    eigvals: float (M, 1)
        a vector of real eigenvalues derived from the signal
    
    pctvar: float (M, 1)
        same vector, expressed in % variance accounted for by each mode. 
    
    eigvals_q: float (M, 2)
        array containing the 5% and 95% quantiles of the Monte-Carlo eigenvalue spectrum [ assigned NaNs if unused ]
        
    eigvecs : float (M, M)
        a matrix of the temporal eigenvectors (T-EOFs), i.e. the temporal patterns that explain most of the variations in the original series.

    PC : float (N - M + 1, M) 
        array of principal components, i.e. the loadings that, convolved with the T-EOFs, produce the reconstructed components, or RCs

    RCmat : float (N,  M) 
        array of reconstructed components, One can think of each RC as the contribution of each mode to the timeseries, weighted by their eigenvalue (loosely speaking, their "amplitude"). Summing over all columns of RC recovers the original series. (synthesis, the reciprocal operation of analysis).

    mode_idx: list 
        index of retained modes 
        
    RCseries : float (N, 1)
        reconstructed series based on the RCs of mode_idx (scaled to original series; mean must be added after the fact)


    See also
    --------

    pyleoclim.utils.decomposition.ssa : Singular Spectrum Analysis
    '''
    def __init__(self, time, original, name, eigvals, eigvecs, pctvar, PC, RCmat, RCseries,mode_idx, eigvals_q=None):
        self.time       = time
        self.original   = original
        self.name       = name
        self.eigvals    = eigvals
        self.eigvals_q  = eigvals_q
        self.eigvecs    = eigvecs
        self.pctvar     = pctvar
        self.PC         = PC
        self.RCseries   = RCseries
        self.RCmat      = RCmat
        self.mode_idx   = mode_idx
        
        

    def screeplot(self, figsize=[6, 4], title='SSA scree plot', ax=None, savefig_settings=None, title_kwargs=None, xlim=None,
             clr_mcssa=sns.xkcd_rgb['red'], clr_signif=sns.xkcd_rgb['teal'],
             clr_eig='black',  mute=False):
        ''' Scree plot for SSA, visualizing the eigenvalue spectrum and indicating which modes were retained.  

        Parameters
        ----------
        figsize : list, optional
            The figure size. The default is [6, 4].

        title : str, optional
            Plot title. The default is 'SSA scree plot'.

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        title_kwargs : dict
            the keyword arguments for ax.set_title()

        ax : matplotlib.axis, optional
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        mute : {True,False}
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)

        xlim : list, optional
            x-axis limits. The default is None.
            
        clr_mcssa : str, optional
            color of the Monte Carlo SSA AR(1) shading (if data are provided)
            default: red
            
        clr_eig : str, optional
            color of the eigenvalues, default: black
            
        clr_signif: str, optional 
            color of the highlights for significant eigenvalue.
               default: teal 

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        v = self.eigvals
        n = self.PC.shape[0] #sample size
        dv = v*np.sqrt(2/(n-1)) 
        idx = np.arange(len(v))+1 
        if self.eigvals_q is not None:
            plt.fill_between(idx,self.eigvals_q[:,0],self.eigvals_q[:,1], color=clr_mcssa, alpha = 0.3, label='AR(1) 5-95% quantiles') 
            
        plt.errorbar(x=idx,y=v,yerr = dv, color=clr_eig,marker='o',ls='',alpha=1.0,label=self.name)
        plt.plot(idx[self.mode_idx],v[self.mode_idx],color=clr_signif,marker='o',ls='',
                 markersize=4, label='modes retained',zorder=10)
        plt.title(title,fontweight='bold'); plt.legend() 
        plt.xlabel(r'Mode index $i$'); plt.ylabel(r'$\lambda_i$')    
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # enforce integer values


        if xlim is not None:
            ax.set_xlim(0.5,min(max(xlim),len(v)))

        if title is not None:
            title_kwargs = {} if title_kwargs is None else title_kwargs.copy()
            t_args = {'y': 1.1, 'weight': 'bold'}
            t_args.update(title_kwargs)
            ax.set_title(title, **t_args)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)
        # else:
        #     if not mute:
        #         plotting.showfig(fig)
        return fig, ax

    def modeplot(self, index=0, figsize=[10, 5], ax=None, savefig_settings=None, 
             title_kwargs=None, mute=False, spec_method = 'mtm', plot_original=False):
        ''' Dashboard visualizing the properties of a given SSA mode, including:
            1. the analyzing function (T-EOF)
            2. the reconstructed component (RC)
            3. its spectrum

        Parameters
        ----------
        index : int
            the (0-based) index of the mode to visualize. 
            Default is 0, corresponding to the first mode. 
        
        figsize : list, optional
            The figure size. The default is [10, 5].
        
        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        title_kwargs : dict
            the keyword arguments for ax.set_title()

        ax : matplotlib.axis, optional
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        mute : {True,False}
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            (going to be deprecated)

        spec_method: str, optional
            The name of the spectral method to be applied on the PC. Default: MTM
            Note that the data are evenly-spaced, so any spectral method that
            assumes even spacing is applicable here:  'mtm', 'welch', 'periodogram'
            'wwz' is relevant too if scaling exponents need to be estimated. 
      
        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        RC = self.RCmat[:,index]    
        fig = plt.figure(tight_layout=True,figsize=figsize)
        gs = gridspec.GridSpec(2, 2)
        # plot RC
        ax = fig.add_subplot(gs[0, :])
        
        ax.plot(self.time,RC,label='mode '+str(index+1),zorder=99) 
        if plot_original:
            ax.plot(self.time,self.original,color='Silver',lw=1,label='original')
            ax.legend()
        ax.set_xlabel('Time'),  ax.set_ylabel('RC')
        ax.set_title('SSA Mode '+str(index+1)+' RC, '+ '{:3.2f}'.format(self.pctvar[index]) + '% variance explained',weight='bold')
        # plot T-EOF
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(self.eigvecs[:,index])
        ax.set_title('Analyzing function')
        ax.set_xlabel('Time'), ax.set_ylabel('T-EOF')
        # plot spectrum
        ax = fig.add_subplot(gs[1, 1])
        ts_rc = Series(time=self.time, value=RC) # define timeseries object for the RC
        psd_mtm_rc = ts_rc.interp().spectral(method=spec_method)
        _ = psd_mtm_rc.plot(ax=ax)
        ax.set_xlabel('Period')
        ax.set_title('Spectrum ('+spec_method+')')

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)
        # else:
        #     if not mute:
        #         plotting.showfig(fig)
        return fig, ax


class Lipd:
    '''Create a Lipd object from Lipd Files

    Parameters
    ----------

    usr_path : str
        path to the Lipd file(s). Can be URL (LiPD utilities only support loading one file at a time from a URL)
        If it's a URL, it must start with "http", "https", or "ftp".

    lidp_dict : dict
        LiPD files already loaded into Python through the LiPD utilities

    validate : bool
        Validate the LiPD files upon loading. Note that for a large library this can take up to half an hour.

    remove : bool
        If validate is True and remove is True, ignores non-valid Lipd files. Note that loading unvalidated Lipd files may result in errors for some functionalities but not all.

    TODO
    ----

    Support querying the LinkedEarth platform

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        import pyleoclim as pyleo
        url='http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
        d=pyleo.Lipd(usr_path=url)
    '''

    def __init__(self, usr_path=None, lipd_dict=None, validate=False, remove=False):
        self.plot_default = {'ice-other': ['#FFD600','h'],
                'ice/rock': ['#FFD600', 'h'],
                'coral': ['#FF8B00','o'],
                'documents':['k','p'],
                'glacierice':['#86CDFA', 'd'],
                'hybrid': ['#00BEFF','*'],
                'lakesediment': ['#4169E0','s'],
                'marinesediment': ['#8A4513', 's'],
                'sclerosponge' : ['r','o'],
                'speleothem' : ['#FF1492','d'],
                'wood' : ['#32CC32','^'],
                'molluskshells' : ['#FFD600','h'],
                'peat' : ['#2F4F4F','*'],
                'midden' : ['#824E2B','o'],
                'other':['k','o']}

        if validate==False and remove==True:
            print('Removal of unvalidated LiPD files require validation')
            validate=True

        #prepare the dictionaries for all possible scenarios
        if usr_path!=None:
            # since readLipd() takes only absolute path and it will change the current working directory (CWD) without turning back,
            # we need to record CWD manually and turn back after the data loading is finished
            cwd = os.getcwd()
            if usr_path[:4] == 'http' or usr_path[:3] == 'ftp':
                # URL
                D_path = lpd.readLipd(usr_path)
            else:
                # local path
                abs_path = os.path.abspath(usr_path)
                D_path = lpd.readLipd(abs_path)

            os.chdir(cwd)

            #make sure that it's more than one
            if 'archiveType' in D_path.keys():
                D_path={D_path['dataSetName']:D_path}
            if validate==True:
                cwd = os.getcwd()
                res=lpd.validate(D_path,detailed=False)
                os.chdir(cwd)
                if remove == True:
                    for item in res:
                        if item['status'] == 'FAIL':
                           c=item['feedback']['errMsgs']
                           check = []
                           for i in c:
                               if i.startswith('Mismatched columns'):
                                   check.append(1)
                               else: check.append(0)
                           if 0 in check:
                               del D_path[item['filename'].strip('.lpd')]
        else:
            D_path={}
        if lipd_dict!=None:
            D_dict=lipd_dict
            if 'archiveType' in D_dict.keys():
                D_dict={D_dict['dataSetName']:D_dict}
            if validate==True:
                cwd = os.getcwd()
                res=lpd.validate(D_dict,detailed=False)
                os.chdir(cwd)
                if remove == True:
                    for item in res:
                        if item['status'] == 'FAIL':
                           c=item['feedback']['errMsgs']
                           check = []
                           for i in c:
                               if i.startswith('Mismatched columns'):
                                   check.append(1)
                               else: check.append(0)
                           if 0 in check:
                               del D_dict[item['filename'].strip('.lpd')]
        else:
            D_dict={}

        # raise an error if empty
        if not bool(D_dict) and not bool(D_path) == True:
            raise ValueError('No valid files; try without validation.')
        #assemble
        self.lipd={}
        self.lipd.update(D_path)
        self.lipd.update(D_dict)

    def __repr__(self):
        return str(self.__dict__)

    def copy(self):
        '''Copy the object
        '''
        return deepcopy(self)

    def to_tso(self):
        '''Extracts all the timeseries objects to a list of LiPD tso

        Returns
        -------
        ts_list : list
            List of Lipd timeseries objects as defined by LiPD utilities

        See also
        --------

        pyleoclim.ui.LipdSeries : LiPD Series object.


        '''
        cwd = os.getcwd()
        ts_list=lpd.extractTs(self.__dict__['lipd'])
        os.chdir(cwd)
        return ts_list

    def extract(self,dataSetName):
        '''
        Parameters
        ----------
        dataSetName : str
            Extract a particular dataset

        Returns
        -------
        new : pyleoclim.Lipd
            A new object corresponding to a particular dataset

        '''
        new = self.copy()
        try:
            dict_out=self.__dict__['lipd'][dataSetName]
            new.lipd=dict_out
        except:
            pass

        return new

    def to_LipdSeriesList(self, mode='paleo'):
        '''Extracts all LiPD timeseries objects to a list of LipdSeries objects

        Parameters
        ----------

        mode : {'paleo','chron'}
            Whether to extract the timeseries information from the paleo tables or chron tables

        Returns
        -------
        res : list
            A list of LiPDSeries objects

        See also
        --------
        pyleoclim.ui.LipdSeries : LipdSeries object

        '''
        cwd = os.getcwd()
        ts_list=lpd.extractTs(self.__dict__['lipd'], mode=mode)
        os.chdir(cwd)

        res=[]

        for idx, item in enumerate(ts_list):
            try:
                res.append(LipdSeries(item))
            except:
                if mode == 'paleo':
                    txt = 'The timeseries from ' + str(idx) + ': ' +\
                            item['dataSetName'] + ': ' + \
                            item['paleoData_variableName'] + \
                            ' could not be coerced into a LipdSeries object, passing'
                else:
                    txt = 'The timeseries from ' + str(idx) + ': ' +\
                            item['dataSetName'] + ': ' + \
                            item['chronData_variableName'] + \
                            ' could not be coerced into a LipdSeries object, passing'  
                warnings.warn(txt)
                pass

        return res

    def to_LipdSeries(self, number = None, mode = 'paleo'):
        '''Extracts one timeseries from the Lipd object

        Note that this function may require user interaction.

        Parameters
        ----------

        number : int
            the number of the timeseries object

        mode : {'paleo','chron'}
            whether to extract the paleo or chron series.

        Returns
        -------
        ts : pyleoclim.LipdSeries
            A LipdSeries object

        See also
        --------
        pyleoclim.ui.LipdSeries : LipdSeries object

        '''
        cwd = os.getcwd()
        ts_list = lpd.extractTs(self.__dict__['lipd'], mode=mode)
        os.chdir(cwd)
        if number is None:
            ts = LipdSeries(ts_list)
        else:
            try:
                number = int(number)
            except:
                raise TypeError('Number needs to be an integer or should be coerced into an integer.')
            ts = LipdSeries(ts_list[number])
        return ts



    def mapAllArchive(self, projection = 'Robinson', proj_default = True,
           background = True,borders = False, rivers = False, lakes = False,
           figsize = None, ax = None, marker=None, color=None,
           markersize = None, scatter_kwargs=None,
           legend=True, lgd_kwargs=None, savefig_settings=None, mute=False):

        '''Map the records contained in LiPD files by archive type

        Parameters
        ----------
        projection : str, optional
            The projection to use. The default is 'Robinson'.
        proj_default : bool, optional
            Wether to use the Pyleoclim defaults for each projection type. The default is True.
        background : bool, optional
            Wether to use a backgound. The default is True.
        borders : bool, optional
            Draw borders. The default is False.
        rivers : bool, optional
            Draw rivers. The default is False.
        lakes : bool, optional
            Draw lakes. The default is False.
        figsize : list, optional
            The size of the figure. The default is None.
        ax : matplotlib.ax, optional
            The matplotlib axis onto which to return the map. The default is None.
        marker : str, optional
            The marker type for each archive. The default is None. Uses plot_default
        color : str, optional
            Color for each acrhive. The default is None. Uses plot_default
        markersize : float, optional
            Size of the marker. The default is None.
        scatter_kwargs : dict, optional
            Parameters for the scatter plot. The default is None.
        legend : bool, optional
            Whether to plot the legend. The default is True.
        lgd_kwargs : dict, optional
            Arguments for the legend. The default is None.
        savefig_settings : dictionary, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax. The default is False.
            (going to be deprecated)

        Returns
        -------
        res : figure
            The figure

        See also
        --------

        pyleoclim.utils.mapping.map_all : Underlying mapping function for Pyleoclim

        Examples
        --------

        For speed, we are only using one LiPD file. But these functions can load and map multiple.

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            @savefig mapallarchive.png
            fig, ax = data.mapAllArchive()
            pyleo.closefig(fig)

        Change the markersize

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            @savefig mapallarchive_marker.png
            fig, ax = data.mapAllArchive(markersize=100)
            pyleo.closefig(fig)


        '''
        scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()


        #get the information from the LiPD dict
        lat=[]
        lon=[]
        archiveType=[]

        for idx, key in enumerate(self.lipd):
            d = self.lipd[key]
            lat.append(d['geo']['geometry']['coordinates'][1])
            lon.append(d['geo']['geometry']['coordinates'][0])
            if 'archiveType' in d.keys():
                archiveType.append(lipdutils.LipdToOntology(d['archiveType']).lower().replace(" ",""))
            else:
                archiveType.append('other')

        # make sure criteria is in the plot_default list
        for idx,val in enumerate(archiveType):
            if val not in self.plot_default.keys():
                archiveType[idx] = 'other'

        if markersize is not None:
            scatter_kwargs.update({'s': markersize})

        if marker==None:
            marker=[]
            for item in archiveType:
                marker.append(self.plot_default[item][1])

        if color==None:
            color=[]
            for item in archiveType:
                color.append(self.plot_default[item][0])

        res = mapping.map_all(lat=lat, lon=lon, criteria=archiveType,
                              marker=marker, color =color,
                              projection = projection, proj_default = proj_default,
                              background = background,borders = borders,
                              rivers = rivers, lakes = lakes,
                              figsize = figsize, ax = ax,
                              scatter_kwargs=scatter_kwargs, legend=legend,
                              lgd_kwargs=lgd_kwargs,savefig_settings=savefig_settings,
                              mute=mute)

        return res

class LipdSeries(Series):
    '''Lipd time series object


    These objects can be obtained from a LiPD file/object either through Pyleoclim or the LiPD utilities.
    If multiple objects (i.e., a list) is given, then the user will be prompted to choose one timeseries.

    LipdSeries is a child of Series, therefore all the methods available for Series apply to LipdSeries in addition to some specific methods.

    Examples
    --------

    In this example, we will import a LiPD file and explore the various options to create a series object.

    First, let's look at the Lipd.to_tso option. This method is attractive because the object is a list of dictionaries that are easily explored in Python.

    .. ipython:: python
        :okwarning:
        :okexcept:
            
        import pyleoclim as pyleo
        url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
        data = pyleo.Lipd(usr_path = url)
        ts_list = data.to_tso()
        # Print out the dataset name and the variable name
        for item in ts_list:
            print(item['dataSetName']+': '+item['paleoData_variableName'])
        # Load the sst data into a LipdSeries. Since Python indexing starts at zero, sst has index 5.
        ts = pyleo.LipdSeries(ts_list[5])

    If you attempt to pass the full list of series, Pyleoclim will prompt you to choose a series by printing out something similar as above.
    If you already now the number of the timeseries object you're interested in, then you should use the following:

    .. ipython:: python
        :okwarning:
        :okexcept:    

        ts1 = data.to_LipdSeries(number=5)

    If number is not specified, Pyleoclim will prompt you for the number automatically.

    Sometimes, one may want to create a MultipleSeries object from a collection of LiPD files. In this case, we recommend using the following:

    .. ipython:: python
        :okwarning:
        :okexcept:    

        ts_list = data.to_LipdSeriesList()
        # only keep the Mg/Ca and SST
        ts_list=ts_list[4:]
        #create a MultipleSeries object
        ms=pyleo.MultipleSeries(ts_list)


    '''
    def __init__(self, tso, clean_ts=True, verbose=False):
        if type(tso) is list:
            self.lipd_ts=lipdutils.getTs(tso)
        else:
            self.lipd_ts=tso

        self.plot_default = {'ice-other': ['#FFD600','h'],
                'ice/rock': ['#FFD600', 'h'],
                'coral': ['#FF8B00','o'],
                'documents':['k','p'],
                'glacierice':['#86CDFA', 'd'],
                'hybrid': ['#00BEFF','*'],
                'lakesediment': ['#4169E0','s'],
                'marinesediment': ['#8A4513', 's'],
                'sclerosponge' : ['r','o'],
                'speleothem' : ['#FF1492','d'],
                'wood' : ['#32CC32','^'],
                'molluskshells' : ['#FFD600','h'],
                'peat' : ['#2F4F4F','*'],
                'midden' : ['#824E2B','o'],
                'other':['k','o']}
        
        try:
            time, label= lipdutils.checkTimeAxis(self.lipd_ts)
            if label=='age':
                time_name='Age'
                if 'ageUnits' in self.lipd_ts.keys():
                    time_unit=self.lipd_ts['ageUnits']
                else:
                    time_unit=None
            elif label=='year':
                time_name='Year'
                if 'yearUnits' in self.lipd_ts.keys():
                    time_unit=self.lipd_ts['yearUnits']
                else:
                    time_unit=None
            try:
                if self.lipd_ts['mode'] == 'paleoData':
                    value=np.array(self.lipd_ts['paleoData_values'],dtype='float64')
                    value_name=self.lipd_ts['paleoData_variableName']
                    if 'paleoData_units' in self.lipd_ts.keys():
                        value_unit=self.lipd_ts['paleoData_units']
                    else:
                        value_unit=None
                    label=self.lipd_ts['dataSetName']
                    super(LipdSeries,self).__init__(time=time,value=value,time_name=time_name,
                         time_unit=time_unit,value_name=value_name,value_unit=value_unit,
                         label=label,clean_ts=clean_ts,verbose=verbose)
                elif self.lipd_ts['mode'] == 'chronData':
                    value=np.array(self.lipd_ts['chronData_values'],dtype='float64')
                    value_name=self.lipd_ts['chronData_variableName']
                    if 'paleoData_units' in self.lipd_ts.keys():
                        value_unit=self.lipd_ts['chronData_units']
                    else:
                        value_unit=None
                    label=self.lipd_ts['dataSetName']
                    super(LipdSeries,self).__init__(time=time,value=value,time_name=time_name,
                         time_unit=time_unit,value_name=value_name,value_unit=value_unit,
                         label=label,clean_ts=clean_ts,verbose=verbose)
            except:
                raise ValueError("paleoData_values should contain floats")
        except:
            raise KeyError("No time information present")


    def copy(self):
        '''Copy the object
        '''
        return deepcopy(self)

    def chronEnsembleToPaleo(self,D,number=None,chronNumber=None, modelNumber=None,tableNumber=None):
        '''Fetch chron ensembles from a lipd object and return the ensemble as MultipleSeries

        Parameters
        ----------
        D : a LiPD object
        number: int, optional
            The number of ensemble members to store. Default is None, which corresponds to all present
        chronNumber: int, optional
            The chron object number. The default is None. 
        modelNumber : int, optional
            Age model number. The default is None.
        tableNumber : int, optional
            Table Number. The default is None.

        Raises
        ------
        ValueError

        Returns
        -------
        ms : pyleoclim.EnsembleSeries
            An EnsembleSeries object with each series representing a possible realization of the age model

        '''
        #get the corresponding LiPD
        dataSetName=self.lipd_ts['dataSetName']
        if type(D) is dict:
            try:
                lipd=D[dataSetName]
            except:
                lipd=D
        else:
            a=D.extract(dataSetName)
            lipd=a.__dict__['lipd']
        #Look for the ensemble and get values
        cwd = os.getcwd()
        csv_dict=lpd.getCsv(lipd)
        os.chdir(cwd)
        chron,paleo = lipdutils.isEnsemble(csv_dict)
        if len(chron)==0:
            raise ValueError("No ChronMeasurementTables available")
        elif len(chron)>1:
            if chronNumber==None or modelNumber==None or tableNumber==None:
                csvName=lipdutils.whichEnsemble(chron)
            else:
                str0='chron'+str(chronNumber)
                str1='model'+str(modelNumber)
                str2='ensemble'+str(tableNumber)
                for item in chron:
                    if str0 in item and str1 in item and str2 in item:
                        csvName=item
            depth, ensembleValues =lipdutils.getEnsemble(csv_dict,csvName)
        else:
            depth, ensembleValues =lipdutils.getEnsemble(csv_dict,chron[0])
        #make sure it's sorted
        sort_ind = np.argsort(depth)
        depth=list(np.array(depth)[sort_ind])
        ensembleValues=ensembleValues[sort_ind,:]
        
        if number is not None:
            if number>np.shape(ensembleValues)[1]:
                warnings.warn('Selected number of ensemble members is greater than number of members in the ensemble table; passing')
                pass
            else:
                ensembleValues=ensembleValues[:,0:number]
        
        #Map to paleovalues
        key=[]
        for item in self.lipd_ts.keys():
            if 'depth' in item and 'Units' not in item:
                key.append(item)
        key=key[0]
        ds= np.array(self.lipd_ts[key],dtype='float64')
        if 'paleoData_values' in self.lipd_ts.keys():
            ys= np.array(self.lipd_ts['paleoData_values'],dtype='float64')
        elif 'chronData_values' in self.lipd_ts.keys():
            ys= np.array(self.lipd_ts['chronData_values'],dtype='float64')
        else:
            raise KeyError('no y-axis values available')
        #Remove NaNs
        ys_tmp=np.copy(ys)
        ds=ds[~np.isnan(ys_tmp)]
        sort_ind2=np.argsort(ds)
        ds=np.array(ds[sort_ind2])
        ys=np.array(ys[sort_ind2])
        ensembleValuestoPaleo=lipdutils.mapAgeEnsembleToPaleoData(ensembleValues, depth, ds)
        #create multipleseries
        s_list=[]
        for i, s in enumerate(ensembleValuestoPaleo.T):
            s_tmp = Series(time=s,value=ys,
                           verbose=i==0, 
                           clean_ts=False, 
                           value_name=self.value_name,
                           value_unit=self.value_unit,
                           time_name=self.time_name,
                           time_unit=self.time_unit)
            s_list.append(s_tmp)

        ens = EnsembleSeries(series_list=s_list)

        return ens

    def map(self,projection = 'Orthographic', proj_default = True,
           background = True,borders = False, rivers = False, lakes = False,
           figsize = None, ax = None, marker=None, color=None,
           markersize = None, scatter_kwargs=None,
           legend=True, lgd_kwargs=None, savefig_settings=None, mute=False):
        '''Map the location of the record

        Parameters
        ----------
        projection : str, optional
            The projection to use. The default is 'Robinson'.
        proj_default : bool, optional
            Wether to use the Pyleoclim defaults for each projection type. The default is True.
        background : bool, optional
            Wether to use a backgound. The default is True.
        borders : bool, optional
            Draw borders. The default is False.
        rivers : bool, optional
            Draw rivers. The default is False.
        lakes : bool, optional
            Draw lakes. The default is False.
        figsize : list, optional
            The size of the figure. The default is None.
        ax : matplotlib.ax, optional
            The matplotlib axis onto which to return the map. The default is None.
        marker : str, optional
            The marker type for each archive. The default is None. Uses plot_default
        color : str, optional
            Color for each acrhive. The default is None. Uses plot_default
        markersize : float, optional
            Size of the marker. The default is None.
        scatter_kwargs : dict, optional
            Parameters for the scatter plot. The default is None.
        legend : bool, optional
            Whether to plot the legend. The default is True.
        lgd_kwargs : dict, optional
            Arguments for the legend. The default is None.
        savefig_settings : dictionary, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax. The default is False.
            (going to be deprecated)

        Returns
        -------
        res : fig,ax

        See also
        --------

        pyleoclim.utils.mapping.map_all : Underlying mapping function for Pyleoclim

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            ts = data.to_LipdSeries(number=5)
            @savefig mapone.png
            fig, ax = ts.map()
            pyleo.closefig(fig)

        '''
        scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()
        #get the information from the timeseries
        lat=[self.lipd_ts['geo_meanLat']]
        lon=[self.lipd_ts['geo_meanLon']]
        
        if 'archiveType' in self.lipd_ts.keys():
            archiveType=lipdutils.LipdToOntology(self.lipd_ts['archiveType']).lower().replace(" ","")
        else:
            archiveType='other'

        # make sure criteria is in the plot_default list
        if archiveType not in self.plot_default.keys():
            archiveType = 'other'

        if markersize is not None:
            scatter_kwargs.update({'s': markersize})

        if marker==None:
            marker= self.plot_default[archiveType][1]

        if color==None:
            color=self.plot_default[archiveType][0]

        if proj_default==True:
            proj1={'central_latitude':lat[0],
                   'central_longitude':lon[0]}
            proj2={'central_latitude':lat[0]}
            proj3={'central_longitude':lon[0]}

        archiveType=[archiveType] #list so it will work with map_all
        marker=[marker]
        color=[color]

        if proj_default==True:

            try:
                res = mapping.map_all(lat=lat, lon=lon, criteria=archiveType,
                              marker=marker, color =color,
                              projection = projection, proj_default = proj1,
                              background = background,borders = borders,
                              rivers = rivers, lakes = lakes,
                              figsize = figsize, ax = ax,
                              scatter_kwargs=scatter_kwargs, legend=legend,
                              lgd_kwargs=lgd_kwargs,savefig_settings=savefig_settings,
                              mute=mute)

            except:
                try:
                    res = mapping.map_all(lat=lat, lon=lon, criteria=archiveType,
                              marker=marker, color =color,
                              projection = projection, proj_default = proj3,
                              background = background,borders = borders,
                              rivers = rivers, lakes = lakes,
                              figsize = figsize, ax = ax,
                              scatter_kwargs=scatter_kwargs, legend=legend,
                              lgd_kwargs=lgd_kwargs,savefig_settings=savefig_settings,
                              mute=mute)
                except:
                    res = mapping.map_all(lat=lat, lon=lon, criteria=archiveType,
                              marker=marker, color =color,
                              projection = projection, proj_default = proj2,
                              background = background,borders = borders,
                              rivers = rivers, lakes = lakes,
                              figsize = figsize, ax = ax,
                              scatter_kwargs=scatter_kwargs, legend=legend,
                              lgd_kwargs=lgd_kwargs,savefig_settings=savefig_settings,
                              mute=mute)

        else:
            res = mapping.map_all(lat=lat, lon=lon, criteria=archiveType,
                              marker=marker, color =color,
                              projection = projection, proj_default = proj_default,
                              background = background,borders = borders,
                              rivers = rivers, lakes = lakes,
                              figsize = figsize, ax = ax,
                              scatter_kwargs=scatter_kwargs, legend=legend,
                              lgd_kwargs=lgd_kwargs,savefig_settings=savefig_settings,
                              mute=mute)
        return res

    def getMetadata(self):

        """ Get the necessary metadata for the ensemble plots

        Parameters
        ----------

        timeseries : object
                    a specific timeseries object.

        Returns
        -------

        res : dict
                  A dictionary containing the following metadata:
                    archiveType
                    Authors (if more than 2, replace by et al)
                    PublicationYear
                    Publication DOI
                    Variable Name
                    Units
                    Climate Interpretation
                    Calibration Equation
                    Calibration References
                    Calibration Notes

        """

        # Get all the necessary information
        # Top level information
        if "archiveType" in self.lipd_ts.keys():
            archiveType = self.lipd_ts["archiveType"]
        else:
            archiveType = "NA"

        if "pub1_author" in self.lipd_ts.keys():
            authors = self.lipd_ts["pub1_author"]
        else:
            authors = "NA"

        #Truncate if more than two authors
        idx = [pos for pos, char in enumerate(authors) if char == ";"]
        if  len(idx)>2:
            authors = authors[0:idx[1]+1] + "et al."

        if "pub1_year" in self.lipd_ts.keys():
            Year = str(self.lipd_ts["pub1_year"])
        else:
            Year = "NA"

        if "pub1_doi" in self.lipd_ts.keys():
            DOI = self.lipd_ts["pub1_doi"]
        else:
            DOI = "NA"
        
        if self.lipd_ts['mode'] == 'paleoData':
            prefix = 'paleo'
        else:
            prefix = 'chron'

        if prefix+"Data_InferredVariableType" in self.lipd_ts.keys():
            if type(self.lipd_ts[prefix+"Data_InferredVariableType"]) is list:
                Variable = self.lipd_ts[prefix+"Data_InferredVariableType"][0]
            else:
                Variable = self.lipd_ts[prefix+"Data_InferredVariableType"]
        elif prefix+"Data_ProxyObservationType" in self.lipd_ts.keys():
            if type(self.lipd_ts[prefix+"Data_ProxyObservationType"]) is list:
                Variable = self.lipd_ts[prefix+"Data_ProxyObservationType"][0]
            else:
                Variable = self.lipd_ts[prefix+"Data_ProxyObservationType"]
        else:
            Variable = self.lipd_ts[prefix+"Data_variableName"]

        if prefix+"Data_units" in self.lipd_ts.keys():
            units = self.lipd_ts[prefix+"Data_units"]
        else:
            units = "NA"

        #Climate interpretation information
        if prefix+"Data_interpretation" in self.lipd_ts.keys():
            interpretation = self.lipd_ts[prefix+"Data_interpretation"][0]
            if "name" in interpretation.keys():
                ClimateVar = interpretation["name"]
            elif "variable" in interpretation.keys():
                ClimateVar = interpretation["variable"]
            else:
                ClimateVar = "NA"
            if "detail" in interpretation.keys():
                Detail = interpretation["detail"]
            elif "variableDetail" in interpretation.keys():
                Detail = interpretation['variableDetail']
            else:
                Detail = "NA"
            if "scope" in interpretation.keys():
                Scope = interpretation['scope']
            else:
                Scope = "NA"
            if "seasonality" in interpretation.keys():
                Seasonality = interpretation["seasonality"]
            else:
                Seasonality = "NA"
            if "interpdirection" in interpretation.keys():
                Direction = interpretation["interpdirection"]
            else:
                Direction = "NA"
        else:
            ClimateVar = "NA"
            Detail = "NA"
            Scope = "NA"
            Seasonality = "NA"
            Direction = "NA"

        # Calibration information
        if prefix+"Data_calibration" in self.lipd_ts.keys():
            calibration = self.lipd_ts[prefix+'Data_calibration'][0]
            if "equation" in calibration.keys():
                Calibration_equation = calibration["equation"]
            else:
                Calibration_equation = "NA"
            if  "calibrationReferences" in calibration.keys():
                ref = calibration["calibrationReferences"]
                if "author" in ref.keys():
                    ref_author = ref["author"][0] # get the first author
                else:
                    ref_author = "NA"
                if  "publicationYear" in ref.keys():
                    ref_year = str(ref["publicationYear"])
                else: ref_year="NA"
                Calibration_notes = ref_author +"."+ref_year
            elif "notes" in calibration.keys():
                Calibration_notes = calibration["notes"]
            else: Calibration_notes = "NA"
        else:
            Calibration_equation = "NA"
            Calibration_notes = "NA"

        #Truncate the notes if too long
        charlim = 30;
        if len(Calibration_notes)>charlim:
            Calibration_notes = Calibration_notes[0:charlim] + " ..."

        res = {"archiveType" : archiveType,
                    "authors" : authors,
                    "Year": Year,
                    "DOI": DOI,
                    "Variable": Variable,
                    "units": units,
                    "Climate_Variable" : ClimateVar,
                    "Detail" : Detail,
                    "Scope":Scope,
                    "Seasonality" : Seasonality,
                    "Interpretation_Direction" : Direction,
                    "Calibration_equation" : Calibration_equation,
                    "Calibration_notes" : Calibration_notes}

        return res

    def dashboard(self, figsize = [11,8], plt_kwargs=None, distplt_kwargs=None, spectral_kwargs=None,
                  spectralsignif_kwargs=None, spectralfig_kwargs=None, map_kwargs=None, metadata = True,
                  savefig_settings=None, mute=False, ensemble = False, D=None):
        '''


        Parameters
        ----------
        figsize : list, optional
            Figure size. The default is [11,8].
        plt_kwargs : dict, optional
            Optional arguments for the timeseries plot. See Series.plot() or EnsembleSeries.plot_envelope(). The default is None.
        distplt_kwargs : dict, optional
            Optional arguments for the distribution plot. See Series.distplot() or EnsembleSeries.plot_displot(). The default is None.
        spectral_kwargs : dict, optional
            Optional arguments for the spectral method. Default is to use Lomb-Scargle method. See Series.spectral() or EnsembleSeries.spectral(). The default is None.
        spectralsignif_kwargs : dict, optional
            Optional arguments to estimate the significance of the power spectrum. See PSD.signif_test. Note that we currently do not support significance testing for ensembles. The default is None.
        spectralfig_kwargs : dict, optional
            Optional arguments for the power spectrum figure. See PSD.plot() or MultiplePSD.plot_envelope(). The default is None.
        map_kwargs : dict, optional
            Optional arguments for the map. See LipdSeries.map(). The default is None.
        metadata : {True,False}, optional
            Whether or not to produce a dashboard with printed metadata. The default is True.
        savefig_settings : dict, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}.
            The default is None.
        mute : {True,False}, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax. The default is False.
            (going to be deprecated)
        ensemble : {True, False}, optional
            If True, will return the dashboard in ensemble modes if ensembles are available
        D : pyleoclim.Lipd
            If asking for an ensemble plot, a pyleoclim.Lipd object must be provided

        Returns
        -------
        fig : matplotlib.figure
            The figure
        ax : matplolib.axis
            The axis.

        See also
        --------

        pyleoclim.Series.plot : plot a timeseries
        
        pyleoclim.EnsembleSeries.plot_envelope: Envelope plots for an ensemble

        pyleoclim.Series.distplot : plot a distribution of the timeseries
        
        pyleoclim.EnsembleSeries.distplot : plot a distribution of the timeseries across ensembles

        pyleoclim.Series.spectral : spectral analysis method.
        
        pyleoclim.MultipleSeries.spectral : spectral analysis method for multiple series.

        pyleoclim.PSD.signif_test : significance test for timeseries analysis

        pyleoclim.PSD.plot : plot power spectrum
        
        pyleoclim.MulitplePSD.plot : plot envelope of power spectrum

        pyleoclim.LipdSeries.map : map location of dataset

        pyleolim.LipdSeries.getMetadata : get relevant metadata from the timeseries object

        pyleoclim.utils.mapping.map_all : Underlying mapping function for Pyleoclim

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:    

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            ts = data.to_LipdSeries(number=5)
            @savefig ts_dashboard.png
            fig, ax = ts.dashboard()
            pyleo.closefig(fig)

        '''
        if ensemble == True and D is None:
            raise ValueError("When an ensemble dashboard is requested, the corresponsind Lipd object must be supplied")
            
        if ensemble == True:
            warnings.warn('Some of the computation in ensemble mode can require a few minutes to complete.')
            
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        res=self.getMetadata()
        # start plotting
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2,5)
        gs.update(left=0,right=1.1)
        
        if ensemble==True:
           ens = self.chronEnsembleToPaleo(D)
           ensc = ens.common_time()

        ax={}
       # Plot the timeseries
        plt_kwargs={} if plt_kwargs is None else plt_kwargs.copy()
        ax['ts'] = plt.subplot(gs[0,:-3])
        plt_kwargs.update({'ax':ax['ts']})
        # use the defaults if color/markers not specified
        if ensemble == False:
            if 'marker' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ","")
                plt_kwargs.update({'marker':self.plot_default[archiveType][1]})
            if 'color' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ","")
                plt_kwargs.update({'color':self.plot_default[archiveType][0]})
            ax['ts'] = self.plot(**plt_kwargs)
        elif ensemble == True:
            if 'curve_clr' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ","")
                plt_kwargs.update({'curve_clr':self.plot_default[archiveType][0]})
            if 'shade_clr' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ","")
                plt_kwargs.update({'shade_clr':self.plot_default[archiveType][0]})
            #plt_kwargs.update({'ylabel':self.value_name})
            ax['ts'] = ensc.plot_envelope(**plt_kwargs)
        else:
            raise ValueError("Invalid argument value for ensemble")
        ymin, ymax = ax['ts'].get_ylim()
            

        #plot the distplot
        distplt_kwargs={} if distplt_kwargs is None else distplt_kwargs.copy()
        ax['dts'] = plt.subplot(gs[0,2])
        distplt_kwargs.update({'ax':ax['dts']})
        distplt_kwargs.update({'ylabel':'PDF'})
        distplt_kwargs.update({'vertical':True})
        if 'color' not in distplt_kwargs.keys():
            archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ","")
            distplt_kwargs.update({'color':self.plot_default[archiveType][0]})
        if ensemble == False:   
            ax['dts'] = self.distplot(**distplt_kwargs)
        elif ensemble == True:
            ax['dts'] = ensc.distplot(**distplt_kwargs)
        ax['dts'].set_ylim([ymin,ymax])
        ax['dts'].set_yticklabels([])
        ax['dts'].set_ylabel('')
        ax['dts'].set_yticks([])

        #make the map - brute force since projection is not being returned properly
        lat=[self.lipd_ts['geo_meanLat']]
        lon=[self.lipd_ts['geo_meanLon']]

        map_kwargs={} if map_kwargs is None else map_kwargs.copy()
        if 'projection' in map_kwargs.keys():
            projection=map_kwargs['projection']
        else:
            projection='Orthographic'
        if 'proj_default' in map_kwargs.keys():
            proj_default=map_kwargs['proj_default']
        else:
            proj_default=True
        if proj_default==True:
            proj1={'central_latitude':lat[0],
                   'central_longitude':lon[0]}
            proj2={'central_latitude':lat[0]}
            proj3={'central_longitude':lon[0]}
            try:
                proj = mapping.set_proj(projection=projection, proj_default=proj1)
            except:
                try:
                    proj = mapping.set_proj(projection=projection, proj_default=proj3)
                except:
                    proj = mapping.set_proj(projection=projection, proj_default=proj2)
        if 'marker' in map_kwargs.keys():
            marker = map_kwargs['marker']
        else:
            marker = self.plot_default[archiveType][1]
        if 'color' in map_kwargs.keys():
            color = map_kwargs['color']
        else:
            color = self.plot_default[archiveType][0]
        if 'background' in map_kwargs.keys():
            background = map_kwargs['background']
        else:
            background = True
        if 'borders' in map_kwargs.keys():
            borders= map_kwargs['borders']
        else:
            borders = False
        if 'rivers' in map_kwargs.keys():
            rivers= map_kwargs['rivers']
        else:
            rivers = False
        if 'lakes' in map_kwargs.keys():
            lakes = map_kwargs['lakes']
        else:
            lakes = False
        if 'scatter_kwargs' in map_kwargs.keys():
            scatter_kwargs = map_kwargs['scatter_kwargs']
        else:
            scatter_kwargs={}
        if 'markersize' in map_kwargs.keys():
            scatter_kwargs.update({'s': map_kwargs['markersize']})
        else:
            pass
        if 'lgd_kwargs' in map_kwargs.keys():
            lgd_kwargs = map_kwargs['lgd_kwargs']
        else:
            lgd_kwargs ={}
        if 'legend' in map_kwargs.keys():
            legend = map_kwargs['legend']
        else:
            legend = False
        #make the plot map

        data_crs = ccrs.PlateCarree()
        ax['map'] = plt.subplot(gs[1,0],projection=proj)
        ax['map'].coastlines()
        if background is True:
            ax['map'].stock_img()
        #Other extra information
        if borders is True:
            ax['map'].add_feature(cfeature.BORDERS)
        if lakes is True:
            ax['map'].add_feature(cfeature.LAKES)
        if rivers is True:
            ax['map'].add_feature(cfeature.RIVERS)
        ax['map'].scatter(lon,lat,zorder=10,label=marker,facecolor=color,transform=data_crs, **scatter_kwargs)
        if legend == True:
            ax.legend(**lgd_kwargs)

        #spectral analysis
        spectral_kwargs={} if spectral_kwargs is None else spectral_kwargs.copy()
        if 'method' in spectral_kwargs.keys():
            pass
        else:
            spectral_kwargs.update({'method':'lomb_scargle'})
        if 'freq_method' in spectral_kwargs.keys():
            pass
        else:
            if ensemble == False:
                spectral_kwargs.update({'freq_method':'lomb_scargle'})
            elif ensemble == True:
                pass
        
        ax['spec'] = plt.subplot(gs[1,1:3])
        spectralfig_kwargs={} if spectralfig_kwargs is None else spectralfig_kwargs.copy()
        spectralfig_kwargs.update({'ax':ax['spec']})
        
        if ensemble == False:
            ts_preprocess = self.detrend().standardize()
            psd = ts_preprocess.spectral(**spectral_kwargs)

            #Significance test
            spectralsignif_kwargs={} if spectralsignif_kwargs is None else spectralsignif_kwargs.copy()
            psd_signif = psd.signif_test(**spectralsignif_kwargs)
           #plot
            if 'color' not in spectralfig_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ","")
                spectralfig_kwargs.update({'color':self.plot_default[archiveType][0]})
            if 'signif_clr' not in spectralfig_kwargs.keys():
                spectralfig_kwargs.update({'signif_clr':'grey'})
            ax['spec'] = psd_signif.plot(**spectralfig_kwargs)
        
        elif ensemble == True:
            if 'curve_clr' not in spectralfig_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ","")
                spectralfig_kwargs.update({'curve_clr':self.plot_default[archiveType][0]})
            if 'shade_clr' not in spectralfig_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ","")
                spectralfig_kwargs.update({'shade_clr':self.plot_default[archiveType][0]})
            psd = ensc.detrend().standardize().spectral(**spectral_kwargs)
            #plot
            ax['spec'] = psd.plot_envelope(**spectralfig_kwargs)

        #Make the plot
        

        if metadata == True:
            # get metadata
            textstr = "archiveType: " + res["archiveType"]+"\n"+"\n"+\
              "Authors: " + res["authors"]+"\n"+"\n"+\
              "Year: " + res["Year"]+"\n"+"\n"+\
              "DOI: " + res["DOI"]+"\n"+"\n"+\
              "Variable: " + res["Variable"]+"\n"+"\n"+\
              "units: " + res["units"]+"\n"+"\n"+\
              "Climate Interpretation: " +"\n"+\
              "    Climate Variable: " + res["Climate_Variable"] +"\n"+\
              "    Detail: " + res["Detail"]+"\n"+\
              "    Seasonality: " + res["Seasonality"]+"\n"+\
              "    Direction: " + res["Interpretation_Direction"]+"\n \n"+\
              "Calibration: \n" + \
              "    Equation: " + res["Calibration_equation"] + "\n" +\
              "    Notes: " + res["Calibration_notes"]
            plt.figtext(0.7, 0.4, textstr, fontsize = 12)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)
        # else:
        #     if not mute:
        #         plotting.showfig(fig)
        return fig, ax

    def mapNearRecord(self, D, n=5, radius = None, sameArchive = False, 
                      projection='Orthographic',proj_default = True,
                      background = True,borders = False, rivers = False, 
                      lakes = False, figsize = None, ax = None, 
                      marker_ref= None, color_ref=None, marker=None, color=None,
                      markersize_adjust=False, scale_factor = 100, scatter_kwargs=None,
                      legend = True, lgd_kwargs=None, savefig_settings=None, 
                      mute=False):
        """ Map records that are near the timeseries of interest
        

        Parameters
        ----------
        D : pyleoclim.Lipd
            A pyleoclim LiPD object
        n : int, optional
            The n number of closest records. The default is 5.
        radius : float, optional
            The radius to take into consideration when looking for records (in km). The default is None.
        sameArchive : {True, False}, optional
            Whether to consider records from the same archiveType as the original record. The default is False.
        projection : string, optional
            A valid cartopy projection. The default is 'Orthographic'.
        proj_default : True or dict, optional
            The projection arguments. If not True, then use a dictionary to pass the appropriate arguments depending on the projection. The default is True.
        background : {True,False}, optional
            Whether to use a background. The default is True.
        borders : {True, False}, optional
            Whether to plot country borders. The default is False.
        rivers : {True, False}, optional
            Whether to plot rivers. The default is False.
        lakes : {True, False}, optional
            Whether to plot rivers. The default is False.
        figsize : list, optional
            the size of the figure. The default is None.
        ax : matplotlib.ax, optional
            The matplotlib axis onto which to return the map. The default is None.
        marker_ref : str, optional
            Marker shape to use for the main record. The default is None, which corresponds to the default marker for the archiveType
        color_ref : str, optional
            The color for the main record. The default is None, which corresponds to the default color for the archiveType.
        marker : str or list, optional
            Marker shape to use for the other records. The default is None, which corresponds to the marker shape for each archiveType.
        color : str or list, optional
            Color for each marker. The default is None, which corresponds to the color for each archiveType
        markersize_adjust : {True, False}, optional
            Whether to adjust the marker size according to distance from record of interest. The default is False.
        scale_factor : int, optional
            The maximum marker size. The default is 100.
        scatter_kwargs : dict, optional
            Parameters for the scatter plot. The default is None.
        legend : {True, False}, optional
            Whether to show the legend. The default is True.
        lgd_kwargs : dict, optional
            Parameters for the legend. The default is None.
        savefig_settings : dict, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.
        mute : {True, False}, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax. The default is False.
            (going to be deprecated)

        See also
        --------

        pyleoclim.utils.mapping.map_all : Underlying mapping function for Pyleoclim
        
        pyleoclim.utils.mapping.dist_sphere: Calculate distance on a sphere
        
        pyleoclim.utils.mapping.compute_dist: Compute the distance between a point and an array
        
        pyleoclim.utils.mapping.within_distance: Returns point in an array within a certain distance

        Returns
        -------
        res : dict
            contains fig and ax

        """
        scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()
        
        #get the information about the original timeseries
        lat_ref=[self.lipd_ts['geo_meanLat']]
        lon_ref=[self.lipd_ts['geo_meanLon']]
        
        if 'archiveType' in self.lipd_ts.keys():
            archiveType_ref=lipdutils.LipdToOntology(self.lipd_ts['archiveType']).lower().replace(" ","")
        else:
            archiveType_ref='other'

        # make sure criteria is in the plot_default list
        if archiveType_ref not in self.plot_default.keys():
            archiveType_ref = 'other'
        
        # get information about the other timeseries
        lat=[]
        lon=[]
        archiveType=[]
        
        dataSetName_ref = self.lipd_ts['dataSetName']

        for idx, key in enumerate(D.lipd):
            if key != dataSetName_ref:
                d = D.lipd[key]
                lat.append(d['geo']['geometry']['coordinates'][1])
                lon.append(d['geo']['geometry']['coordinates'][0])
                if 'archiveType' in d.keys():
                    archiveType.append(lipdutils.LipdToOntology(d['archiveType']).lower().replace(" ",""))
                else:
                    archiveType.append('other')

        # make sure criteria is in the plot_default list
        for idx,val in enumerate(archiveType):
            if val not in self.plot_default.keys():
                archiveType[idx] = 'other'
        
        if len(lat)==0: #this should not happen unless the coordinates are not available in the LiPD file
            raise ValueError('no matching record found')
        
        # Filter by the same type of archive if asked
        if sameArchive == True:
            idx_archive = [idx for idx,val in enumerate(archiveType) if val==archiveType_ref]
            if len(idx_archive)==0:
                raise ValueError('No records corresponding to the same archiveType available. Widen your search criteria.')
            else:
                lat = [lat[idx] for idx in idx_archive]
                lon = [lon[idx] for idx in idx_archive]
                archiveType=[archiveType[idx] for idx in idx_archive]
                       
        #compute the distance
        dist = mapping.compute_dist(lat_ref,lon_ref,lat,lon)
        
        if radius: 
            idx_radius = mapping.within_distance(dist, radius)
            if len(idx_radius) == 0:
                raise ValueError('No records withing matching radius distance. Widen your search criteria')
            else:
                lat = [lat[idx] for idx in idx_radius]
                lon = [lon[idx] for idx in idx_radius]
                archiveType = [archiveType[idx] for idx in idx_radius]
                dist = [dist[idx] for idx in idx_radius]
        
        #print a warning if plotting less than asked because of the filters
        
        if n>len(dist):
            warnings.warn("Number of matching records is less"+\
              " than the number of neighbors chosen. Including all records "+\
              " in the analysis.")
            n=len(dist)
        
        #Sort the distance array
        sort_idx = np.argsort(dist)
        dist = [dist[idx] for idx in sort_idx]
        lat = [lat[idx] for idx in sort_idx]
        lon = [lon[idx] for idx in sort_idx]
        archiveType = [archiveType[idx] for idx in sort_idx]
        
        # Grab the right number of records
        dist = dist[0:n]
        lat = lat[0:n]
        lon = lon[0:n]
        archiveType = archiveType[0:n]
         
        # Get plotting information
        
        if marker_ref == None:
            marker_ref = self.plot_default[archiveType_ref][1]
        if color_ref == None:
            color_ref = self.plot_default[archiveType_ref][0] 
        
        if marker == None:
            marker=[]
            for item in archiveType:
                marker.append(self.plot_default[item][1])
        elif type(marker) ==list:
            if len(marker)!=len(lon):
                raise ValueError('When providing a list, it should be the same length as the number of records')
        elif type(marker) == str:
            marker = [marker]*len(lon)

        if color == None:
            color=[]
            for item in archiveType:
                color.append(self.plot_default[item][0])
        elif type(color) ==list:
            if len(color)!=len(lon):
                raise ValueError('When providing a list, it should be the same length as the number of records')
        elif type(color) == str:
            color = [color]*len(lon)
        
        if 'edgecolors' not in scatter_kwargs.keys():
            edgecolors = []
            for item in marker:
                edgecolors.append('w')
            edgecolors.append('k')
            scatter_kwargs.update({'edgecolors':edgecolors})
        
        #Start plotting
        lat_all = lat + lat_ref
        lon_all = lon + lon_ref
        dist_all = dist + [0]
        archiveType_all = archiveType
        archiveType_all.append(archiveType_ref) 
        
        color_all = color
        color_all.append(color_ref)
        marker_all= marker
        marker_all.append(marker_ref)
        
        if markersize_adjust == True:
            scale = dist_all[-1]/(scale_factor-30)
            s = list(np.array(dist_all)*1/(scale)+30)
            s.reverse()
            scatter_kwargs.update({'s':s})
        
        proj1={'central_latitude':lat_ref[0],
       'central_longitude':lon_ref[0]}
        proj2={'central_latitude':lat_ref[0]}
        proj3={'central_longitude':lon_ref[0]}
        
        if proj_default==True:
            try:
                res = mapping.map_all(lat=lat_all, lon=lon_all, 
                      criteria=archiveType_all,
                      marker=marker_all, color =color_all,
                      projection = projection, proj_default = proj1,
                      background = background,borders = borders,
                      rivers = rivers, lakes = lakes,
                      figsize = figsize, ax = ax,
                      scatter_kwargs=scatter_kwargs, legend=legend,
                      lgd_kwargs=lgd_kwargs,savefig_settings=savefig_settings,
                      mute=mute)
            except:
                try:
                    res = mapping.map_all(lat=lat_all, lon=lon_all, 
                      criteria=archiveType_all,
                      marker=marker_all, color =color_all,
                      projection = projection, proj_default = proj2,
                      background = background,borders = borders,
                      rivers = rivers, lakes = lakes,
                      figsize = figsize, ax = ax,
                      scatter_kwargs=scatter_kwargs, legend=legend,
                      lgd_kwargs=lgd_kwargs,savefig_settings=savefig_settings,
                      mute=mute)
                except:
                    res = mapping.map_all(lat=lat_all, lon=lon_all, 
                      criteria=archiveType_all,
                      marker=marker_all, color =color_all,
                      projection = projection, proj_default = proj3,
                      background = background,borders = borders,
                      rivers = rivers, lakes = lakes,
                      figsize = figsize, ax = ax,
                      scatter_kwargs=scatter_kwargs, legend=legend,
                      lgd_kwargs=lgd_kwargs,savefig_settings=savefig_settings,
                      mute=mute)
        
        else:
            res = mapping.map_all(lat=lat_all, lon=lon_all, 
                      criteria=archiveType_all,
                      marker=marker_all, color =color_all,
                      projection = projection, proj_default = proj_default,
                      background = background,borders = borders,
                      rivers = rivers, lakes = lakes,
                      figsize = figsize, ax = ax,
                      scatter_kwargs=scatter_kwargs, legend=legend,
                      lgd_kwargs=lgd_kwargs,savefig_settings=savefig_settings,
                      mute=mute)
        
        return res

        
    def plot_age_depth(self,figsize = [10,4], plt_kwargs=None,  
                       savefig_settings=None, mute=False, 
                       ensemble = False, D=None, num_traces = 10, ensemble_kwargs=None,
                       envelope_kwargs = None, traces_kwargs = None):
        
        '''

        Parameters
        ----------
        figsize : List, optional
            Size of the figure. The default is [10,4].
        plt_kwargs : dict, optional
            Arguments for basic plot. See Series.plot() for details. The default is None.
        savefig_settings : dict, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.
        mute : {True,False}, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax. The default is False.
            (going to be deprecated)
        ensemble : {True,False}, optional
            Whether to use age model ensembles stored in the file for the plot. The default is False.
            If no ensemble can be found, will error out.
        D : pyleoclim.Lipd, optional
            The pyleoclim.Lipd object from which the pyleoclim.LipdSeries is derived. The default is None.
        num_traces : int, optional
            Number of individual age models to plot. If not interested in plotting individual traces, set this parameter to 0 or None. The default is 10.
        ensemble_kwargs : dict, optional
            Parameters associated with identifying the chronEnsemble tables. See pyleoclim.LipdSeries.chronEnsembleToPaleo() for details. The default is None.
        envelope_kwargs : dict, optional
            Parameters to control the envelope plot. See pyleoclim.EnsembleSeries.plot_envelope() for details. The default is None.
        traces_kwargs : TYPE, optional
            Parameters to control the traces plot. See pyleoclim.EnsembleSeries.plot_traces() for details. The default is None.

        Raises
        ------
        ValueError
            In ensemble mode, make sure that the LiPD object is given 
        KeyError
            Depth information needed.

        Returns
        -------
        fig,ax
            The figure
        
        See also
        --------
        
        pyleoclim.core.ui.Lipd : Pyleoclim internal representation of a LiPD file
        pyleoclim.core.ui.Series.plot : Basic plotting in pyleoclim
        pyleoclim.core.ui.LipdSeries.chronEnsembleToPaleo : Function to map the ensemble table to a paleo depth. 
        pyleoclim.core.ui.EnsembleSeries.plot_envelope : Create an envelope plot from an ensemble
        pyleoclim.core.ui.EnsembleSeries.plot_traces : Create a trace plot from an ensemble
        
        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:    
                
            D = pyleo.Lipd('http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=Crystal.McCabe-Glynn.2013')
            ts=D.to_LipdSeries(number=2)
            ts.plot_age_depth()
            pyleo.closefig(fig)
        
        '''
        if ensemble == True and D is None:
            raise ValueError("When an ensemble is requested, the corresponsind Lipd object must be supplied")
        
        meta=self.getMetadata()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plt_kwargs={} if plt_kwargs is None else plt_kwargs.copy()
        # get depth
        try:
            value_depth, label_depth =  lipdutils.checkXaxis(self.lipd_ts,'depth')
            if 'depthUnits' in self.lipd_ts.keys():
                units_depth = self.lipd_ts['depthUnits']
            else:
                units_depth = 'NA'
        except:
            raise KeyError('No depth available in this record')
        
        # create a series for which time is actually depth
        
        if ensemble  == False:
            ts = Series(time = self.time,value=value_depth,
                              time_name=self.time_name,time_unit=self.time_unit,
                              value_name=label_depth,value_unit=units_depth)
            plt_kwargs={} if plt_kwargs is None else plt_kwargs.copy()
            if 'marker' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(meta['archiveType']).lower().replace(" ","")
                plt_kwargs.update({'marker':self.plot_default[archiveType][1]})
            if 'color' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(meta['archiveType']).lower().replace(" ","")
                plt_kwargs.update({'color':self.plot_default[archiveType][0]})
            
            fig,ax = ts.plot(**plt_kwargs)
        elif ensemble == True:
            ensemble_kwargs = {} if ensemble_kwargs is None else ensemble_kwargs.copy()
            ens = self.chronEnsembleToPaleo(D,**ensemble_kwargs)
            # NOT  A VERY ELEGANT SOLUTION: replace depth in the dictionary
            for item in ens.__dict__['series_list']:
                item.__dict__['value'] = value_depth
                item.__dict__['value_unit']=units_depth
                item.__dict__['value_name']='depth'
            envelope_kwargs={} if envelope_kwargs is None else envelope_kwargs.copy()
            if 'curve_clr'not in envelope_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(meta['archiveType']).lower().replace(" ","")
                envelope_kwargs.update({'curve_clr':self.plot_default[archiveType][0]})
            if 'shade_clr'not in envelope_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(meta['archiveType']).lower().replace(" ","")
                envelope_kwargs.update({'shade_clr':self.plot_default[archiveType][0]})
            ens2=ens.common_time()
            if num_traces > 0:
                envelope_kwargs.update({'mute':True})
                fig,ax = ens2.plot_envelope(**envelope_kwargs)
                traces_kwargs={} if traces_kwargs is None else traces_kwargs.copy()
                if 'color' not in traces_kwargs.keys():
                    archiveType = lipdutils.LipdToOntology(meta['archiveType']).lower().replace(" ","")
                    traces_kwargs.update({'color':self.plot_default[archiveType][0]})
                if 'linestyle' not in traces_kwargs.keys():
                    traces_kwargs.update({'linestyle':'dashed'})
                traces_kwargs.update({'ax':ax,'num_traces':num_traces})
                ens2.plot_traces(**traces_kwargs) 
            else:
                fig,ax=ens2.plot_envelope(**envelope_kwargs)
            
        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            # else:
            #     if not mute:
            #         plotting.showfig(fig)
            return fig, ax
        else:
            return ax
        
        
            