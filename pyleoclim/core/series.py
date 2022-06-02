"""
The Series class describes the most basic objects in Pyleoclim. A Series is a simple `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_ that contains 3 things:
- a series of real-valued numbers;
- a time axis at which those values were measured/simulated ;
- optionally, some metadata about both axes, like units, labels and the like.

How to create and manipulate such objects is described in a short example below, while `this notebook <https://nbviewer.jupyter.org/github/LinkedEarth/Pyleoclim_util/blob/master/example_notebooks/pyleoclim_ui_tutorial.ipynb>`_ demonstrates how to apply various Pyleoclim methods to Series objects.
"""

from ..utils import tsutils, plotting, tsmodel, tsbase
from ..utils import wavelet as waveutils
from ..utils import spectral as specutils
from ..utils import correlation as corrutils
from ..utils import causality as causalutils
from ..utils import decomposition
from ..utils import filter as filterutils

from ..core.psds import PSD
from ..core.ssares import SsaRes
from ..core.multipleseries import MultipleSeries
from ..core.scalograms import Scalogram
from ..core.coherence import Coherence
from ..core.corr import Corr
from ..core.surrogateseries import SurrogateSeries

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import namedtuple
from copy import deepcopy

from matplotlib import gridspec
import warnings
import collections

def dict2namedtuple(d):
    ''' Convert a dictionary to a namedtuple
    '''
    tupletype = namedtuple('tupletype', sorted(d))
    return tupletype(**d)

class Series:
    '''The Series class describes the most basic objects in Pyleoclim. 
    A Series is a simple `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_ that contains 3 things:
        
    * a series of real-valued numbers;
    
    * a time axis at which those values were measured/simulated ;
    
    * optionally, some metadata about both axes, like units, labels and the like.

    How to create and manipulate such objects is described in a short example below, while `this notebook <https://nbviewer.jupyter.org/github/LinkedEarth/Pyleoclim_util/blob/master/example_notebooks/pyleoclim_ui_tutorial.ipynb>`_ demonstrates how to apply various Pyleoclim methods to Series objects.

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

    In this example, we import the Southern Oscillation Index (SOI) into a pandas dataframe and create aSeries object.

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
        
    For a quick look at the values, one may use the `print()` method: 
        
    .. ipython:: python
        :okwarning:
        :okexcept:
        
        print(ts)
    
    '''

    def __init__(self, time, value, time_name=None, time_unit=None, value_name=None, value_unit=None, label=None, clean_ts=True, verbose=False):

        if clean_ts==True:
            value, time = tsbase.clean_ts(np.array(value), np.array(time), verbose=verbose)

        self.time = np.array(time)
        self.value = np.array(value)
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
        Initialization of plot labels based on Series metadata

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

        _ = print(tabulate(table, headers='keys'))
        return f'Length: {np.size(self.time)}'

    def stats(self):
        """ Compute basic statistics from a Series

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
              savefig_settings=None, ax=None, invert_xaxis=False):
        ''' Plot the timeseries

        Parameters
        ----------

        figsize : list
            a list of two integers indicating the figure size

        marker : str
            e.g., 'o' for dots
            See [matplotlib.markers](https://matplotlib.org/stable/api/markers_api.html) for details

        markersize : float
            the size of the marker

        color : str, list
            the color for the line plot
            e.g., 'r' for red
            See [matplotlib colors](https://matplotlib.org/stable/gallery/color/color_demo.html) for details

        linestyle : str
            e.g., '--' for dashed line
            See [matplotlib.linestyles](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html) for details

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
            See [matplotlib.pyplot.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html) for details

        lgd_kwargs : dict
            the dictionary of keyword arguments for ax.legend()
            See [matplotlib.pyplot.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html) for details

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


        Returns
        -------

        fig : matplotlib.figure
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/stable/api/figure_api.html) for details.

        ax : matplotlib.axis
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/stable/api/axes_api.html) for details.

        Notes
        -----

        When `ax` is passed, the return will be `ax` only; otherwise, both `fig` and `ax` will be returned.

        See also
        --------

        pyleoclim.utils.plotting.savefig : saving a figure in Pyleoclim

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

        Save the figure. Two options available, only one is needed:
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
            invert_xaxis=invert_xaxis,
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
                (2) 'mcssa': Monte-Carlo SSA (use modes above the 95% threshold)
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

        References
        ----------
        [1]_ Vautard, R., and M. Ghil (1989), Singular spectrum analysis in nonlinear
        dynamics, with applications to paleoclimatic time series, Physica D, 35,
        395–424.

        [2]_ Ghil, M., R. M. Allen, M. D. Dettinger, K. Ide, D. Kondrashov, M. E. Mann,
        A. Robertson, A. Saunders, Y. Tian, F. Varadi, and P. Yiou (2002),
        Advanced spectral methods for climatic time series, Rev. Geophys., 40(1),
        1003–1052, doi:10.1029/2000RG000092.

        [3]_ Allen, M. R., and L. A. Smith (1996), Monte Carlo SSA: Detecting irregular
        oscillations in the presence of coloured noise, J. Clim., 9, 3373–3404.

        [4]_ Schoellhamer, D. H. (2001), Singular spectrum analysis for time series with
        missing data, Geophysical Research Letters, 28(16), 3187–3190, doi:10.1029/2000GL012698.

        See also
        --------
        
        pyleoclim.core.utils.decomposition.ssa : Singular Spectrum Analysis utility
        
        pyleoclim.core.ssares.SsaRes.modeplot : plot SSA modes
        
        pyleoclim.core.ssares.SsaRes.screeplot : plot SSA eigenvalue spectrum

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
            @savefig ts_eigen.png
            nino_ssa.screeplot()


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
            @savefig ssa_recon.png
            fig, ax = ts.plot(title='ONI')
            ax.plot(time,RCk,label='SSA reconstruction, 14 modes',color='orange')
            ax.legend()

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
            @savefig scree_nmc.png
            nino_mcssa.screeplot()

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
        '''Check if the Series time axis is evenly-spaced, within tolerance
        
        Parameters
        ----------
        tol : float
            tolerance. If time increments are all within tolerance, the series
            is declared evenly-spaced. default = 1e-3

        Returns
        -------

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

        By default, this method implements a lowpass filter, though it can easily
        be turned into a bandpass or high-pass filter (see examples below).

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
            @savefig ts_filter1.png
            fig, ax = ts.plot(label='mix')
            ts1.plot(ax=ax, label='10 Hz')
            ts2.plot(ax=ax, label='20 Hz')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=3)

        - Applying a low-pass filter

        .. ipython:: python
            :okwarning:
            :okexcept:

            fig, ax = ts.plot(label='mix')
            ts.filter(cutoff_freq=15).plot(ax=ax, label='After 15 Hz low-pass filter')
            @savefig ts_filter2.png
            ts1.plot(ax=ax, label='10 Hz')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=3)

        - Applying a band-pass filter

        .. ipython:: python
            :okwarning:
            :okexcept:

            fig, ax = ts.plot(label='mix')
            ts.filter(cutoff_freq=[15, 25]).plot(ax=ax, label='After 15-25 Hz band-pass filter')
            @savefig ts_filter3.png
            ts2.plot(ax=ax, label='20 Hz')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=3)

        Above is using the default Butterworth filtering. To use FIR filtering with a window like Hanning is also simple:

        .. ipython:: python
            :okwarning:
            :okexcept:

                    
            fig, ax = ts.plot(label='mix')
            ts.filter(cutoff_freq=[15, 25], method='firwin', window='hanning').plot(ax=ax, label='After 15-25 Hz band-pass filter')
            @savefig ts_filter4.png
            ts2.plot(ax=ax, label='20 Hz')
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=3)

        - Applying a high-pass filter

        .. ipython:: python
            :okwarning:
            :okexcept:

            fig, ax = ts.plot(label='mix')
            ts_low  = ts.filter(cutoff_freq=15)
            ts_high = ts.copy()
            ts_high.value = ts.value - ts_low.value # subtract low-pass filtered series from original one
            @savefig ts_filter5.png
            ts_high.plot(label='High-pass filter @ 15Hz',ax=ax)
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=3)

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

        if method not in method_func.keys():
            raise ValueError('Method value is not an appropriate method for filters')

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
            if cutoff_scale is None and cutoff_freq is None:
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
                  ax=None, ylabel='KDE', vertical=False, edgecolor='w', **plot_kwargs):
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

    def summary_plot(self, psd, scalogram, figsize=[8, 10], title=None,
                    time_lim=None, value_lim=None, period_lim=None, psd_lim=None,
                    time_label=None, value_label=None, period_label=None, psd_label=None,
                    ts_plot_kwargs = None, wavelet_plot_kwargs = None,
                    psd_plot_kwargs = None, y_label_loc = -.15, savefig_settings=None):

        ''' Produce summary plot of timeseries.

        Generate cohesive plot of timeseries alongside results of wavelet analysis and spectral analysis on said timeseries.
        Requires wavelet and spectral analysis to be conducted outside of plotting function, psd and scalogram must be passed as arguments.

        Parameters
        ----------

        psd : PSD
            the PSD object of a Series.

        scalogram : Scalogram
            the Scalogram object of a Series. 
            If the passed scalogram object contains stored signif_scals these will be plotted.

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

        time_label : str
            the label for the time axis

        value_label : str
            the label for the value axis of the timeseries

        period_label : str
            the label for the period axis

        psd_label : str
            the label for the amplitude axis of PDS

        ts_plot_kwargs : dict
            arguments to be passed to the timeseries subplot, see pyleoclim.Series.plot for details

        wavelet_plot_kwargs : dict
            arguments to be passed to the scalogram plot, see pyleoclim.Scalogram.plot for details

        psd_plot_kwargs : dict
            arguments to be passed to the psd plot, see pyleoclim.PSD.plot for details
                Certain psd plot settings are required by summary plot formatting. These include:
                    - ylabel
                    - legend
                    - tick parameters
                These will be overriden by summary plot to prevent formatting errors

        y_label_loc : float
            Plot parameter to adjust horizontal location of y labels to avoid conflict with axis labels, default value is -0.15

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        See also
        --------

        pyleoclim.core.series.Series.spectral : Spectral analysis for a timeseries

        pyleoclim.core.series.Series.wavelet : Wavelet analysis for a timeseries

        pyleoclim.utils.plotting.savefig : saving figure in Pyleoclim

        pyleoclim.core.psds.PSD : PSD object

        pyleoclim.core.psds.MultiplePSD : Multiple PSD object

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

        '''

        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(6, 12)
        gs.update(wspace=0, hspace=0)

        wavelet_plot_kwargs={} if wavelet_plot_kwargs is None else wavelet_plot_kwargs.copy()
        psd_plot_kwargs={} if psd_plot_kwargs is None else psd_plot_kwargs.copy()
        ts_plot_kwargs={} if ts_plot_kwargs is None else ts_plot_kwargs.copy()

        ax = {}
        ax['ts'] = plt.subplot(gs[0:1, :-3])
        ax['ts'] = self.plot(ax=ax['ts'], **ts_plot_kwargs)
        ax['ts'].xaxis.set_visible(False)
        ax['ts'].get_yaxis().set_label_coords(y_label_loc,0.5)

        if time_lim is not None:
            ax['ts'].set_xlim(time_lim)
            if 'xlim' in ts_plot_kwargs:
                print('Xlim passed to time series plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument.')

        if value_lim is not None:
            ax['ts'].set_ylim(value_lim)
            if 'ylim' in ts_plot_kwargs:
                print('Ylim passed to time series plot through exposed argument and key word argument. The exposed argument takes precedence and will overwrite relevant key word argument.')

        ax['scal'] = plt.subplot(gs[1:5, :-3], sharex=ax['ts'])

        #Need variable for plotting purposes
        if 'variable' not in wavelet_plot_kwargs:
            wavelet_plot_kwargs.update({'variable':'amplitude'})

        if 'title' not in wavelet_plot_kwargs:
            wavelet_plot_kwargs.update({'title':None})

        if 'cbar_style' not in wavelet_plot_kwargs:
            wavelet_plot_kwargs.update({'cbar_style':{'orientation': 'horizontal', 'pad': 0.12, 
                                        'label': wavelet_plot_kwargs['variable'].capitalize() + ' from ' + scalogram.wave_method}})

        ax['scal'] = scalogram.plot(ax=ax['scal'], **wavelet_plot_kwargs)
        ax['scal'].invert_yaxis()
        ax['scal'].get_yaxis().set_label_coords(y_label_loc,0.5)

        ax['psd'] = plt.subplot(gs[1:4, -3:], sharey=ax['scal'])

        ax['psd'] = psd.plot(ax=ax['psd'], transpose=True, ylabel = 'PSD from \n' + str(psd.spec_method), **psd_plot_kwargs)

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
        return fig, ax

    def copy(self):
        '''Make a copy of the Series object

        Returns
        -------
        Series : pyleoclim.Series
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
        new : pyleoclim.Series
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
        new : pyleoclim.Series
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

        factor : float
            The factor that adjusts the threshold for gap detection

        Returns
        -------

        res : pyleoclim.MultipleSeries Object or pyleoclim.Series Object
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
            
        Examples
        --------

        slice the SOI from 1972 to 1998

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts = pyleo.Series(time=time, value=value, time_name='Year C.E', value_name='SOI', label='SOI')
            
            ts_slice = ts.slice([1972, 1998])
            print("New time bounds:",ts_slice.time.min(),ts_slice.time.max())

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

        We will generate a harmonic signal with a nonlinear trend and use two  detrending options to recover the original signal.

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

            # Detrending with default parameters (using EMD method with 1 mode)
            ts_emd1 = ts.detrend()
            ts_emd1.label = 'default detrending (EMD, last mode)'
            @savefig ts_emd1.png
            fig, ax = ts_emd1.plot(title='Detrended with EMD method')
            ax.plot(time,signal_noise,label='target signal')
            ax.legend()
            
        

        We see that the default function call results in a "Hockey Stick" at the end, which is undesirable.
        There is no automated way to do this, but with a little trial and error, we find that removing 
        the 2 smoothest modes performs reasonably well:
                
        .. ipython:: python
            :okwarning:
            :okexcept:

            ts_emd2 = ts.detrend(method='emd', n=2)
            ts_emd2.label = 'EMD detrending, last 2 modes'
            @savefig ts_emd_n2.png
            fig, ax = ts_emd2.plot(title='Detrended with EMD (n=2)')
            ax.plot(time,signal_noise,label='target signal')
            ax.legend()

        Another option for removing a nonlinear trend is a Savitzky-Golay filter:
            
        .. ipython:: python
            :okwarning:
            :okexcept:
                
            ts_sg = ts.detrend(method='savitzky-golay')
            ts_sg.label = 'savitzky-golay detrending, default parameters'
            @savefig ts_sg.png
            fig, ax = ts_sg.plot(title='Detrended with Savitzky-Golay filter')
            ax.plot(time,signal_noise,label='target signal')
            ax.legend()

        As we can see, the result is even worse than with EMD (default). Here it pays to look into the underlying method, which comes from SciPy.
        It turns out that by default, the Savitzky-Golay filter fits a polynomial to the last "window_length" values of the edges.
        By default, this value is close to the length of the series. Choosing a value 10x smaller fixes the problem here, though you will have to tinker with that parameter until you get the result you seek.
        
        .. ipython:: python
            :okwarning:
            :okexcept:
                
            ts_sg2 = ts.detrend(method='savitzky-golay',sg_kwargs={'window_length':201})
            ts_sg2.label = 'savitzky-golay detrending, window_length = 201'
            @savefig ts_sg2.png
            fig, ax = ts_sg2.plot(title='Detrended with Savitzky-Golay filter')
            ax.plot(time,signal_noise,label='target signal')
            ax.legend()

        '''
        new = self.copy()
        v_mod = tsutils.detrend(self.value, x=self.time, method=method, **kwargs)
        new.value = v_mod
        return new

    def spectral(self, method='lomb_scargle', freq_method='log', freq_kwargs=None, settings=None, label=None, scalogram=None, verbose=False):
        ''' Perform spectral analysis on the timeseries

        Parameters
        ----------

        method : str; 
            {'wwz', 'mtm', 'lomb_scargle', 'welch', 'periodogram', 'cwt'}

        freq_method : str
            {'log','scale', 'nfft', 'lomb_scargle', 'welch'}

        freq_kwargs : dict
            Arguments for frequency vector

        settings : dict
            Arguments for the specific spectral method

        label : str
            Label for the PSD object

        scalogram : pyleoclim.core.series.Series.Scalogram
            The return of the wavelet analysis; effective only when the method is 'wwz' or 'cwt'

        verbose : bool
            If True, will print warning messages if there is any

        Returns
        -------

        psd : pyleoclim.PSD
            A PSD object

        See also
        --------
        pyleoclim.utils.spectral.mtm : Spectral analysis using the Multitaper approach

        pyleoclim.utils.spectral.lomb_scargle : Spectral analysis using the Lomb-Scargle method

        pyleoclim.utils.spectral.welch: Spectral analysis using the Welch segement approach

        pyleoclim.utils.spectral.periodogram: Spectral anaysis using the basic Fourier transform

        pyleoclim.utils.spectral.wwz_psd : Spectral analysis using the Wavelet Weighted Z transform

        pyleoclim.utils.spectral.cwt_psd : Spectral analysis using the continuous Wavelet Transform as implemented by Torrence and Compo

        pyleoclim.utils.spectral.make_freq_vector : Functions to create the frequency vector

        pyleoclim.utils.tsutils.detrend : Detrending function

        pyleoclim.core.psds.PSD : PSD object

        pyleoclim.core.psds.MultiplePSD : Multiple PSD object


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
            @savefig spec_ls_n50.png
            fig, ax = psd_LS_n50.plot(
                title='PSD using Lomb-Scargle method with 4 overlapping segments',
                label='settings={"n50": 4}')
            psd_ls.plot(ax=ax, label='settings={"n50": 3}', marker='o')

            @savefig spec_ls_freq.png
            fig, ax = psd_LS_freq.plot(
                title='PSD using Lomb-Scargle method with different frequency vectors',
                label='freq=np.linspace(1/20, 1/0.2, 51)', marker='o')
            psd_ls.plot(ax=ax, label='freq_method="log"', marker='o')

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

        We may take advantage of a pre-calculated scalogram using WWZ to accelerate the spectral analysis
        (although note that the default parameters for spectral and wavelet analysis using WWZ are different):

        .. ipython:: python
            :okwarning:
            :okexcept:

            scal_wwz = ts_std.wavelet(method='wwz')  # wwz is the default method
            psd_wwz_fast = ts_std.spectral(method='wwz', scalogram=scal_wwz)
            @savefig spec_wwz_fast.png
            fig, ax = psd_wwz_fast.plot(title='PSD using WWZ method w/ pre-calculated scalogram')

        - Periodogram

        .. ipython:: python
            :okwarning:
            :okexcept:

            ts_interp = ts_std.interp()
            psd_perio = ts_interp.spectral(method='periodogram')
            psd_perio_signif = psd_perio.signif_test(number=20, method='ar1sim') #in practice, need more AR1 simulations
            @savefig spec_perio.png
            fig, ax = psd_perio_signif.plot(title='PSD using Periodogram method')

        - Welch

        .. ipython:: python
            :okwarning:
            :okexcept:

            ts_interp = ts_std.interp()
            psd_welch = ts_interp.spectral(method='welch')
            psd_welch_signif = psd_welch.signif_test(number=20, method='ar1sim') #in practice, need more AR1 simulations
            @savefig spec_welch.png
            fig, ax = psd_welch_signif.plot(title='PSD using Welch method')

        - MTM

        .. ipython:: python
            :okwarning:
            :okexcept:

            ts_interp = ts_std.interp()
            psd_mtm = ts_interp.spectral(method='mtm')
            psd_mtm_signif = psd_mtm.signif_test(number=20, method='ar1sim') #in practice, need more AR1 simulations
            @savefig spec_mtm.png
            fig, ax = psd_mtm_signif.plot(title='PSD using MTM method')

        - Continuous Wavelet Transform

        .. ipython:: python
            :okwarning:
            :okexcept:

            ts_interp = ts_std.interp()
            psd_cwt = ts_interp.spectral(method='cwt')
            psd_cwt_signif = psd_cwt.signif_test(number=20)
            @savefig spec_cwt.png
            fig, ax = psd_cwt_signif.plot(title='PSD using CWT method')

        '''
        if not verbose:
            warnings.simplefilter('ignore')

        settings = {} if settings is None else settings.copy()
        spec_func = {
            'wwz': specutils.wwz_psd,
            'mtm': specutils.mtm,
            'lomb_scargle': specutils.lomb_scargle,
            'welch': specutils.welch,
            'periodogram': specutils.periodogram,
            'cwt': specutils.cwt_psd
        }
        args = {}
        freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
        freq = specutils.make_freq_vector(self.time, method=freq_method, **freq_kwargs)

        args['wwz'] = {'freq': freq}
        args['cwt'] = {'freq': freq}
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

        if method == 'cwt' and scalogram is not None:
            Results = collections.namedtuple('Results', ['amplitude', 'coi', 'freq', 'time', 'scale', 'mother','param'])
            res = Results(amplitude=scalogram.amplitude, coi=scalogram.coi,
                          freq=scalogram.frequency, time=scalogram.time,
                          scale=scalogram.wave_args['scale'],
                          mother=scalogram.wave_args['mother'],
                          param=scalogram.wave_args['param'])
            args['cwt'].update({'cwt_res':res})


        spec_res = spec_func[method](self.value, self.time, **args[method])
        if type(spec_res) is dict:
            spec_res = dict2namedtuple(spec_res)

        if label is None:
            label = self.label

        if method == 'wwz' and scalogram is not None:
            args['wwz'].pop('wwa')
            args['wwz'].pop('wwz_Neffs')
            args['wwz'].pop('wwz_freq')

        if method == 'cwt':
            args['cwt'].update({'scale':spec_res.scale,'mother':spec_res.mother,'param':spec_res.param})
            if scalogram is not None:
                args['cwt'].pop('cwt_res')


        psd = PSD(
            frequency=spec_res.freq,
            amplitude=spec_res.psd,
            label=label,
            timeseries=self,
            spec_method=method,
            spec_args=args[method]
        )

        return psd

    def wavelet(self, method='cwt', settings=None, freq_method='log', freq_kwargs=None, verbose=False):
        ''' Perform wavelet analysis on a timeseries

        Parameters
        ----------

        method : str {wwz, cwt}
            cwt - the continuous wavelet transform (as per Torrence and Compo [1998])
                is appropriate for evenly-spaced series.
            wwz - the weighted wavelet Z-transform (as per Foster [1996])
                is appropriate for unevenly-spaced series.
            Default is cwt, returning an error if the Series is unevenly-spaced.

        freq_method : str
            {'log', 'scale', 'nfft', 'lomb_scargle', 'welch'}

        freq_kwargs : dict
            Arguments for the frequency vector

        settings : dict
            Arguments for the specific wavelet method

        verbose : bool
            If True, will print warning messages if there is any

        Returns
        -------

        scal : Scalogram object

        See also
        --------

        pyleoclim.utils.wavelet.wwz : wwz function

        pyleoclim.utils.wavelet.cwt : cwt function

        pyleoclim.utils.spectral.make_freq_vector : Functions to create the frequency vector

        pyleoclim.utils.tsutils.detrend : Detrending function
        
        pyleoclim.core.series.Series.spectral : spectral analysis tools

        pyleoclim.core.scalograms.Scalogram : Scalogram object

        pyleoclim.core.scalograms.MultipleScalogram : Multiple Scalogram object

        References
        ----------

        Torrence, C. and G. P. Compo, 1998: A Practical Guide to Wavelet Analysis. Bull. Amer. Meteor. Soc., 79, 61-78.
        Python routines available at http://paos.colorado.edu/research/wavelets/

        Foster, G., 1996: Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal, 112, 1709.

        Examples
        --------

        Wavelet analysis on the evenly-spaced SOI record. The CWT method will be applied by default.
        
        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts = pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')

            scal1 = ts.wavelet() 
            scal_signif = scal1.signif_test(number=20)  # for research-grade work, use number=200 or larger
            @savefig scal_cwt.png
            fig, ax = scal_signif.plot(title='CWT scalogram')
            pyleo.closefig()
                        
        If you wanted to invoke the WWZ instead (here with no significance testing, to lower computational cost):
            
        .. ipython:: python
            :okwarning:
            :okexcept:
                
            scal2 = ts.wavelet(method='wwz') 
            @savefig scal_wwz.png
            fig, ax = scal2.plot(title='WWZ scalogram')
            pyleo.closefig()

        Notice that the two scalograms have different units, which are arbitrary.  Method-specific arguments
        may be passed via `settings`.  For instance, if you wanted to change the default mother wavelet
        ('MORLET') to a derivative of a Gaussian (DOG), with degree 2 by default ("Mexican Hat wavelet"):

        .. ipython:: python
            :okwarning:
            :okexcept:
                
            scal3 = ts.wavelet(settings = {'mother':'DOG'}) 
            @savefig scal_dog.png
            fig, ax = scal3.plot(title='CWT scalogram with DOG mother wavelet')
            pyleo.closefig()
            
        As for WWZ, note that, for computational efficiency, the time axis is coarse-grained
        by default to 50 time points, which explains in part the diffence with the CWT scalogram.

        If you need a custom axis, it (and other method-specific  parameters) can also be passed 
        via the `settings` dictionary:
            
        .. ipython:: python
            :okwarning:
            :okexcept:
            
            tau = np.linspace(np.min(ts.time), np.max(ts.time), 60)
            scal4 = ts.wavelet(method='wwz', settings={'tau':tau}) 
            @savefig scal_tau.png
            fig, ax = scal4.plot(title='WWZ scalogram with finer time axis')
            pyleo.closefig()
            
        '''
        if not verbose:
            warnings.simplefilter('ignore')

        # Assign method
        if method == 'cwt' and not(self.is_evenly_spaced()):
            raise ValueError("The chosen method is cwt but the series is unevenly spaced. You can either interpolate/bin or set method='wwz'.")

        wave_func = {'wwz': waveutils.wwz,
                      'cwt': waveutils.cwt
                      }

        # Process options
        settings = {} if settings is None else settings.copy()
        freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
        freq = specutils.make_freq_vector(self.time, method=freq_method, **freq_kwargs)
        args = {}
        args['wwz'] = {'freq': freq}
        args['cwt'] = {'freq': freq}

        if method == 'wwz':
            if 'ntau' in settings.keys():
                ntau = settings['ntau']
            else:
                ntau = np.min([np.size(self.time), 50])
            tau = np.linspace(np.min(self.time), np.max(self.time), ntau)
            settings.update({'tau': tau})

        args[method].update(settings)

        # Apply wavelet method
        wave_res = wave_func[method](self.value, self.time, **args[method])

        # Export result
        if method == 'wwz':
            wwz_Neffs = wave_res.Neffs
        elif method=='cwt':
            wwz_Neffs = None
            args[method].update({'scale':wave_res.scale,'mother':wave_res.mother,'param':wave_res.param})

        scal = Scalogram(
            frequency=wave_res.freq,
            scale = wave_res.scale,
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

    def wavelet_coherence(self, target_series, method='cwt', settings=None,
                          freq_method='log', freq_kwargs=None, verbose=False,
                          common_time_kwargs=None):
        ''' Performs wavelet coherence analysis with the target timeseries


        Parameters
        ----------

        target_series : pyleoclim.Series
            A pyleoclim Series object on which to perform the coherence analysis

        method : str
            Possible methods {'wwz','cwt'}. Default is 'cwt', which only works
            if the series share the same evenly-spaced time axis.
            'wwz' is designed for unevenly-spaced data, but is far slower.

        freq_method : str
            {'log','scale', 'nfft', 'lomb_scargle', 'welch'}

        freq_kwargs : dict
            Arguments for frequency vector

        common_time_kwargs : dict
            Parameters for the method `MultipleSeries.common_time()`. Will use interpolation by default.

        settings : dict
            Arguments for the specific wavelet method (e.g. decay constant for WWZ, mother wavelet for CWT)
            and common properties like standardize, detrend, gaussianize, pad, etc.

        verbose : bool
            If True, will print warning messages, if any

        Returns
        -------

        coh : pyleoclim.core.coherence.Coherence

        References
        ----------

        Grinsted, A., Moore, J. C. & Jevrejeva, S. Application of the cross wavelet transform and
        wavelet coherence to geophysical time series. Nonlin. Processes Geophys. 11, 561–566 (2004). 

        See also
        --------

        pyleoclim.utils.spectral.make_freq_vector : Functions to create the frequency vector
        
        pyleoclim.utils.tsutils.detrend : Detrending function
        
        pyleoclim.core.multipleseries.MultipleSeries.common_time : put timeseries on common time axis
        
        pyleoclim.core.series.Series.wavelet : wavelet analysis

        Examples
        --------

        Calculate the wavelet coherence of NINO3 and All India Rainfall with default arguments:

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino_even.csv')
            time = data['t'].values
            air = data['air'].values
            nino = data['nino'].values
            ts_air = pyleo.Series(time=time, value=air, time_name='Year (CE)')
            ts_nino = pyleo.Series(time=time, value=nino, time_name='Year (CE)')

            coh = ts_air.wavelet_coherence(ts_nino)

            @savefig coh.png
            fig, ax = coh.plot()
            pyleo.closefig()

        Note that in this example both timeseries area already on a common,
        evenly-spaced time axis. If they are not (either because the data are unevenly spaced,
        or because the time axes are different in some other way), an error will be raised.
        To circumvent this error, you can either put the series
        on a common time axis (e.g. using common_time()) prior to applying CWT, or you
        can use the Weighted Wavelet Z-transform (WWZ) instead, as it is designed for
        unevenly-spaced data. However, it is usually far slower:

        .. ipython:: python
            :okwarning:
            :okexcept:

             coh_wwz = ts_air.wavelet_coherence(ts_nino, method = 'wwz')
             @savefig coh_wwz.png
             fig, ax = coh_wwz.plot()
            
        As with wavelet analysis, both CWT and WWZ admit optional arguments through `settings`.
        Significance is assessed similarly as with PSD or Scalogram objects:
            
        .. ipython:: python
            :okwarning:
            :okexcept:

            cwt_sig = coh.signif_test(number=20, qs=[.9,.95]) # specifiying 2 significance thresholds does not take any more time.
            @savefig cwt_sig.png
            # by default, the plot function will look for the closest quantile to 0.95, but it is easy to adjust:
            cwt_sig.plot(signif_thresh = 0.9)
            
        Another plotting option, `dashboard`, allows to visualize both
        timeseries as well as the wavelet transform coherency (WTC), which quantifies where 
        two timeseries exhibit similar behavior in time-frequency space, and the cross-wavelet
        transform (XWT), which indicates regions of high common power. 
        
        .. ipython:: python
            :okwarning:
            :okexcept:

            @savefig cwt_sig_dash.png
            cwt_sig.dashboard()
             
        Note: this design balances many considerations, and is not easily customizable. 
        '''
        if not verbose:
            warnings.simplefilter('ignore')

        settings = {} if settings is None else settings.copy()

        wtc_func = {
            'wwz': waveutils.wwz_coherence,
            'cwt': waveutils.cwt_coherence
        }

        # Process options
        settings = {} if settings is None else settings.copy()
        freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
        freq = specutils.make_freq_vector(self.time, method=freq_method, **freq_kwargs)
        args = {}
        args['wwz'] = {'freq': freq}
        args['cwt'] = {'freq': freq}

        # put on same time axes if necessary
        if method == 'cwt' and not np.array_equal(self.time, target_series.time):
            warnings.warn("Series have different time axes. Applying common_time().")
            ms = MultipleSeries([self, target_series])
            common_time_kwargs = {} if common_time_kwargs is None else common_time_kwargs.copy()
            ct_args = {'method': 'interp'}
            ct_args.update(common_time_kwargs)
            ms = ms.common_time(**ct_args)
            ts1 = ms.series_list[0]
            ts2 = ms.series_list[1]
        elif method == 'cwt' and (not self.is_evenly_spaced() or not target_series.is_evenly_spaced()):
            raise ValueError("The chosen method is cwt but at least one the series is unevenly spaced. You can either apply common_time() or use 'wwz'.")

        else:
            ts1 = self
            ts2 = target_series

        if method == 'wwz':
            if 'ntau' in settings.keys():
                ntau = settings['ntau']
            else:
                ntau = np.min([np.size(ts1.time), np.size(ts2.time), 50])

            tau = np.linspace(np.min(self.time), np.max(self.time), ntau)
            settings.update({'tau': tau})

        args[method].update(settings)

        # Apply WTC method
        wtc_res = wtc_func[method](ts1.value, ts1.time, ts2.value, ts2.time, **args[method])

        # Export result
        coh = Coherence(
            frequency=wtc_res.freq,
            scale = wtc_res.scale,
            time=wtc_res.time,
            wtc=wtc_res.xw_coherence,
            xwt=wtc_res.xw_amplitude,
            phase=wtc_res.xw_phase,
            coi=wtc_res.coi,
            timeseries1= ts1,
            timeseries2= ts2,
            wave_method = method,
            wave_args = args[method],
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
            # set an arbitrary random seed to fix the result
            corr_res = ts_nino.correlation(ts_air, settings={'nsim': 20}, seed=2333)
            print(corr_res)

            # using a simple t-test
            # set an arbitrary random seed to fix the result
            corr_res = ts_nino.correlation(ts_air, settings={'method': 'ttest'})
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

    def surrogates(self, method='ar1sim', number=1, length=None, seed=None, settings=None):
        ''' Generate surrogates with increasing time axis

        Parameters
        ----------

        method : {ar1sim}
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

        pyleoclim.utils.tsmodel.ar1_sim : AR(1) simulator
        
        '''
        settings = {} if settings is None else settings.copy()
        surrogate_func = {
            'ar1sim': tsmodel.ar1_sim,
        }
        args = {}
        args['ar1sim'] = {'t': self.time}
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
                  saveknee_settings=None,saveoutliers_settings=None):
        '''
        Detects outliers in a timeseries and removes if specified. The method uses clustering to locate outliers.

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

        Returns
        -------
        new : Series
            Time series with outliers removed if they exist

        See also
        --------

        pyleoclim.utils.tsutils.remove_outliers : remove outliers function

        pyleoclim.utils.plotting.plot_xy : basic x-y plot

        pyleoclim.utils.plotting.plot_scatter_xy : Scatter plot on top of a line plot

        Examples
        --------

        Let's create a "perfect" sinusoidal signal and add outliers

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import numpy as np

            # create the signal
            freqs=[1/20,1/80]
            time=np.arange(2001)
            signals=[]
            for freq in freqs:
                signals.append(np.cos(2*np.pi*freq*time))
            signal=sum(signals)

            #add outliers
            outliers_start = np.mean(signal)+5*np.std(signal)
            outliers_end = np.mean(signal)+7*np.std(signal)
            outlier_values = np.arange(outliers_start,outliers_end,0.1)
            index = np.random.randint(0,len(signal),6)
            signal_out = signal
            for i,ind in enumerate(index):
                signal_out[ind] = outlier_values[i]

            #Make a Series object
            ts = pyleo.Series(time=time,value=signal_out)
            @savefig outliers.png
            fig, ax = ts.plot()
            pyleo.closefig(fig)

            #Detect and remove outliers
            ts_new=ts.outliers()
            @savefig outliers_remove.png
            fig, ax = ts_new.plot()
            pyleo.closefig(fig)

        '''
        new = self.copy()
        outlier_indices = tsutils.detect_outliers(
            self.time, self.value, auto=auto, plot_knee=fig_knee,plot_outliers=fig_outliers,
            figsize=figsize,saveknee_settings=saveknee_settings,saveoutliers_settings=saveoutliers_settings,
            plot_outliers_kwargs=plot_outliers_kwargs,plot_knee_kwargs=plot_knee_kwargs)
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
