''' The application interface for the users

@author: fengzhu

Created on Jan 31, 2020
'''
from ..utils import tsutils, plotting, mapping, lipdutils, tsmodel
from ..utils import wavelet as waveutils
from ..utils import spectral as specutils
from ..utils import correlation as corrutils
from ..utils import causality as causalutils
from ..utils import decomposition

#from textwrap import dedent

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from tabulate import tabulate
from collections import namedtuple
from copy import deepcopy

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib import cm
import matplotlib.pylab as pl
from matplotlib import gridspec
#from matplotlib.colors import BoundaryNorm, Normalize

from tqdm import tqdm
from scipy.stats.mstats import mquantiles
import warnings
import os

import lipd as lpd


def dict2namedtuple(d):
    tupletype = namedtuple('tupletype', sorted(d))
    return tupletype(**d)

class Series:
    ''' Create a pyleoSeries object

    Parameters
    ----------

    time : list or numpy.array
        Time values for the time series

    value : list of numpy.array
        ordinate values for the time series

    time_name : string
        Name of the time vector (e.g., 'Age').
        Default is None. This is used to label the time axis on plots

    time_unit : string
        Units for the time vector (e.g., 'yr B.P.').
        Default is None

    value_name : string
        Name of the value vector (e.g., 'temperature')
        Default is None

    value_unit : string
        Units for the value vector (e.g., 'deg C')

    label : string
        Name of the time series (e.g., 'Nino 3.4')

    clean_ts : bool
        remove the NaNs and let the time axis to be increasing if True
    
    Examples
    --------
    
    In this example, we import the Southern Oscillation Index (SOI) into a pandas dataframe and create a PyleoSeries object. 

    .. ipython:: python
        :okwarning:
        
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
            time_name='Year (CE)', value_name='SOI', label='SOI'
        )
        ts
        ts.__dict__.keys()
    '''

    def __init__(self, time, value, time_name=None, time_unit=None, value_name=None, value_unit=None, label=None, clean_ts=True):
        
        if clean_ts==True:
            value, time = tsutils.clean_ts(np.array(value), np.array(time))

        self.time = time
        self.value = value
        self.time_name = time_name
        self.time_unit = time_unit
        self.value_name = value_name
        self.value_unit = value_unit
        self.label = label

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
             savefig_settings=None, ax=None, mute=False):
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
                    
                import pyleoclim as pyleo
                import pandas as pd
                from matplotlib import pyplot as plt
                data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
                time=data.iloc[:,1]
                value=data.iloc[:,2]
                ts=pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')
                @savefig ts_plot.png
                fig,ax = ts.plot()
                plt.close(fig)
        
        Change the line color
        
            .. ipython:: python
                :okwarning:
            
                @savefig ts_plot2.png
                fig, ax = ts.plot(color='r')
                plt.close(fig)
        
        Save the figure. Two options available:
            * Within the plotting command
            * After the figure has been generated
        
            .. ipython:: python
                :okwarning:
            
                #@savefig ts_plot3.png 
                fig,ax = ts.plot(color='k',savefig_settings={'path':'ts_plot3.png'})
                pyleo.savefig(fig,path='ts_plot3.png')
                plt.close(fig)
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

    def ssa(self, M=None, nMC=0, f=0.5):
        '''Singular Spectrum Analysis
        
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
            maximum allowable fraction of missing values. The default is 0.5.

        Returns
        -------
        res : dict
            Containing:

            - eig_val : (M, 1) array of eigenvalue spectrum of length r, the number of SSA modes. As in Principal Component Analysis, eigenvaluesare closely related to the fraction of variance accounted for ("explained", a common but not-so-helpful term) by each mode.
            
            - eig_vec : is a matrix of the temporal eigenvectors (T-EOFs), i.e. the temporal patterns that explain most of the variations in the original series.
                
            - PC : (N - M + 1, M) array of principal components, i.e. the loadings that, convolved with the T-EOFs, produce the reconstructed components, or RCs

            - RC : (N,  M) array of reconstructed components, One can think of each RC as the contribution of each mode to the timeseries, weighted by their eigenvalue (loosely speaking, their "amplitude"). Summing over all columns of RC recovers the original series. (synthesis, the reciprocal operation of analysis).

            - eig_val_q : (M, 2) array containing the 5% and 95% quantiles of the Monte-Carlo eigenvalue spectrum [ if MC >0 ]
        
        Examples
        --------
        
        SSA with SOI
        
        .. ipython:: python
            :okwarning:
            
            import pyleoclim as pyleo
            import pandas as pd
            from matplotlib import pyplot as plt
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time=data.iloc[:,1]
            value=data.iloc[:,2]
            ts=pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')
            #plot
            @savefig ts_plot.png
            fig,ax = ts.plot()
            plt.close(fig)
            #SSA
            nino_ssa = ts.ssa(M=60)
        
        Let us now see how to make use of all these arrays. The first step is too inspect the eigenvalue spectrum ("scree plot") to identify remarkable modes. Let us restrict ourselves to the first 40, so we can see something:
            
        .. ipython:: python
            :okwarning:
        
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import numpy as np
            
            d  = nino_ssa['eig_val'] # extract eigenvalue vector
            M  = len(d)  # infer window size
            de = d*np.sqrt(2/(M-1))
            var_pct = d**2/np.sum(d**2)*100  # extract the fraction of variance attributable to each mode

            # plot eigenvalues
            r = 20
            rk = np.arange(0,r)+1
            fig,ax = plt.subplots()
            ax.errorbar(rk,d[:r],yerr=de[:r],label='SSA eigenvalues w/ 95% CI')
            ax.set_title('Scree plot of SSA eigenvalues')
            ax.set_xlabel('Rank $i$'); plt.ylabel(r'$\lambda_i$')
            ax.legend(loc='upper right')   
            @savefig scree_plot.png
            pyleo.showfig(fig)
            plt.close(fig)
        
        This highlights a few common phenomena with SSA:
            * the eigenvalues are in descending order
            * their uncertainties are proportional to the eigenvalues themselves
            * the eigenvalues tend to come in pairs : (1,2) (3,4), are all clustered within uncertainties . (5,6) looks like another doublet 
            * around i=15, the eigenvalues appear to reach a floor, and all subsequent eigenvalues explain a very small amount of variance.
        
        So, summing the variance of all modes higher than 19, we get:
            
        .. ipython:: python
            :okwarning:
            
            print(var_pct[15:].sum()*100)
        
        That is, over 95% of the variance is in the first 15 modes. That is a typical result for a "warm-colored" timeseries, which is most geophysical timeseries; a few modes do the vast majority of the work. That means we can focus our attention on these modes and capture most of the interesting behavior. To see this, let's use the reconstructed components (RCs), and sum the RC matrix over the first 15 columns:
        
        .. ipython:: python
            :okwarning:
        
            RCk = nino_ssa['RC'][:,:14].sum(axis=1)
            fig, ax = ts.plot(title='ONI',mute=True) # we mute the first call to only get the plot with 2 lines
            ax.plot(time,RCk,label='SSA reconstruction, 14 modes',color='orange')
            ax.legend()
            @savefig ssa_recon.png
            pyleo.showfig(fig) 
            plt.close(fig)
        
        Indeed, these first few modes capture the vast majority of the low-frequency behavior, including all the El Niño/La Niña events. What is left (the blue wiggles not captured in the orange curve) are high-frequency oscillations that might be considered "noise" from the standpoint of ENSO dynamics. This illustrates how SSA might be used for filtering a timeseries. One must be careful however:
            * there was not much rhyme or reason for picking 15 modes. Why not 5, or 39? All we have seen so far is that they gather >95% of the variance, which is by no means a magic number.
            * there is no guarantee that the first few modes will filter out high-frequency behavior, or at what frequency cutoff they will do so. If you need to cut out specific frequencies, you are better off doing it with a classical filter, like the butterworth filter implemented in Pyleoclim. However, in many instances the choice of a cutoff frequency is itself rather arbitrary. In such cases, SSA provides a principled alternative for generating a version of a timeseries that preserves features and excludes others (i.e, a filter).
            * as with all orthgonal decompositions, summing over all RCs will recover the original signal within numerical precision.
        
        Monte-Carlo SSA
        
        Selecting meaningful modes in eigenproblems (e.g. EOF analysis) is more art than science. However, one technique stands out: Monte Carlo SSA, introduced by Allen & Smith, (1996) to identiy SSA modes that rise above what one would expect from "red noise", specifically an AR(1) process_process). To run it, simply provide the parameter MC, ideally with a number of iterations sufficient to get decent statistics. Here's let's use MC = 1000. The result will be stored in the eig_val_q array, which has the same length as eig_val, and its two columns contain the 5% and 95% quantiles of the ensemble of MC-SSA eigenvalues.
        
        .. ipython:: python
            :okwarning:
        
            nino_mcssa = ts.ssa(M = 60, nMC=1000)
        
        Now let's look at the result:
            
        .. ipython:: python
            :okwarning:
        
            d  = nino_mcssa['eig_val'] # extract eigenvalue vector
            de = d*np.sqrt(2/(M-1))
            du = nino_mcssa['eig_val_q'][:,0]  # extract upper quantile of MC-SSA eigenvalues
            dl = nino_mcssa['eig_val_q'][:,1]  # extract lower quantile of MC-SSA eigenvalues

            # plot eigenvalues
            rk = np.arange(0,20)+1
            fig=plt.figure()
            plt.fill_between(rk,dl[:20],du[:20],color='silver',alpha=0.5,label='MC-SSA 95% CI')
            plt.errorbar(rk,d[:20],yerr=de[:20],label='SSA eigenvalues w/ 95% CI')
            plt.title('Scree plot of SSA eigenvalues, w/ MC-SSA bounds')
            plt.xlabel('Rank $i$'); plt.ylabel(r'$\lambda_i$')
            plt.legend(loc='upper right')
            @savefig scree_nmc.png
            pyleo.showfig(fig)
            plt.close(fig)
    
        This suggests that modes 1-5 fall above the red noise benchmark.
        
        '''

        res = decomposition.ssa(self.value, M=M, nMC=nMC, f=f)
        return res

    def distplot(self, figsize=[10, 4], title=None, savefig_settings=None,
                 ax=None, ylabel='KDE', mute=False, **plot_kwargs):
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
        
        See also
        --------
        
        pyleoclim.utils.plotting.savefig : saving figure in Pyleoclim
        
        Examples
        --------
        
        Distribution of the SOI record
        
        .. ipython:: python
            :okwarning:
            
            import pyleoclim as pyleo
            import pandas as pd
            from matplotlib import pyplot as plt
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time=data.iloc[:,1]
            value=data.iloc[:,2]
            ts=pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')
            @savefig ts_plot.png
            fig,ax = ts.plot()
            @savefig ts_dist.png
            fig,ax = ts.distplot()
            plt.close(fig)
        
        
        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
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
                plotting.savefig(fig, settings=savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def summary_plot(self, psd=None, scalogram=None, figsize=[8, 10], title=None, savefig_settings=None,
                    time_lim=None, value_lim=None, period_lim=None, psd_lim=None, n_signif_test=100,
                    time_label=None, value_label=None, period_label=None, psd_label=None, mute=False):
        ''' Generate a plot of the timeseries and its frequency content through spectral and wavelet analyses. 

        Parameters
        ----------

        psd : PSD
            the PSD object of a Series. If None, will be calculated. This process can be slow as it will be using the WWZ method.

        scalogram : Scalogram
            the Scalogram object of a Series. If None, will be calculated. This process can be slow as it will be using the WWZ method.

        figsize : list
            a list of two integers indicating the figure size

        title : str
            the title for the figure

        time_lim : list or tuple
            the limitation of the time axis

        value_lim : list or tuple
            the limitation of the value axis of the timeseries

        period_lim : list or tuple
            the limitation of the period axis

        psd_lim : list or tuple
            the limitation of the psd axis
        
        n_signif_test=100 : int
            Number of Monte-Carlo simulations to perform for significance testing. Used when psd=None or scalogram=None

        time_label : str
            the label for the time axis

        value_label : str
            the label for the value axis of the timeseries

        period_label : str
            the label for the period axis

        psd_label : str
            the label for the amplitude axis of PDS

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        mute : {True,False}
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
        
        See also
        --------
        
        pyleoclim.core.ui.Series.spectral : Spectral analysis for a timeseries
        
        pyleoclim.core.ui.Series.wavelet : Wavelet analysis for a timeseries
        
        pyleoclim.utils.plotting.savefig : saving figure in Pyleoclim 
        
        pyleoclim.core.ui.PSD : PSD object
        
        pyleoclim.core.ui.MultiplePSD : Multiple PSD object
        
        Examples
        --------
        
        Create a summary plot for the SOI dataset. Note: because the wwz method can be slow, only 10 AR1 models are generated in this example. For normal applications, we recommend at least 200. 
        
        .. ipython:: python
            :okwarning:
            
            import pyleoclim as pyleo
            import pandas as pd
            from matplotlib import pyplot as plt
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time=data.iloc[:,1]
            value=data.iloc[:,2]
            ts=pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')
            #Perform spectral analysis
            psd=ts.spectral()
            # Significance testing
            psd_signif=psd.signif_test(number=1)
            # Perform wavelet analysis
            scal=ts.wavelet()
            # Significance testing
            scal_signif = scal.signif_test(number=1)
            @savefig ts_summary_plot.png
            fig, ax = ts.summary_plot(
                        scalogram=scal_signif, psd=psd_signif,
                        psd_lim=[1e-2, 1e2],
                        period_lim=[0.2, 50],
                        value_label='SOI [K]',
                        period_label='Period [yrs]',
                        time_label='Year (CE)',
                        psd_label='PSD',
                        title='Summary of SOI timeseries'
                        )
            plt.close(fig)
        
        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(6, 12)
        gs.update(wspace=0, hspace=0)

        ax = {}
        ax['ts'] = plt.subplot(gs[0:1, :-3])
        ax['ts'] = self.plot(ax=ax['ts'])
        if time_lim is not None:
            ax['ts'].set_xlim(time_lim)
        if value_lim is not None:
            ax['ts'].set_ylim(value_lim)

        ax['ts'].spines['bottom'].set_visible(False)

        ax['scal'] = plt.subplot(gs[1:5, :-3], sharex=ax['ts'])
        if scalogram is None:
            scalogram = self.wavelet().signif_test(number=n_signif_test)

        ax['scal'] = scalogram.plot(ax=ax['scal'], cbar_style={'orientation': 'horizontal', 'pad': 0.1})

        ax['psd'] = plt.subplot(gs[1:4, -3:], sharey=ax['scal'])
        if psd is None:
            psd = self.spectral().signif_test(number=n_signif_test)

        ax['psd'] = psd.plot(ax=ax['psd'], transpose=True)
        if period_lim is not None:
            ax['psd'].set_ylim(period_lim)
        ax['psd'].set_ylabel(None)
        ax['psd'].tick_params(axis='y', direction='in', labelleft=False)
        ax['psd'].legend().remove()

        if psd_lim is not None:
            ax['psd'].set_xlim(psd_lim)

        if title is not None:
            ax['ts'].set_title(title)

        if value_label is not None:
            ax['ts'].set_ylabel(value_label)

        if time_label is not None:
            ax['scal'].set_xlabel(time_label)

        if period_label is not None:
            ax['scal'].set_ylabel(period_label)

        if psd_label is not None:
            ax['psd'].set_xlabel(psd_label)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)
        else:
            if not mute:
                plotting.showfig(fig)
        return fig, ax

    def copy(self):
        '''Make a copy of the Series object

        Returns
        -------
        Series
            A copy of the Series object

        '''
        return deepcopy(self)

    def clean(self):
        ''' Clean up the timeseries by removing NaNs and sort with increasing time points
        
        Returns
        -------
        Series
            Series object with removed NaNs and sorting
        
        '''
        new = self.copy()
        v_mod, t_mod = tsutils.clean_ts(self.value, self.time)
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
        '''Standardizes the time series

        Returns
        -------
        new : pyleoclim.Series
            The standardized series object

        '''
        new = self.copy()
        v_mod = tsutils.standardize(self.value)[0]
        new.value = v_mod
        return new

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

    def detrend(self, method='emd', **kwargs):
        '''Detrend Series object        

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
        new : pyleoclim.Series
            Detrended Series object
        
        See also
        --------
        pyleoclim.utils.tsutils.detrend : detrending wrapper functions
        
        Examples
        --------
        
        We will generate a random signal and use the different detrending functions
        
        .. ipython:: python
            :okwarning:
            
            import pyleoclim as pyleo
            import numpy as np
            from matplotlib import pyplot as plt
            
            # Generate a mixed signal with known frequencies
            freqs=[1/20,1/80]
            time=np.arange(2001)
            signals=[]
            for freq in freqs:
                signals.append(np.cos(2*np.pi*freq*time))
            signal=sum(signals)
            
            # Add a non-linear trend
            slope = 1e-5
            intercept = -1
            nonlinear_trend = slope*time**2 + intercept
            signal_trend = signal + nonlinear_trend

            #Add white noise
            sig_var = np.var(signal)
            noise_var = sig_var / 2 #signal is twice the size of noise
            white_noise = np.random.normal(0, np.sqrt(noise_var), size=np.size(signal))
            signal_noise = signal_trend + white_noise
            
            #Create a series object
            ts=pyleo.Series(time=time,value=signal_noise)
            @savefig random_series.png
            fig,ax = ts.plot(title='Timeseries with nonlinear trend')
            plt.close(fig)
            
            #Standardize
            ts_std=ts.standardize()
            
            #Detrend using EMD 
            ts_emd = ts_std.detrend()
            @savefig ts_emd.png
            fig,ax=ts_emd.plot(title='Detrended with EMD method')
            plt.close(fig)
            
            #Detrend using Savitzky-Golay filter
            ts_sg = ts_std.detrend(method='savitzky-golay')
            @savefig ts_sg.png
            fig,ax=ts_sg.plot(title='Detrended with Savitzky-Golay filter')
            plt.close(fig)

        '''
        new = self.copy()
        v_mod = tsutils.detrend(self.value, x=self.time, method=method, **kwargs)
        new.value = v_mod
        return new

    def spectral(self, method='wwz', freq_method='log', freq_kwargs=None, settings=None, label=None, verbose=False):
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
        
        verbose : {True, False}
        
        Returns
        -------
        
        psd : pyleoclim.Psd
            A PSD object
            
        See also
        --------
        pyleoclim.utils.spectral.mtm : Spectral analysis using the Multitaper approach
        
        pyleoclim.utils.spectral.lomb_scargle : Spectral analysis using the Lomb-Scargle method
        
        pyleoclim.utils.spectral.welch: Spectral analysis using the Welch segement approach
        
        pyleoclim.utils.spectral.periodogram: Spectral anaysis using the basic Fourier transform
        
        pyleoclim.utils.spectral.wwz_psd : Spectral analysis using the Wavelet Weighted Z transform
        
        pyleoclim.utils.wavelet.make_freq : Functions to create the frequency vector
        
        pyleoclim.utils.tsutils.detrend : Detrending function
        
        pyleoclim.core.ui.PSD : PSD object
        
        pyleoclim.core.ui.MultiplePSD : Multiple PSD object
        
        
        Examples
        --------
        
        Calculate the spectrum of SOI using the various methods and compute significance
        
        .. ipython:: python
            :okwarning:
        
            import pyleoclim as pyleo
            import pandas as pd
            from matplotlib import pyplot as plt
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time=data.iloc[:,1]
            value=data.iloc[:,2]
            ts=pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')
            # Standardize the time series
            ts_std=ts.standardize()
            # WWZ
            psd_wwz=ts_std.spectral()
            psd_wwz_signif=psd_wwz.signif_test(number=1)  # for real work, should use number=200 or even larger
            @savefig spec_wwz.png
            fig,ax=psd_wwz_signif.plot(title='PSD using WWZ method')
            plt.close(fig)
            #Periodogram
            ts_interp = ts_std.interp()
            psd_perio=ts_interp.spectral(method='periodogram')
            psd_perio_signif=psd_perio.signif_test()
            @savefig spec_perio.png
            fig,ax=psd_perio_signif.plot(title='PSD using Periodogram method')
            plt.close(fig)
            #Welch
            ts_interp = ts_std.interp()
            psd_welch=ts_interp.spectral(method='welch')
            psd_welch_signif=psd_welch.signif_test()
            @savefig spec_welch.png
            fig,ax=psd_welch_signif.plot(title='PSD using Welch method')
            plt.close(fig)
            #MTM
            ts_interp = ts_std.interp()
            psd_mtm=ts_interp.spectral(method='mtm')
            psd_mtm_signif=psd_mtm.signif_test()
            @savefig spec_mtm.png
            fig,ax=psd_mtm_signif.plot(title='PSD using MTM method')
            plt.close(fig)
            #Lomb-Scargle
            psd_ls=ts_std.spectral(method='lomb_scargle')
            psd_ls_signif=psd_ls.signif_test()
            @savefig spec_ls.png
            fig,ax=psd_ls_signif.plot(title='PSD using Lomb-Scargle method')
            plt.close(fig)
        
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

    def wavelet(self, method='wwz', settings=None, freq_method='log', freq_kwargs=None, verbose=False):
        ''' Perform wavelet analysis on the timeseries

        cwt wavelets documented on https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
        
        Parameters
        ----------
        
        method : {wwz,cwt}
            Whether to use the wwz method for unevenly spaced timeseries or traditional cwt (from pywavelets)
            
        freq_method : str
            {'log','scale', 'nfft', 'lomb_scargle', 'welch'}
        
        freq_kwargs : dict
            Arguments for frequency vector
        
        settings : dict
            Arguments for the specific spectral method
        
        verbose : {True, False}
        
        Returns
        -------
        
        scal : Series.Scalogram
        
        See also
        --------
        
        pyleoclim.utils.wavelet.wwz : wwz function
        
        pyleoclim.utils.wavelet.cwt : cwt function
        
        pyleoclim.utils.wavelet.make_freq : Functions to create the frequency vector
        
        pyleoclim.utils.tsutils.detrend : Detrending function
        
        pyleoclim.core.ui.Scalogram : Scalogram object
        
        pyleoclim.core.ui.MultipleScalogram : Multiple Scalogram object
        
        Examples
        --------
        
        Wavelet analysis on the SOI record. 
        
        .. ipython:: python
            :okwarning:
        
            import pyleoclim as pyleo
            import pandas as pd
            from matplotlib import pyplot as plt
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            time=data.iloc[:,1]
            value=data.iloc[:,2]
            ts=pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')
            #WWZ
            scal = ts.wavelet()
            scal_signif = scal.signif_test(number=1)  # for real work, should use number=200 or even larger
            @savefig wave_wwz.png
            fig,ax=scal_signif.plot()
            plt.close(fig)
        
        '''
        if not verbose:
            warnings.simplefilter('ignore')

        settings = {} if settings is None else settings.copy()
        wave_func = {
            'wwz': waveutils.wwz,
            'cwt': waveutils.cwt,
        }
        
        if method == 'cwt' and 'freq' in settings.keys():
            scales=1/np.array(settings['freq'])
            settings.update({'scales':scales})
            del settings['freq']

        freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
        freq = waveutils.make_freq_vector(self.time, method=freq_method, **freq_kwargs)

        args = {}

        args['wwz'] = {'tau': self.time, 'freq': freq}
        args['cwt'] = {'wavelet' : 'morl', 'scales':1/freq}


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
            freq_method=freq_method,
            freq_kwargs=freq_kwargs,
            wave_args=args[method],
        )

        return scal

    def wavelet_coherence(self, target_series, method='wwz', settings=None, freq_method='log', freq_kwargs=None, verbose=False):
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
        
        settings : dict
            Arguments for the specific spectral method
        
        verbose : {True, False}
        
        Returns
        -------
        
        coh : pyleoclim.Coherence
        
        See also
        --------
        
        pyleoclim.utils.wavelet.xwt : Cross-wavelet analysis based on WWZ method
        
        pyleoclim.utils.wavelet.make_freq : Functions to create the frequency vector
        
        pyleoclim.utils.tsutils.detrend : Detrending function
        
        pyleoclim.core.ui.Coherence : Coherence object

        Examples
        --------

        .. ipython:: python
            :okwarning:
        
            import pyleoclim as pyleo
            import pandas as pd
            from matplotlib import pyplot as plt
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino.csv')
            t=data.iloc[:,0]
            air=data.iloc[:,1]
            nino=data.iloc[:,2]
            ts_nino=pyleo.Series(time=t,value=nino)
            ts_air=pyleo.Series(time=t,value=air)
            #plot the two timeseries
            @savefig ts_nino.png
            fig, ax = ts_nino.plot(title='El Nino Region 3 -- SST Anomalies')
            plt.close(fig)
            @savefig ts_air.png
            fig, ax = ts_air.plot(title='Deasonalized All Indian Rainfall Index')
            plt.close(fig)
            ts_air_std=ts_air.standardize()
            ts_nino_std=ts_nino.standardize()
            coh = ts_nino.wavelet_coherence(ts_air)
            coh_signif = coh.signif_test(number=1, qs=[0.99])  # for real work, should use number=200 or even larger
            @savefig coh_plot.png
            fig, ax = coh_signif.plot(phase_style={'skip_x': 50, 'skip_y': 10}) 
            plt.close(fig)
        
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
            freq_method=freq_method,
            freq_kwargs=freq_kwargs,
        )

        return coh

    def correlation(self, target_series, timespan=None, settings=None):
        ''' Estimates the Pearson's correlation and associated significance between two non IID time series
    
        The significance of the correlation is assessed using one of the following methods:
        
        1) 'ttest': T-test adjusted for effective sample size.
        2) 'isopersistent': AR(1) modeling of x and y.
        3) 'isospectral': phase randomization of original inputs. (default)
    
        The T-test is a parametric test, hence computationally cheap but can only be performed in idyllic circumstances.
        The others are non-parametric, but their computational requirements scales with nsim.
        
        The choise of significance test and associated number of Monte-Carlo simulations are passed through the settings parameter.
        
        Parameters
        ----------
        
        target_series : pyleoclim.Series
            A pyleoclim Series object
        
        timespan : tuple
            The time interval over which to perform the calculation
        
        settings : dict
            Parameters for the correlation function (singificance testing and number of simulation)
        
        Returns
        -------
        
        res : dict
            Containing the Pearson's correlation coefficient, associated significance and p-value. 
        
        See also
        --------
        
        pyleoclim.utils.correlation.corr_sig : Correlation function

        Examples
        --------

        Correlation between the Nino3.4 index and the Deasonalized All Indian Rainfall Index
        
        .. ipython:: python
            :okwarning:
        
            import pyleoclim as pyleo
            import pandas as pd
            from matplotlib import pyplot as plt
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino.csv')
            t=data.iloc[:,0]
            air=data.iloc[:,1]
            nino=data.iloc[:,2]
            ts_nino=pyleo.Series(time=t,value=nino)
            ts_air=pyleo.Series(time=t,value=air)
            #plot the two timeseries
            @savefig ts_nino.png
            fig, ax = ts_nino.plot(title='El Nino Region 3 -- SST Anomalies')
            plt.close(fig)
            @savefig ts_air.png
            fig, ax = ts_air.plot(title='Deasonalized All Indian Rainfall Index')
            plt.close(fig)
            corr_res = ts_nino.correlation(ts_air)
            print(corr_res)
        '''
        settings = {} if settings is None else settings.copy()
        args = {}
        args.update(settings)

        if timespan is None:
            value1 = self.value
            value2 = target_series.value
        else:
            value1 = self.slice(timespan).value
            value2 = target_series.slice(timespan).value

        corr_res = corrutils.corr_sig(value1, value2, **args)
        return corr_res

    def causality(self, target_series, method='liang', settings=None):
        ''' Perform causality analysis with the target timeseries
        
        Parameters
        ----------
        
        target_series : pyleoclim.Series
            A pyleoclim Series object on which to compute causality
        
        method : {'liang', 'granger'}
            The causality method to use.
        
        settings : dict
            Parameters associated with the causality methods  . Note that each method has different parameters. See individual methods for details
        
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
        
            import pyleoclim as pyleo
            import pandas as pd
            from matplotlib import pyplot as plt
            data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino.csv')
            t=data.iloc[:,0]
            air=data.iloc[:,1]
            nino=data.iloc[:,2]
            ts_nino=pyleo.Series(time=t,value=nino)
            ts_air=pyleo.Series(time=t,value=air)
            #plot the two timeseries
            @savefig ts_nino.png
            fig, ax = ts_nino.plot(title='El Nino Region 3 -- SST Anomalies')
            plt.close(fig)
            @savefig ts_air.png
            fig, ax = ts_air.plot(title='Deasonalized All Indian Rainfall Index')
            plt.close(fig)
            # we use the specific params below in ts_nino.causality() just to make the example less heavier;
            # please drop the `settings` for real work
            caus_res = ts_nino.causality(ts_air, settings={'nsim': 2, 'signif_test': 'isopersist'})
            print(caus_res)
        
        Granger causality
        
        .. ipython:: python
            :okwarning:
                
            caus_res = ts_nino.causality(ts_air, method='granger')
            print(caus_res)
        
        '''
        settings = {} if settings is None else settings.copy()
        spec_func={
            'liang':causalutils.liang_causality,
            'granger':causalutils.granger_causality}
        args = {}
        args['liang'] = {}
        args['granger'] = {}
        args[method].update(settings)
        causal_res = spec_func[method](self.value, target_series.value, **args[method])
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
            plot_outliers_kwargs=plot_outliers_kwargs,plot_knee_kwargs=plot_knee_kwargs, mute=mute,
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
        '''Interpolate a time series onto  a new  time axis

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
        x_mod, v_mod = tsutils.interp(self.time,self.value,interp_type=method,**kwargs)
        new.time = x_mod
        new.value = v_mod
        return new

    def bin(self,**kwargs):
        '''Bin values in a time series
        
        Parameters
        ----------
        
        kwargs : 
            Arguments for binning function. See pyleoclim.utils.tsutils.bin_values for details
        
        Returns
        -------
        
        new : pyleoclim.Series
            An binned Series object
        
        See also
        --------
        
        pyleoclim.utils.tsutils.bin_values : bin the time series into evenly-spaced bins
                
        '''
        new=self.copy()
        res_dict = tsutils.bin_values(self.time,self.value,**kwargs)
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
                 spec_method=None, spec_args=None, signif_qs=None, signif_method=None, period_unit=None):
        self.frequency = np.array(frequency)
        self.amplitude = np.array(amplitude)
        self.label = label
        self.timeseries = timeseries
        self.spec_method = spec_method
        self.spec_args = spec_args
        self.signif_qs = signif_qs
        self.signif_method = signif_method
        self.plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        if period_unit is not None:
            self.period_unit = period_unit
        elif timeseries is not None:
            self.period_unit = f'{timeseries.time_unit}s'
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

    def signif_test(self, number=200, method='ar1', seed=None, qs=[0.95],
                    settings=None):
        '''
        

        Parameters
        ----------
        number : int, optional
            Number of surrogate series to generate for significance testing. The default is 200.
        method : {ar1}, optional
            Method to generate surrogates. The default is 'ar1'.
        seed : int, optional
            Option to set the seed for reproducibility. The default is None.
        qs : list, optional
            Singificance levels to return. The default is [0.95].
        settings : dict, optional
            Parameters. The default is None.

        Returns
        -------
        new : pyleoclim.PSD
            New PSD object with appropriate significance test

        '''
        new = self.copy()
        surr = self.timeseries.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )
        surr_psd = surr.spectral(method=self.spec_method, settings=self.spec_args)
        new.signif_qs = surr_psd.quantiles(qs=qs)
        new.signif_method = method

        return new

    def beta_est(self, fmin=None, fmax=None, verbose=False):
        ''' Estimate the scaling factor beta of the PSD in a log-log space

        Parameters
        ----------

        fmin : float
            the minimum frequency edge for beta estimation; the default is the minimum of the frequency vector of the PSD obj

        fmax : float
            the maximum frequency edge for beta estimation; the default is the maximum of the frequency vector of the PSD obj

        verbose : bool
            if True, will print out debug information

        Returns
        -------

        res_dict : dictionary
            - beta: the scaling factor
            - std_err: the one standard deviation error of the scaling factor
            - f_binned: the binned frequency series, used as X for linear regression
            - psd_binned: the binned PSD series, used as Y for linear regression
            - Y_reg: the predicted Y from linear regression, used with f_binned for the slope curve plotting

        '''
        if fmin is None:
            fmin = np.min(self.frequency)

        if fmax is None:
            fmax = np.max(self.frequency)

        res = waveutils.beta_estimation(self.amplitude, self.frequency, fmin=fmin, fmax=fmax, verbose=verbose)
        res_dict = {
            'beta': res.beta,
            'std_err': res.std_err,
            'f_binned': res.f_binned,
            'psd_binned': res.psd_binned,
            'Y_reg': res.Y_reg,
        }

        return res_dict

    def plot(self, in_loglog=True, in_period=True, label=None, xlabel=None, ylabel='Amplitude', title=None,
             marker=None, markersize=None, color=None, linestyle=None, linewidth=None, transpose=False,
             xlim=None, ylim=None, figsize=[10, 4], savefig_settings=None, ax=None, mute=False,
             plot_legend=True, lgd_kwargs=None, xticks=None, yticks=None, alpha=None, zorder=None,
             plot_kwargs=None, signif_clr='red', signif_linestyles=['--', '-.', ':'], signif_linewidth=1):
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
            Label for the y-axis. The default is 'Amplitude'. Will guess based on Series
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
        plot_legend : bool, optional
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

        Returns
        -------
        fig, ax
        
        See also
        --------
        
        pyleoclim.core.ui.Series.spectral : spectral analysis

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = self.plot_kwargs if plot_kwargs is None else plot_kwargs.copy()
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
                xlabel = f'Period [{self.period_unit}]' if self.period_unit is not None else 'Period'

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

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

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
            ax.set_xscale('log', nonpositive='clip')
            ax.set_yscale('log', nonpositive='clip')

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
                plotting.savefig(fig, settings=savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
            return fig, ax
        else:
            return ax

class Scalogram:
    def __init__(self, frequency, time, amplitude, coi=None, label=None, Neff=3, timeseries=None,
                 wave_method=None, wave_args=None, signif_qs=None, signif_method=None, freq_method=None, freq_kwargs=None,
                 period_unit=None, time_label=None):
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
        self.freq_method = freq_method
        self.freq_kwargs = freq_kwargs
        if period_unit is not None:
            self.period_unit = period_unit
        elif timeseries is not None:
            self.period_unit = timeseries.time_unit
        if time_label is not None:
            self.time_label = time_label
        elif timeseries is not None:
            self.time_label = f'{timeseries.time_name}'

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
                yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
                mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
                yticks = yticks_default[mask]
        else:
            y_axis = self.frequency
            if ylabel is None:
                ylabel = f'Frequency [1/{self.period_unit}]' if self.period_unit is not None else 'Frequency'

        if xlabel is None:
            xlabel = self.time_label

        cont = ax.contourf(self.time, y_axis, self.amplitude.T, **contourf_args)
        ax.set_yscale('log', nonpositive='clip')

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
                plotting.savefig(fig, settings=savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    def signif_test(self, number=200, method='ar1', seed=None, qs=[0.95],
                    settings=None):
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

        Raises
        ------
        ValueError
            qs should be a list with at least one value.

        Returns
        -------
        new : pyleoclim.Scalogram
            A new Scalogram object with the significance level
        
        See also
        --------
        
        pyleoclim.core.ui.Series.wavelet : wavelet analysis

        '''
        
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
            self.period_unit = timeseries1.time_unit
        if time_label is not None:
            self.time_label = time_label
        elif timeseries1 is not None:
            self.time_label = f'{timeseries1.time_name}'

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
        contourf_style : dict, optional
            Arguments for the contour plot. The default is {}.
        phase_style : dict, optional
            Arguments for the phase arrows. The default is {}.
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

        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if in_period:
            y_axis = 1/self.frequency
            if ylabel is None:
                ylabel = f'Period [{self.period_unit}]' if self.period_unit is not None else 'Period'

            if yticks is None:
                yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
                mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
                yticks = yticks_default[mask]
        else:
            y_axis = self.frequency
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
        yaxis_range = np.max(y_axis) - np.min(y_axis)
        xaxis_range = np.max(self.time) - np.min(self.time)
        phase_args = {'pt': 0.5, 'skip_x': int(xaxis_range//10), 'skip_y': int(yaxis_range//50), 'scale': 30, 'width': 0.004}
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
                plotting.savefig(fig, settings=savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
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
        
        
        new = self.copy()
        surr1 = self.timeseries1.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )
        surr2 = self.timeseries2.surrogates(
            number=number, seed=seed, method=method, settings=settings
        )

        cohs = []
        for i in tqdm(range(number), desc='Performing wavelet coherence on surrogate pairs', position=0, leave=True, disable=mute_pbar):
            coh_tmp = surr1.series_list[i].wavelet_coherence(surr2.series_list[i], freq_method=self.freq_method, freq_kwargs=self.freq_kwargs)
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
    '''Define a multiple series object.
    
    This is useful for ensembling or working with multiple timeseries in the same workflow
    
    Parameters
    ----------
    
    series_list : list
        a list of pyleoclim.Series objects
    
    Examples
    --------
    
    Create a MultipleSeries object for the Nino and All Indian Rainfall indices
    
    .. ipython:: python
        :okwarning:
        
        import pyleoclim as pyleo
        import pandas as pd
        data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/wtc_test_data_nino.csv')
        t=data.iloc[:,0]
        air=data.iloc[:,1]
        nino=data.iloc[:,2]
        ts_nino=pyleo.Series(time=t,value=nino)
        ts_air=pyleo.Series(time=t,value=air)
        series_list=[ts_nino,ts_air]
        ts_all = pyleo.MultipleSeries(series_list)
    '''
    def __init__(self, series_list):
        self.series_list = series_list

    def copy(self):
        '''Copy the object 
        '''
        return deepcopy(self)

    def standardize(self):
        '''Standardize each series object

        Returns
        -------
        new : pyleoclim.MultipleSeries
            The standardized Series

        '''
        new=self.copy()
        for idx,item in enumerate(new.series_list):
            s=item.copy()
            v_mod=tsutils.standardize(item.value)[0]
            s.value=v_mod
            new.series_list[idx]=s
        return new

    # def mssa(self, M, MC=0, f=0.5):
    #     data = []
    #     for val in self.series_list:
    #         data.append(val.value)
    #     data = np.transpose(np.asarray(data))


    #     res = decomposition.mssa(data, M=M, MC=MC, f=f)
    #     return res

    # def pca(self):
    #     data = []
    #     for val in self.series_list:
    #         data.append(val.value)
    #     a = len(data[0])
    #     r = data[1:]
    #     flag = all (len(v)==a for v in r)
    #     if flag==False:
    #         print('All Time Series should be of same length')
    #         return
    #     data = np.transpose(np.asarray(data))
    #     res = decomposition.pca(data)
    #     return res

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
        new : pyleoclim.MultipleSeries
            The detrended timeseries
            
        See also
        --------
        
        pyleoclim.core.ui.Series.detrend : Detrending for a single series
        pyleoclim.utils.tsutils.detrend : Detrending function

        '''
        new=self.copy()
        for idx,item in enumerate(new.series_list):
            s=item.copy()
            v_mod=tsutils.detrend(item.value,x=item.time,method=method,**kwargs)
            s.value=v_mod
            new.series_list[idx]=s
        return new

    def spectral(self, method='wwz', settings={}, mute_pbar=False, freq_method='log', freq_kwargs=None):
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
        
        mute_pbar : {True, False}
            Mute the progress bar. Default is False. 
        
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
        
        pyleoclim.utils.wavelet.make_freq : Functions to create the frequency vector
        
        pyleoclim.utils.tsutils.detrend : Detrending function
        
        pyleoclim.core.ui.Series.spectral : Spectral analysis for a single timeseries
        
        pyleoclim.core.ui.PSD : PSD object
        
        pyleoclim.core.ui.MultiplePSD : Multiple PSD object
        '''
        settings = {} if settings is None else settings.copy()
        if method in ['wwz', 'lomb_scargle'] and 'freq' not in settings.keys():
            res=[]
            for s in(self.series_list):
                c=np.mean(np.diff(s.value))
                res.append(c)
            res=np.array(res)
            idx = np.argmin(res)
            ts=self.series_list[idx].time
            freq_kwargs = {} if freq_kwargs is None else freq_kwargs.copy()
            freq=waveutils.make_freq_vector(ts, freq_method=freq_method, **freq_kwargs)
            settings.update({'freq':freq})
        psd_list = []
        for s in tqdm(self.series_list, desc='Performing spectral analysis on surrogates', position=0, leave=True, disable=mute_pbar):
            psd_tmp = s.spectral(method=method, settings=settings)
            psd_list.append(psd_tmp)

        psds = MultiplePSD(psd_list=psd_list)

        return psds

    def wavelet(self, method='wwz', settings={}, mute_pbar=False):
        '''Wavelet analysis        

        Parameters
        ----------
        method : {'wwz'}, optional
            Method for the wavelet analysis. The default is 'wwz'.
        settings : dict, optional
            Settings for the particular method. The default is {}.
        mute_pbar : bool, optional
            Whether to mute the progress bar. The default is False.

        Returns
        -------
        scals : pyleoclim.MultipleScalograms
        
        See also
        --------
        pyleoclim.utils.wavelet.wwz : wwz function
        
        pyleoclim.utils.wavelet.make_freq : Functions to create the frequency vector
        
        pyleoclim.utils.tsutils.detrend : Detrending function
        
        pyleoclim.core.ui.Series.wavelet : wavelet analysis on single object
        
        pyleoclim.core.ui.MultipleScalogram : Multiple Scalogram object
        
        '''
        settings = {} if settings is None else settings.copy()

        scal_list = []
        for s in tqdm(self.series_list, desc='Performing wavelet analysis on surrogates', position=0, leave=True, disable=mute_pbar):
            scal_tmp = s.wavelet(method=method, settings=settings)
            scal_list.append(scal_tmp)

        scals = MultipleScalogram(scalogram_list=scal_list)

        return scals

    def plot(self, figsize=[10, 4],
             marker=None, markersize=None, color=None,
             linestyle=None, linewidth=None,
             label=None, xlabel=None, ylabel=None, title=None,
             legend=True, plot_kwargs=None, lgd_kwargs=None,
             savefig_settings=None, ax=None, mute=False):
        '''Plot multiple timeseries on the same axis

        Parameters
        ----------
        figsize : list, optional
            Size of the figure. The default is [10, 4].
        marker : str, optional
            marker type. The default is None.
        markersize : float, optional
            marker size. The default is None.
        color : str, optional
            color. The default is None.
        linestyle : str, optional
            Line style. The default is None.
        linewidth : float, optional
            The width of the line. The default is None.
        label : str, optional
            Label for the series. The default is None.
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
        savefig_settings : TYPE, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.
        ax : matplotlib.ax, optional
            The matplotlib axis onto which to return the figure. The default is None.
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            
        Returns
        -------
        fig, ax
            
        '''

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
                plotting.savefig(fig, settings=savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
            return fig, ax
        else:
            return ax

    # def stackplot(self, figsize=None, xlabel=None, ylabel=None, 
    #           xlim=None, ylim=None, title=None,
    #           savefig_settings=None, ax=None, style=None, 
    #           plot_kwargs=None, mute=False,color=None):
    #     x = []
    #     y = []
    #     for s in self.series_list:
    #         x.append(s.time)
    #         y.append(s.value)
    #     color = pl.cm.jet(np.linspace(0.3, 1, len(x)))
    #     fig,ax = plotting.stackplot(x,y,figsize=figsize,color=color,xlabel=xlabel,
    #                                 ylabel=ylabel,title=title,plot_kwargs=plot_kwargs,
    #                                 savefig_settings=savefig_settings,ax=ax,mute=mute)
    #     return fig,ax

class SurrogateSeries(MultipleSeries):
    ''' Object containing surrogate timeseries
    '''
    def __init__(self, series_list, surrogate_method=None, surrogate_args=None):
        self.series_list = series_list
        self.surrogate_method = surrogate_method
        self.surrogate_args = surrogate_args

class MultiplePSD:
    ''' Object for multiple PSD.
    
    Used for significance level
    '''
    def __init__(self, psd_list):
        self.psd_list = psd_list

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

    def beta_est(self, fmin=None, fmax=None, verbose=False):
        ''' Estimate the scaling factor beta of the each PSD from the psd_list in a log-log space

        Parameters
        ----------

        fmin : float
            the minimum frequency edge for beta estimation; the default is the minimum of the frequency vector of the PSD obj

        fmax : float
            the maximum frequency edge for beta estimation; the default is the maximum of the frequency vector of the PSD obj

        verbose : bool
            if True, will print out debug information

        Returns
        -------

        res_dict : dictionary
            - beta: list of the scaling factors
            - std_err: list of one standard deviation errors of the scaling factor
            - f_binned: list of the binned frequency series, used as X for linear regression
            - psd_binned: list of the binned PSD series, used as Y for linear regression
            - Y_reg: list of the predicted Y from linear regression, used with f_binned for the slope curve plotting

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
        for psd_obj in self.psd_list:
            res = psd_obj.beta_est(fmin=fmin, fmax=fmax, verbose=verbose)
            for k in res_dict.keys():
                res_dict[k].append(res[k])

        return res_dict


    def plot(self, figsize=[10, 4], in_loglog=True, in_period=True, xlabel=None, ylabel='Amplitude', title=None,
             xlim=None, ylim=None, savefig_settings=None, ax=None, xticks=None, yticks=None, plot_legend=True,
             plot_kwargs=None, lgd_kwargs=None, mute=False):
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
        plot_legend : bool, optional
            Whether to plot the legend. The default is True.
        plot_kwargs : TYPE, optional
            Parameters for plot function. The default is None.
        lgd_kwargs : TYPE, optional
            Parameters for legend. The default is None.
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax
            The default is False.

        Returns
        -------
        fig,ax
        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for psd in self.psd_list:
            tmp_plot_kwargs = {}
            if psd.plot_kwargs is not None:
                tmp_plot_kwargs.update(psd.plot_kwargs)
            tmp_plot_kwargs.update(plot_kwargs)
            ax = psd.plot(
                figsize=figsize, in_loglog=in_loglog, in_period=in_period, xlabel=xlabel, ylabel=ylabel,
                title=title, xlim=xlim, ylim=ylim, savefig_settings=savefig_settings, ax=ax,
                xticks=xticks, yticks=yticks, plot_legend=plot_legend, plot_kwargs=tmp_plot_kwargs, lgd_kwargs=lgd_kwargs,
            )

        if title is not None:
            ax.set_title(title)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            else:
                if not mute:
                    plotting.showfig(fig)
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
        in_period : TYPE, optional
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
            else:
                if not mute:
                    plotting.showfig(fig)
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


class Lipd:
    '''Create a Lipd object from Lipd Files
    
    Parameters
    ----------
    
    usr_path : str
        path to the Lipd file(s). Can be URL (LiPD utilities only support loading one file at a time from a URL)
        If it's a URL, it must start with "http", "https", or "ftp.
    
    lidp_dict : dict
        LiPD files already loaded into Python through the LiPD utilities
    
    TODO
    ----
    
    Support querying the LinkedEarth platform

    Examples
    --------
    
    .. ipython:: python
        :okwarning:
        
        import pyleoclim as pyleo
        url='http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
        d=pyleo.Lipd(usr_path=url)
    '''
    
    def __init__(self, query=False, query_args={}, usr_path=None, lipd_dict=None):
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
                'mollusk shells' : ['#FFD600','h'],
                'peat' : ['#2F4F4F','*'],
                'other':['k','o']}

        #check that query has matching terms
        if query==True and bool(query_args)==False:
            raise ValueError('When query is set to true, you must define query terms')
        if query==False and usr_path==None and lipd_dict==None:
            usr_path==''

        #deal with the query dictionary
        if query==True and bool(query_args)==True:
            if 'archiveType' in query_args.keys():
                archiveType=query_args['archiveType']
                if type(archiveType) == str:
                    archiveType=lipdutils.pre_process_str(archiveType)
                    archiveType=[archiveType]
                else:
                    archiveType=lipdutils.pre_process_list(archiveType)
                availableType=lipdutils.whatArchives(print_response=False)
                availableTypeP=lipdutils.pre_process_list(availableType)
                res=[]
                for item in archiveType:
                    indices = [i for i, x in enumerate(availableTypeP) if x == item]
                if len(indices)!=0:
                    res.append(np.array(availableType)[indices].tolist())
                res=np.unique(np.array(res)).tolist()
                if len(res)==0:
                    archiveType = [ ]
                else:
                    archiveType=res
            else:
                archiveType = [ ]

            if 'proxyObsType' in query_args.keys():
                proxyObsType=query_args['proxyObsType']
                if type(proxyObsType) == str:
                    proxyObsType=lipdutils.pre_process_str(proxyObsType)
                    proxyObsType=[proxyObsType]
                else:
                    proxyObsType=lipdutils.pre_process_list(proxyObsType)
                availableProxy=lipdutils.whatProxyObservations(print_response=False)
                availableProxyP=lipdutils.pre_process_list(availableProxy)
                res=[]
                for item in proxyObsType:
                    indices = [i for i, x in enumerate(availableProxyP) if x == item]
                if len(indices)!=0:
                    res.append(np.array(availableProxy)[indices].tolist())
                res=np.unique(np.array(res)).tolist()
                if len(res)==0:
                    proxyObsType = [ ]
                else:
                    proxyObsType=res
            else:
                proxyObsType=[ ]

            if 'infVarType' in query_args.keys():
                infVarType=query_args['infVarType']
            else:
                infVarType=[ ]
            if 'sensorGenus' in query_args.keys():
                sensorGenus = query_args['sensorGenus']
            else:
                sensorGenus = [ ]
            if 'sensorSpecies' in query_args.keys():
                sensorSpecies = query_args['sensorSpecies']
            else:
                sensorSpecies=[ ]
            if 'interpName' in query_args.keys():
                interpName = query_args['interpName']
            else:
                interpName=[ ]
            if 'interpDetail' in query_args.keys():
                interpDetail = query_args['interpDetail']
            else:
                interpDetail = [ ]
            if 'ageUnits' in query_args.keys():
                ageUnits = query_args['ageUnits']
            else:
                ageUnits = [ ]
            if 'ageBound' in query_args.keys():
                ageBound = query_args['ageBound']
            else:
                ageBound=[ ]
            if 'ageBoundType' in query_args.keys():
                ageBoundType = query_args['ageBoundType']
            else:
                ageBoundType = [ ]
            if 'recordLength' in query_args.keys():
                recordLength = query_args['recordLength']
            else:
                recordLength = [ ]
            if 'resolution' in query_args.keys():
                resolution = query_args['resolution']
            else:
                resolution = [ ]
            if 'lat' in query_args.keys():
                lat = query_args['lat']
            else:
                lat = [ ]
            if 'lon' in query_args.keys():
                lon = query_args['lon']
            else:
                lon = [ ]
            if 'alt' in query_args.keys():
                alt = query_args['alt']
            else:
                alt= [ ]
            if 'download_folder' in query_args.keys():
                download_folder = query_args['download_folder']
            else:
                download_folder=os.getcwd()+'/'

            lipdutils.queryLinkedEarth(archiveType=archiveType, proxyObsType=proxyObsType, infVarType = infVarType, sensorGenus=sensorGenus,
                    sensorSpecies=sensorSpecies, interpName =interpName, interpDetail =interpDetail, ageUnits = ageUnits,
                    ageBound = ageBound, ageBoundType = ageBoundType, recordLength = recordLength, resolution = resolution,
                    lat = lat, lon = lon, alt = alt, print_response = False, download_lipd = True,
                    download_folder = download_folder)

            D_query = lpd.readLipd(download_folder)
            if 'archiveType' in D_query.keys():
                D_query={D_query['dataSetName']:D_query}
        else:
            D_query={}
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
        else:
            D_path={}
        if lipd_dict!=None:
            D_dict=lipd_dict
            if 'archiveType' in D_dict.keys():
                D_dict={D_dict['dataSetName']:D_dict}
        else:
            D_dict={}

        #assemble
        self.lipd={}
        self.lipd.update(D_query)
        self.lipd.update(D_path)
        self.lipd.update(D_dict)

    def __repr__(self):
        return str(self.__dict__)

    def copy(self):
        '''Copy the object
        '''
        return deepcopy(self)

    def to_tso(self):
        '''

        Returns
        -------
        ts_list : list
            List of Lipd timeseries objects

        '''
        ts_list=lpd.extractTs(self.__dict__['lipd'])
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
        savefig_settings : TYPE, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax. The default is False.

        Returns
        -------
        res : figure
            The figure

        '''
        #get the information from the LiPD dict
        lat=[]
        lon=[]
        archiveType=[]

        for idx, key in enumerate(self.lipd):
            d = self.lipd[key]
            lat.append(d['geo']['geometry']['coordinates'][1])
            lon.append(d['geo']['geometry']['coordinates'][0])
            archiveType.append(lipdutils.LipdToOntology(d['archiveType']).lower())

        # make sure criteria is in the plot_default list
        for idx,val in enumerate(archiveType):
            if val not in self.plot_default.keys():
                archiveType[idx] = 'other'

        if markersize is not None:
            scatter_kwargs.update({'markersize': markersize})

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

    #def mapNearRecord():

        #res={}

        #return res

class LipdSeries(Series):
    '''Lipd time series object
    '''
    def __init__(self, tso):
        if type(tso) is list:
            self.lipd_ts=lipdutils.getTs(tso)
        else:
            self.lipd_ts=tso

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

        value=np.array(self.lipd_ts['paleoData_values'],dtype='float64')
        #Remove NaNs
        ys_tmp=np.copy(value)
        value=value[~np.isnan(ys_tmp)]
        time=time[~np.isnan(ys_tmp)]
        value_name=self.lipd_ts['paleoData_variableName']
        if 'paleoData_units' in self.lipd_ts.keys():
            value_unit=self.lipd_ts['paleoData_units']
        else:
            value_unit=None
        label=self.lipd_ts['dataSetName']
        super(LipdSeries,self).__init__(time=time,value=value,time_name=time_name,
             time_unit=time_unit,value_name=value_name,value_unit=value_unit,
             label=label)

    def copy(self):
        '''Copy the object
        '''
        return deepcopy(self)

    def chronEnsembleToPaleo(self,D,modelNumber=None,tableNumber=None):
        '''Fetch chron ensembles from a lipd object and return the ensemble as MultipleSeries

        Parameters
        ----------
        D : a LiPD object
        modelNumber : int, optional
            Age model number. The default is None.
        tableNumber : int, optional
            Table Number. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        ms : pyleoclim.MultipleSeries
            A MultipleSeries object with each series representing a possible realization of the age model

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
        csv_dict=lpd.getCsv(lipd)
        chron,paleo = lipdutils.isEnsemble(csv_dict)
        if len(chron)==0:
            raise ValueError("No ChronMeasurementTables available")
        elif len(chron)>1:
            if modelNumber==None or tableNumber==None:
                csvName=lipdutils.whichEnsemble(chron)
            else:
                str1='model'+str(modelNumber)
                str2='ensemble'+str(tableNumber)
                for item in chron:
                    if str1 in item and str2 in item:
                        csvName=item
            depth, ensembleValues =lipdutils.getEnsemble(csv_dict,csvName)
        else:
            depth, ensembleValues =lipdutils.getEnsemble(csv_dict,chron[0])
        #make sure it's sorted
        sort_ind = np.argsort(depth)
        depth=list(np.array(depth)[sort_ind])
        ensembleValues=ensembleValues[sort_ind,:]
        #Map to paleovalues
        key=[]
        for item in self.lipd_ts.keys():
            if 'depth' in item and 'Units' not in item:
                key.append(item)
        key=key[0]
        ds= np.array(self.lipd_ts[key],dtype='float64')
        ys= np.array(self.lipd_ts['paleoData_values'],dtype='float64')
        #Remove NaNs
        ys_tmp=np.copy(ys)
        ds=ds[~np.isnan(ys_tmp)]
        ensembleValuestoPaleo=lipdutils.mapAgeEnsembleToPaleoData(ensembleValues, depth, ds)
        #create multipleseries
        s_list=[]
        for s in ensembleValuestoPaleo.T:
            s_tmp=Series(time=s,value=self.value)
            s_list.append(s_tmp)

        ms = MultipleSeries(series_list=s_list)

        return ms

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
        savefig_settings : TYPE, optional
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.
        mute : bool, optional
            if True, the plot will not show;
            recommend to turn on when more modifications are going to be made on ax. The default is False.

        Returns
        -------
        res : fig
        '''
        #get the information from the timeseries
        lat=[self.lipd_ts['geo_meanLat']]
        lon=[self.lipd_ts['geo_meanLon']]
        archiveType=lipdutils.LipdToOntology(self.lipd_ts['archiveType'])

        # make sure criteria is in the plot_default list
        if archiveType not in self.plot_default.keys():
            archiveType = 'other'

        if markersize is not None:
            scatter_kwargs.update({'markersize': markersize})

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
