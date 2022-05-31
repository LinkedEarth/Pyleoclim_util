'''
This class is meant to hold the output of the Singular Spectrum Analysis (SSA) method,
which applies to Series objets. Two functions are enabled by this class:
* `screeplot`, which plots the eigenvalue spectrum to help determine what modes to keep
* `modeplot`, which plots the individual mode temporal EOF and temporal PC
'''

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import MaxNLocator

from ..core import series
from ..utils import plotting


class SsaRes:
    '''This class is meant to hold the output of the Singular Spectrum Analysis (SSA) method,
    which applies to Series objets. Two functions are enabled by this class:
        
    * `screeplot`, which plots the eigenvalue spectrum to help determine what modes to keep
    
    * `modeplot`, which plots the individual mode temporal EOF and temporal PC

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
             clr_eig='black'):
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

        xlim : list, optional
            x-axis limits. The default is None.

        clr_mcssa : str, optional
            color of the Monte Carlo SSA AR(1) shading (if data are provided)
            default: red

        clr_eig : str, optional
            color of the eigenvalues, default: black

        clr_signif : str, optional
            color of the highlights for significant eigenvalue. (default: teal)
       
        See also
        --------
        
        pyleoclim.core.series.Series.ssa : Singular Spectrum Analysis for timeseries objects
        
        pyleoclim.core.utils.decomposition.ssa : Singular Spectrum Analysis utility
        
        pyleoclim.core.ssares.SsaRes.modeplot : plot SSA modes
        
        
        Examples
        --------

        Plot the SSA eig envalue spectrum of the Southern Oscillation Index:

        .. ipython:: python
            :okwarning:
            :okexcept:

            data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            ts = pyleo.Series(time=data.iloc[:,1], value=data.iloc[:,2], time_name='Year C.E', value_name='SOI', label='SOI')
            ssa = ts.ssa()
            
            @savefig ssa_screeplot.png
            fig, ax = ssa.screeplot()
            pyleo.closefig(fig)

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

        return fig, ax

    def modeplot(self, index=0, figsize=[10, 5], ax=None, savefig_settings=None,
             title_kwargs=None, spec_method = 'mtm', plot_original=False):
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

        spec_method: str, optional
            The name of the spectral method to be applied on the PC. Default: MTM
            Note that the data are evenly-spaced, so any spectral method that
            assumes even spacing is applicable here:  'mtm', 'welch', 'periodogram'
            'wwz' is relevant too if scaling exponents need to be estimated.
          
        See also
        --------
        
        pyleoclim.core.series.Series.ssa : Singular Spectrum Analysis for timeseries objects
        
        pyleoclim.core.utils.decomposition.ssa : Singular Spectrum Analysis utility
        
        pyleoclim.core.ssares.SsaRes.screeplot : plot SSA eigenvalue spectrum
         
        Examples
        --------

        Plot the first SSA mode of the Southern Oscillation Index:

        .. ipython:: python
            :okwarning:
            :okexcept:

            data = pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
            ts = pyleo.Series(time=data.iloc[:,1], value=data.iloc[:,2], time_name='Year C.E', value_name='SOI', label='SOI')
            ssa = ts.ssa()
            
            @savefig ssa_modeplot1.png
            fig, ax = ssa.modeplot()
            pyleo.closefig(fig)
            
        Plot the second mode (note 0-based indexing):
            
         .. ipython:: python
             :okwarning:
             :okexcept:

             @savefig ssa_modeplot2.png
             fig, ax = ssa.modeplot(index=1)
             pyleo.closefig(fig)

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
        ax.set_xlabel('Time'),  ax.set_ylabel(r'$RC_'+str(index+1)+'$')
        ax.set_title('SSA Mode '+str(index+1)+' RC, '+ '{:3.2f}'.format(self.pctvar[index]) + '% variance explained',weight='bold')
        # plot T-EOF
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(self.eigvecs[:,index])
        ax.set_title('Analyzing function')
        ax.set_xlabel('Time'), ax.set_ylabel('T-EOF')
        # plot spectrum
        ax = fig.add_subplot(gs[1, 1])
        ts_rc = series.Series(time=self.time, value=RC) # define timeseries object for the RC
        psd_mtm_rc = ts_rc.interp().spectral(method=spec_method)
        _ = psd_mtm_rc.plot(ax=ax)
        ax.set_xlabel('Period')
        ax.set_title('RC Spectrum ('+spec_method+')')

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)

        return fig, ax