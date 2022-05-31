#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CorrEns objects store the result of an ensemble correlation calculation between timeseries and/or ensemble of timeseries.
The class enables a print and plot function to easily visualize the result. 
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, transforms as transforms
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate

from ..utils import plotting

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

class CorrEns:
    ''' CorrEns objects store the result of an ensemble correlation calculation 
    between timeseries and/or ensemble of timeseries. The class enables a print and 
    plot function to easily visualize the result. 

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
    
    pyleoclim.utils.correlation.fdr : FDR (False Discovery Rate) function
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

    def plot(self, figsize=[4, 4], title=None, ax=None, savefig_settings=None, hist_kwargs=None, title_kwargs=None,
             xlim=None,
             clr_insignif=sns.xkcd_rgb['grey'], clr_signif=sns.xkcd_rgb['teal'],
             clr_signif_fdr=sns.xkcd_rgb['pale orange'],
             clr_percentile=sns.xkcd_rgb['salmon'], rwidth=0.8, bins=None, vrange=None):
        ''' Plot the distribution of correlation values as a histogram
        
        Color-coding is used to indicate significance, with or without applying 
        the False Discovery Rate (FDR) method. 

        Parameters
        ----------
        figsize : list, optional
        
            The figure size. The default is [4, 4].

        title : str, optional
        
            Plot title. The default is None.

        savefig_settings : dict
        
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existing or new path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        hist_kwargs : dict
        
            the keyword arguments for ax.hist()

        title_kwargs : dict
        
            the keyword arguments for ax.set_title()

        ax : matplotlib.axis, optional
        
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        xlim : list, optional
        
            x-axis limits. The default is None.

        See also
        --------
        
        pyleoclim.core.series.Series.correlation: correlation with significance
        
        pyleoclim.utils.correlation.fdr: False Discovery Rate

        matplotlib.pyplot.hist: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.hist.html
        
        pyleoclim.utils.plotting.savefig : save figures in Pyleoclim
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
        ax.legend([f'p ≥ {self.alpha}', f'p < {self.alpha} (w/o FDR)', f'p < {self.alpha} (w/ FDR)'], loc='upper left',
                  bbox_to_anchor=(1.1, 1), ncol=1)

        frac_signif = np.size(r_signif) / np.size(self.r)
        frac_signif_fdr = np.size(r_signif_fdr) / np.size(self.r)
        ax.text(x=1.1, y=0.5, s=f'Fraction significant: {frac_signif * 100:.1f}%', transform=ax.transAxes, fontsize=10,
                color=clr_signif)
        ax.text(x=1.1, y=0.4, s=f'Fraction significant: {frac_signif_fdr * 100:.1f}%', transform=ax.transAxes,
                fontsize=10, color=clr_signif_fdr)

        r_pcts = np.percentile(self.r, [2.5, 25, 50, 75, 97.5])
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for r_pct, pt, ls in zip(r_pcts, np.array([2.5, 25, 50, 75, 97.5]) / 100, [':', '--', '-', '--', ':']):
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
            return fig, ax
        else:
            return ax

