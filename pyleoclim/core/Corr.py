#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corr objects are used to store the results of the correlation method between Series
and/or MultipleSeries. Its main purpose is to allow a print function to display the output.
"""

import numpy as np
from tabulate import tabulate

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

class Corr:
    ''' The object for correlation results in order to format the print message

    Parameters
    ----------

    r: float
        the correlation coefficient

    p: float
        the p-value

    p_fmt_td: float
        the threshold for p-value formatting (0.01 by default, i.e., if p<0.01, will print "< 0.01" instead of "0")

    p_fmt_style: str
        the style for p-value formatting (exponential notation by default)

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
