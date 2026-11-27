"""
Helpers for how pyleoclim objects display themselves (``__repr__``, ``__str__`` and Jupyter's ``_repr_html_``).

These helpers return text; they never print.
"""

import html

import numpy as np
import pandas as pd
from tabulate import tabulate

__all__ = []  # internal module: not exported by ``pyleoclim.utils``

#: default number of rows shown by truncated tables (half at the top, half at the bottom)
MAX_ROWS = 10


def text_table(table, max_rows=MAX_ROWS):
    ''' Format a dict of equal-length columns with tabulate, keeping only the first and last rows if it is long

    Parameters
    ----------
    table : dict
        column name -> sequence of values
    max_rows : int
        number of rows kept when the table is truncated (half at the top, half at the bottom)

    Returns
    -------
    str
    '''
    columns = {key: list(val) for key, val in table.items()}
    nrows = len(next(iter(columns.values()))) if columns else 0
    if nrows > max_rows:
        head, tail = max_rows // 2, max_rows - max_rows // 2
        # the gap row is None, shown as '...' by tabulate, so number columns stay numeric (and right-aligned)
        columns = {key: val[:head] + [None] + val[-tail:] for key, val in columns.items()}
    out = tabulate(columns, headers='keys', missingval='...')
    if nrows > max_rows:
        out += f'\n[{nrows} rows]'
    return out


def html_table(table, max_rows=MAX_ROWS):
    ''' HTML version of ``text_table``, via pandas (which truncates long tables the same way)
    '''
    return pd.DataFrame(table).to_html(max_rows=max_rows, index=False, notebook=True)


def html_header(obj, label=None, details=None):
    ''' Bold class name, optional label and details in parentheses, e.g. "pyleoclim.PSD: SOI (mtm, 959 frequencies)"
    '''
    out = f'<b>pyleoclim.{type(obj).__name__}</b>'
    if label is not None:
        out += f': {html.escape(str(label))}'
    if details:
        out += f' ({html.escape(details)})'
    return f'<p>{out}</p>'


def text_header(obj, label=None, details=None):
    ''' Plain-text version of ``html_header``
    '''
    out = f'pyleoclim.{type(obj).__name__}'
    if label is not None:
        out += f': {label}'
    if details:
        out += f' ({details})'
    return out


def html_fields(fields):
    ''' Two-column HTML table of name/value pairs
    '''
    rows = ''.join(f'<tr><th style="text-align:left">{html.escape(str(k))}</th>'
                   f'<td style="text-align:left">{html.escape(str(v))}</td></tr>'
                   for k, v in fields.items())
    return f'<table>{rows}</table>'


def text_fields(fields):
    ''' Plain-text version of ``html_fields``
    '''
    width = max((len(str(k)) for k in fields), default=0)
    return '\n'.join(f'{str(k):<{width}}  {v}' for k, v in fields.items())


def value_range(x, unit=None):
    ''' "n values, min to max [unit]" for an array, ignoring NaNs
    '''
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    out = f'{np.size(x)} values'
    if finite.size > 0:
        out += f', {finite.min():.4g} to {finite.max():.4g}'
    if unit:
        out += f' [{unit}]'
    return out
