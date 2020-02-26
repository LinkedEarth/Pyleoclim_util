#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 05:40:16 2020

@author: deborahkhider

Contains all basic block functions for pyleoclim. 
Defaults set as per the original function
"""

from .plotting import set_style
from .plotting import showfig
from .spectral import *

__all__ = [
    'set_style',
    'showfig',
]
