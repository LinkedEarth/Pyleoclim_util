#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 05:38:44 2020

@author: deborahkhider

Core concerns objects. The pyleoclim defaults (if different from original base
function should be set in these files)
"""

from .api_deprec import Series, Coherence, PSD, Scalogram

__all__=['Series',
         'Coherence',
         'PSD',
         'Scalogram']