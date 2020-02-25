#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 05:40:16 2020

@author: deborahkhider

Contains all basic block functions for pyleoclim. 
Defaults set as per the original function
"""

from .correlation import corr_sig, fdr

__all__ = [
        'corr_sig',
        'fdr']