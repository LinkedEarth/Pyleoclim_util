# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:42:34 2016

@author: deborahkhider

License agreement - GNU GENERAL PUBLIC LICENSE v3
https://github.com/LinkedEarth/Pyleoclim_util/blob/master/license

"""
#Import pyleoclim modules

from .utils import *
from .core import *

from .utils import plotting
plotting.set_style(style='journal', font_scale=1.4)

# get the version
from importlib.metadata import version
__version__ = version('pyleoclim')
