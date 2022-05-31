#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the various Classes and in which Modules they may be found.
"""

from .series import Series
from .psds import PSD, MultiplePSD
from .multipleseries import MultipleSeries
from .surrogateseries import SurrogateSeries
from .ensembleseries import EnsembleSeries
from .scalograms import Scalogram, MultipleScalogram
from .coherence import Coherence
#from .multiplepsd import MultiplePSD
#from .MultipleScalogram import MultipleScalogram
from .corr import Corr
from .correns import CorrEns
from .spatialdecomp import SpatialDecomp
from .ssares import SsaRes
from .lipd import Lipd
from .lipdseries import LipdSeries
