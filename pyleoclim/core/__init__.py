#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the various Classes and in which Modules they may be found.
"""

from .series import Series
from .geoseries import GeoSeries
from .psds import PSD, MultiplePSD
from .multipleseries import MultipleSeries
from .multiplegeoseries import MultipleGeoSeries
from .surrogateseries import SurrogateSeries
from .ensembleseries import EnsembleSeries
from .scalograms import Scalogram, MultipleScalogram
from .coherence import Coherence
from .corr import Corr
from .correns import CorrEns
from .multivardecomp import MultivariateDecomp
from .ssares import SsaRes
from .resolutions import Resolution, MultipleResolution
from .ensemblegeoseries import EnsembleGeoSeries
from .mulensgeoseries import MulEnsGeoSeries
from .ensmultivardecomp import EnsMultivarDecomp
