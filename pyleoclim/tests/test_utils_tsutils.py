
import pytest
from pyleoclim.utils import tsutils, tsbase
import numpy as np


def test_bin_t0(unevenly_spaced_series):
    res_dict = tsutils.bin(unevenly_spaced_series.time,unevenly_spaced_series.value)
    t = res_dict['bins']
    v = res_dict['binned_values']
    assert isinstance(t,np.ndarray)
    assert isinstance(v,np.ndarray)

def test_bin_t1(unevenly_spaced_series):
    res_dict = tsutils.bin(unevenly_spaced_series.time,unevenly_spaced_series.value,evenly_spaced=True)
    t = res_dict['bins']
    assert tsbase.is_evenly_spaced(t)