
import pytest

import numpy as np

from pyleoclim.utils import tsutils, tsbase
from numpy.testing import assert_array_equal

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

def test_bin_t2(unevenly_spaced_series):
    bins = np.arange(0,100,10)
    res_dict = tsutils.bin(unevenly_spaced_series.time,unevenly_spaced_series.value,bins=bins)
    t = res_dict['bins']
    assert_array_equal(t,(bins[1:]+bins[:-1])/2)

@pytest.mark.parametrize('statistic',['mean','std','median','count','sum','min','max'])
def test_bin_t3(unevenly_spaced_series,statistic):
    res_dict = tsutils.bin(unevenly_spaced_series.time,unevenly_spaced_series.value,statistic=statistic)

def test_gkernel_t0(unevenly_spaced_series):
    t,v = tsutils.gkernel(unevenly_spaced_series.time,unevenly_spaced_series.value)
    assert isinstance(t,np.ndarray)
    assert isinstance(v,np.ndarray)

def test_gkernel_t1(unevenly_spaced_series):
    t,v = tsutils.gkernel(unevenly_spaced_series.time,unevenly_spaced_series.value)
    assert tsbase.is_evenly_spaced(t)

def test_gkernel_t2(unevenly_spaced_series):
    bins = np.arange(0,100,10)
    t,v = tsutils.gkernel(unevenly_spaced_series.time,unevenly_spaced_series.value,bins=bins)
    assert_array_equal(t,(bins[1:]+bins[:-1])/2)