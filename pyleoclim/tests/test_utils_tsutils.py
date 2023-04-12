
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
    res_dict = tsutils.bin(unevenly_spaced_series.time,unevenly_spaced_series.value,no_nans=True)
    t = res_dict['bins']
    assert tsbase.is_evenly_spaced(t)

def test_bin_t2(unevenly_spaced_series):
    bin_edges = np.arange(0,100,10)
    res_dict = tsutils.bin(unevenly_spaced_series.time,unevenly_spaced_series.value,bin_edges=bin_edges)
    t = res_dict['bins']
    assert_array_equal(t,(bin_edges[1:]+bin_edges[:-1])/2)

def test_bin_t3(unevenly_spaced_series):
    time_axis = np.arange(0,100,10)
    res_dict = tsutils.bin(unevenly_spaced_series.time,unevenly_spaced_series.value,time_axis=time_axis)
    t = res_dict['bins']
    assert_array_equal(time_axis,t)

@pytest.mark.parametrize('statistic',['mean','std','median','count','sum','min','max'])
def test_bin_t4(unevenly_spaced_series,statistic):
    tsutils.bin(unevenly_spaced_series.time,unevenly_spaced_series.value,statistic=statistic)

@pytest.mark.parametrize('start',[None,10])
@pytest.mark.parametrize('stop',[None,90])
@pytest.mark.parametrize('bin_size',[None,20])
@pytest.mark.parametrize('step_style',[None,'median'])
@pytest.mark.parametrize('no_nans',[False,True])
def test_bin_t5(unevenly_spaced_series,start,stop,bin_size,step_style,no_nans):
    tsutils.bin(unevenly_spaced_series.time,unevenly_spaced_series.value,start=start,stop=stop,bin_size=bin_size,step_style=step_style,no_nans=no_nans)

def test_gkernel_t0(unevenly_spaced_series):
    t,v = tsutils.gkernel(unevenly_spaced_series.time,unevenly_spaced_series.value)
    assert isinstance(t,np.ndarray)
    assert isinstance(v,np.ndarray)

def test_gkernel_t1(unevenly_spaced_series):
    t,v = tsutils.gkernel(unevenly_spaced_series.time,unevenly_spaced_series.value)
    assert tsbase.is_evenly_spaced(t)

def test_gkernel_t2(unevenly_spaced_series):
    bin_edges = np.arange(0,100,10)
    t,v = tsutils.gkernel(unevenly_spaced_series.time,unevenly_spaced_series.value,bin_edges=bin_edges)
    assert_array_equal(t,(bin_edges[1:]+bin_edges[:-1])/2)

def test_interp_t0(unevenly_spaced_series):
    t,v = tsutils.interp(unevenly_spaced_series.time,unevenly_spaced_series.value)
    assert isinstance(t,np.ndarray)
    assert isinstance(v,np.ndarray)

def test_interp_t1(unevenly_spaced_series):
    time_axis = np.arange(1,100,10)
    t,v = tsutils.interp(unevenly_spaced_series.time,unevenly_spaced_series.value,time_axis=time_axis)
    assert_array_equal(time_axis,t)

@pytest.mark.parametrize('start',[None,10])
@pytest.mark.parametrize('stop',[None,90])
@pytest.mark.parametrize('step',[None,20])
@pytest.mark.parametrize('step_style',[None,'median'])
def test_interp_t2(unevenly_spaced_series,start,stop,step,step_style):
    tsutils.interp(unevenly_spaced_series.time,unevenly_spaced_series.value,start=start,stop=stop,step=step,step_style=step_style)