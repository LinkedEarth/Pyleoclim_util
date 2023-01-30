import pandas as pd
import numpy as np
import pyleoclim as pyleo
import pytest 


@pytest.fixture
def dataframe_dt():
    """Pandas dataframe with a datetime index and random values"""
    length = 5
    dti = pd.date_range("2018-01-01", periods=length, freq="Y")
    df = pd.DataFrame(np.array(range(length)), index=dti)
    return df

@pytest.fixture
def dataframe():
    """Pandas dataframe with a non-datetime index and random values"""
    length = 5
    df = pd.DataFrame(np.ones(length))
    return df

@pytest.fixture
def unevenly_spaced_series()
    """Pyleoclim series with unevenly spaced time axis"""
    length = 10
    t = np.linspace(1,length,length) ** 2
    v = np.ones(length)
    series = pyleo.Series(t,v)
    return series