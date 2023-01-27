import pandas as pd
import numpy as np
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
def metadata():
    return {'time_unit': 'years CE',
        'time_name': 'Time',
        'value_unit': 'mb',
        'value_name': 'SOI',
        'label': 'Southern Oscillation Index',
        'lat': None,
        'lon': None,
        'archiveType': None,
        'importedFrom': None,
        'log': ({0: 'clean_ts', 'applied': True, 'verbose': False},)
    }