import pandas as pd
import numpy as np
import pyleoclim as pyleo
import pytest 


@pytest.fixture
def dataframe_dt():
    """Pandas dataframe with a datetime index and random values"""
    length = 5
    dti = pd.date_range("2018-01-01", periods=length, freq="Y", unit='s')
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
        'archiveType': 'Instrumental',
        'importedFrom': None,
        'log': (
                {0: 'dropna', 'applied': True, 'verbose': True},
                {1: 'sort_ts', 'direction': 'ascending'}
            )
    }

@pytest.fixture
def gen_ts():
    """ Generate realistic-ish Series for testing """
    t,v = pyleo.utils.gen_ts(model='colored_noise',nt=50)
    ts = pyleo.Series(t,v, verbose=False)
    return ts

@pytest.fixture
def unevenly_spaced_series():
    """Pyleoclim series with unevenly spaced time axis"""
    length = 10
    t = np.linspace(1,length,length) ** 2
    v = np.ones(length)
    series = pyleo.Series(time=t, value =v, verbose=False)
    return series

@pytest.fixture
def unevenly_spaced_series_nans():
    """Pyleoclim series with unevenly spaced time axis"""
    length = 10
    t = np.linspace(1,length,length) ** 2
    v = np.ones(length)
    v[2:4] = np.nan
    series = pyleo.Series(time=t, value =v, dropna=False, verbose=False)
    return series

@pytest.fixture
def evenly_spaced_series():
    """Pyleoclim series with evenly spaced time axis"""
    length = 10
    t = np.linspace(1,length,length)
    v = np.cos(2*np.pi*t/10)
    series = pyleo.Series(time=t, value=v, verbose=False)
    series.label = 'cosine'
    return series

@pytest.fixture
def pinkseries():
    """Pyleoclim geoseries with 1/f spectrum """
    t,v = pyleo.utils.gen_ts(model='colored_noise',alpha=1.0, nt=100, seed=251)
    ts = pyleo.Series(t,v, sort_ts='none', verbose=False)
    ts.label = 'pink noise'
    return ts

# Geoseries fixtures
@pytest.fixture
def geometadata():
    return {'lat': -75.1011,
            'lon': 123.3478,
            'elevation': 3233,
            'time_unit': 'ky BP',
            'time_name': 'Age',
            'value_unit': '‰',
            'value_name': '$\\delta \\mathrm{D}$',
            'label': 'pink noise geoseries',
            'archiveType': 'GlacierIce',
            'importedFrom': 'who knows where',
            'log': None
    }

@pytest.fixture
def pinkgeoseries(geometadata):
    """Pyleoclim geoseries based on 1/f (pink) temporal structure"""
    t,v = pyleo.utils.gen_ts(model='colored_noise',alpha=1.0, nt=200, seed=251)
    ts = pyleo.GeoSeries(t,v, verbose=False, **geometadata).standardize()
    return ts

@pytest.fixture
def multipleseries_basic():
    ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]))
    ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]))
    ms = pyleo.MultipleSeries([ts1, ts2])
    return ms

@pytest.fixture
def multipleseries_science():
    soi = pyleo.utils.load_dataset('SOI')
    nino = pyleo.utils.load_dataset('NINO3')
    ms = soi & nino
    ms.name = 'ENSO'
    return ms

