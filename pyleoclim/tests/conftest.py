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
    ts = pyleo.Series(t,v, verbose=False, auto_time_params=True)
    return ts

@pytest.fixture
def unevenly_spaced_series():
    """Pyleoclim series with unevenly spaced time axis"""
    length = 10
    t = np.linspace(1,length,length) ** 2
    v = np.ones(length)
    series = pyleo.Series(time=t, value =v, verbose=False, auto_time_params=True)
    return series

@pytest.fixture
def unevenly_spaced_series_nans():
    """Pyleoclim series with unevenly spaced time axis"""
    length = 10
    t = np.linspace(1,length,length) ** 2
    v = np.ones(length)
    v[2:4] = np.nan
    series = pyleo.Series(time=t, value =v, dropna=False, verbose=False, auto_time_params=True)
    return series

@pytest.fixture
def evenly_spaced_series():
    """Pyleoclim series with evenly spaced time axis"""
    length = 10
    t = np.linspace(1,length,length)
    v = np.cos(2*np.pi*t/10)
    series = pyleo.Series(time=t, value=v, verbose=False, auto_time_params=True)
    series.label = 'cosine'
    return series

@pytest.fixture
def pinkseries():
    """Pyleoclim geoseries with 1/f spectrum """
    t = np.arange(100)
    v = pyleo.utils.colored_noise(alpha=1.0, t=t, seed=251, std=2.5)
    ts = pyleo.Series(t,v, verbose=False, auto_time_params=True)
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
def multipleseries_nans():
    t1 = np.arange(1,10)
    v1 = np.ones(len(t1))
    ts1 = pyleo.Series(time=t1, value=v1)
    
    t2 = np.arange(1,10)
    v2 = np.ones(len(t1))
    v2[2:4] = np.nan
    ts2 = pyleo.Series(time=t2, value =v2, dropna=False, verbose=False)
    ms = pyleo.MultipleSeries([ts1, ts2])
    return ms

@pytest.fixture
def multipleseries_science():
    soi = pyleo.utils.load_dataset('SOI')
    nino = pyleo.utils.load_dataset('NINO3')
    ms = soi & nino
    ms.name = 'ENSO'
    return ms

@pytest.fixture
def ensembleseries_basic():
    ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]))
    ts2 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 8, 1]))
    ens = pyleo.EnsembleSeries([ts1, ts2])
    return ens

@pytest.fixture
def ensembleseries_nans():
    t1 = np.arange(1,10)
    v1 = np.ones(len(t1))
    ts1 = pyleo.Series(time=t1, value=v1)
    
    t2 = np.arange(1,10)
    v2 = np.ones(len(t1))
    v2[2:4] = np.nan
    ts2 = pyleo.Series(time=t2, value =v2, dropna=False, verbose=False)
    ens = pyleo.EnsembleSeries([ts1, ts2])
    return ens

@pytest.fixture
def ensembleseries_science():
    soi = pyleo.utils.load_dataset('SOI')
    ens = pyleo.EnsembleSeries([soi for _ in range(5)])
    for series in ens.series_list:
        series.value += np.random.randn(len(series))
    return ens

@pytest.fixture
def ensemblegeoseries_basic():
    time = np.arange(50)
    ts1 = pyleo.GeoSeries(time=time, value=np.random.randn(len(time)),lat=0,lon=0)
    ts2 = pyleo.GeoSeries(time=time, value=np.random.randn(len(time)),lat=0,lon=0)
    ens = pyleo.EnsembleGeoSeries([ts1, ts2])
    return ens

@pytest.fixture
def ensemblegeoseries_nans():
    t1 = np.arange(50)
    v1 = np.random.randn(len(t1))
    ts1 = pyleo.GeoSeries(time=t1, value=v1,lat=0,lon=0)
    
    t2 = np.arange(50)
    v2 = np.random.randn(len(t2))
    v2[2:4] = np.nan
    ts2 = pyleo.GeoSeries(time=t2, value =v2, dropna=False, verbose=False,lat=0,lon=0)
    ens = pyleo.EnsembleGeoSeries([ts1, ts2])
    return ens