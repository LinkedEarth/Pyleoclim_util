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
def unevenly_spaced_series():
    """Pyleoclim series with unevenly spaced time axis"""
    length = 10
    t = np.linspace(1,length,length) ** 2
    v = np.ones(length)
    series = pyleo.Series(time=t, value =v, verbose=False)
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
            'archiveType': 'Glacier Ice',
            'importedFrom': 'who knows where',
            'log': None
    }

@pytest.fixture
def pinkgeoseries(geometadata):
    """Pyleoclim geoseries based on 1/f (pink) noise"""
    t,v = pyleo.utils.gen_ts(model='colored_noise',alpha=1.0, nt=200, seed=251)
    ts = pyleo.GeoSeries(t,v, verbose=False, **geometadata).standardize()
    return ts

@pytest.fixture
def multiple_pinkgeoseries():
    """Pyleoclim geoseries with """
    nrecs = 10
    seed = 101
    nt = 200
    lats = np.random.default_rng(seed=seed).uniform(30.0,60.0,nrecs)
    lons = np.random.default_rng(seed=seed).uniform(-20.0,60.0,nrecs)
    elevs = np.random.default_rng(seed=seed).uniform(0,4000,nrecs)
    
    archives = np.random.default_rng(seed=seed).choice(list(pyleo.utils.PLOT_DEFAULT.keys()),size=nrecs)
    obsTypes = np.random.default_rng(seed=seed).choice(['MXD', 'd18O', 'Sr/Ca'],size=nrecs)
    
    
    ts_list = []
    for i in range(nrecs):
        t,v = pyleo.utils.gen_ts(model='colored_noise',alpha=1.0, nt=nt)
        ts = pyleo.GeoSeries(t,v, verbose=False, label = f'pink series {i}',
                             archiveType=archives[i], observationType=obsTypes[i],
                             lat=lats[i], lon = lons[i], elevation=elevs[i]).standardize()
        ts_list.append(ts)
        
    return pyleo.MultipleGeoSeries(ts_list, label='Multiple Pink GeoSeries')