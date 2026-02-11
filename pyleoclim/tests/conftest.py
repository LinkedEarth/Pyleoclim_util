#from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest

import pyleoclim as pyleo

# ==== Base fixtures for generating testing datasets ====


@pytest.fixture
def random_seed():
    """Provide consistent random seed for reproducible tests"""
    return 251


@pytest.fixture
def base_time(random_seed):
    """Generate basic time axis data"""
    return np.arange(100)


@pytest.fixture
def basic_metadata():
    """Basic default metadata for Series objects"""
    return {
        "time_unit": "years CE",
        "time_name": "Time",
        "value_unit": "unit",
        "value_name": "Value",
        "label": "Test Series",
    }


@pytest.fixture
def soi_metadata():
    return {
        "time_unit": "years CE",
        "time_name": "Time",
        "value_unit": "mb",
        "value_name": "SOI",
        "label": "Southern Oscillation Index",
        "archiveType": "Instrumental",
        "importedFrom": None,
        "log": (
            {0: "dropna", "applied": True, "verbose": True},
            {1: "sort_ts", "direction": "ascending"},
        ),
    }


@pytest.fixture
def geo_metadata():
    """Default metadata for GeoSeries objects"""
    return {
        "lat": -75.1011,
        "lon": 123.3478,
        "elevation": 3233,
        "time_unit": "ky BP",
        "time_name": "Age",
        "value_unit": "‰",
        "value_name": "$\\delta \\mathrm{D}$",
        "label": "pink noise geoseries",
        "archiveType": "GlacierIce",
        "importedFrom": "test data",
        "log": None,
    }


# ==== Pandas DataFrame fixtures ====


@pytest.fixture
def dataframe_dt():
    """Pandas dataframe with a datetime index and random values"""
    length = 5
    dti = pd.date_range("2018-01-01", periods=length, freq="YE", unit="s")
    df = pd.DataFrame(np.array(range(length)), index=dti)
    return df


@pytest.fixture
def dataframe():
    """Pandas dataframe with a non-datetime index and random values"""
    length = 5
    df = pd.DataFrame(np.ones(length))
    return df


# ==== Timeseries pattern generators ====


@pytest.fixture
def gen_ts():
    """Generate realistic-ish Series for testing using colored noise"""

    def _gen(model="colored_noise", nt=50, alpha=1, f0=None, m=None, random_seed=42):
        t, v = pyleo.utils.gen_ts(
            model=model, nt=nt, alpha=alpha, f0=f0, m=m, seed=random_seed
        )
        ts = pyleo.Series(t, v, verbose=False, auto_time_params=True)
        return ts

    return _gen


@pytest.fixture
def gen_colored_noise():
    """Generate timeseries with colored noise"""

    def _gen(alpha=1, nt=100, std=1.0, f0=None, m=None, random_seed=None):
        t, v = pyleo.utils.gen_ts(
            model="colored_noise",
            alpha=alpha,
            nt=nt,
            std=std,
            f0=f0,
            m=m,
            seed=random_seed,
        )
        ts = pyleo.Series(t, v, verbose=False, auto_time_params=True)
        return ts

    return _gen


@pytest.fixture
def gen_sine_wave():
    """Generate sine wave data"""

    def _gen(period=10, nt=100, amplitude=1, offset=0):
        t = np.linspace(1, nt, nt)
        v = amplitude * np.sin(2 * np.pi * t / period) + offset
        ts = pyleo.Series(t, v, verbose=False, auto_time_params=True)
        return ts

    return _gen


@pytest.fixture
def gen_normal():
    """Generate random data with a Gaussian distribution"""

    def _gen(loc=0, scale=1, nt=100):
        t = np.arange(nt)
        v = np.random.normal(loc=loc, scale=scale, size=nt)
        ts = pyleo.Series(t, v, verbose=False, auto_time_params=True)
        return ts

    return _gen


@pytest.fixture
def gen_evenly_spaced():
    """Generate evenly spaced timeseries data (using cosine by default)"""

    def _gen(length=10, pattern="cosine"):
        t = np.linspace(1, length, length)
        if pattern == "cosine":
            v = np.cos(2 * np.pi * t / length)
        elif pattern == "sine":
            v = np.sin(2 * np.pi * t / length)
        elif pattern == "linear":
            v = t.copy()
        elif pattern == "constant":
            v = np.ones(length)
        else:  # random
            v = np.random.randn(length)
        ts = pyleo.Series(t, v, verbose=False, auto_time_params=True)
        return ts

    return _gen


@pytest.fixture
def gen_unevenly_spaced():
    """Generate unevenly spaced timeseries data"""

    def _gen(length=10, pattern="squared"):
        if pattern == "squared":
            t = np.linspace(1, length, length) ** 2
            v = np.ones(length)
        elif pattern == "random":
            t = np.sort(np.random.rand(length) * length)
            v = np.ones(length)
        ts = pyleo.Series(t, v, verbose=False, auto_time_params=True)
        return ts

    return _gen


# ==== Base Series fixtures ====


@pytest.fixture
def create_series():
    """Factory fixture to create a generic Series object with various options"""

    def _create(time, value, with_nans=False, dropna=False, **kwargs):
        if with_nans:
            # Add NaNs at fixed positions for reproducibility
            mask = np.zeros_like(value, dtype=bool)
            nan_indices = [2, 4] if len(value) > 5 else [0]
            mask[nan_indices] = True
            value = np.where(mask, np.nan, value)

        return pyleo.Series(
            time=time,
            value=value,
            dropna=dropna,
            verbose=False,
            auto_time_params=True,
            **kwargs,
        )

    return _create


@pytest.fixture
def create_geoseries():
    """Factory fixture to create a GeoSeries object with various options"""

    def _create(
        time, value, lat=0, lon=0, elevation=0, with_nans=False, dropna=False, **kwargs
    ):
        if with_nans:
            # Add NaNs at fixed positions for reproducibility
            mask = np.zeros_like(value, dtype=bool)
            nan_indices = [2, 4] if len(value) > 5 else [0]
            mask[nan_indices] = True
            value = np.where(mask, np.nan, value)

        return pyleo.GeoSeries(
            time=time,
            value=value,
            lat=lat,
            lon=lon,
            elevation=elevation,
            dropna=dropna,
            verbose=False,
            auto_time_params=True,
            **kwargs,
        )

    return _create


# ==== Generated Series fixtures ====


@pytest.fixture
def evenly_spaced_series(gen_evenly_spaced):
    """Series with evenly spaced time axis (cosine function)"""
    series = gen_evenly_spaced(length=10, pattern="cosine")
    series.label = "cosine"
    return series


@pytest.fixture
def unevenly_spaced_series(gen_unevenly_spaced):
    """Series with unevenly spaced time axis"""
    series = gen_unevenly_spaced(length=10, pattern="squared")
    return series


@pytest.fixture
def unevenly_spaced_series_nans(gen_unevenly_spaced):
    """Series with unevenly spaced time axis and NaN values"""
    series = gen_unevenly_spaced(length=10, pattern="squared")
    series.value[2:4] = np.nan
    return series


@pytest.fixture
def pinkseries(random_seed):
    """Series with 1/f spectrum (pink noise)"""
    t = np.arange(100)
    v = pyleo.utils.colored_noise(alpha=1.0, t=t, seed=random_seed, std=2.5)
    ts = pyleo.Series(t, v, verbose=False, auto_time_params=True)
    ts.label = "pink noise"
    return ts


# ==== GeoSeries fixtures ====


@pytest.fixture
def pinkgeoseries(geo_metadata, random_seed):
    """GeoSeries based on 1/f (pink) temporal structure"""
    t, v = pyleo.utils.gen_ts(
        model="colored_noise", alpha=1.0, nt=200, seed=random_seed
    )
    ts = pyleo.GeoSeries(t, v, verbose=False, **geo_metadata).standardize()
    return ts


@pytest.fixture
def multiple_pinkgeoseries():
    """Function to create multiple GeoSeries objects with pink noise

    This was moved from test_core_GeoSeries.py to centralize all fixtures
    """

    def _create(nrecs=20, seed=108, geobox=[-85.0, 85.0, -180, 180]):
        nt = 200
        lats = np.random.default_rng(seed=seed).uniform(geobox[0], geobox[1], nrecs)
        lons = np.random.default_rng(seed=seed + 1).uniform(geobox[2], geobox[3], nrecs)
        elevs = np.random.default_rng(seed=seed + 2).uniform(0, 4000, nrecs)
        unknowns = np.random.randint(0, len(elevs) - 1, size=2)
        for ik in unknowns:
            elevs[ik] = None

        archives = np.random.default_rng(seed=seed).choice(
            list(pyleo.utils.PLOT_DEFAULT.keys()) + [None], size=nrecs
        )
        obsTypes = np.random.default_rng(seed=seed).choice(
            ["MXD", "d18O", "Sr/Ca", None], size=nrecs
        )

        ts_list = []
        for i in range(nrecs):
            t, v = pyleo.utils.gen_ts(model="colored_noise", alpha=1.0, nt=nt)
            ts = pyleo.GeoSeries(
                t,
                v,
                verbose=False,
                label=f"pink series {i}",
                archiveType=archives[i],
                observationType=obsTypes[i],
                lat=lats[i],
                lon=lons[i],
                elevation=elevs[i],
            ).standardize()
            ts_list.append(ts)

        return pyleo.MultipleGeoSeries(ts_list, label="Multiple Pink GeoSeries")

    return _create


# ==== MultipleSeries fixtures ====


@pytest.fixture
def multipleseries_basic():
    """Basic MultipleSeries with two simple series"""
    ts1 = pyleo.Series(
        time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), auto_time_params=True
    )
    ts2 = pyleo.Series(
        time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), auto_time_params=True
    )
    ms = pyleo.MultipleSeries([ts1, ts2])
    return ms


@pytest.fixture
def multipleseries_nans():
    """MultipleSeries containing a series with NaN values"""
    t1 = np.arange(1, 10)
    v1 = np.ones(len(t1))
    ts1 = pyleo.Series(time=t1, value=v1, auto_time_params=True)

    t2 = np.arange(1, 10)
    v2 = np.ones(len(t1))
    v2[2:4] = np.nan
    ts2 = pyleo.Series(
        time=t2, value=v2, dropna=False, verbose=False, auto_time_params=True
    )
    ms = pyleo.MultipleSeries([ts1, ts2])
    return ms


@pytest.fixture
def multipleseries_science():
    """MultipleSeries created from actual scientific datasets"""
    soi = pyleo.utils.load_dataset("SOI")
    nino = pyleo.utils.load_dataset("NINO3")
    ms = soi & nino
    ms.name = "ENSO"
    return ms


# ==== EnsembleSeries fixtures ====


@pytest.fixture
def ensembleseries_basic():
    """Basic EnsembleSeries with two simple series"""
    ts1 = pyleo.Series(
        time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), auto_time_params=True
    )
    ts2 = pyleo.Series(
        time=np.array([1, 2, 4]), value=np.array([7, 8, 1]), auto_time_params=True
    )
    ens = pyleo.EnsembleSeries([ts1, ts2])
    return ens


@pytest.fixture
def ensembleseries_nans():
    """EnsembleSeries with a series containing NaN values"""
    t1 = np.arange(1, 10)
    v1 = np.ones(len(t1))
    ts1 = pyleo.Series(time=t1, value=v1)

    t2 = np.arange(1, 10)
    v2 = np.ones(len(t1))
    v2[2:4] = np.nan
    ts2 = pyleo.Series(
        time=t2, value=v2, dropna=False, verbose=False, auto_time_params=True
    )
    ens = pyleo.EnsembleSeries([ts1, ts2])
    return ens


@pytest.fixture
def ensembleseries_science(random_seed):
    """EnsembleSeries based on real scientific data with added noise"""
    np.random.seed(random_seed)
    soi = pyleo.utils.load_dataset("SOI")
    ens = pyleo.EnsembleSeries([soi for _ in range(5)])
    for series in ens.series_list:
        series.value += np.random.randn(len(series.value))
    return ens


# ==== EnsembleGeoSeries fixtures ====


@pytest.fixture
def ensemblegeoseries_basic():
    """Basic EnsembleGeoSeries with two GeoSeries"""
    time = np.arange(50)
    ts1 = pyleo.GeoSeries(
        time=time, value=np.random.randn(len(time)), lat=0, lon=0, auto_time_params=True
    )
    ts2 = pyleo.GeoSeries(
        time=time, value=np.random.randn(len(time)), lat=0, lon=0, auto_time_params=True
    )
    ens = pyleo.EnsembleGeoSeries([ts1, ts2])
    return ens


@pytest.fixture
def ensemblegeoseries_nans():
    """EnsembleGeoSeries with NaN values"""
    t1 = np.arange(50)
    v1 = np.random.randn(len(t1))
    ts1 = pyleo.GeoSeries(time=t1, value=v1, lat=0, lon=0, auto_time_params=True)

    t2 = np.arange(50)
    v2 = np.random.randn(len(t2))
    v2[2:4] = np.nan
    ts2 = pyleo.GeoSeries(
        time=t2,
        value=v2,
        dropna=False,
        verbose=False,
        lat=0,
        lon=0,
        auto_time_params=True,
    )
    ens = pyleo.EnsembleGeoSeries([ts1, ts2])
    return ens
