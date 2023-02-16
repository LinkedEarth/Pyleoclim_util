
import pytest
from pyleoclim.utils import tsbase
import numpy as np


def test_time_unit_to_datum_exp_dir_unknown_time_unit():
    # unknown time unit, gets default values
    time_unit = 'unknown_value'
    with pytest.warns(match='Time unit'):
        (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(time_unit)
        assert datum == 0
        assert exponent == 0
        assert direction == 'prograde'

    with pytest.warns(match='Time unit'):
        (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(time_unit, time_name='unknown')
        assert datum == 0
        assert exponent == 0
        assert direction == 'prograde'

    # with pytest.warns(match='Time unit'):
    #     (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(time_unit, time_name='age')
    #     assert datum == 0
    #     assert exponent == 0
    #     assert direction == 'retrograde'   


@pytest.mark.parametrize('time_unit', tsbase.MATCH_KA)
def test_time_unit_to_datum_exp_dir_ka(time_unit):
    (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(time_unit)
    assert datum == 1950
    assert exponent == 3
    assert direction == 'retrograde'


@pytest.mark.parametrize('time_unit', tsbase.MATCH_MA)
def test_time_unit_to_datum_exp_dir_ma(time_unit):
    (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(time_unit)
    assert datum == 1950
    assert exponent == 6
    assert direction == 'retrograde'


@pytest.mark.parametrize('time_unit', tsbase.MATCH_GA)
def test_time_unit_to_datum_exp_dir_ga(time_unit):
    (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(time_unit)
    assert datum == 1950
    assert exponent == 9
    assert direction == 'retrograde'


def test_time_unit_to_datum_exp_dir_b2k():
    time_unit = 'years b2k'
    (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(time_unit)
    assert datum == 2000
    assert exponent == 0
    assert direction == 'retrograde' 


@pytest.mark.parametrize('time_unit', ['years bp', 'years bnf'])
def test_time_unit_to_datum_exp_dir_bp(time_unit):
    (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(time_unit)
    assert datum == 1950
    assert exponent == 0
    assert direction == 'retrograde'


@pytest.mark.parametrize('time_unit', ['years AD', 'yr CE'])
def test_time_unit_to_datum_exp_dir_ad(time_unit):
    (datum, exponent, direction) = tsbase.time_unit_to_datum_exp_dir(time_unit)
    assert datum == 0
    assert exponent == 0
    assert direction == 'prograde'


def test_convert_datetime_index_ka(dataframe_dt):
    time_unit = 'ka'
    time_name = None
    time = tsbase.convert_datetime_index_to_time(
        dataframe_dt.index, 
        time_unit, 
        time_name=time_name,
        )
    expected = np.array([-0.06899658, -0.06999658, -0.07099932, -0.07199658, -0.07299658])
    assert np.allclose(time.values, expected, rtol=1e-05, atol=1e-08)


def test_convert_datetime_index_ma(dataframe_dt):
    time_unit = 'ma'
    time_name = None
    time = tsbase.convert_datetime_index_to_time(
        dataframe_dt.index, 
        time_unit, 
        time_name=time_name,
        )
    expected = np.array([-6.89965777e-05, -6.99965777e-05, -7.09993155e-05, -7.19965777e-05, -7.29965777e-05])
    assert np.allclose(time.values, expected, rtol=1e-05, atol=1e-08)


def test_convert_datetime_index_ga(dataframe_dt):
    time_unit = 'ga'
    time_name = None
    time = tsbase.convert_datetime_index_to_time(
        dataframe_dt.index, 
        time_unit, 
        time_name=time_name,
        )
    expected = np.array([-6.89965777e-08, -6.99965777e-08, -7.09993155e-08, -7.19965777e-08, -7.29965777e-08])
    assert np.allclose(time.values, expected, rtol=1e-05, atol=1e-08)


def test_convert_datetime_index_bp(dataframe_dt):
    time_unit = 'years B.P.'
    time_name = None
    time = tsbase.convert_datetime_index_to_time(
        dataframe_dt.index, 
        time_unit, 
        time_name=time_name,
        )
    expected = np.array([-68.99657769, -69.99657769, -70.99931554, -71.99657769, -72.99657769])
    assert np.allclose(time.values, expected, rtol=1e-05, atol=1e-08)


def test_convert_datetime_index_ad(dataframe_dt):
    time_unit = 'AD'
    time_name = None
    time = tsbase.convert_datetime_index_to_time(
        dataframe_dt.index, 
        time_unit, 
        time_name=time_name,
        )
    expected = np.array([2018.99657769, 2019.99657769, 2020.99931554, 2021.99657769, 2022.99657769])
    assert np.allclose(time.values, expected, rtol=1e-05, atol=1e-08)


def test_convert_datetime_index_nondt_index(dataframe):
    time_unit = 'yr'
    time_name = None
    with pytest.raises(ValueError, match='not a proper DatetimeIndex object'):
        tsbase.convert_datetime_index_to_time(
            dataframe.index, 
            time_unit, 
            time_name=time_name,
            )

