
import pytest

import numpy as np
import pandas as pd
import calendar
from datetime import datetime

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


# Tests for custom_year_averages function

class TestCustomYearAverages:
    """Test suite for custom_year_averages function"""
    
    @pytest.fixture
    def monthly_data(self):
        """Create monthly test data"""
        dates = pd.date_range('2020-01-31', '2023-12-31', freq='ME')
        values = np.arange(len(dates))  # Sequential values for predictable results
        return pd.Series(values, index=dates, name='monthly_test')
    
    @pytest.fixture
    def daily_data(self):
        """Create daily test data with uneven spacing"""
        dates = pd.to_datetime([
            '2021-01-01', '2021-01-15', '2021-02-01', 
            '2021-02-15', '2021-03-01', '2021-03-15'
        ])
        values = [10, 20, 30, 40, 50, 60]
        return pd.Series(values, index=dates, name='daily_test')
    
    def test_basic_functionality(self, monthly_data):
        """Test basic functionality with monthly data"""
        result = tsutils.custom_year_averages(monthly_data, 1, 12, years=2021)
        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert 2021 in result.index
        assert not np.isnan(result.iloc[0])
    
    def test_single_year_int(self, monthly_data):
        """Test with single year as integer"""
        result = tsutils.custom_year_averages(monthly_data, 1, 3, years=2021)
        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.index[0] == 2021
    
    def test_multiple_years_list(self, monthly_data):
        """Test with multiple years as list"""
        years = [2021, 2022]
        result = tsutils.custom_year_averages(monthly_data, 1, 6, years=years)
        assert len(result) == 2
        assert all(year in result.index for year in years)
    
    def test_year_range(self, monthly_data):
        """Test with range of years"""
        years = range(2021, 2024)
        result = tsutils.custom_year_averages(monthly_data, 7, 12, years=years)
        assert len(result) == 3
        assert all(year in result.index for year in years)
    
    def test_all_years_none(self, monthly_data):
        """Test with years=None (all available years)"""
        result = tsutils.custom_year_averages(monthly_data, 1, 12, years=None)
        expected_years = [2020, 2021, 2022, 2023]
        assert len(result) == 4
        assert all(year in result.index for year in expected_years)
    
    def test_straddling_years(self, monthly_data):
        """Test periods that straddle calendar years (e.g., Apr-Mar)"""
        result = tsutils.custom_year_averages(monthly_data, 4, 3, years=2021)
        assert len(result) == 1
        assert 2021 in result.index
        # Should include Apr 2020 through Mar 2021
    
    def test_straddling_years_all(self, monthly_data):
        """Test straddling periods with years=None"""
        result = tsutils.custom_year_averages(monthly_data, 10, 9, years=None)
        # Should exclude first year since we need data from previous year
        expected_years = [2021, 2022, 2023]
        assert len(result) == 3
        assert all(year in result.index for year in expected_years)
    
    def test_edge_months(self, monthly_data):
        """Test with edge months (Jan and Dec)"""
        result_jan_dec = tsutils.custom_year_averages(monthly_data, 1, 12, years=2021)
        result_dec_jan = tsutils.custom_year_averages(monthly_data, 12, 1, years=2021)
        
        assert len(result_jan_dec) == 1
        assert len(result_dec_jan) == 1
        assert not np.isnan(result_jan_dec.iloc[0])
        assert not np.isnan(result_dec_jan.iloc[0])
    
    def test_single_month_period(self, monthly_data):
        """Test with same start and end month"""
        result = tsutils.custom_year_averages(monthly_data, 6, 6, years=2021)
        assert len(result) == 1
        # Should return the value for June 2021
    
    def test_daily_data_weighting(self, daily_data):
        """Test weighted averaging with daily data"""
        result = tsutils.custom_year_averages(daily_data, 1, 3, years=2021)
        assert len(result) == 1
        assert not np.isnan(result.iloc[0])
        # Weighted average should be different from simple mean due to uneven spacing
        simple_mean = daily_data.mean()
        assert result.iloc[0] != simple_mean  # Should be different due to weighting
    
    def test_missing_data_year(self, monthly_data):
        """Test with year that has no data"""
        result = tsutils.custom_year_averages(monthly_data, 1, 12, years=2019)
        # Function should return empty series when no data is available for requested year
        assert len(result) == 0
    
    def test_partial_year_data(self, monthly_data):
        """Test with partial year data"""
        # Test with a year that only has partial data
        result = tsutils.custom_year_averages(monthly_data, 6, 12, years=2020)
        assert len(result) == 1
        assert not np.isnan(result.iloc[0])
    
    def test_invalid_months(self, monthly_data):
        """Test with invalid month values"""
        with pytest.raises(ValueError, match="Months must be between 1 and 12"):
            tsutils.custom_year_averages(monthly_data, 0, 12, years=2021)
        
        with pytest.raises(ValueError, match="Months must be between 1 and 12"):
            tsutils.custom_year_averages(monthly_data, 1, 13, years=2021)
    
    def test_non_datetime_index(self):
        """Test with non-datetime index"""
        data = pd.Series([1, 2, 3], index=[1, 2, 3])
        with pytest.raises(ValueError, match="Data must have a DatetimeIndex"):
            tsutils.custom_year_averages(data, 1, 12, years=2021)
    
    def test_empty_series(self):
        """Test with empty series"""
        empty_data = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        result = tsutils.custom_year_averages(empty_data, 1, 12, years=2021)
        # Should return empty series when no data is available
        assert len(result) == 0
    
    def test_series_name_preservation(self, monthly_data):
        """Test that series name is preserved in result"""
        result = tsutils.custom_year_averages(monthly_data, 1, 12, years=2021)
        assert result.name == monthly_data.name
    
    def test_leap_year_handling(self):
        """Test handling of leap years"""
        # Create data that includes Feb 29
        dates = pd.date_range('2020-01-31', '2020-12-31', freq='ME')  # 2020 is leap year
        values = np.arange(len(dates))
        data = pd.Series(values, index=dates)
        
        result = tsutils.custom_year_averages(data, 1, 12, years=2020)
        assert len(result) == 1
        assert not np.isnan(result.iloc[0])


# Tests for _calculate_weighted_average function

class TestCalculateWeightedAverage:
    """Test suite for _calculate_weighted_average helper function"""
    
    def test_empty_data(self):
        """Test with empty data"""
        empty_data = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        start = pd.Timestamp('2021-01-01')
        end = pd.Timestamp('2021-01-31')
        result = tsutils._calculate_weighted_average(empty_data, start, end)
        assert np.isnan(result)
    
    def test_single_observation(self):
        """Test with single observation"""
        data = pd.Series([42.0], index=[pd.Timestamp('2021-01-15')])
        start = pd.Timestamp('2021-01-01')
        end = pd.Timestamp('2021-01-31')
        result = tsutils._calculate_weighted_average(data, start, end)
        assert result == 42.0
    
    def test_monthly_data_equal_weights(self):
        """Test that monthly data gets equal weights"""
        # Create monthly data (end of month dates)
        dates = pd.to_datetime(['2021-01-31', '2021-02-28', '2021-03-31'])
        values = [10, 20, 30]
        data = pd.Series(values, index=dates)
        start = pd.Timestamp('2021-01-01')
        end = pd.Timestamp('2021-03-31')
        
        result = tsutils._calculate_weighted_average(data, start, end)
        expected = np.mean(values)  # Should be simple mean for monthly data
        assert result == expected
    
    def test_daily_data_time_weighting(self):
        """Test that daily data uses time-based weighting"""
        # Create unevenly spaced daily data
        dates = pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-10'])
        values = [10, 20, 30]
        data = pd.Series(values, index=dates)
        start = pd.Timestamp('2021-01-01')
        end = pd.Timestamp('2021-01-31')
        
        result = tsutils._calculate_weighted_average(data, start, end)
        simple_mean = np.mean(values)
        # Result should be different from simple mean due to time weighting
        assert result != simple_mean
    
    def test_unsorted_data(self):
        """Test that function handles unsorted data correctly"""
        # Create unsorted data
        dates = pd.to_datetime(['2021-01-10', '2021-01-01', '2021-01-20'])
        values = [20, 10, 30]
        data = pd.Series(values, index=dates)
        start = pd.Timestamp('2021-01-01')
        end = pd.Timestamp('2021-01-31')
        
        result = tsutils._calculate_weighted_average(data, start, end)
        assert not np.isnan(result)
        assert isinstance(result, float)
    
    def test_duplicate_timestamps(self):
        """Test handling of duplicate timestamps"""
        # This is an edge case that might occur with duplicate timestamps
        dates = pd.to_datetime(['2021-01-15', '2021-01-15'])  # Same timestamp
        values = [10, 20]
        data = pd.Series(values, index=dates)
        start = pd.Timestamp('2021-01-01')
        end = pd.Timestamp('2021-01-31')
        
        result = tsutils._calculate_weighted_average(data, start, end)
        # Should return a finite value (handling edge case gracefully)
        assert not np.isnan(result)
        assert isinstance(result, float)
