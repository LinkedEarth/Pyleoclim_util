"""
A MultipleGeoSeries object is a collection (more precisely, a 
list) of GeoSeries objects. This is handy in case you want to apply the same method 
to such a collection at once (e.g. process a bunch of series in a consistent fashion).
Compared to its parent class MultipleSeries, MultipleGeoSeries opens new possibilites regarding mapping.
"""
from ..core.multipleseries import MultipleSeries
import warnings

class MultipleGeoSeries(MultipleSeries):
    '''MultipleGeoSeries object.

    This object handles a collection of the type GeoSeries and can be created from a list of such objects.
    MultipleGeoSeries should be used when the need to run analysis on multiple records arises, such as running principal component analysis.
    Some of the methods automatically transform the time axis prior to analysis to ensure consistency.

    Parameters
    ----------

    series_list : list
    
        a list of pyleoclim.Series objects

    time_unit : str
    
        The target time unit for every series in the list.
        If None, then no conversion will be applied;
        Otherwise, the time unit of every series in the list will be converted to the target.

    label : str
   
        label of the collection of timeseries (e.g. 'Euro 2k')

    Examples
    --------
    TODO: Euro2k example
                
    '''
    def __init__(self, series_list, time_unit=None, label=None, name=None):
        from ..core.geoseries import GeoSeries
        
        self.series_list = series_list
        self.time_unit = time_unit
        self.label = label
        self.name = name
        if name is not None:
            warnings.warn("`name` is a deprecated property, which will be removed in future releases. Please use `label` instead.",
                          DeprecationWarning, stacklevel=2)
        # check that all components are Series
        if not all([isinstance(ts, GeoSeries) for ts in self.series_list]):
            raise ValueError('All components must be GeoSeries objects')

        if self.time_unit is not None:
            new_ts_list = []
            for ts in self.series_list:
                new_ts = ts.convert_time_unit(time_unit=self.time_unit)
                new_ts_list.append(new_ts)

            self.series_list = new_ts_list
