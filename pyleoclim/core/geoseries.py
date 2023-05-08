"""
The GeoSeries class is a child of Series, with additional metadata latitude (lat) and longitude (lon)
This unlocks plotting capabilities like map() and dashboard(). 
"""
from ..utils import plotting
from ..core.series import Series
from ..core.multipleseries import MultipleSeries

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy

from tqdm import tqdm

class GeoSeries(Series):
    '''The geoSeries class describes the most basic objects in Pyleoclim.
    
    Parameters
    ----------

    time : list or numpy.array
        independent variable (t)

    value : list of numpy.array
        values of the dependent variable (y)
        
    lat : float
        latitude N in decimal degrees.
        
    lon : float
        longitude East in decimal degrees. Negative values will be converted to an angle in [0 , 360)
       
    elevation : float
        elevation of the sample, in meters above sea level.                                                                                          

    time_unit : string
        Units for the time vector (e.g., 'years').
        Default is None

    time_name : string
        Name of the time vector (e.g., 'Time','Age').
        Default is None. This is used to label the time axis on plots

    value_name : string
        Name of the value vector (e.g., 'temperature')
        Default is None

    value_unit : string
        Units for the value vector (e.g., 'deg C')
        Default is None

    label : string
        Name of the time series (e.g., 'Nino 3.4')
        Default is None

    log : dict
        Dictionary of tuples documentating the various transformations applied to the object
        
    keep_log : bool
        Whether to keep a log of applied transformations. False by default
        
  
    importedFrom : string
        source of the dataset. If it came from a LiPD file, this could be the datasetID property 

    archiveType : string
        climate archive, one of ....                                                                                    

    dropna : bool
        Whether to drop NaNs from the series to prevent downstream functions from choking on them
        defaults to True
        
    sort_ts : str
        Direction of sorting over the time coordinate; 'ascending' or 'descending'
        Defaults to 'ascending'
        
    verbose : bool
        If True, will print warning messages if there is any
        
    clean_ts : boolean flag
         set to True to remove the NaNs and make time axis strictly prograde with duplicated timestamps reduced by averaging the values
         Default is None (marked for deprecation)

    Examples
    --------

    Import the Southern Oscillation Index (SOI) and display a quick synopsis:

    >>> soi = pyleo.utils.load_dataset('SOI')
    >>> soi.view()
          
    '''

    def __init__(self, time, value, lat, lon, elevation = None, time_unit=None, time_name=None, 
                 value_name=None, value_unit=None, label=None, 
                 importedFrom=None, archiveType = None, log=None, keep_log=False,
                 sort_ts = 'ascending', dropna = True, verbose=True, clean_ts=False):
        
       
        # assign latitude
        if lat is not None:
            lat = float(lat) 
            if -90 <= lat <= 90: 
                self.lat = lat
            else:
                ValueError('Latitude must be a number in [-90; 90]')
        else:
            self.lat = None # assign a default value to prevent bugs ?
            
        # assign longitude
        if lon is not None:
            lon = float(lon)
            if 0 <= lon < 360:     
                self.lon = lon
            elif -180 <= lon < 0:
                self.lon = 360 - lon
            else:
                ValueError('Longitude must be a number in [-180,360]')
        else:
            self.lon = None # assign a default value to prevent bugs ?
            
        # elevation
        self.elevation = elevation
            
        #assign all the rest
        super().__init__(time, value, time_unit, time_name, value_name,
                         value_unit, label, importedFrom, archiveType,
                         log, keep_log, sort_ts, dropna, verbose, clean_ts)
                      

    @property
    def metadata(self):
        return dict(
            lat = self.lat,
            lon = self.lon,
            elevation = self.elevation,
            time_unit = self.time_unit,
            time_name = self.time_name,
            value_unit = self.value_unit,
            value_name = self.value_name,
            label = self.label,
            archiveType = self.archiveType,
            importedFrom = self.importedFrom,
            log = self.log,
        )