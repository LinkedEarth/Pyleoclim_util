"""
The EnsembleSeries class is a child of EnsembleSeries, designed for ensemble applications (e.g. draws from a posterior distribution of ages, model ensembles with randomized initial conditions, or some other stochastic ensemble).
In addition to an EnsembleSeries object, an EnsembleGeoSeries object has the following properties:
- The addition of location data (latitude, longitude, and optionally elevation).   
- Optional depth data.
- Optional proxy sensor type, observation type, and archive type metadata

"""

from ..core.ensembleseries import EnsembleSeries
from ..core.geoseries import GeoSeries

import numpy as np
import seaborn as sns

class EnsembleGeoSeries(EnsembleSeries):
    ''' EnsembleSeries object

    The EnsembleSeries object is a child of the MultipleSeries object, that is, a special case of MultipleSeries, aiming for ensembles of similar series.
    Ensembles usually arise from age modeling or Bayesian calibrations. All members of an EnsembleSeries object are assumed to share identical labels and units.

    All methods available for MultipleSeries are available for EnsembleSeries. Some functions were modified for the special case of ensembles.
    The class enables ensemble-oriented methods for computation (e.g., quantiles) 
    and visualization (e.g., envelope plot) that are unavailable to other classes.

    Parameters
    ----------

    series_list : list
        List of GeoSeries objects

    lat : float
        latitude N in decimal degrees. Must be in the range [-90;+90]
        
    lon : float
        longitude East in decimal degrees. Must be in the range [-180;+360]
        No conversion is applied as mapping utilities convert to [-180,+180] internally
       
    elevation : float
        elevation of the sample, in meters above sea level. Negative numbers indicate depth below global mean sea level, therefore.                                                                                          

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
        climate archive, one of 'Borehole', 'Coral', 'FluvialSediment', 'GlacierIce', 'GroundIce', 'LakeSediment', 'MarineSediment', 'Midden', 'MolluskShell', 'Peat', 'Sclerosponge', 'Shoreline', 'Speleothem', 'TerrestrialSediment', 'Wood'                                                                                   
        Reference: https://lipdverse.org/vocabulary/archivetype/
    
    control_archiveType  : [True, False]
        Whether to standardize the name of the archiveType agains the vocabulary from: https://lipdverse.org/vocabulary/paleodata_proxy/. 
        If set to True, will only allow for these terms and automatically convert known synonyms to the standardized name. Only standardized variable names will be automatically assigned a color scheme.  
        Default is False. 
        
    sensorType : string
        sensor, e.g. a paleoclimate proxy sensor. This property can be used to differentiate between species of foraminifera
        
    observationType : string
        observation type,  e.g. a proxy observation. See https://lipdverse.org/vocabulary/paleodata_proxy/. Note: this is preferred terminology but not enforced
        
    depth : array
        depth at which the values were collected
        
    depth_name : string
        name of the field, e.g. 'mid-depth', 'top-depth', etc   
        
    depth_unit : string
         units of the depth axis, e.g. 'cm'

    '''
    def __init__(self, series_list,lat,lon,elevation=None,archiveType=None,control_archiveType = False, 
                 sensorType = None, observationType = None, depth = None, depth_name = None, depth_unit= None):

        super().__init__(series_list)

        if lat is None:
            # check that all components are GeoSeries
            if not all([isinstance(ts, GeoSeries) for ts in series_list]):
                raise ValueError('If lat is not passed, all components must be GeoSeries objects')
            else:
                self.lat = series_list[0].lat
        elif lon is None:
            if not all([isinstance(ts, GeoSeries) for ts in series_list]):
                raise ValueError('If lon is not passed, all components must be GeoSeries objects')
            else:
                self.lon = series_list[0].lon

        else:
            lat = float(lat) 
            if -90 <= lat <= 90: 
                self.lat = lat
            else:
                ValueError('Latitude must be a number in [-90; 90]')
            
            lon = float(lon)
            if -180 < lon <= 360:     
                self.lon = lon
            else:
                ValueError('Longitude must be a number in [-180,360]')

        self.elevation = elevation
        self.archiveType = archiveType
        self.control_archiveType = control_archiveType
        self.sensorType = sensorType
        self.observationType = observationType
        self.depth = depth
        self.depth_name = depth_name
        self.depth_unit = depth_unit

    def make_labels(self):
        '''Initialization of labels

        Returns
        -------

        time_header : str

            Label for the time axis

        value_header : str

            Label for the value axis

        '''
        ts_list = self.series_list

        if ts_list[0].time_name is not None:
            time_name_str = ts_list[0].time_name
        else:
            time_name_str = 'time'

        if ts_list[0].value_name is not None:
            value_name_str = ts_list[0].value_name
        else:
            value_name_str = 'value'

        if ts_list[0].value_unit is not None:
            value_header = f'{value_name_str} [{ts_list[0].value_unit}]'
        else:
            value_header = f'{value_name_str}'

        if ts_list[0].time_unit is not None:
            time_header = f'{time_name_str} [{ts_list[0].time_unit}]'
        else:
            time_header = f'{time_name_str}'

        return time_header, value_header