#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LipdSeries are (you guessed it), Series objects that are created from LiPD objects. As a subclass of Series, they inherit all its methods.
When created, LiPDSeries automatically instantiates the time, value and other parameters from what’s in the lipd file.

"""

from ..utils import plotting, mapping, lipdutils
from ..core.series import Series
from ..core.ensembleseries import EnsembleSeries

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
import os
import lipd as lpd


class LipdSeries(Series):
    '''LipdSeries are (you guessed it), Series objects that are created from LiPD objects. As a subclass of Series, they inherit all its methods.
    When created, LiPDSeries automatically instantiates the time, value and other parameters from what’s in the lipd file.
    These objects can be obtained from a LiPD file/object either through Pyleoclim or the LiPD utilities.
    If multiple objects (i.e., a list) are given, then the user will be prompted to choose one timeseries.

    Returns
    -------
    object : pyleoclim.LipdSeries

    See also
    --------

    pyleoclim.core.lipd.Lipd : Creates a Lipd object from LiPD Files

    pyleoclim.core.series.Series : Creates pyleoclim Series object

    pyleoclim.core.multipleseries.MultipleSeries : a collection of multiple Series objects

    Examples
    --------

    In this example, we will import a LiPD file and explore the various options to create a series object.

    First, let's look at the Lipd.to_tso option. This method is attractive because the object is a list of dictionaries that are easily explored in Python.

    .. ipython:: python
        :okwarning:
        :okexcept:

        import pyleoclim as pyleo
        url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
        data = pyleo.Lipd(usr_path = url)
        ts_list = data.to_tso()
        # Print out the dataset name and the variable name
        for item in ts_list:
            print(item['dataSetName']+': '+item['paleoData_variableName'])
        # Load the sst data into a LipdSeries. Since Python indexing starts at zero, sst has index 5.
        ts = pyleo.LipdSeries(ts_list[5])

    If you attempt to pass the full list of series, Pyleoclim will prompt you to choose a series by printing out something similar as above.
    If you already now the number of the timeseries object you're interested in, then you should use the following:

    .. ipython:: python
        :okwarning:
        :okexcept:

        ts1 = data.to_LipdSeries(number=5)

    If number is not specified, Pyleoclim will prompt you for the number automatically.

    Sometimes, one may want to create a MultipleSeries object from a collection of LiPD files. In this case, we recommend using the following:

    .. ipython:: python
        :okwarning:
        :okexcept:

        ts_list = data.to_LipdSeriesList()
        # only keep the Mg/Ca and SST
        ts_list=ts_list[4:]
        #create a MultipleSeries object
        ms=pyleo.MultipleSeries(ts_list)

    '''

    def __init__(self, tso, clean_ts=True, verbose=False):
        if type(tso) is list:
            self.lipd_ts = lipdutils.getTs(tso)
        else:
            self.lipd_ts = tso

        self.plot_default = {'ice-other': ['#FFD600', 'h'],
                             'ice/rock': ['#FFD600', 'h'],
                             'coral': ['#FF8B00', 'o'],
                             'documents': ['k', 'p'],
                             'glacierice': ['#86CDFA', 'd'],
                             'hybrid': ['#00BEFF', '*'],
                             'lakesediment': ['#4169E0', 's'],
                             'marinesediment': ['#8A4513', 's'],
                             'sclerosponge': ['r', 'o'],
                             'speleothem': ['#FF1492', 'd'],
                             'wood': ['#32CC32', '^'],
                             'molluskshells': ['#FFD600', 'h'],
                             'peat': ['#2F4F4F', '*'],
                             'midden': ['#824E2B', 'o'],
                             'other': ['k', 'o']}

        try:
            time, label = lipdutils.checkTimeAxis(self.lipd_ts)
            if label == 'age':
                time_name = 'Age'
                if 'ageUnits' in self.lipd_ts.keys():
                    time_unit = self.lipd_ts['ageUnits']
                else:
                    time_unit = None
            elif label == 'year':
                time_name = 'Year'
                if 'yearUnits' in self.lipd_ts.keys():
                    time_unit = self.lipd_ts['yearUnits']
                else:
                    time_unit = None
            try:
                if self.lipd_ts['mode'] == 'paleoData':
                    value = np.array(self.lipd_ts['paleoData_values'], dtype='float64')
                    value_name = self.lipd_ts['paleoData_variableName']
                    if 'paleoData_units' in self.lipd_ts.keys():
                        value_unit = self.lipd_ts['paleoData_units']
                    else:
                        value_unit = None
                    label = self.lipd_ts['dataSetName']
                    super(LipdSeries, self).__init__(time=time, value=value, time_name=time_name,
                                                     time_unit=time_unit, value_name=value_name, value_unit=value_unit,
                                                     label=label, clean_ts=clean_ts, verbose=verbose)
                elif self.lipd_ts['mode'] == 'chronData':
                    value = np.array(self.lipd_ts['chronData_values'], dtype='float64')
                    value_name = self.lipd_ts['chronData_variableName']
                    if 'paleoData_units' in self.lipd_ts.keys():
                        value_unit = self.lipd_ts['chronData_units']
                    else:
                        value_unit = None
                    label = self.lipd_ts['dataSetName']
                    super(LipdSeries, self).__init__(time=time, value=value, time_name=time_name,
                                                     time_unit=time_unit, value_name=value_name, value_unit=value_unit,
                                                     label=label, clean_ts=clean_ts, verbose=verbose)
            except:
                raise ValueError("paleoData_values should contain floats")
        except:
            raise KeyError("No time information present")

    def copy(self):
        '''Copy the object

        Returns
        -------
        object : pyleoclim.LipdSeries
            New object with data copied from original

        '''
        return deepcopy(self)

    def chronEnsembleToPaleo(self, D, number=None, chronNumber=None, modelNumber=None, tableNumber=None):
        '''Fetch chron ensembles from a Lipd object and return the ensemble as MultipleSeries

        Parameters
        ----------

        D : a LiPD object

        number: int, optional

            The number of ensemble members to store. Default is None, which corresponds to all present

        chronNumber: int, optional

            The chron object number. The default is None.

        modelNumber : int, optional

            Age model number. The default is None.

        tableNumber : int, optional

            Table number. The default is None.

        Raises
        ------

        ValueError

        Returns
        -------

        ens : pyleoclim.EnsembleSeries
            An EnsembleSeries object with each series representing a possible realization of the age model

        See also
        --------

        pyleoclim.core.ensembleseries.EnsembleSeries : An EnsembleSeries object with each series representing a possible realization of the age model

        pyleoclim.utils.lipdutils.mapAgeEnsembleToPaleoData : Map the depth for the ensemble age values to the paleo depth

        '''
        # get the corresponding LiPD
        dataSetName = self.lipd_ts['dataSetName']
        if type(D) is dict:
            try:
                lipd = D[dataSetName]
            except:
                lipd = D
        else:
            a = D.extract(dataSetName)
            lipd = a.__dict__['lipd']
        # Look for the ensemble and get values
        cwd = os.getcwd()
        csv_dict = lpd.getCsv(lipd)
        os.chdir(cwd)
        chron, paleo = lipdutils.isEnsemble(csv_dict)
        if len(chron) == 0:
            raise ValueError("No ChronMeasurementTables available")
        elif len(chron) > 1:
            if chronNumber == None or modelNumber == None or tableNumber == None:
                csvName = lipdutils.whichEnsemble(chron)
            else:
                str0 = 'chron' + str(chronNumber)
                str1 = 'model' + str(modelNumber)
                str2 = 'ensemble' + str(tableNumber)
                for item in chron:
                    if str0 in item and str1 in item and str2 in item:
                        csvName = item
            depth, ensembleValues = lipdutils.getEnsemble(csv_dict, csvName)
        else:
            depth, ensembleValues = lipdutils.getEnsemble(csv_dict, chron[0])
        # make sure it's sorted
        sort_ind = np.argsort(depth)
        depth = list(np.array(depth)[sort_ind])
        ensembleValues = ensembleValues[sort_ind, :]

        if number is not None:
            if number > np.shape(ensembleValues)[1]:
                warnings.warn(
                    'Selected number of ensemble members is greater than number of members in the ensemble table; passing')
                pass
            else:
                ensembleValues = ensembleValues[:, 0:number]

        # Map to paleovalues
        key = []
        for item in self.lipd_ts.keys():
            if 'depth' in item and 'Units' not in item:
                key.append(item)
        key = key[0]
        ds = np.array(self.lipd_ts[key], dtype='float64')
        if 'paleoData_values' in self.lipd_ts.keys():
            ys = np.array(self.lipd_ts['paleoData_values'], dtype='float64')
        elif 'chronData_values' in self.lipd_ts.keys():
            ys = np.array(self.lipd_ts['chronData_values'], dtype='float64')
        else:
            raise KeyError('no y-axis values available')
        # Remove NaNs
        ys_tmp = np.copy(ys)
        ds = ds[~np.isnan(ys_tmp)]
        sort_ind2 = np.argsort(ds)
        ds = np.array(ds[sort_ind2])
        ys = np.array(ys[sort_ind2])
        ensembleValuestoPaleo = lipdutils.mapAgeEnsembleToPaleoData(ensembleValues, depth, ds)
        # create multipleseries
        s_list = []
        for i, s in enumerate(ensembleValuestoPaleo.T):
            s_tmp = Series(time=s, value=ys,
                           verbose=i == 0,
                           clean_ts=False,
                           value_name=self.value_name,
                           value_unit=self.value_unit,
                           time_name=self.time_name,
                           time_unit=self.time_unit)
            s_list.append(s_tmp)

        ens = EnsembleSeries(series_list=s_list)

        return ens

    def map(self, projection='Orthographic', proj_default=True,
            background=True, borders=False, rivers=False, lakes=False,
            figsize=None, ax=None, marker=None, color=None,
            markersize=None, scatter_kwargs=None,
            legend=True, lgd_kwargs=None, savefig_settings=None):
        '''Map the location of the record

        Parameters
        ----------
        projection : str, optional

            The projection to use. The default is 'Robinson'.

        proj_default : bool; {True, False}, optional

            Whether to use the Pyleoclim defaults for each projection type. The default is True.

        background :  bool; {True, False}, optional

            Whether to use a background. The default is True.

        borders :  bool; {True, False}, optional

            Draw borders. The default is False.

        rivers :  bool; {True, False}, optional

            Draw rivers. The default is False.

        lakes :  bool; {True, False}, optional

            Draw lakes. The default is False.

        figsize : list or tuple, optional

            The size of the figure. The default is None.

        ax : matplotlib.ax, optional

            The matplotlib axis onto which to return the map. The default is None.

        marker : str, optional

            The marker type for each archive. The default is None. Uses plot_default

        color : str, optional

            Color for each archive. The default is None. Uses plot_default

        markersize : float, optional

            Size of the marker. The default is None.

        scatter_kwargs : dict, optional

            Parameters for the scatter plot. The default is None.

        legend :  bool; {True, False}, optional

            Whether to plot the legend. The default is True.

        lgd_kwargs : dict, optional

            Arguments for the legend. The default is None.

        savefig_settings : dict, optional

            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.

        Returns
        -------

        res : fig,ax

        See also
        --------

        pyleoclim.utils.mapping.map : Underlying mapping function for Pyleoclim

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            ts = data.to_LipdSeries(number=5)
            @savefig mapone.png
            fig, ax = ts.map()
            pyleo.closefig(fig)

        '''
        scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()
        # get the information from the timeseries
        lat = [self.lipd_ts['geo_meanLat']]
        lon = [self.lipd_ts['geo_meanLon']]

        if 'archiveType' in self.lipd_ts.keys():
            archiveType = lipdutils.LipdToOntology(self.lipd_ts['archiveType']).lower().replace(" ", "")
        else:
            archiveType = 'other'

        # make sure criteria is in the plot_default list
        if archiveType not in self.plot_default.keys():
            archiveType = 'other'

        if markersize is not None:
            scatter_kwargs.update({'s': markersize})

        if marker == None:
            marker = self.plot_default[archiveType][1]

        if color == None:
            color = self.plot_default[archiveType][0]

        if proj_default == True:
            proj1 = {'central_latitude': lat[0],
                     'central_longitude': lon[0]}
            proj2 = {'central_latitude': lat[0]}
            proj3 = {'central_longitude': lon[0]}

        archiveType = [archiveType]  # list so it will work with map
        marker = [marker]
        color = [color]

        if proj_default == True:

            try:
                res = mapping.map(lat=lat, lon=lon, criteria=archiveType,
                                  marker=marker, color=color,
                                  projection=projection, proj_default=proj1,
                                  background=background, borders=borders,
                                  rivers=rivers, lakes=lakes,
                                  figsize=figsize, ax=ax,
                                  scatter_kwargs=scatter_kwargs, legend=legend,
                                  lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings, )

            except:
                try:
                    res = mapping.map(lat=lat, lon=lon, criteria=archiveType,
                                      marker=marker, color=color,
                                      projection=projection, proj_default=proj3,
                                      background=background, borders=borders,
                                      rivers=rivers, lakes=lakes,
                                      figsize=figsize, ax=ax,
                                      scatter_kwargs=scatter_kwargs, legend=legend,
                                      lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)
                except:
                    res = mapping.map(lat=lat, lon=lon, criteria=archiveType,
                                      marker=marker, color=color,
                                      projection=projection, proj_default=proj2,
                                      background=background, borders=borders,
                                      rivers=rivers, lakes=lakes,
                                      figsize=figsize, ax=ax,
                                      scatter_kwargs=scatter_kwargs, legend=legend,
                                      lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)

        else:
            res = mapping.map(lat=lat, lon=lon, criteria=archiveType,
                              marker=marker, color=color,
                              projection=projection, proj_default=proj_default,
                              background=background, borders=borders,
                              rivers=rivers, lakes=lakes,
                              figsize=figsize, ax=ax,
                              scatter_kwargs=scatter_kwargs, legend=legend,
                              lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)
        return res

    def getMetadata(self):

        """ Get the necessary metadata for the ensemble plots

        Parameters
        ----------

        timeseries : object

                    a specific timeseries object.

        Returns
        -------

        res : dict
                  A dictionary containing the following metadata:
                    archiveType,
                    Authors (if more than 2, replace by et al),
                    PublicationYear,
                    Publication DOI,
                    Variable Name,
                    Units,
                    Climate Interpretation,
                    Calibration Equation,
                    Calibration References,
                    Calibration Notes

        """

        # Get all the necessary information
        # Top level information
        if "archiveType" in self.lipd_ts.keys():
            archiveType = self.lipd_ts["archiveType"]
        else:
            archiveType = "NA"

        if "pub1_author" in self.lipd_ts.keys():
            authors = self.lipd_ts["pub1_author"]
        else:
            authors = "NA"

        # Truncate if more than two authors
        idx = [pos for pos, char in enumerate(authors) if char == ";"]
        if len(idx) > 2:
            authors = authors[0:idx[1] + 1] + "et al."

        if "pub1_year" in self.lipd_ts.keys():
            Year = str(self.lipd_ts["pub1_year"])
        else:
            Year = "NA"

        if "pub1_doi" in self.lipd_ts.keys():
            DOI = self.lipd_ts["pub1_doi"]
        else:
            DOI = "NA"

        if self.lipd_ts['mode'] == 'paleoData':
            prefix = 'paleo'
        else:
            prefix = 'chron'

        if prefix + "Data_InferredVariableType" in self.lipd_ts.keys():
            if type(self.lipd_ts[prefix + "Data_InferredVariableType"]) is list:
                Variable = self.lipd_ts[prefix + "Data_InferredVariableType"][0]
            else:
                Variable = self.lipd_ts[prefix + "Data_InferredVariableType"]
        elif prefix + "Data_ProxyObservationType" in self.lipd_ts.keys():
            if type(self.lipd_ts[prefix + "Data_ProxyObservationType"]) is list:
                Variable = self.lipd_ts[prefix + "Data_ProxyObservationType"][0]
            else:
                Variable = self.lipd_ts[prefix + "Data_ProxyObservationType"]
        else:
            Variable = self.lipd_ts[prefix + "Data_variableName"]

        if prefix + "Data_units" in self.lipd_ts.keys():
            units = self.lipd_ts[prefix + "Data_units"]
        else:
            units = "NA"

        # Climate interpretation information
        if prefix + "Data_interpretation" in self.lipd_ts.keys():
            interpretation = self.lipd_ts[prefix + "Data_interpretation"][0]
            if "name" in interpretation.keys():
                ClimateVar = interpretation["name"]
            elif "variable" in interpretation.keys():
                ClimateVar = interpretation["variable"]
            else:
                ClimateVar = "NA"
            if "detail" in interpretation.keys():
                Detail = interpretation["detail"]
            elif "variableDetail" in interpretation.keys():
                Detail = interpretation['variableDetail']
            else:
                Detail = "NA"
            if "scope" in interpretation.keys():
                Scope = interpretation['scope']
            else:
                Scope = "NA"
            if "seasonality" in interpretation.keys():
                Seasonality = interpretation["seasonality"]
            else:
                Seasonality = "NA"
            if "interpdirection" in interpretation.keys():
                Direction = interpretation["interpdirection"]
            else:
                Direction = "NA"
        else:
            ClimateVar = "NA"
            Detail = "NA"
            Scope = "NA"
            Seasonality = "NA"
            Direction = "NA"

        # Calibration information
        if prefix + "Data_calibration" in self.lipd_ts.keys():
            calibration = self.lipd_ts[prefix + 'Data_calibration'][0]
            if "equation" in calibration.keys():
                Calibration_equation = calibration["equation"]
            else:
                Calibration_equation = "NA"
            if "calibrationReferences" in calibration.keys():
                ref = calibration["calibrationReferences"]
                if "author" in ref.keys():
                    ref_author = ref["author"][0]  # get the first author
                else:
                    ref_author = "NA"
                if "publicationYear" in ref.keys():
                    ref_year = str(ref["publicationYear"])
                else:
                    ref_year = "NA"
                Calibration_notes = ref_author + "." + ref_year
            elif "notes" in calibration.keys():
                Calibration_notes = calibration["notes"]
            else:
                Calibration_notes = "NA"
        else:
            Calibration_equation = "NA"
            Calibration_notes = "NA"

        # Truncate the notes if too long
        charlim = 30;
        if len(Calibration_notes) > charlim:
            Calibration_notes = Calibration_notes[0:charlim] + " ..."

        res = {"archiveType": archiveType,
               "authors": authors,
               "Year": Year,
               "DOI": DOI,
               "Variable": Variable,
               "units": units,
               "Climate_Variable": ClimateVar,
               "Detail": Detail,
               "Scope": Scope,
               "Seasonality": Seasonality,
               "Interpretation_Direction": Direction,
               "Calibration_equation": Calibration_equation,
               "Calibration_notes": Calibration_notes}

        return res

    def dashboard(self, figsize=[11, 8], plt_kwargs=None, distplt_kwargs=None, spectral_kwargs=None,
                  spectralsignif_kwargs=None, spectralfig_kwargs=None, map_kwargs=None, metadata=True,
                  savefig_settings=None, ensemble=False, D=None):
        '''

        Parameters
        ----------
        figsize : list or tuple, optional

            Figure size. The default is [11,8].

        plt_kwargs : dict, optional

            Optional arguments for the timeseries plot. See Series.plot() or EnsembleSeries.plot_envelope(). The default is None.

        distplt_kwargs : dict, optional

            Optional arguments for the distribution plot. See Series.distplot() or EnsembleSeries.plot_displot(). The default is None.

        spectral_kwargs : dict, optional

            Optional arguments for the spectral method. Default is to use Lomb-Scargle method. See Series.spectral() or EnsembleSeries.spectral(). The default is None.

        spectralsignif_kwargs : dict, optional

            Optional arguments to estimate the significance of the power spectrum. See PSD.signif_test. Note that we currently do not support significance testing for ensembles. The default is None.

        spectralfig_kwargs : dict, optional

            Optional arguments for the power spectrum figure. See PSD.plot() or MultiplePSD.plot_envelope(). The default is None.

        map_kwargs : dict, optional

            Optional arguments for the map. See LipdSeries.map(). The default is None.

        metadata : bool; {True,False}, optional

            Whether or not to produce a dashboard with printed metadata. The default is True.

        savefig_settings : dict, optional

            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}.
            The default is None.

        ensemble : bool; {True, False}, optional

            If True, will return the dashboard in ensemble modes if ensembles are available

        D : pyleoclim.Lipd object

            If asking for an ensemble plot, a pyleoclim.Lipd object must be provided

        Returns
        -------
        fig : matplotlib.figure

            The figure

        ax : matplolib.axis

            The axis

        See also
        --------

        pyleoclim.core.series.Series.plot : plot a timeseries

        pyleoclim.core.ensembleseries.EnsembleSeries.plot_envelope: Envelope plots for an ensemble

        pyleoclim.core.series.Series.distplot : plot a distribution of the timeseries

        pyleoclim.core.ensembleseries.EnsembleSeries.distplot : plot a distribution of the timeseries across ensembles

        pyleoclim.core.series.Series.spectral : spectral analysis method.

        pyleoclim.core.multipleseries.MultipleSeries.spectral : spectral analysis method for multiple series.

        pyleoclim.core.PSD.PSD.signif_test : significance test for timeseries analysis

        pyleoclim.core.PSD.PSD.plot : plot power spectrum

        pyleoclim.core.MulitplePSD.MulitplePSD.plot : plot envelope of power spectrum

        pyleoclim.core.lipdseries.LipdSeries.map : map location of dataset

        pyleolim.core.lipdseries.LipdSeries.getMetadata : get relevant metadata from the timeseries object

        pyleoclim.utils.mapping.map : Underlying mapping function for Pyleoclim

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            ts = data.to_LipdSeries(number=5)
            @savefig ts_dashboard.png
            fig, ax = ts.dashboard()
            pyleo.closefig(fig)

        '''
        if ensemble == True and D is None:
            raise ValueError("When an ensemble dashboard is requested, the corresponsind Lipd object must be supplied")

        if ensemble == True:
            warnings.warn('Some of the computation in ensemble mode can require a few minutes to complete.')

        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        res = self.getMetadata()
        # start plotting
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 5)
        gs.update(left=0, right=1.1)

        if ensemble == True:
            ens = self.chronEnsembleToPaleo(D)
            ensc = ens.common_time()

        ax = {}
        # Plot the timeseries
        plt_kwargs = {} if plt_kwargs is None else plt_kwargs.copy()
        ax['ts'] = plt.subplot(gs[0, :-3])
        plt_kwargs.update({'ax': ax['ts']})
        # use the defaults if color/markers not specified
        if ensemble == False:
            if 'marker' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ", "")
                plt_kwargs.update({'marker': self.plot_default[archiveType][1]})
            if 'color' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ", "")
                plt_kwargs.update({'color': self.plot_default[archiveType][0]})
            ax['ts'] = self.plot(**plt_kwargs)
        elif ensemble == True:
            if 'curve_clr' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ", "")
                plt_kwargs.update({'curve_clr': self.plot_default[archiveType][0]})
            if 'shade_clr' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ", "")
                plt_kwargs.update({'shade_clr': self.plot_default[archiveType][0]})
            # plt_kwargs.update({'ylabel':self.value_name})
            ax['ts'] = ensc.plot_envelope(**plt_kwargs)
        else:
            raise ValueError("Invalid argument value for ensemble")
        ymin, ymax = ax['ts'].get_ylim()

        # plot the distplot
        distplt_kwargs = {} if distplt_kwargs is None else distplt_kwargs.copy()
        ax['dts'] = plt.subplot(gs[0, 2])
        distplt_kwargs.update({'ax': ax['dts']})
        distplt_kwargs.update({'ylabel': 'PDF'})
        distplt_kwargs.update({'vertical': True})
        if 'color' not in distplt_kwargs.keys():
            archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ", "")
            distplt_kwargs.update({'color': self.plot_default[archiveType][0]})
        if ensemble == False:
            ax['dts'] = self.distplot(**distplt_kwargs)
        elif ensemble == True:
            ax['dts'] = ensc.distplot(**distplt_kwargs)
        ax['dts'].set_ylim([ymin, ymax])
        ax['dts'].set_yticklabels([])
        ax['dts'].set_ylabel('')
        ax['dts'].set_yticks([])

        # make the map - brute force since projection is not being returned properly
        lat = [self.lipd_ts['geo_meanLat']]
        lon = [self.lipd_ts['geo_meanLon']]

        map_kwargs = {} if map_kwargs is None else map_kwargs.copy()
        if 'projection' in map_kwargs.keys():
            projection = map_kwargs['projection']
        else:
            projection = 'Orthographic'
        if 'proj_default' in map_kwargs.keys():
            proj_default = map_kwargs['proj_default']
        else:
            proj_default = True
        if proj_default == True:
            proj1 = {'central_latitude': lat[0],
                     'central_longitude': lon[0]}
            proj2 = {'central_latitude': lat[0]}
            proj3 = {'central_longitude': lon[0]}
            try:
                proj = mapping.set_proj(projection=projection, proj_default=proj1)
            except:
                try:
                    proj = mapping.set_proj(projection=projection, proj_default=proj3)
                except:
                    proj = mapping.set_proj(projection=projection, proj_default=proj2)
        if 'marker' in map_kwargs.keys():
            marker = map_kwargs['marker']
        else:
            marker = self.plot_default[archiveType][1]
        if 'color' in map_kwargs.keys():
            color = map_kwargs['color']
        else:
            color = self.plot_default[archiveType][0]
        if 'background' in map_kwargs.keys():
            background = map_kwargs['background']
        else:
            background = True
        if 'borders' in map_kwargs.keys():
            borders = map_kwargs['borders']
        else:
            borders = False
        if 'rivers' in map_kwargs.keys():
            rivers = map_kwargs['rivers']
        else:
            rivers = False
        if 'lakes' in map_kwargs.keys():
            lakes = map_kwargs['lakes']
        else:
            lakes = False
        if 'scatter_kwargs' in map_kwargs.keys():
            scatter_kwargs = map_kwargs['scatter_kwargs']
        else:
            scatter_kwargs = {}
        if 'markersize' in map_kwargs.keys():
            scatter_kwargs.update({'s': map_kwargs['markersize']})
        else:
            pass
        if 'lgd_kwargs' in map_kwargs.keys():
            lgd_kwargs = map_kwargs['lgd_kwargs']
        else:
            lgd_kwargs = {}
        if 'legend' in map_kwargs.keys():
            legend = map_kwargs['legend']
        else:
            legend = False
        # make the plot map

        data_crs = ccrs.PlateCarree()
        ax['map'] = plt.subplot(gs[1, 0], projection=proj)
        ax['map'].coastlines()
        if background is True:
            ax['map'].stock_img()
        # Other extra information
        if borders is True:
            ax['map'].add_feature(cfeature.BORDERS)
        if lakes is True:
            ax['map'].add_feature(cfeature.LAKES)
        if rivers is True:
            ax['map'].add_feature(cfeature.RIVERS)
        ax['map'].scatter(lon, lat, zorder=10, label=marker, facecolor=color, transform=data_crs, **scatter_kwargs)
        if legend == True:
            ax.legend(**lgd_kwargs)

        # spectral analysis
        spectral_kwargs = {} if spectral_kwargs is None else spectral_kwargs.copy()
        if 'method' in spectral_kwargs.keys():
            pass
        else:
            spectral_kwargs.update({'method': 'lomb_scargle'})
        if 'freq_method' in spectral_kwargs.keys():
            pass
        else:
            if ensemble == False:
                spectral_kwargs.update({'freq_method': 'lomb_scargle'})
            elif ensemble == True:
                pass

        ax['spec'] = plt.subplot(gs[1, 1:3])
        spectralfig_kwargs = {} if spectralfig_kwargs is None else spectralfig_kwargs.copy()
        spectralfig_kwargs.update({'ax': ax['spec']})

        if ensemble == False:
            ts_preprocess = self.detrend().standardize()
            psd = ts_preprocess.spectral(**spectral_kwargs)

            # Significance test
            spectralsignif_kwargs = {} if spectralsignif_kwargs is None else spectralsignif_kwargs.copy()
            psd_signif = psd.signif_test(**spectralsignif_kwargs)
            # plot
            if 'color' not in spectralfig_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ", "")
                spectralfig_kwargs.update({'color': self.plot_default[archiveType][0]})
            if 'signif_clr' not in spectralfig_kwargs.keys():
                spectralfig_kwargs.update({'signif_clr': 'grey'})
            ax['spec'] = psd_signif.plot(**spectralfig_kwargs)

        elif ensemble == True:
            if 'curve_clr' not in spectralfig_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ", "")
                spectralfig_kwargs.update({'curve_clr': self.plot_default[archiveType][0]})
            if 'shade_clr' not in spectralfig_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(res['archiveType']).lower().replace(" ", "")
                spectralfig_kwargs.update({'shade_clr': self.plot_default[archiveType][0]})
            psd = ensc.detrend().standardize().spectral(**spectral_kwargs)
            # plot
            ax['spec'] = psd.plot_envelope(**spectralfig_kwargs)

        # Make the plot

        if metadata == True:
            # get metadata
            textstr = "archiveType: " + res["archiveType"] + "\n" + "\n" + \
                      "Authors: " + res["authors"] + "\n" + "\n" + \
                      "Year: " + res["Year"] + "\n" + "\n" + \
                      "DOI: " + res["DOI"] + "\n" + "\n" + \
                      "Variable: " + res["Variable"] + "\n" + "\n" + \
                      "units: " + res["units"] + "\n" + "\n" + \
                      "Climate Interpretation: " + "\n" + \
                      "    Climate Variable: " + res["Climate_Variable"] + "\n" + \
                      "    Detail: " + res["Detail"] + "\n" + \
                      "    Seasonality: " + res["Seasonality"] + "\n" + \
                      "    Direction: " + res["Interpretation_Direction"] + "\n \n" + \
                      "Calibration: \n" + \
                      "    Equation: " + res["Calibration_equation"] + "\n" + \
                      "    Notes: " + res["Calibration_notes"]
            plt.figtext(0.7, 0.4, textstr, fontsize=12)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)

        return fig, ax

    def mapNearRecord(self, D, n=5, radius=None, sameArchive=False,
                      projection='Orthographic', proj_default=True,
                      background=True, borders=False, rivers=False,
                      lakes=False, figsize=None, ax=None,
                      marker_ref=None, color_ref=None, marker=None, color=None,
                      markersize_adjust=False, scale_factor=100, scatter_kwargs=None,
                      legend=True, lgd_kwargs=None, savefig_settings=None):
        """ Map records that are near the timeseries of interest

        Parameters
        ----------
        D : pyleoclim.Lipd

            A pyleoclim LiPD object

        n : int, optional

            The n number of closest records. The default is 5.

        radius : float, optional

            The radius to take into consideration when looking for records (in km). The default is None.

        sameArchive :  bool; {True, False}, optional

            Whether to consider records from the same archiveType as the original record. The default is False.

        projection : str, optional

            A valid cartopy projection. The default is 'Orthographic'.
            See pyleoclim.utils.mapping for a list of supported projections.

        proj_default : True or dict, optional

            The projection arguments. If not True, then use a dictionary to pass the appropriate arguments depending on the projection. The default is True.

        background : bool; {True, False}, optional

            Whether to use a background. The default is True.

        borders :  bool; {True, False}, optional

            Whether to plot country borders. The default is False.

        rivers :  bool; {True, False}, optional

            Whether to plot rivers. The default is False.

        lakes :  bool; {True, False}, optional

            Whether to plot rivers. The default is False.

        figsize : list or tuple, optional

            the size of the figure. The default is None.

        ax : matplotlib.ax, optional

            The matplotlib axis onto which to return the map. The default is None.

        marker_ref : str, optional

            Marker shape to use for the main record. The default is None, which corresponds to the default marker for the archiveType

        color_ref : str, optional

            The color for the main record. The default is None, which corresponds to the default color for the archiveType.

        marker : str or list, optional

            Marker shape to use for the other records. The default is None, which corresponds to the marker shape for each archiveType.

        color : str or list, optional

            Color for each marker. The default is None, which corresponds to the color for each archiveType

        markersize_adjust :  bool; {True, False}, optional

            Whether to adjust the marker size according to distance from record of interest. The default is False.

        scale_factor : int, optional

            The maximum marker size. The default is 100.

        scatter_kwargs : dict, optional

            Parameters for the scatter plot. The default is None.

        legend :  bool; {True, False}, optional

            Whether to show the legend. The default is True.

        lgd_kwargs : dict, optional

            Parameters for the legend. The default is None.

        savefig_settings : dict, optional

            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.

        Returns
        -------

        res : dict
            contains fig and ax


        See also
        --------

        pyleoclim.utils.mapping.map: Underlying mapping function for Pyleoclim

        pyleoclim.utils.mapping.dist_sphere: Calculate distance on a sphere

        pyleoclim.utils.mapping.compute_dist: Compute the distance between a point and an array

        pyleoclim.utils.mapping.within_distance: Returns point in an array within a certain distance

        """
        scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()

        # get the information about the original timeseries
        lat_ref = [self.lipd_ts['geo_meanLat']]
        lon_ref = [self.lipd_ts['geo_meanLon']]

        if 'archiveType' in self.lipd_ts.keys():
            archiveType_ref = lipdutils.LipdToOntology(self.lipd_ts['archiveType']).lower().replace(" ", "")
        else:
            archiveType_ref = 'other'

        # make sure criteria is in the plot_default list
        if archiveType_ref not in self.plot_default.keys():
            archiveType_ref = 'other'

        # get information about the other timeseries
        lat = []
        lon = []
        archiveType = []

        dataSetName_ref = self.lipd_ts['dataSetName']

        for idx, key in enumerate(D.lipd):
            if key != dataSetName_ref:
                d = D.lipd[key]
                lat.append(d['geo']['geometry']['coordinates'][1])
                lon.append(d['geo']['geometry']['coordinates'][0])
                if 'archiveType' in d.keys():
                    archiveType.append(lipdutils.LipdToOntology(d['archiveType']).lower().replace(" ", ""))
                else:
                    archiveType.append('other')

        # make sure criteria is in the plot_default list
        for idx, val in enumerate(archiveType):
            if val not in self.plot_default.keys():
                archiveType[idx] = 'other'

        if len(lat) == 0:  # this should not happen unless the coordinates are not available in the LiPD file
            raise ValueError('no matching record found')

        # Filter by the same type of archive if asked
        if sameArchive == True:
            idx_archive = [idx for idx, val in enumerate(archiveType) if val == archiveType_ref]
            if len(idx_archive) == 0:
                raise ValueError(
                    'No records corresponding to the same archiveType available. Widen your search criteria.')
            else:
                lat = [lat[idx] for idx in idx_archive]
                lon = [lon[idx] for idx in idx_archive]
                archiveType = [archiveType[idx] for idx in idx_archive]

        # compute the distance
        dist = mapping.compute_dist(lat_ref, lon_ref, lat, lon)

        if radius:
            idx_radius = mapping.within_distance(dist, radius)
            if len(idx_radius) == 0:
                raise ValueError('No records withing matching radius distance. Widen your search criteria')
            else:
                lat = [lat[idx] for idx in idx_radius]
                lon = [lon[idx] for idx in idx_radius]
                archiveType = [archiveType[idx] for idx in idx_radius]
                dist = [dist[idx] for idx in idx_radius]

        # print a warning if plotting less than asked because of the filters

        if n > len(dist):
            warnings.warn("Number of matching records is less" + \
                          " than the number of neighbors chosen. Including all records " + \
                          " in the analysis.")
            n = len(dist)

        # Sort the distance array
        sort_idx = np.argsort(dist)
        dist = [dist[idx] for idx in sort_idx]
        lat = [lat[idx] for idx in sort_idx]
        lon = [lon[idx] for idx in sort_idx]
        archiveType = [archiveType[idx] for idx in sort_idx]

        # Grab the right number of records
        dist = dist[0:n]
        lat = lat[0:n]
        lon = lon[0:n]
        archiveType = archiveType[0:n]

        # Get plotting information

        if marker_ref == None:
            marker_ref = self.plot_default[archiveType_ref][1]
        if color_ref == None:
            color_ref = self.plot_default[archiveType_ref][0]

        if marker == None:
            marker = []
            for item in archiveType:
                marker.append(self.plot_default[item][1])
        elif type(marker) == list:
            if len(marker) != len(lon):
                raise ValueError('When providing a list, it should be the same length as the number of records')
        elif type(marker) == str:
            marker = [marker] * len(lon)

        if color == None:
            color = []
            for item in archiveType:
                color.append(self.plot_default[item][0])
        elif type(color) == list:
            if len(color) != len(lon):
                raise ValueError('When providing a list, it should be the same length as the number of records')
        elif type(color) == str:
            color = [color] * len(lon)

        if 'edgecolors' not in scatter_kwargs.keys():
            edgecolors = []
            for item in marker:
                edgecolors.append('w')
            edgecolors.append('k')
            scatter_kwargs.update({'edgecolors': edgecolors})

        # Start plotting
        lat_all = lat + lat_ref
        lon_all = lon + lon_ref
        dist_all = dist + [0]
        archiveType_all = archiveType
        archiveType_all.append(archiveType_ref)

        color_all = color
        color_all.append(color_ref)
        marker_all = marker
        marker_all.append(marker_ref)

        if markersize_adjust == True:
            scale = dist_all[-1] / (scale_factor - 30)
            s = list(np.array(dist_all) * 1 / (scale) + 30)
            s.reverse()
            scatter_kwargs.update({'s': s})

        proj1 = {'central_latitude': lat_ref[0],
                 'central_longitude': lon_ref[0]}
        proj2 = {'central_latitude': lat_ref[0]}
        proj3 = {'central_longitude': lon_ref[0]}

        if proj_default == True:
            try:
                res = mapping.map(lat=lat_all, lon=lon_all,
                                  criteria=archiveType_all,
                                  marker=marker_all, color=color_all,
                                  projection=projection, proj_default=proj1,
                                  background=background, borders=borders,
                                  rivers=rivers, lakes=lakes,
                                  figsize=figsize, ax=ax,
                                  scatter_kwargs=scatter_kwargs, legend=legend,
                                  lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)
            except:
                try:
                    res = mapping.map(lat=lat_all, lon=lon_all,
                                      criteria=archiveType_all,
                                      marker=marker_all, color=color_all,
                                      projection=projection, proj_default=proj2,
                                      background=background, borders=borders,
                                      rivers=rivers, lakes=lakes,
                                      figsize=figsize, ax=ax,
                                      scatter_kwargs=scatter_kwargs, legend=legend,
                                      lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)
                except:
                    res = mapping.map(lat=lat_all, lon=lon_all,
                                      criteria=archiveType_all,
                                      marker=marker_all, color=color_all,
                                      projection=projection, proj_default=proj3,
                                      background=background, borders=borders,
                                      rivers=rivers, lakes=lakes,
                                      figsize=figsize, ax=ax,
                                      scatter_kwargs=scatter_kwargs, legend=legend,
                                      lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)

        else:
            res = mapping.map(lat=lat_all, lon=lon_all,
                              criteria=archiveType_all,
                              marker=marker_all, color=color_all,
                              projection=projection, proj_default=proj_default,
                              background=background, borders=borders,
                              rivers=rivers, lakes=lakes,
                              figsize=figsize, ax=ax,
                              scatter_kwargs=scatter_kwargs, legend=legend,
                              lgd_kwargs=lgd_kwargs, savefig_settings=savefig_settings)

        return res

    def plot_age_depth(self, figsize=[10, 4], plt_kwargs=None,
                       savefig_settings=None,
                       ensemble=False, D=None, num_traces=10, ensemble_kwargs=None,
                       envelope_kwargs=None, traces_kwargs=None):

        '''

        Parameters
        ----------

        figsize : list or tuple, optional

            Size of the figure. The default is [10,4].

        plt_kwargs : dict, optional

            Arguments for basic plot. See Series.plot() for details. The default is None.

        savefig_settings : dict, optional

            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. The default is None.

        ensemble : bool; {True, False}, optional

            Whether to use age model ensembles stored in the file for the plot. The default is False.
            If no ensemble can be found, will error out.

        D : pyleoclim.Lipd, optional

            The pyleoclim.Lipd object from which the pyleoclim.LipdSeries is derived. The default is None.

        num_traces : int, optional

            Number of individual age models to plot. To plot only the envelope and median value, set this parameter to 0 or None. The default is 10.

        ensemble_kwargs : dict, optional

            Parameters associated with identifying the chronEnsemble tables. See pyleoclim.core.lipdseries.LipdSeries.chronEnsembleToPaleo() for details. The default is None.

        envelope_kwargs : dict, optional

            Parameters to control the envelope plot. See pyleoclim.EnsembleSeries.plot_envelope() for details. The default is None.

        traces_kwargs : dict, optional

            Parameters to control the traces plot. See pyleoclim.EnsembleSeries.plot_traces() for details. The default is None.

        Raises
        ------

        ValueError

            In ensemble mode, make sure that the LiPD object is given

        KeyError

            Depth information needed.

        Returns
        -------

        fig,ax

            The figure

        See also
        --------

        pyleoclim.core.lipd.Lipd : Pyleoclim internal representation of a LiPD file

        pyleoclim.core.series.Series.plot : Basic plotting in pyleoclim

        pyleoclim.core.lipdseries.LipdSeries.chronEnsembleToPaleo : Function to map the ensemble table to a paleo depth.

        pyleoclim.core.ensembleseries.EnsembleSeries.plot_envelope : Create an envelope plot from an ensemble

        pyleoclim.core.ensembleseries.EnsembleSeries.plot_traces : Create a trace plot from an ensemble

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            D = pyleo.Lipd('http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=Crystal.McCabe-Glynn.2013')
            ts=D.to_LipdSeries(number=2)
            @savefig lipdseries_age_depth.png
            fig, ax = ts.plot_age_depth()
            pyleo.closefig(fig)

        '''
        if ensemble == True and D is None:
            raise ValueError("When an ensemble is requested, the corresponsind Lipd object must be supplied")

        meta = self.getMetadata()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plt_kwargs = {} if plt_kwargs is None else plt_kwargs.copy()
        # get depth
        try:
            value_depth, label_depth = lipdutils.checkXaxis(self.lipd_ts, 'depth')
            if 'depthUnits' in self.lipd_ts.keys():
                units_depth = self.lipd_ts['depthUnits']
            else:
                units_depth = 'NA'
        except:
            raise KeyError('No depth available in this record')

        # create a series for which time is actually depth

        if ensemble == False:
            ts = Series(time=self.time, value=value_depth,
                        time_name=self.time_name, time_unit=self.time_unit,
                        value_name=label_depth, value_unit=units_depth)
            plt_kwargs = {} if plt_kwargs is None else plt_kwargs.copy()
            if 'marker' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(meta['archiveType']).lower().replace(" ", "")
                plt_kwargs.update({'marker': self.plot_default[archiveType][1]})
            if 'color' not in plt_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(meta['archiveType']).lower().replace(" ", "")
                plt_kwargs.update({'color': self.plot_default[archiveType][0]})

            fig, ax = ts.plot(**plt_kwargs)
        elif ensemble == True:
            ensemble_kwargs = {} if ensemble_kwargs is None else ensemble_kwargs.copy()
            ens = self.chronEnsembleToPaleo(D, **ensemble_kwargs)
            # NOT  A VERY ELEGANT SOLUTION: replace depth in the dictionary
            for item in ens.__dict__['series_list']:
                item.__dict__['value'] = value_depth
                item.__dict__['value_unit'] = units_depth
                item.__dict__['value_name'] = 'depth'
            envelope_kwargs = {} if envelope_kwargs is None else envelope_kwargs.copy()
            if 'curve_clr' not in envelope_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(meta['archiveType']).lower().replace(" ", "")
                envelope_kwargs.update({'curve_clr': self.plot_default[archiveType][0]})
            if 'shade_clr' not in envelope_kwargs.keys():
                archiveType = lipdutils.LipdToOntology(meta['archiveType']).lower().replace(" ", "")
                envelope_kwargs.update({'shade_clr': self.plot_default[archiveType][0]})
            ens2 = ens.common_time()
            if num_traces > 0:
                # envelope_kwargs.update({'mute':True})
                fig, ax = ens2.plot_envelope(**envelope_kwargs)
                traces_kwargs = {} if traces_kwargs is None else traces_kwargs.copy()
                if 'color' not in traces_kwargs.keys():
                    archiveType = lipdutils.LipdToOntology(meta['archiveType']).lower().replace(" ", "")
                    traces_kwargs.update({'color': self.plot_default[archiveType][0]})
                if 'linestyle' not in traces_kwargs.keys():
                    traces_kwargs.update({'linestyle': 'dashed'})
                traces_kwargs.update({'ax': ax, 'num_traces': num_traces})
                ens2.plot_traces(**traces_kwargs)
            else:
                fig, ax = ens2.plot_envelope(**envelope_kwargs)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax
