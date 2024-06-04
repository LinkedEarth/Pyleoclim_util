"""
A MultipleSeries object is a collection (more precisely, a 
list) of multiple Series objects. This is handy in case you want to apply the same method 
to such a collection at once (e.g. process a bunch of series in a consistent fashion).
"""

from ..utils import tsutils, plotting, jsonutils
from ..utils import correlation as corrutils

from ..core.correns import CorrEns
from ..core.scalograms import MultipleScalogram
from ..core.psds import MultiplePSD
from ..core.multivardecomp import MultivariateDecomp
from ..core.resolutions import MultipleResolution

import warnings
import numpy as np
from copy import deepcopy

from matplotlib.ticker import FormatStrFormatter
import matplotlib.transforms as transforms
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm
from scipy import stats
from statsmodels.multivariate.pca import PCA

class MultipleSeries:
    '''MultipleSeries object.

    This object handles a collection of the type Series and can be created from a list of such objects.
    MultipleSeries should be used when the need to run analysis on multiple records arises, such as running principal component analysis.
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
   
        label of the collection of timeseries (e.g. 'PAGES 2k ice cores')

    Examples
    --------

    .. jupyter-execute::

        soi = pyleo.utils.load_dataset('SOI')
        nino = pyleo.utils.load_dataset('NINO3')
        ms = soi & nino
        ms.label = 'ENSO'
        ms
                
    '''
    def __init__(self, series_list, time_unit=None, label=None, name=None):
        
        self.series_list = series_list
        self.time_unit = time_unit
        self.label = label
        self.name = name
        if name is not None:
            warnings.warn("`name` is a deprecated property, which will be removed in future releases. Please use `label` instead.",
                          DeprecationWarning, stacklevel=2)
        # check that all components are Series
        from ..core.series import Series
        from ..core.geoseries import GeoSeries
        
        if not all([isinstance(ts, (Series, GeoSeries)) for ts in self.series_list]):
            raise ValueError('All components must be of the same type')

        if self.time_unit is not None:
            new_ts_list = []
            for ts in self.series_list:
                new_ts = ts.convert_time_unit(time_unit=self.time_unit)
                new_ts_list.append(new_ts)

            self.series_list = new_ts_list
            
    def __repr__(self):
        return repr(self.to_pandas()) 
    
    def __len__(self):
        return self.series_list.__len__()
    
    def view(self):
        '''
        Generates a DataFrame version of the MultipleSeries object, suitable for viewing in a Jupyter Notebook

        Returns
        -------
        pd.DataFrame
        
        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo

            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            ms.name = 'ENSO'
            ms.view()

        '''
        return self.to_pandas(paleo_style=True)
    
    def remove(self, label):
        """
        Remove Series based on given label.

        Modifies the MultipleSeries, does not return anything.
        """
        to_remove = None
        for series in self.series_list:
            if series.metadata['label'] == label:
                to_remove = series
                break
        if to_remove is None:
            labels = [series.metadata['label'] for series in self.series_list]
            raise ValueError(f"Label {label} not found, expected one of: {labels}")
        self.series_list.remove(series)
    
    def __sub__(self, label):
        """
        Remove Series based on given label.

        Modifies the MultipleSeries, does not return anything.

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            import numpy as np
            ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='foo', verbose=False)
            ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='bar', verbose=False)
            ms = pyleo.MultipleSeries([ts1, ts2])
            # Remove pyleo.Series labelled 'bar' from the multiple series:
            ms - 'bar'

        """
        self.remove(label)

    def __add__(self, other):
        """
        Append a pyleo.Series, or combine with another pyleo.MultipleSeries.
        
        Parameters
        ----------
        other
            pyleo.Series or pyleo.MultipleSeries to combine with.
        
        Returns
        -------
        pyleo.MultipleSeries

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            import numpy as np
            ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='ts1', verbose=False)
            ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='ts2', verbose=False)
            ts3 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='ts3', verbose=False)
            ts4 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='ts4', verbose=False)
            ts5 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='ts5', verbose=False)
            ms1 = pyleo.MultipleSeries([ts1, ts2, ts3])
            ms2 = pyleo.MultipleSeries([ts4, ts5])
        
            # Combine the Multiple Series ms1 and ms2 by using the addition operator:
            ms = ms1 + ms2

        """
        from ..core.series import Series
        if isinstance(other, Series):
            return self.append(other)
        if isinstance(other, MultipleSeries):
            for series in other.series_list:
                self = self.append(series)
            return self
        else:
           raise TypeError(f"Expected pyleo.Series or pyleo.MultipleSeries, got: {type(other)}")
         
    
    def __and__(self, other):
        """
        Append a Series.

        Parameters
        ----------
        other
            Series to append.

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            import numpy as np
            ts1 = pyleo.Series(time=np.array([1, 2, 4]), value=np.array([7, 4, 9]), time_unit='years CE', label='ts1', verbose=False)
            ts2 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='ts2', verbose=False)
            ts3 = pyleo.Series(time=np.array([1, 3, 4]), value=np.array([7, 8, 1]), time_unit='years CE', label='ts3', verbose=False)
            # Combine ts1, ts2, and ts3 into a multiple series:
            ms = ts1 & ts2 & ts3

        """
        from ..core.series import Series
        if not isinstance(other, Series):
            raise TypeError(f"Expected pyleo.Series, got: {type(other)}")
        return self.append(other)

    def convert_time_unit(self, time_unit=None):
        ''' Convert the time units of the object

        Parameters
        ----------
        time_unit : str
        
            the target time unit, possible input:
            {
                'year', 'years', 'yr', 'yrs',
                'y BP', 'yr BP', 'yrs BP', 'year BP', 'years BP',
                'ky BP', 'kyr BP', 'kyrs BP', 'ka BP', 'ka',
                'my BP', 'myr BP', 'myrs BP', 'ma BP', 'ma',
            }

        Examples
        --------
        
        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            new_ms = ms.convert_time_unit('yr BP')
            print('Original timeseries:')
            print('time unit:', ms.time_unit)
            print()
            print('Converted timeseries:')
            print('time unit:', new_ms.time_unit)

        '''
        
        if time_unit is None: # if not provided, find a common time unit
            units = [ts.time_unit for ts in self.series_list]
            unique_units = np.unique(units).tolist()
            count_units = np.zeros(len(unique_units))
            for i, u in enumerate(unique_units):
                count_units[i] = units.count(u)
            time_unit = unique_units[count_units.argmax()]                

        new_ms = self.copy()
        new_ts_list = []
        for ts in self.series_list:
            new_ts = ts.convert_time_unit(time_unit=time_unit)
            new_ts_list.append(new_ts)

        new_ms.time_unit = time_unit
        new_ms.series_list = new_ts_list
        return new_ms

    def filter(self, cutoff_freq=None, cutoff_scale=None, method='butterworth', **kwargs):
        ''' Filtering the timeseries in the MultipleSeries object

        Parameters
        ----------
        method : str; {'savitzky-golay', 'butterworth', 'firwin', 'lanczos'}
        
            The filtering method  
            - 'butterworth': the Butterworth method (default)
            - 'savitzky-golay': the Savitzky-Golay method
            - 'firwin': FIR filter design using the window method, with default window as Hamming
            - 'lanczos': lowpass filter via Lanczos resampling

        cutoff_freq : float or list
        
            The cutoff frequency only works with the Butterworth method.
            If a float, it is interpreted as a low-frequency cutoff (lowpass).
            If a list,  it is interpreted as a frequency band (f1, f2), with f1 < f2 (bandpass).

        cutoff_scale : float or list
        
            cutoff_freq = 1 / cutoff_scale
            The cutoff scale only works with the Butterworth method and when cutoff_freq is None.
            If a float, it is interpreted as a low-frequency (high-scale) cutoff (lowpass).
            If a list,  it is interpreted as a frequency band (f1, f2), with f1 < f2 (bandpass).

        kwargs : dict
        
            A dictionary of the keyword arguments for the filtering method,
            See pyleoclim.utils.filter.savitzky_golay, pyleoclim.utils.filter.butterworth, pyleoclim.utils.filter.firwin, and pyleoclim.utils.filter.lanczos for the details

        Returns
        -------
        ms : MultipleSeries

        See also
        --------

        pyleoclim.series.Series.filter : filtering for Series objects    

        pyleoclim.utils.filter.butterworth : Butterworth method

        pyleoclim.utils.filter.savitzky_golay : Savitzky-Golay method

        pyleoclim.utils.filter.firwin : FIR filter design using the window method

        pyleoclim.utils.filter.lanczos : lowpass filter via Lanczos resampling

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            ms_filter = ms.filter(method='lanczos',cutoff_scale=20)

        '''

        ms = self.copy()

        new_tslist = []

        for ts in self.series_list:
            new_tslist.append(ts.filter(cutoff_freq=cutoff_freq, cutoff_scale=cutoff_scale, method=method, **kwargs))

        ms.series_list = new_tslist

        return ms

    def append(self,ts, inplace=False):
        '''Append timeseries ts to MultipleSeries object

        Parameters
        ----------
        ts : pyleoclim.Series
        
            The pyleoclim Series object to be appended to the MultipleSeries object

        Returns
        -------
        ms : MultipleSeries
        
            The augmented object, comprising the old one plus `ts`

        See also
        --------

        pyleoclim.core.series.Series : A Pyleoclim Series object

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            NINO3 = pyleo.utils.load_dataset('NINO3')
            ms = pyleo.MultipleSeries([soi], label = 'ENSO')
            ms.append(NINO3)

        '''
        for series in self.series_list:
            if series.equals(ts) == (True, True):
                raise ValueError(f"Given series is identical to existing series {series}")

        ms = self.copy()
        ts_list = deepcopy(ms.series_list)
        ts_list.append(ts)
        ms = MultipleSeries(ts_list)
        if inplace is True:
            self.series_list = ts_list
        return ms

    def copy(self):
        '''Copy the object

        Returns
        -------
        ms : MultipleSeries
        
            The copied version of the pyleoclim.MultipleSeries object

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            ms_copy = ms.copy()

        '''
        return deepcopy(self)
    
    def flip(self, axis='value'):
        '''
        Flips the Series along one or both axes

        Parameters
        ----------
        axis : str, optional
            The axis along which the Series will be flipped. The default is 'value'.
            Other acceptable options are 'time' or 'both'.
            
        Returns
        -------
        ms : MultipleSeries
            The flipped object
            
         Examples
         --------

        .. jupyter-execute::

            import pyleoclim as pyleo        
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            ms.name = 'ENSO'
            fig, ax = ms.flip().stackplot()
        
        
        Note that labels have been updated to reflect the flip
        '''
        
        ms=self.copy()
        for idx,item in enumerate(ms.series_list):
            s=item.flip(axis=axis, keep_log=False)
            ms.series_list[idx]=s
        
        return ms

    def standardize(self):
        '''Standardize each series object in a collection

        Returns
        -------
        ms : MultipleSeries
        
            The standardized pyleoclim.MultipleSeries object

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            ms_std = ms.standardize()

        '''
        ms=self.copy()
        for idx,item in enumerate(ms.series_list):
            s=item.copy()
            v_mod=tsutils.standardize(item.value)[0]
            s.value=v_mod
            ms.series_list[idx]=s
        return ms

    def increments(self, step_style='median', verbose=False):
        '''
        Extract grid properties (start, stop, step) of all the Series objects in a collection.

        Parameters
        ----------
        step_style : str; {'median','mean','mode','max'}
        
            Method to obtain a representative step if x is not evenly spaced.
            Valid entries: 'median' [default], 'mean', 'mode' or 'max'.
            The "mode" is the most frequent entry in a dataset, and may be a good choice if the timeseries
            is nearly equally spaced but for a few gaps.

            "max" is a conservative choice, appropriate for binning methods and Gaussian kernel coarse-graining
        
        verbose : bool
            If True, will print out warning messages when they appear
            
        Returns
        -------
        increments : numpy.array
        
            n x 3 array, where n is the number of series,
            
            * index 0 is the earliest time among all Series
            
            * index 1 is the latest time among all Series
            
            * index 2 is the step, chosen according to step_style
            
        See also
        --------
        
        pyleoclim.utils.tsutils.increments :  underlying array-level utility

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            increments = ms.increments()

        '''
        gp = np.empty((len(self.series_list),3)) # obtain grid parameters
        for idx,item in enumerate(self.series_list):
            item      = item.clean(verbose=verbose)
            gp[idx,:] = tsutils.increments(item.time, step_style=step_style)

        return gp
    
    def common_time(self, method='interp', step = None, start = None, stop = None, step_style = None, time_axis = None, **kwargs):
        ''' Aligns the time axes of a MultipleSeries object
        
        The alignment is achieved via binning, interpolation, or Gaussian kernel. Alignment is critical for workflows
        that need to assume a common time axis for the group of series under consideration.

        The common time axis is characterized by the following parameters:

        start : the latest start date of the bunch (maximun of the minima)

        stop  : the earliest stop date of the bunch (minimum of the maxima)

        step  : The representative spacing between consecutive values

        Optional arguments for binning, Gaussian kernel (gkernel) interpolation are those of the underling functions.

        If any of the time axes are retrograde, this step makes them prograde.

        Parameters
        ----------
        method :  string; {'bin','interp','gkernel'}
        
            either 'bin', 'interp' [default] or 'gkernel'
            
        step : float
        
            common step for all time axes. Default is None and inferred from the timeseries spacing
            
        start : float
        
            starting point of the common time axis. Default is None and inferred as the max of the min of the time axes for the timeseries.
            
        stop : float
        
            end point of the common time axis. Default is None and inferred as the min of the max of the time axes for the timeseries.
        
        step_style : string; {'median', 'mean', 'mode', 'max'}
        
            Method to obtain a representative step among all Series (using tsutils.increments).
            Default value is None, so that it will be chosen according to the method: 'max' for bin and gkernel, 'mean' for interp. 

        time_axis : array
            Time axis onto which all the series will be aligned. Will override step,start,stop, and step_style if they are passed.

        kwargs: dict
        
            keyword arguments (dictionary) of the bin, gkernel or interp methods

        Returns
        -------
        ms : MultipleSeries
        
            The MultipleSeries objects with all series aligned to the same time axis.

        Notes
        -----

        `start`, `stop`, `step`, and `step_style` are interpreted differently depending on the method used. 
        Interp uses these to specify the `time_axis` onto which interpolation will be applied.
        Bin and gkernel use these to specify the `bin_edges` which define the "buckets" used for the
        respective methods.


        See also
        --------

        pyleoclim.utils.tsutils.bin : put timeseries values into bins of equal size (possibly leaving NaNs in).

        pyleoclim.utils.tsutils.gkernel : coarse-graining using a Gaussian kernel

        pyleoclim.utils.tsutils.interp : interpolation onto a regular grid (default = linear interpolation)

        pyleoclim.utils.tsutils.increments : infer grid properties

        Examples
        --------

        .. jupyter-execute::

            import numpy as np
            import pyleoclim as pyleo
            import matplotlib.pyplot as plt
            from pyleoclim.utils.tsmodel import colored_noise

            # create 2 incompletely sampled series
            ns = 2 ; nt = 200; n_del = 20
            serieslist = []

            for j in range(ns):
                t = np.arange(nt)
                v = colored_noise(alpha=1, t=t)
                deleted_idx = np.random.choice(range(np.size(t)), n_del, replace=False)
                tu =  np.delete(t, deleted_idx)
                vu =  np.delete(v, deleted_idx)
                ts = pyleo.Series(time = tu, value = vu, label = 'series {}'.format(j+1), verbose=False)
                serieslist.append(ts)

            # create MS object from the list
            ms = pyleo.MultipleSeries(serieslist)

            fig, ax = plt.subplots(2,2,sharex=True,sharey=True, figsize=(10,8))
            ax = ax.flatten()
            # apply common_time with default parameters
            msc = ms.common_time()
            msc.plot(title='linear interpolation',ax=ax[0], legend=False)

            # apply common_time with binning
            msc = ms.common_time(method='bin')
            msc.plot(title='Binning',ax=ax[1], legend=False)

            # apply common_time with gkernel
            msc = ms.common_time(method='gkernel')
            msc.plot(title=r'Gaussian kernel ($h=3$)',ax=ax[2],legend=False)

            # apply common_time with gkernel and a large bandwidth
            msc = ms.common_time(method='gkernel', h=.5)
            msc.plot(title=r'Gaussian kernel ($h=.5$)',ax=ax[3],legend=False)
            fig.tight_layout()
            # Optional close fig after plotting

        '''
        
        if time_axis is not None:
            if start is not None or stop is not None or step is not None or step_style is not None:
                warnings.warn('The time axis has been passed with other time axis relevant arguments {start,stop,step,step_style}. Time_axis takes priority and will be used.')
            even_axis=None
        else:
            # specify stepping style
            if step_style is None: # if step style isn't specified, pick a robust choice according to method
                if method == 'bin' or method == 'gkernel':
                    step_style = 'max'
                elif  method == 'interp':
                    step_style = 'mean'
                
            # obtain grid properties with given step_style
            gp = self.increments(step_style=step_style)
            
            # define grid step     
            if step is not None and step > 0:
                common_step = step 
            else:
                if step_style == 'mean':
                    common_step = gp[:,2].mean()
                elif step_style == 'max':
                    common_step = gp[:,2].max()
                elif step_style == 'mode':
                    common_step = stats.mode(gp[:,2])[0][0]
                else:
                    common_step = np.median(gp[:,2])
            # define start and stop
            if start is None: 
                start = gp[:,0].max() # pick the latest of the start times
            if stop is None:
                stop  = gp[:,1].min() # pick the earliest of the stop times
            if start > stop:
                raise ValueError('At least one series has no common time interval with others. Please check the time axis of the series.')
            
            even_axis = tsutils.make_even_axis(start=start,stop=stop,step=common_step)
        
        ms = self.copy()

        # apply each method
        if method == 'bin':
            for idx,item in enumerate(self.series_list):
                ts = item.copy()
                d = tsutils.bin(ts.time, ts.value, bin_edges=even_axis, time_axis=time_axis, no_nans=False, **kwargs)
                ts.time  = d['bins']
                ts.value = d['binned_values']
                ms.series_list[idx] = ts

        elif method == 'interp':

            if time_axis is None:
                time_axis = even_axis
            for idx,item in enumerate(self.series_list):
                ts = item.copy()
                ti, vi = tsutils.interp(ts.time, ts.value, time_axis=time_axis, **kwargs)
                ts.time  = ti
                ts.value = vi
                ms.series_list[idx] = ts

        elif method == 'gkernel':
            for idx,item in enumerate(self.series_list):
                ts = item.copy()
                ti, vi = tsutils.gkernel(ts.time,ts.value,bin_edges=even_axis, time_axis=time_axis, no_nans=False,**kwargs)
                ts.time  = ti
                ts.value = vi
                ms.series_list[idx] = ts.clean() # remove NaNs

        else:
            raise NameError('Unknown methods; no action taken')

        return ms

    def correlation(self, target=None, timespan=None, alpha=0.05, settings=None, method='phaseran', number=1000,
                    fdr_kwargs=None, common_time_kwargs=None, mute_pbar=False, seed=None):
        ''' Calculate the correlation between a MultipleSeries and a target Series

        Parameters
        ----------
        target : pyleoclim.Series, optional
        
            The Series against which to take the correlation. If the target Series is not specified, then the 1st member of MultipleSeries will be used as the target

        timespan : tuple, optional
        
            The time interval over which to perform the calculation

        alpha : float
        
            The significance level (0.05 by default)

        settings : dict
        
            Parameters for the correlation function (per scipy)

        number : int

                the number of simulations (default: 1000)

        method : str, {'ttest', 'ar1sim', 'phaseran' (default)}

                method for significance testing

        fdr_kwargs : dict
        
            Parameters for the FDR function

        common_time_kwargs : dict
        
            Parameters for the method MultipleSeries.common_time()

        mute_pbar : bool; {True,False}
        
            If True, the progressbar will be muted. Default is False.
        
        seed : float or int
        
            random seed for isopersistent and isospectral methods

        Returns
        -------
        corr : CorrEns
        
            the result object

        See also
        --------

        pyleoclim.utils.correlation.corr_sig : Correlation function

        pyleoclim.utils.correlation.fdr : FDR function
        
        pyleoclim.core.correns.CorrEns : the correlation ensemble object

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            from pyleoclim.utils.tsmodel import colored_noise
            import numpy as np

            nt = 100
            t0 = np.arange(nt)
            v0 = colored_noise(alpha=1, t=t0)
            noise = np.random.normal(loc=0, scale=1, size=nt)

            ts0 = pyleo.Series(time=t0, value=v0, verbose=False)
            ts1 = pyleo.Series(time=t0, value=v0+noise, verbose=False)
            ts2 = pyleo.Series(time=t0, value=v0+2*noise, verbose=False)
            ts3 = pyleo.Series(time=t0, value=v0+1/2*noise, verbose=False)

            ts_list = [ts1, ts2, ts3]

            ms = pyleo.MultipleSeries(ts_list)
            ts_target = ts0

        Correlation between the MultipleSeries object and a target Series. We also set an arbitrary random seed to ensure reproducibility:
       
        .. jupyter-execute::

            corr_res = ms.correlation(ts_target, number=20, seed=2333)
            print(corr_res)
        
        Correlation among the series of the MultipleSeries object
        
        .. jupyter-execute::

            corr_res = ms.correlation(number=20, seed=2333)
            print(corr_res)

        '''
        r_list = []
        signif_list = []
        p_list = []

        if target is None:
            target = self.series_list[0]

        print("Looping over "+ str(len(self.series_list)) +" Series in collection")
        for idx, ts in tqdm(enumerate(self.series_list),  total=len(self.series_list), disable=mute_pbar):
            corr_res = ts.correlation(target, timespan=timespan, alpha=alpha, settings=settings,
                                      method=method, number=number,
                                      common_time_kwargs=common_time_kwargs, seed=seed)
            r_list.append(corr_res.r)
            signif_list.append(corr_res.signif)
            p_list.append(corr_res.p)

        r_list = np.array(r_list)
        signif_fdr_list = []
        fdr_kwargs = {} if fdr_kwargs is None else fdr_kwargs.copy()
        args = {}
        args.update(fdr_kwargs)
        for i in range(np.size(signif_list)):
            signif_fdr_list.append(False)

        fdr_res = corrutils.fdr(p_list, **fdr_kwargs)
        if fdr_res is not None:
            for i in fdr_res:
                signif_fdr_list[i] = True

        corr_ens = CorrEns(r_list, p_list, signif_list, signif_fdr_list, alpha)
        return corr_ens


    def equal_lengths(self):
        ''' Test whether all series in object have equal length

        Returns
        -------
        flag : bool
        
            Whether or not the Series in the pyleo.MultipleSeries object are of equal length

        lengths : list 
        
            List of the lengths of the series in object
            
        See also
        --------
        
        pyleoclim.core.multipleseries.MultipleSeries.common_time : Aligns the time axes of a MultipleSeries object
            
        Examples
        --------    

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            flag, lengths = ms.equal_lengths()
            print(flag)

        '''

        lengths = []
        for ts in self.series_list:
            lengths.append(len(ts.value))

        L = lengths[0]
        r = lengths[1:]
        flag = all(l==L for l in r)

        return flag, lengths

    def pca(self,weights=None, name=None, missing='fill-em',tol_em=5e-03, max_em_iter=100,**pca_kwargs):
        '''Principal Component Analysis (Empirical Orthogonal Functions)

        Decomposition of MultipleSeries in terms of orthogonal basis functions.
        Tolerant to missing values, infilled by an EM algorithm.

        Do make sure the time axes are aligned, however! (e.g. use `common_time()`)

        Algorithm from statsmodels: https://www.statsmodels.org/stable/generated/statsmodels.multivariate.pca.PCA.html

        Parameters
        ----------
        weights : ndarray, optional
        
            Series weights to use after transforming data according to standardize
            or demean when computing the principal components.

        missing : {str, None}
        
            Method for missing data.  Choices are:

            * 'drop-row' - drop rows with missing values.
            * 'drop-col' - drop columns with missing values.
            * 'drop-min' - drop either rows or columns, choosing by data retention.
            * 'fill-em' - use EM algorithm to fill missing value.  ncomp should be
              set to the number of factors required.
            * `None` raises if data contains NaN values.

        tol_em : float
        
            Tolerance to use when checking for convergence of the EM algorithm.
            
        max_em_iter : int
        
            Maximum iterations for the EM algorithm.

        Returns
        -------
        res: MultivariateDecomp

            Resulting pyleoclim.MultivariateDecomp object
        
        See also
        --------
        
        pyleoclim.utils.tsutils.eff_sample_size : Effective Sample Size of timeseries y

        pyleoclim.core.multivardecomp.MultivariateDecomp : The spatial decomposition object
        
        pyleoclim.core.mulitpleseries.MulitpleSeries.common_time : align time axes
        
        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = (soi & nino).common_time()
            ms.label = ms.series_list[0].label
            res = ms.pca() # carry out PCA

            fig1, ax1 = res.screeplot() # plot the eigenvalue spectrum
            fig2, ax2 = res.modeplot() # plot the first mode

        '''
        flag, lengths = self.equal_lengths()

        if flag==False:
            print('All Time Series should be of same length. Apply common_time() first')
        else: # if all series have equal length
            p = len(lengths)
            n = lengths[0]
            ys = np.empty((n,p))
            for j in range(p):
                ys[:,j] = self.series_list[j].value  # fill in data matrix

            #nc = min(ys.shape) # number of components to return

            out  = PCA(ys,weights=weights,missing=missing,tol_em=tol_em, max_em_iter=max_em_iter,**pca_kwargs)

            # compute effective sample size
            PC1 = out.factors[:,0]
            neff = tsutils.eff_sample_size(PC1)

            # compute percent variance
            pctvar = out.eigenvals/np.sum(out.eigenvals)*100

            # assign name
            if name is not None:
                name_str = name + ' PCA'
            elif self.label is not None:
                name_str = self.label + ' PCA'
            else:
                name_str = 'PCA of unlabelled object'
            # assign result to MultivariateDecomp class
            res = MultivariateDecomp(name=name_str, neff= neff,
                                pcs = out.scores, pctvar = pctvar, eigvals = out.eigenvals,
                                eigvecs = out.eigenvecs, orig=self)
            return res

    def bin(self, **kwargs):
        '''Aligns the time axes of a MultipleSeries object, via binning.

        This is critical for workflows that need to assume a common time axis
        for the group of series under consideration.

        The common time axis is characterized by the following parameters:

        start : the latest start date of the bunch (maximin of the minima)

        stop  : the earliest stop date of the bunch (minimum of the maxima)

        step  : The representative spacing between consecutive values (mean of the median spacings)

        This is a special case of the common_time function.

        Parameters
        ----------
        kwargs : dict
        
            Arguments for the binning function. See pyleoclim.utils.tsutils.bin

        Returns
        -------
        ms : MultipleSeries
        
            The MultipleSeries objects with all series aligned to the same time axis.

        See also
        --------

        pyleoclim.core.multipleseries.MultipleSeries.common_time: Base function on which this operates

        pyleoclim.utils.tsutils.bin: Underlying binning function

        pyleoclim.core.series.Series.bin: Bin function for Series object

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            msbin = ms.bin()

        '''

        ms = self.copy()

        ms = ms.common_time(method = 'bin', **kwargs)

        return ms

    def gkernel(self, **kwargs):
        ''' Aligns the time axes of a MultipleSeries object, via Gaussian kernel.
        
        This is critical for workflows that need to assume a common time axis
        for the group of series under consideration.

        The common time axis is characterized by the following parameters:

        start : the latest start date of the bunch (maximin of the minima)

        stop  : the earliest stop date of the bunch (minimum of the maxima)

        step  : The representative spacing between consecutive values (mean of the median spacings)

        This is a special case of the common_time function.

        Parameters
        ----------
        kwargs : dict
        
            Arguments for gkernel. See pyleoclim.utils.tsutils.gkernel for details.

        Returns
        -------
        ms : MultipleSeries
            The MultipleSeries objects with all series aligned to the same time axis.

        See also
        --------

        pyleoclim.core.multipleseries.MultipleSeries.common_time: Base function on which this operates

        pyleoclim.utils.tsutils.gkernel: Underlying kernel module

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            msk = ms.gkernel()

        '''

        ms = self.copy()

        ms = ms.common_time(method = 'gkernel', **kwargs)

        return ms

    def interp(self, **kwargs):
        ''' Aligns the time axes of a MultipleSeries object, via interpolation.
        
        This is critical for workflows that need to assume a common time axis
        for the group of series under consideration.


        The common time axis is characterized by the following parameters:

        start : the latest start date of the bunch (maximin of the minima)

        stop  : the earliest stop date of the bunch (minimum of the maxima)

        step  : The representative spacing between consecutive values (mean of the median spacings)

        This is a special case of the common_time function.

        Parameters
        ----------
        kwargs: keyword arguments (dictionary) for the interpolation method

        Returns
        -------
        ms : MultipleSeries
        
            The MultipleSeries objects with all series aligned to the same time axis.

        See also
        --------

        pyleoclim.core.multipleseries.MultipleSeries.common_time: Base function on which this operates

        pyleoclim.utils.tsutils.interp: Underlying interpolation function

        pyleoclim.core.series.Series.interp: Interpolation function for Series object

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            msinterp = ms.interp()

        '''
        ms = self.copy()

        ms = ms.common_time(method='interp', **kwargs)

        return ms

    def detrend(self,method='emd',**kwargs):
        '''Detrend timeseries

        Parameters
        ----------
        method : str, optional
        
            The method for detrending. The default is 'emd'.
            Options include:
                * linear: the result of a linear least-squares fit to y is subtracted from y.
                * constant: only the mean of data is subtrated.
                * 'savitzky-golay', y is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
                * 'emd' (default): Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
                
        **kwargs : dict
            Relevant arguments for each of the methods.

        Returns
        -------
        ms : MultipleSeries
        
            The detrended timeseries

        See also
        --------

        pyleoclim.core.series.Series.detrend : Detrending for a single series

        pyleoclim.utils.tsutils.detrend : Detrending function
        '''
        ms=self.copy()
        for idx,item in enumerate(ms.series_list):
            s=item.copy()
            v_mod, _=tsutils.detrend(item.value,x=item.time,method=method,**kwargs)
            s.value=v_mod
            ms.series_list[idx]=s
        return ms

    def spectral(self, method='lomb_scargle', freq=None, settings=None, mute_pbar=False, 
                freq_kwargs=None, label=None, verbose=False, scalogram_list=None):
        ''' Perform spectral analysis on the timeseries

        Parameters
        ----------
        method : str; {'wwz', 'mtm', 'lomb_scargle', 'welch', 'periodogram', 'cwt'}

        freq : str or array, optional
           Information to produce the frequency vector. 
           This can be 'log','scale', 'nfft', 'lomb_scargle' or 'welch' or a NumPy array.
           If a string, will use make_freq_vector with the specified frequency-generating method.
           If an array, this will be passed directly to the spectral method.
           If None (default), will use 'log' for WWZ and 'lomb_scargle' for Lomb-Scargle. 
           This parameter is highly consequential for the WWZ and Lomb-Scargle methods, 
           but is otherwise ignored, as other spectral methods generate their frequency vector internally.

        freq_kwargs : dict
        
            Arguments for frequency vector

        settings : dict
        
            Arguments for the specific spectral method

        label : str
        
            Label for the PSD object

        verbose : bool
        
            If True, will print warning messages if there is any

        mute_pbar : bool
        
            Mute the progress bar. Default is False.

        scalogram_list : pyleoclim.MultipleScalogram
        
            Multiple scalogram object containing pre-computed scalograms to use when calculating spectra, only works with wwz or cwt

        Returns
        -------
        psd : MultiplePSD
        
            A Multiple PSD object

        See also
        --------
        
        pyleoclim.utils.spectral.mtm : Spectral analysis using the Multitaper approach

        pyleoclim.utils.spectral.lomb_scargle : Spectral analysis using the Lomb-Scargle method

        pyleoclim.utils.spectral.welch: Spectral analysis using the Welch segement approach

        pyleoclim.utils.spectral.periodogram: Spectral anaysis using the basic Fourier transform

        pyleoclim.utils.spectral.wwz_psd : Spectral analysis using the Wavelet Weighted Z transform

        pyleoclim.utils.spectral.cwt_psd : Spectral analysis using the continuous Wavelet Transform as implemented by Torrence and Compo

        pyleoclim.utils.wavelet.make_freq_vector : Functions to create the frequency vector

        pyleoclim.utils.tsutils.detrend : Detrending function

        pyleoclim.core.series.Series.spectral : Spectral analysis for a single timeseries

        pyleoclim.core.PSD.PSD : PSD object

        pyleoclim.core.psds.MultiplePSD : Multiple PSD object

        Examples
        --------

        .. jupyter-execute::

            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            ms_psd = ms.spectral(method='mtm')
            ms_psd.plot()
                    
        '''
        settings = {} if settings is None else settings.copy()

        psd_list = []
        if method in ['wwz','cwt'] and scalogram_list:
            scalogram_list_len = len(scalogram_list.scalogram_list)
            series_len = len(self.series_list)

            #In the case where the scalogram list and series list are the same we can re-use scalograms in a one to one fashion
            #OR if the scalogram list is longer than the series list we use as many scalograms from the scalogram list as we need
            if scalogram_list_len >= series_len:
                for idx, s in enumerate(tqdm(self.series_list, desc='Performing spectral analysis on individual series', position=0, leave=True, disable=mute_pbar)):
                    psd_tmp = s.spectral(method=method, settings=settings, freq=freq, freq_kwargs=freq_kwargs, label=label, verbose=verbose,scalogram = scalogram_list.scalogram_list[idx])
                    psd_list.append(psd_tmp)
            #If the scalogram list isn't as long as the series list, we re-use all the scalograms we can and then calculate the rest
            elif scalogram_list_len < series_len:
                for idx, s in enumerate(tqdm(self.series_list, desc='Performing spectral analysis on individual series', position=0, leave=True, disable=mute_pbar)):
                    if idx < scalogram_list_len:
                        psd_tmp = s.spectral(method=method, settings=settings, freq=freq, freq_kwargs=freq_kwargs, label=label, verbose=verbose,scalogram = scalogram_list.scalogram_list[idx])
                        psd_list.append(psd_tmp)
                    else:
                        psd_tmp = s.spectral(method=method, settings=settings, freq=freq, freq_kwargs=freq_kwargs, label=label, verbose=verbose)
                        psd_list.append(psd_tmp)
        else:
            for s in tqdm(self.series_list, desc='Performing spectral analysis on individual series', position=0, leave=True, disable=mute_pbar):
                psd_tmp = s.spectral(method=method, settings=settings, freq=freq, freq_kwargs=freq_kwargs, label=label, verbose=verbose)
                psd_list.append(psd_tmp)

        psds = MultiplePSD(psd_list=psd_list)

        return psds

    def wavelet(self, method='cwt', settings={}, freq_method='log', freq_kwargs=None, verbose=False, mute_pbar=False):
        '''Wavelet analysis

        Parameters
        ----------
        method : str {wwz, cwt}
        
            - cwt - the continuous wavelet transform (as per Torrence and Compo [1998])
                is appropriate only for evenly-spaced series.
            - wwz - the weighted wavelet Z-transform (as per Foster [1996])
                is appropriate for both evenly and unevenly-spaced series.

            Default is cwt, returning an error if the Series is unevenly-spaced.

        settings : dict, optional
        
            Settings for the particular method. The default is {}.

        freq_method : str; {'log', 'scale', 'nfft', 'lomb_scargle', 'welch'}

        freq_kwargs : dict
        
            Arguments for frequency vector

        settings : dict
        
            Arguments for the specific spectral method

        verbose : bool
        
            If True, will print warning messages if there is any

        mute_pbar : bool, optional
        
            Whether to mute the progress bar. The default is False.

        Returns
        -------
        scals : MultipleScalograms
        
            A Multiple Scalogram object

        See also
        --------
        
        pyleoclim.utils.wavelet.wwz : wwz function

        pyleoclim.utils.wavelet.cwt : cwt function

        pyleoclim.utils.wavelet.make_freq_vector : Functions to create the frequency vector

        pyleoclim.utils.tsutils.detrend : Detrending function

        pyleoclim.core.series.Series.wavelet : wavelet analysis on single object

        pyleoclim.core.scalograms.MultipleScalogram : Multiple Scalogram object

        References
        ----------

        Torrence, C. and G. P. Compo, 1998: A Practical Guide to Wavelet Analysis. Bull. Amer. Meteor. Soc., 79, 61-78.
        Python routines available at http://paos.colorado.edu/research/wavelets/
        

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = (soi & nino)
            wav = ms.wavelet(method='wwz')

        '''
        settings = {} if settings is None else settings.copy()

        scal_list = []
        for s in tqdm(self.series_list, desc='Performing wavelet analysis on individual series', position=0, leave=True, disable=mute_pbar):
            scal_tmp = s.wavelet(method=method, settings=settings, freq_method=freq_method, freq_kwargs=freq_kwargs, verbose=verbose)
            scal_list.append(scal_tmp)

        scals = MultipleScalogram(scalogram_list=scal_list)

        return scals

    def plot(self, figsize=[10, 4],
             marker=None, markersize=None,
             linestyle=None, linewidth=None, colors=None, cmap='tab10', norm=None,
             xlabel=None, ylabel=None, title=None, time_unit = None,
             legend=True, plot_kwargs=None, lgd_kwargs=None,
             savefig_settings=None, ax=None, invert_xaxis=False):

        '''Plot multiple timeseries on the same axis

        Parameters
        ----------
        figsize : list, optional
        
            Size of the figure. The default is [10, 4].
            
        marker : str, optional
        
            Marker type. The default is None.
            
        markersize : float, optional
        
            Marker size. The default is None.
            
        linestyle : str, optional
        
            Line style. The default is None.
            
        linewidth : float, optional
        
            The width of the line. The default is None.
            
        colors : a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)
        
            Colors for plotting.
            If None, the plotting will cycle the 'tab10' colormap;
            if only one color is specified, then all curves will be plotted with that single color;
            if a list of colors are specified, then the plotting will cycle that color list.
            
        cmap : str
        
            The colormap to use when "colors" is None.
            
        norm : matplotlib.colors.Normalize
       
            The normalization for the colormap.
            If None, a linear normalization will be used.
            
        xlabel : str, optional
        
            x-axis label. The default is None.
            
        ylabel : str, optional
        
            y-axis label. The default is None.
            
        title : str, optional
        
            Title. The default is None.
            
        time_unit : str
        
            the target time unit, possible input:
            {
                'year', 'years', 'yr', 'yrs',
                'y BP', 'yr BP', 'yrs BP', 'year BP', 'years BP',
                'ky BP', 'kyr BP', 'kyrs BP', 'ka BP', 'ka',
                'my BP', 'myr BP', 'myrs BP', 'ma BP', 'ma',
            }
            default is None, in which case the code picks the most common time unit in the collection.
            If no unambiguous winner can be found, the unit of the first series in the collection is used. 
            
        legend : bool, optional
        
            Whether the show the legend. The default is True.
            
        plot_kwargs : dict, optional
        
            Plot parameters. The default is None.
            
        lgd_kwargs : dict, optional
        
            Legend parameters. The default is None.
            
        savefig_settings : dictionary, optional
        
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.
            
        ax : matplotlib.ax, optional
        
            The matplotlib axis onto which to return the figure. The default is None.
            
        invert_xaxis : bool, optional
        
            if True, the x-axis of the plot will be inverted

        Returns
        -------
        fig : matplotlib.figure
        
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html) for details.

        ax : matplotlib.axis
        
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        See also
        --------

        pyleoclim.utils.plotting.savefig : Saving figure in Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            ms.name = 'ENSO'
            fig, ax = ms.plot()

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        # deal with time units
        self = self.convert_time_unit(time_unit=time_unit)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        if title is None and self.label is not None:
            ax.set_title(self.label)

        if ylabel is None:
            consistent_ylabels = True
            time_label, value_label = self.series_list[0].make_labels()
            for s in self.series_list[1:]:
                time_label_tmp, value_label_tmp = s.make_labels()
                if value_label_tmp != value_label:
                    consistent_ylabels = False

            if consistent_ylabels:
                ylabel = value_label
            else:
                ylabel = 'value'

        for idx, s in enumerate(self.series_list):
            if colors is None:
                cmap_obj = plt.get_cmap(cmap)
                if hasattr(cmap_obj, 'colors'):
                    nc = len(cmap_obj.colors)
                else:
                    nc = len(self.series_list)

                if norm is None:
                    norm = mpl.colors.Normalize(vmin=0, vmax=nc-1)

                clr = cmap_obj(norm(idx%nc))
            elif type(colors) is str:
                clr = colors
            elif type(colors) is list:
                nc = len(colors)
                clr = colors[idx%nc]
            else:
                raise TypeError('"colors" should be a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)')

            ax = s.plot(
                figsize=figsize, marker=marker, markersize=markersize, color=clr, linestyle=linestyle,
                linewidth=linewidth, label=s.label, xlabel=xlabel, ylabel=ylabel, title=title,
                legend=legend, lgd_kwargs=lgd_kwargs, plot_kwargs=plot_kwargs, ax=ax,
            )

        if invert_xaxis:
            ax.invert_xaxis()

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax

    def stackplot(self, figsize=None, savefig_settings=None, time_unit = None, 
                  xlim=None, fill_between_alpha=0.2, colors=None, cmap='tab10', 
                  norm=None, labels='auto', ylabel_fontsize = 8, spine_lw=1.5, grid_lw=0.5,
                  label_x_loc=-0.15, v_shift_factor=3/4, linewidth=1.5, plot_kwargs=None):
        ''' Stack plot of multiple series

        Time units are harmonized prior to plotting. 
        Note that the plotting style is uniquely designed for this one and cannot be properly reset with `pyleoclim.set_style()`.

        Parameters
        ----------
        figsize : list
        
            Size of the figure.
            
        savefig_settings : dictionary
        
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.
            
        time_unit : str
        
            the target time unit, possible inputs:
            {
                'year', 'years', 'yr', 'yrs',
                'y BP', 'yr BP', 'yrs BP', 'year BP', 'years BP',
                'ky BP', 'kyr BP', 'kyrs BP', 'ka BP', 'ka',
                'my BP', 'myr BP', 'myrs BP', 'ma BP', 'ma',
            }
            default is None, in which case the code picks the most common time unit in the collection.
            If no discernible winner can be found, the unit of the first series in the collection is used. 
            
        xlim : list
        
            The x-axis limit.
            
        fill_between_alpha : float
        
            The transparency for the fill_between shades.
            
        colors : a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)
        
            Colors for plotting.
            If None, the plotting will cycle the 'tab10' colormap;
            if only one color is specified, then all curves will be plotted with that single color;
            if a list of colors are specified, then the plotting will cycle that color list.
            
        cmap : str
        
            The colormap to use when "colors" is None.
            
        norm : matplotlib.colors.Normalize like
        
            The normalization for the colormap.
            If None, a linear normalization will be used.
            
        labels: None, 'auto' or list
        
            If None, doesn't add labels to the subplots
            If 'auto', uses the labels passed during the creation of pyleoclim.Series
            If list, pass a list of strings for each labels.
            Default is 'auto'
            
        spine_lw : float
        
            The linewidth for the spines of the axes.
            
        grid_lw : float
        
            The linewidth for the gridlines.
            
        label_x_loc : float
        
            The x location for the label of each curve.
            
        v_shift_factor : float
        
            The factor for the vertical shift of each axis.
            The default value 3/4 means the top of the next axis will be located at 3/4 of the height of the previous one.
            
        linewidth : float
        
            The linewidth for the curves.
            
        ylabel_fontsize : int
            
            Size for ylabel font. Default is 8, to avoid crowding. 
            
        plot_kwargs: dict or list of dict
        
            Arguments to further customize the plot from matplotlib.pyplot.plot.

            - Dictionary: Arguments will be applied to all lines in the stackplots
            - List of dictionaries: Allows to customize one line at a time.

        Returns
        -------
        fig : matplotlib.figure
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html) for details.

        ax : matplotlib.axis
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        See also
        --------

        pyleoclim.utils.plotting.savefig : Saving figure in Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            fig, ax = ms.stackplot()

        Let's change the labels on the left

        .. jupyter-execute::

            fig, ax = ms.stackplot(labels=['SOI','NINO3'])

        And let's remove them completely

        .. jupyter-execute::

            fig, ax = ms.stackplot(labels=None)

        Now, let's add markers to the timeseries.

        .. jupyter-execute::

            fig, ax = ms.stackplot(labels=None, plot_kwargs={'marker':'o'})

        Using different marker types on each series:

        .. jupyter-execute::

            fig, ax = ms.stackplot(labels=None, plot_kwargs=[{'marker':'o'},{'marker':'^'}])

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        n_ts = len(self.series_list)

        if type(labels)==list:
            if len(labels) != n_ts:
                raise ValueError("The length of the label list should match the number of timeseries to be plotted")
        
        # deal with time units
        self = self.convert_time_unit(time_unit=time_unit)

        # Deal with plotting arguments
        if type(plot_kwargs)==dict:
            plot_kwargs = [plot_kwargs]*n_ts

        if plot_kwargs is not None and len(plot_kwargs) != n_ts:
            raise ValueError("When passing a list of dictionaries for kwargs arguments, the number of items should be the same as the number of timeseries")


        fig = plt.figure(figsize=figsize)
        
        if xlim is None:
            time_min = np.inf
            time_max = -np.inf
            for ts in self.series_list:
                if np.min(ts.time) <= time_min:
                    time_min = np.min(ts.time)
                if np.max(ts.time) >= time_max:
                    time_max = np.max(ts.time)
            xlim = [time_min, time_max]

        ax = {}
        left = 0
        width = 1
        height = 1/n_ts
        bottom = 1
        for idx, ts in enumerate(self.series_list):
            if colors is None:
                cmap_obj = plt.get_cmap(cmap)
                if hasattr(cmap_obj, 'colors'):
                    nc = len(cmap_obj.colors)
                else:
                    nc = len(self.series_list)

                if norm is None:
                    norm = mpl.colors.Normalize(vmin=0, vmax=nc-1)

                clr = cmap_obj(norm(idx%nc))
            elif type(colors) is str:
                clr = colors
            elif type(colors) is list:
                nc = len(colors)
                clr = colors[idx%nc]
            else:
                raise TypeError('"colors" should be a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)')
            #deal with other plotting arguments
            if plot_kwargs is None:
                p_kwargs = {}
            else:
                p_kwargs = plot_kwargs[idx]

            bottom -= height*v_shift_factor
            ax[idx] = fig.add_axes([left, bottom, width, height])
            ax[idx].plot(ts.time, ts.value, color=clr, lw=linewidth,**p_kwargs)
            ax[idx].patch.set_alpha(0)
            ax[idx].set_xlim(xlim)
            time_label, value_label = ts.make_labels()
            ax[idx].set_ylabel(value_label, weight='bold', size=ylabel_fontsize)

            mu = np.nanmean(ts.value)
            std = np.nanstd(ts.value)
            ylim = [mu-4*std, mu+4*std]
            ax[idx].fill_between(ts.time, ts.value, y2=mu, alpha=fill_between_alpha, color=clr)
            trans = transforms.blended_transform_factory(ax[idx].transAxes, ax[idx].transData)
            if labels == 'auto':
                if ts.label is not None:
                    ax[idx].text(label_x_loc, mu, ts.label, horizontalalignment='right', transform=trans, color=clr, weight='bold')
            elif type(labels) ==list:
                ax[idx].text(label_x_loc, mu, labels[idx], horizontalalignment='right', transform=trans, color=clr, weight='bold')
            elif labels==None:
                pass
            ax[idx].set_ylim(ylim)
            ax[idx].set_yticks(ylim)
            ax[idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax[idx].grid(False)
            if idx % 2 == 0:
                ax[idx].spines['left'].set_visible(True)
                ax[idx].spines['left'].set_linewidth(spine_lw)
                ax[idx].spines['left'].set_color(clr)
                ax[idx].spines['right'].set_visible(False)
                ax[idx].yaxis.set_label_position('left')
                ax[idx].yaxis.tick_left()
            else:
                ax[idx].spines['left'].set_visible(False)
                ax[idx].spines['right'].set_visible(True)
                ax[idx].spines['right'].set_linewidth(spine_lw)
                ax[idx].spines['right'].set_color(clr)
                ax[idx].yaxis.set_label_position('right')
                ax[idx].yaxis.tick_right()

            ax[idx].yaxis.label.set_color(clr)
            ax[idx].tick_params(axis='y', colors=clr)
            ax[idx].spines['top'].set_visible(False)
            ax[idx].spines['bottom'].set_visible(False)
            ax[idx].tick_params(axis='x', which='both', length=0)
            ax[idx].set_xlabel('')
            ax[idx].set_xticklabels([])
            xt = ax[idx].get_xticks()[1:-1]
            for x in xt:
                ax[idx].axvline(x=x, color='lightgray', linewidth=grid_lw, ls='-', zorder=-1)
            ax[idx].axhline(y=mu, color='lightgray', linewidth=grid_lw, ls='-', zorder=-1)

        # subplots_height = 1-height*(1-v_shift_factor)
        # ax['subplots_canvas'] = fig.add_axes([left, bottom, width, subplots_height],
        #                                      **{'zorder':-1})
        # ax['subplots_canvas'].spines['left'].set_visible(False)
        # ax['subplots_canvas'].spines['right'].set_visible(False)
        # ax['subplots_canvas'].spines['bottom'].set_visible(False)
        # ax['subplots_canvas'].set_yticks([])
        # ax['subplots_canvas'].set_xlim(xlim)
        # ax['subplots_canvas'].tick_params(axis='x', which='both', length=0)
        #
        # ax['subplots_canvas'].set_xlabel('')
        # ax['subplots_canvas'].set_ylabel('')
        # ax['subplots_canvas'].set_xticklabels([])
        # ax['subplots_canvas'].set_yticklabels([])
        # ax['subplots_canvas'].grid(False)

        bottom -= height*(1-v_shift_factor)
        # other subplots are set inside the subplot that controls the time axis
        # trying to make that time axis subplot the whole size of the figure

        x_axis_key = 'x_axis'
        # x_axis_key = n_ts

        ax[x_axis_key] = fig.add_axes([left, bottom, width, height])
        ax[x_axis_key].set_xlabel(time_label)
        ax[x_axis_key].spines['left'].set_visible(False)
        ax[x_axis_key].spines['right'].set_visible(False)
        ax[x_axis_key].spines['bottom'].set_visible(True)
        ax[x_axis_key].spines['bottom'].set_linewidth(spine_lw)
        ax[x_axis_key].set_yticks([])
        ax[x_axis_key].patch.set_alpha(0)
        ax[x_axis_key].set_xlim(xlim)
        ax[x_axis_key].grid(False)
        ax[x_axis_key].tick_params(axis='x', which='both', length=3.5)
        xt = ax[x_axis_key].get_xticks()[1:-1]
        for x in xt:
            ax[x_axis_key].axvline(x=x, color='lightgray', linewidth=grid_lw,
                                   ls='-', zorder=-1)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax
        
    def stripes(self, cmap = 'RdBu_r', sat=1.0, ref_period=None,
                figsize=None, savefig_settings=None,  time_unit=None,
                labels='auto',  label_color = 'gray', show_xaxis=False,
                common_time_kwargs=None, xlim=None, font_scale=0.8, x_offset = 0.05):
        '''
        Represents a MultipleSeries object as a quilt of Ed Hawkins' "stripes" patterns
        
        To ensure comparability, constituent series are placed on a common time axis, using
        `MultipleSeries.common_time()`. To ensure consistent scaling, all series are Gaussianized
        prior to plotting. 
    
        Credit: https://showyourstripes.info/,
        Implementation: https://matplotlib.org/matplotblog/posts/warming-stripes/

        Parameters
        ----------
        cmap: str
            colormap name (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
            Default is 'RdBu_r'
            
        ref_period : TYPE, optional
            dates of the reference period, in the form "(first, last)".
            The default is None, which will pick the beginning and end of the common time axis.
        
        figsize : list
            a list of two integers indicating the figure size (in inches)
        
        sat : float > 0
            Controls the saturation of the colormap normalization by scaling the vmin, vmax in https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
            default = 1.0
            
        show_xaxis : bool
            flag indicating whether or not the x-axis should be shown (default = False) 
            
        savefig_settings : dictionary
            the dictionary of arguments for plt.savefig(); some notes below:

            - 'path' must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in 'path', it will follow 'format'
            - 'format' can be one of {"pdf", 'eps', 'png', ps'} The default is None.
            
        time_unit : str
            the target time unit, possible inputs:
            {
                'year', 'years', 'yr', 'yrs',
                'y BP', 'yr BP', 'yrs BP', 'year BP', 'years BP',
                'ky BP', 'kyr BP', 'kyrs BP', 'ka BP', 'ka',
                'my BP', 'myr BP', 'myrs BP', 'ma BP', 'ma',
            }
            default is None, in which case the code picks the most common time unit in the collection.
            If no discernible winner can be found, the unit of the first series in the collection is used. 
                
        xlim : list
            The x-axis limit.
            
        x_offset : float
            value controlling the horizontal offset between stripes and labels (default = 0.05)          
            
        labels: None, 'auto' or list
        
            If None, doesn't add labels to the subplots

            If 'auto', uses the labels passed during the creation of pyleoclim.Series

            If list, pass a list of strings for each labels.
            Default is 'auto'
            
        common_time_kwargs : dict
            Optional arguments for common_time()
            
        font_scale : float
            The scale for the font sizes. Default is 0.8.   

        Returns
        -------
        fig : matplotlib.figure
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/stable/api/figure_api.html) for details.

        ax : matplotlib.axis
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/stable/api/axes_api.html) for details.
            
        See also
        --------
        
        pyleoclim.core.multipleseries.MultipleSeries.common_time : aligns the time axes of a MultipleSeries object

        pyleoclim.utils.plotting.savefig : saving a figure in Pyleoclim
        
        pyleoclim.core.series.Series.stripes : stripes representation in Pyleoclim   
        
        pyleoclim.utils.tsutils.gaussianize : mapping to a standard Normal distribution
            
        Examples
        --------

        .. jupyter-execute::

            co2ts = pyleo.utils.load_dataset('AACO2')
            lr04 = pyleo.utils.load_dataset('LR04')
            edc = pyleo.utils.load_dataset('EDC-dD')
            ms = lr04.flip() & edc & co2ts # create MS object
            fig, ax = ms.stripes()
             
        The default style has rather thick bands, intense colors, and too many stripes.
        The first issue can be solved by passing a figsize tuple; the second by increasing the LIM parameter; 
        the third by passing a step of 0.5 (500y) to common_time(). Finally, the 
        labels are too close to the edge of the plot, which can be adjusted with x_offset, like so:  

        .. jupyter-execute::

             co2ts = pyleo.utils.load_dataset('AACO2')
             lr04 = pyleo.utils.load_dataset('LR04')
             edc = pyleo.utils.load_dataset('EDC-dD')
             ms = lr04.flip() & edc & co2ts # create MS object
             fig, ax = ms.stripes(figsize=(8,2.5),show_xaxis=True, sat = 0.8)

        '''
        current_style = deepcopy(mpl.rcParams)
        plotting.set_style('journal', font_scale=font_scale)
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        common_time_kwargs = {} if common_time_kwargs is None else common_time_kwargs.copy()
        
        if len(self.series_list)>20:
            warnings.warn("You are trying to plot over 20 series; results will be hard to see",
                          UserWarning, stacklevel=2)
        
        # deal with time units
        self = self.convert_time_unit(time_unit=time_unit)

        # put on common timescale
        msc = self.common_time(**common_time_kwargs)
        
        ts0 = msc.series_list[0]
        time = ts0.time
        # generate default axis labels
        time_label, _ = ts0.make_labels()
           
        if ref_period is None:
            ref_period = [time.min(), time.max()]
        
        n_ts = len(msc.series_list)
        last = n_ts-1

        if n_ts < 2:
            raise ValueError("There is only one series in this object. Please use the Series class instead")

        if type(labels)==list:
            if len(labels) != n_ts:
                raise ValueError("The length of the label list should match the number of timeseries to be plotted")

        fig, axs = plt.subplots(n_ts, 1, sharex=True, figsize=figsize, layout = 'tight')
        ax = axs.flatten()

        for idx in range(n_ts-1):  # loop over series
            ts = msc.series_list[idx]
            ts.stripes(ref_period, sat=sat, cmap= cmap,
                       label_color = label_color,
                       ax=ax[idx], x_offset=x_offset) 
            
        # handle bottom plot
        ts = msc.series_list[last]
        ts.stripes(ref_period, sat=sat, cmap=cmap,
                   label_color = label_color, show_xaxis=show_xaxis,
                   ax=ax[last], x_offset=x_offset) 
        
        if xlim is None:
            xlim = [time.min(), time.max()]
        ax[last].set_xlim(xlim)
        
        fig.tight_layout()

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            mpl.rcParams.update(current_style)
            return fig, ax
        else:
            # reset the plotting style
            mpl.rcParams.update(current_style)
            return ax

    def to_pandas(self, paleo_style=False, *args, use_common_time=False, **kwargs):
        """
        Align Series and place in DataFrame.

        Column names will be taken from each Series' label. 

        Parameters
        ----------
        paleo_style : boolean, optional
            If True, will format datetime as the common time vector and assign as 
            index name the time_name of the first series in the object. 
            
        *args, **kwargs
            Arguments and keyword arguments to pass to ``common_time``.
        use_common_time, bool
            Pass True if you want to use ``common_time`` to align the Series
            to have common times. Else, times for which some Series doesn't
            have values will be filled with NaN (default).
         
        Returns
        -------
        pandas.DataFrame

        """
        if use_common_time:
            ms = self.common_time(*args, **kwargs)
        else:
            ms = self
        
        df = pd.DataFrame({ser.metadata['label']: ser.to_pandas(paleo_style=paleo_style) for ser in ms.series_list})
        if paleo_style:
            tl = ms.series_list[0].time_name
            df.index.name = tl if tl is not None else 'time' 
        return df
        
    def to_csv(self, path = None, *args, use_common_time=False,  **kwargs):
        '''
        Export MultipleSeries to CSV

        Parameters
        ----------
        path : str, optional
            system path to save the file. The default is None, in which case the filename defaults to the poetic 'MultipleSeries.csv' in the current directory.
        *args, **kwargs
            Arguments and keyword arguments to pass to ``common_time``.
        use_common_time, bool
            Set to True if you want to use ``common_time`` to align the Series
            to a common timescale. Else, times for which some Series don't
            have values will be filled with NaN (default).
        Returns
        -------
        None.
    
        Examples
        --------

        This will place the NINO3 and SOI datasets into a MultipleSeries object and export it to enso.csv.

        .. jupyter-execute::

            import pyleoclim as pyleo
            soi = pyleo.utils.load_dataset('SOI')
            nino = pyleo.utils.load_dataset('NINO3')
            ms = soi & nino
            ms.label = 'enso'
            ms.to_csv()

        '''
        if path is None:  
            path = self.label.split('.')[0].replace(" ", "_") + '.csv' if self.label is not None else 'MultipleSeries.csv' 
            
        self.to_pandas(paleo_style=True, *args,
                       use_common_time=use_common_time,
                       **kwargs).to_csv(path, header = True)
    
    def sel(self, value=None, time=None, tolerance=0):
        '''
        Slice MulitpleSeries based on 'value' or 'time'. See examples in pyleoclim.series.Series for usage. 

        Parameters
        ----------
        value : int, float, slice
            If int/float, then the Series will be sliced so that `self.value` is
            equal to `value` (+/- `tolerance`).
            If slice, then the Series will be sliced so `self.value` is between
            slice.start and slice.stop (+/- tolerance).
        time : int, float, slice
            If int/float, then the Series will be sliced so that `self.time` is
            equal to `time`. (+/- `tolerance`)
            If slice of int/float, then the Series will be sliced so that
            `self.time` is between slice.start and slice.stop.
            If slice of `datetime` (or str containing datetime, such as `'2020-01-01'`),
            then the Series will be sliced so that `self.datetime_index` is
            between `time.start` and `time.stop` (+/- `tolerance`, which needs to be
            a `timedelta`).
        tolerance : int, float, default 0.
            Used by `value` and `time`, see above.

        Returns
        -------
        ms_new : pyleoclim.mulitpleseries.MultipleSeries
            Copy of `self`, sliced according to `value` and `time`.
            
        See also
        --------
        
        pyleoclim.series.Series.sel : Slicing a series by `value` and `time`. 

        '''
        
        if value is not None:
            warnings.warn('You are selecting by values. Make sure the units are consistent across all timeseries or that they have been standardized')
            
        #loop it
        
        new_list = []
        
        for item in self.series_list:
            new_list.append(item.sel(value=value,time=time,tolerance=tolerance))
        
        ms_new = self.copy()
        ms_new.series_list=new_list
        
        return ms_new
    
    def to_json(self, path=None):
        '''
        Export the pyleoclim.MultipleSeries object to a json file

        Parameters
        ----------
        path : string, optional
            The path to the file. The default is None, resulting in a file saved in the current working directory using the label for the dataset as filename if available or 'mulitpleseries.json' if label is not provided.

        Returns
        -------
        None.
        
        '''
        
        if path is None:        
            path = self.series_list[0].label.replace(" ", "_") + '.json' if self.series_list[0].label is not None else 'multipleseries.json' 
        
        jsonutils.PyleoObj_to_json(self, path)
    
    @classmethod    
    def from_json(cls, path):
        ''' Creates a pyleoclim.MulitpleSeries from a JSON file
        
        The keys in the JSON file must correspond to the parameter associated with MulitpleSeries and Series objects

        Parameters
        ----------
        path : str
            Path to the JSON file

        Returns
        -------
        ts : pyleoclim.core.series.MulitplesSeries
            A Pyleoclim MultipleSeries object. 

        '''
        
        a = jsonutils.open_json(path)
        b = jsonutils.iterate_through_dict(a, 'MultipleSeries')
        
        return cls(**b)

    def time_coverage_plot(self, figsize=[10, 3],
             marker=None, markersize=None, alpha = .8,
             linestyle=None, linewidth=10, colors=None, cmap='turbo',
             norm=None, xlabel=None, ylabel=None, title=None, time_unit = None,
             legend=True, inline_legend=True, plot_kwargs=None, lgd_kwargs=None,
             label_x_offset=200,label_y_offset=0,savefig_settings=None, ax=None, ypad=None,
             invert_xaxis=False, invert_yaxis=False):
        '''A plot of the temporal coverage of the records in a MultipleSeries object organized by ranked length.

        Inspired by Dr. Mara Y. McPartland.
        
        Parameters
        ----------
        figsize : list, optional
        
            Size of the figure. The default is [10, 4].
            
        marker : str, optional
        
            Marker type. The default is None.
            
        markersize : float, optional
        
            Marker size. The default is None.

        alpha : float, optional
        
            Alpha of the lines
            
        linestyle : str, optional
        
            Line style. The default is None.
            
        linewidth : float, optional
        
            The width of the line. The default is 10.
            
        colors : a list of, or one, Python supported color code (a string of hex code or a tuple of rgba values)
        
            Colors for plotting.
            If None, the plotting will cycle the 'viridis' colormap;
            if only one color is specified, then all curves will be plotted with that single color;
            if a list of colors are specified, then the plotting will cycle that color list.
            
        cmap : str
        
            The colormap to use when "colors" is None. Default is 'turbo'
            
        norm : matplotlib.colors.Normalize
       
            The normalization for the colormap.
            If None, a linear normalization will be used.
            
        xlabel : str, optional
        
            x-axis label. The default is None.
            
        ylabel : str, optional
        
            y-axis label. The default is None.
            
        title : str, optional
        
            Title. The default is None.
            
        time_unit : str
        
            the target time unit, possible input:
            {
                'year', 'years', 'yr', 'yrs',
                'y BP', 'yr BP', 'yrs BP', 'year BP', 'years BP',
                'ky BP', 'kyr BP', 'kyrs BP', 'ka BP', 'ka',
                'my BP', 'myr BP', 'myrs BP', 'ma BP', 'ma',
            }
            default is None, in which case the code picks the most common time unit in the collection.
            If no unambiguous winner can be found, the unit of the first series in the collection is used. 
            
        legend : bool, optional
        
            Whether the show the legend. The default is True.

        inline_legend : bool, optional

            Whether to use inline labels or the default pyleoclim legend. This option overrides lgd_kwargs
            
        plot_kwargs : dict, optional
        
            Plot parameters. The default is None.
            
        lgd_kwargs : dict, optional
        
            Legend parameters. The default is None.

            If inline_legend is True, lgd_kwargs will be passed to ax.text() (see matplotlib.axes.Axes.text documentation)
            If inline_legend is False, lgd_kwargs will be passed to ax.legend() (see matplotlib.axes.Axes.legend documentation)

        label_x_offset: float or list, optional

            Amount to offset label by in the x direction. Only used if inline_legend is True. Default is 200.
            If list, should have the same number of elements as the MultipleSeries object.

        label_y_offset : float or list, optional

            Amount to offset label by in the y direction. Only used if inline_legend is True. Default is 0.
            If list, should have the same number of elements as the MultipleSeries object.
            
        savefig_settings : dictionary, optional
        
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"} The default is None.
            
        ax : matplotlib.ax, optional
        
            The matplotlib axis onto which to return the figure. The default is None.
            
        invert_xaxis : bool, optional
        
            if True, the x-axis of the plot will be inverted

        invert_yaxis : bool, optional
        
            if True, the y-axis of the plot will be inverted

        Returns
        -------
        fig : matplotlib.figure
        
            the figure object from matplotlib
            See [matplotlib.pyplot.figure](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html) for details.

        ax : matplotlib.axis
        
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        See also
        --------

        pyleoclim.utils.plotting.savefig : Saving figure in Pyleoclim

        Examples
        --------

        .. jupyter-execute::

            import pyleoclim as pyleo

            co2ts = pyleo.utils.load_dataset('AACO2')
            lr04 = pyleo.utils.load_dataset('LR04')
            edc = pyleo.utils.load_dataset('EDC-dD')
            ms = lr04.flip() & edc & co2ts # create MS object
            fig, ax = ms.time_coverage_plot(label_y_offset=-.08) #Fiddling with label offsets is sometimes necessary for aesthetic

        Awkward vertical spacing can be adjusted by varying linewidth and figure size
        
        .. jupyter-execute::

            import pyleoclim as pyleo
            
            co2ts = pyleo.utils.load_dataset('AACO2')
            lr04 = pyleo.utils.load_dataset('LR04')
            edc = pyleo.utils.load_dataset('EDC-dD')
            ms = lr04.flip() & edc & co2ts # create MS object
            fig, ax = ms.time_coverage_plot(linewidth=20,figsize=[10,2],label_y_offset=-.1)

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        # deal with time units
        self = self.convert_time_unit(time_unit=time_unit)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        if title is None and self.label is not None:
            ax.set_title(self.label, fontweight='bold')

        if ylabel is None:
            ylabel = 'Length Rank'
            
        sorted_series_list = list(dict(sorted({max(series.time)-min(series.time):series for series in self.series_list}.items())).values())

        for idx, s in enumerate(sorted_series_list):
            if colors is None:
                cmap_obj = plt.get_cmap(cmap)
                nc = len(self.series_list)
                if norm is None:
                    norm = mpl.colors.Normalize(vmin=0, vmax=nc-1)
                clr = cmap_obj(norm(idx%nc))
            elif type(colors) is str:
                clr = colors
            elif type(colors) is list:
                nc = len(colors)
                clr = colors[idx%nc]
            else:
                raise TypeError("'colors' should be a list of, or one of, Python's supported color codes (a string of hex code or a tuple of rgba values)")
            
            s.value = np.ones(len(s.value))*(idx+1)

            if legend and inline_legend:
            
                ax = s.plot(
                    figsize=figsize, marker=marker, markersize=markersize, alpha=alpha, color=clr, linestyle=linestyle,
                    linewidth=linewidth, label=s.label, xlabel=xlabel, ylabel=ylabel, title=title,
                    legend=False, lgd_kwargs=None, plot_kwargs=plot_kwargs, ax=ax,
                )

                if isinstance(label_x_offset,list):
                    x_offset = label_x_offset[idx]
                else:
                    x_offset=label_x_offset
                if isinstance(label_y_offset,list):
                    y_offset = label_y_offset[idx]
                else:
                    y_offset=label_y_offset

                ax.text(s.time[-1]+x_offset, s.value[-1]+y_offset, s.label, color=clr, **lgd_kwargs)

            else:
                
                ax = s.plot(
                    figsize=figsize, marker=marker, markersize=markersize, alpha=alpha, color=clr, linestyle=linestyle,
                    linewidth=linewidth, label=s.label, xlabel=xlabel, ylabel=ylabel, title=title,
                    legend=legend, lgd_kwargs=lgd_kwargs, plot_kwargs=plot_kwargs, ax=ax,
                )

        #Don't need the y-axis for these plots, can just remove it.
        ax.get_yaxis().set_visible(False)
        ax.spines[['left']].set_visible(False)

        #Increase padding to minimize cutoff likelihood.
        ylim = ax.get_ylim()
        ax.set_ylim([0.5,ylim[-1]+.2])

        if invert_xaxis:
            ax.invert_xaxis()

        if invert_yaxis:
            ax.invert_yaxis()

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax
        
    def resolution(self,time_unit=None,verbose=True,statistic='median'):
        """Generate a MultipleResolution object

        Increments are assigned to the preceding time value.
        E.g. for time_axis = [0,1,3], resolution.resolution = [1,2] resolution.time = [0,1].
        Note that the MultipleResolution class requires a shared time unit. If the time_unit parameter is not passed, a time unit will be automatically determined.

        Returns
        -------
        multipleresolution : pyleoclim.MultipleResolution
            MultipleResolution object

        time_unit : str
            Time unit to convert objects to. See pyleo.Series.convert_time_unit for options.

        verbose : bool
            Whether or not to print messages warning the user about automated decisions.

        statistic : str; {'median','mean',None}
            If a recognized statistic is passed, this function will simply output that statistic applied to the resolution of each series in the MulitipleSeries object. Options are 'mean' or 'median'.
            If statistic is None, then the function will return a new MultipleResolution class with plotting capabilities.

        See also
        --------
        
        pyleoclim.core.resolutions.MultipleResolution

        pyleoclim.core.series.Series.convert_time_unit

        Examples
        --------

        To create a resolution object, apply the .resolution() method to a Series object with `statistic=None`.

        .. jupyter-execute::

            import pyleoclim as pyleo

            co2ts = pyleo.utils.load_dataset('AACO2')
            edc = pyleo.utils.load_dataset('EDC-dD')
            ms = edc & co2ts # create MS object
            ms_resolution = ms.resolution(statistic=None)

        Several methods are then available:

        Summary statistics can be obtained via .describe()

        .. jupyter-execute::

            ms_resolution.describe()

        A simple plot can be created using .summary_plot()

        .. jupyter-execute::

            ms_resolution.summary_plot()
            """
         
        if statistic=='median':
            warnings.warn('The statistic parameter will be deprecated in a future release. Statistic = None will become the default behavior.',DeprecationWarning)
            res = [np.median(ts.resolution().resolution) for ts in self.series_list]
        elif statistic=='mean':
            warnings.warn('The statistic parameter will be deprecated in a future release. Statistic = None will become the default behavior.',DeprecationWarning)
            res = [np.mean(ts.resolution().resolution) for ts in self.series_list]
        elif statistic is None:
            resolution_list = []

            if time_unit:
                series_list = self.series_list
                for series in series_list:
                    resolution = series.convert_time_unit(time_unit).resolution()
                    resolution_list.append(resolution)
            else:
                if self.time_unit:
                    series_list = self.series_list
                    for series in series_list:
                        resolution = series.resolution()
                        resolution_list.append(resolution)
                else:
                    if verbose:
                        print('Time unit not found, attempting conversion.')
                    new_ms = self.convert_time_unit()
                    time_unit = new_ms.time_unit
                    series_list = new_ms.series_list
                    if verbose:
                        print(f'Converted to {time_unit}')
                    for series in series_list:
                        resolution = series.resolution()
                        resolution_list.append(resolution)

            res = MultipleResolution(resolution_list=resolution_list,time_unit=time_unit)
        else:
            raise ValueError('Unrecognized statistic, please use "mean", "median", or None')

        return res