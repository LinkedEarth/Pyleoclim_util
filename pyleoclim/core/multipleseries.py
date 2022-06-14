"""
A MultipleSeries object is a collection (more precisely, a 
list) of multiple Series objects. This is handy in case you want to apply the same method 
to such a collection at once (e.g. process a bunch of series in a consistent fashion).
"""

from ..utils import tsutils, plotting
from ..utils import correlation as corrutils

from ..core.correns import CorrEns
from ..core.scalograms import MultipleScalogram
from ..core.psds import MultiplePSD
from ..core.spatialdecomp import SpatialDecomp

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from matplotlib.ticker import FormatStrFormatter
import matplotlib.transforms as transforms
import matplotlib as mpl

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

   name : str
   
        name of the collection of timeseries (e.g. 'PAGES 2k ice cores')

    Examples
    --------
    .. ipython:: python
        :okwarning:
        :okexcept:

        import pyleoclim as pyleo
        import pandas as pd
        data = pd.read_csv(
            'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
            skiprows=0, header=1
        )
        time = data.iloc[:,1]
        value = data.iloc[:,2]
        ts1 = pyleo.Series(time=time, value=value, time_unit='years')
        ts2 = pyleo.Series(time=time, value=value, time_unit='years')
        ms = pyleo.MultipleSeries([ts1, ts2], name = 'SOI x2')
    '''
    def __init__(self, series_list, time_unit=None, name=None):
        self.series_list = series_list
        self.time_unit = time_unit
        self.name = name

        if self.time_unit is not None:
            new_ts_list = []
            for ts in self.series_list:
                new_ts = ts.convert_time_unit(time_unit=self.time_unit)
                new_ts_list.append(new_ts)

            self.series_list = new_ts_list

    def convert_time_unit(self, time_unit='years'):
        ''' Convert the time unit of the timeseries

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
        
        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv(
                'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
                skiprows=0, header=1
            )
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts1 = pyleo.Series(time=time, value=value, time_unit='years')
            ts2 = pyleo.Series(time=time, value=value, time_unit='years')
            ms = pyleo.MultipleSeries([ts1, ts2])
            new_ms = ms.convert_time_unit('yr BP')
            print('Original timeseries:')
            print('time unit:', ms.time_unit)
            print()
            print('Converted timeseries:')
            print('time unit:', new_ms.time_unit)
        '''

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

        ms : pyleoclim.core.multipleseries.MultipleSeries

        See also
        --------

        pyleoclim.series.Series.filter : filtering for Series objects    

        pyleoclim.utils.filter.butterworth : Butterworth method

        pyleoclim.utils.filter.savitzky_golay : Savitzky-Golay method

        pyleoclim.utils.filter.firwin : FIR filter design using the window method

        pyleoclim.utils.filter.lanczos : lowpass filter via Lanczos resampling

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv(
                'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
                skiprows=0, header=1
            )
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts1 = pyleo.Series(time=time, value=value, time_unit='years')
            ts2 = pyleo.Series(time=time, value=value, time_unit='years')
            ms = pyleo.MultipleSeries([ts1, ts2], name = 'SOI x2')
            ms_filter = ms.filter(method='lanczos',cutoff_scale=20)
        '''

        ms = self.copy()

        new_tslist = []

        for ts in self.series_list:
            new_tslist.append(ts.filter(cutoff_freq=cutoff_freq, cutoff_scale=cutoff_scale, method=method, **kwargs))

        ms.series_list = new_tslist

        return ms

    def append(self,ts):
        '''Append timeseries ts to MultipleSeries object

        Parameters
        ----------

        ts : pyleoclim.Series
        
            The pyleoclim Series object to be appended to the MultipleSeries object

        Returns
        -------

        ms : pyleoclim.MultipleSeries
        
            The augmented object, comprising the old one plus `ts`

        See also
        --------

        pyleoclim.core.series.Series : A Pyleoclim Series object

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv(
                'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
                skiprows=0, header=1
            )
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts1 = pyleo.Series(time=time, value=value, time_unit='years')
            ts2 = pyleo.Series(time=time, value=value, time_unit='years')
            ms = pyleo.MultipleSeries([ts1], name = 'SOI x2')
            ms.append(ts2)
        '''
        ms = self.copy()
        ts_list = deepcopy(ms.series_list)
        ts_list.append(ts)
        ms = MultipleSeries(ts_list)
        return ms

    def copy(self):
        '''Copy the object

        Returns
        -------

        ms : pyleoclim.MultipleSeries
        
            The copied version of the pyleoclim.MultipleSeries object

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv(
                'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
                skiprows=0, header=1
            )
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts1 = pyleo.Series(time=time, value=value, time_unit='years')
            ts2 = pyleo.Series(time=time, value=value, time_unit='years')
            ms = pyleo.MultipleSeries([ts1], name = 'SOI x2')
            ms_copy = ms.copy()
        '''
        return deepcopy(self)

    def standardize(self):
        '''Standardize each series object in a collection

        Returns
        -------

        ms : pyleoclim.MultipleSeries
        
            The standardized pyleoclim.MultipleSeries object

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv(
                'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
                skiprows=0, header=1
            )
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts1 = pyleo.Series(time=time, value=value, time_unit='years')
            ts2 = pyleo.Series(time=time, value=value, time_unit='years')
            ms = pyleo.MultipleSeries([ts1], name = 'SOI x2')
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
        
        step_style : str; {"median","mean,"mode","max"}
        
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

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv(
                'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
                skiprows=0, header=1
            )
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts1 = pyleo.Series(time=time, value=value, time_unit='years')
            ts2 = pyleo.Series(time=time, value=value, time_unit='years')
            ms = pyleo.MultipleSeries([ts1], name = 'SOI x2')
            increments = ms.increments()

        '''
        gp = np.empty((len(self.series_list),3)) # obtain grid parameters
        for idx,item in enumerate(self.series_list):
            item      = item.clean(verbose=verbose)
            gp[idx,:] = tsutils.increments(item.time, step_style=step_style)

        return gp

    def common_time(self, method='interp', step = None, start = None, stop = None, step_style = None, **kwargs):
        ''' Aligns the time axes of a MultipleSeries object
        
        The alignment is achieved via binning, interpolation., or Gaussian kernel. Alignment is critical for workflows
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

        kwargs: dict
        
            keyword arguments (dictionary) of the bin, gkernel or interp methods

        Returns
        -------

        ms : pyleoclim.MultipleSeries
        
            The MultipleSeries objects with all series aligned to the same time axis.


        See also
        --------

        pyleoclim.utils.tsutils.bin : put timeseries values into bins of equal size (possibly leaving NaNs in).

        pyleoclim.utils.tsutils.gkernel : coarse-graining using a Gaussian kernel

        pyleoclim.utils.tsutils.interp : interpolation onto a regular grid (default = linear interpolation)

        pyleoclim.utils.tsutils.increments : infer grid properties

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

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
                ts = pyleo.Series(time = tu, value = vu, label = 'series ' + str(j+1))
                serieslist.append(ts)

            # create MS object from the list
            ms = pyleo.MultipleSeries(serieslist)

            @savefig ms_common_time.png
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
            pyleo.closefig(fig)
        '''
        
        # specify stepping style
        if step_style == None: # if step style isn't specified, pick a robust choice according to method
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
    
        ms = self.copy()

        # apply each method
        if method == 'bin':
            for idx,item in enumerate(self.series_list):
                ts = item.copy()
                d = tsutils.bin(ts.time, ts.value, bin_size=common_step, start=start, stop=stop, evenly_spaced = False, **kwargs)
                ts.time  = d['bins']
                ts.value = d['binned_values']
                ms.series_list[idx] = ts

        elif method == 'interp':
            for idx,item in enumerate(self.series_list):
                ts = item.copy()
                ti, vi = tsutils.interp(ts.time, ts.value, step=common_step, start=start, stop=stop,**kwargs)
                ts.time  = ti
                ts.value = vi
                ms.series_list[idx] = ts

        elif method == 'gkernel':
            for idx,item in enumerate(self.series_list):
                ts = item.copy()
                ti, vi = tsutils.gkernel(ts.time,ts.value,step=common_step, start=start, stop=stop, **kwargs)
                ts.time  = ti
                ts.value = vi
                ms.series_list[idx] = ts.clean() # remove NaNs

        else:
            raise NameError('Unknown methods; no action taken')

        return ms

    def correlation(self, target=None, timespan=None, alpha=0.05, settings=None, 
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
        
            Parameters for the correlation function, including:

            nsim : int
                the number of simulations (default: 1000)
            method : str, {'ttest','isopersistent','isospectral' (default)}
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

        corr : pyleoclim.CorrEns.CorrEns
        
            the result object

        See also
        --------

        pyleoclim.utils.correlation.corr_sig : Correlation function

        pyleoclim.utils.correlation.fdr : FDR function
        
        pyleoclim.core.correns.CorrEns : the correlation ensemble object

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            from pyleoclim.utils.tsmodel import colored_noise
            import numpy as np

            nt = 100
            t0 = np.arange(nt)
            v0 = colored_noise(alpha=1, t=t0)
            noise = np.random.normal(loc=0, scale=1, size=nt)

            ts0 = pyleo.Series(time=t0, value=v0)
            ts1 = pyleo.Series(time=t0, value=v0+noise)
            ts2 = pyleo.Series(time=t0, value=v0+2*noise)
            ts3 = pyleo.Series(time=t0, value=v0+1/2*noise)

            ts_list = [ts1, ts2, ts3]

            ms = pyleo.MultipleSeries(ts_list)
            ts_target = ts0

        Correlation between the MultipleSeries object and a target Series. We also set an arbitrary random seed to ensure reproducibility:
       
        .. ipython:: python
            :okwarning:
            :okexcept:
           
            corr_res = ms.correlation(ts_target, settings={'nsim': 20}, seed=2333)
            print(corr_res)
        
        Correlation among the series of the MultipleSeries object
        
        .. ipython:: python
            :okwarning:
            :okexcept:

            corr_res = ms.correlation(settings={'nsim': 20}, seed=2333)
            print(corr_res)

        '''
        r_list = []
        signif_list = []
        p_list = []

        if target is None:
            target = self.series_list[0]

        print("Looping over "+ str(len(self.series_list)) +" Series in collection")
        for idx, ts in tqdm(enumerate(self.series_list),  total=len(self.series_list), disable=mute_pbar):
            corr_res = ts.correlation(target, timespan=timespan, alpha=alpha, settings=settings, common_time_kwargs=common_time_kwargs, seed=seed)
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

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            data = pd.read_csv(
                'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
                skiprows=0, header=1
            )
            time = data.iloc[:,1]
            value = data.iloc[:,2]
            ts1 = pyleo.Series(time=time, value=value, time_unit='years')
            ts2 = pyleo.Series(time=time, value=value, time_unit='years')
            ms = pyleo.MultipleSeries([ts1], name = 'SOI x2')
            flag, lengths = ms.equal_lengths()
            print(flag)
        '''

        lengths = []
        for ts in self.series_list:
            lengths.append(len(ts.value))

        L = lengths[0]
        r = lengths[1:]
        flag = all (l==L for l in r)

        return flag, lengths

    def pca(self,weights=None,missing='fill-em',tol_em=5e-03, max_em_iter=100,**pca_kwargs):
        '''Principal Component Analysis (Empirical Orthogonal Functions)

        Decomposition of dataset ys in terms of orthogonal basis functions.
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

        res: pyleoclim.SpatialDecomp

            Resulting pyleoclim.SpatialDecomp object
        
        See also
        --------
        
        pyleoclim.utils.tsutils.eff_sample_size : Effective Sample Size of timeseries y

        pyleoclim.core.spatialdecomp.SpatialDecomp : The spatial decomposition object
        
        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist).common_time()

            res = ms.pca() # carry out PCA

            @savefig ms_pca1.png
            fig1, ax1 = res.screeplot() # plot the eigenvalue spectrum
            pyleo.closefig(fig1)    # Optional close fig after plotting

            @savefig ms_pca2.png
            fig2, ax2 = res.modeplot() # plot the first mode
            pyleo.closefig(fig2)    # Optional close fig after plotting
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

        nc = min(ys.shape) # number of components to return

        out  = PCA(ys,weights=weights,missing=missing,tol_em=tol_em, max_em_iter=max_em_iter,**pca_kwargs)

        # compute effective sample size
        PC1  = out.factors[:,0]
        neff = tsutils.eff_sample_size(PC1)

        # compute percent variance
        pctvar = out.eigenvals**2/np.sum(out.eigenvals**2)*100

        # assign result to SpatiamDecomp class
        # Note: need to grab coordinates from Series or LiPDSeries
        res = SpatialDecomp(name='PCA', time = self.series_list[0].time, neff= neff,
                            pcs = out.scores, pctvar = pctvar,  locs = None,
                            eigvals = out.eigenvals, eigvecs = out.eigenvecs)
        return res

    # def mcpca(self,nMC=200,**pca_kwargs):
    #     ''' Monte Carlo Principal Component Analysis

    #     (UNDER REPAIR)

    #     Parameters
    #     ----------

    #     nMC : int
    #         number of Monte Carlo simulations

    #     pca_kwargs : tuple


    #     Returns
    #     -------
    #     res : dictionary containing:

    #         - eigval : eigenvalues (nrec,)
    #         - eig_ar1 : eigenvalues of the AR(1) ensemble (nrec, nMC)
    #         - pcs  : PC series of all components (nrec, nt)
    #         - eofs : EOFs of all components (nrec, nrec)

    #     References:
    #     ----------
    #     Deininger, M., McDermott, F., Mudelsee, M. et al. (2017): Coherency of late Holocene
    #     European speleothem δ18O records linked to North Atlantic Ocean circulation.
    #     Climate Dynamics, 49, 595–618. https://doi.org/10.1007/s00382-016-3360-8

    #     See also
    #     --------

    #     pyleoclim.utils.decomposition.mcpca: Monte Carlo PCA

    #     Examples
    #     --------

    #     .. ipython:: python
    #         :okwarning:

    #         import pyleoclim as pyleo
    #         url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
    #         data = pyleo.Lipd(usr_path = url)
    #         tslist = data.to_LipdSeriesList()
    #         tslist = tslist[2:] # drop the first two series which only concerns age and depth
    #         ms = pyleo.MultipleSeries(tslist)

    #         # msc = ms.common_time()

    #         # res = msc.pca(nMC=20)

    #     '''
    #     flag, lengths = self.equal_lengths()

    #     if flag==False:
    #         print('All Time Series should be of same length. Apply common_time() first')
    #     else: # if all series have equal length
    #         p = len(lengths)
    #         n = lengths[0]
    #         ys = np.empty((n,p))
    #         for j in range(p):
    #             ys[:,j] = self.series_list[j].value

    #     res = decomposition.mcpca(ys, nMC, **pca_kwargs)
    #     return res

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

        ms : pyleoclim.MultipleSeries
        
            The MultipleSeries objects with all series aligned to the same time axis.

        See also
        --------

        pyleoclim.core.multipleseries.MultipleSeries.common_time: Base function on which this operates

        pyleoclim.utils.tsutils.bin: Underlying binning function

        pyleoclim.core.series.Series.bin: Bin function for Series object

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist)
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
        ms : pyleoclim.MultipleSeries
            The MultipleSeries objects with all series aligned to the same time axis.

        See also
        --------

        pyleoclim.core.multipleseries.MultipleSeries.common_time: Base function on which this operates

        pyleoclim.utils.tsutils.gkernel: Underlying kernel module

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist)
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
        ms : pyleoclim.MultipleSeries
        
            The MultipleSeries objects with all series aligned to the same time axis.

        See also
        --------

        pyleoclim.core.multipleseries.MultipleSeries.common_time: Base function on which this operates

        pyleoclim.utils.tsutils.interp: Underlying interpolation function

        pyleoclim.core.series.Series.interp: Interpolation function for Series object

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist)
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
                * "savitzky-golay", y is filtered using the Savitzky-Golay filters and the resulting filtered series is subtracted from y.
                * "emd" (default): Empirical mode decomposition. The last mode is assumed to be the trend and removed from the series
                
        **kwargs : dict
            Relevant arguments for each of the methods.

        Returns
        -------
        
        ms : pyleoclim.MultipleSeries
        
            The detrended timeseries

        See also
        --------

        pyleoclim.core.series.Series.detrend : Detrending for a single series

        pyleoclim.utils.tsutils.detrend : Detrending function
        '''
        ms=self.copy()
        for idx,item in enumerate(ms.series_list):
            s=item.copy()
            v_mod=tsutils.detrend(item.value,x=item.time,method=method,**kwargs)
            s.value=v_mod
            ms.series_list[idx]=s
        return ms

    def spectral(self, method='lomb_scargle', settings=None, mute_pbar=False, freq_method='log', 
                freq_kwargs=None, label=None, verbose=False, scalogram_list=None):
        ''' Perform spectral analysis on the timeseries

        Parameters
        ----------

        method : str; {'wwz', 'mtm', 'lomb_scargle', 'welch', 'periodogram', 'cwt'}

        freq_method : str; {'log','scale', 'nfft', 'lomb_scargle', 'welch'}

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

        psd : pyleoclim.MultiplePSD
        
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

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist)
            ms_psd = ms.spectral()
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
                    psd_tmp = s.spectral(method=method, settings=settings, freq_method=freq_method, freq_kwargs=freq_kwargs, label=label, verbose=verbose,scalogram = scalogram_list.scalogram_list[idx])
                    psd_list.append(psd_tmp)
            #If the scalogram list isn't as long as the series list, we re-use all the scalograms we can and then calculate the rest
            elif scalogram_list_len < series_len:
                for idx, s in enumerate(tqdm(self.series_list, desc='Performing spectral analysis on individual series', position=0, leave=True, disable=mute_pbar)):
                    if idx < scalogram_list_len:
                        psd_tmp = s.spectral(method=method, settings=settings, freq_method=freq_method, freq_kwargs=freq_kwargs, label=label, verbose=verbose,scalogram = scalogram_list.scalogram_list[idx])
                        psd_list.append(psd_tmp)
                    else:
                        psd_tmp = s.spectral(method=method, settings=settings, freq_method=freq_method, freq_kwargs=freq_kwargs, label=label, verbose=verbose)
                        psd_list.append(psd_tmp)
        else:
            for s in tqdm(self.series_list, desc='Performing spectral analysis on individual series', position=0, leave=True, disable=mute_pbar):
                psd_tmp = s.spectral(method=method, settings=settings, freq_method=freq_method, freq_kwargs=freq_kwargs, label=label, verbose=verbose)
                psd_list.append(psd_tmp)

        psds = MultiplePSD(psd_list=psd_list)

        return psds

    def wavelet(self, method='cwt', settings={}, freq_method='log', freq_kwargs=None, verbose=False, mute_pbar=False):
        '''Wavelet analysis

        Parameters
        ----------
        
        method : str {wwz, cwt}
        
            cwt - the continuous wavelet transform (as per Torrence and Compo [1998])
                is appropriate only for evenly-spaced series.
            wwz - the weighted wavelet Z-transform (as per Foster [1996])
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

        scals : pyleoclim.MultipleScalograms
        
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

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only contain age and depth
            ms = pyleo.MultipleSeries(tslist)
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
             xlabel=None, ylabel=None, title=None,
             legend=True, plot_kwargs=None, lgd_kwargs=None,
             savefig_settings=None, ax=None, invert_xaxis=False):

        '''Plot multiple timeseries on the same axis

        Parameters
        ----------
        
        figsize : list, optional
        
            Size of the figure. The default is [10, 4].
            
        marker : str, optional
        
            marker type. The default is None.
            
        markersize : float, optional
        
            marker size. The default is None.
            
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
            
        legend : bool, optional
        
            Wether the show the legend. The default is True.
            
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

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            tslist = data.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist)

            @savefig ms_basic_plot.png
            fig, ax = ms.plot()
            pyleo.closefig(fig) #Optional close fig after plotting

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs.copy()
        lgd_kwargs = {} if lgd_kwargs is None else lgd_kwargs.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

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

    def stackplot(self, figsize=None, savefig_settings=None,  xlim=None, fill_between_alpha=0.2, colors=None, cmap='tab10', norm=None, labels='auto',
                  spine_lw=1.5, grid_lw=0.5, font_scale=0.8, label_x_loc=-0.15, v_shift_factor=3/4, linewidth=1.5, plot_kwargs=None):
        ''' Stack plot of multiple series

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
        
            The nomorlization for the colormap.
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
            
        font_scale : float
        
            The scale for the font sizes. Default is 0.8.
            
        label_x_loc : float
        
            The x location for the label of each curve.
            
        v_shift_factor : float
        
            The factor for the vertical shift of each axis.
            The default value 3/4 means the top of the next axis will be located at 3/4 of the height of the previous one.
            
        linewidth : float
        
            The linewidth for the curves.
            
        plot_kwargs: dict or list of dict
        
            Arguments to further customize the plot from matplotlib.pyplot.plot.
            Dictionary: Arguments will be applied to all lines in the stackplots
            List of dictionary: Allows to customize one line at a time.

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

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            d = pyleo.Lipd(usr_path = url)
            tslist = d.to_LipdSeriesList()
            tslist = tslist[2:] # drop the first two series which only concerns age and depth
            ms = pyleo.MultipleSeries(tslist)
            @savefig mts_stackplot.png
            fig, ax = ms.stackplot()
            pyleo.closefig(fig)

        Let's change the labels on the left

        .. ipython:: python
            :okwarning:
            :okexcept:

            sst = d.to_LipdSeries(number=5)
            d18Osw = d.to_LipdSeries(number=3)
            ms = pyleo.MultipleSeries([sst,d18Osw])

            @savefig mts_stackplot_customlabels.png
            fig, ax = ms.stackplot(labels=['sst','d18Osw'])
            pyleo.closefig(fig) 

        And let's remove them completely

        .. ipython:: python
            :okwarning:
            :okexcept:

            @savefig mts_stackplot_nolabels.png
            fig, ax = ms.stackplot(labels=None)
            pyleo.closefig(fig) #Optional figure close after plotting

        Now, let's add markers to the timeseries.

        .. ipython:: python
            :okwarning:
            :okexcept:

            @savefig mts_stackplot_samemarkers.png
            fig, ax = ms.stackplot(labels=None, plot_kwargs={'marker':'o'})
            pyleo.closefig(fig) #Optional figure close after plotting

        Using different marker types on each series:

        .. ipython:: python
            :okwarning:
            :okexcept:

            @savefig mts_stackplot_differentmarkers.png
            fig, ax = ms.stackplot(labels=None, plot_kwargs=[{'marker':'o'},{'marker':'^'}])
            pyleo.closefig(fig) #Optional figure close after plotting

        '''
        current_style = deepcopy(mpl.rcParams)
        plotting.set_style('journal', font_scale=font_scale)
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        n_ts = len(self.series_list)

        if type(labels)==list:
            if len(labels) != n_ts:
                raise ValueError("The length of the label list should match the number of timeseries to be plotted")

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
            ax[idx].set_ylabel(value_label, weight='bold')

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

        bottom -= height*(1-v_shift_factor)
        ax[n_ts] = fig.add_axes([left, bottom, width, height])
        ax[n_ts].set_xlabel(time_label)
        ax[n_ts].spines['left'].set_visible(False)
        ax[n_ts].spines['right'].set_visible(False)
        ax[n_ts].spines['bottom'].set_visible(True)
        ax[n_ts].spines['bottom'].set_linewidth(spine_lw)
        ax[n_ts].set_yticks([])
        ax[n_ts].patch.set_alpha(0)
        ax[n_ts].set_xlim(xlim)
        ax[n_ts].grid(False)
        ax[n_ts].tick_params(axis='x', which='both', length=3.5)
        xt = ax[n_ts].get_xticks()[1:-1]
        for x in xt:
            ax[n_ts].axvline(x=x, color='lightgray', linewidth=grid_lw, ls='-', zorder=-1)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            mpl.rcParams.update(current_style)
            return fig, ax
        else:
            plotting.showfig(fig)
            # reset the plotting style
            mpl.rcParams.update(current_style)
            return ax
