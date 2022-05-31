
# It is unclear why the documentation for these two modules does not build automatically using automodule. It therefore had to be built using autoclass

from ..utils import plotting, lipdutils
from ..utils import wavelet as waveutils

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from copy import deepcopy

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter #, MaxNLocator
from scipy.stats.mstats import mquantiles

#from ..core import MultipleScalogram
def infer_period_unit_from_time_unit(time_unit):
    ''' infer a period unit based on the given time unit

    '''
    if time_unit is None:
        period_unit = None
    else:
        unit_group = lipdutils.timeUnitsCheck(time_unit)
        if unit_group != 'unknown':
            if unit_group == 'kage_units':
                period_unit = 'kyrs'
            else:
                period_unit = 'yrs'
        else:
            if time_unit[-1] == 's':
                period_unit = time_unit
            else:
                period_unit = f'{time_unit}s'

    return period_unit


class Scalogram:
    '''
    The Scalogram class is analogous to PSD, but for wavelet spectra (scalograms).
    '''
    def __init__(self, frequency, scale, time, amplitude, coi=None, label=None, Neff_threshold=3, wwz_Neffs=None, timeseries=None,
                 wave_method=None, wave_args=None, signif_qs=None, signif_method=None, freq_method=None, freq_kwargs=None,
                 scale_unit=None, time_label=None, signif_scals=None, qs = None):
        '''
        Parameters
        ----------

            frequency : array

                The frequency axis

            scale : array

                The scale axis

            time : array

                The time axis

            amplitude : array

                The amplitude at each (frequency, time) point;
                note the dimension is assumed to be (frequency, time)

            coi : array

                Cone of influence

            label : str

                Label for the Series

            Neff_threshold : int

                The threshold of the number of effective samples

            wwz_Neffs : array

                The matrix of effective number of points in the time-scale coordinates obtained from wwz

            timeseries : pyleoclim.Series

                A copy of the timeseries for which the scalogram was obtained

            wave_method: str

                The method used to obtain the scalogram

            wave_args: dict

                The parameters values of the wavelet method

            qs : list

                Quantiles at which significance levels were evaluated & exported

            signif_qs : dict

                MultipleScalogram object containing the quantiles qs of the surrogate scalogram distribution

            signif_method: str

                The method used to obtain the significance level

            freq_method: str

                The method used to obtain the frequency vector

            freq_kwargs: dict

                Arguments for the frequency vector

            scale_unit: str

                Units for the scale axis

            time_label: str

                Label for the time axis

            signif_scals: pyleoclim.MultipleScalogram

                A list of the scalogram from the AR1 MC significance testing. Useful when obtaining a PSD.

        '''
        self.frequency = np.array(frequency)
        self.scale = np.array(scale)
        self.time = np.array(time)
        self.amplitude = np.array(amplitude)
        if coi is not None:
            self.coi = np.array(coi)
        else:
            self.coi = waveutils.make_coi(self.time, Neff_threshold=Neff_threshold)
        self.label = label
        self.timeseries = timeseries
        self.wave_method = wave_method
        if wave_args is not None:
            if 'freq' in wave_args.keys():
                wave_args['freq'] = np.array(wave_args['freq'])
            if 'tau' in wave_args.keys():
                wave_args['tau'] = np.array(wave_args['tau'])
        self.wave_args = wave_args
        self.signif_qs = signif_qs
        self.qs        = qs
        self.signif_method = signif_method
        self.freq_method = freq_method
        self.freq_kwargs = freq_kwargs
        self.signif_scals = signif_scals
        #if wave_method == 'wwz':
        if wwz_Neffs is None:
            self.wwz_Neffs = wwz_Neffs
        else:
            self.wwz_Neffs=np.array(wwz_Neffs)

        if scale_unit is not None:
            self.scale_unit = scale_unit
        elif timeseries is not None:
            self.scale_unit = infer_period_unit_from_time_unit(timeseries.time_unit)
        else:
            self.scale_unit = None

        if time_label is not None:
            self.time_label = time_label
        elif timeseries is not None:
            if timeseries.time_unit is not None:
                self.time_label = f'{timeseries.time_name} [{timeseries.time_unit}]'
            else:
                self.time_label = f'{timeseries.time_name}'
        else:
            self.time_label = None


    def copy(self):
        '''Copy object

        Returns
        -------

        scal : pyleoclim.Scalogram

            The copied version of the pyleoclim.Scalogram object

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            ts=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/soi_data.csv',skiprows = 1)
            series = pyleo.Series(time = ts['Year'],value = ts['Value'], time_name = 'Years', time_unit = 'AD')

            scalogram = series.wavelet()
            scalogram_copy = scalogram.copy()

        '''
        return deepcopy(self)

    def __str__(self):
        table = {
            'Frequency': self.frequency,
            'Time': self.time,
            'Amplitude': self.amplitude,
        }

        msg = print(tabulate(table, headers='keys'))
        return f'Dimension: {np.size(self.frequency)} x {np.size(self.time)}'

    def plot(self, variable = 'amplitude', in_scale=True, xlabel=None, ylabel=None, title='default',
             ylim=None, xlim=None, yticks=None, figsize=[10, 8],
             signif_clr='white', signif_linestyles='-', signif_linewidths=1,
             contourf_style={}, cbar_style={}, savefig_settings={}, ax=None,
             signif_thresh = 0.95):
        ''' Plot the scalogram

        Parameters
        ----------

        in_scale : bool, optional

            Plot the in scale instead of frequency space. The default is True.

        variable : {'amplitude','power'}

            Whether to plot the amplitude or power. Default is amplitude

        xlabel : str, optional

            Label for the x-axis. The default is None.

        ylabel : str, optional

            Label for the y-axis. The default is None.

        title : str, optional

            Title for the figure. The default is 'default', which auto-generates a title.

        ylim : list, optional

            Limits for the y-axis. The default is None.

        xlim : list, optional

            Limits for the x-axis. The default is None.

        yticks : list, optional

            yticks label. The default is None.

        figsize : list, optional

            Figure size The default is [10, 8].

        signif_clr : str, optional

            Color of the singificance line. The default is 'white'.

        signif_thresh: float in [0, 1]

            Significance threshold. Default is 0.95. If this quantile is not
            found in the qs field of the Coherence object, the closest quantile
            will be picked.

        signif_linestyles : str, optional

            Linestyle of the significance line. The default is '-'.

        signif_linewidths : float, optional

            Width for the significance line. The default is 1.

        contourf_style : dict, optional

            Arguments for the contour plot. The default is {}.

        cbar_style : dict, optional

            Arguments for the colarbar. The default is {}.

        savefig_settings : dict, optional

            saving options for the figure. The default is {}.

        ax : ax, optional

            Matplotlib Axis on which to return the figure. The default is None.

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

        pyleoclim.core.series.Series.wavelet : Wavelet analysis

        pyleoclim.utils.plotting.savefig : Saving figure in Pyleoclim

        Examples
        --------

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            ts=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/soi_data.csv',skiprows = 1)
            series = pyleo.Series(time = ts['Year'],value = ts['Value'], time_name = 'Years', time_unit = 'AD')

            scalogram = series.wavelet()

            @savefig scal_basic.png
            fig,ax = scalogram.plot()
            pyleo.closefig(fig)
        '''
        contourf_args = {'cmap': 'magma', 'origin': 'lower', 'levels': 11}
        contourf_args.update(contourf_style)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if in_scale:
            y_axis = 1/self.frequency
            if ylabel is None:
                ylabel = f'Scale [{self.scale_unit}]' if self.scale_unit is not None else 'Scale'

            if yticks is None:
                yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])
                mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
                yticks = yticks_default[mask]
        else:
            y_axis = self.frequency
            if ylabel is None:
                ylabel = f'Frequency [1/{self.scale_unit}]' if self.scale_unit is not None else 'Frequency'

        if variable == 'amplitude':
            cont = ax.contourf(self.time, y_axis, self.amplitude.T, **contourf_args)
        elif variable=='power':
            cont = ax.contourf(self.time, y_axis, self.amplitude.T**2, **contourf_args)
        else:
            raise ValueError('Variable should be either "amplitude" or "power"')
        ax.set_yscale('log')

        # plot colorbar
        cbar_args = {'drawedges': False, 'orientation': 'vertical', 'fraction': 0.15, 'pad': 0.05, 'label':variable.capitalize()}
        cbar_args.update(cbar_style)

        cb = plt.colorbar(cont, ax = ax, **cbar_args)

        # plot cone of influence
        if self.coi is not None:
            ax.plot(self.time, self.coi, 'k--')

        if yticks is not None:
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

        if title is not None:
            ax.set_title(title)

        if ylim is None:
            ylim = [np.min(y_axis), np.min([np.max(y_axis), np.max(self.coi)])]

        ax.fill_between(self.time, self.coi, np.max(self.coi), color='white', alpha=0.5)

        # plot significance levels
        if self.signif_qs is not None:
            signif_method_label = {
                'ar1': 'AR(1)',
            }
            if signif_thresh not in self.qs:
                isig = np.abs(np.array(self.qs) - signif_thresh).argmin()
                print("Significance threshold {:3.2f} not found in qs. Picking the closest, which is {:3.2f}".format(signif_thresh,self.qs[isig]))
            else:
                isig = self.qs.index(signif_thresh)
            signif_scal = self.signif_qs.scalogram_list[isig]
            signif_boundary = self.amplitude.T / signif_scal.amplitude.T
            ax.contour(
                self.time, y_axis, signif_boundary, [-99, 1],
                colors=signif_clr,
                linestyles=signif_linestyles,
                linewidths=signif_linewidths,
            )
            if title == 'default':
                if self.label is not None:
                    ax.set_title(self.label + " scalogram with " + str(round(self.qs[isig]*100))+"% threshold")
                else:
                    ax.set_title("Scalogram with " + str(round(self.qs[isig]*100))+"% threshold")

        if xlabel is None:
            xlabel = self.time_label

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        ax.set_ylim(ylim)

        if 'fig' in locals():
            if 'path' in savefig_settings:
                plotting.savefig(fig, settings=savefig_settings)
            return fig, ax
        else:
            return ax

    def signif_test(self, method='ar1sim', number=None, seed=None, qs=[0.95],
                    settings=None, export_scal = False):
        ''' Significance test for scalograms

        Parameters
        ----------

        method : {'ar1asym', 'ar1sim'}

            Method to use to generate the surrogates.  ar1sim uses simulated timeseries with similar persistence. 
            ar1asym represents the theoretical, closed-form solution. The default is ar1sim
            
       number: int     
            Number of surrogates to generate for significance analysis based on simulations. 
            The default is 200.

        seed : int, optional

            Set the seed for the random number generator. Useful for reproducibility The default is None.

        qs : list, optional

            Significane level to consider. The default is [0.95].

        settings : dict, optional

            Parameters for the model. The default is None.

        export_scal : bool; {True,False}

            Whether or not to export the scalograms used in the noise realizations. Note: For the wwz method, the scalograms used for wavelet analysis are slightly different
            than those used for spectral analysis (different decay constant). As such, this functionality should be used only to expedite exploratory analysis.

        Raises
        ------

        ValueError

            qs should be a list with at least one value.

        Returns
        -------

        new : pyleoclim.Scalogram

            A new Scalogram object with the significance level


        See also
        --------

        pyleoclim.core.series.Series.wavelet : Wavelet analysis

        pyleoclim.core.scalograms.MultipleScalogram : MultipleScalogram object

        pyleoclim.utils.wavelet.tc_wave_signif : Asymptotic significance calculation

        Examples
        --------

        Generating scalogram, running significance tests, and saving the output for future use in generating psd objects or in summary_plot()

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            import pandas as pd
            ts=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/master/example_data/soi_data.csv',skiprows = 1)
            series = pyleo.Series(time = ts['Year'],value = ts['Value'], time_name = 'Years', time_unit = 'AD')

        By setting export_scal to True, the noise realizations used to generate the significance test will be saved. 
        These come in handy for generating summary plots and for running significance tests on spectral objects.
        
        .. ipython:: python
            :okwarning:
            :okexcept:
                
            scalogram = series.wavelet().signif_test(number=2, export_scal=True)

        '''
        if self.wave_method == 'wwz' and method == 'ar1asym':
            raise ValueError('Asymptotic solution is not supported for the wwz method')

        if method not in ['ar1sim', 'ar1asym']:
                raise ValueError("The available methods are ar1sim'and 'ar1asym'")

        if method == 'ar1sim':

            if hasattr(self,'signif_scals'):
                signif_scals = self.signif_scals

            #Allow for a few different configurations of passed number of signif tests, default behavior is to set number = 200
            if number is None and signif_scals is not None:
                number = len(signif_scals.scalogram_list)
            elif number is None and signif_scals is None:
                number = 200
            elif number == 0:
                return self

            new = self.copy()

            if signif_scals:
                scalogram_list = signif_scals.scalogram_list
                #If signif_scals already in scalogram object are more than those requested for significance testing, use as many of them as required
                if len(scalogram_list) > number:
                    surr_scal = MultipleScalogram(scalogram_list=scalogram_list[:number])
                #If number is the same as the length of signif_scals, just use signif_scals
                elif len(scalogram_list) == number:
                    surr_scal = signif_scals
                #If the number is more than the length of signif_scals, reuse what is available and calculate the rest
                elif len(scalogram_list) < number:
                    number -= len(scalogram_list)
                    surr_scal_tmp = []
                    surr_scal_tmp.extend(scalogram_list)
                    surr = self.timeseries.surrogates(number=number, seed=seed,
                                                      method=method, settings=settings)

                    surr_scal_tmp.extend(surr.wavelet(method=self.wave_method, settings=self.wave_args).scalogram_list)
                    surr_scal = MultipleScalogram(scalogram_list=surr_scal_tmp)
            else:
                surr = self.timeseries.surrogates(number=number, seed=seed,
                                                  method=method, settings=settings)
                surr_scal = surr.wavelet(method=self.wave_method, settings=self.wave_args)

            if type(qs) is not list:
                raise TypeError('qs should be a list')

            new.signif_qs = surr_scal.quantiles(qs=qs)

            if export_scal == True:
                new.signif_scals = surr_scal

        elif method == 'ar1asym':

            new = self.copy()

            if type(qs) is not list:
                raise TypeError('qs should be a list')

            settings = {} if settings is None else settings.copy()

            signif_levels=waveutils.tc_wave_signif(self.timeseries.value,
                                                   self.timeseries.time,
                                                   self.wave_args['scale'],
                                                   self.wave_args['mother'],
                                                   self.wave_args['param'],
                                                   qs=qs, **settings)

            #Create a scalogram for each of the significance levels
            ms_base =[]
            for idx, item in enumerate(signif_levels):
                label = str(int(qs[idx]*100))+'%'
                # expand
                signif = item[:, np.newaxis].dot(np.ones(len(self.timeseries.value))[np.newaxis, :])
                s = Scalogram(frequency=self.frequency, time =self.time, scale = self.scale,
                              amplitude = signif.T, label=label)
                ms_base.append(s)

            new.signif_qs = MultipleScalogram(ms_base)


        new.signif_method = method
        new.qs = qs

        return new
    
class MultipleScalogram:
    '''MultipleScalogram objects are used to store the results of significance testing for wavelet analysis
    '''
    
    def __init__(self, scalogram_list):
        ''' Multiple Scalogram objects.
        
        This object is mainly used to store the results of wavelet significance testing in the signif_qs arguments of wavelet. 
        
        See also
        --------
        
        pyleoclim.core.scalograms.Scalogram : Scalogram object
        
        pyleoclim.core.series.Series.wavelet : Wavelet analysis
        
        pyleoclim.core.scalograms.Scalogram.signif_test : Significance testing for wavelet analysis
        
        '''
        
        self.scalogram_list = scalogram_list

    def copy(self):
        ''' Copy the object
        
        See also
        --------
        
        pyleoclim.core.scalograms.Scalogram.copy : Scalogram object copy
        
        '''
        return deepcopy(self)

    def quantiles(self, qs=[0.05, 0.5, 0.95]):
        '''Calculate quantiles

        Parameters
        ----------
        
        qs : list, optional
        
            List of quantiles to consider for the calculation. The default is [0.05, 0.5, 0.95].

        Raises
        ------
        
        ValueError
        
            Frequency axis not consistent across the PSD list!

        Value Error
        
            Time axis not consistent across the scalogram list!

        Returns
        -------
        
        scals : pyleoclim.MultipleScalogram
        
        '''
        freq = np.copy(self.scalogram_list[0].frequency)
        scale = np.copy(self.scalogram_list[0].scale)
        time = np.copy(self.scalogram_list[0].time)
        coi = np.copy(self.scalogram_list[0].coi)
        amps = []
        for scal in self.scalogram_list:
            if not np.array_equal(scal.frequency, freq):
                raise ValueError('Frequency axis not consistent across the scalogram list!')

            if not np.array_equal(scal.time, time):
                raise ValueError('Time axis not consistent across the scalogram list!')

            amps.append(scal.amplitude)

        amps = np.array(amps)
        ne, nf, nt = np.shape(amps)
        amp_qs = np.ndarray(shape=(np.size(qs), nf, nt))

        for i in range(nf):
            for j in range(nt):
                amp_qs[:,i,j] = mquantiles(amps[:,i,j], qs)

        scal_list = []
        for i, amp in enumerate(amp_qs):
            scal_tmp = Scalogram(frequency=freq, time=time, amplitude=amp,
                                 scale = scale, coi=coi, label=f'{qs[i]*100:g}%')
            scal_list.append(scal_tmp)

        scals = MultipleScalogram(scalogram_list=scal_list)
        return scals