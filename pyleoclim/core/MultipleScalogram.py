"""
This module contains the MultipleScalogram object, which is used to store the results of significance testing for wavelet analysis in signif_qs
"""

from ..core import Scalogram

import numpy as np
from copy import deepcopy
from scipy.stats.mstats import mquantiles

class MultipleScalogram:
    
    def __init__(self, scalogram_list):
        ''' Multiple Scalogram objects.
        
        This object is mainly used to store the results of wavelet significance testing in the signif_qs arguments of wavelet. 
        
        See also
        --------
        
        pyleoclim.core.Scalogram.Scalogram : Scalogram object
        
        pyleoclim.core.Series.Series.wavelet : Wavelet analysis
        
        pyleoclim.core.Scalogram.Scalogram.signif_test : Significance testing for wavelet analysis
        
        '''
        
        self.scalogram_list = scalogram_list

    def copy(self):
        ''' Copy the object
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
            scal_tmp = Scalogram.Scalogram(frequency=freq, time=time, amplitude=amp,
                                 scale = scale, coi=coi, label=f'{qs[i]*100:g}%')
            scal_list.append(scal_tmp)

        scals = MultipleScalogram(scalogram_list=scal_list)
        return scals