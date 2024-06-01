"""
A MultipleEnsembleGeoSeries object is a collection (more precisely, a 
list) of EnsembleGeoSeries objects. This class currently exists primarily for the application of MC-PCA
"""

from tqdm import tqdm

import scipy as sp
import numpy as np

from ..core.series import Series
from ..core.multiplegeoseries import MultipleGeoSeries
from ..core.ensmultivardecomp import EnsMultivarDecomp

class MulEnsGeoSeries():
    def __init__(self, ensemble_series_list,label=None):
        self.ensemble_series_list = ensemble_series_list
        self.label = label

    def mcpca(self,nsim=1000, seed=None, common_time_kwargs=None, pca_kwargs=None,align_method='dot',delta=0,phase_max=.5):
        '''Function to conduct Monte-Carlo PCA using ensembles included in this object
        
        Parameters
        ----------
        nsim : int
            Number of simulations to carry out. Default is 1000
            
        seed : int
            Seed to use for random calculations
            
        common_time_kwargs : dict
            Key word arguments for MultipleSeries.common_time()
            
        pca_kwargs : dict
            Key word arguments for MultipleGeoSeries.pca()

        align_method : str; {'correlation','dot','phase','cosine'}
            How to align pcs. 
            Correlation computes the correlation between individual principal components and the first set of PCs, flipping those that have a negative correlation.
            Dot computes the dot product between the eigenvectors, and flips those that have a negative dot product.
            Phase computes the phase difference between the eigenvectors, and flips those that have a phase difference greater than phase_max.
            Cosine computes the cosine similarity between the eigenvectors, and flips those that have a negative cosine similarity.

        delta : float
            A number between -1 and 1, that serves as the threshold for the correlation, dot, and cosine methods. Default is 0.

        phase_max : float
            A float between 0 and 1, that serves as the threshold for the phase method. Default is 0.5.
            
        Returns
        -------
        EnsembleMvD : pyleo.EnsMultivarDecomp
            Ensemble Multivariate Decomposition object
            
        Examples
        --------
        
        .. jupyter-execute::
        
            n = 3 # number of ensembles
            nn = 30 # number of noise realizations
            nt = 500
            ens_list = []

            t,v = pyleo.utils.gen_ts(model='colored_noise',nt=nt,alpha=1.0)
            signal = pyleo.Series(t,v)

            for _ in range(n): 
                series_list = []
                lat = np.random.randint(-90,90)
                lon = np.random.randint(-180,180)
                for idx in range(nn):  # noise
                    noise = np.random.randn(nt,nn)*100
                    ts = pyleo.GeoSeries(time=signal.time, value=signal.value+noise[:,idx], lat=lat, lon=lon, verbose=False)
                    series_list.append(ts)

                ts_ens = pyleo.EnsembleSeries(series_list)
                ens_list.append(ts_ens)

            mul_ens = pyleo.MulEnsGeoSeries([ts_ens])
            mul_ens.mcpca(nsim=10,seed=42)'''
        
        common_time_kwargs = {} if common_time_kwargs is None else common_time_kwargs.copy()
        pca_kwargs = {} if pca_kwargs is None else pca_kwargs.copy()

        if seed:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        pca_list = []

        for i in tqdm(range(nsim),desc='Iterating over simulations'):
            ensemble_list = []
            for ensemble in self.ensemble_series_list:
                rng_index = rng.integers(low=0,high=len(ensemble.series_list))
                ensemble_list.append(ensemble.series_list[rng_index])
            mgs_tmp = MultipleGeoSeries(ensemble_list).common_time(**common_time_kwargs)
            pca_tmp = mgs_tmp.pca(**pca_kwargs)

            #Create reference pca
            if i == 0:
                base_pca = pca_tmp
                base_pcs = pca_tmp.pcs
                base_t = pca_tmp.orig.series_list[0].time
                base_eigvecs = pca_tmp.eigvecs.T
                pca_list.append(pca_tmp)
                continue

            if align_method == 'correlation':
                for idx,pcs in enumerate(pca_tmp.pcs.T):
                    t = pca_tmp.orig.series_list[0].time
                    pc_series = Series(time=t,value=pcs,verbose=False)
                    base_pc_series = Series(time=base_t,value=base_pcs[:,idx],verbose=False)
                    r = pc_series.correlation(base_pc_series,settings={'nsim':1},seed=seed,mute_pbar=True).r
                    if r > delta:
                        pass
                    else:
                        pca_tmp.eigvecs[:,idx] *= -1
                        pca_tmp.pcs[:,idx] *= -1
            elif align_method == 'phase':
                for idx,pcs in enumerate(pca_tmp.pcs.T):
                    t = pca_tmp.orig.series_list[0].time
                    pc_series = Series(time=t,value=pcs,verbose=False)
                    base_pc_series = Series(time=base_t,value=base_pcs[:,idx],verbose=False)
                    filtered_pcs = pc_series.filter(method='lanczos',cutoff_freq=.005)
                    filtered_base_pcs = base_pc_series.filter(method='lanczos',cutoff_freq=.005)
                    correlation = sp.signal.correlate(filtered_base_pcs.value, filtered_pcs.value, mode="full")
                    lags = sp.signal.correlation_lags(filtered_base_pcs.value.size, filtered_pcs.value.size, mode="full")
                    shift = np.abs(lags[np.argmax(correlation)])/(len(lags)/2)
                    # plt.plot(np.linspace(-180,180,len(correlation)),correlation)
                    if shift < phase_max:
                        pass
                    else:
                        pca_tmp.eigvecs[:,idx] *= -1
                        pca_tmp.pcs[:,idx] *= -1
            elif align_method == 'dot':
                for idx,eigvecs in enumerate(pca_tmp.eigvecs.T):
                    dot = np.dot(eigvecs,base_eigvecs[idx])
                    if dot > delta:
                        pass
                    else:
                        pca_tmp.eigvecs[:,idx] *= -1
                        pca_tmp.pcs[:,idx] *= -1
            elif align_method == 'cosine':
                for idx,eigvecs in enumerate(pca_tmp.eigvecs.T):
                    cos = sp.spatial.distance.cosine(eigvecs,base_eigvecs[idx])
                    if cos > delta:
                        pass
                    else:
                        pca_tmp.eigvecs[:,idx] *= -1
                        pca_tmp.pcs[:,idx] *= -1
            else:
                raise ValueError('Align method not recognized. Please pass "phase", "cosine", "correlation", or "dot".')

            pca_list.append(pca_tmp)

        # assign name
        if self.label is not None:
            label = self.label + ' PCA'
        else:
            label = 'PCA of unlabelled object'

        EnsembleMvD = EnsMultivarDecomp(pca_list=pca_list,label=label)
        return EnsembleMvD
