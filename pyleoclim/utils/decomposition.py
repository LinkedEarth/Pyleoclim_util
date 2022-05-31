#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigendecomposition methods:
Singular Spectrum Analysis (SSA). 
soon: Monte-Carlo Principal Component Analysis, Multi-Channel SSA
"""

__all__ = [
    'ssa'
]

import numpy as np
#from sklearn.decomposition import PCA
#from statsmodels.multivariate.pca import PCA
from .tsutils import standardize
from .tsmodel import ar1_sim
from scipy.linalg import eigh, toeplitz
#from nitime import algorithms as alg
#import copy

#------
# Main functions
#------

# def mcpca(ys, nMC=200, **pca_kwargs):
#     '''Monte-Carlo Principal Component Analysis

#     Carries out Principal Component Analysis  (most unfortunately named EOF analysis in the meteorology
#     and climate literature) on a data matrix ys.

#     The significance of eigenvalues is gauged against those of AR(1) surrogates fit to the data.

#     TODO: enable for ensembles and generally debug

#     Parameters
#     ----------
#     ys : 2D numpy array (nt, nrec)
#         nt   = number of time samples (assumed identical for all records)
#         nrec = number of records (aka variables, channels, etc)

#     nMC : int
#         the number of Monte-Carlo simulations

#     pca_kwargs : tuple
#         keyword arguments for the PCA method

#     Returns
#     -------
#     res : dict containing:

#         - eigvals : eigenvalues (nrec,)

#         - eigvals95 : eigenvalues of the AR(1) ensemble (nrec, nMC)

#         - pcs : PC series of all components (nt, rec)

#         - eofs : EOFs of all components (nrec, nrec)

#     References:
#     ----------
#     Deininger, M., McDermott, F., Mudelsee, M. et al. (2017): Coherency of late Holocene
#     European speleothem δ18O records linked to North Atlantic Ocean circulation.
#     Climate Dynamics, 49, 595–618. https://doi.org/10.1007/s00382-016-3360-8

#     Written by Jun Hu (Rice University).
#     Adapted for pyleoclim by Julien Emile-Geay (USC)
#     '''
#     nt, nrec = ys.shape

#     ncomp = min(nt,nrec)

#     pc_mc = np.zeros((nt,nrec)) # principal components
#     eof_mc = np.zeros((nrec,nrec))  #eof (spatial loadings)
#     #eigenvalue matrices
#     eigvals = np.zeros((nrec))
#     eig_ar1 = np.zeros((nrec,nMC))

#     # apply PCA algorithm to the data matrix
#     pca_res = PCA(ys,ncomp=ncomp, **pca_kwargs)
#     eigvals = pca_res.eigenvals

#     # generate surrogate matrix
#     y_ar1 = np.full((nt,nrec,nMC), 0, dtype=np.double)

#     for i in range(nrec):
#         yi = copy.deepcopy(ys[:,i])
#         # generate nMC AR(1) surrogates
#         y_ar1[:,i,:] = ar1_sim(yi, nMC)
#         # assign PC and EOF matrices
#         if pc.loadings[:,i][0]>0:
#             eof_mc[:,i]  = pc.loadings[:,i]
#             pc_mc[:,i]   = pc.factors[:,i]
#         else:   # flip sign (arbitrary)
#             eof_mc[:,i]  = -pc.loadings[:,i]
#             pc_mc[:,i]   = -pc.factors[:,i]
#         # estimate effective sample size
#         #PC1 =
#         neff[i] = tsutils.eff_sample_size(PC1)

#     # loop over Monte Carlo iterations
#     for m in range(nMC):
#         pc_ar1 = PCA(y_ar1[:,:,m],ncomp=nrec,**pca_kwargs)
#         eig_ar1[:,m] = pc_ar1.eigenvals

#     eig95 = np.percentile(eig_ar1, 95, axis=1)


#     # assign result
#     #res = {'eigvals': eigvals, 'eigvals95': eig95, 'pcs': pc_mc, 'eofs': eof_mc}

#     # compute effective sample size
#     PC1  = out.factors[:,0]
#     neff = tsutils.eff_sample_size(PC1)

#     # compute percent variance
#     pctvar = out.eigenvals**2/np.sum(out.eigenvals**2)*100

#     # assign result to SpatiamDecomp class
#     # Note: need to grab coordinates from Series or LiPDSeries
#     res = SpatialDecomp(name='PCA', time = self.series_list[0].time, neff= neff,
#                         pcs = out.scores, pctvar = pctvar,  locs = None,
#                         eigvals = out.eigenvals, eigvecs = out.eigenvecs)

#     return res


# def mssa(ys, M=None, nMC=0, f=0.3):
#     '''Multi-channel singular spectrum analysis analysis

#     Multivariate generalization of SSA [2], using the original algorithm of [1].
#     Each variable is called a channel, hence the name.

#     Parameters
#     ----------

#     ys : array
#           multiple time series (dimension: length of time series x total number of time series)

#     M : int
#        window size (embedding dimension, default: 10% of the length of the series)

#     nMC : int
#        Number of iteration in the Monte-Carlo process [default=0, no Monte Carlo process]

#     f : float
#        fraction (0<f<=1) of good data points for identifying significant PCs [f = 0.3]

#     Returns
#     -------
#     res : dict
#         Containing:

#         - eigvals : array of eigenvalue spectrum

#         - eigvals05 : The 5% percentile of eigenvalues

#         - eigvals95 : The 95% percentile of eigenvalues

#         - PC : matrix of principal components (2D array)

#         - RC : matrix of RCs (nrec,N,nrec*M) (2D array)

#     References
#     ----------
#     [1]_ Vautard, R., and M. Ghil (1989), Singular spectrum analysis in nonlinear
#     dynamics, with applications to paleoclimatic time series, Physica D, 35,
#     395–424.

#     [2]_ Jiang, N., J. D. Neelin, and M. Ghil (1995), Quasi-quadrennial and
#     quasi-biennial variability in the equatorial Pacific, Clim. Dyn., 12, 101-112.

#     See Also
#     --------

#     pyleoclim.utils.decomposition.ssa : Singular Spectrum Analysis (single channel)

#     '''
#     N = len(ys[:, 0])
#     nrec = len(ys[0, :])
#     if M == None:
#         M=int(N/10)
#     Y = np.zeros((N - M + 1, nrec * M))
#     for irec in np.arange(nrec):
#         for m in np.arange(0, M):
#             Y[:, m + irec * M] = ys[m:N - M + 1 + m, irec]

#     C = np.dot(np.nan_to_num(np.transpose(Y)), np.nan_to_num(Y)) / (N - M + 1)
#     D, eigvecs = eigh(C)

#     sort_tmp = np.sort(D)
#     eigvals = sort_tmp[::-1]
#     sortarg = np.argsort(-D)

#     eigvecs = eigvecs[:, sortarg]

#     # test the signifiance using Monte-Carlo
#     Ym = np.zeros((N - M + 1, nrec * M))
#     noise = np.zeros((nrec, N, nMC))
#     for irec in np.arange(nrec):
#         noise[irec, 0, :] = ys[0, irec]
#     eigvals_R = np.zeros((nrec * M, nMC))
#     # estimate coefficents of ar1 processes, and then generate ar1 time series (noise)
#     # TODO : update to use ar1_sim(), as in ssa()
#     for irec in np.arange(nrec):
#         Xs = ys[:, irec]
#         coefs_est, var_est = alg.AR_est_YW(Xs[~np.isnan(Xs)], 1)
#         sigma_est = np.sqrt(var_est)

#         for jt in range(1, N):
#             noise[irec, jt, :] = coefs_est * noise[irec, jt - 1, :] + sigma_est * np.random.randn(1, nMC)

#     for m in range(nMC):
#         for irec in np.arange(nrec):
#             noise[irec, :, m] = (noise[irec, :, m] - np.mean(noise[irec, :, m])) / (
#                 np.std(noise[irec, :, m], ddof=1))
#             for im in np.arange(0, M):
#                 Ym[:, im + irec * M] = noise[irec, im:N - M + 1 + im, m]
#         Cn = np.dot(np.nan_to_num(np.transpose(Ym)), np.nan_to_num(Ym)) / (N - M + 1)
#         # eigvals_R[:,m] = np.diag(np.dot(np.dot(eigvecs,Cn),np.transpose(eigvecs)))
#         eigvals_R[:, m] = np.diag(np.dot(np.dot(np.transpose(eigvecs), Cn), eigvecs))

#     eigvals95 = np.percentile(eigvals_R, 95, axis=1)
#     eigvals05 = np.percentile(eigvals_R, 5, axis=1)


#     # determine principal component time series
#     PC = np.zeros((N - M + 1, nrec * M))
#     PC[:, :] = np.nan
#     for k in np.arange(nrec * M):
#         for i in np.arange(0, N - M + 1):
#             #   modify for nan
#             prod = Y[i, :] * eigvecs[:, k]
#             ngood = sum(~np.isnan(prod))
#             #   must have at least m*f good points
#             if ngood >= M * f:
#                 PC[i, k] = sum(prod[~np.isnan(prod)])  # the columns of this matrix are Ak(t), k=1 to M (T-PCs)

#     # compute reconstructed timeseries
#     Np = N - M + 1

#     RC = np.zeros((nrec, N, nrec * M))

#     for k in np.arange(nrec):
#         for im in np.arange(M):
#             x2 = np.dot(np.expand_dims(PC[:, im], axis=1), np.expand_dims(eigvecs[0 + k * M:M + k * M, im], axis=0))
#             x2 = np.flipud(x2)

#             for n in np.arange(N):
#                 RC[k, n, im] = np.diagonal(x2, offset=-(Np - 1 - n)).mean()
#     res = {'eigvals': eigvals, 'eigvecs': eigvecs, 'q05': eigvals05, 'q95': eigvals95, 'PC': PC, 'RC': RC}

#     return res

def ssa(y, M=None, nMC=0, f=0.5, trunc=None, var_thresh = 80):
    '''Singular spectrum analysis

    Nonparametric eigendecomposition of timeseries into orthogonal oscillations.
    This implementation  uses the method of [1], with applications presented in [2].
    Optionally (nMC>0), the significance of eigenvalues is assessed by Monte-Carlo simulations of an AR(1) model fit to X, using [3].
    The method expects regular spacing, but is tolerant to missing values, up to a fraction 0<f<1 (see [4]).

    Parameters
    ----------

    y : array of length N
          time series (evenly-spaced, possibly with up to f*N NaNs)

    M : int
       window size (default: 10% of N)

    nMC : int
        Number of iterations in the Monte-Carlo simulation (default nMC=0, bypasses Monte Carlo SSA)
        Currently only supported for evenly-spaced, gap-free data.

    f : float
        maximum allowable fraction of missing values. (Default is 0.5)

    trunc : str
        if present, truncates the expansion to a level K < M owing to one of 3 criteria:
            (1) 'kaiser': variant of the Kaiser-Guttman rule, retaining eigenvalues larger than the median
            (2) 'mcssa': Monte-Carlo SSA (use modes above the 95% quantile from an AR(1) process)
            (3) 'var': first K modes that explain at least var_thresh % of the variance.
        Default is None, which bypasses truncation (K = M)

    var_thresh : float
        variance threshold for reconstruction (only impactful if trunc is set to 'var')

    Returns
    -------
    res : dict containing:

        - eigvals : (M, ) array of eigenvalues

        - eigvecs : (M, M) Matrix of temporal eigenvectors (T-EOFs)

        - PC : (N - M + 1, M) array of principal components (T-PCs)

        - RCmat : (N,  M) array of reconstructed components

        - RCseries : (N,) reconstructed series, with mean and variance restored

        - pctvar: (M, ) array of the fraction of variance (%) associated with each mode

        - eigvals_q : (M, 2) array containting the 5% and 95% quantiles of the Monte-Carlo eigenvalue spectrum [ if nMC >0 ]

        - mode_idx : array of indices of eigenvalues >=eigvals_q

    References
    ----------
    [1]_ Vautard, R., and M. Ghil (1989), Singular spectrum analysis in nonlinear
    dynamics, with applications to paleoclimatic time series, Physica D, 35,
    395–424.

    [2]_ Ghil, M., R. M. Allen, M. D. Dettinger, K. Ide, D. Kondrashov, M. E. Mann,
    A. Robertson, A. Saunders, Y. Tian, F. Varadi, and P. Yiou (2002),
    Advanced spectral methods for climatic time series, Rev. Geophys., 40(1),
    1003–1052, doi:10.1029/2000RG000092.

    [3]_ Allen, M. R., and L. A. Smith (1996), Monte Carlo SSA: Detecting irregular
    oscillations in the presence of coloured noise, J. Clim., 9, 3373–3404.

    [4]_ Schoellhamer, D. H. (2001), Singular spectrum analysis for time series with
    missing data, Geophysical Research Letters, 28(16), 3187–3190, doi:10.1029/2000GL012698.

    See also
    --------
    
    pyleoclim.core.series.Series.ssa : Singular Spectrum Analysis for timeseries objects
    
    pyleoclim.core.ssares.SsaRes.modeplot : plot SSA modes
    
    pyleoclim.core.ssares.SsaRes.screeplot : plot SSA eigenvalue spectrum

    '''

    ys, mu, scale = standardize(y)

    N = len(ys)

    if M == None:
        M=int(N/10)
    c = np.zeros(M)

    for j in range(M):
        prod = ys[0:N - j] * ys[j:N]
        c[j] = sum(prod[~np.isnan(prod)]) / (sum(~np.isnan(prod)) - 1)


    C = toeplitz(c[0:M])  #form correlation matrix

    D, eigvecs = eigh(C) # solve eigendecomposition

    sort_tmp = np.sort(D)
    eigvals = sort_tmp[::-1]
    sortarg = np.argsort(-D)
    eigvecs = eigvecs[:, sortarg]

    # determine principal component time series
    PC = np.zeros((N - M + 1, M))
    PC[:, :] = np.nan
    for k in np.arange(M):
        for i in np.arange(0, N - M + 1):
            #   modify for nan
            prod = ys[i:i + M] * eigvecs[:, k]
            ngood = sum(~np.isnan(prod))
            #   must have at least m*f good points
            if ngood >= M * f:
                PC[i, k] = sum(
                    prod[~np.isnan(prod)]) * M / ngood  # the columns of this matrix are Ak(t), k=1 to M (T-PCs)

    pctvar = eigvals**2/np.sum(eigvals**2)*100 # percent variance

    if nMC > 0: # If Monte-Carlo SSA is requested.
        trunc == 'mcssa'
        noise = ar1_sim(ys, nMC)  # generate MC AR(1) surrogates of y
        eigvals_R = np.zeros((M,nMC)) # define eigenvalue matrix
        lgs = np.arange(-N + 1, N)

        for m in range(nMC):
            xn, _ , _ = standardize(noise[:, m]) # center and standardize
            Gn = np.correlate(xn, xn, "full")
            Gn = Gn / (N - abs(lgs))
            Cn = toeplitz(Gn[N - 1:N - 1 + M])
            eigvals_R[:, m] = np.diag(np.dot(np.dot(np.transpose(eigvecs), Cn), eigvecs))

        eigvals_q = np.empty((M,2))
        eigvals_q[:,0] = np.percentile(eigvals_R, 5, axis=1)
        eigvals_q[:,1] = np.percentile(eigvals_R, 95, axis=1)
        mode_idx = np.where(eigvals>=eigvals_q[:,1])[0]
    else:
        eigvals_q = None


    if trunc is None:
        mode_idx = np.arange(M)
    elif trunc == 'kaiser':
        mval     = np.median(eigvals) # median eigenvalues
        mode_idx = np.where(eigvals>=mval)[0]
    elif trunc == 'var':
        mode_idx = np.arange(np.argwhere(np.cumsum(pctvar)>=var_thresh)[0]+1)
    if nMC == 0 and trunc == 'mcssa':
        raise ValueError('nMC must be larger than 0 to enable MC-SSA truncation')
    elif nMC>0:
       mode_idx = np.where(eigvals>=eigvals_q[:,1])[0]
       
    nmodes = len(mode_idx)

    # compute reconstructed timeseries
    Np = N - M + 1
    RCmat = np.zeros((N, M))

    for im in range(M):
        xdum = np.dot(np.expand_dims(PC[:, im], axis=1), np.expand_dims(eigvecs[0:M, im], axis=0))
        xdum = np.flipud(xdum)

        for n in np.arange(N):
            RCmat[n, im] = np.diagonal(xdum, offset=-(Np - 1 - n)).mean()
        del xdum

    RCmat = scale*RCmat + np.tile(mu, reps=[N, M])  # restore the mean and variance

    #RCseries = scale*RCmat[:,mode_idx].sum(axis=1) + mu

    RCseries = RCmat[:,mode_idx].sum(axis=1)
    if nmodes > 1:
        RCseries -= mu*(nmodes-1)

    # export results
    res = {'eigvals': eigvals, 'eigvecs': eigvecs, 'PC': PC, 'RCseries': RCseries, 'RCmat': RCmat, 'pctvar': pctvar, 'eigvals_q': eigvals_q, 'mode_idx': mode_idx}
    return res
