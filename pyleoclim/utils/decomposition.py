#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:29:36 2020

@author: deborahkhider
Contains decompoistion methods (PCA, SSA...)
"""

__all__ = [
    'pca',
    'ssa',
    'mssa',
]

import numpy as np
from sklearn.decomposition import PCA
from .tsutils import standardize
from .tsmodel import ar1_sim
from scipy.linalg import eigh, toeplitz
from nitime import algorithms as alg

#------
#Main functions
#------

def pca(ys,n_components=None,copy=True,whiten=False, svd_solver='auto',tol=0.0,iterated_power='auto',random_state=None):
    '''Principal Component Analysis (Empirical Orthogonal Functions)

    Decomposition of a signal or data set in terms of orthogonal basis functions.

    From scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Parameters
    ----------

    ys : array
        timeseries

    n_components : int,None,or str
         [default: None]
        Number of components to keep. if n_components is not set all components are kept:
        If n_components == 'mle' and svd_solver == 'full', Minka’s MLE is used to guess the dimension. Use of n_components == 'mle' will interpret svd_solver == 'auto' as svd_solver == 'full'.
        If 0 < n_components < 1 and svd_solver == 'full', select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.
        If svd_solver == 'arpack', the number of components must be strictly less than the minimum of n_features and n_samples.

    copy : bool,optional
        [default: True]
        If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, use fit_transform(X) instead.

    whiten : bool,optional
        [default: False]
        When True (False by default) the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.

    svd_solver : str {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
        If auto :
            The solver is selected by a default policy based on X.shape and n_components: if the input data is larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension of the data, then the more efficient ‘randomized’ method is enabled.
            Otherwise the exact full SVD is computed and optionally truncated afterwards.

        If full :
            run exact full SVD calling the standard LAPACK solver via scipy.linalg.svd and select the components by postprocessing

        If arpack :
            run SVD truncated to n_components calling ARPACK solver via scipy.sparse.linalg.svds. It requires strictly 0 < n_components < min(X.shape)

        If randomized :
            run randomized SVD by the method of Halko et al.

    tol : float >= 0 ,optional
        [default: 0]
        Tolerance for singular values computed by svd_solver == ‘arpack’.

    iterated_power : int >= 0, or string {'auto'}
        [default: 'auto']
        Number of iterations for the power method computed by svd_solver == ‘randomized’.

    random_state : int, RandomState instance, or None, optional
        [default: None]
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.
        Used when svd_solver == ‘arpack’ or ‘randomized’.

    Returns
    -------

    dict
        Sklearn PCA object dictionary of all attributes and values.

        -components_array, shape (n_components, n_features)
            Principal axes in feature space, representing the directions of maximum variance in the data. The components are sorted by explained_variance_.

        -explained_variance_array, shape (n_components,)
            The amount of variance explained by each of the selected components.
            Equal to n_components largest eigenvalues of the covariance matrix of X.
            New in version 0.18.

        -explained_variance_ratio_array, shape (n_components,)
            Percentage of variance explained by each of the selected components.
            If n_components is not set then all components are stored and the sum of the ratios is equal to 1.0.

        -singular_values_array, shape (n_components,)
            The singular values corresponding to each of the selected components. The singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.
            New in version 0.19.

        -mean_array, shape (n_features,)
            Per-feature empirical mean, estimated from the training set.
            Equal to X.mean(axis=0).

        -n_components_int
            The estimated number of components. When n_components is set to ‘mle’ or a number between 0 and 1 (with svd_solver == ‘full’) this number is estimated from input data. Otherwise it equals the parameter n_components, or the lesser value of n_features and n_samples if n_components is None.

        -n_features_int
            Number of features in the training data.

        -n_samples_int
            Number of samples in the training data.

        -noise_variance_float
            The estimated noise covariance following the Probabilistic PCA model from Tipping and Bishop 1999. See “Pattern Recognition and Machine Learning” by C. Bishop, 12.2.1 p. 574 or http://www.miketipping.com/papers/met-mppca.pdf. It is required to compute the estimated data covariance and score samples.
            Equal to the average of (min(n_features, n_samples) - n_components) smallest eigenvalues of the covariance matrix of X.
    '''
    if np.any(np.isnan(ys)):
        raise ValueError('matrix may not have null values.')
    pca=PCA(n_components=n_components,copy=copy,whiten=whiten,svd_solver=svd_solver,tol=tol,iterated_power=iterated_power,random_state=random_state)
    return pca.fit(ys).__dict__


def mssa(ys, M=None, nMC=0, f=0.3):
    '''Multi-channel singular spectrum analysis analysis

    Multivariate generalization of SSA [2], using the original algorithm of [1].

    Parameters
    ----------

    ys : array
          multiple time series (dimension: length of time series x total number of time series)

    M : int
       window size (embedding dimension, default: 10% of the length of the series)

    nMC : int
       Number of iteration in the Monte-Carlo process [default=0, no Monte Carlo process]

    f : float
       fraction (0<f<=1) of good data points for identifying significant PCs [f = 0.3]

    Returns
    -------
    res : dict
        Containing:

        - eig_val : array of eigenvalue spectrum

        - eig_val05 : The 5% percentile of eigenvalues

        - eig_val95 : The 95% percentile of eigenvalues

        - PC : matrix of principal components (2D array)

        - RC : matrix of RCs (nrec,N,nrec*M) (2D array)

    References
    ----------
    [1]_ Vautard, R., and M. Ghil (1989), Singular spectrum analysis in nonlinear
    dynamics, with applications to paleoclimatic time series, Physica D, 35,
    395–424.

    [2]_ Jiang, N., J. D. Neelin, and M. Ghil (1995), Quasi-quadrennial and
    quasi-biennial variability in the equatorial Pacific, Clim. Dyn., 12, 101-112.

    See Also
    --------

    pyleoclim.utils.decomposition.ssa : Singular Spectrum Analysis (one channel)

    '''
    N = len(ys[:, 0])
    nrec = len(ys[0, :])
    if M == None:
        M=int(N/10)
    Y = np.zeros((N - M + 1, nrec * M))
    for irec in np.arange(nrec):
        for m in np.arange(0, M):
            Y[:, m + irec * M] = ys[m:N - M + 1 + m, irec]

    C = np.dot(np.nan_to_num(np.transpose(Y)), np.nan_to_num(Y)) / (N - M + 1)
    D, eig_vec = eigh(C)

    sort_tmp = np.sort(D)
    eig_val = sort_tmp[::-1]
    sortarg = np.argsort(-D)

    eig_vec = eig_vec[:, sortarg]

    # test the signifiance using Monte-Carlo
    Ym = np.zeros((N - M + 1, nrec * M))
    noise = np.zeros((nrec, N, nMC))
    for irec in np.arange(nrec):
        noise[irec, 0, :] = ys[0, irec]
    eig_val_R = np.zeros((nrec * M, nMC))
    # estimate coefficents of ar1 processes, and then generate ar1 time series (noise)
    for irec in np.arange(nrec):
        Xs = ys[:, irec]
        coefs_est, var_est = alg.AR_est_YW(Xs[~np.isnan(Xs)], 1)
        sigma_est = np.sqrt(var_est)

        for jt in range(1, N):
            noise[irec, jt, :] = coefs_est * noise[irec, jt - 1, :] + sigma_est * np.random.randn(1, nMC)

    for m in range(nMC):
        for irec in np.arange(nrec):
            noise[irec, :, m] = (noise[irec, :, m] - np.mean(noise[irec, :, m])) / (
                np.std(noise[irec, :, m], ddof=1))
            for im in np.arange(0, M):
                Ym[:, im + irec * M] = noise[irec, im:N - M + 1 + im, m]
        Cn = np.dot(np.nan_to_num(np.transpose(Ym)), np.nan_to_num(Ym)) / (N - M + 1)
        # eig_val_R[:,m] = np.diag(np.dot(np.dot(eig_vec,Cn),np.transpose(eig_vec)))
        eig_val_R[:, m] = np.diag(np.dot(np.dot(np.transpose(eig_vec), Cn), eig_vec))

    eig_val95 = np.percentile(eig_val_R, 95, axis=1)
    eig_val05 = np.percentile(eig_val_R, 5, axis=1)


    # determine principal component time series
    PC = np.zeros((N - M + 1, nrec * M))
    PC[:, :] = np.nan
    for k in np.arange(nrec * M):
        for i in np.arange(0, N - M + 1):
            #   modify for nan
            prod = Y[i, :] * eig_vec[:, k]
            ngood = sum(~np.isnan(prod))
            #   must have at least m*f good points
            if ngood >= M * f:
                PC[i, k] = sum(prod[~np.isnan(prod)])  # the columns of this matrix are Ak(t), k=1 to M (T-PCs)

    # compute reconstructed timeseries
    Np = N - M + 1

    RC = np.zeros((nrec, N, nrec * M))

    for k in np.arange(nrec):
        for im in np.arange(M):
            x2 = np.dot(np.expand_dims(PC[:, im], axis=1), np.expand_dims(eig_vec[0 + k * M:M + k * M, im], axis=0))
            x2 = np.flipud(x2)

            for n in np.arange(N):
                RC[k, n, im] = np.diagonal(x2, offset=-(Np - 1 - n)).mean()
    res = {'eig_val': eig_val, 'eig_vec': eig_vec, 'q05': eig_val95, 'q95': eig_val05, 'PC': PC, 'RC': RC}

    return res

def ssa(ys, M=None, nMC=0, f=0.5):
    '''Singular spectrum analysis

    Nonparametric, orthogonal decomposition of timeseries into constituent oscillations.
    This implementation  uses the method of [1], with applications presented in [2]
    Optionally (MC>0), the significance of eigenvalues is assessed by Monte-Carlo simulations of an AR(1) model fit to X, using [3].
    The method expects regular spacing, but is tolerant to missing values, up to a fraction 0<f<1 (see [4]).

    Parameters
    ----------

    ys : array of length N
          time series (evenly-spaced, possibly with with up to f*N NaNs)

    M : int
       window size (default: 10% of the length of the series)

    nMC : int
        Number of iteration in the Monte-Carlo process (default nMC=0, bypasses Monte Carlo SSA)
        Note: currently only supported for evenly-spaced, gap-free data.

    f : float
        maximum allowable fraction of missing values.

    Returns
    -------

    res : dict
        Containing:

        - eig_val : (M, 1) array of eigenvalue spectrum
        
        - eig_vec : Matrix of temporal eigenvectors

        - PC : (N - M + 1, M) array of principal components

        - RC : (N,  M) array of reconstructed components

        - eig_val_q : (M, 2) array contaitning the 5% and 95% quantiles of the Monte-Carlo eigenvalue spectrum [ if MC >0 ]

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

    See Also
    --------

    pyleoclim.utils.decomposition.mssa : Multi-channel SSA

    '''

    ys, mu, _ = standardize(ys)

    N = len(ys)

    if M == None:
        M=int(N/10)
    c = np.zeros(M)

    for j in range(M):
        prod = ys[0:N - j] * ys[j:N]
        c[j] = sum(prod[~np.isnan(prod)]) / (sum(~np.isnan(prod)) - 1)


    C = toeplitz(c[0:M])  #form correlation matrix

    D, eig_vec = eigh(C) # solve eigendecomposition

    sort_tmp = np.sort(D)
    eig_val = sort_tmp[::-1]
    sortarg = np.argsort(-D)
    eig_vec = eig_vec[:, sortarg]

    # determine principal component time series
    PC = np.zeros((N - M + 1, M))
    PC[:, :] = np.nan
    for k in np.arange(M):
        for i in np.arange(0, N - M + 1):
            #   modify for nan
            prod = ys[i:i + M] * eig_vec[:, k]
            ngood = sum(~np.isnan(prod))
            #   must have at least m*f good points
            if ngood >= M * f:
                PC[i, k] = sum(
                    prod[~np.isnan(prod)]) * M / ngood  # the columns of this matrix are Ak(t), k=1 to M (T-PCs)

    # compute reconstructed timeseries
    Np = N - M + 1

    RC = np.zeros((N, M))

    for im in np.arange(M):
        x2 = np.dot(np.expand_dims(PC[:, im], axis=1), np.expand_dims(eig_vec[0:M, im], axis=0))
        x2 = np.flipud(x2)

        for n in np.arange(N):
            RC[n, im] = np.diagonal(x2, offset=-(Np - 1 - n)).mean()

    RC = RC + np.tile(mu, reps=[N, M])  # put the mean back in

    # TODO: implement automatic truncation criteria: (1) Kaiser rule (2) significant MC-SSA modes
    # and (3) variance % criterion (e.g. first K modes that explain at least 90% of the variance).


    if nMC > 0: # If Monte-Carlo SSA is requested.
        # TODO: translate and use https://github.com/SMAC-Group/uAR1 here

        noise = ar1_sim(ys, nMC)  # generate MC AR(1) surrogates of y

        eig_val_R = np.zeros((M,nMC)) # define eigenvalue matrix

        lgs = np.arange(-N + 1, N)

        for m in range(nMC):
            xn, _ , _ = standardize(noise[:, m]) # center and standardize
            Gn = np.correlate(xn, xn, "full")
            Gn = Gn / (N - abs(lgs))
            Cn = toeplitz(Gn[N - 1:N - 1 + M])
            eig_val_R[:, m] = np.diag(np.dot(np.dot(np.transpose(eig_vec), Cn), eig_vec))

        eig_val_q = np.empty((M,2))
        eig_val_q[:,0] = np.percentile(eig_val_R, 5, axis=1)
        eig_val_q[:,1] = np.percentile(eig_val_R, 95, axis=1)
    else:
        eig_val_q = None
    res = {'eig_val': eig_val, 'eig_vec': eig_vec, 'PC': PC, 'RC': RC, 'eig_val_q': eig_val_q}
    return res
