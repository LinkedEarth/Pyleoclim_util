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
from .wavelet import preprocess
from .tsutils import clean_ts, standardize
from scipy.linalg import eigh
from nitime import algorithms as alg
from scipy.linalg import toeplitz


#------
#Main functions
#------

def pca(x,n_components=None,copy=True,whiten=False, svd_solver='auto',tol=0.0,iterated_power='auto',random_state=None):
    '''
    scikit-learn PCA

    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Args
    ----
    x : array
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
    if np.any(np.isnan(x)):
        raise ValueError('matrix may not have null values.')
    pca=PCA(n_components=n_components,copy=copy,whiten=whiten,svd_solver=svd_solver,tol=tol,iterated_power=iterated_power,random_state=random_state)
    return pca.fit(x).__dict__


def mssa(data, M, MC=1000, f=0.3):
    '''Multi-channel SSA analysis
    (applicable for data including missing values)
    and test the significance by Monte-Carlo method

    Args
    ----

    data : array
          multiple time series (dimension: length of time series x total number of time series)
    M : int
       window size
    MC : int
       Number of iteration in the Monte-Carlo process
    f : float
       fraction (0<f<=1) of good data points for identifying significant PCs [f = 0.3]

    Returns
    -------

    eig_val : array
           eigenvalue spectrum
    eig_val05 : float
         The 5% percentile of eigenvalues
    eig_val95 : float
         The 95% percentile of eigenvalues
    PC : 2D array
         matrix of principal components
    RC : 2D array
        matrix of RCs (nrec,N,nrec*M)

    '''
    N = len(data[:, 0])
    nrec = len(data[0, :])
    Y = np.zeros((N - M + 1, nrec * M))
    for irec in np.arange(nrec):
        for m in np.arange(0, M):
            Y[:, m + irec * M] = data[m:N - M + 1 + m, irec]

    C = np.dot(np.nan_to_num(np.transpose(Y)), np.nan_to_num(Y)) / (N - M + 1)
    D, eig_vec = eigh(C)

    sort_tmp = np.sort(D)
    eig_val = sort_tmp[::-1]
    sortarg = np.argsort(-D)

    eig_vec = eig_vec[:, sortarg]

    # test the signifiance using Monte-Carlo
    Ym = np.zeros((N - M + 1, nrec * M))
    noise = np.zeros((nrec, N, MC))
    for irec in np.arange(nrec):
        noise[irec, 0, :] = data[0, irec]
    eig_val_R = np.zeros((nrec * M, MC))
    # estimate coefficents of ar1 processes, and then generate ar1 time series (noise)
    for irec in np.arange(nrec):
        Xr = data[:, irec]
        coefs_est, var_est = alg.AR_est_YW(Xr[~np.isnan(Xr)], 1)
        sigma_est = np.sqrt(var_est)

        for jt in range(1, N):
            noise[irec, jt, :] = coefs_est * noise[irec, jt - 1, :] + sigma_est * np.random.randn(1, MC)

    for m in range(MC):
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

    return eig_val, eig_vec, eig_val95, eig_val05, PC, RC

def ssa(X, M=None, MC=0, f=0.3):
    '''Singular spectrum analysis for a time series X, using the method of [1] and
    the formulation of [3]. Optionally (MC>0), the significance of eigenvalues
    is assessed by Monte-Carlo simulations of an AR(1) model fit to X, using [2].

    Args
    ----

    X : array of length N
          time series (evenly-spaced, possibly with with up to f*N NaNs)
    M : int
       window size (default: 10% of the length of the series)
    MC : int
        Number of iteration in the Monte-Carlo process (default M=0, bypasses Monte Carlo SSA)

    Returns
    -------

    eig_val : (M, 1) array
        eigenvalue spectrum
    PC : (N - M + 1, M) array
        matrix of principal components
    RC : (N,  M) array
        matrix of reconstructed components
    eig_val_q : (M, 2) array
         The 5% and 95% quantiles of the Monte-Carlo eigenvalue spectrum [ if MC >0 ]

    References:
    -----------
    [1] Vautard, R., and M. Ghil (1989), Singular spectrum analysis in nonlinear
    dynamics, with applications to paleoclimatic time series, Physica D, 35,
    395–424.

    [2] Allen, M. R., and L. A. Smith (1996), Monte Carlo SSA: Detecting irregular
    oscillations in the presence of coloured noise, J. Clim., 9, 3373–3404.

    [3] Ghil, M., R. M. Allen, M. D. Dettinger, K. Ide, D. Kondrashov, M. E. Mann,
    A. Robertson, A. Saunders, Y. Tian, F. Varadi, and P. Yiou (2002),
    Advanced spectral methods for climatic time series, Rev. Geophys., 40(1),
    1003–1052, doi:10.1029/2000RG000092.

    '''

    Xr, mu, _ = standardize(X)

    N = len(X)

    if not M:
        M=int(N/10)
    c = np.zeros(M)

    for j in range(M):
        prod = Xr[0:N - j] * Xr[j:N]
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
            prod = Xr[i:i + M] * eig_vec[:, k]
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

    RC = RC + np.repmat(mu, (N, M))  # put the mean back in
    # TODO: implement automatic truncation criteria.

    if MC > 0:
        # If Monte-Carlo SSA is requested. NOTE: DO NOT ATTEMPT IF MISSING DATA. Use https://github.com/SMAC-Group/uAR1 instead.
        coefs_est, var_est = alg.AR_est_YW(Xr, 1)
        sigma_est = np.sqrt(var_est)

        noise = np.zeros((N, MC))
        noise[0, :] = Xr[0]
        eig_val_R = np.zeros((M, MC))

        for jt in range(1, N):
            # TODO: update to proper AR simulation, e.g. with statsmodels
            noise[jt, :] = coefs_est * noise[jt - 1, :] + sigma_est * np.random.randn(1, MC)

        for m in range(MC):
            noise[:, m] = (noise[:, m] - np.mean(noise[:, m])) / (np.std(noise[:, m], ddof=1))
            Gn = np.correlate(noise[:, m], noise[:, m], "full")
            lgs = np.arange(-N + 1, N)
            Gn = Gn / (N - abs(lgs))
            Cn = toeplitz(Gn[N - 1:N - 1 + M])
            eig_val_R[:, m] = np.diag(np.dot(np.dot(np.transpose(eig_vec), Cn), eig_vec))

        eig_val_q = np.empty((M,2))
        eig_val_q[:,0] = np.percentile(eig_val_R, 5, axis=1)
        eig_val_q[:,1] = np.percentile(eig_val_R, 95, axis=1)
    else:
        eig_val_q = None

    return eig_val, eig_vec, PC, RC, eig_val_q
