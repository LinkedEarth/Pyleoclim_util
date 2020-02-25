#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:29:36 2020

@author: deborahkhider
Contains decompoistion methods (PCA, SSA...)
"""
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


    '''
    if np.any(np.isnan(x)):
        raise ValueError('matrix may not have null values.')
    pca=PCA(n_components=n_components,copy=copy,whiten=whiten,svd_solver=svd_solver,tol=tol,iterated_power=iterated_power,random_state=random_state)
    return pca.fit(x).__dict__

def ssa(ys, ts, M, MC=1000, f=0.3, method='SSA', prep_args={}):
    '''
    Args
    ----

    ys : array
        time series
    ts: array
       time axis
    M : int
       window size
    MC : int
        Number of iteration in the Monte-Carlo process
    f : float
       fraction (0<f<=1) of good data points for identifying
    method (str, {'SSA', 'MSSA'}) : str({'SSA','MSSA'})
                                   perform SSA or MSSA

    prep_args : dict
              the arguments for preprocess, including
                detrend : str
                         'none' - the original time series is assumed to have no trend;
                         'linear' - a linear least-squares fit to `ys` is subtracted;
                         'constant' - the mean of `ys` is subtracted
                         'savitzy-golay' - ys is filtered using the Savitzky-Golay
                                             filters and the resulting filtered series is subtracted from y.
                          'hht' - detrending with Hilbert-Huang Transform
                params  : list
                           The paramters for the Savitzky-Golay filters. The first parameter
                           corresponds to the window size (default it set to half of the data)
                           while the second parameter correspond to the order of the filter
                          (default is 4). The third parameter is the order of the derivative
                          (the default is zero, which means only smoothing.)
                gaussianize : bool
                             If True, gaussianizes the timeseries
                standardize : bool
                             If True, standardizes the timeseries

    Returns
    -------

    res_dict : dictionary
              the result dictionary, including
                 deval : array
                        eigenvalue spectrum
                 eig_vec : array
                          eigenvalue vector
                 q05 : float
                      The 5% percentile of eigenvalues
                 q95 : float
                      The 95% percentile of eigenvalues
                 pc: 2D array
                    matrix of principal components
                 rc: 2D array
                    matrix of RCs (nrec,N,nrec*M) (only if K>0)
    '''

    
    ys, ts = clean_ts(ys, ts)
    ys = preprocess(ys, ts, **prep_args)

    ssa_func = {
        'SSA': ssa_all,
        'MSSA': mssa,
    }
    deval, eig_vec, q05, q95, pc, rc = ssa_func[method](ys, M, MC=MC, f=f)

    res_dict = {
        'deval': deval,
        'eig_vec': eig_vec,
        'q05': q05,
        'q95': q95,
        'pc': pc,
        'rc': rc,
    }

    return res_dict

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

    deval : array
           eigenvalue spectrum
    q05 : float
         The 5% percentile of eigenvalues
    q95 : float
         The 95% percentile of eigenvalues
    PC : 2D array
         matrix of principal components
    RC : 2D array
        matrix of RCs (nrec,N,nrec*M) (only if K>0)

    '''
    N = len(data[:, 0])
    nrec = len(data[0, :])
    Y = np.zeros((N - M + 1, nrec * M))
    for irec in np.arange(nrec):
        for m in np.arange(0, M):
            Y[:, m + irec * M] = data[m:N - M + 1 + m, irec]

    C = np.dot(np.nan_to_num(np.transpose(Y)), np.nan_to_num(Y)) / (N - M + 1)
    eig_val, eig_vec = eigh(C)

    sort_tmp = np.sort(eig_val)
    deval = sort_tmp[::-1]
    sortarg = np.argsort(-eig_val)

    eig_vec = eig_vec[:, sortarg]

    # test the signifiance using Monte-Carlo
    Ym = np.zeros((N - M + 1, nrec * M))
    noise = np.zeros((nrec, N, MC))
    for irec in np.arange(nrec):
        noise[irec, 0, :] = data[0, irec]
    Lamda_R = np.zeros((nrec * M, MC))
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
        # Lamda_R[:,m] = np.diag(np.dot(np.dot(eig_vec,Cn),np.transpose(eig_vec)))
        Lamda_R[:, m] = np.diag(np.dot(np.dot(np.transpose(eig_vec), Cn), eig_vec))

    q95 = np.percentile(Lamda_R, 95, axis=1)
    q05 = np.percentile(Lamda_R, 5, axis=1)


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

    return deval, eig_vec, q95, q05, PC, RC

def ssa_all(data, M, MC=1000, f=0.3):
    '''SSA analysis for a time series
    (applicable for data including missing values)
    and test the significance by Monte-Carlo method

    Args
    ----

    data : array
          time series
    M : int
       window size
    MC : int
        Number of iteration in the Monte-Carlo process
    f : fraction
       fraction (0<f<=1) of good data points for identifying significant PCs [f = 0.3]

    Returns
    -------

    deval : array
           eigenvalue spectrum
    q05 : float
         The 5% percentile of eigenvalues
    q95 : float
         The 95% percentile of eigenvalues
    PC : 2D array
        matrix of principal components
    RC : 2D array
        matrix of RCs (N*M, nmode) (only if K>0)
    '''


    Xr = standardize(data)
    N = len(data)
    c = np.zeros(M)

    for j in range(M):
        prod = Xr[0:N - j] * Xr[j:N]
        c[j] = sum(prod[~np.isnan(prod)]) / (sum(~np.isnan(prod)) - 1)


    C = toeplitz(c[0:M])

    eig_val, eig_vec = eigh(C)

    sort_tmp = np.sort(eig_val)
    deval = sort_tmp[::-1]
    sortarg = np.argsort(-eig_val)

    eig_vec = eig_vec[:, sortarg]

    coefs_est, var_est = alg.AR_est_YW(Xr[~np.isnan(Xr)], 1)
    sigma_est = np.sqrt(var_est)

    noise = np.zeros((N, MC))
    noise[0, :] = Xr[0]
    Lamda_R = np.zeros((M, MC))

    for jt in range(1, N):
        noise[jt, :] = coefs_est * noise[jt - 1, :] + sigma_est * np.random.randn(1, MC)

    for m in range(MC):
        noise[:, m] = (noise[:, m] - np.mean(noise[:, m])) / (np.std(noise[:, m], ddof=1))
        Gn = np.correlate(noise[:, m], noise[:, m], "full")
        lgs = np.arange(-N + 1, N)
        Gn = Gn / (N - abs(lgs))
        Cn = toeplitz(Gn[N - 1:N - 1 + M])
        # Lamda_R[:,m] = np.diag(np.dot(np.dot(eig_vec,Cn),np.transpose(eig_vec)))
        Lamda_R[:, m] = np.diag(np.dot(np.dot(np.transpose(eig_vec), Cn), eig_vec))

    q95 = np.percentile(Lamda_R, 95, axis=1)
    q05 = np.percentile(Lamda_R, 5, axis=1)

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

    return deval, eig_vec, q05, q95, PC, RC