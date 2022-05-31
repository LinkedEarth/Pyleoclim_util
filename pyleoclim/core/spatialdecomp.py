import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import MaxNLocator

from ..core import series
from ..utils import plotting


class SpatialDecomp:
    ''' Class to hold the results of spatial decompositions
        applies to : `pca()`, `mcpca()`, `mssa()`

        Attributes
        ----------

        time: float
            the common time axis

        locs: float (p, 2)
            a p x 2 array of coordinates (latitude, longitude) for mapping the spatial patterns ("EOFs")

        name: str
            name of the dataset/analysis to use in plots

        eigvals: float
            vector of eigenvalues from the decomposition

        eigvecs: float
            array of eigenvectors from the decomposition

        pctvar: float
            array of pct variance accounted for by each mode

        neff: float
            scalar representing the effective sample size of the leading mode

    '''

    def __init__(self, time, locs, name, eigvals, eigvecs, pctvar, pcs, neff):
        self.time = time
        self.name = name
        self.locs = locs
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.pctvar = pctvar
        self.pcs = pcs
        self.neff = neff

    def screeplot(self, figsize=[6, 4], uq='N82', title='scree plot', ax=None, savefig_settings=None,
                  title_kwargs=None, xlim=[0, 10], clr_eig='C0'):
        ''' Plot the eigenvalue spectrum with uncertainties

        Parameters
        ----------
        figsize : list, optional
            The figure size. The default is [6, 4].

        title : str, optional
            Plot title. The default is 'scree plot'.

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        title_kwargs : dict, optional
            the keyword arguments for ax.set_title()

        ax : matplotlib.axis, optional
            the axis object from matplotlib
            See [matplotlib.axes](https://matplotlib.org/api/axes_api.html) for details.

        xlim : list, optional
            x-axis limits. The default is [0, 10] (first 10 eigenvalues)

        uq : str, optional
            Method used for uncertainty quantification of the eigenvalues.
            'N82' uses the North et al "rule of thumb" [1] with effective sample size
            computed as in [2].
            'MC' uses Monte-Carlo simulations (e.g. MC-EOF). Returns an error if no ensemble is found.

        clr_eig : str, optional
            color to be used for plotting eigenvalues


        References
        ----------
        _[1] North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng (1982), Sampling errors in the estimation of empirical orthogonal functions, Mon. Weather Rev., 110, 699–706.
        
        _[2] Hannachi, A., I. T. Jolliffe, and D. B. Stephenson (2007), Empirical orthogonal functions and related techniques in atmospheric science: A review, International Journal of Climatology, 27(9), 1119–1152, doi:10.1002/joc.1499.

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if self.neff < 2:
            self.neff = 2

        # compute 95% CI
        if uq == 'N82':
            eb_lbl = r'95% CI ($n_\mathrm{eff} = $' + '{:.1f}'.format(self.neff) + ')'  # declare method
            Lc = self.eigvals  # central estimate
            Lerr = np.tile(Lc, (2, 1))  # declare array
            Lerr[0, :] = Lc * np.sqrt(1 - np.sqrt(2 / self.neff))
            Lerr[1, :] = Lc * np.sqrt(1 + np.sqrt(2 / self.neff))
        elif uq == 'MC':
            eb_lbl = '95% CI (Monte Carlo)'  # declare method
            try:
                Lq = np.quantile(self.eigvals, [0.025, 0.5, 0.975], axis=1)
                Lc = Lq[1, :]
                Lerr = np.tile(Lc, (2, 1))  # declare array
                Lerr[0, :] = Lq[0, :]
                Lerr[1, :] = Lq[2, :]

            except ValueError:
                print("Eigenvalue array must have more than 1 non-singleton dimension.")
        else:
            raise NameError("unknown UQ method. No action taken")

        idx = np.arange(len(Lc)) + 1

        ax.errorbar(x=idx, y=Lc, yerr=Lerr, color=clr_eig, marker='o', ls='',
                    alpha=1.0, label=eb_lbl)

        ax.set_title(title, fontweight='bold');
        ax.legend();
        ax.set_xlabel(r'Mode index $i$');
        ax.set_ylabel(r'$\lambda_i$')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # enforce integer values

        if xlim is not None:
            ax.set_xlim(0.5, min(max(xlim), len(Lc)))

        if title is not None:
            title_kwargs = {} if title_kwargs is None else title_kwargs.copy()
            t_args = {'y': 1.1, 'weight': 'bold'}
            t_args.update(title_kwargs)
            ax.set_title(title, **t_args)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)

        return fig, ax

    def modeplot(self, index=0, figsize=[10, 5], ax=None, savefig_settings=None,
                 title_kwargs=None, spec_method='mtm'):
        ''' Dashboard visualizing the properties of a given mode, including:
            1. The temporal coefficient (PC or similar)
            2. its spectrum
            3. The spatial loadings (EOF or similar)

        Parameters
        ----------
        index : int
            the (0-based) index of the mode to visualize.
            Default is 0, corresponding to the first mode.

        figsize : list, optional
            The figure size. The default is [10, 5].

        savefig_settings : dict
            the dictionary of arguments for plt.savefig(); some notes below:
            - "path" must be specified; it can be any existed or non-existed path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}

        title_kwargs : dict
            the keyword arguments for ax.set_title()

        gs : matplotlib.gridspec object, optional
            the axis object from matplotlib
            See [matplotlib.gridspec.GridSpec](https://matplotlib.org/stable/tutorials/intermediate/gridspec.html) for details.

        spec_method: str, optional
            The name of the spectral method to be applied on the PC. Default: MTM
            Note that the data are evenly-spaced, so any spectral method that
            assumes even spacing is applicable here:  'mtm', 'welch', 'periodogram'
            'wwz' is relevant if scaling exponents need to be estimated, but ill-advised otherwise, as it is very slow.

        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        PC = self.pcs[:, index]
        ts = series.Series(time=self.time, value=PC)  # define timeseries object for the PC

        fig = plt.figure(tight_layout=True, figsize=figsize)
        gs = gridspec.GridSpec(2, 2)  # define grid for subplots
        ax1 = fig.add_subplot(gs[0, :])
        ts.plot(ax=ax1)
        ax1.set_ylabel('PC ' + str(index + 1))
        ax1.set_title('Mode ' + str(index + 1) + ', ' + '{:3.2f}'.format(self.pctvar[index]) + '% variance explained',
                      weight='bold')

        # plot spectrum
        ax2 = fig.add_subplot(gs[1, 0])
        psd_mtm_rc = ts.interp().spectral(method=spec_method)
        _ = psd_mtm_rc.plot(ax=ax2)
        ax2.set_xlabel('Period')
        ax2.set_title('Power spectrum (' + spec_method + ')', weight='bold')

        # plot T-EOF
        ax3 = fig.add_subplot(gs[1, 1])
        # EOF = self.eigvecs[:,mode]
        ax3.set_title('Spatial loadings \n (under construction)', weight='bold')

        # if title is not None:
        #     title_kwargs = {} if title_kwargs is None else title_kwargs.copy()
        #     t_args = {'y': 1.1, 'weight': 'bold'}
        #     t_args.update(title_kwargs)
        #     ax.set_title(title, **t_args)

        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)

        return fig, gs
