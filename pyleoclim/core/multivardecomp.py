import numpy as np
from matplotlib import pyplot as plt, gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ..core import series
from ..utils import plotting, lipdutils
from ..utils import mapping as mp


class MultivariateDecomp:
    ''' Class to hold the results of multivariate decompositions
        applies to : `pca()`, `mcpca()`, `mssa()`

        Parameters
        ----------

        time: float
        
            the common time axis

        name: str
        
            name of the dataset/analysis to use in plots

        eigvals: 1d array
        
            vector of eigenvalues from the decomposition

        eigvecs: 2d array
        
            array of eigenvectors from the decomposition (e.g. EOFs)
            
        pcs : 1d array
        
            array containing the temporal expansion coefficients (e.g. "principal components" in the climate lore)

        pctvar: float
        
            array of pct variance accounted for by each mode
            
        orig : MultipleSeries, or MultipleGeoSeries object
        
            original data, on a common time axis 
            
        locs: float (p, 2)
        
            a p x 2 array of coordinates (latitude, longitude) for mapping spatial patterns. Defaults to None 

        neff: float
        
            scalar representing the effective sample size of the leading mode
            
            

    '''

    def __init__(self, name, eigvals, eigvecs, pctvar, pcs, neff, orig, locs = None):
        self.name = name
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.pctvar = pctvar
        self.pcs = pcs
        self.neff = neff
        self.orig = orig
        self.locs = locs

    def screeplot(self, figsize=[6, 4], uq='N82', title=None, ax=None, savefig_settings=None,
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
            
        See Also
        --------
        pyleoclim.core.MultipleSeries.pca : Principal Component Analysis

        References
        ----------
        
        [1]_ North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng (1982), Sampling errors in the estimation of empirical orthogonal functions, Mon. Weather Rev., 110, 699–706.
        
        [2]_ Hannachi, A., I. T. Jolliffe, and D. B. Stephenson (2007), Empirical orthogonal functions and related techniques in atmospheric science: A review, International Journal of Climatology, 27(9), 1119–1152, doi:10.1002/joc.1499.

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
        if title is None:
            title = self.name + ' eigenvalues'
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

    def modeplot(self, index=0, figsize=[8, 8], ax=None, savefig_settings=None,
                 title_kwargs=None, spec_method='mtm', cmap='RdBu_r', cb_scale = 0.8,
                 flip = False, map_kwargs=None, scatter_kwargs=None):
        ''' Dashboard visualizing the properties of a given mode, including:
            1. The temporal coefficient (PC or similar)
            2. its spectrum
            3. The loadings (EOF or similar), possibly geolocated.

        Parameters
        ----------
        
        index : int
        
            the (0-based) index of the mode to visualize.
            Default is 0, corresponding to the first mode.

        figsize : list, optional
        
            The figure size. The default is [8, 8].

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
            
        cmap: str
           colormap name for the loadings (https://matplotlib.org/stable/tutorials/colors/colormaps.html)  
           
        cb_scale : float in [0, 1) 
                             
           scale of the colorbar, called "shrink" in https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
           default is 0.8, which works well with default size. Change at your own risk. 
           
        map_kwargs : dict, optional

            Optional arguments for the map. See GeoSeries.map(). The default is None.
            
        scatter_kwargs : dict, optional
            
            Optional arguments for the scatterplot. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter
            
        See Also
        --------
        pyleoclim.core.MultipleSeries.pca : Principal Component Analysis
        
        pyleoclim.utils.tsutils.eff_sample_size : Effective sample size
        
        '''
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
        
        if flip:
            PC = -self.pcs[:, index]
            EOF = -self.eigvecs[:, index]
        else:
            PC = self.pcs[:, index]
            EOF = self.eigvecs[:, index]
            
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(4, 3, wspace=0.3, hspace=0.3)
        gs.update(left=0, right=1.1)
        
        ax = {}
        # plot the PC
        ax['pc'] = fig.add_subplot(gs[0, :2])
        label = rf'$PC_{index + 1}$' 
        t = self.orig.series_list[0].time
        ts = series.Series(time=t, value=PC, verbose=False)  # define timeseries object for the PC
        ts.plot(ax=ax['pc'])
        ax['pc'].set_ylabel(label)
               
        # plot its PSD
        ax['psd'] = fig.add_subplot(gs[0, 2])
        psd = ts.interp().spectral(method=spec_method)
        _ = psd.plot(ax=ax['psd'], label=label)

        # plot spatial pattern or spaghetti
       
        if self.locs is not None:
            # make the map - brute force since projection is not being returned properly
            lats = self.locs[:,0]
            lons = self.locs[:,1]
            
            map_kwargs = {} if map_kwargs is None else map_kwargs.copy()
            if 'projection' in map_kwargs.keys():
                projection = map_kwargs['projection']
            else:
                projection = 'Robinson'
            if 'proj_default' in map_kwargs.keys():
                proj_default = map_kwargs['proj_default']
            else:
                proj_default = True
            if proj_default == True:
                proj1 = {'central_latitude': lats.mean(),
                         'central_longitude': lons.mean()}
                proj2 = {'central_latitude': lats.mean()}
                proj3 = {'central_longitude': lons.mean()}
                try:
                    proj = mp.set_proj(projection=projection, proj_default=proj1)
                except:
                    try:
                        proj = mp.set_proj(projection=projection, proj_default=proj3)
                    except:
                        proj = mp.set_proj(projection=projection, proj_default=proj2)
            if 'marker' in map_kwargs.keys():
                marker = map_kwargs['marker']
            else:
                marker = 'o'
                #marker = [lipdutils.PLOT_DEFAULT[ts.archiveType][1] for ts in self.orig.series_list] 
        
            if 'background' in map_kwargs.keys():
                background = map_kwargs['background']
            else:
                background = False
            if 'force_global' in map_kwargs.keys():
                force_global = map_kwargs['force_global']
            else:
                force_global = True
            if 'land' in map_kwargs.keys():
                land = map_kwargs['land']
            else:
                land = True
            if 'borders' in map_kwargs.keys():
                borders = map_kwargs['borders']
            else:
                borders = True
            if 'rivers' in map_kwargs.keys():
                rivers = map_kwargs['rivers']
            else:
                rivers = False
            if 'lakes' in map_kwargs.keys():
                lakes = map_kwargs['lakes']
            else:
                lakes = False
            if 'scatter_kwargs' in map_kwargs.keys():
                scatter_kwargs = map_kwargs['scatter_kwargs']
            else:
                scatter_kwargs = {}
            if 'markersize' in map_kwargs.keys():
                scatter_kwargs.update({'s': map_kwargs['markersize']})
            else:
                scatter_kwargs.update({'s': 100})
                
            if 'edgecolors' in map_kwargs.keys():
                scatter_kwargs.update({'edgecolors': map_kwargs['edgecolors']})
            else:
                scatter_kwargs.update({'edgecolors':'white'})
            
            # prepare the map
            data_crs = ccrs.PlateCarree() 
            ax['map'] = fig.add_subplot(gs[1:, :], projection=proj)
            if force_global:
                ax['map'].set_global()
            ax['map'].coastlines()
            if background is True:
                ax['map'].stock_img()
            # Additional information
            if land is True:
                ax['map'].add_feature(cfeature.LAND)
                #ax['map'].add_feature(cfeature.OCEAN, alpha=0.5)
                
            if borders is True:
                ax['map'].add_feature(cfeature.BORDERS, alpha=0.5)
            if lakes is True:
                ax['map'].add_feature(cfeature.LAKES, alpha=0.5)
            if rivers is True:
                ax['map'].add_feature(cfeature.RIVERS)
                
            # h/t to this solution: https://stackoverflow.com/a/66578339  
            # right now, marker is ignored ; need to loop over values but then the colors get messed up
            vext = np.abs(EOF).max()
            
            sc = ax['map'].scatter(lons, lats, marker=marker, 
                              c=EOF, cmap=cmap, vmin = -vext, vmax = vext,
                              transform=data_crs, **scatter_kwargs)
            # if legend == True:
            #     ax.legend(**lgd_kwargs)
         
            # make colorbar, h/t https://stackoverflow.com/a/73061877
            fig.colorbar(sc, ax=ax['map'], label=rf'$EOF_{index + 1}$' , 
                         shrink=cb_scale, orientation="vertical")
                             
        else: # plot the original data
            ax['map'] = fig.add_subplot(gs[1:, :])
            self.orig.standardize().plot(ax=ax['map'], title='',
                                         ylabel = 'Original Data (standardized)')   

        # if title is not None:
        #     title_kwargs = {} if title_kwargs is None else title_kwargs.copy()
        #     t_args = {'y': 1.1, 'weight': 'bold'}
        #     t_args.update(title_kwargs)
        #     fig.suptitle(title, **t_args)
        fig.suptitle(self.name + ' mode ' + str(index + 1) + ', ' + '{:3.2f}'.format(self.pctvar[index]) + '% variance explained',
                      weight='bold', y=0.92)
        
        fig.tight_layout()
        
        if 'path' in savefig_settings:
            plotting.savefig(fig, settings=savefig_settings)

        return fig, gs
