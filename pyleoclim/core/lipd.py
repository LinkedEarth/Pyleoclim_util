"""
This allows to manipulate LiPD objects and take advantage of the metadata information for specific functionalities. Lipd objects are needed to create LipdSeries objects, which carry most of the timeseries functionalities.
"""

from ..utils import mapping, lipdutils

from ..core.lipdseries import LipdSeries
from copy import deepcopy
import warnings
import os

import lipd as lpd

class Lipd:
    '''The Lipd class allows to create a Lipd object from Lipd files. 
    This allows to manipulate LiPD objects and take advantage of the metadata information 
    for specific functionalities. Lipd objects are needed to create LipdSeries objects, 
    which carry most of the timeseries functionalities.


    Parameters
    ----------

    usr_path : str
    
        Path to the Lipd file(s). Can be URL (LiPD utilities only support loading one file at a time from a URL).
        If it's a URL, it must start with "http", "https", or "ftp".

    lidp_dict : dict
    
        LiPD files already loaded into Python through the LiPD utilities

    validate : bool
    
        Validate the LiPD files upon loading. Note that for a large library (>300files) this can take up to half an hour.

    remove : bool
    
        If validate is True and remove is True, ignores non-valid LiPD files. Note that loading unvalidated Lipd files may result in errors for some functionalities but not all.

    TODO
    ----

    Support querying the LinkedEarth platform
    
    References
    ----------
    
    McKay, N. P., & Emile-Geay, J. (2016). Technical Note: The Linked Paleo Data framework – a common tongue for paleoclimatology. Climate of the Past, 12, 1093-1100. 

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        import pyleoclim as pyleo
        url='http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
        d=pyleo.Lipd(usr_path=url)
    '''

    def __init__(self, usr_path=None, lipd_dict=None, validate=False, remove=False):
        self.plot_default = {'ice-other': ['#FFD600','h'],
                'ice/rock': ['#FFD600', 'h'],
                'coral': ['#FF8B00','o'],
                'documents':['k','p'],
                'glacierice':['#86CDFA', 'd'],
                'hybrid': ['#00BEFF','*'],
                'lakesediment': ['#4169E0','s'],
                'marinesediment': ['#8A4513', 's'],
                'sclerosponge' : ['r','o'],
                'speleothem' : ['#FF1492','d'],
                'wood' : ['#32CC32','^'],
                'molluskshells' : ['#FFD600','h'],
                'peat' : ['#2F4F4F','*'],
                'midden' : ['#824E2B','o'],
                'other':['k','o']}

        if validate==False and remove==True:
            print('Removal of unvalidated LiPD files require validation')
            validate=True

        #prepare the dictionaries for all possible scenarios
        if usr_path!=None:
            # since readLipd() takes only absolute path and it will change the current working directory (CWD) without turning back,
            # we need to record CWD manually and turn back after the data loading is finished
            cwd = os.getcwd()
            if usr_path[:4] == 'http' or usr_path[:3] == 'ftp':
                # URL
                D_path = lpd.readLipd(usr_path)
            else:
                # local path
                abs_path = os.path.abspath(usr_path)
                D_path = lpd.readLipd(abs_path)

            os.chdir(cwd)

            #make sure that it's more than one
            if 'archiveType' in D_path.keys():
                D_path={D_path['dataSetName']:D_path}
            if validate==True:
                cwd = os.getcwd()
                res=lpd.validate(D_path,detailed=False)
                os.chdir(cwd)
                if remove == True:
                    for item in res:
                        if item['status'] == 'FAIL':
                           c=item['feedback']['errMsgs']
                           check = []
                           for i in c:
                               if i.startswith('Mismatched columns'):
                                   check.append(1)
                               else: check.append(0)
                           if 0 in check:
                               del D_path[item['filename'].strip('.lpd')]
        else:
            D_path={}
        if lipd_dict!=None:
            D_dict=lipd_dict
            if 'archiveType' in D_dict.keys():
                D_dict={D_dict['dataSetName']:D_dict}
            if validate==True:
                cwd = os.getcwd()
                res=lpd.validate(D_dict,detailed=False)
                os.chdir(cwd)
                if remove == True:
                    for item in res:
                        if item['status'] == 'FAIL':
                           c=item['feedback']['errMsgs']
                           check = []
                           for i in c:
                               if i.startswith('Mismatched columns'):
                                   check.append(1)
                               else: check.append(0)
                           if 0 in check:
                               del D_dict[item['filename'].strip('.lpd')]
        else:
            D_dict={}

        # raise an error if empty
        if not bool(D_dict) and not bool(D_path) == True:
            raise ValueError('No valid files; try without validation.')
        #assemble
        self.lipd={}
        self.lipd.update(D_path)
        self.lipd.update(D_dict)

    def __repr__(self):
        return str(self.__dict__)

    def copy(self):
        '''Copy the object
        '''
        return deepcopy(self)

    def to_tso(self, mode='paleo'):
        '''Extracts all the variables to a list of LiPD timeseries objects
        
        In LiPD, timeseries objects are flatten dictionaries that contain the values for the time and variable axes as well as relevant metadata. 

        Parameters
        ----------
        
        mode : {'paleo','chron'}
        
            Whether to extract the timeseries information from the paleo tables or chron tables

        Returns
        -------
        
        ts_list : list
        
            List of LiPD timeseries objects

        
        References
        ----------
    
        McKay, N. P., & Emile-Geay, J. (2016). Technical Note: The Linked Paleo Data framework – a common tongue for paleoclimatology. Climate of the Past, 12, 1093-1100. 
        '''
        cwd = os.getcwd()
        ts_list=lpd.extractTs(self.__dict__['lipd'], mode=mode)
        os.chdir(cwd)
        return ts_list

    def extract(self,dataSetName):
        '''
        Parameters
        ----------
        
        dataSetName : str
        
            Extract a particular dataset

        Returns
        -------
        
        new : pyleoclim.Lipd
        
            A new object corresponding to a particular dataset

        '''
        new = self.copy()
        try:
            dict_out=self.__dict__['lipd'][dataSetName]
            new.lipd=dict_out
        except:
            pass

        return new

    def to_LipdSeriesList(self, mode='paleo'):
        '''Extracts all LiPD timeseries objects to a list of LipdSeries objects
        
        In LiPD, timeseries objects are flatten dictionaries that contain the values for the time and variable axes as well as relevant metadata. 

        Parameters
        ----------

        mode : {'paleo','chron'}
        
            Whether to extract the timeseries information from the paleo tables or chron tables

        Returns
        -------
        res : list
        
            A list of LiPDSeries objects
        
        References
        ----------
    
        McKay, N. P., & Emile-Geay, J. (2016). Technical Note: The Linked Paleo Data framework – a common tongue for paleoclimatology. Climate of the Past, 12, 1093-1100. 

        See also
        --------
        
        pyleoclim.core.lipdseries.LipdSeries : a LipdSeries object

        '''
        cwd = os.getcwd()
        ts_list=lpd.extractTs(self.__dict__['lipd'], mode=mode)
        os.chdir(cwd)

        res=[]

        for idx, item in enumerate(ts_list):
            try:
                res.append(LipdSeries(item))
            except:
                if mode == 'paleo':
                    txt = 'The timeseries from ' + str(idx) + ': ' +\
                            item['dataSetName'] + ': ' + \
                            item['paleoData_variableName'] + \
                            ' could not be coerced into a LipdSeries object, passing'
                else:
                    txt = 'The timeseries from ' + str(idx) + ': ' +\
                            item['dataSetName'] + ': ' + \
                            item['chronData_variableName'] + \
                            ' could not be coerced into a LipdSeries object, passing'
                warnings.warn(txt)
                pass

        return res

    def to_LipdSeries(self, number = None, mode = 'paleo'):
        '''Extracts one timeseries from the Lipd object

        In LiPD, timeseries objects are flatten dictionaries that contain the values for the time and variable axes as well as relevant metadata. 
        Note that this function may require user interaction if the number of the column in the file is unknown. The numbers are fixed so automating the code is as simple as retaining a series of numbers when reopening the files. 

        Parameters
        ----------

        number : int
        
            the number of the timeseries object

        mode : str; {'paleo','chron'}
        
            whether to extract the paleo or chron series.

        Returns
        -------
        
        ts : pyleoclim.LipdSeries
        
            A LipdSeries object

        See also
        --------
        
        pyleoclim.core.lipdseries.LipdSeries : LipdSeries object

        '''
        cwd = os.getcwd()
        ts_list = lpd.extractTs(self.__dict__['lipd'], mode=mode)
        os.chdir(cwd)
        if number is None:
            ts = LipdSeries(ts_list)
        else:
            try:
                number = int(number)
            except:
                raise TypeError('Number needs to be an integer or should be coerced into an integer.')
            ts = LipdSeries(ts_list[number])
        return ts



    def mapAllArchive(self, projection = 'Robinson', proj_default = True,
           background = True,borders = False, rivers = False, lakes = False,
           figsize = None, ax = None, marker=None, color=None,
           markersize = None, scatter_kwargs=None,
           legend=True, lgd_kwargs=None, savefig_settings=None):

        '''Map all the records contained in the LiPD object by the type of archive
        
        Note that the map is fully cusomizable by using the optional parameters. 

        Parameters
        ----------
        projection : str, optional
        
            The projection to use. The default is 'Robinson'.
            
        proj_default : bool, optional
        
            Wether to use the Pyleoclim defaults for each projection type. The default is True.
            
        background : bool, optional
        
            Wether to use a backgound. The default is True.
            
        borders : bool, optional
        
            Draw borders. The default is False.
            
        rivers : bool, optional
        
            Draw rivers. The default is False.
            
        lakes : bool, optional
        
            Draw lakes. The default is False.
            
        figsize : list, optional
        
            The size of the figure. The default is None.
            
        ax : matplotlib.ax, optional
        
            The matplotlib axis onto which to return the map. The default is None.
            
        marker : str, optional
        
            The marker type for each archive. The default is None, which uses a pre-defined palette in Pyleoclim.  
            To see the default option, run Lipd.plot_default where Lipd is the name of the object. 
            
        color : str, optional
        
            Color for each acrhive. The default is None. The default is None, which uses a pre-defined palette in Pyleoclim.  
            To see the default option, run Lipd.plot_default where Lipd is the name of the object. 
            
        markersize : float, optional
        
            Size of the marker. The default is None. 
            
        scatter_kwargs : dict, optional
        
            Parameters for the scatter plot. The default is None.
            
        legend : bool; {True,False}, optional
        
            Whether to plot the legend. The default is True.
            
        lgd_kwargs : dict, optional
        
            Arguments for the legend. The default is None.
            
        savefig_settings : dictionary, optional
        
            The dictionary of arguments for plt.savefig(); some notes below:  
            - "path" must be specified; it can be any existing or non-existing path,
              with or without a suffix; if the suffix is not given in "path", it will follow "format"
            - "format" can be one of {"pdf", "eps", "png", "ps"}. 
            The default is None.
        
        Returns
        -------
        res : tuple or fig
        
            The figure and axis if asked. 

        See also
        --------

        pyleoclim.utils.mapping.map : Underlying mapping function for Pyleoclim

        Examples
        --------

        For speed, we are only using one LiPD file. But these functions can load and map multiple.

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            @savefig mapallarchive.png
            fig, ax = data.mapAllArchive()
            pyleo.closefig(fig)

        Change the markersize

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyleoclim as pyleo
            url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid=MD982176.Stott.2004'
            data = pyleo.Lipd(usr_path = url)
            @savefig mapallarchive_marker.png
            fig, ax = data.mapAllArchive(markersize=100)
            pyleo.closefig(fig)


        '''
        scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs.copy()


        #get the information from the LiPD dict
        lat=[]
        lon=[]
        archiveType=[]

        for idx, key in enumerate(self.lipd):
            d = self.lipd[key]
            lat.append(d['geo']['geometry']['coordinates'][1])
            lon.append(d['geo']['geometry']['coordinates'][0])
            if 'archiveType' in d.keys():
                archiveType.append(lipdutils.LipdToOntology(d['archiveType']).lower().replace(" ",""))
            else:
                archiveType.append('other')

        # make sure criteria is in the plot_default list
        for idx,val in enumerate(archiveType):
            if val not in self.plot_default.keys():
                archiveType[idx] = 'other'

        if markersize is not None:
            scatter_kwargs.update({'s': markersize})

        if marker==None:
            marker=[]
            for item in archiveType:
                marker.append(self.plot_default[item][1])

        if color==None:
            color=[]
            for item in archiveType:
                color.append(self.plot_default[item][0])

        res = mapping.map(lat=lat, lon=lon, criteria=archiveType,
                              marker=marker, color =color,
                              projection = projection, proj_default = proj_default,
                              background = background,borders = borders,
                              rivers = rivers, lakes = lakes,
                              figsize = figsize, ax = ax,
                              scatter_kwargs=scatter_kwargs, legend=legend,
                              lgd_kwargs=lgd_kwargs,savefig_settings=savefig_settings)

        return res