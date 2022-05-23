#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module converts Pyleoclim objects to and from JSON files. 

Useful for obtaining a human-readable output and keeping the results of an analysis. The JSON file can also be used to swap analysis results between programming language. 

Please note that these utilities are maintained on a as-needed basis and that not all objects are currently available.
"""

__all__ =['PyleoObj_to_json', 'json_to_PyleoObj', 'isPyleoclim']

import pyleoclim as pyleo
import numpy as np
import inspect
import json
from urllib.request import urlopen
import re

def isPyleoclim(obj):
    '''
    Check whether an object is a valid type for Pyleoclim ui object

    Parameters
    ----------
    obj : pyleoclim.core.ui
        Object from the Pyleoclim UI module

    Returns
    -------
    Bool : {True,False}
        

    '''
    class_names=[]
    for name, ob in inspect.getmembers(pyleo.core.ui):
            if inspect.isclass(ob):
                class_names.append(ob)
    return type(obj) in class_names

def PyleoObj_to_dict(obj):
    '''Transform a pyleoclim object into a dictionary. 
    
    The transformation ensures that all the objects are JSON serializable (i.e. all numpy arrays have been converted to lists.)
    

    Parameters
    ----------
    obj : pyleoclim.core.io
        A pyleoclim object from the UI module

    Returns
    -------
    s : dict
        A JSON-encodable dictionary
        
    See also
    --------
    
    pyleoclim.utils.jsonutils.isPyleoclim : Whether an object is a valid Pyleoclim object

    '''
    
    if isinstance(obj,(dict)):
        s=obj
    else:
        s=vars(obj)
    for k in s.keys():
        #print(k)
        if isinstance(s[k],(np.ndarray)):            
            s[k] = s[k].astype('float64').tolist()
        elif isinstance(s[k],(dict)):
            s[k]=PyleoObj_to_dict(s[k])
        elif isPyleoclim(s[k])==True:
            s[k]=PyleoObj_to_dict(s[k])
        elif isinstance(s[k],(list)):
            if isPyleoclim(s[k][0])==True:
                new_list=[]
                for item in s[k]:
                    new_list.append(PyleoObj_to_dict(item))
                s[k]=new_list                    
    
    return s

def PyleoObj_to_json(obj, filename):
    '''Serializes a Pyleoclim object into a JSON file

    Parameters
    ----------
    obj : pyleoclim.core.ui
        A Pyleoclim object from the UI module
        
    filename : str
        Filename or path to save the JSON to.

    Returns
    -------
    None 
    
    See also
    --------
    pyleoclim.utils.jsonutils.PyleoObj_to_dict : Encodes a Pyleoclim UI object into a dictionary that is JSON serializable
    
    '''
    
    s = PyleoObj_to_dict(obj)
    with open(filename,'w') as f:
        json.dump(s, f)
        f.close()
    

def open_json(filename):
    '''Open a json file.

    Parameters
    ----------
    filename : str
        Path to the json file or URL

    Returns
    -------
    t : dict
        A Python dictionary from the json

    '''
    
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if (re.match(regex, filename) is not None)==True:
        response = urlopen(filename)
        t = json.loads(response.read())
    else:    
        with open(filename,'r') as f:
            t=json.load(f)
    return t

def objname_to_obj(objname):
    '''Returns the correct obj type for the name of a Pyleoclim UI object

    Parameters
    ----------
    objname : str
        Name of the object (e.g., Series, Scalogram, MultipleSeries...)

    Raises
    ------
    ValueError
        If the name of the object is not valid

    Returns
    -------
    obj : pyleoclim.core.ui
        A valid Pyleoclim object for the UI module

    '''
    
    possible_objects={'Series':pyleo.core.ui.Series,
                      'PSD':pyleo.core.ui.PSD,
                      'Scalogram':pyleo.core.ui.Scalogram,
                      'Coherence':pyleo.core.ui.Coherence,
                      'MultipleSeries':pyleo.core.ui.MultipleSeries,
                      'SurrogateSeries':pyleo.core.ui.SurrogateSeries,
                      'EnsembleSeries':pyleo.core.ui.EnsembleSeries,
                      'MultiplePSD':pyleo.core.ui.MultiplePSD,
                      'MultipleScalogram':pyleo.core.ui.MultipleScalogram,
                      'Corr':pyleo.core.ui.Corr,
                      'CorrEns':pyleo.core.ui.CorrEns,
                      'SpatialDecomp':pyleo.core.ui.SpatialDecomp,
                      'SsaRes':pyleo.core.ui.SsaRes,
                      'Lipd':pyleo.core.ui.Lipd,
                      'LipdSeries':pyleo.core.ui.LipdSeries
        }
    
    try:
        obj=possible_objects[objname]
    except:
        raise ValueError("The object is not a proper Pyleoclim object")
        
    return obj

def json_to_PyleoObj(filename,objname):
    '''Reads a JSON serialized file into a Pyleoclim object

    Parameters
    ----------
    filename : str
        The filename/path/URL of the JSON-serialized object
    objname : str
        Name of the object (e.g., Series, Scalogram, MultipleSeries...)

    Returns
    -------
    pyleoObj : pyleoclim.core.ui
        A Pyleoclim UI object
        
    See also
    --------
    pyleoclim.utils.jsonutils.open_json : open a json file from a local source or URL
    
    pyleoclim.utils.jsonutils.objename_to_obj : create a valid Pyleoclim object from a string   

    '''
    
    obj = objname_to_obj(objname)
    a = open_json(filename)

    for k in a.keys():
        if k == 'timeseries':
            a[k]=pyleo.Series(**a[k])
        if k == 'signif_qs' and a[k] is not None:
            if obj==pyleo.core.ui.PSD:
                for idx,item in enumerate(a[k]['psd_list']):
                    if item['timeseries'] is not None:
                        item['timeseries'] = pyleo.Series(**item['timeseries'])
                    a[k]['psd_list'][idx]=pyleo.PSD(**a[k]['psd_list'][idx])
                a[k] = pyleo.MultiplePSD(**a[k])
            elif obj == pyleo.core.ui.Scalogram or obj == pyleo.core.ui.Coherence:
                for idx,item in enumerate(a[k]['scalogram_list']):
                    if item['timeseries'] is not None:
                        item['timeseries'] = pyleo.Series(**item['timeseries'])
                    a[k]['scalogram_list'][idx]=pyleo.Scalogram(**a[k]['scalogram_list'][idx])
                a[k] = pyleo.MultipleScalogram(**a[k])
        if k == 'signif_scals' and a[k] is not None:
            for idx,item in enumerate(a[k]['scalogram_list']):
                if item['timeseries'] is not None:
                    item['timeseries'] = pyleo.Series(**item['timeseries'])
                a[k]['scalogram_list'][idx]=pyleo.Scalogram(**a[k]['scalogram_list'][idx])
            a[k] = pyleo.MultipleScalogram(**a[k])     

    pyleoObj=obj(**a)
    
    return pyleoObj