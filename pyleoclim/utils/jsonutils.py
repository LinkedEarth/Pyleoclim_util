#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:09:03 2020

@author: deborahkhider

Utilities to import/export Pyleoclim objects from JSON files
"""

__all__ =['PyleoObj_to_json', 'json_to_Series','json_to_PSD','json_to_Scalogram']

import numpy as np
import json
import pyleoclim as pyleo

def transform(obj_dict):
    '''
    This function recursively transform pyleoclim object into dictionaries

    Parameters
    ----------
    obj_dict : dict
        A dictionary-like object

    Returns
    -------
    obj_dict : dict
        A dictionary of the pyleoclim objects

    '''
    for k in obj_dict.keys():
        if isinstance(obj_dict[k],(np.ndarray)):
            obj_dict[k] = obj_dict[k].tolist()
        elif isinstance(obj_dict[k],(pyleo.Series,pyleo.Scalogram)):
            obj_dict[k]=PyleoObj_to_json(obj_dict[k],dict_return=True)
        elif isinstance(obj_dict[k],pyleo.MultiplePSD):
            obj_dict[k]=PyleoObj_to_json(obj_dict[k],dict_return=True)
            c=[]
            idx = np.arange(0,len(obj_dict[k]['psd_list']),1).tolist()
            for item in idx:
                PSD_dict=PyleoObj_to_json(obj_dict[k]['psd_list'][item],dict_return=True)
                c.append(PSD_dict)
            obj_dict[k]['psd_list']=c    
        elif isinstance(obj_dict[k],pyleo.core.ui.MultipleScalogram):
            obj_dict[k]=PyleoObj_to_json(obj_dict[k],dict_return=True)
            c=[]
            idx = np.arange(0,len(obj_dict[k]['scalogram_list']),1).tolist()
            for item in idx:
                PSD_dict=PyleoObj_to_json(obj_dict[k]['scalogram_list'][item],dict_return=True)
                c.append(PSD_dict)
            obj_dict[k]['scalogram_list']=c            
        elif isinstance(obj_dict[k],(dict)):
            obj_dict[k]=transform(obj_dict[k])
    return obj_dict

def list_to_array(obj_dict):
    '''
    Recusively transforms lists of dictionaries

    Parameters
    ----------
    obj_dict : dict
        A dict-like object

    Returns
    -------
    obj_dict : dict
        A dictionary

    '''
    for k in obj_dict:
        if type(obj_dict[k])is dict:
            obj_dict[k]=list_to_array(obj_dict[k])
        elif type(obj_dict[k]) is list:
            obj_dict[k]=np.array(obj_dict[k])
        else:
            obj_dict[k]=obj_dict[k]
    return obj_dict

def PyleoObj_to_json(PyleoObj,filename,dict_return=False):
    '''
    
    Parameters
    ----------
    PyleoObj : a Pyleoclim object
       Can be a Series, PSD, Scalogram, MultipleSeries, MultiplePSD object
    filename : str
       The name of the output JSON file - ignored if dict_return == True
    dict_return : {True,False}, optional
        Whether the return the dictionary for further use. Default is False.

    Returns
    -------
    obj_dict : dict
        If dict_return is True, returns a dictionary like object from the JSON file. 

    '''
    obj_dict = PyleoObj.__dict__
    obj_dict = transform(obj_dict, filename)
    if dict_return == False:
        with open(filename,'w') as f:
            json.dump(obj_dict, f)
            f.close()
    elif dict_return == True:
        return  obj_dict

def json_to_Series(filename):
    '''
    Open a JSON file and returns a pyleoclim.Series object

    Parameters
    ----------
    filename : str
        The name of the JSON file containing the Series information

    Returns
    -------
    ts : pyleoclim.Series
        A pyleoclim.Series object

    '''
    with  open(filename,'r') as f:
        t = json.load(f)
    ts = pyleo.Series(time=np.array(t['time']),
                     value=np.array(t['value']),
                     time_name=t['time_name'],
                     time_unit=t['time_unit'],
                     value_name=t['value_name'],
                     value_unit=t['value_unit'],
                     label=t['label']) 
    return ts

def PSD_to_MultiplePSD(series_list):
    '''
    Transforms a list of PSD into a MutiplePSD object

    Parameters
    ----------
    series_list : list
        A list of multiple PSD objects.

    Returns
    -------
    MPSD : pyleoclim.MultiplePSD
        A pyleoclim MultiplePSD object

    '''
    idx = np.arange(0,len(series_list),1).tolist()
    d=[]
    for item in idx:
        t=series_list[item]
        PSD_obj=pyleo.PSD(frequency=np.array(t['frequency']),
                        amplitude=np.array(t['amplitude']),
                        label=t['label'],
                        timeseries = t['timeseries'],
                        spec_method = t['spec_method'],
                        spec_args =  t['spec_args'],
                        signif_qs = t['signif_qs'],
                    signif_method = t['signif_method'])
        d.append(PSD_obj)
    MPSD=pyleo.MultiplePSD(psd_list=d)
    return MPSD
        

def json_to_PSD(filename):
    '''
    Creates a PSD object from a JSON file

    Parameters
    ----------
    filename : str
        Name of the json file containing the necessary information

    Returns
    -------
    psd : pyleoclim.PSD
        a pyleoclim PSD object

    '''
    with  open(filename,'r') as f:
        t = json.load(f)
    t = list_to_array(t)
    
    #Deal with significance testing 
    if type(t['signif_qs']) is dict:
        c = PSD_to_MultiplePSD(t['signif_qs']['psd_list'])
    else:
        c = t['signif_qs']
        
    psd = pyleo.PSD(frequency=np.array(t['frequency']),
                    amplitude=np.array(t['amplitude']),
                    label=t['label'],
                    timeseries = pyleo.Series(time=np.array(t['timeseries']['time']),
                                             value=np.array(t['timeseries']['value']),
                                             time_name=t['timeseries']['time_name'],
                                             time_unit=t['timeseries']['time_unit'],
                                             value_name=t['timeseries']['value_name'],
                                             value_unit=t['timeseries']['value_unit'],
                                             label=t['timeseries']['label']),
                    spec_method = t['spec_method'],
                    spec_args =  t['spec_args'],
                    signif_qs = c,
                    signif_method = t['signif_method'])
    
    return  psd

def Scalogram_to_MultipleScalogram(series_list):
    '''
    Creates a MultipleScalogram object from a list of Scalogram objects

    Parameters
    ----------
    series_list : list
        List of Scalograms object

    Returns
    -------
    mscalogram : pyleoclim.MultipleScalogram
        A pyleoclim MultipleScalogram object.

    '''
    idx = np.arange(0,len(series_list),1).tolist()
    d=[]
    for item in idx:
        t=series_list[item]
        scalogram_obj= pyleo.Scalogram(frequency=np.array(t['frequency']),time=np.array(t['time']),
                               amplitude=np.array(t['amplitude']),coi=t['coi']
                               ,label=t['label'],timeseries=t['timeseries'],
                               wave_method = t['wave_method']
                               ,wave_args=t['wave_args'],
                               signif_qs = t['signif_qs'],
                               signif_method=t['signif_method'])

        d.append(scalogram_obj)
    mscalogram=pyleo.core.ui.MultipleScalogram(scalogram_list=d)
    return mscalogram

def json_to_Scalogram(filename):
    with open(filename,'r') as f:
        t = json.load(f)
    t = list_to_array(t)
    temp = t['timeseries']
    ts = pyleo.Series(time=np.array(temp['time']),
                     value=np.array(temp['value']),
                     time_name=temp['time_name'],
                     time_unit=temp['time_unit'],
                     value_name=temp['value_name'],
                     value_unit=temp['value_unit'],
                     label=t['label'])
    c = None
    if type(t['signif_qs']) is dict:
        c = Scalogram_to_MultipleScalogram(t['signif_qs']['scalogram_list'])
    else:
        c = t['signif_qs']
                             
    scalogram = pyleo.Scalogram(frequency=np.array(t['frequency']),time=np.array(t['time']),
                               amplitude=np.array(t['amplitude']),coi=t['coi']
                               ,label=t['label'],timeseries=ts,
                               wave_method = t['wave_method']
                               ,wave_args=t['wave_args'],
                               signif_qs = c,
                               signif_method=t['signif_method'])
    return scalogram