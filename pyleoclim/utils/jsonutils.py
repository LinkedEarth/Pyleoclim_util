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
        elif isinstance(obj_dict[k],(pyleo.core.ui.Series,pyleo.core.ui.Scalogram)):
            obj_dict[k]=PyleoObj_to_json(obj_dict[k],dict_return=True)
        elif isinstance(obj_dict[k],pyleo.core.ui.MultiplePSD):
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

def PyleoObj_to_json(PyleoObj,filename=None,dict_return=False):
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
    
    if filename is None and dict_return is False:
        raise ValueError('If a dictionary is not returned, filename must be provided')
    
    obj_dict = PyleoObj.__dict__
    obj_dict = transform(obj_dict)
    if dict_return == False:
        with open(filename,'w') as f:
            json.dump(obj_dict, f)
            f.close()
    elif dict_return == True:
        return  obj_dict
    
def open_json(filename):
    '''
    Open a json file.

    Parameters
    ----------
    filename : str
        Path to the json file

    Returns
    -------
    t : dict
        A Python diciotanary from the json

    '''
    with open(filename,'r') as f:
        t=json.load(f)
    return t

def json_to_Series(filename):
    '''
    Open a JSON file and returns a pyleoclim.Series object

    Parameters
    ----------
    filename : str
        The name of the JSON file containing the Series information. 

    Returns
    -------
    ts : pyleoclim.Series
        A pyleoclim.Series object

    '''
    
    t = open_json(filename)
    ts = pyleo.Series(time=np.array(t['time']),
                     value=np.array(t['value']),
                     time_name=t['time_name'],
                     time_unit=t['time_unit'],
                     value_name=t['value_name'],
                     value_unit=t['value_unit'],
                     label=t['label'],
                     clean_ts=t['clean_ts'], 
                     verbose=t['verbose']) 
    return ts

def json_to_LipdSeries(filename):
    '''
    Open a JSON file and returns a pyleoclim.LiPDSeries object. Note not extremely useful as compared to reopening the LiPD file.

    Parameters
    ----------
    filename : str
        The name of the JSON file containing the Series information

    Returns
    -------
    ts : pyleoclim.Series
        A pyleoclim.Series object

    '''
    t = open_json(filename)
    ts = pyleo.LipdSeries(tso=t['lipd_ts'],clean_ts=t['clean_ts'],verbose=t['verbose']) 
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
        if t['timeseries'] == None:
            v = t['timeseries']
        elif type(t['timeseries']) is dict:
            v = pyleo.Series(time=np.array(t['timeseries']['time']),
                             value=np.array(t['timeseries']['value']),
                             time_name=t['timeseries']['time_name'],
                             time_unit=t['timeseries']['time_unit'],
                             value_name=t['timeseries']['value_name'],
                             value_unit=t['timeseries']['value_unit'],
                             label=t['timeseries']['label'],
                             clean_ts=t['timeseries']['clean_ts'], 
                             verbose=t['timeseries']['verbose'])
        PSD_obj=pyleo.PSD(frequency=np.array(t['frequency']),
                        amplitude=np.array(t['amplitude']),
                        label=t['label'],
                        timeseries = v,
                        spec_method = t['spec_method'],
                        spec_args =  t['spec_args'],
                        signif_qs = t['signif_qs'],
                    signif_method = t['signif_method'],
                    plot_kwargs=t['plot_kwargs'],
                    period_unit=t['period_unit'],
                    beta_est_res=t['beta_est_res'])
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
                                             label=t['timeseries']['label'],
                                             clean_ts=t['timeseries']['clean_ts'], 
                                             verbose=t['timeseries']['verbose']),
                    spec_method = t['spec_method'],
                    spec_args =  t['spec_args'],
                    signif_qs = c,
                    signif_method = t['signif_method'],
                    plot_kwargs=t['plot_kwargs'],
                    period_unit=t['period_unit'],
                    beta_est_res=t['beta_est_res'])
    
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
        t = list_to_array(t)
        if t['timeseries'] == None:
            v = t['timeseries']
        elif type(t['timeseries']) is dict:
            v = pyleo.Series(time=np.array(t['timeseries']['time']),
                             value=np.array(t['timeseries']['value']),
                             time_name=t['timeseries']['time_name'],
                             time_unit=t['timeseries']['time_unit'],
                             value_name=t['timeseries']['value_name'],
                             value_unit=t['timeseries']['value_unit'],
                             label=t['timeseries']['label'],
                             clean_ts=t['timeseries']['clean_ts'], 
                             verbose=t['timeseries']['verbose'])
        scalogram_obj= pyleo.Scalogram(frequency=np.array(t['frequency']),time=np.array(t['time']),
                               amplitude=np.array(t['amplitude']),coi=t['coi']
                               ,label=t['label'],timeseries=v,
                               wave_method = t['wave_method']
                               ,wave_args=t['wave_args'],
                               signif_qs = t['signif_qs'],
                               signif_method=t['signif_method'],
                               freq_method=t['freq_method'],
                               freq_kwargs=t['freq_kwargs'],
                               period_unit=t['period_unit'],
                               time_label=t['time_label'],
                               wwz_Neffs=t['wwz_Neffs'],
                               signif_scals=t['signif_scals'])

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
                     label=temp['label'],
                     clean_ts=temp['clean_ts'], 
                     verbose=temp['verbose'])
    c = None
    if type(t['signif_qs']) is dict:
        c = Scalogram_to_MultipleScalogram(t['signif_qs']['scalogram_list'])
    else:
        c = t['signif_qs']
    
    d = None
    if type(t['signif_scals']) is dict:
       d = Scalogram_to_MultipleScalogram(t['signif_scals']['scalogram_list'])
    else:
       d = t['signif_scals']
        
    scalogram = pyleo.Scalogram(frequency=np.array(t['frequency']),time=np.array(t['time']),
                               amplitude=np.array(t['amplitude']),coi=t['coi']
                               ,label=t['label'],timeseries=ts,
                               wave_method = t['wave_method']
                               ,wave_args=t['wave_args'],
                               signif_qs = c,
                               signif_method=t['signif_method'],
                               freq_method=t['freq_method'],
                               freq_kwargs=t['freq_kwargs'],
                               period_unit=t['period_unit'],
                               time_label=t['time_label'],
                               wwz_Neffs=t['wwz_Neffs'],
                               signif_scals=d)
    
    return scalogram


def json_to_Coherence(filename):
    '''
    load a Coherence object from a JSON file. 

    Parameters
    ----------
    filename : str
        json file to unpack.

    Returns
    -------
    coherence : pyleoclim.Coherence
        A coherence object

    '''
    with open(filename,'r') as f:
        t = json.load(f)
    t = list_to_array(t)
    temp1 = t['timeseries1']
    ts1 = pyleo.Series(time=np.array(temp1['time']),
                     value=np.array(temp1['value']),
                     time_name=temp1['time_name'],
                     time_unit=temp1['time_unit'],
                     value_name=temp1['value_name'],
                     value_unit=temp1['value_unit'],
                     label=temp1['label'],
                     clean_ts=temp1['clean_ts'], 
                     verbose=temp1['verbose'])
    
    temp2 = t['timeseries2']
    ts2 = pyleo.Series(time=np.array(temp2['time']),
                     value=np.array(temp2['value']),
                     time_name=temp2['time_name'],
                     time_unit=temp2['time_unit'],
                     value_name=temp2['value_name'],
                     value_unit=temp2['value_unit'],
                     label=temp2['label'],
                     clean_ts=temp2['clean_ts'], 
                     verbose=temp2['verbose'])
    
    c = None
    
    if type(t['signif_qs']) is dict:
        c = Scalogram_to_MultipleScalogram(t['signif_qs']['scalogram_list'])
    else:
        c = t['signif_qs']
                             
    coherence = pyleo.Coherence(frequency=np.array(t['frequency']),time=np.array(t['time']),
                               coherence=np.array(t['coherence']),coi=t['coi'], phase=t['phase'],
                               timeseries1=ts1,timeseries2=ts2,
                               freq_method = t['freq_method'],
                               freq_kwargs=t['freq_kwargs'],
                               period_unit=t['period_unit'],
                               time_label=t['time_label'],
                               signif_qs = c,
                               signif_method=t['signif_method'])
    return coherence