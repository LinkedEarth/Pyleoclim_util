 # -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:07:07 2016

@author: deborahkhider

LiPD file manipulations. Except for maps, most manipulations are done on the timeseries objects.
See the LiPD documentation for more information on timeseries objects (TSO)

Also handles integration with the LinkedEarth wiki and the LinkedEarth Ontology

"""

__all__=['whatArchives',
        'whatProxyObservations',
        'whatProxySensors',
        'whatInferredVariables',
        'whatInterpretations',
        'queryLinkedEarth']

import lipd as lpd
import numpy as np
import os
import json
import requests
import wget
#from unidecode import unidecode
import string


"""
The following functions handle the LiPD files
"""

def enumerateLipds(lipds):
    """Enumerate the LiPD files loaded in the workspace

    Parameters
    ----------

    lipds : dict
        A dictionary of LiPD files.

    """
    print("Below are the available records")
    lipds_list = [val for val in lipds.keys()]
    for idx, val in enumerate(lipds_list):
        print(idx,': ',val)

def getLipd(lipds):
    """Prompt for a LiPD file

    Ask the user to select a LiPD file from a list
    Use this function in conjunction with enumerateLipds()

    Parameters
    ----------

    lipds : dict
        A dictionary of LiPD files. Can be obtained from
        pyleoclim.readLipd()

    Returns
    -------

    select_lipd : int
        The index of the LiPD file

    """
    enumerateLipds(lipds)
    lipds_list = [val for val in lipds.keys()]
    choice = int(input("Enter the number of the file: "))
    lipd_name = lipds_list[choice]
    select_lipd = lipds[lipd_name]

    return select_lipd


"""
The following functions work at the variables level
"""

def promptForVariable():
    """Prompt for a specific variable

    Ask the user to select the variable they are interested in.
    Use this function in conjunction with readHeaders() or getTSO()

    Returns
    -------

    select_var : int
        The index of the variable

    """
    select_var = int(input("Enter the number of the variable you wish to use: "))
    return select_var

def xAxisTs(timeseries):
    """ Get the x-axis for the timeseries.

    Parameters
    ----------

    timeseries : dict
        a timeseries object

    Returns
    -------

    x_axis : array
        the values for the x-axis representation
    label : string
        returns either "age", "year", or "depth"
    """

    if "depth" in timeseries.keys() and "age" in timeseries.keys() or\
            "depth" in timeseries.keys() and "year" in timeseries.keys():
        print("Both time and depth information available, selecting time")
        if "age" in timeseries.keys() and "year" in timeseries.keys():
            print("Both age and year representation available, selecting age")
            x_axis = timeseries["age"]
            label = "age"
        elif "year" in timeseries.keys():
            x_axis = timeseries["year"]
            label = "year"
        elif "age" in timeseries.keys():
            x_axis = timeseries["age"]
    elif "depth" in timeseries.keys():
        x_axis =  timeseries["depth"]
        label = "depth"
    elif "age" in timeseries.keys():
        x_axis = timeseries["age"]
        label = "age"
    elif "year" in timeseries.keys():
        x_axis = timeseries["year"]
        label = "year"
    else:
        raise KeyError("No age or depth information available")

    return x_axis, label

def checkXaxis(timeseries, x_axis= None):
    """Check that a x-axis is present for the timeseries

    Parameters
    ----------

    timeseries : dict
        a timeseries

    x_axis : string
        the x-axis representation, either depth, age or year

    Returns
    -------

    x : array
        the values for the x-axis representation

    label : string
        returns either "age", "year", or "depth"

    """
    if x_axis is None:
        x, label = xAxisTs(timeseries)
        x = np.array(x, dtype = 'float64')
    elif x_axis == "depth":
        if not "depth" in timeseries.keys():
            raise ValueError("Depth not available for this record")
        else:
            x = np.array(timeseries['depth'], dtype = 'float64')
            label = "depth"
    elif x_axis == "age":
        if not "age" in timeseries.keys():
            raise ValueError("Age not available for this record")
        else:
            x = np.array(timeseries['age'], dtype = 'float64')
            label = "age"
    elif x_axis == "year":
        if not "year" in timeseries.keys():
            raise ValueError("Year not available for this record")
        else:
            x = np.array(timeseries['year'], dtype = 'float64')
            label = "year"
    else:
        raise KeyError("enter either 'depth','age',or 'year'")

    return x, label

def checkTimeAxis(timeseries, x_axis = None):
    """ This function makes sure that time is available for the timeseries

    Parameters
    ----------

    timeseries : dict
        A LiPD timeseries object

    Returns
    -------

    x : array
        the time values for the timeseries

    label : string
        the time representation for the timeseries
    """
    if x_axis is None:
        if not 'age' in timeseries.keys() and not 'year' in timeseries.keys():
            raise KeyError("No time information available")
        elif 'age' in timeseries.keys() and 'year' in timeseries.keys():
            print("Both age and year information are available, using age")
            label = 'age'
        elif 'age' in timeseries.keys():
            label = 'age'
        elif 'year' in timeseries.keys():
            label = 'year'
    elif x_axis == 'age':
        if not 'age' in timeseries.keys():
            raise KeyError('Age is not available for this record')
        else:
            label = 'age'
    elif x_axis == 'year':
        if not 'year' in timeseries.keys():
            raise KeyError('Year is not available for this record')
        else:
            label='year'
    else:
        raise KeyError('Only None, year and age are valid entries for x_axis parameter')

    x = np.array(timeseries[label], dtype = 'float64')

    return x, label

def searchVar(timeseries_list, key, exact = True, override = True):
    """ This function searched for keywords (exact match) for a variable

    Parameters
    ----------

    timeseries_list : list
        A list of available series

    key : list
        A list of keys to search

    exact : bool
        if True, looks for an exact match.

    override : bool
        if True, override the exact match if no match is found

    Returns
    -------

    match : list
        A list of keys for the timeseries that match the selection
        criteria.
    """

    # Make sure thaat the keys are contained in a list
    if type(key) is not list:
       if type(key) is str:
           key = [key]
       else:
           raise TypeError("Key terms should be entered as a list")

    match = []

    if exact == True:
    #Search for exact match with the key
        for keyVal in key:
            for val in timeseries_list.keys():
                ts_temp = timeseries_list[val]
                if "variableName" in ts_temp.keys():
                    name = ts_temp["variableName"]
                    if keyVal.lower() == name.lower():
                        match.append(val)
                elif "paleoData_variableName" in ts_temp.keys():
                    name = ts_temp["paleoData_variableName"]
                    if keyVal.lower() == name.lower():
                        match.append(val)
                elif "chronData_variableName" in ts_temp.keys():
                    name = ts_temp["chronData_variableName"]
                    if keyVal.lower() == name.lower():
                        match.append(val)
                elif "ProxyObservationType" in ts_temp.keys():
                    name = ts_temp["ProxyObservationType"]
                    if keyVal.lower() == name.lower():
                        match.append(val)
                elif "paleoData_proxyObservationType" in ts_temp.keys():
                    name = ts_temp["paleoData_proxyObservationType"]
                    if keyVal.lower() == name.lower():
                        match.append(val)
                elif "chronData_proxyObservationType" in ts_temp.keys():
                    name = ts_temp["chronData_proxyObservationType"]
                    if keyVal.lower() == name.lower():
                        match.append(val)
                elif "InferredVariableType" in ts_temp.keys():
                    name = ts_temp["InferredVariableType"]
                    if keyVal.lower() == name.lower():
                        match.append(val)
                elif "paleoData_inferredVariableType" in ts_temp.keys():
                    name = ts_temp["paleoData_inferredVariableType"]
                    if keyVal.lower() == name.lower():
                        match.append(val)
                elif "chronData_inferredVariableType" in ts_temp.keys():
                    name = ts_temp["chronData_inferredVariableType"]
                    if keyVal.lower() == name.lower():
                        match.append(val)
    else:
    # Search for the word in the ley
        for keyVal in key:
            for val in timeseries_list.keys():
                ts_temp = timeseries_list[val]
                if "variableName" in ts_temp.keys():
                    name = ts_temp["variableName"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "paleoData_variableName" in ts_temp.keys():
                    name = ts_temp["paleoData_variableName"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "chronData_variableName" in ts_temp.keys():
                    name = ts_temp["chronData_variableName"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "ProxyObservationType" in ts_temp.keys():
                    name = ts_temp["ProxyObservationType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "paleoData_proxyObservationType" in ts_temp.keys():
                    name = ts_temp["paleoData_proxyObservationType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "chronData_proxyObservationType" in ts_temp.keys():
                    name = ts_temp["chronData_proxyObservationType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "InferredVariableType" in ts_temp.keys():
                    name = ts_temp["InferredVariableType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "paleoData_inferredVariableType" in ts_temp.keys():
                    name = ts_temp["paleoData_inferredVariableType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "chronData_inferredVariableType" in ts_temp.keys():
                    name = ts_temp["chronData_inferredVariableType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)

    # Expand the search if asked
    if not match and exact == True and override == True:
        print("No match found on exact search, running partial match")
        for keyVal in key:
            for val in timeseries_list.keys():
                ts_temp = timeseries_list[val]
                if "variableName" in ts_temp.keys():
                    name = ts_temp["variableName"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "paleoData_variableName" in ts_temp.keys():
                    name = ts_temp["paleoData_variableName"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "chronData_variableName" in ts_temp.keys():
                    name = ts_temp["chronData_variableName"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "ProxyObservationType" in ts_temp.keys():
                    name = ts_temp["ProxyObservationType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "paleoData_proxyObservationType" in ts_temp.keys():
                    name = ts_temp["paleoData_proxyObservationType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "chronData_proxyObservationType" in ts_temp.keys():
                    name = ts_temp["chronData_proxyObservationType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "InferredVariableType" in ts_temp.keys():
                    name = ts_temp["InferredVariableType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "paleoData_inferredVariableType" in ts_temp.keys():
                    name = ts_temp["paleoData_inferredVariableType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)
                elif "chronData_inferredVariableType" in ts_temp.keys():
                    name = ts_temp["chronData_inferredVariableType"]
                    if keyVal.lower() in name.lower():
                        match.append(val)

    # Get the unique entries
    match = list(set(match))

    # Narrow down if more than one match is found by asking the user
    if len(match) > 1:
        print("More than one series match your search criteria")
        for idx, val in enumerate(match):
            print(idx,": ", val)
        choice = int(input("Enter the number for the variable: "))
        match = match[choice]
    elif not match:
        print("No match found.")
        print("Here are the available variables: ")
        v = list(timeseries_list.keys())
        for idx, val in enumerate(v):
            print(idx,": ",val)
        choice = input("Please select the variable you'd like to use or enter to continue: ")
        if not choice:
            match =""
        else:
            choice = int(choice)
            match = v[choice]
    else:
        match = match[0]

    return match

"""
The following functions handle the time series objects
"""
def enumerateTs(timeseries_list):
    """Enumerate the available time series objects

    Parameters
    ----------

    timeseries_list : list
        a  list of available timeseries objects.


    """
    available_y = []
    dataSetName =[]
    at =[]
    for item in timeseries_list:
        if 'dataSetName' in item.keys():
            dataSetName.append(item['dataSetName'])
        else:
            dataSetName.append('NA')
        if 'paleoData_variableName' in item.keys():
            available_y.append(item['paleoData_variableName'])
        elif 'chronData_variableName' in item.keys():
            available_y.append(item['chronData_variableName'])
        else:
            available_y.append('NA')
        if 'archiveType' in item.keys():
            at.append(item['archiveType'])
        else:
            at.append('NA')

    for idx,val in enumerate(available_y):
        print(idx,': ',dataSetName[idx], ': ',at[idx],': ', val)

def getTs(timeseries_list, option = None):
    """Get a specific timeseries object from a dictionary of timeseries

    Parameters
    ----------

    timeseries_list : list
        a  list of available timeseries objects.

    option : string
        An expression to filter the datasets. Uses lipd.filterTs()

    Returns
    -------

    timeseries : single timeseries object or list of timeseries
        A single timeseries object if not optional filter selected or a filtered
        list if optional arguments given

    """
    if not option:
        enumerateTs(timeseries_list)
        select_TSO = promptForVariable()
        timeseries = timeseries_list[select_TSO]
    else:
        timeseries = lpd.filterTs(timeseries_list, option)

    return timeseries

"""
Functions to handle data on the LinkedEarth Platform
"""
def LipdToOntology(archiveType):
    """ standardize archiveType

    Transform the archiveType from their LiPD name to their ontology counterpart

    Parameters
    ----------

    archiveType : string
        name of the archiveType from the LiPD file

    Returns
    -------

    archiveType : string
        archiveType according to the ontology

    """
    #Align with the ontology
    if archiveType.lower().replace(" ", "") == "icecore":
        archiveType = 'glacierice'
    elif archiveType.lower().replace(" ", "") == "ice-other":
        archiveType = 'glacierice'
    elif archiveType.lower().replace(" ", "") == 'tree':
        archiveType = 'wood'
    elif archiveType.lower().replace(" ", "") == 'borehole':
        archiveType = 'ice/rock'
    elif archiveType.lower().replace(" ", "") == 'bivalve':
        archiveType = 'molluskshells'

    return archiveType

def timeUnitsCheck(units):
    """ This function attempts to make sense of the time units by checking for equivalence

    Parameters
    ----------

    units : string
        The units string for the timeseries

    Returns
    -------

    unit_group : string
        Whether the units belongs to age_units, kage_units, year_units, ma_units, or undefined
    """

    age_units = ['year B.P.','yr B.P.','yr BP','BP','yrs BP','years B.P.',\
                 'yr. BP','yr. B.P.', 'cal. BP', 'cal B.P.', \
                 'year BP','years BP']
    kage_units = ['kyr BP','kaBP','ka BP','ky','kyr','kyr B.P.', 'ka B.P.', 'ky BP',\
                  'kyrs BP','ky B.P.', 'kyrs B.P.', 'kyBP', 'kyrBP']
    year_units = ['AD','CE','year C.E.','year A.D.', 'year CE','year AD',\
                  'years C.E.','years A.D.','yr CE','yr AD','yr C.E.'\
                  'yr A.D.', 'yrs C.E.', 'yrs A.D.', 'yrs CE', 'yrs AD']
    mage_units = ['my BP', 'myr BP', 'myrs BP', 'ma BP', 'ma',\
                  'my B.P.', 'myr B.P.', 'myrs B.P.', 'ma B.P.']
    undefined = ['years', 'yr','year','yrs']

    if units in age_units:
        unit_group = 'age_units'
    elif units in kage_units:
        unit_group = 'kage_units'
    elif units in year_units:
        unit_group = 'year_units'
    elif units in undefined:
        unit_group = 'undefined'
    elif units in mage_units:
        unit_group = 'ma_units'
    else:
        unit_group = 'unknown'

    return unit_group



def whatArchives(print_response=True):
    """ Get the names for ArchiveType from LinkedEarth Ontology

    Parameters
    ----------

    print_response : bool
        Whether to print the results on the console. Default is True

    Returns
    -------

    res : JSON-object with the request from LinkedEarth wiki api

    """
    url = "http://wiki.linked.earth/store/ds/query"

    query = """PREFIX core: <http://linked.earth/ontology#>
    PREFIX wiki: <http://wiki.linked.earth/Special:URIResolver/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT distinct ?a
    WHERE {
    {
        ?dataset wiki:Property-3AArchiveType ?a.
    }UNION
    {
        ?w core:proxyArchiveType ?t.
        ?t rdfs:label ?a
    }
    }"""

    response = requests.post(url, data = {'query': query})
    res_i = json.loads(response.text)

    if print_response == True:
        print("The following archive types are available on the wiki:")
        for item in res_i['results']['bindings']:
            print ("*" + item['a']['value'])
    res=[]
    for item in res_i['results']['bindings']:
        res.append(item['a']['value'])

    return res

def whatProxyObservations(print_response=True):
    """ Get the names for ProxyObservations from LinkedEarth Ontology

    Parameters
    ----------

    print_response : bool
        Whether to print the results on the console. Default is True

    Returns
    -------

    res : JSON-object with the request from LinkedEarth wiki api
    """

    url = "http://wiki.linked.earth/store/ds/query"

    query = """PREFIX core: <http://linked.earth/ontology#>
    PREFIX wiki: <http://wiki.linked.earth/Special:URIResolver/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT distinct ?a
    WHERE
    {
        ?w core:proxyObservationType ?t.
        ?t rdfs:label ?a
    }"""

    response = requests.post(url, data = {'query': query})
    res_i = json.loads(response.text)

    if print_response==True:
        print("The following proxy observation types are available on the wiki: ")
        for item in res_i['results']['bindings']:
            print (item['a']['value'])

    res=[]
    for item in res_i['results']['bindings']:
        res.append(item['a']['value'])

    return res

def whatProxySensors(print_response=True):
    """ Get the names for ProxySensors from LinkedEarth Ontology

    Parameters
    ----------

    print_response : bool
        Whether to print the results on the console. Default is True

    Returns
    -------

    res : JSON-object with the request from LinkedEarth wiki api
    """
    url = "http://wiki.linked.earth/store/ds/query"

    query = """PREFIX core: <http://linked.earth/ontology#>
    PREFIX wiki: <http://wiki.linked.earth/Special:URIResolver/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT distinct ?a ?b
    WHERE
    {
        ?w core:sensorGenus ?a.
        ?w core:sensorSpecies ?b .

    }"""

    response = requests.post(url, data = {'query': query})
    res_i = json.loads(response.text)

    if print_response == True:
        print("The available sensor genus/species are: ")
        for item in res_i['results']['bindings']:
            print ("*" + 'Genus: '+item['a']['value']+' Species: ' +item['b']['value'])

    res=[]
    for item in res_i['results']['bindings']:
        res.append(item['a']['value'])

    return res

def whatInferredVariables(print_response=True):
    """ Get the names for InferredVariables from LinkedEarth Ontology

    Parameters
    ----------

    print_response : bool
        Whether to print the results on the console. Default is True

    Returns
    -------

    res : JSON-object with the request from LinkedEarth wiki api
    """
    url = "http://wiki.linked.earth/store/ds/query"

    query = """PREFIX core: <http://linked.earth/ontology#>
    PREFIX wiki: <http://wiki.linked.earth/Special:URIResolver/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT distinct ?a
    WHERE
    {
        ?w core:inferredVariableType ?t.
        ?t rdfs:label ?a
    }"""

    response = requests.post(url, data = {'query': query})
    res_i = json.loads(response.text)

    if print_response == True:
        print("The following Inferred Variable types are available on the wiki: ")
        for item in res_i['results']['bindings']:
            print ("*" + item['a']['value'])
            res=[]
    for item in res_i['results']['bindings']:
        res.append(item['a']['value'])
    return res

def whatInterpretations(print_response=True):
    """ Get the names for interpretations from LinkedEarth Ontology

    Parameters
    ----------

    print_response : bool
        Whether to print the results on the console. Default is True

    Returns
    -------

    res : JSON-object with the request from LinkedEarth wiki api
    """

    url = "http://wiki.linked.earth/store/ds/query"

    query = """PREFIX core: <http://linked.earth/ontology#>
    PREFIX wiki: <http://wiki.linked.earth/Special:URIResolver/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT distinct ?a ?b
    WHERE
    {
        ?w core:name ?a.
        ?w core:detail ?b .

    }"""

    response = requests.post(url, data = {'query': query})
    res_i = json.loads(response.text)

    if print_response==True:
        print("The following interpretation are available on the wiki: ")
        for item in res_i['results']['bindings']:
            print ("*" + 'Name: '+item['a']['value']+' Detail: ' +item['b']['value'])
    res=[]
    for item in res_i['results']['bindings']:
        res.append(item['a']['value'])
    return res

def queryLinkedEarth(archiveType=[ ], proxyObsType=[ ], infVarType = [ ], sensorGenus=[ ],
                    sensorSpecies=[ ], interpName =[ ], interpDetail =[ ], ageUnits = [ ],
                    ageBound = [ ], ageBoundType = [ ], recordLength = [ ], resolution = [ ],
                    lat = [ ], lon = [ ], alt = [ ], print_response = True, download_lipd = True,
                    download_folder = 'default'):
    """ This function allows to query the LinkedEarth wiki for records.

    This function allows to query the LinkedEarth wiki for specific catagories.
    If you have more than one keyword per catagory, enter them in a list. If you don't
    wish to use a particular terms, leave a blank in-between the brackets.

    Parameters
    ----------

    archiveType : list of strings
        The type of archive (enter all query terms, separated by a comma)

    proxyObsType : list of strings
        The type of proxy observation (enter all query terms, separated by a comma)

    infVarType : list of strings
        The type of inferred variable (enter all query terms, separated by a comma)

    sensorGenus : list of strings
        The Genus of the sensor (enter all query terms, separated by a comma)

    sensorSpecies : list of strings
        The Species of the sensor (enter all query terms, separated by a comma)

    interpName : list of strings
        The name of the interpretation (enter all query terms, separated by a comma)

    interpDetail : list of strings
        The detail of the interpretation (enter all query terms, separated by a comma)

    ageUnits : list of strings
        The units of in which the age (year) is expressed in.
        Warning: Separate each query if need to run across multiple age queries (i.e., yr B.P. vs kyr B.P.). If the units are different but the meaning is the same (e.g., yr B.P. vs yr BP, enter all search terms separated by a comma).

    ageBound : list of floats
        Enter the minimum and maximum age value to search for.
        Warning: You MUST enter a minimum AND maximum value. If you wish to perform a query such as "all ages before 2000 A.D.", enter a minimum value of -99999 to cover all bases.

    ageBoundType : list of strings
        The type of querying to perform. Possible values include: "any", "entire", and "entirely".
        - any: Overlap any portions of matching datasets (default)
        - entirely: are entirely overlapped by matching datasets
        - entire: overlap entire matching datasets but dataset can be shorter than the bounds

    recordLength : list of floats
        The minimum length the record needs to have while matching the ageBound criteria. For instance, "look for all records between 3000 and 6000 year BP with a record length of at least 1500 year".

    resolution : list of floats
        The maximum resolution of the resord. Resolution has the same units as age/year. For instance, "look for all records with a resolution of at least 100 years".
        Warning: Resolution applies to specific variables rather than an entire dataset. Imagine the case where some measurements are made every cm while others are made every 5cm. If you require a specific variable to have the needed resolution, make sure that either the proxyObservationType, inferredVariableType, and/or Interpretation fields are completed.

    lat : list of floats
        The minimum and maximum latitude. South is expressed with negative numbers.
        Warning: You MUST enter a minimum AND maximum value. If you wish to perform a query looking for records from the Northern Hemisphere, enter [0,90].

    lon : list of floats
        The minimum and maximum longitude. West is expressed with negative numbers.
        Warning: You MUST enter a minimum AND a maximum value. If you wish to perform a query looking for records from the Western Hemisphere, enter [-180,0].

    alt : list of floats
        The minimum and maximum altitude. Depth below sea level is expressed as negative numbers.
        Warning: You MUST enter a minimum AND a maximum value. If you wish to perform a query looking for records below a certain depth (e.g., 500), enter [-99999,-500].

    print_response : bool
        If True, prints the URLs to the matching LiPD files

    download_lipd : bool
        If True, download the matching LiPD files

    download_folder : string
        Location to download the LiPD files. If "default", will download in the current directory.

    Returns
    -------

    res : the response to the query

    """
    # Perform a lot of checks
    if len(ageBound)==1:
        raise ValueError("You need to provide a minimum and maximum boundary.")

    if ageBound and not ageUnits:
        raise ValueError("When providing age limits, you must also enter the units")

    if recordLength and not ageUnits:
        raise ValueError("When providing a record length, you must also enter the units")

    if ageBound and ageBound[0]>ageBound[1]:
        ageBound = [ageBound[1],ageBound[0]]

    if len(ageBoundType)>1:
        raise ValueError("Only one search possible at a time.")
        if ageBoundType not in ["any","entirely","entire"]:
            raise ValueError("ageBoundType is not recognized")

    if recordLength and ageBound and recordLength[0] > (ageBound[1]-ageBound[0]):
        raise ValueError("The required recordLength is greater than the provided age bounds")

    if len(resolution)>1:
        raise ValueError("You can only search for a maximum resolution one at a time.")

    if len(lat)==1:
        raise ValueError("Please enter a lower AND upper boundary for the latitude search")

    if lat and lat[1]<lat[0]:
        lat = [lat[1],lat[0]]

    if len(lon)==1:
        raise ValueError("Please enter a lower AND upper boundary for the longitude search")

    if lon and lon[1]<lon[0]:
        lon = [lon[1],lon[0]]

    if len(alt)==1:
        raise ValueError("Please enter a lower AND upper boundary for the altitude search")

    if alt and alt[1]<alt[0]:
        alt = [alt[1],alt[0]]

    # Perform the query
    url = "http://wiki.linked.earth/store/ds/query"

    query = """PREFIX core: <http://linked.earth/ontology#>
    PREFIX wiki: <http://wiki.linked.earth/Special:URIResolver/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT  distinct ?dataset ?dataset_label
    WHERE {
    ?dataset rdfs:label ?dataset_label
    """

    ### Look for data field
    dataQ=""
    if archiveType or proxyObsType or infVarType or sensorGenus or sensorSpecies or interpName or interpDetail or ageUnits or ageBound or recordLength or resolution:
        dataQ = "?dataset core:includesChronData|core:includesPaleoData ?data."

    ### Look for variable
    ## measuredVariable
    measuredVarQ=""
    if proxyObsType or archiveType or sensorGenus or sensorSpecies or interpName or interpDetail or resolution:
        measuredVarQ = "?data core:foundInMeasurementTable / core:includesVariable ?v."

    ## InferredVar
    inferredVarQ=""
    if infVarType or interpName or interpDetail or resolution:
        inferredVarQ = "?data core:foundInMeasurementTable / core:includesVariable ?v1."

    ### Archive Query
    archiveTypeQ=""
    if len(archiveType)>0:
        #add values for the archiveType
        query += "VALUES ?a {"
        for item in archiveType:
            query +="\""+item+"\" "
        query += "}\n"
        # Create the query
        archiveTypeQ = """
    #Archive Type query
    {
        ?dataset wiki:Property-3AArchiveType ?a.
    }UNION
    {
        ?p core:proxyArchiveType / rdfs:label ?a.
    }"""

    ### ProxyObservationQuery
    proxyObsTypeQ=""
    if len(proxyObsType)>0:
       #  add values for the proxyObservationType
       query+="VALUES ?b {"
       for item in proxyObsType:
           query += "\""+item+"\""
       query += "}\n"
       # Create the query
       proxyObsTypeQ="?v core:proxyObservationType/rdfs:label ?b."

    ### InferredVariableQuery
    infVarTypeQ=""
    if len(infVarType)>0:
        query+="VALUES ?c {"
        for item in infVarType:
            query+="\""+item+"\""
        query+="}\n"
        # create the query
        infVarTypeQ="""
    ?v1 core:inferredVariableType ?t.
    ?t rdfs:label ?c.
    """
    ### ProxySensorQuery
    sensorQ=""
    if len(sensorGenus)>0 or len(sensorSpecies)>0:
        sensorQ="""
    ?p core:proxySensorType ?sensor.
    """
    ## Genus query
    genusQ=""
    if len(sensorGenus)>0:
        query+="VALUES ?genus {"
        for item in sensorGenus:
            query+="\""+item+"\""
        query+="}\n"
        # create the query
        genusQ = "?sensor core:sensorGenus ?genus."

    ## Species query
    speciesQ=""
    if len(sensorSpecies)>0:
        query+=  "VALUES ?species {"
        for item in sensorSpecies:
            query+="\""+item+"\""
        query+="}\n"
        #Create the query
        speciesQ = "?sensor core:sensorSpecies ?species."

    ### Proxy system query
    proxySystemQ = ""
    if len(archiveType)>0 or len(sensorGenus)>0 or len(sensorSpecies)>0:
        proxySystemQ="?v ?proxySystem ?p."

    ### Deal with interpretation
    ## Make sure there is an interpretation to begin with
    interpQ = ""
    if len(interpName)>0 or len(interpDetail)>0:
        interpQ= """
    {?v1 core:interpretedAs ?interpretation}
    UNION
    {?v core:interpretedAs ?interpretation}
    """

    ## Name
    interpNameQ=""
    if len(interpName)>0:
        query+= "VALUES ?intName {"
        for item in interpName:
            query+="\""+item+"\""
        query+=  "}\n"
        #Create the query
        interpNameQ = "?interpretation core:name ?intName."

    ## detail
    interpDetailQ = ""
    if len(interpDetail)>0:
        query+= "VALUES ?intDetail {"
        for item in interpDetail:
            query+="\""+item+"\""
        query+="}\n"
        #Create the query
        interpDetailQ = "?interpretation core:detail ?intDetail."

    ### Age
    ## Units
    ageUnitsQ = ""
    if len(ageUnits)>0:
        query+= "VALUES ?units {"
        for item in ageUnits:
            query+="\""+item+"\""
        query+="}\n"
        query+="""VALUES ?ageOrYear{"Age" "Year"}\n"""
        # create the query
        ageUnitsQ ="""
    ?data core:foundInMeasurementTable / core:includesVariable ?v2.
    ?v2 core:inferredVariableType ?aoy.
    ?aoy rdfs:label ?ageOrYear.
    ?v2 core:hasUnits ?units .
    """
    ## Minimum and maximum
    ageQ = ""
    if ageBoundType[0] == "entirely":
        if len(ageBound)>0 and len(recordLength)>0:
            ageQ="""
    ?v2 core:hasMinValue ?e1.
    ?v2 core:hasMaxValue ?e2.
    filter(?e1<=""" +str(ageBound[0])+ """&& ?e2>="""+str(ageBound[1])+""" && abs(?e1-?e2)>="""+str(recordLength[0])+""").
    """
        elif len(ageBound)>0 and len(recordLength)==0:
            ageQ="""
    ?v2 core:hasMinValue ?e1.
    ?v2 core:hasMaxValue ?e2.
    filter(?e1<=""" +str(ageBound[0])+ """&& ?e2>="""+str(ageBound[1])+""").
    """
    elif ageBoundType[0] == "entire":
        if len(ageBound)>0 and len(recordLength)>0:
            ageQ="""
    ?v2 core:hasMinValue ?e1.
    ?v2 core:hasMaxValue ?e2.
    filter(?e1>=""" +str(ageBound[0])+ """&& ?e2<="""+str(ageBound[1])+""" && abs(?e1-?e2)>="""+str(recordLength[0])+""").
    """
        elif len(ageBound)>0 and len(recordLength)==0:
            ageQ="""
    ?v2 core:hasMinValue ?e1.
    ?v2 core:hasMaxValue ?e2.
    filter(?e1>=""" +str(ageBound[0])+ """&& ?e2<="""+str(ageBound[1])+""").
    """
    elif ageBoundType[0] == "any":
        if len(ageBound)>0 and len(recordLength)>0:
            ageQ="""
    ?v2 core:hasMinValue ?e1.
    filter(?e1<=""" +str(ageBound[1])+ """ && abs(?e1-"""+str(ageBound[1])+""")>="""+str(recordLength[0])+""").
    """
        elif len(ageBound)>0 and len(recordLength)==0:
            ageQ="""
    ?v2 core:hasMinValue ?e1.
    filter(?e1<=""" +str(ageBound[1])+ """).
    """

    ### Resolution
    resQ=""
    if len(resolution)>0:
        resQ = """
    {
    ?v core:hasResolution/(core:hasMeanValue |core:hasMedianValue) ?resValue.
    filter (xsd:float(?resValue)<100)
    }
    UNION
    {
    ?v1 core:hasResolution/(core:hasMeanValue |core:hasMedianValue) ?resValue1.
    filter (xsd:float(?resValue1)<"""+str(resolution[0])+""")
    }
    """

    ### Location
    locQ=""
    if lon or lat or alt:
           locQ = "?dataset core:collectedFrom ?z."

    ## Altitude
    latQ=""
    if len(lat)>0:
        latQ="""
    ?z <http://www.w3.org/2003/01/geo/wgs84_pos#lat> ?lat.
    filter(xsd:float(?lat)<"""+str(lat[1])+""" && xsd:float(?lat)>"""+str(lat[0])+""").
    """

    ##Longitude
    lonQ=""
    if len(lon)>0:
        lonQ = """
    ?z <http://www.w3.org/2003/01/geo/wgs84_pos#long> ?long.
    filter(xsd:float(?long)<"""+str(lon[1])+""" && xsd:float(?long)>"""+str(lon[0])+""").
    """

    ## Altitude
    altQ=""
    if len(alt)>0:
        altQ="""
    ?z <http://www.w3.org/2003/01/geo/wgs84_pos#alt> ?alt.
    filter(xsd:float(?alt)<"""+str(alt[1])+""" && xsd:float(?alt)>"""+str(alt[0])+""").
    """

    query += """
    ?dataset a core:Dataset.
    """+dataQ+"""
    """+measuredVarQ+"""
    # By proxyObservationType
    """+proxyObsTypeQ+"""
    """+inferredVarQ+"""
    # By InferredVariableType
    """+infVarTypeQ+"""
    # Look for the proxy system model: needed for sensor and archive queries
    """+proxySystemQ+"""
    # Sensor query
    """+sensorQ+"""
    """+genusQ+"""
    """+speciesQ+"""
    # Archive query (looks in both places)
    """+archiveTypeQ+"""
    # Interpretation query
    """+interpQ+"""
    """+interpNameQ+"""
    """+interpDetailQ+"""
    # Age Query
    """+ageUnitsQ+"""
    """+ageQ+"""
    # Location Query
    """+locQ+"""
    #Latitude
    """+latQ+"""
    #Longitude
    """+lonQ+"""
    #Altitude
    """+altQ+"""
    #Resolution Query
    """+resQ+"""
    }"""

    #print(query)
    response = requests.post(url, data = {'query': query})
    res = json.loads(response.text)
    if print_response == True:
        for item in res['results']['bindings']:
            print (item['dataset']['value'])

    #download files
    if download_lipd == True:
        for item in res['results']['bindings']:
            dataset = (item['dataset_label']['value'])
            download_url = 'http://wiki.linked.earth/wiki/index.php/Special:WTLiPD?op=export&lipdid='+dataset
            if download_folder == 'default':
                path = os.getcwd()+'/'
            else:
                if download_folder[-1] == '/':
                    path = download_folder
                    wget.download(download_url, path)
                else:
                    path = download_folder+'/'
            if os.path.exists(path) == False:
                os.mkdir(path)
            wget.download(download_url, path+'/'+dataset+'.lpd')

    return res

def pre_process_list(list_str):
    """ Pre-process a series of strings for capitalized letters, space, and punctuation

    Parameters
    ----------

    list_str : list
        A list of strings from which to strip capitals, spaces, and other characters

    Returns
    -------

    res : list
        A list of strings with capitalization, spaces, and punctuation removed
    """
    res=[]
    for item in list_str:
        res.append(pre_process_str(item))
    return res

def similar_string(list_str, search):
    """ Returns a list of indices for strings with similar values

    Parameters
    ----------

    list_str : list
        A list of strings

    search : str
        A keyword search

    Returns
    -------

    indices: list
        A list of indices with similar value as the keyword
    """
    #exact matches
    indices = [i for i, x in enumerate(list_str) if x == search]
    # proximity matches
    return indices

def pre_process_str(word):
    """Pre-process a string for capitalized letters, space, and punctuation

    Parameters
    ----------

    string : str
        A string from which to strip capitals, spaces, and other characters

    Returns
    -------
    res : str
        A string with capitalization, spaces, and punctuation removed
    """
    d=word.replace(" ","").lower()
    stopset=list(string.punctuation)
    res="".join([i for i in d if i not in stopset])
    return res

"""
Deal with models
"""
def isModel(csvName, lipd):
    """Check for the presence of a model in the same object as the measurement table

    Parameters
    ----------

    csvName : string
        The name of the csv file corresponding to the measurement table

    lipd : dict
        A LiPD object

    Returns
    -------

    model : list
        List of models already available

    dataObject : string
        The name of the paleoData or ChronData
        object in which the model(s) are stored
    """
    csvNameSplit = csvName.split('.')
    for val in csvNameSplit:
        if "chron" in val or "paleo" in val:
            tableName = val

    if tableName[0] == 'c':
        objectName = 'chron'+tableName.split('chron')[1][0]
        dataObject = lipd["chronData"][objectName]
    elif tableName[0] == 'p':
        objectName = 'paleo'+tableName.split('paleo')[1][0]
        dataObject = lipd["paleoData"][objectName]
    else:
        raise KeyError("Key name should only include 'chron' or 'paleo'")

    if "model" in dataObject.keys():
        model_list = dataObject["model"]
        model = list(model_list.keys())
    else:
        model=[]

    return model, objectName


def modelNumber(model):
    """Assign a new or existing model number

    Parameters
    ----------

    model : list
        List of possible model number. Obtained from isModel

    Returns
    -------

    modelNum : int
        The number of the model
    """
    if model:
        print("There is " + str(len(model)) + " model(s) already available.")
        print("creating a new model...")
        modelNum = len(model)
        print("Your new model number is "+ str(modelNum))
    else:
        print("No previous model available. Creating a new model...")
        modelNum = 0
        print("Your model number is "+ str(modelNum))

    return modelNum

"""
Get entire tables
"""

def isMeasurement(csv_dict):
    """ Check whether measurement tables are available

    Parameters
    ----------

    csv_dict : dict
        Dictionary of available csv

    Returns
    -------

    paleoMeasurementTables : list
        List of available paleoMeasurementTables
    chronMeasurementTables : list
        List of available chronMeasurementTables
    """
    chronMeasurementTables = []
    paleoMeasurementTables =[]

    for val in csv_dict.keys():
        if "measurement" in val and "chron" in val:
            chronMeasurementTables.append(val)
        if "measurement" in val and "paleo" in val:
            paleoMeasurementTables.append(val)

    return chronMeasurementTables, paleoMeasurementTables

def whichMeasurement(measurementTableList):
    """Select a measurement table from a list

    Use in conjunction with the function isMeasurement

    Parameters
    ----------

    measurementTableList : list
        List of measurement tables contained in the LiPD file. Output from the isMeasurement function

    Returns
    -------

    csvName : string
        the name of the csv file

    """
    if len(measurementTableList)>1:
        print("More than one table is available.")
        for idx, val in enumerate(measurementTableList):
            print(idx, ": ", val)
        csvName = measurementTableList[int(input("Which one would you like to use? "))]
    else:
        csvName = measurementTableList[0]

    return csvName

def getMeasurement(csvName, lipd):
    """Extract the dictionary corresponding to the measurement table

    Parameters
    ----------

    csvName : string
        The name of the csv file
    lipd : dict
        The LiPD object from which to extract the data

    Returns
    -------

    ts_list : dict
        A dictionary containing data and metadata for each column in the
        csv file.

    """
    csvNameSplit = csvName.split('.')
    for val in csvNameSplit:
        if "chron" in val or "paleo" in val:
            tableName = val

    if tableName[0] == 'c':
        objectName = 'chron'+tableName.split('chron')[1][0]
        ts_list = lipd["chronData"][objectName]["measurementTable"][tableName]["columns"]
    elif tableName[0] == 'p':
        objectName = 'paleo'+tableName.split('paleo')[1][0]
        ts_list = lipd["paleoData"][objectName]["measurementTable"][tableName]["columns"]
    else:
        raise KeyError("Key name should only include 'chron' or 'paleo'")

    return ts_list

"""
Deal with ensembles
"""

def isEnsemble(csv_dict):
    """ Check whether ensembles are available

    Parameters
    ----------

    csv_dict : dict
        Dictionary of available csv

    Returns
    -------

    paleoEnsembleTables : list
        List of available paleoEnsembleTables
    chronEnsembleTables : list
        List of availale chronEnsemble Tables

    """
    chronEnsembleTables =[]
    paleoEnsembleTables =[]
    for val in csv_dict.keys():
        if "ensemble" in val and "chron" in val:
            chronEnsembleTables.append(val)
        elif "ensemble" in val and "paleo" in val:
            paleoEnsembleTables.append(val)

    return chronEnsembleTables, paleoEnsembleTables

def whichEnsemble(ensembleTableList):
    """Select an ensemble table from a list

    Use in conjunction with the function isMeasurement

    Parameters
    ----------

    measurementTableList : list
        List of measurement tables contained in the LiPD file. Output from the isMeasurement function
    csv_list : list
        Dictionary of available csv

    Returns
    -------

    csvName : string
        the name of the csv file

    """
    if len(ensembleTableList)>1:
        print("More than one table is available.")
        for idx, val in enumerate(ensembleTableList):
            print(idx, ": ", val)
        csvName = ensembleTableList[int(input("Which one would you like to use? "))]
    else:
        csvName = ensembleTableList[0]

    return csvName

def getEnsemble(csv_dict, csvName):
    """ Extracts the ensemble values and depth vector from the dictionary and
    returns them into two numpy arrays.

    Parameters
    ----------

    csv_dict : dict
        dictionary containing the availableTables

    csvName : str
        Name of the csv

    Returns
    -------

    depth : array
        Vector of depth

    ensembleValues : array
        The matrix of Ensemble values
    """
    ensemble_dict=csv_dict[csvName]
    ensembleValues=[]
    for val in ensemble_dict.keys():
        if 'depth' in val:
            depth = ensemble_dict[val]["values"]
        else:
            ensembleValues.append(ensemble_dict[val]["values"])
    ensembleValues= np.transpose(np.array(ensembleValues))

    return depth, ensembleValues

def mapAgeEnsembleToPaleoData(ensembleValues, depthEnsemble, depthPaleo):
    """ Map the depth for the ensemble age values to the paleo depth

    Parameters
    ----------

    ensembleValues : array
        A matrix of possible age models. Realizations
        should be stored in columns
    depthEnsemble : array
        A vector of depth. The vector should have the same
        length as the number of rows in the ensembleValues
    depthPaleo : array
        A vector corresponding to the depth at which there
        are paleodata information

    Returns
    -------

    ensembleValuesToPaleo : array
        A matrix of age ensemble on the PaleoData scale

    """
    if len(depthEnsemble)!=np.shape(ensembleValues)[0]:
        raise ValueError("Depth and age need to have the same length")

    #Make sure that numpy arrays were given
    ensembleValues=np.array(ensembleValues)
    depthEnsemble=np.array(depthEnsemble)
    depthPaleo = np.array(depthPaleo)

    #Interpolate
    ensembleValuesToPaleo = np.zeros((len(depthPaleo),np.shape(ensembleValues)[1])) #placeholder
    for i in np.arange(0,np.shape(ensembleValues)[1]):
        ensembleValuesToPaleo[:,i]=np.interp(depthPaleo,depthEnsemble,ensembleValues[:,i])

    return ensembleValuesToPaleo

def gen_dict_extract(key, var):
    '''Recursively searches for all the values in nested dictionaries corresponding
    to a particular key

    Parameters
    ----------

    key : str
        The key to search for
    var : dict
        The dictionary to search

    '''
    if hasattr(var,'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result
