 # -*- coding: utf-8 -*-
"""
Utilities to manipulate LiPD files and automate data transformation whenever possible. 
These functions are used throughout Pyleoclim but are not meant for direct interaction by users.
Also handles integration with the LinkedEarth wiki and the LinkedEarth Ontology.
"""

import numpy as np
import os
import json
import requests
import wget
from bs4 import BeautifulSoup
import string


class CaseInsensitiveDict(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key.lower().replace(" ", ""), value)

    def __getitem__(self, key):
        return super().__getitem__(key.lower().replace(" ", ""))

PLOT_DEFAULT = {'GroundIce': ['#86CDFA', 'h'],
                     'Borehole': ['#00008b', 'h'],
                     'Coral': ['#FF8B00', 'o'],
                     'Documents': ['#f8d568', 'p'],
                     'GlacierIce': ['#86CDFA', 'd'],
                     'Hybrid': ['#808000', '*'],
                     'LakeSediment': ['#8A4513', '^'],
                     'MarineSediment': ['#8A4513', 's'],
                     'Sclerosponge': ['r', 'o'],
                     'Speleothem': ['#FF1492', 'd'],
                     'Wood': ['#32CC32', '^'],
                     'MolluskShell': ['#FFD600', 'h'],
                     'Peat': ['#2F4F4F', '*'],
                     'Midden': ['#824E2B', 'o'],
                     'FluvialSediment': ['#4169E0','d'],
                     'TerrestrialSediment': ['#8A4513','o'],
                     'Shoreline': ['#add8e6','o'],
                     'Instrumental' : ['#8f21d8', '*'],
                     'Model' : ['#b4a7d6', "d"],
                     'Other': ['k', 'o']
                    }


"""
The followng functions handle web scrapping to grab information regarding the controlled vocabulary

"""

def get_archive_type():
    '''
    Scrape the LiPDverse website to obtain the list of possible archives and associated synonyms

    Returns
    -------
    res : Dictionary
        Keys correspond to the preferred terms and values represent known synonyms

    '''
    url = "https://lipdverse.org/vocabulary/archivetype/"
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the content of the request with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get the names of the archiveTypes   
        
        h3_tags = soup.find_all('h3')
        archiveName = []
        for item in h3_tags:
            archiveName.append(item.get_text())
            
        
            
        # Get the known synonyms
        h4_tags = soup.find_all('h4', string="Known synonyms")
        
        synonyms=[]
        
        
        for h4_tag in h4_tags:
            next_element = h4_tag.find_next_sibling()

            found_p = False
            while next_element and next_element.name != 'div':
                if next_element.name == 'p':
                    synonyms_text = next_element.get_text()
                    words = [word.strip() for word in synonyms_text.split(',')]
                    synonyms.append(words)
                    found_p = True
                    break

                next_element = next_element.find_next_sibling()
            
            # If a <p> tag was not found, insert an empty string
            if not found_p:
                synonyms.append([])
        
        #create a dictionary for the results
        res= {}
        for idx,item in enumerate(archiveName):
            res[item]=synonyms[idx]

    else:
        print("failed to retrieve the webpage; returning static list, which may be out of date")
        
        res = ["Borehole",
                       "Coral",
                       "FluvialSediment",
                       "GlacierIce",
                       "GroundIce",
                       "LakeSediment",
                       "MarineSediment",
                       "Midden",
                       "MolluskShell",
                       "Peat",
                       "Scelorosponge",
                       "Shoreline",
                       "Spleleothem",
                       "TerrestrialSediment",
                       "Wood"]
    
    return res


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
    
    if archiveType != None:
    
        if archiveType.lower().replace(" ", "") == "icecore":
            archiveType = 'GlacierIce'
        elif archiveType.lower().replace(" ", "") == "ice-other":
            archiveType = 'GlacierIce'
        elif archiveType.lower().replace(" ", "") == 'tree':
            archiveType = 'Wood'
        elif archiveType.lower().replace(" ","") not in [key.lower() for key in PLOT_DEFAULT.keys()]:
            archiveType='Other'

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

def mapAgeEnsembleToPaleoData(ensembleValues, depthEnsemble, depthPaleo,extrapolate=True):
    """ Map the depth for the ensemble age values to the paleo depth

    Parameters
    ----------
    ensembleValues : array
        A matrix of possible age models. Realizations should be stored in columns

    depthEnsemble : array
        A vector of depth. The vector should have the same length as the number of rows in the ensembleValues

    depthPaleo : array
        A vector corresponding to the depth at which there are paleodata information

    extrapolate : bool
        Whether to extrapolate the ensemble values to the paleo depth. Default is True

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

    if extrapolate is True:
        for i in np.arange(0,np.shape(ensembleValues)[1]):
            ensembleValuesToPaleo[:,i]=np.interp(depthPaleo,depthEnsemble,ensembleValues[:,i])
    elif extrapolate is False:
        for i in np.arange(0,np.shape(ensembleValues)[1]):
            ensembleValuesToPaleo[:,i]=np.interp(depthPaleo,depthEnsemble,ensembleValues[:,i],left=np.nan,right=np.nan)

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
