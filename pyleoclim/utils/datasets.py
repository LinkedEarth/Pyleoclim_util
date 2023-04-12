


from pathlib import Path 
import pyleoclim as pyleo
import yaml
import pandas as pd
from ..utils import jsonutils

DATA_DIR = Path(__file__).parents[1].joinpath("data").resolve()
METADATA_PATH = DATA_DIR.joinpath('metadata.yml')


def load_datasets_metadata(path=METADATA_PATH):
    """Load datasets metadata yaml file
    
    Parameters
    ----------
    path: str or pathlib.Path
        (Optional) path to dataset metadata yaml
        
    Returns
    -------
    Dictionary of sample dataset metadata

    Examples
    -------
    >>> from pyleoclim.utils.datasets import load_datasets_metadata
    Load the metadata yaml file (with the default path)
    >>> metadata = load_datasets_metadata()
    Alternately, if you have a new metadata file, you can load it directly using
    this method as well.
    >>> metadata = load_datasets_metadata(path='path/to/metadata.yml')

    """
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def available_dataset_names():
    """Helper function to easily see what datasets are available to load

    Returns
    -------
    List of datasets available via the `load_dataset` method. 
    
    Examples
    --------
    >>> from pyleoclim.utils.datasets import available_dataset_names
    >>> available_dataset_names()

    """
    meta = load_datasets_metadata()
    return list(meta.keys())


def get_metadata(name):
    """Load the metadata for a given dataset. 
    Note: Available datasets can be seen via `available_dataset_names`

    Parameters
    ----------
    name: str
        name of the dataset
    
    Returns
    -------
    Dictionary of metadata for this dataset

    Examples
    --------
    >>> from pyleoclim.utils.datasets import get_metadata
    >>> meta = get_metadata('LR04')
    """
    all_metadata = load_datasets_metadata()
    metadata = all_metadata.get(name, None)
    if metadata is None:
        avail_names = available_dataset_names()
        raise RuntimeError(f'Metadata not found for dataset {name}. Available dataset names are: {avail_names}')
    return metadata

def load_dataset(name):
    """Load example dataset given the nickname
    Note: Available datasets can be seen via `available_dataset_names`
    
    Parameters
    ----------
    name: str
        name of the dataset
    
    Returns
    -------
    pyleoclim_util.Series of the dataset

    Examples
    --------
    >>> from pyleoclim.utils.datasets import load_dataset
    >>> pyleo_series = load_dataset('LR04')
    """
    # load the metadata for this dataset
    metadata = get_metadata(name)
    # construct the full path to the file in the data directory
    path = DATA_DIR.joinpath(f"{metadata['filename']}.{metadata['file_extension']}")

    # if this is a csv
    if metadata['file_extension'] == 'csv':
        
        time_column =  metadata['time_column']
        value_column = metadata['value_column']
        pandas_kwargs = metadata.get('pandas_kwargs', {})
        pyleo_kwargs = metadata.get('pyleo_kwargs', {})
        
        # load into pandas
        df = pd.read_csv(path, **pandas_kwargs)

        # use iloc if we're given an int
        if isinstance(time_column, int):
            time=df.iloc[:, time_column]
        # use column name otherwise
        else:
            # if its a column
            if time_column in df.columns:
                time = df[time_column]
            else:
                time = df.Index

        if isinstance(value_column, int):
            value=df.iloc[:, value_column]
        else:
            value = df[value_column]

        # convert to pyleo.Series
        ts=pyleo.Series(
            time=time, 
            value=value,
            **pyleo_kwargs, 
            verbose=False
        )
    # if this is a json
    elif metadata['file_extension'] == 'json':
        ts=jsonutils.json_to_PyleoObj(str(path), 'Series')
        
    else:
        raise RuntimeError(f"Unable to load dataset with file extension {metadata['file_extension']}.")

    return ts
