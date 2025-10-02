


from pathlib import Path 
import pyleoclim as pyleo
import yaml
import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import SKOS, RDF
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

    .. jupyter-execute::

        from pyleoclim.utils.datasets import get_metadata
        meta = get_metadata('LR04')

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
    Series object

    Examples
    --------

    .. jupyter-execute::
        
        LR04 = pyleo.utils.load_dataset('LR04')
        LR04.view()

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

        # Check if the index is a DatetimeIndex - if so, use from_pandas()
        if isinstance(df.index, pd.DatetimeIndex):
            # Filter out rows with NaT (Not a Time) values and missing data
            valid_mask = df.index.notna()
            if isinstance(value_column, int):
                value_series = df.iloc[:, value_column]
            else:
                value_series = df[value_column]
            
            # Also filter out rows where values are NaN
            valid_mask = valid_mask & value_series.notna()
            
            # Apply the mask to get clean data
            clean_index = df.index[valid_mask]
            clean_values = value_series[valid_mask]
            
            # Create a pandas Series with the clean DatetimeIndex
            pandas_series = pd.Series(clean_values.values, index=clean_index)
            
            # Use from_pandas() method which handles DatetimeIndex properly
            if 'lat' in pyleo_kwargs.keys() and 'lon' in pyleo_kwargs.keys():
                if pyleo_kwargs['lat'] is not None and pyleo_kwargs['lon'] is not None:
                    ts = pyleo.GeoSeries.from_pandas(
                        pandas_series,
                        metadata=pyleo_kwargs,
                        verbose=False
                    )
                else:
                    ts = pyleo.Series.from_pandas(
                        pandas_series,
                        metadata=pyleo_kwargs,
                        verbose=False
                    )
            else:
                ts = pyleo.Series.from_pandas(
                    pandas_series,
                    metadata=pyleo_kwargs,
                    verbose=False
                )
        else:
            # Original logic for non-DatetimeIndex data
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
            
            if 'lat' in pyleo_kwargs.keys() and 'lon' in pyleo_kwargs.keys():
                if pyleo_kwargs['lat'] is not None and pyleo_kwargs['lon'] is not None:
                    ts=pyleo.GeoSeries(
                        time=time, 
                        value=value,
                        **pyleo_kwargs, 
                        verbose=False
                    )
                else:
                   ts=pyleo.Series(
                       time=time, 
                       value=value,
                       **pyleo_kwargs, 
                       verbose=False
                   ) 

            else:
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


TIME = Namespace("http://www.w3.org/2006/time#")
SCHEMA = Namespace("https://schema.org/")
GTS = Namespace("http://resource.geosciml.org/ontology/timescale/gts#")

def _first(it):
    for x in it:
        return x
    return None

def _localname(term):
    s = str(term)
    if "#" in s:
        return s.rsplit("#", 1)[1]
    return s.rstrip("/").rsplit("/", 1)[-1]

def _find_ns_for_local(graph: Graph, local: str):
    """
    Find a predicate in the graph whose localname == `local` and
    return its namespace base as an rdflib Namespace. Fallback to None.
    """
    for s, p, o in graph.triples((None, None, None)):
        if isinstance(p, URIRef) and _localname(p) == local:
            base = str(p)[: -len(local)]
            return Namespace(base)
    return None

def load_ics_chart_to_df(ttl_path_or_url="https://raw.githubusercontent.com/i-c-stratigraphy/chart/refs/heads/main/chart.ttl", time_units='Ma') -> pd.DataFrame:
    g = Graph()
    g.parse(ttl_path_or_url, format="turtle")

    # Resolve namespaces robustly
    ischart_ns = _find_ns_for_local(g, "inMYA")  # e.g., https://w3id.org/isc/isc2020#
    if ischart_ns is None:
        # Hard fallback (current ICS charts use this)
        ischart_ns = Namespace("https://w3id.org/isc/isc2020#")

    # Rank mapping to your requested Rank labels
    rank_map = {
        "Eon": "Eon", "Eonothem": "Eon",
        "Era": "Era", "Erathem": "Era",
        "Period": "Period", "System": "Period",
        "Epoch": "Epoch", "Series": "Epoch",
        "Age": "Stage", "Stage": "Stage",
        "Subepoch": "Subepoch", "Subseries": "Subepoch",
        "Substage": "Substage", "Subperiod": "Subperiod",
    }

    rows = []

    # An interval concept typically has skos:prefLabel and gts:rank
    for subj in set(g.subjects(SKOS.prefLabel, None)):
        rank_uri = _first(g.objects(subj, GTS.rank))
        if rank_uri is None:
            continue  # skip non-interval resources

        # Type
        rank_local = _localname(rank_uri)
        Rank = rank_map.get(rank_local, rank_local)

        # Name
        name_lit = _first(g.objects(subj, SKOS.prefLabel))
        Name = str(name_lit) if name_lit is not None else ""

        # Abbrev: skos:notation (typed literal is fine)
        notation = _first(g.objects(subj, SKOS.notation))
        Abbrev = str(notation) if notation is not None else ""

        # Color (hex) if present
        col = _first(g.objects(subj, SCHEMA.color))
        Color = str(col) if col is not None else ""

        # Helper to read a boundary node (blank node or URI) → inMYA float
        def boundary_ma(node_pred):
            node = _first(g.objects(subj, node_pred))
            if node is None:
                return None
            # find a predicate with localname 'inMYA' under this node
            val = None
            for p, o in g.predicate_objects(node):
                if _localname(p) == "inMYA":
                    val = o
                    break
            try:
                return float(val) if val is not None else None
            except Exception:
                return None

        start_ma = boundary_ma(TIME.hasBeginning)
        end_ma   = boundary_ma(TIME.hasEnd)

        # UpperBoundary (younger/smaller Ma), LowerBoundary (older/larger Ma)
        UpperBoundary = LowerBoundary = None
        if start_ma is not None and end_ma is not None:
            UpperBoundary = min(start_ma, end_ma)
            LowerBoundary = max(start_ma, end_ma)
        elif start_ma is not None:
            UpperBoundary = start_ma
        elif end_ma is not None:
            UpperBoundary = end_ma

        # Keep intervals only
        if not Name or (UpperBoundary is None and LowerBoundary is None):
            continue

        rows.append({
            "Rank": Rank,
            "Name": Name,
            "Abbrev": Abbrev,
            "Color": Color,
            "UpperBoundary": UpperBoundary,
            "LowerBoundary": LowerBoundary
        })

    df = pd.DataFrame(rows).dropna(subset=["Name", "Rank"])
    # Optional: order types, youngest first within each type
    type_order = ["Eon", "Era", "Period", "Epoch", "Stage", "Subepoch", "Substage", "Subperiod"]
    df["Rank"] = pd.Categorical(df["Rank"], categories=type_order + sorted(set(df["Rank"]) - set(type_order)), ordered=True)
    df = df.sort_values(["Rank", "UpperBoundary"], na_position="last").reset_index(drop=True)

    if time_units not in ['Ma', 'Mya', 'My']:
        if time_units in ['a', 'ya', 'y', 'yr']:
            df['UpperBoundary'] = df['UpperBoundary'] * 1e6
            df['LowerBoundary'] = df['LowerBoundary'] * 1e6
        elif time_units in ['ka', 'kya', 'ky', 'kyr']:
            df['UpperBoundary'] = df['UpperBoundary'] * 1e3
            df['LowerBoundary'] = df['LowerBoundary'] * 1e3
        elif time_units in ['Ga', 'Gya', 'Gy', 'Gyr']:
            df['UpperBoundary'] = df['UpperBoundary'] * 1e-3
            df['LowerBoundary'] = df['LowerBoundary'] * 1e-3
        else:
            raise ValueError(f"Unsupported time_units '{time_units}'. Supported: 'Ma', 'ka', 'Ga'.")

    return df

def apply_custom_traits(df, specs, target_col="Abbrev"):
    """
    Apply custom overrides to a DataFrame by matching on traits.

    Parameters
    ----------
    df : pandas.DataFrame
        Must include the target column and all columns named in specs.
    specs : list of dict
        Each dict has at least the target_col key plus one or more
        trait columns used for matching.
        Example: {"Type":"Period", "Name":"Cambrian", "Abbrev":"Cam."}
    target_col : str, default "Abbrev"
        The column to update.

    Returns
    -------
    pandas.DataFrame
        Copy of df with updates applied.
    """
    df = df.copy()
    if not isinstance(specs, list):
        specs = [specs]
    for spec in specs:
        if target_col not in spec:
            raise ValueError(f"Spec {spec} missing '{target_col}' key")

        # Start with all True, then AND each condition
        mask = pd.Series(True, index=df.index)
        for col, val in spec.items():
            if col == target_col:
                continue
            mask &= df[col] == val

        df.loc[mask, target_col] = spec[target_col]
    return df
