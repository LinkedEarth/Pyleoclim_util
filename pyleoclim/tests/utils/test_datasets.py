# from pyleoclim.utils import load_dataset
import pyleoclim as pyleo
import pytest
from pyleoclim import utils
from pyleoclim.utils import datasets 

TEST_DATASETS = [
    'soi',
    'nino3',
    'nino_json'
]


def test_load_datasets_metadata():
    """test loading all metadata for all datasets"""
    meta = datasets.load_datasets_metadata()
    assert isinstance(meta, dict)


def test_available_dataset_names():
    """test getting available dataset names"""
    names = datasets.available_dataset_names()
    assert len(names) > 0


@pytest.mark.parametrize('name', TEST_DATASETS)
def test_get_metadata(name):
    meta = datasets.get_metadata(name)
    assert isinstance(meta, dict)


def test_get_metadata_invalid_name():
    """ensure error is raised with an invalid dataset name"""
    with pytest.raises(RuntimeError, match='Metadata not found'):
        datasets.get_metadata('invalid_name')


@pytest.mark.parametrize('name', TEST_DATASETS)
def test_load_dataset(name):
    ts = datasets.load_dataset(name)
    assert isinstance(ts, pyleo.Series)
