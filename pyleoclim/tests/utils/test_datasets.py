import pyleoclim as pyleo
import pytest
from pyleoclim.utils import datasets 

# Datasets tested in this file
TEST_DATASETS = [
    'SOI',
    'NINO3',
    'HadCRUT5',
    'AIR',
    'LR04',
    'AACO2',
    'GISP2',
    'EDC-dD'
]

def test_load_datasets_metadata():
    """Test loading all metadata for all datasets"""
    meta = datasets.load_datasets_metadata()
    assert isinstance(meta, dict)

def test_load_datasets_metadata_with_path():
    """Test loading all metadata using specific path"""
    meta = datasets.load_datasets_metadata(path=datasets.METADATA_PATH)
    assert isinstance(meta, dict)


def test_available_dataset_names():
    """Test getting available dataset names"""
    names = datasets.available_dataset_names()
    assert len(names) > 0


@pytest.mark.parametrize('name', TEST_DATASETS)
def test_get_metadata(name):
    """Test getting metdata for the test datasets"""
    meta = datasets.get_metadata(name)
    assert isinstance(meta, dict)


def test_get_metadata_invalid_name():
    """Test getting metadata using an invalid name. Ensure error is raised."""
    with pytest.raises(RuntimeError, match='Metadata not found'):
        datasets.get_metadata('invalid_name')


@pytest.mark.parametrize('name', TEST_DATASETS)
def test_load_dataset(name):
    """Test loading datasets"""
    ts = datasets.load_dataset(name)
    assert isinstance(ts, pyleo.Series)
