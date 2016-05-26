from setuptools import setup, find_packages
import os
import sys
import io

setup(
    name = 'Pyleoclim',
    packages = find_packages(),
    version = '0.1.3',
    license = 'GNU Public',
    description = 'A Python package for paleoclimate data analysis',
    author = 'Deborah Khider',
    author_email = 'dkhider@gmail.com',
    url = 'https://github.com/LinkedEarth/Pyleoclim_util/Pyleoclim',
    download_url = 'https://github.com/LinkedEarth/Pyleoclim_util/Pyleoclim/tarball/0.1',
    keywords = ['Paleoclimate, Data Analysis'],
    classifiers = [],
    install_requires = [
    "LiPD>=0.1.2.7",
    "pandas>=0.18.1",
    "numpy>=1.11.0",
    "matplotlib>=1.5.1",
    "Cartopy>=0.13.1"
    ]
)
