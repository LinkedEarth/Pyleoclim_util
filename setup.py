from setuptools import setup, find_packages
import os
import sys
import io
import path

version = '0.1.5'

# Read the readme file contents into variable
os.system('pandoc README.md -f markdown -t rst -s -o README.txt')
readme_file = io.open('README.txt', encoding='utf-8')

# Fallback long_description in case errors with readme file.
long_description = "Welcome to Pyleoclim. Please reference the README file in the package for information"
with readme_file:
    long_description = readme_file.read()

# Remove the text file once the description is added
os.remove('README.txt')

setup(
    name = 'Pyleoclim',
    packages = find_packages(),
    version = version,
    license = 'GNU Public',
    description = 'A Python package for paleoclimate data analysis',
    long_description=long_description,
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
    ]
)
