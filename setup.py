import os
#import sys
#import io

from setuptools import setup, find_packages


version = '1.0.0b0'

# Read the readme file contents into variable
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='pyleoclim',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    version=version,
    license='GPL-3.0 License',
    description='A Python package for paleoclimate data analysis',
    long_description=read("README.md"),
    long_description_content_type = 'text/markdown',
    author='Deborah Khider, Feng Zhu, Julien Emile-Geay, Jun Hu, Myron Kwan, Pratheek Athreya, Alexander James, Daniel Garijo',
    author_email='linkedearth@gmail.com',
    url='https://github.com/LinkedEarth/Pyleoclim_util/pyleoclim',
    download_url='https://github.com/LinkedEarth/Pyleoclim_util/tarball/'+version,
    keywords=['Paleoclimate, Data Analysis, LiPD'],
    classifiers=[],
    install_requires=[
        "cartopy",
        "kneed>=0.7.0",
        "LiPD==0.2.8.8",
        "nitime>=0.9",
        "numba>=0.56",
        "numpy<=1.24.0",
        "matplotlib>=3.6.0",
        "pandas>=2.0.0",
        "pathos>=0.2.8",
        "pyhht>=0.1.0",
        "pyyaml",
        "requests",
        "scikit-learn>=0.24.2",
        "scipy>=1.9.1",
        "seaborn>=0.12.0",
        "shapely",
        "statsmodels>=0.13.2",
        "tabulate>=0.8.9",
        "tftb>=0.1.3",
        "tqdm>=4.61.2",
        "Unidecode>=1.1.1",
        "wget>=3.2",
    ],
    python_requires=">=3.9.0",
)
