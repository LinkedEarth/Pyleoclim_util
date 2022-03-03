import os
import sys
import io

from setuptools import setup, find_packages

version = '0.7.2'

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
        "LiPD==0.2.8.8",
        "pandas>=1.3.0",
        "numpy>=1.21.1",
        "matplotlib>=3.4.2",
        "scipy>=1.7.0",
        "statsmodels>=0.11.1",
        "seaborn>=0.11",
        "scikit-learn>=0.24.2",
        "pathos>=0.2.6",
        "tqdm>=4.47.0",
        "tftb>=0.1.1",
        "pyhht>=0.1.0",
        "wget>=3.2",
        "numba>=0.50.1",
        "nitime>=0.8.1",
        "tabulate>=0.8.7",
        "Unidecode>=1.1.1",
    ],
    python_requires=">=3.8.0"
)
