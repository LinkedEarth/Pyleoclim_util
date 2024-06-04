import os

from setuptools import setup, find_packages

version = '1.0.0'

# Read the readme file contents into variable
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='pyleoclim',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['data/*.csv','data/metadata.yml']},
    package_dir={"": "."},
    zip_safe=False,
    version=version,
    license='GPL-3.0 License',
    description='A Python package for paleoclimate data analysis',
    long_description=read("README.md"),
    long_description_content_type = 'text/markdown',
    author='Deborah Khider, Julien Emile-Geay, Feng Zhu, Jordan Landers, Alexander James, Jun Hu, Myron Kwan, Pratheek Athreya, Daniel Garijo',
    author_email='linkedearth@gmail.com',
    url='https://github.com/LinkedEarth/Pyleoclim_util/pyleoclim',
    download_url='https://github.com/LinkedEarth/Pyleoclim_util/tarball/'+version,
    keywords=['Paleoclimate, Data Analysis, LiPD'],
    classifiers=[],
    install_requires=[
        "pandas==2.1.4",
        "kneed>=0.7.0",
        "statsmodels>=0.13.2",
        "seaborn>=0.13.0",
        "scikit-learn>=0.24.2",
        "pathos>=0.2.8",
        "tqdm>=4.61.2",
        "tftb>=0.1.3",
        "wget>=3.2",
        "numba>=0.56",
        "nitime>=0.9",
        "tabulate>=0.8.9",
        "Unidecode>=1.1.1",
        "cartopy>=0.22.0",
        "pyyaml",
        "beautifulsoup4",
        "scipy",
        "requests",
    ],
    python_requires=">=3.9",
)
