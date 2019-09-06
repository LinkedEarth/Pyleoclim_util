import os
import sys
import io

#  from setuptools import setup, find_packages
from setuptools import find_packages
from numpy.distutils.core import Extension
from numpy.distutils.core import setup

version = '0.4.10'

# Read the readme file contents into variable
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Publish the package to the live server
if sys.argv[-1] == 'publish':
    # Register the tarball, upload it, and trash the temp readme rst file
    #os.system('python3 setup.py register -r pypi')
    os.system('python3 setup.py sdist')
    os.system('twine upload dist/pyleoclim-'+version+'tar.gz')
    sys.exit()

# Publish the package to the test server
elif sys.argv[-1] == 'publishtest':
    # Create dist tarball, register it to test site, upload tarball, and remove temp readme file
    #os.system('python3 setup.py register -r pypitest')
    os.system('python3 setup.py sdist')
    os.system('twine upload --repository-url https://test.pypi.org/legacy/ dist/pyleoclim-'+version+'tar.gz')

    sys.exit()

f2py_wwz = Extension(
    name='pyleoclim.f2py_wwz',
    extra_compile_args=['-O3'],
    extra_f90_compile_args=['-fopenmp', '-O3'],  # compiling with gfortran
    #  extra_f90_compile_args=['-qopenmp', '-O3', '-mkl=parallel'],  # compiling with ifort
    sources=['pyleoclim/src/f2py_wwz.f90']
)

setup(
    name='pyleoclim',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    version=version,
    license='GNU Public',
    description='A Python package for paleoclimate data analysis',
    long_description=read("README.md"),
    long_description_content_type = 'text/markdown',
    author='Deborah Khider',
    author_email='khider@usc.edu',
    url='https://github.com/LinkedEarth/Pyleoclim_util/pyleoclim',
    download_url='https://github.com/LinkedEarth/Pyleoclim_util/tarball/'+version,
    keywords=['Paleoclimate, Data Analysis'],
    classifiers=[],
    ext_modules=[f2py_wwz],
    install_requires=[
        "LiPD>=0.2.7",
        "pandas>=0.25.0",
        "numpy>=1.16.4",
        "matplotlib>=3.1.0",
        "scipy>=1.3.1",
        "statsmodels>=0.8.0",
        "seaborn>=0.9.0",
        "scikit-learn>=0.21.3",
        "pathos>=0.2.4",
        "tqdm>=4.33.0",
        "rpy2>=3.0.5"],
    python_requires=">=3.6.0"
)
