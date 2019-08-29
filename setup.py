import os
import sys
import io

from setuptools import setup, find_packages

version = '0.4.9'

# Read the readme file contents into variable
this_directory = os.path.abspath(path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Publish the package to the live server
if sys.argv[-1] == 'publish':
    # Register the tarball, upload it, and trash the temp readme rst file
    os.system('python setup.py register -r pypi')
    os.system('python setup.py sdist upload -r pypi')
    os.remove('README.txt')
    sys.exit()

# Publish the package to the test server
elif sys.argv[-1] == 'publishtest':
    # Create dist tarball, register it to test site, upload tarball, and remove temp readme file
    os.system('python setup.py register upload -r pypitest')
    os.system('python setup.py sdist upload -r pypitest')
    # Trash the temp rst readme file
    os.remove('README.txt')
    sys.exit()

setup(
    name='pyleoclim',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    version=version,
    license='GNU Public',
    description='A Python package for paleoclimate data analysis',
    long_description=long_description,
    long_description_content_type = 'text/markdown'
    author='Deborah Khider',
    author_email='dkhider@gmail.com',
    url='https://github.com/LinkedEarth/Pyleoclim_util/pyleoclim',
    download_url='https://github.com/LinkedEarth/Pyleoclim_util/tarball/0.4.8',
    keywords=['Paleoclimate, Data Analysis'],
    classifiers=[],
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
        "rpy2>=3.0.5"]
)
