from distutils.core import setup
import os
import sys
import io

os.system('pandoc README.md -f markdown -t rst -s -o README.txt')
readme_file = io.open('README.txt', encoding='utf-8')
long_description = readme_file.open()
os.remove('README.txt') #remove the text file

setup(
    name = 'Pyleoclim',
    packages = ['Pyleoclim', 'pkg_resources'],
    version = 0.1,
    license = 'GNU Public',
    description = 'A Python package for paleoclimate data analysis',
    long_description = long_description,
    author = 'Deborah Khider',
    author_email = 'dkhider@gmail.com',
    url = 'https://github.com/LinkedEarth/Pyleoclim_util/Pyleoclim',
    download_url = 'https://github.com/LinkedEarth/Pyleoclim_util/Pyleoclim/tarball/0.1',
    keywords = ['Paleoclimate, Data Analysis'],
    classifiers = [],
)
