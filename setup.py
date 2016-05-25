from setuptools import setup, find_packages
from os import path
import shutil
import os
import io
import sys
from distutils.command.install import install


class PostInstall(install):
    """ Custom install script that runs post-install."""
    def run(self):
        # Make the notebooks folder in the user directory
        src_nb = os.path.join(here, 'lipd', 'files')
        src_ex = os.path.join(src_nb, 'examples')
        dst_nb = os.path.join(os.path.expanduser('~'), 'LiPD_Notebooks')
        dst_ex = os.path.join(dst_nb, 'examples')

        # Make folders if needed
        if not os.path.isdir(dst_nb):
            os.mkdir(dst_nb)
        if not os.path.isdir(dst_ex):
            os.mkdir(dst_ex)

        # Copy example files (don't overwrite directory)
        f = [x for x in os.listdir(src_ex) if not (x.startswith('.'))]
        for file in f:
            print(file)
            if file == 'quickstart_functions.py':
                shutil.copy(os.path.join(src_ex, file), dst_ex)
            elif os.path.isfile(os.path.join(src_ex, file)):
                shutil.copy(os.path.join(src_ex, file), dst_nb)

        # Copy / Overwrite Quickstart notebook
        shutil.copy(os.path.join(src_nb, 'Welcome LiPD - Quickstart.ipynb'), dst_nb)

        # Open the install folder so the user can see the documentation and instructions
        # os.system('open .')
        install.run(self)


here = path.abspath(path.dirname(__file__))
version = '0.1.2.9'

# Read the readme file contents into variable
if sys.argv[-1] == 'publish' or sys.argv[-1] == 'publishtest':
    os.system('pandoc README.md -f markdown -t rst -s -o README.txt')

readme_file = io.open('README.txt', encoding='utf-8')
# Fallback long_description in case errors with readme file.
long_description = "Welcome to LiPD. Please reference the README file in the package for information"
with readme_file:
    long_description = readme_file.read()

# Publish the package to the live server
if sys.argv[-1] == 'publish':
    # Register the tarball, upload it, and trash the temp readme rst file
    os.system('python3 setup.py register')
    os.system('python3 setup.py sdist upload')
    os.remove('README.txt')
    sys.exit()

# Publish the package to the test server
elif sys.argv[-1] == 'publishtest':
    # Create dist tarball, register it to test site, upload tarball, and remove temp readme file
    os.system('python3 setup.py sdist')
    os.system('python3 setup.py register -r https://testpypi.python.org/pypi')
    os.system('twine upload -r test dist/LiPD-' + version + '.tar.gz')
    # Trash the temp rst readme file
    os.remove('README.txt')
    sys.exit()


# Do all the setup work
setup(
    name='LiPD',
    version=version,
    author='C. Heiser',
    author_email='heiser@nau.edu',
    packages=find_packages(exclude=['build', '_docs', 'templates']),
    # packages = ["lipd", "doi", "noaa", "excel"],
    entry_points={
        "console_scripts": [
            'lipd= lipd.__main__:main'
        ]
    },
    cmdclass={
        'install': PostInstall,
    },
    url='https://github.com/nickmckay/LiPD-utilities',
    license='GNU Public',
    description='LiPD utilities to process, convert, and analyze data.',
    long_description=long_description,
    keywords="paleo R matlab python paleoclimatology linkedearth",
    install_requires=[
        "bagit>=1.5.4",
        "beautifulsoup4>=4.4.1",
        "bokeh>=0.11.1",
        "demjson>=2.2.4",
        "matplotlib>=1.4.2",
        "xlrd>=0.9.3",
        "Pillow>=3.1.1",
        "jupyter>=1.0.0",
        "pandas>=0.18.0",
        "requests>=2.9.1",
        "google-api-python-client>=1.4.2",
        "virtualenv>=15.0.1"
    ],
    include_package_data=True,
    package_data={'noaa': ['*.txt'],
                  'helpers': ['*.json']
                  },
)
