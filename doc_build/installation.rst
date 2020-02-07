Installation
============

We recommend the use of Anaconda or Miniconda, with Pyleoclim setup in
its own environment.

Installing Anaconda or Miniconda
"""""""""""""""""""""""""""""""""

To install Anaconda or Miniconda on your platform, follow the instructions from `this page <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

Creating a new Anaconda environment
"""""""""""""""""""""""""""""""""""

To create a new environment using Python 3.6 via command line:

  conda create -n pyleoenv python=3.6


To view a list of available environment:

  conda env list

To activate your new environment:

  conda activate pyleoenv

To view the list of packages in your environment:

  conda list

To remove the environment:

  conda remove --name pyleoenv --all

More information about managing conda environments can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#>`_.

Installing Pyleoclim
""""""""""""""""""""
Make sure that the pyleoenv environment is activated.

**First** install numpy and Cartopy:

  conda install numpy
  conda install -c conda-forge cartopy

Make sure that the package proj4 is version 5.2+

  conda list

Install Pyleoclim

  pip install pyleoclim

Building from source
""""""""""""""""""""

Note that the pip command line above will trigger the installation of (most of) the dependencies, as well as the local compilation of the Fortran code for WWZ with the GNU Fortran compiler gfortran. If you have the Intel's Fortran compiler ifort installed, then further accerlation for WWZ could be achieved by compiling the Fortran code with ifort, and below are the steps:

- download the source code, either via git clone or just download the .zip file
- modify setup.py by commenting out the line of extra_f90_compile_args for gfortran, and use the line below for ifort
- run python setup.py build_ext --fcompiler=intelem && python setup.py install

Installing R
""""""""""""

Some functionalities require an installation of `R <https://www.r-project.org?>`_.

To install R, download a mirror `here <https://cran.r-project.org/mirrors.html>`_`.
Note that Rstudio is not needed when calling R from Python.

Pyleoclim requires the `Bchron package <https://cran.r-project.org/web/packages/Bchron/index.html>`_`. Pyleoclim will check for an installation of Bchron. It it doesn't exist, it will be installed.
