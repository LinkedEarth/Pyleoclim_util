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

Installing R
""""""""""""

Some functionalities require an installation of `R <https://www.r-project.org?>`_.

To install R, download a mirror `here <https://cran.r-project.org/mirrors.html>`_`.
Note that Rstudio is not needed when calling R from Python.

Pyleoclim requires the `Bchron package <https://cran.r-project.org/web/packages/Bchron/index.html>`_`. Pyleoclim will check for an installation of Bchron. It it doesn't exist, it will be installed.
