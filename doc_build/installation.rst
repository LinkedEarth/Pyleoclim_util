Installation
============

We recommend the use of Anaconda or Miniconda, with Pyleoclim setup in
its own environment.

Installing Anaconda or Miniconda
"""""""""""""""""""""""""""""""""

To install Anaconda or Miniconda on your platform, follow the instructions from `this page <https://github.com/LinkedEarth/Pyleoclim_util/blob/Development/help/python-env-setup.md>`_ or `this page <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

Creating a new Anaconda environment
"""""""""""""""""""""""""""""""""""

To create a new environment using Python 3.7 via command line:

.. code-block:: bash

  conda create -n pyleoenv python=3.7

To view a list of available environment:

.. code-block:: bash

  conda env list

To activate your new environment:

.. code-block:: bash

  conda activate pyleoenv

To view the list of packages in your environment:

.. code-block:: bash

  conda list

To remove the environment:

.. code-block:: bash

  conda remove --name pyleoenv --all

More information about managing conda environments can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#>`_.

Installing Pyleoclim
""""""""""""""""""""
Make sure that the pyleoenv environment is activated.

**First** install numpy and Cartopy:

.. code-block:: bash

  conda install numpy
  conda install -c conda-forge cartopy

Make sure that the package proj4 is version 5.2+

.. code-block:: bash

  conda list

Install Pyleoclim

.. code-block:: bash

  pip install pyleoclim

Building from source for the f2py feature of WWZ
""""""""""""""""""""""""""""""""""""""""""""""""

The default version of WWZ is relying on `Numba <http://numba.pydata.org/>`_.
To achieve accelartion of the alogrithm, one may build the f2py vesion from the source code:

- download the source code, either via git clone or just download the .zip file from the `Github repo <https://github.com/LinkedEarth/Pyleoclim_util>`_
- go to the directory :code:`Pyleoclim_util/pyleoclim/f2py`, and then type :code:`make` to compile the .f90 source code with :code:`gfortran`
- one may also edit the :code:`Makefile` to use :code:`ifort` as the compiler to achieve further acceleration; just comment out the line for :code:`gfortran` and use the line for :code:`ifort` below


Installing R
""""""""""""

Some functionalities require an installation of `R <https://www.r-project.org?>`_.

To install R, download a mirror `here <https://cran.r-project.org/mirrors.html>`_.
Note that Rstudio is not needed when calling R from Python.

Pyleoclim requires the `Bchron package <https://cran.r-project.org/web/packages/Bchron/index.html>`_. Pyleoclim will check for an installation of Bchron. It it doesn't exist, it will be installed.
