.. _installation:

.. note::

   Pyleoclim requires the use of Python 3.8 or 3.9.

Installing Pyleoclim
====================

We recommend the use of Anaconda or Miniconda, with Pyleoclim setup in
its own `conda` environment. Some default packages shipping with the full Anaconda distribution are known to cause conflicts with the required Pyleoclim packages, so we recommend Miniconda, especially for beginners.

Installing Anaconda or Miniconda
"""""""""""""""""""""""""""""""""

To install Anaconda or Miniconda on your platform, follow the instructions from `this page <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

Click :ref:`here <anaconda_installation>` for a quick tutorial on MacOs and Linux systems.

Creating a new conda environment
"""""""""""""""""""""""""""""""""""

To create a new environment using Python 3.9 via command line:

.. code-block:: bash

  conda create -n pyleo python=3.9

To view a list of available environment:

.. code-block:: bash

  conda env list

To activate your new environment:

.. code-block:: bash

  conda activate pyleo

To view the list of packages in your environment:

.. code-block:: bash

  conda list

To remove the environment:

.. code-block:: bash

  conda remove --name pyleo --all

More information about managing conda environments can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#>`_.

Installing Pyleoclim
""""""""""""""""""""
Make sure that the pyleo environment is activated.

First install Cartopy:

.. code-block:: bash

  conda install -c conda-forge cartopy

Then install Pyleoclim through Pypi, which contains the most stable version of Pyleoclim:

.. code-block:: bash

  pip install pyleoclim

To install the development version, which contains the most up-to-date features:

.. code-block:: bash

  pip install git+https://github.com/LinkedEarth/Pyleoclim_util.git@Development

If you would like to use Jupyter Notebooks or Spyder for code development, install these packages in your environment:

.. code-block:: bash

  conda install spyder
  conda install jupyter

Optional libraries
""""""""""""""""""

To run the :ref:`tutorial notebooks <tutorials>`, we recommend installing the `xarray <https://docs.xarray.dev/en/stable/getting-started-guide/installing.html>`_ package suite.

.. code-block:: bash

  conda install -c conda-forge xarray dask netCDF4 bottleneck

You will also need `climlab <https://climlab.readthedocs.io/en/latest/>`_:

.. code-block:: bash

  conda install climlab

Building from source for the f2py feature of WWZ
""""""""""""""""""""""""""""""""""""""""""""""""

The default version of WWZ that comes with the installation steps mentioned above is relying on `Numba <http://numba.pydata.org/>`_.
It is fast enough for lightweight spectral & wavelet analysis tasks, in which case we recommend using the default installation.

However, it could be slow for heavy use (e.g. performing it for hundreds of times on timeseries with length longer than 1000 points), in which case we recommend activating the f2py feature to achieve an acceleration of around 50%.

To do that, a Fortran compiler (e.g. :code:`gfortran` or :code:`ifort`) is required on your local machine, and the related Fortran source code should be compiled locally following the steps below:

- download the source code, either via git clone or just download the .zip file from the `Github repo <https://github.com/LinkedEarth/Pyleoclim_util>`_
- go to the directory :code:`Pyleoclim_util/pyleoclim/f2py`, and then type :code:`make` to compile the .f90 source code with :code:`gfortran`
- one may also edit the :code:`Makefile` to use :code:`ifort` as the compiler to achieve further acceleration; just comment out the line for :code:`gfortran` and use the line for :code:`ifort` below
- a :code:`.so` file will be generated if the compilation is successful
- copy the :code:`.so` file into the directory :code:`Pyleoclim_util/pyleoclim/utils` where Pyleoclim is installed on your machine. To find out the location, one may import the package in Python and "print" it:

.. code-block:: python

  import pyleoclim as pyleo
  print(pyleo)

Again, unless you are planning to make heavy use of the WWZ functionality, we recommend using the default installation.

Docker Container
""""""""""""""""
Docker containers with various versions of Pyleoclim are available at: `https://quay.io/repository/2i2c/paleohack-2021?tab=tags <https://quay.io/repository/2i2c/paleohack-2021?tab=tags>`_.

To pull an image:

.. code-block:: bash

  docker pull quay.io/2i2c/paleohack-2021:latest
