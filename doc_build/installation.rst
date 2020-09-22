.. _installation:

.. toctree::
   :hidden:
   anaconda_installation.rst

.. note::
  Pyleoclim requires the use of Python 3.8.

Installation
============

We recommend the use of Anaconda or Miniconda, with Pyleoclim setup in
its own `conda` environment. Some default packages shipping with the full Anaconda distribution are known to cause conflicts with the required Pyleoclim packages, so we recommend Miniconda, especially for beginners.

Installing Anaconda or Miniconda
"""""""""""""""""""""""""""""""""

To install Anaconda or Miniconda on your platform, follow the instructions from `this page <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

Click :ref:`here <anaconda_installation>` for a quick tutorial on MacOs and Linux systems.

Creating a new Anaconda environment
"""""""""""""""""""""""""""""""""""

To create a new environment using Python 3.8 via command line:

.. code-block:: bash

  conda create -n pyleoenv python=3.8

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

Install Pyleoclim through Pypi

The Pypi release contains the most stable version of Pyleoclim.

.. code-block:: bash

  pip install pyleoclim

To install the development version, which contains the most up-to-date features:

.. code-block:: bash

  pip install git+https://github.com/LinkedEarth/Pyleoclim_util.git@Development

If you would like to use Jupyter Notebooks or Spyder for code development, install these packages in your environment:

.. code-block:: bash

  conda install spyder
  conda install jupyter

.. warning::
  The GUI framework used by the LiPD packages may cause a known conflict with the GUI framework for spyder. If this is the case it is safe to downgrade the conflicting packages.

.. code-block:: bash

  pip install 'pyqt5<5.13'
  pip install 'pyqtwebengine<5.13'

Building from source for the f2py feature of WWZ
""""""""""""""""""""""""""""""""""""""""""""""""

The default version of WWZ that comes with the installation steps mentioned above is relying on `Numba <http://numba.pydata.org/>`_.
It is fast enough for lightweight spectral & wavelet analysis tasks, in which case we recommend using the default installation.

However, it could be slow for heavy use (e.g. performing it for hundreds of times on timeseries with length longer than 1000 points), in which case we recommend activating the f2py feature to achieve an acceleration of around 50% (see a loose benchmark notebook `here <https://github.com/LinkedEarth/Pyleoclim_util/blob/Development/example_notebooks/WWZ_numba.ipynb>`_).

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
