.. _installation:

.. note::

   Pyleoclim requires the use of Python 3.9 or above

Installing Pyleoclim
====================

If you know what you are doing, you may install Pyleoclim in any suitable Python environment, with a Python version >=3.9.

However, we have not and cannot possibly, try every situation. 

If you are new to Python, we recommend the use of Anaconda (or its minimal version Miniconda), to set up such an environment. Then you may install Pyleoclim via pip.


Installing Anaconda or Miniconda
"""""""""""""""""""""""""""""""""

To install Anaconda or Miniconda on your platform, follow the instructions from `this page <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

Click :ref:`here <anaconda_installation>` for a quick tutorial on MacOs and Linux systems.

Creating a new conda environment
"""""""""""""""""""""""""""""""""""
As of Nov 2023, we recommend Python 3.11. Create an environment via the command line (e.g. Terminal app in MacOS):

.. code-block:: bash

  conda create -n pyleo python=3.11

To view a list of available environments:

.. code-block:: bash

  conda env list

To activate the new environment:

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
Once the pyleo environment is activated, simply run:

.. code-block:: bash

  pip install pyleoclim

This will install the latest official release, which you can view `here <https://pypi.org/project/pyleoclim/>`_. To install the latest version, which contains the most up-to-date features, you can install directly from the GitHub source:

.. code-block:: bash

  pip install git+https://github.com/LinkedEarth/Pyleoclim_util.git

This version may contain bugs not caught by our continuous integration test suite; if so, please report them via `github issues <https://github.com/LinkedEarth/Pyleoclim_util/issues>`_
If you would like to use Spyder for code development:

.. code-block:: bash

  conda install spyder
  
If you intend on using Pyleoclim within a Jupyter Notebook, we recommend using `ipykernel <https://anaconda.org/anaconda/ipykernel>`_.   
  
.. code-block:: bash

  conda install ipykernel    
  python -m ipykernel install --user --name=pyleo       
  
The first line will install ipykernel and its dependencies, including IPython, Jupyter, etc. The second line will make sure the pyleo environment is visible to Jupyter (see `this page for context <https://queirozf.com/entries/jupyter-kernels-how-to-add-change-remove>`_)


Building from source for the f2py feature of WWZ
""""""""""""""""""""""""""""""""""""""""""""""""

The default version of WWZ that comes with the installation steps mentioned above is relying on `Numba <http://numba.pydata.org/>`_.
It is fast enough for lightweight spectral & wavelet analysis tasks, in which case we recommend using the default installation.

However, it could be slow for heavy use (e.g. performing it hundreds of times on timeseries longer than 1000 points), in which case we recommend activating the f2py feature to achieve a speedup of ~50%.

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
Docker containers with various versions of Pyleoclim are available `on quay.io <https://quay.io/repository/linkedearth/pyleoclim?tab=tags>`_.

To pull an image:

.. code-block:: bash

  docker pull quay.io/linkedearth/pyleoclim:latest

To run the image:

.. code-block:: bash

  docker run -it -p 8888:8888 quay.io/linkedearth/pyleoclim:latest

The container will start a Jupyter server automatically. You need to copy the link to the server (localhost) into your web browser on your machine (the command -p 8888:8888 opens the communication port between your machine and the container). You can then create notebook and upload notebook and data using the Jupyter interface. Remember that the container will not save any of your work if you close it. So make sure you donwload your work before closing the container.
