.. _anaconda_installation:

Installing Miniconda (Mac, Linux)
=================================

Anaconda is a very large install, containing not only Python, but also R, and a host of other things that are not necessary to run pyleoclim.
Users may find it preferable to install the minimalist "miniconda" package.

Step 1: Download the installation script for miniconda3
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

macOS (Intel)
'''''''''''''

.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

macOS (Apple Silicon)
'''''''''''''''''''''

.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

Linux
-----
.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Step 2: Install Miniconda3
"""""""""""""""""""""""""""

.. code-block:: bash

  chmod +x Miniconda3-latest-*.sh && ./Miniconda3-latest-Linux-x86_64.sh

During the installation, a path `<base-path>` needs to be specified as the base location of the python environment.
After the installation is done, we need to add the two lines into your shell environment (e.g., `~/.bashrc` or `~/.zshrc`) as below to enable the `conda` package manager (remember to change `<base-path>` with your real location):

.. code-block:: bash

  export PATH="<base-path>/bin:$PATH"
  . <base-path>/etc/profile.d/conda.sh

Step 3: Test your Installation
"""""""""""""""""""""""""""""""

.. code-block:: bash

  source ~/.bashrc  # assumes you are using Bash shell
  which python  # should return a path under <base-path>
  which conda  # should return a path under <base-path>
