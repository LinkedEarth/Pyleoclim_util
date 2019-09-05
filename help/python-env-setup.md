# Python Environment Setup for Pyleoclim

In this tutorial, we will present how to setup a python environment for
Pyleoclim and other general purposes for geosciences.

## Step 1: install miniconda3

First, let's download the installation script for miniconda3 on macOS or Linux:

```bash
# if your OS is macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# if your OS is Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Now execute the installation script:
```bash
chmod +x Miniconda3-latest-*.sh && ./Miniconda3-latest-Linux-x86_64.sh
```

During the installation, a path `<base-path>` needs to be specified as the base location of the python environment.
After the installation is done, we need to add the two lines into your shell environment (e.g., `~/.bashrc` or `~/.zshrc`) as below to enable the `conda` package manager (remember to change `<base-path>` with your real location):
```bash
export PATH="<base-path>/bin:$PATH"
. <base-path>/etc/profile.d/conda.sh
```

To test if miniconda3 is correctly installed:
```
source ~/.bashrc  # assume you are using Bash
which python  # should return a path under <base-path>
which conda  # should return a path under <base-path>
```

## Step 2: create an environment for Pyleoclim

Currently, we recommend to use Python 3.6:
```bash
conda create -n pyleoclim python=3.6
```

Now let's activate the environment:
```bash
conda activate pyleoclim
```

Now you should see a string `(pyleoclim)` ahead of your command prompt, which means that the `pyleoclim` environment has been activated.
You may check your python environment list via:
```bash
conda env list
```


## Step 3.1: install essential packages for geosciences (optional)
Below are some essential packages for geosciences I find useful, but you can skip this step if you want.
It should not affect the installation of pyleoclim.

```bash
conda install -n pyleoclim numpy scipy matplotlib pandas jupyter  # essentials
conda install -n pyleoclim netCDF4 click seaborn statsmodels  # some add-ons
pip install nitime pathos xarray tqdm termcolor  # more add-ons
conda install -n pyleoclim -c conda-forge cartopy basemap  # we still need these to plot maps
pip install rpy2  # rpy2 is useful if you want to load R packages
conda install -n pyleoclim -c conda-forge nco  # NCO

# below is for LMR
conda install -n pyleoclim -c conda-forge esmpy pyspharm
pip install pyyaml
```
Note that when you use `pip`, make sure your targeted environment is activated.

## Step 3.2: install Pyleoclim
Make sure your targeted environment is activated, and we can install Pyleoclim via:
```bash
pip install pyleoclim
```


