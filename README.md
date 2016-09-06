# Pyleoclim
**Python Package for the Analysis of Paleoclimate Data**

About the current version: This version includes basic functionality such as mapping and plotting. Check the Jupyter Notebook for a documentation of the various functions.

More advanced functionalities are currently under development.

## What is it?

Pyleoclim is a Python package for the analyses of paleoclimate data.

## Installation

Python v3.5+ is required.

Pyleoclim is published through PyPi and easily installed via `pip`
```
pip install Pyleoclim
```

## Quickstart guide

1. Open your command line application (Terminal or Command Prompt).

2. Install with command: `pip install Pyleoclim`

3. Wait for installation to complete, then:

    3a. Import the package into your favorite Python environment (we recommend the use of Spyder, which comes standard with the Anaconda package)

    3b. Use Jupyter Notebook to go through the tutorial contained in the `PyleoclimQuickstart.ipynb` Notebook, which can be downloaded [here](https://github.com/LinkedEarth/Pyleoclim_util).

## Features

- Mapping
    - Map all the records by proxy time
    - Map a specific record
- Plotting
    - Plot time series of a specific record

## Requirements

- LiPD v0.1+
- pandas v0.18+
- numpy v1.11+
- matplotlib v1.5+
- Cartopy v0.13+

The installer will automatically check for the needed updates ***except*** for Cartopy.

<div class="alert alert-warning" role="alert" style="margin: 10px">
<p><b>NOTE</b></p>
<p>Cartopy does not install properly through pip. The recommended method is through <a href="http://conda.pydata.org/miniconda.html"> Conda</a>. See the instructions on the <a href="http://scitools.org.uk/cartopy/docs/latest/installing.html"> developer website</a>.</p>
</div>

## Further information

GitHub: https://github.com/LinkedEarth/Pyleoclim_util

LinkedEarth: http://linked.earth

Python and Anaconda: http://conda.pydata.org/docs/test-drive.html

Jupyter Notebook: http://jupyter.org

## Contact

Please report issues to <linkedearth@gmail.com>

## License

The project is licensed under the GNU Public License. Please refer to the file call license.

## Disclaimer

This material is based upon work supported by the National Science Foundation under Grant Number ICER-1541029. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the investigators and do not necessarily reflect the views of the National Science Foundation.
