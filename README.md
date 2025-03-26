<!---[![PyPI](https://img.shields.io/pypi/dm/pyleoclim.svg)](https://pypi.python.org/pypi/Pyleoclim)-->
[![PyPI version](https://badge.fury.io/py/pyleoclim.svg)](https://badge.fury.io/py/pyleoclim)
[![PyPI](https://img.shields.io/badge/python-3.10-yellow.svg)]()
[![license](https://img.shields.io/github/license/linkedearth/Pyleoclim_util.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6999279.svg)](https://doi.org/10.5281/zenodo.6999279)
[![NSF-1541029](https://img.shields.io/badge/NSF-1541029-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1541029)


[![Downloads](https://static.pepy.tech/badge/pyleoclim)](https://pepy.tech/project/pyleoclim)
[![Downloads](https://static.pepy.tech/badge/pyleoclim/month)](https://pepy.tech/project/pyleoclim)
[![Downloads](https://static.pepy.tech/badge/pyleoclim/week)](https://pepy.tech/project/pyleoclim)

![](https://github.com/LinkedEarth/Logos/raw/master/pyleoclim_logo_full_white.png)

**Python Package for the Analysis of Paleoclimate Data**

[Paleoclimate](https://www.ncdc.noaa.gov/news/what-paleoclimatology) data, whether from observations or model simulations, offer unique challenges to the analyst, as they usually come in the form of timeseries with missing values and age uncertainties, which trip up off-the-shelf methods.
Pyleoclim is a Python package primarily geared towards the analysis and visualization of such timeseries. The package includes several low-level methods to deal with these issues under the hood, leaving paleoscientists to interact with intuitive, high-level analysis and plotting methods that support publication-quality scientific workflows.

There are many entry points to Pyleoclim, thanks to its underlying [data structures](https://pyleoclim-util.readthedocs.io/en/latest/core/api.html). Low-level modules work on [NumPy](http://www.numpy.org) arrays or [Pandas](https://pandas.pydata.org) dataframes.

We've worked hard to make Pyleoclim accessible to a wide variety of users, from establisher researchers to high-school students, and from seasoned Pythonistas to first-time programmers. A progressive introduction to the package is available at [PyleoTutorials](http://linked.earth/PyleoTutorials/). Examples of scientific use are given [this paper](https://doi.org/10.1029/2022PA004509).  A growing collection of research-grade workflows using Pyleoclim and the LinkedEarth research ecosystem are available as Jupyter notebooks on [paleoBooks](http://linked.earth/PaleoBooks/), with video tutorials on the LinkedEarth [YouTube channel](https://www.youtube.com/watch?v=LJaQBFMK2-Q&list=PL93NbaRnKAuF4WpIQf-4y_U4lo-GqcrcW). Pyleoclim is part of the broader Python ecosystem of [Computational Tools for Climate Science](https://neuromatch.io/computational-tools-for-climate-science-course/).  Python novices are encouraged to follow these [self-paced tutorials](http://linked.earth/LeapFROGS) before trying Pyleoclim.

Science-based training materials are also available from the [paleoHackathon repository](https://github.com/LinkedEarth/paleoHackathon). We also run live training workshops every so often. Follow us on [Twitter](https://twitter.com/Linked_Earth), or join our [Discourse Forum](https://discourse.linked.earth) for more information.

### Versions

See our [releases page](https://github.com/LinkedEarth/Pyleoclim_util/releases) for details on what's included in each version.

### Documentation

Online documentation is available through [readthedocs](https://pyleoclim-util.readthedocs.io/en/latest/).

### Dependencies

pyleoclim **only** supports Python 3.11

### Installation

The latest stable release is available through Pypi. We recommend using Anaconda or Miniconda with a dedicated environment. Full installation instructions are available in the [package documentation](https://pyleoclim-util.readthedocs.io/en/latest/installation.html)

## Citation
If you use our code in any way, please consider adding these citations to your publications:

- Khider, D., Emile-Geay, J., Zhu, F., James, A., Landers, J., Ratnakar, V., & Gil, Y. (2022). Pyleoclim: Paleoclimate timeseries analysis and visualization with Python. Paleoceanography and Paleoclimatology, 37, e2022PA004509. https://doi.org/10.1029/2022PA004509
- Khider, Deborah, Emile-Geay, Julien, Zhu, Feng, James, Alexander, Landers, Jordan, Kwan, Myron, & Athreya, Pratheek. (2022). Pyleoclim: A Python package for the analysis and visualization of paleoclimate data (v0.9.1). Zenodo. https://doi.org/10.5281/zenodo.7523617

### Development

Pyleoclim development takes place on GitHub: https://github.com/LinkedEarth/Pyleoclim_util

Please submit any reproducible bugs you encounter to the [issue tracker](https://github.com/LinkedEarth/Pyleoclim_util/issues). For usage questions, please use [Discourse](https://discourse.linked.earth).


### License

The project is licensed under the GNU Public License. Please refer to the file call license.
If you use the code in publications, please credit the work using the citation file. 


### Disclaimer

This material is based upon work supported by the National Science Foundation under Grant Number ICER-1541029. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the investigators and do not necessarily reflect the views of the National Science Foundation.

This research is funded in part by JP Morgan Chase & Co. Any views or opinions expressed herein are solely those of the authors listed, and may differ from the views and opinions expressed by JP Morgan Chase & Co. or its affilitates. This material is not a product of the Research Department of J.P. Morgan Securities LLC. This material should not be construed as an individual recommendation of for any particular client and is not intended as a recommendation of particular securities, financial instruments or strategies for a particular client. This material does not constitute a solicitation or offer in any jurisdiction.
