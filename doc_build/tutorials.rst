.. _tutorials:

Tutorials
=========

Tutorials (and functional science examples) for Pyleoclim exist in the form of:

* Jupyter Notebooks, which are found on the following repositories:

  * `PyleoTutorials <http://linked.earth/PyleoTutorials/>`_ contains notebooks demonstrating simple workflows with Pyleoclim, such as:

    * Loading data from different formats into Pyleoclim objects

    * Basic plotting and time series manipulation

    * Timeseries analysis such as spectral analysis, wavelet analysis, coherence, singular spectrum analysis, paleo-aware correlation

  * `PaleoBooks <https://linked.earth/PaleoBooks/>`_ contain more advanced scientific workflows, some featuring Pyleoclim. 

  * `Pyleoclim paper notebooks <https://github.com/LinkedEarth/PyleoclimPaper>`_ highlights three scientific case studies featuring Pyleoclim.

  * `PaleoHackathon notebooks <https://github.com/LinkedEarth/paleoHackathon>`_ (contact the crew for solutions).

* `The LinkedEarth YouTube Channel <https://www.youtube.com/playlist?list=PL93NbaRnKAuF4WpIQf-4y_U4lo-GqcrcW>`_.

Note that additional packages may need to be installed to run these various scientific examples. In particular, we recommend installing the `xarray <https://docs.xarray.dev/en/stable/getting-started-guide/installing.html>`_ package suite.

.. code-block:: bash

  conda install -c conda-forge xarray dask netCDF4 bottleneck

You may also need `climlab <https://climlab.readthedocs.io/en/latest/>`_:

.. code-block:: bash

  conda install climlab

The various repositories listed above have environment files that detail the necessary packages beyond Pyleoclim itself. No package is an island! Another option to run these tutorials is the `LinkedEarth Hub <http://linked.earth/research_hub.html>`_

If you still have questions, please see our  `Discourse forum <https://discourse.linked.earth>`_.
