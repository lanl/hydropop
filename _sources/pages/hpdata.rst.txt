.. _hpdata:
.. role:: raw-html(raw)
   :format: html

=============
Hydropop Data
=============

A full "run" of hydropop creation results in a set of files that provide hydropop delineations and a suite of attributes and parameters we have identified as relevant to the E3SM-Land Model. This documentation outlines the structure and contents of these data. Note that additional data may be provided (for example, data related to streamflow gages and watersheds). Information about those tables are also provided here, but are not necessarily part of the core hydropop functionality.

Each "run" must be named, and that name is shown as {name} in the following documentation. Some filenames depend on the ids they represent, and these are shown in {} in the tree below. When the number of files in a directory can be variable, `...` is used.

Directory and file structure
----------------------------

The following tree shows how hydropop exports are structured on disk. Note that all core files are contained in the parent directory (called ``{name}``), while auxiliary/optional files will be in subdirectories that may or may not be present depending on the run parameters. That is, all files in the ``{name}`` directory will always exist for any hydropop export, but other directories (e.g. ``streamflow``, ``watersheds``, ``forcings``, etc.) might not.


::

   {name} <-- parent directory
   ├── {name}_hpus.gpkg
   ├── {name}_hpus.tif
   ├── {name}_hpus.shp
   ├── {name}_hpu_classes.gpkg
   ├── {name}_hpu_classes.tif
   ├── {name}_areagrid.tif
   ├── {name}_adjacency.csv
   ├── LICENCE.txt
   ├── streamflow          
      ├── {id_gage_0}.csv
      ├── {id_gage_...}.csv
      └── {id_gage_N}.csv
   ├── watersheds          
      ├── {name}_basins.gpkg
      ├── {name}_hpu_gages.csv
      └── {name}_gages.gpkg
   ├── forcings          
      ├── daily
            ├── {hpu_id_0}.csv
            ├── {hpu_id_...}.cosv
            └── {hpu_id_N}.csv
      ├── hourly
            ├── {hpu_id_0}.csv
            ├── {hpu_id_...}.cosv
            └── {hpu_id_N}.csv


Core files
----------

The following files will be present for any hydropop exports. Note that hydropop processing is all done in unprojected coordinates (EPSG:4326), but care is taken to correctly compute areas and distances when appropriate. All georeferenced outputs therefore also are in the EPSG:4326 coordinate reference system.

.. csv-table::
   :file: ../../doctables/hp_output_files.csv
   :widths: 10,90
   :header-rows: 1


Auxiliary files
---------------

Auxiliary files are not considered part of the core hydropop functionality, and therefore these files may not be present for a general user.

The ``watersheds`` directory contains files about streamflow gages and their watersheds. These data were obtained from the Veins of the Earth (VotE) data platform, which is currently only available to LANL employees. Information about VotE can be found `here <https://www.essoar.org/doi/10.1002/essoar.10509913.2>`_, and LANL collaborators can access the private VotE repository upon request.

The ``streamflow`` directory contains one `csv` per streamflow gage, and each filename corresponds to the `id_gage` provided by VotE (and found within the `watersheds` files). These data were obtained from the Veins of the Earth (VotE) data platform, which is currently only available to LANL employees. Information about VotE can be found `here <https://www.essoar.org/doi/10.1002/essoar.10509913.2>`_, and LANL collaborators can access the private VotE repository upon request.

The ``forcings`` directory contains meterologic and other time-series data either required for or useful to running E3SM-Land models on hydropop units. The data were sampled from the `ERA5-Land Hourly <https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY>`_ dataset on Google Earth Engine (GEE). Some postprocessing to bring units to more standard formats (and other details) is performed, so the band descriptions provided by the GEE Data Catalog might not be exactly accuarte. Each `csv` file is named a corresponding hydropop id and contains time series. See the `hpus.gpkg`_ for more detailed descriptions of the contents of each file.

.. csv-table::
   :file: ../../doctables/auxiliary_files.csv
   :widths: 10, 20, 70
   :header-rows: 1


Individual file contents
------------------------

_`hpus.gpkg`
^^^^^^^^^^^^
.. csv-table::
   :file: ../../doctables/hpu_gpkg.csv
   :widths: 20, 10, 40, 30
   :header-rows: 1


_`basins.gpkg` and _`gages.gpkg`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table:: 
   :file: ../../doctables/basins_gpkg.csv
   :widths: 20, 80
   :header-rows: 1


Forcings csvs aka _`{hpu_id_n}.csv`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. csv-table::
   :file: ../../doctables/forcings.csv
   :widths: 20, 10, 70
   :header-rows: 1
