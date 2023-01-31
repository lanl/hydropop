# Hydropop Data

# Background
Hydropop units are spatially-continuous regions that feature similar 1) human population density and 2) hydrotopo habitat index (HTHI) (roughly speaking, mosquito habitat suitability). There are a number of ways to draw boundaries given these data, and we've played with two in particular: k-means (unsupervised clustering) and range-setting. We have settled on range-setting because it allows a level of control over the hydropop unit resolution that k-means (or other automated clustering methods) do not. 

range-setting refers to pre-determining ranges of population density and HTHI that define hydropop unit classes. Since these may be selected somewhat arbitrarily (see Figure 1 below which shows how we used the global histograms to guide our decision), there are many possible combinations of ranges. We therefore have designed a _coarse_ and a _fine_ range set:


### Output files
The following table lists the minimum set of files a hydropop "run" will create. For each, `{rn}` refers to "run name", a parameter provided at the initialization of hydropop creation. Some of the files have their own tables that provide further details about the attributes found within them.

| file | description | details |
| - | ------ | -- |
| roi.shp | A polygon (EPSG:4326) representing the study area considered for these files. | Any polygon can be provided; if you want a different study area, send me a message. |
| hpus.gpkg | A GeoPackage (EPSG:4326) containing the HPUs as polygons (or MultiPolygons) and some attributes. | Attributes include: <br /> **hpu_id** : unique HPU identifier <br /> **hthi_mean**: the mean value of the hydrotopo habitat index within that HPU <br /> **pop_mean**: the mean value of the human population within the HPU (need to check units here) <br /> **area_sum**: the area of the HPU in square km <br /> **hpu_class**: the class to which the HPU belongs <br /> **centroid_x**: the longitude of the centroid of the HPU <br /> **centroid_y**: the latitude of the centroid of the HPU |
| hpus.tif | A geotiff (EPSG:4326) of Hydropop Units. Pixel values represent the unit to which the HPU belongs. |  |
| hpu_classes.tif | A geotiff (EPSG:4326) of Hydropop classes. Pixel values represent the class to which the HPU belongs. A HP class is the "cluster" to which the pixel belongs. There are as many classes as groups specified to the clustering algorithm (kmeans in this case). Classes, in general, have similar human populations and hydrotopo habitat potential values. HPUs are derived from classes by considering spatial connectivity. |  |
| areagrid.tif | A geotiff (EPSG:4326) for which pixel values represent the area of the pixel in square km. This is used for computing actual HPU areas, as working in unprojected coordinate systems (4326) require a bit of extra work to estimate pixel areas in meaningful units (km instead of degrees). |  |
| adjacency.csv | Provides connectivity information among HPUs. | Essentially a .csv with two columns: **hpu_id** of each hydropop unit in the dataset, and **adjacency** that specifies each hpu_id's neighbors as a comma-separated string |


<table style=tight-table>
<thead>
<tr>
<th>head1</th>
<th>head2</th>
<th>head3</th>
<th>head4</th>
</tr>
</thead>
<tfoot>
<tr>
<td>foot1</td>
<td>foot2</td>
<td>foot3</td>
<td>foot4</td>
</tr>
</tfoot>
<tbody>
<tr>
<td>cell1_1</td><td>cell2_1cell2_1cell2_1cell2_1cell2_1cell2_1cell2_1cell2_1cell2_1cell2_1cell2_1cell2_1cell2_1cell2_1cell2_1</td><td>cell3_1</td><td>cell4_1</td></tr>
<tr>
<td>cell1_2</td><td>cell2_2</td><td>cell3_2</td><td>cell4_2</td></tr>
<tr>
<td>cell1_3</td><td>cell2_3</td><td>cell3_3</td><td>cell4_3</td></tr>
<tr>
<td>cell1_4</td><td>cell2_4</td><td>cell3_4</td><td>cell4_4</td></tr>
<tr>
<td>cell1_5</td><td>cell2_5</td><td>cell3_5</td><td>cell4_5</td></tr>
</tbody>
</tr>
</table>

<!-- |  | coarse | fine |
| ------ | ------ | ------ |
| **N HPU classes** | 12 | 30 |
| **population intervals**| [-10], (-10, -4], (-4, 0], (0, >5] | [-10], (-10,-4], (-4, -1], (-1, 1], (1, 2], (2, >5] |
| **HTHI intervals**| [0, 0.4], (0.4, 0.7], (0.7, 1] | [0, 0.3], (0.3, 0.55], (0.55, 0.75], (0.75, 0.9], (0.9, 1] | -->

<!-- Note that population density pixels whose values were == 0 were set to 10^-10. This is because population density values are log-transformed.

![image](/uploads/42e512980899ae847fc1d4952f88ec32/image.png)
_Figure 1. Histograms of pixel counts for population density and hydrotopop-habitat index covering all the Americas._

# Where are the data?
Hydropop files are located on Darwin, per the table below. _kmeans_ and _coarse_ and _fine_ are explained in the **Background** section above.

| id | location| description |
| ------ | ------ | ------ |
| [**deprecated**] Toronto - kmeans | /projects/cimmid/hydropops/HPU_Toronto_1 | covers greater Toronto, used a k-means with 10 classes |
| Iquitos - coarse | /projects/cimmid/hydropops/hpu_iquitos_1/coarse | covers Iquitos and surrounding area using the coarse HPU ranges |
| Iquitos - fine | /projects/cimmid/hydropops/hpu_iquitos_1/fine | covers Iquitos and surrounding area using the fine HPU ranges |

# What do the files contain?
Note that these files and structures are subject to change; **however**, files uploaded to Darwin are considered final and will only be changed if an error is found. Otherwise, we will simply add more HPU delineations as needed. 

**Contents of an HPU data directory**
| file | description | details |
| ------ | ------ | ------ |
| roi.shp | A polygon (EPSG:4326) representing the study area considered for these files. | Any polygon can be provided; if you want a different study area, send me a message. |
| hpus.gpkg | A GeoPackage (EPSG:4326) containing the HPUs as polygons (or MultiPolygons) and some attributes. | Attributes include: <br /> **hpu_id** : unique HPU identifier <br /> **hthi_mean**: the mean value of the hydrotopo habitat index within that HPU <br /> **pop_mean**: the mean value of the human population within the HPU (need to check units here) <br /> **area_sum**: the area of the HPU in square km <br /> **hpu_class**: the class to which the HPU belongs <br /> **centroid_x**: the longitude of the centroid of the HPU <br /> **centroid_y**: the latitude of the centroid of the HPU |
| hpus.tif | A geotiff (EPSG:4326) of Hydropop Units. Pixel values represent the unit to which the HPU belongs. |  |
| hpu_classes.tif | A geotiff (EPSG:4326) of Hydropop classes. Pixel values represent the class to which the HPU belongs. A HP class is the "cluster" to which the pixel belongs. There are as many classes as groups specified to the clustering algorithm (kmeans in this case). Classes, in general, have similar human populations and hydrotopo habitat potential values. HPUs are derived from classes by considering spatial connectivity. |  |
| areagrid.tif | A geotiff (EPSG:4326) for which pixel values represent the area of the pixel in square km. This is used for computing actual HPU areas, as working in unprojected coordinate systems (4326) require a bit of extra work to estimate pixel areas in meaningful units (km instead of degrees). |  |
| adjacency.csv | Provides connectivity information among HPUs. | Essentially a .csv with two columns: **hpu_id** of each hydropop unit in the dataset, and **adjacency** that specifies each hpu_id's neighbors as a comma-separated string |


## Attribute description for HU geopackage exports
**Metadata for HU fields**
| attribute | units | description | source |
| ------ | ------ | ------ | ------ |
| fid | n/a | Unique feature id created upon export; essentially meaningless. | n/a |
| hpu_id | n/a | Unique hydropop id. | n/a |
| hpu_class | n/a | Class to which this HU belongs. | n/a |
| hthi_mean | n/a | Average hydrotopo-habitat index value across the HU. | n/a |
| pop_mean | n people per 0.01 km^2 | Average population density across the HU. | Worldpop Estimated Residential Population per 100x100m Grid Square for 2020 |
| area_sum | km^2 | Area of the HU. | n/a |
| centroid_x, centroid_y| degrees | coordinates of the HU centroid, EPSG:4326. | n/a |
| fmax | n/a | fmax parameter required by ELM, see above for description. | MERIT-DEM + GEE + custom function |
| elevation_mean | m.a.s.l. | Average elevation across the HU. | MERIT-DEM |
| stdDev [need to change this] | m | Standard deviation of elevations across the HU. | MERIT-DEM |
| soil_depth_mean | m | Average soil depth across the HU. | Pelletier, 2016 |
| topo_slope_mean | degrees | Average topographic slope across the HU | MERIT-DEM via https://www.nature.com/articles/s41597-020-0479-6 | 
| soc_d1-d2_mean | dg/kg | Average soil organic carbon across the HU between depths d1 and d2. | SoilGrids 2.0 via https://soil.copernicus.org/articles/7/217/2021/soil-7-217-2021.html |
| clay_d1-d2_mean | g/kg | Average soil clay content across the HU between depths d1 and d2. | SoilGrids 2.0 via https://soil.copernicus.org/articles/7/217/2021/soil-7-217-2021.html |
| sand_d1-d2_mean | g/kg | Average soil sand content across the HU between depths d1 and d2. | SoilGrids 2.0 via https://soil.copernicus.org/articles/7/217/2021/soil-7-217-2021.html |
| lc_XXX | Between 0-1 | Fraction of HU covered by land cover type XXX. | MCD12Q1.006 MODIS Land Cover Type Yearly Global 500m for 2015 via https://lpdaac.usgs.gov/documents/101/MCD12_User_Guide_V6.pdf | -->


<!-- ## Ingestion

### Sources

Daily streamflow data were curated from at least 11 sources, with other sources used to fill metadata gaps or provide catchment boundaries where available. Some sources, such as CAMELS and CAMELS-like, were provided as published datasets while others were provided via direct APIs from the agencies. The following table describes the sources that have been ingested into VotE thus far.

<span style="font-size:0.7em;">[GSIM: Global Streamflow and Metadata Archive](https://essd.copernicus.org/articles/10/765/2018/)

[GAGES II: Geospatial Attributes of Gages for Evaluating Streamflow](https://water.usgs.gov/GIS/metadata/usgswrd/XML/gagesII_Sept2011.xml)

[HYSETS: The Hydrometeorological Sandbox - École de technologie supérieure](https://www.nature.com/articles/s41597-020-00583-2)</span>

### Ingestion scripts

An ingestion script was written for each source as generalizing across all sources was not effective due to 1) different filetypes (e.g. .mdb, .csv, direct API, etc.), 2) different headers and column names, 3) different available metadata, 4) use of auxiliary datasets for some sources that required special handling (e.g. matching gage IDs from a source that provided catchment polygons), and 5) other minor differences not stated here. These ingestion scripts can be found [here XXX](XXX).

The purpose of each ingestion script is to standardize the source data and push it into the VotE streamgage database. Streamgage data are stored in two tables; `gages` contains gage metadata including basin geometries and streamflow availability metrics, while `streamflow` stores the daily streamflow values and their quality flags. See XXX for details about these tables. In general, ingestion scripts followed the following steps, with modifications depending on the needs of the data:

1. Download streamflow, metadata from API OR load static source data into pandas (Geo)DataFrames
2. Load catchment geofiles if available
3. Ensure VotE tables (`gages` and `streamflow`) exist; create if not
4. Compute unique ids for each gage based on its coordinates (see XXX); slightly "jitter" the coordinates if the id already exists in the `gages` table to ensure uniqueness
5. Reshape streamflow data such that each row contains a single daily streamflow observation
6. Rename (Geo)DataFrames for consistency

For each gage:

   7. Ensure gage id not in `gages` table
   8. Convert streamflow to cubic meters per second if necessary
   9. Remove nodata and obviously errant streamflow values (e.g. <0)
   10. Fetch appropriate catchment geometry if exists
   11. Compute streamflow availability metrics
   12. Determine `naturalish` flag if possible; see XXX
   13. Use auxiliary sources to complete metadata where possible
   14. Rename metadata columns for consistency with VotE tables
   15. Push streamgage metadata, catchments, etc. into `gages`
   16. Push daily streamflow to `streamflow`

The steps above are general and do not cover all operations performed in order to ingest streamflow data; each ingestion script contains elements of the above but implementation is different for each. We do not provide explicit documentation for each ingestion script, but the guidance of these docs paired with in-script comments should make clear the steps taken to ingest each source.

## Storage

Only two tables store all streamgage information including metadata, catchments, availability info, VotE-integration info, streamflow data, and quality flags: `gages` and `streamflow`.

### The `gages` table


#### Naturalish

The `naturalish` flag is provided as a convenient way to identify gages whose catchments have been relatively unimpacted by human alterations (dams, diversions, etc.). We chose the term "naturalish" to indicate that practically no watersheds are fully natural. These flags are derived from information provided by the sources themselves, and as such **are not defined the same way** across all sources. Some sources provide one or multiple metrics of human impacts; in these cases we followed the source'(s) guidance for designating the `naturalish` flag. The following table describes how we determined `naturalish` for each source that provided sufficient information.

put table here XXX

### The `streamflow` table

Streamflow data are stored in "deflated" format, which simply means that no NoData values are stored, so there may be date discontinuities for a given record. See the docs for accessing streamflow data XXX for tools to "inflate" a record.


#### Quality flags

There are two sources of quality flags, and each has its own column in the `streamflow` table. The first,  `q_quality` contains those flags provided by the source. The following table details these flags' documentation locations.


The second, `whatever it's called XXX` contains flags computed by VotE following the GSIM [] paper. These flags indicate the following:


## VotE-Fusion of gages

blah blah

## Deduplication

blah blah

## Test

| :memo:        | Take note of this test       |
|---------------|:------------------------|

## Querying

It is impossible to query this data, we just put it together to tease you. -->
