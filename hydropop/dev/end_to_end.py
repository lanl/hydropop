# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:30:14 2022

@author: 318596
"""

import os
from osgeo import gdal
from skimage.future.graph import RAG
import pandas as pd
import geopandas as gpd
from rivgraph.io_utils import write_geotiff as wg
from rabpro import basin_stats

import hpu.hp_class as hpc
import hpu.hp_utils as hut
import hpu.gee_stats as gee

""" Adjustable parameters """
# HPU creation parameters
pop_breaks = [-11, -10, -4, 0, 100] # coarse = [-11, -10, -4, 0, 100], fine =  [-11, -10, -4, -1, 1, 2, 100]
hthi_breaks = [-.01, .4, .7, 1.01] # coarse = [-.01, .4, .7, 1.01], fine = [-.01, 0.3, 0.55, 0.75, 0.9, 1.01]
min_hpu_size = 20 # in pixels - each HPU will have at least this many pixels

# Path parameters
path_bounding_box = r"X:\Research\CIMMID\Data\Hydropop Layers\Finals\toronto_coarse\roi.gpkg" # shapefile of ROI
path_results = r'X:\Research\CIMMID\Data\Hydropop Layers\Finals\toronto_coarse' # folder to store results
run_name = 'toronto_coarse' # string to prepend to exports
gee_asset = 'projects/cimmid/assets/toronto_coarse_hpus' # the asset path to the hydropop shapefile--this might not be known beforehand but is created upon asset loading to GEE
gdrive_folder_name = 'CIMMID_{}'.format(run_name)

""" Pseduo-fixed parameters/variables """
# Paths to data
path_hthi = r"X:\Research\CIMMID\Data\Hydropop Layers\Hydrotopo Index\hydrotopo_hab_index.tif"
path_pop = r"X:\Research\CIMMID\Data\Hydropop Layers\pop_density_americas.tif"
path_gee_csvs = r'X:\Research\CIMMID\Data\Hydropop Layers\Finals\toronto_coarse\gee' 

""" Path preparation """
path_hpu_class_raster = os.path.join(path_results, '{}_hpu_classes.tif'.format(run_name))
path_hpu_raster = os.path.join(path_results, '{}_hpus.tif'.format(run_name))
path_hpu_gpkg = os.path.join(path_results, '{}_hpus.gpkg'.format(run_name))
path_hpu_shapefile = os.path.join(path_results, '{}_hpus.shp'.format(run_name))
path_adjacency = os.path.join(path_results, '{}_adjacency.csv'.format(run_name))
path_areagrid = os.path.join(path_results, '{}_areagrid.tif'.format(run_name))

# Ensure results folder exists
if os.path.isdir(path_results) is False:
    os.mkdir(path_results)

""" Generate HPUs """
# Instantiate hpu class - can take awhile to load images and do some preprocessing
hpugen = hpc.hpu(path_pop, path_hthi, path_bounding_box)

# Compute classes
breaks = {'hthi':hthi_breaks, 'pop':pop_breaks}
hpugen.compute_hp_classes_ranges(breaks)

# Simplify classes
hpugen.simplify_hpu_classes(minpatch=min_hpu_size, unique_neighbor=False, maxiter=10)

# Assign each class cluster a unique id
hpugen.compute_hpus()

# Export adjacency
rag = RAG(hpugen.I['hpu'], connectivity=2)
adj_dict = {i: list(rag.neighbors(i)) for i in list(rag.nodes)}
hpu_ids = adj_dict.keys()
adj_vals = []
for hid in hpu_ids:
    adj_vals.append(','.join([str(v) for v in adj_dict[hid]]))
adj_df = pd.DataFrame({'hpu_id':hpu_ids, 'adjacency':adj_vals})
adj_df.to_csv(path_adjacency, index=False)

# Export HPU rasters
hpugen.export_raster('hpu', path_hpu_raster)
hpugen.export_raster('hpu_simplified', path_hpu_class_raster)

# Compute areagrid required for computing HP unit areas
agrid = hut.areagrid(path_hpu_raster)
gdobj = gdal.Open(path_hpu_raster)
wg(agrid, gdobj.GetGeoTransform(), gdobj.GetProjection(), path_areagrid, dtype=gdal.GDT_Float32)

""" Compute statistics for HPUs """
# First, we do zonal stats on the locally-available rasters
# HPU stats and properties
do_stats = {'hthi' : [path_hthi, ['mean']],
           'pop' : [path_pop, ['mean']],
           'area' : [path_areagrid, ['sum'], path_hpu_raster],
           'hpu_class' :[path_hpu_class_raster, ['majority']]}
hpugen.compute_hpu_stats(do_stats)
# Export the geopackage that contains all the HPU attributes
hpugen.hpus.to_file(path_hpu_gpkg, driver='GPKG')
# For the shapefile export, we only need the HPU id and the polygon
hpus_shp = gpd.GeoDataFrame(hpugen.hpus[['hpu_id', 'geometry']])
hpus_shp.crs = hpugen.hpus.crs
hpus_shp.to_file(path_hpu_shapefile) # shapefile needed to upload to GEE

""" STOP. Here you need to upload the hpu shapefile as a GEE asset. """

""" Update the gee_asset variable. """
datasets, Datasets = gee.generate_datasets()

# check and do fmax
if 'fmax' in datasets.keys():
    filename_out = 'fmax'
    gee.export_fmax(gee_asset, filename_out, gdrive_folder_name)
 
# Spin up other datasets
urls, tasks = basin_stats.compute(Datasets,
                                  gee_feature_path=gee_asset,
                                  folder=gdrive_folder_name)

""" STOP. Download the GEE exports (csvs) to path_gee_csvs """
hpus = gpd.read_file(path_hpu_gpkg)
gee_csvs = os.listdir(path_gee_csvs)
for key in datasets.keys():

    # Find the csv associated with a dataset
    if key == 'fmax':
        look_for = 'fmax'
    else:
        look_for = datasets[key]['path']
        if datasets[key]['band'] != 'None':
            look_for = look_for + '__' + datasets[key]['band']
        look_for = look_for.replace('/', '-')
    this_csv = [c for c in gee_csvs if look_for in c][0]
    
    # Ingest it
    csv = pd.read_csv(os.path.join(path_gee_csvs,this_csv))
    
    # Handle special cases first
    if key == 'fmax':
        csv = csv[['fmax', 'hpu_id']]
    elif key == 'land_use':
        csv = csv[['histogram', 'hpu_id']]
        csv = gee.format_lc_type1(csv, fractionalize=True, prepend='lc_')
    else:
        keepcols = ['hpu_id']
        renamer = {}
        if 'mean' in datasets[key]['stats']:
            keepcols.append('mean')
            renamer.update({'mean' : key + '_mean'}) 
        if 'std' in datasets[key]['stats'] or 'stdDev' in datasets[key]['stats']:
            keepcols.append('stdDev')
            renamer.update({'stdDev' : key + '_std'}) 
        csv = csv[keepcols]
        csv = csv.rename({'mean': key + '_mean'}, axis=1)
        
    hpus = pd.merge(hpus, csv, left_on='hpu_id', right_on='hpu_id')
        
hpus.to_file(path_hpu_gpkg, driver='GPKG')

# Overlay watersheds
path_hpus = r"X:\Research\CIMMID\Data\Hydropop Layers\Finals\toronto_coarse\toronto_coarse_hpus.shp"
path_watersheds = r"X:\Research\CIMMID\Data\Watersheds\Toronto\initial_basins.gpkg"
hpus = gpd.read_file(path_hpus)
watersheds = gpd.read_file(path_watersheds)
df = hut.overlay_watersheds(hpus, watersheds)
df.to_csv(r'X:\Research\CIMMID\Data\Hydropop Layers\Finals\toronto_coarse\watershed_info.csv', index=False)
        








