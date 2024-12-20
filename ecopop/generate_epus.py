# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:30:14 2022

@author: 318596
"""
import os
import sys
sys.path.append(r'C:\Users\318596\Desktop\ecopop\ecopop\ecopop')
import ep_class as epc
import ep_utils as eut
# import gee_stats as gee
from osgeo import gdal
import pandas as pd
import geopandas as gpd
from rivgraph_ports import write_geotiff as wg
import rabpro
from matplotlib import pyplot as plt
import numpy as np

""" Pseduo-fixed parameters/variables """
# Paths to data
path_hthi = r"C:\Users\318596\Desktop\ecopop\data\hydrotopo_hab_index.tif"
path_pop = r"C:\Users\318596\Desktop\ecopop\data\pop_density_americas.tif"
path_watermask = r"C:\Users\318596\Desktop\ecopop\data\hydrolakes_areas_americas.tif"

""" Adjustable parameters """
# epu creation parameters
pop_breaks = [-11, -4, 0, 100] # coarse = [-11, -10, -4, 0, 100], fine =  [-11, -10, -4, -1, 1, 2, 100]
hthi_breaks = [-.01, .4, .7, 1.01] # coarse = [-.01, .4, .7, 1.01], fine = [-.01, 0.3, 0.55, 0.75, 0.9, 1.01]
# target_epu_size = 1200 # in pixels - not guaranteed, but will try to make each epu this size
# Waterbody size based on square km, nominally each pixel is 1km2
# max_waterbody_size = target_epu_size # Waterbodies bigger than this will be masked

# Path parameters
path_bounding_box = r"C:\Users\318596\Desktop\ecopop\data\bounding_box_small.gpkg" # shapefile of ROI
path_results = r'C:\Users\318596\Desktop\ecopop\results' # folder to store results
run_name = 'test_port' # string to prepend to exports
gdrive_folder_name = 'CIMMID_{}'.format(run_name)

""" Area calcs """
max_waterbody_size = 1000 # square km, maximum size to be included in epus (anything larger is NoData)
target_no_epus = 10000 # total number of discrete epus (not classes)
forarea = epc.epu(path_pop, path_hthi, bounding=path_bounding_box, path_water_raster=path_watermask, waterbody_thresh=max_waterbody_size)
path_mask_temp = os.path.join(path_results, 'temp_mask.tif')
forarea.export_raster('mask', path_mask_temp)
gdobj = gdal.Open(path_mask_temp)
Iagrid = eut.areagrid(path_mask_temp)
Iagrid[forarea.I['mask']==0] = np.nan
area_km2 = np.nansum(Iagrid)
avg_pix_area = np.nanmean(Iagrid)
target_pix_per_epu = forarea.I['mask'].sum() / target_no_epus
target_epu_area_km2 = area_km2 / target_no_epus
min_epu_size = int(target_pix_per_epu/10)


""" Here we go """
paths = eut.prepare_export_paths(path_results, run_name)

# Ensure results folder exists
if os.path.isdir(path_results) is False:
    os.mkdir(path_results)

""" Generate epus """
# Instantiate epu class - can take awhile to load images and do some preprocessing and masking
epugen = epc.epu(path_pop, path_hthi, bounding=path_bounding_box, path_water_raster=path_watermask, waterbody_thresh=max_waterbody_size)

# Compute classes
breaks = {'hthi':hthi_breaks, 'pop':pop_breaks}
epugen.compute_ep_classes_ranges(breaks)

# Simplify classes
epugen.simplify_epu_classes(min_class_size=min_epu_size, unique_neighbor=False, maxiter=10)

# Compute epus from classes image - this will need to be sped up to run at this scale
epugen.compute_epus(target_pix_per_epu, min_epu_size)

""" Exporting """
# Export epu rasters
epugen.export_raster('epu_simplified', paths['epu_raster'])
epugen.export_raster('epu_class_simplified', paths['epu_class_raster'])

# Export all epu geopackage (including waterbodies/masked ones) in case needed later
epugen.epus_all.to_file(paths['epu_all'], driver='GPKG')

# Export epu class polygons
classes = eut.polygonize_epu(epugen.I['epu_class_simplified'], epugen.gt, epugen.wkt)
classes.to_file(paths['epu_class_gpkg'], driver='GPKG')

# Export adjacency
adjacency = epugen.compute_adjacency()
adjacency.to_csv(paths['adjacency'], index=False)

# Export areagrid required for computing epu areas
agrid = eut.areagrid(paths['epu_raster'])
gdobj = gdal.Open(paths['epu_raster'])
wg(agrid, gdobj.GetGeoTransform(), gdobj.GetProjection(), paths['areagrid'], dtype=gdal.GDT_Float32)

""" Compute statistics for epus """
# First, we do zonal stats on the locally-available rasters
# epu stats and properties
do_stats = {'hthi' : [path_hthi, ['mean']],
           'pop' : [path_pop, ['mean']],
           'area' : [paths['areagrid'], ['sum'], paths['epu_raster']],
           'epu_class' :[paths['epu_class_raster'], ['majority']]}
epugen.compute_epu_stats(do_stats)
# Export the geopackage that contains all the epu attributes
epugen.epus.to_file(paths['epu_gpkg'], driver='GPKG')
# For the shapefile export, we only need the epu id and the polygon
epus_shp = gpd.GeoDataFrame(epugen.epus[['epu_id', 'geometry']])
epus_shp.crs = epugen.epus.crs
epus_shp.to_file(paths['epu_shapefile']) # shapefile needed to upload to GEE

""" STOP. Here you need to upload the epu shapefile as a GEE asset. """
gee_asset = 'projects/cimmid/assets/na_10k_epus' # the asset path to the ecopop shapefile

""" Update the gee_asset variable. """
import gee_stats as gee
datasets, Datasets = gee.generate_datasets()

# Do fmax
if 'fmax' in datasets.keys():
    filename_out = 'fmax'
    gee.export_fmax(gee_asset, filename_out, gdrive_folder_name)
 
# Spin up other datasets
urls, tasks = rabpro.basin_stats.compute(Datasets,
                           gee_feature_path=gee_asset,
                           folder=gdrive_folder_name)

""" STOP. Download the GEE exports (csvs) to path_gee_csvs """
path_gee_csvs = r'X:\Research\CIMMID\Data\ecopop Layers\Finals\na_10k\gee_exports'

""" Aggregate the csvs """
epus = gpd.read_file(paths['epu_gpkg'])
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
        csv = csv[['fmax', 'epu_id']]
    elif key == 'land_use':
        csv = csv[['histogram', 'epu_id']]
        csv = gee.format_lc_type1(csv, fractionalize=True, prepend='lc_')
    else:
        keepcols = ['epu_id']
        renamer = {}
        if 'mean' in datasets[key]['stats']:
            keepcols.append('mean')
            renamer.update({'mean' : key + '_mean'}) 
        if 'std' in datasets[key]['stats'] or 'stdDev' in datasets[key]['stats']:
            keepcols.append('stdDev')
            renamer.update({'stdDev' : key + '_std'}) 
        csv = csv[keepcols]
        csv = csv.rename({'mean': key + '_mean'}, axis=1)
        
    epus = pd.merge(epus, csv, left_on='epu_id', right_on='epu_id')
        
epus.to_file(paths['epu_gpkg'], driver='GPKG')

# Export watershed/gage information - keep out of class since this is somewhat
# external...for now
path_watersheds = r"X:\Research\CIMMID\Data\ecopop Layers\Finals\na_10k\gage_selection\trap_data_basins_s001.shp"
epus = gpd.read_file(paths['epu_gpkg'])
watersheds = gpd.read_file(path_watersheds)
df = eut.overlay_watersheds(epus, watersheds, check_coverage=False)
df.rename({'area_sum':'area_epu_km2'}, axis=1, inplace=True)
df.to_csv(paths['gages'], index=False)
df.to_csv(r"X:\Research\CIMMID\Data\ecopop Layers\Finals\na_10k\na_10k_gages.csv", index=False)        

# Export epus that cover gages
epus_gages = epus[epus['epu_id'].isin(df['epu_id'])]
epus_gages = epus_gages[['epu_id', 'geometry']]
epus_gages.to_file(r'X:\Research\CIMMID\Data\ecopop Layers\Finals\na_10k\na_10k_epu_gages.shp')














