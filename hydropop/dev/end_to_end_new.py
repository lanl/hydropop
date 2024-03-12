import os
import sys
import rabpro
import pandas as pd
from osgeo import gdal
import geopandas as gpd

sys.path.append("hydropop")
import hp_class as hpc
import hp_utils as hut
import gee_stats as gee
import rivgraph_ports as wg

### """ Adjustable parameters """
## HPU creation parameters
# fmt: off
pop_breaks = [-11, -10, -4, 0, 100] # coarse = [-11, -10, -4, 0, 100], fine =  [-11, -10, -4, -1, 1, 2, 100]
hthi_breaks = [-.01, .4, .7, 1.01] # coarse = [-.01, .4, .7, 1.01], fine = [-.01, 0.3, 0.55, 0.75, 0.9, 1.01]
# fmt: on
min_hpu_size = 20 # in pixels - each HPU will have at least this many pixels
target_hpu_size = 300 # in pixels - not guaranteed, but will try to make each HPU this size

## Path parameters
path_bounding_box = r"data/roi.gpkg" # shapefile of ROI
path_results = r'results' # folder to store results
run_name = 'toronto_new_method' # string to prepend to exports
gee_asset = 'projects/cimmid/assets/toronto_coarse_hpus' # the asset path to the hydropop shapefile--this might not be known beforehand but is created upon asset loading to GEE
gdrive_folder_name = 'CIMMID_{}'.format(run_name)

## Pseduo-fixed parameters/variables """
# Paths to data
path_hthi = r"data/hydrotopo_hab_index.tif"
path_pop = r"data/pop_density_americas.tif"
path_gee_csvs = r'results/toronto_new_hpu_method/gee' 

## Here we go
paths = hut.prepare_export_paths(path_results, run_name)

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
hpugen.simplify_hpu_classes(min_class_size=min_hpu_size)

# Compute HPUs from classes image
hpugen.compute_hpus(target_hpu_size, min_hpu_size)

# Export adjacency
hpugen.compute_adjacency()
hpugen.adjacency.to_csv(paths['adjacency'], index=False)

# Export HPU rasters
hpugen.export_raster('hpu_simplified', paths['hpu_raster'])
hpugen.export_raster('hpu_class_simplified', paths['hpu_class_raster'])

# Export classes as polygons for plotting
classes = hut.polygonize_hpu(hpugen.I['hpu_class_simplified'], hpugen.gt, hpugen.wkt)
classes.to_file(paths['hpu_class_gpkg'], driver='GPKG')

# Compute areagrid required for computing HP unit areas
agrid = hut.areagrid(paths['hpu_raster'])
gdobj = gdal.Open(paths['hpu_raster'])
wg(agrid, gdobj.GetGeoTransform(), gdobj.GetProjection(), paths['areagrid'], dtype=gdal.GDT_Float32)

""" Compute statistics for HPUs """
# First, we do zonal stats on the locally-available rasters
# HPU stats and properties
do_stats = {'hthi' : [path_hthi, ['mean']],
           'pop' : [path_pop, ['mean']],
           'area' : [paths['areagrid'], ['sum'], path_hpu_raster],
           'hpu_class' :[paths['hpu_class_simplfied'], ['majority']]}
hpugen.compute_hpu_stats(do_stats)
# Export the geopackage that contains all the HPU attributes
hpugen.hpus.to_file(paths['hpu_gpkg'], driver='GPKG')
# For the shapefile export, we only need the HPU id and the polygon
hpus_shp = gpd.GeoDataFrame(hpugen.hpus[['hpu_id', 'geometry']])
hpus_shp.crs = hpugen.hpus.crs
hpus_shp.to_file(paths['hpu_shapefile']) # shapefile needed to upload to GEE

""" STOP. Here you need to upload the hpu shapefile as a GEE asset. """

""" Update the gee_asset variable. """
datasets, Datasets = gee.generate_datasets()

# check and do fmax
if 'fmax' in datasets.keys():
    filename_out = 'fmax'
    gee.export_fmax(gee_asset, filename_out, gdrive_folder_name)
 
# Spin up other datasets
urls, tasks = rabpro.basin_stats.compute(Datasets,
                           gee_feature_path=gee_asset,
                           folder=gdrive_folder_name)

""" STOP. Download the GEE exports (csvs) to path_gee_csvs """
hpus = gpd.read_file(paths['hpu_gpkg'])
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
        
hpus.to_file(paths['hpu_gpkg'], driver='GPKG')

# Export watershed/gage information - keep out of class since this is somewhat
# external...for now
path_watersheds = r"X:\Research\CIMMID\Data\Watersheds\Toronto\initial_basins.gpkg"
hpus = gpd.read_file(paths['hpu_gpkg'])
watersheds = gpd.read_file(path_watersheds)
df = hut.overlay_watersheds(hpus, watersheds)
df.to_csv(paths['gages'], index=False)

# from matplotlib import pyplot as plt
# """
# Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
# :param nlabels: Number of labels (size of colormap)
# :param type: 'bright' for strong colors, 'soft' for pastel colors
# :param first_color_black: Option to use first color as black, True or False
# :param last_color_black: Option to use last color as black, True or False
# :param verbose: Prints the number of labels and shows the colormap. True or False
# :return: colormap for matplotlib
# """
# from matplotlib.colors import LinearSegmentedColormap
# import colorsys
# import numpy as np


# if type not in ('bright', 'soft'):
#     print ('Please choose "bright" or "soft" for type')
#     return

# if verbose:
#     print('Number of labels: ' + str(nlabels))

# # Generate color map for bright colors, based on hsv
# if type == 'bright':
#     randHSVcolors = [(np.random.uniform(low=0.0, high=1),
#                       np.random.uniform(low=0.2, high=1),
#                       np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

#     # Convert HSV list to RGB
#     randRGBcolors = []
#     for HSVcolor in randHSVcolors:
#         randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

#     if first_color_black:
#         randRGBcolors[0] = [0, 0, 0]

#     if last_color_black:
#         randRGBcolors[-1] = [0, 0, 0]

#     random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

# # Generate soft pastel colors, by limiting the RGB spectrum
# if type == 'soft':
#     low = 0.6
#     high = 0.95
#     randRGBcolors = [(np.random.uniform(low=low, high=high),
#                       np.random.uniform(low=low, high=high),
#                       np.random.uniform(low=low, high=high)) for i in range(nlabels)]

#     if first_color_black:
#         randRGBcolors[0] = [0, 0, 0]

#     if last_color_black:
#         randRGBcolors[-1] = [0, 0, 0]
#     random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

# # Display colorbar
# if verbose:
#     from matplotlib import colors, colorbar
#     from matplotlib import pyplot as plt
#     fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

#     bounds = np.linspace(0, nlabels, nlabels + 1)
#     norm = colors.BoundaryNorm(bounds, nlabels)

#     cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
#                                boundaries=bounds, format='%1i', orientation=u'horizontal')

# return random_colormap
