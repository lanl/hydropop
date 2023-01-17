# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:19:34 2021

@author: Jon
"""
import os
import sys
sys.path.append(r'X:\Research\CIMMID\hydropop\make_hpus')
import hp_class as hpc
import hp_utils as hut
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt
from skimage.future.graph import RAG
import pandas as pd
import networkx as nx    
from rivgraph.io_utils import write_geotiff as wg

# Export path
path_export = r'X:\Research\CIMMID\Data\Hydropop Layers\Finals\north_america'
# Paths to data
path_hthi = r"X:\Research\CIMMID\Data\Hydropop Layers\Hydrotopo Index\hydrotopo_hab_index.tif"
path_pop = r"X:\Research\CIMMID\Data\Hydropop Layers\pop_density_americas.tif"
# path_bounding_box = r"X:\Research\CIMMID\Data\Hydropop Layers\Finals\hpu_tor_1\roi.shp"
path_bounding_box = r"X:\Research\CIMMID\Data\Hydropop Layers\Finals\hpu_iquitos_1\iquitos_1_roi.shp"
path_bounding_box = r"X:\Research\CIMMID\Data\Hydropop Layers\Finals\north_america\na_roi.gpkg"

# Instantiate hpu class - can take awhile to load images and do some preprocessing
nsam = hpc.hpu(path_pop, path_hthi, path_bounding_box)

# # Smooth the input layers
# nsam.I['hthi'] = hut.smooth_layer(nsam.I['hthi'], 2)
# nsam.I['pop'] = hut.smooth_layer(nsam.I['pop'], 2)

# Generate the HP classes using pre-defined break ranges.
# Possible breaks
pop_coarse = [-11, -10, -4, 0, 100]
hthi_coarse = [-.01, .4, .7, 1.01]

pop_fine = [-11, -10, -4, -1, 1, 2, 100]
hthi_fine = [-.01, 0.3, 0.55, 0.75, 0.9, 1.01]

breaks = {'hthi':hthi_fine, 'pop':pop_fine}
nsam.compute_hp_classes_ranges(breaks)

# # Generate the HP classes. Uses k-means clustering--you must specify n_groups
# # and more n_groups takes more time. Can take minutes. 
# ngroups = 10
# nsam.compute_hp_classes_kmeans(ngroups)


# Do some HP simplification to remove tiny HP clusters
# Then assign each cluster a unique id
min_hpu_size = 4 # pixels - each HPU will have at least this many pixels
nsam.simplify_hpu_classes(minpatch=min_hpu_size, unique_neighbor=False, maxiter=10)

# Assign each cluster a unique id
nsam.compute_hpus()

# Export HPU rasters in order to compute zonal stats
# Note that zonal_stats can take ndarrays as well, but require an Affine transform
# This can speed zs when raster is already in memory (as is hpu case)
path_hpu_raster = os.path.join(path_export, 'hpus.tif')
path_hpu_class_raster = os.path.join(path_export, 'hpu_classes.tif')
nsam.export_raster('hpu', path_hpu_raster)
nsam.export_raster('hpu_class', path_hpu_class_raster)

# Compute areagrid for sampling HP unit areas
path_areagrid = os.path.join(path_export, 'areagrid.tif')
agrid = hut.areagrid(path_hpu_raster)
gdobj = gdal.Open(path_hpu_raster)
wg(agrid, gdobj.GetGeoTransform(), gdobj.GetProjection(), path_areagrid, dtype=gdal.GDT_Float32)

# HPU stats and properties
do_stats = {'hthi' : [path_hthi, ['mean']],
           'pop' : [path_pop, ['mean']],
           'area' : [path_areagrid, ['sum'], path_hpu_raster],
           'hpu_class' :[path_hpu_class_raster, ['majority']]}
nsam.compute_hpu_stats(do_stats)
path_hpus = os.path.join(path_export, 'hpus.shp')
nsam.hpus.to_file(path_hpus)

# Adjacency
path_adj = os.path.join(path_export, 'adjacency.csv')
path_adjmap = os.path.join(path_export, 'adjacency_map.csv')

# Export adjacency matrix as csv for now
rag = RAG(nsam.I['hpu'], connectivity=2)
adj_dict = {i: list(rag.neighbors(i)) for i in list(rag.nodes)}
hpu_ids = adj_dict.keys()
adj_vals = []
for hid in hpu_ids:
    adj_vals.append(','.join([str(v) for v in adj_dict[hid]]))
    
adj_df = pd.DataFrame({'hpu_id':hpu_ids, 'adjacency':adj_vals})
adj_df.to_csv(path_adj, index=False)
# adj_df = pd.DataFrame(nx.convert_matrix.to_numpy_array(rag))
# adj_map = pd.DataFrame(data={'row_in_matrix':np.arange(0, len(rag.nodes)), 'hpu_id':list(rag.nodes)})
# adj_df.to_csv(path_adj, index=False, header=False)
# adj_map.to_csv(path_adjmap, index=False)




# # 1. Smooth the layers before k-means
# # 2. Turn off neighbors

# import numpy as np
# import seaborn as sns
# plt.close('all')

# pdf = nsam.stats.copy()
# pdf = pdf[['hpu', 'hthi_mean', 'pop_mean']]
# sns.jointplot(x='hthi_mean', y='pop_mean', data=pdf, hue='hpu', kind='hist')

# plt.close('all')
# plt.figure()
# for h in np.unique(nsam.stats.hpu.values):
#     pdf = nsam.stats[nsam.stats.hpu.values==h]
#     print(h, np.mean(pdf['hthi_mean'].values), np.nanmean(pdf['pop_mean'].values))
#     plt.plot(np.mean(pdf['hthi_mean'].values), np.nanmean(pdf['pop_mean'].values), 'o')
    
# # Plotting
# n_pts = 100000
# sidcs = np.array(np.linspace(0, len(nsam.hi_vals)-1, n_pts), dtype=int)

# plt.close()
# xvals = np.log10(nsam.pop_vals[sidcs])
# yvals = nsam.hi_vals[sidcs]

# yvals = yvals[~np.isinf(xvals)]
# xvals = xvals[~np.isinf(xvals)]

# xlims = (-10, 3)
# g = sns.JointGrid(x=xvals, y=yvals, xlim=xlims)
# g = g.plot_joint(plt.kdeplot, cmap='Blues')
# _ = g.ax_marg_x.hist(xvals, color="b", alpha=.6,
#                       bins=np.linspace(xlims[0], xlims[1], 10))
# _ = g.ax_marg_y.hist(yvals, color="r", alpha=.6,
#                       orientation="horizontal",
#                       bins=np.arange(0, 1, 10))


# cmap = plt.cm.jet  # define the colormap
# # extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(cmap.N)]
# # create the new map
# cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)
# plt.close()
# plt.imshow(Itest, cmap=cmap)



