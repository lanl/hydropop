# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:13:47 2021

@author: Jon

Script for preparing Geotiffs needed for HP creation.
"""
import sys
sys.path.append(r'C:\Users\Jon\Desktop\Research\DR Reserve\Code')
sys.path.append(r'C:\Users\Jon\Desktop\Research\CIMMID\hydropop\make_hpus')
import hp_utils as hut
from osgeo import gdal
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.vq import kmeans2

# Clip worldpop downloaded from GEE
path_wpop = r"C:\Users\Jon\Desktop\wpop_2020_1k-0000000000-0000000000.tif"
path_hthi = r"C:\Users\Jon\Desktop\Research\CIMMID\Data\Hydropop Layers\Hydrotopo Index\hydrotopo_hab_index.tif"
path_pop_americas = r"C:\Users\Jon\Desktop\Research\CIMMID\Data\Hydropop Layers\pop_density_americas.tif"
# hut.fit_geotiff_into_another(path_hthi, path_wpop, path_pop_americas, dtype='Float32', matchres=True, src_nodata=None, dst_nodata=None, resampling='bilinear')

path_hthi = r"C:\Users\Jon\Desktop\Research\CIMMID\Data\Hydropop Layers\Hydrotopo Index\hydrotopo_hab_index.tif"
path_pop = r"C:\Users\Jon\Desktop\Research\DR Reserve\Data\MHI Layers\pop_density_1k.tif"


Ihi = gdal.Open(path_hthi).ReadAsArray()
Ipop = gdal.Open(path_pop_americas).ReadAsArray()

Imask = Ihi == Ihi[0][0] # What not to include
Ipop[np.isnan(Ipop)] = 0 # Set nans to 0 in population

idcs = np.where(~Imask)
hi_vals = Ihi[idcs[0], idcs[1]]
pop_vals = Ipop[idcs[0], idcs[1]]


# Plotting - must do some sampling as there are too many points

# Downsample for plotting if wanted
n_pts = 100000
sidcs = np.array(np.linspace(0, len(hi_vals)-1, n_pts), dtype=int)

plt.close()
xvals = np.log10(pop_vals[sidcs])
yvals = hi_vals[sidcs]

yvals = yvals[~np.isinf(xvals)]
xvals = xvals[~np.isinf(xvals)]

xlims = (-10, 3)
g = sns.JointGrid(x=xvals, y=yvals, xlim=xlims)
g = g.plot_joint(plt.kdeplot, cmap='Blues')
_ = g.ax_marg_x.hist(xvals, color="b", alpha=.6,
                      bins=np.linspace(xlims[0], xlims[1], 10))
_ = g.ax_marg_y.hist(yvals, color="r", alpha=.6,
                      orientation="horizontal",
                      bins=np.arange(0, 1, 10))



# plt.close()
# ax = sns.kdeplot(xvals, yvals, cmap = 'Reds')

ngroups = 10
coords = list(zip(hi_vals, pop_vals))
centroid, label = kmeans2(coords, ngroups, minit='points')

Itest = np.zeros(Ihi.shape)
Itest[idcs[0], idcs[1]] = label



