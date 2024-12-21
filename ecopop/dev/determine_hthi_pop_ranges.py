# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:55:43 2022

@author: L318596
"""
from osgeo import gdal
import numpy as np
import sys
sys.path.append(r'X:\Research\CIMMID\ecopop\make_hpus')
import hp_class as hpc
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


# Paths to data
path_hthi = r"X:\Research\CIMMID\Data\Hydropop Layers\Hydrotopo Index\hydrotopo_hab_index.tif"
path_pop = r"X:\Research\CIMMID\Data\Hydropop Layers\pop_density_americas.tif"
path_bounding_box = r"X:\Research\CIMMID\Data\Hydropop Layers\Finals\hpu_iquitos_1\iquitos_1_roi.shp"
path_bounding_box = r"X:\Research\CIMMID\Data\Hydropop Layers\Finals\hpu_tor_1\roi.shp"


# Histogram of log population for all Americas
Ipop = gdal.Open(path_pop).ReadAsArray()
popvals = Ipop[~np.isnan(Ipop)]
popvals[popvals==0] = 1e-10
popvals[popvals<0] = 1e-10
popvals = np.log10(popvals)
plt.close(); plt.hist(popvals)
plt.xlabel('log10(pop density)')
plt.ylabel('count (N pixels)')
plt.title('Population Density for All Americas')

# Histogram of HTHI for all Americas
Ihthi = gdal.Open(path_hthi).ReadAsArray()
hthivals = Ihthi[Ihthi!=Ihthi[0][0]]
plt.close(); plt.hist(hthivals)
plt.xlabel('HTHI')
plt.ylabel('count (N pixels)')
plt.title('HTHI for All Americas')

# Possible breaks
pop_coarse = [-11, -10, -4, 0, 100]
hthi_coarse = [-.01, .4, .7, 1.01]

pop_fine = [-11, -10, -4, -1, 1, 2, 100]
hthi_fine = [-.01, 0.3, 0.55, 0.75, 0.9, 1.01]

# Instantiate hpu class - can take awhile to load images and do some preprocessing
test = hpc.hpu(path_pop, path_hthi, path_bounding_box)

# # HTHI
# plt.hist(test.I['hthi'].flatten())
# # Population
# plt.close(); plt.hist(test.I['pop'].flatten())

# Testing
breaks = {'hthi' : hthi_fine, 'pop' : pop_fine}
test.compute_hp_classes_ranges(breaks)
popl, popu = breaks['pop'][:-1], breaks['pop'][1:]
hthil, hthiu = breaks['hthi'][:-1], breaks['hthi'][1:]

# Plot classes of population
lid = 1
labels = {}
Ic_pop = np.zeros(test.I['pop'].shape, dtype=int)
for hl, hu in zip(popl, popu):
    Ic_pop[np.logical_and(test.I['pop']>=hl, test.I['pop']<hu)] = lid
    labels[lid] = '({},{})'.format(hl, hu)
    lid = lid + 1

all_poss_labels = np.arange(0, len(popl) + 1)
cmap = plt.get_cmap('bone')
actual_labels = sorted(np.unique(Ic_pop))
actual_cmap = {}
for al in actual_labels:
    actual_cmap[al] = list(cmap(int(255*al/len(all_poss_labels))))
    if al == 0:
        actual_cmap[al] = [0, 0, 0, 1] # make nodata black
actual_leg = {}
for al in actual_labels:
    if al == 0:
        actual_leg[al] = 'nodata'
    else:
        actual_leg[al] = labels[al]
patches = [mpatches.Patch(color=actual_cmap[al], label=actual_leg[al]) for al in actual_labels]

Iplot = np.array([[actual_cmap[i] for i in j] for j in Ic_pop])    
plt.close();
plt.figure(figsize=(10,4))
plt.imshow(Iplot, interpolation='none')
plt.legend(handles=patches, loc=0, borderaxespad=0., title='log10(pop) range')


# Plot classes of HTHI
lid = 1
labels = {}
Ic_hthi = np.zeros(test.I['hthi'].shape, dtype=int)
for l, u in zip(hthil, hthiu):
    Ic_hthi[np.logical_and(test.I['hthi']>=l, test.I['hthi']<u)] = lid
    labels[lid] = '({},{})'.format(l, u)
    lid = lid + 1
        
all_poss_labels = np.arange(0, len(hthil) + 1)
cmap = plt.get_cmap('summer')
actual_labels = sorted(np.unique(Ic_hthi))
actual_cmap = {}
for al in actual_labels:
    actual_cmap[al] = list(cmap(int(cmap.N*al/len(all_poss_labels))))
    if al == 0:
        actual_cmap[al] = [0, 0, 0, 1] # make nodata black
actual_leg = {}
for al in actual_labels:
    if al == 0:
        actual_leg[al] = 'nodata'
    else:
        actual_leg[al] = labels[al]
patches = [mpatches.Patch(color=actual_cmap[al], label=actual_leg[al]) for al in actual_labels]

Iplot = np.array([[actual_cmap[i] for i in j] for j in Ic_hthi])    
plt.close();
plt.figure(figsize=(10,4))
plt.imshow(Iplot, interpolation='none')
plt.legend(handles=patches, loc=0, borderaxespad=0., title='HTHI range')

# Plot ecopop units
lid = 1
labels = {}
Ic_hpu = np.zeros(test.I['hthi'].shape, dtype=int)
for pl, pu in zip(popl, popu):
    for hl, hu in zip(hthil, hthiu):
        labels[lid] = '({}, {}) | ({}, {})'.format(pl, pu, hl, hu)
        Ic_hpu[np.logical_and(np.logical_and(test.I['hthi']>=hl, test.I['hthi']<hu),np.logical_and(test.I['pop']>=pl, test.I['pop']<pu))] = lid
        lid = lid + 1
Ic_hpu[test.I['mask']==0] = 0

all_poss_labels = np.arange(0, (len(hthil)*len(popl)) + 1)
cmap = plt.get_cmap('Paired')
actual_labels = sorted(np.unique(Ic_hpu))
actual_cmap = {}
for al in actual_labels:
    actual_cmap[al] = list(cmap(int(cmap.N*al/len(all_poss_labels))))
    if al == 0:
        actual_cmap[al] = [0, 0, 0, 1] # make nodata black
actual_leg = {}
for al in actual_labels:
    if al == 0:
        actual_leg[al] = 'nodata'
    else:
        actual_leg[al] = labels[al]
patches = [mpatches.Patch(color=actual_cmap[al], label=actual_leg[al]) for al in actual_labels]

Iplot = np.array([[actual_cmap[i] for i in j] for j in Ic_hpu])    
plt.close();
plt.figure(figsize=(10,4))
plt.imshow(Iplot, interpolation='none')
plt.legend(handles=patches, borderaxespad=0., bbox_to_anchor=(1.05, 1), loc=2, title='pop | HTHI ranges')

