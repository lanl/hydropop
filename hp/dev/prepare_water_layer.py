# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:21:07 2022

@author: L318596
"""

from osgeo import gdal
from rivgraph import im_utils as iu
import numpy as np
from skimage import morphology, measure, util

# Read the watermask image
path = r"X:\Research\CIMMID\Data\Hydropop Layers\Watermask\gsw_1_4_occ_90_byte_americas_only.tif"
gdobj = gdal.Open(path)
I = gdobj.ReadAsArray()

# Get the area of each blob
getprops = ['coords', 'area']
props, Ilab = iu.regionprops(I, getprops)