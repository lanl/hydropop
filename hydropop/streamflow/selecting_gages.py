# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:45:03 2022

@author: 318596
"""
import geopandas as gpd
from VotE import config
config.vote_db()
from VotE.streamflow import export_streamflow as es


path_bounding_box = r"X:\Research\CIMMID\Data\Hydropop Layers\Finals\toronto_coarse\roi.gpkg" # shapefile of ROI
bb = gpd.read_file(path_bounding_box)
gage_params = {'within' : bb.geometry.values[0],
               'max_drainarea_km2' : 10000,
               'fraction_valid' : .9,
               'vote_addressed' : True,
               'end_date': '2000-01-01',
               'min_span_yrs' : 10
              }
gage_ids = es.gage_selector(gage_params)
gages = es.get_gages(gage_ids)

# Export the gages and their watersheds (two files)
keepkeys = [k for k in gages.keys() if 'geom' not in k]
keepkeys = [k for k in keepkeys if 'chunk' not in k]
keepkeys.remove('id_duplicates')
gage_locs = gpd.GeoDataFrame(data=gages[keepkeys], geometry=gages['mapped_geom'], crs=gages.crs)
basins = gpd.GeoDataFrame(data=gages[keepkeys], geometry=gages['basin_geom_vote'], crs=gages.crs)

gage_locs['start_date'] = gage_locs['start_date'].astype(str)
gage_locs['end_date'] = gage_locs['start_date'].astype(str)
basins['start_date'] = basins['start_date'].astype(str)
basins['end_date'] = basins['start_date'].astype(str)

gage_locs.to_file(r'X:\Research\CIMMID\Data\Watersheds\Toronto\initial_gages.gpkg', driver='GPKG')
basins.to_file(r'X:\Research\CIMMID\Data\Watersheds\Toronto\initial_basins.gpkg', driver='GPKG')