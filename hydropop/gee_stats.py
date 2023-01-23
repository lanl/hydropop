# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:44:11 2022

@author: 318596
"""
import os
os.environ['RABPRO_DATA'] = r'X:\Data'
import pandas as pd
import rabpro
from rabpro.basin_stats import Dataset
from rabpro import utils as ru
from rabpro import data_utils as du
import geopandas as gpd
import ee
ee.Initialize()


def export_fmax(asset, filename_out, gdrive_folder_name):
    """
    A GEE function to compute fmax, which is the fraction of area for a 
    region/polygon above the mean topographic wetness index (or cti).
    """
    
    def fracAreaAboveMean(feature):
      meanval = twi.reduceRegion(
        reducer = ee.Reducer.mean(),
        geometry = feature.geometry(),
        scale = 90,
        maxPixels = 10e18
      ).getNumber('b1');          
      
      I = ee.Image.pixelArea().updateMask(twi.lt(meanval))
      
      area_above = I.reduceRegion(
        reducer = ee.Reducer.sum(),
        geometry = feature.geometry(),
        scale = 90,
        maxPixels = 10e18,
      ).getNumber('area')
      
      # Handle nulls in meanval - return them as None. This happens when the
      # feature's geometry does not cover any of the underlying TWI image.
      # In general, these geometries are useless but some might slip through.
      toreturn = ee.Algorithms.If(ee.Algorithms.IsEqual(None, meanval),
                  None,
                  area_above.divide(feature.geometry().area()).multiply(100))

      return feature.set({'fmax': toreturn})
    
    polys = ee.FeatureCollection(asset)
    twi = ee.ImageCollection("projects/sat-io/open-datasets/Geomorpho90m/cti").mosaic()
    fmax = polys.map(fracAreaAboveMean)
    
    task = ee.batch.Export.table.toDrive(
      collection = fmax.select([".*"], None, False),
      description = filename_out,
      fileFormat = 'CSV',
      folder = gdrive_folder_name,
    )
    
    task.start()


def generate_datasets():
    """
    Returns a list of rabpro.Dataset objects that can be passed to 
    rabpro.subbasin_stats.compute(). These datasets will be sampled over
    each HPU, and this particluar set was chosen in order to provide 
    HPU-specific parameterizations for the E3SM's Land Model.
    """
    # Create rabpro Dataset list -- uses a dictionary so parsing the results is easier
    dataset_dict = {
        'fmax' : 'custom',
        'elevation' : {'path': 'MERIT/DEM/v1_0_3',
                       'band': 'dem',
                       'stats': ['mean', 'std']},
        'soil_depth' : {'path': 'projects/rabpro-datasets/assets/pelletier_average_soil_and_sedimentary_deposit_thickness',
                        'band': 'b1',
                        'stats': ['mean']},
        'topo_slope' : {'path': 'projects/sat-io/open-datasets/Geomorpho90m/slope',
                        'band': 'None',
                        'stats': ['mean'],
                        'mosaic': True},
        'soc_0-5' : {'path': 'projects/soilgrids-isric/soc_mean',
                      'band': 'soc_0-5cm_mean',
                      'stats' : ['mean']},
        'soc_5-15' : {'path': 'projects/soilgrids-isric/soc_mean',
                      'band': 'soc_5-15cm_mean',
                      'stats' : ['mean']},
        'soc_15-30' : {'path': 'projects/soilgrids-isric/soc_mean',
                      'band': 'soc_15-30cm_mean',
                      'stats' : ['mean']},
        'soc_30-60' : {'path': 'projects/soilgrids-isric/soc_mean',
                      'band': 'soc_30-60cm_mean',
                      'stats' : ['mean']},
        'soc_60-100' : {'path': 'projects/soilgrids-isric/soc_mean',
                      'band': 'soc_60-100cm_mean',
                      'stats' : ['mean']},
        'soc_100-200' : {'path': 'projects/soilgrids-isric/soc_mean',
                      'band': 'soc_100-200cm_mean',
                      'stats' : ['mean']},
        'clay_0-5' : {'path': 'projects/soilgrids-isric/clay_mean',
                      'band': 'clay_0-5cm_mean',
                      'stats' : ['mean']},
        'clay_5-15' : {'path': 'projects/soilgrids-isric/clay_mean',
                      'band': 'clay_5-15cm_mean',
                      'stats' : ['mean']},
        'clay_15-30' : {'path': 'projects/soilgrids-isric/clay_mean',
                      'band': 'clay_15-30cm_mean',
                      'stats' : ['mean']},
        'clay_30-60' : {'path': 'projects/soilgrids-isric/clay_mean',
                      'band': 'clay_30-60cm_mean',
                      'stats' : ['mean']},
        'clay_60-100' : {'path': 'projects/soilgrids-isric/clay_mean',
                      'band': 'clay_60-100cm_mean',
                      'stats' : ['mean']},
        'clay_100-200' : {'path': 'projects/soilgrids-isric/clay_mean',
                      'band': 'clay_100-200cm_mean',
                      'stats' : ['mean']},
        'silt_0-5' : {'path': 'projects/soilgrids-isric/silt_mean',
                      'band': 'silt_0-5cm_mean',
                      'stats' : ['mean']},
        'silt_5-15' : {'path': 'projects/soilgrids-isric/silt_mean',
                      'band': 'silt_5-15cm_mean',
                      'stats' : ['mean']},
        'silt_15-30' : {'path': 'projects/soilgrids-isric/silt_mean',
                      'band': 'silt_15-30cm_mean',
                      'stats' : ['mean']},
        'silt_30-60' : {'path': 'projects/soilgrids-isric/silt_mean',
                      'band': 'silt_30-60cm_mean',
                      'stats' : ['mean']},
        'silt_60-100' : {'path': 'projects/soilgrids-isric/silt_mean',
                      'band': 'silt_60-100cm_mean',
                      'stats' : ['mean']},
        'silt_100-200' : {'path': 'projects/soilgrids-isric/silt_mean',
                      'band': 'silt_100-200cm_mean',
                      'stats' : ['mean']},
        'sand_0-5' : {'path': 'projects/soilgrids-isric/sand_mean',
                      'band': 'sand_0-5cm_mean',
                      'stats' : ['mean']},
        'sand_5-15' : {'path': 'projects/soilgrids-isric/sand_mean',
                      'band': 'sand_5-15cm_mean',
                      'stats' : ['mean']},
        'sand_15-30' : {'path': 'projects/soilgrids-isric/sand_mean',
                      'band': 'sand_15-30cm_mean',
                      'stats' : ['mean']},
        'sand_30-60' : {'path': 'projects/soilgrids-isric/sand_mean',
                      'band': 'sand_30-60cm_mean',
                      'stats' : ['mean']},
        'sand_60-100' : {'path': 'projects/soilgrids-isric/sand_mean',
                      'band': 'sand_60-100cm_mean',
                      'stats' : ['mean']},
        'sand_100-200' : {'path': 'projects/soilgrids-isric/sand_mean',
                      'band': 'sand_100-200cm_mean',
                      'stats' : ['mean']},
        'land_use' : {'path' : 'MODIS/006/MCD12Q1',
                      'band' : 'LC_Type1',
                      'stats' : ['freqhist'],
                      'start': '2015-01-01',
                      'end' : '2015-12-31'},
        }

    dataset_list = []
    for key in dataset_dict:
        this_ds = dataset_dict[key]
        mosaic, start, end = False, None, None
        if this_ds == 'custom':
            continue
        if 'mosaic' in this_ds.keys() and this_ds['mosaic'] is True:
            mosaic = True
        if 'start' in this_ds.keys():
            start = this_ds['start']
        if 'end' in this_ds.keys():
            end = this_ds['end']
            
        dataset_list.append(Dataset(this_ds['path'], this_ds['band'], stats=this_ds['stats'], start=start, end=end, mosaic=mosaic))

    return dataset_dict, dataset_list


""" Function for parsing the MODIS Land Use data, see https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD12Q1 """
def format_lc_type1(csv, fractionalize=True, prepend=''):
    """
    Takes a csv returned by GEE's histogram function and expands
    them into a DataFrame. The input csv should have only two columns:
        'histogram' and 'hpu_id'.
    
    This particular function considers the MODIS MCD12Q1
    dataset, Type 1 classes.
    
    If fractionalize is True, the sum of each row will be one (i.e.
    provides fractions rather than absolute areas or N pixels)
    """
    LC_Type1 = {1: 'Evergreen Needleleaf Forests',
                2: 'Evergreen Broadleaf Forests', 
                3: 'Deciduous Needleleaf Forests',
                4: 'Deciduous Broadleaf Forests',
                5: 'Mixed Forests',
                6: 'Closed Shrublands',
                7: 'Open Shrublands',
                8: 'Woody Savannas',
                9: 'Savannas',
                10: 'Grasslands',
                11: 'Permanent Wetlands',
                12: 'Croplands',
                13: 'Urban and Built-up Lands',
                14: 'Cropland/Natural Vegetation mosaics',
                15: 'Permanent Snow and Ice',
                16: 'Barren',
                17: 'Water Bodies'}

    columns = [prepend + v for v in LC_Type1.values()] + ['hpu_id']

    dfs = []
    for _, row in csv.iterrows():
        dstr = row['histogram'].strip('{').strip('}').split(', ')
        if fractionalize is True:
            total = sum([float(ds.split('=')[1]) for ds in dstr])
        else:
            total = 1
        this_df = pd.DataFrame(index=[0], columns=columns, dtype=float)
        this_df.loc[:,:] = 0
        for ds in dstr:
            this_df[prepend + LC_Type1[int(ds.split('=')[0])]] = float(ds.split('=')[1])/total
            this_df['hpu_id'] = row['hpu_id']
        dfs.append(this_df)
    
    dfhc = pd.concat(dfs) 
    
    return dfhc
