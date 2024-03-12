import os
import sys
import json
import scipy
import platform
import subprocess
import numpy as np
import pandas as pd
from pyproj import CRS
import geopandas as gpd
#
from osgeo import gdal, ogr
from math import floor, ceil
from rabpro import utils as ru
from rasterstats import zonal_stats
#
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.geometry import MultiPolygon, shape
#
sys.path.append("hydropop")
import rivgraph_ports as iu

def hp_paths(basepath, basename):
    
    paths = {
        'dem' : os.path.join(basepath, basename + '_meritDEM.tif'),
        'hand' : os.path.join(basepath, basename + '_hand.tif'),
        'occurrence' : os.path.join(basepath, basename + '_occurrence.tif'),
        'occurrence_native' : os.path.join(basepath, basename + '_occurrence_native.tif'),
        'watermask' : os.path.join(basepath, basename + '_watermask.tif'),
        'watermask_native' : os.path.join(basepath, basename + '_watermask_native.tif'),
        'watdist' : os.path.join(basepath, basename + '_watdist.tif'),
        'watdist_native' : os.path.join(basepath, basename + '_watdist_native.tif'),
        'slope' : os.path.join(basepath, basename + '_slope.tif'),
        'tri' : os.path.join(basepath, basename + '_tri.tif'),
        'twi' : os.path.join(basepath, basename + '_twi.tif'),
        'pop' : os.path.join(basepath, basename + '_wpop.tif'),
        'ndvi' : os.path.join(basepath, basename + '_ndvi.tif'),
        'gdp' : os.path.join(basepath, basename + '_gdp.tif'),
        'hch' : os.path.join(basepath, basename + '_hch.tif'),
        'hpus' : os.path.join(basepath, basename + '_hpus.tif'),
        'hpu_polys' : os.path.join(basepath, basename + '_hpu_polys.shp'),
        'mhi' : os.path.join(basepath, basename + '_mhi.tif'),
        'area' : os.path.join(basepath, basename + '_area.tif'),
        'refgrid' : os.path.join(basepath, basename + '_refgrid.tif'),
        'base' : basepath
        }
    
    return paths


def prepare_export_paths(path_results, run_name):
    
    paths = {
        'hpu_class_raster' : os.path.join(path_results, '{}_hpu_classes.tif'.format(run_name)),
        'hpu_class_gpkg' : os.path.join(path_results, '{}_hpu_classes.gpkg'.format(run_name)),
        'hpu_raster' : os.path.join(path_results, '{}_hpus.tif'.format(run_name)),
        'hpu_gpkg' : os.path.join(path_results, '{}_hpus.gpkg'.format(run_name)),
        'hpu_all' : os.path.join(path_results, '{}_hpus_all.gpkg'.format(run_name)), # includes all hpus
        'hpu_shapefile' : os.path.join(path_results, '{}_hpus.shp'.format(run_name)), # does not include masked-out hpus (waterbodies etc.)
        'adjacency' : os.path.join(path_results, '{}_adjacency.csv'.format(run_name)),
        'areagrid' : os.path.join(path_results, '{}_areagrid.tif'.format(run_name)),
        'gages' :  os.path.join(path_results, '{}_gages.csv'.format(run_name))
        }
    
    return paths        



def load_layers(layer_names, paths):
    """
    Loads all the specified layers into a dictionary
    """
    layers = {}
    for l in layer_names:
        gdobj = gdal.Open(paths[l])
        if gdobj is None:
            print('Could not find {} layer at {}.'.format(l, paths[l]))
        else:
            layers[l] = gdal.Open(paths[l]).ReadAsArray()
        
    return layers


def threshold_pop(Ipop, threshold=2.5):
    
    Ipop[Ipop<threshold] = 0
    
    return Ipop


def smooth_layer(layer, sigma):
    """
    Replaces the astropy smoothing method with a much more efficient one. 
    Smooths a layer containing nan values by considering only weights from
    non-nan values. sigma is the size of the Gaussian smoothing kernel. 
    
    https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python/36307291
    """
    
    nans = np.isnan(layer)
    
    V = layer.copy()
    V[nans] = 0
    VS = scipy.ndimage.gaussian_filter(V, sigma=sigma)
    
    W = 0 * layer.copy() + 1
    W[nans] = 0
    WS = scipy.ndimage.gaussian_filter(W, sigma=sigma)
    
    layer_smoothed = VS/WS
    
    return layer_smoothed


def nan_waterbodies(layers, paths):
    """
    Sets all persistent waterbodies to np.nan in all layers. Layers may also
    be an image.
    """
    Iw = gdal.Open(paths['watdist']).ReadAsArray()
    Inan = Iw == 0
        
    if type(layers) is dict:
        for k in layers.keys():
            I = layers[k].astype(np.float)
            I[Inan] = np.nan
            layers[k] = I
    else: # Treat as an image
        layers[Inan] = np.nan
        
    return layers


def layer_means(layers):
    """
    Gets the mean of each layer. Population must be treated separately because
    we want the non-zero mean.
    """
    means = {}
    ls = layers.keys()
    for l in ls:
        if l in ['pop', 'hch']:
            layervals = layers[l].flatten()
            layervals = layervals[layervals != 0]
            mval = np.nanmean(layervals)
        else:
            mval = np.nanmean(layers[l])
            
        means[l] = mval
        
    return means


def call_gdal(callstr):
    """
    Executes a command-line gdal string with subprocess. 
    """
    proc = subprocess.Popen(callstr, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=proc.communicate()
    
    # Don't return warnings/errors related to colortables
    if stderr != b'' and 'color table' not in str(stderr): 
        print(stderr) # A warning about colortable is OK
        
    return stdout


def fit_geotiff_into_another(ref, tofit, outpath, dtype='Byte', matchres=True, src_nodata=None, dst_nodata=None, resampling='bilinear'):
    """
    Clips a geotiff (tofit) by a reference geotiff (ref), then matches the 
    extents of the clipped to that of the reference.
    """    

    tempshp = tofit.split('.')[0] + 'temp.shp'
    callstr = ['gdaltindex',
               tempshp,
               ref]
    proc = subprocess.Popen(callstr, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=proc.communicate()
    
    # Get extents of reference image
    gdobj = gdal.Open(ref)
    gt = gdobj.GetGeoTransform()
    xmin = gt[0]
    xmax = gt[0] + gt[1] * gdobj.RasterXSize
    ymax = gt[3]
    ymin = gt[3] + gt[5] * gdobj.RasterYSize
        
    callstr = ['gdalwarp',
               '-cutline', tempshp,
               '-crop_to_cutline',
               '-te', str(xmin), str(xmax), str(ymin), str(ymax),
               '-overwrite',
               '-ot', dtype,
               '-co', 'COMPRESS=LZW',
               '-r', resampling,
               '-tr', str(gt[1]), str(abs(gt[5])), 
               '-multi',
               '-wm', str(500),
               tofit,
               outpath]
    
    if src_nodata is not None:
        callstr.insert(4, '-srcnodata')
        callstr.insert(5, str(src_nodata))

    if dst_nodata is not None:
        callstr.insert(4, '-dstnodata')
        callstr.insert(5, str(dst_nodata))
    
    if matchres is not True:
        idx = callstr.index('-tr')
        callstr_nor = callstr[0:idx]
        callstr_nor.extend(callstr[idx+3:])
        callstr = callstr_nor
        
    call_gdal(callstr)
    
    # Delete the temporary shapefile
    basetemp = tempshp.split('.')[0]
    for ext in ['shp', 'dbf', 'prj', 'shx']:
        file = basetemp + '.' + ext
        if os.path.isfile(file):
            os.remove(file)
            
    gdobj = None
            
    
def add_raster_stats(path_raster):
    """
    Adds raster statistics to a raster's metadata using GDAL. Can take awhile
    for large rasters as the statistics are not approximated, but computed on
    all the available values. Only needs to be run once for a given raster
    or virtual raster, as the stats are stored in the raster's metadata.

    Parameters
    ----------
    path_raster : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    callstr = ['gdalinfo',
            '-nomd',
            '-noct',
            '-stats',
            path_raster]
    
    _ = call_gdal(callstr)


def get_raster_stats(path_raster):
    """
    Retrieves raster statistics from metadata of a raster. If none are available,
    they will automatically be computed. Currently designed for a single-band
    raster.

    Parameters
    ----------
    path_raster : str
        Path to the raster to fetch statistics.

    Returns
    -------
    minval : float
        Minimum value of the raster.
    maxval : float
        Maximum value of the raster.
    meanval : float
        Mean value of the raster.
    stdval : float
        Standard deviation of the raster.

    """
    callstr = ['gdalinfo',
               path_raster]
        
    output = call_gdal(callstr)
    outputstr = output.decode("utf-8").split('\n')
    bandstats = [s for s in outputstr if ' Minimum=' in s]
    
    if len(bandstats) == 0:
        add_raster_stats(path_raster)
    
    minval_str = bandstats[0].split('Minimum=')[-1].split(',')[0]
    maxval_str = bandstats[0].split('Maximum=')[-1].split(',')[0]
    meanval_str = bandstats[0].split('Mean=')[-1].split(',')[0]
    stdval_str = bandstats[0].split('StdDev=')[-1].split('\r')[0]
    
    if len(minval_str) > 0 and len(maxval_str) > 0 and len(meanval_str) > 0 and len(stdval_str) > 0:
        minval = float(minval_str)
        maxval = float(maxval_str)
        meanval = float(meanval_str)
        stdval = float(stdval_str)
    else:
        add_raster_stats(path_raster)
        minval, maxval, meanval, stdval = get_raster_stats(path_raster)
    
    return minval, maxval, meanval, stdval

# paths_global = global_raster_datapaths()
# # add_raster_stats(paths_global['worldpop'])
# # minv, maxv, meanv, stdv = get_raster_stats(paths_global['worldpop'])
# minv, maxv, meanv, stdv = get_raster_stats(paths_global['gdp'])


def normalize_layers(layers, layerlist):
    """
    Normalizes layers appropriately between 0 and 1, where 0 and 1 correspond
    to the layer's contribution to the MHI. E.g. for HAND, higher values 
    corresponds to lower MHI, so this layer will be inverted when normalizing.
    
    Returns a dictionary of normalized layers.
    gdp - normalized on 0,1 with higher values corresponding to lower gdp
    pop - normalized on 0,1 with higher values corresponding to higher gdp
    
    In order to have HPUs be consistent across all spatial domains, normalization
    parameters are hard-coded based on physical reasoning or global statistics
    of the layer.
    """
    # Dictionary of hard-coded normalization parameters. (min, max) tuples.
    # Note that gdp and pop values are for log-transformed, and pop has been 
    # thresholded at 2.5.
    norm_params = {
            'pop' : (-3, 3), # higher pop -> higher HCH; these are log bases, not raw values
            'gdp' : (-2, 9), # higher gdp ->, lower GDP; these are log bases, not raw values
            'twi' : (5, 15), # higher twi -> higher MHI
            'watdist' : 5000, # higher watdist -> lower MHI, distance in meters at which normalized value should decay to 0
            'hand' : 10 # higher hand -> lower mhi
            }
    
    normlayers = {}
    for l in layerlist:
        if l not in norm_params.keys():
            print('Cannot normalize layer: {}'.format(l))
            continue
        
        # Normalize between 0 and 1
        layer = layers[l].copy()
         
        # Store the zeros and nans           
        lzeros = layer == 0
        lnans = np.isnan(layer)

        # Exponential decay normalization
        if l in ['watdist', 'hand']:
            approx_0 = 0.01
            lam = np.log(approx_0)/norm_params[l]
            layer = np.exp(lam*layer)           

        # Linear normalization
        else:
            upper = norm_params[l][1]
            lower = norm_params[l][0]
            layer[layer>upper] = upper
            layer[layer<lower] = lower
            layer = (layer - lower) / (upper - lower)
            
        # Invert layers with negative relationship to MHI/HCH
        if l in ['gdp']:
            layer = 1 - layer
       
        # Put the zeros and nans back in
        layer[lzeros] = 0
        layer[lnans] = np.nan
            
        normlayers[l] = layer
        
    return normlayers


def simplify_classes(Ilabeled, minpatchsize, nodata=0, unique_neighbor=True, maxiter=10):
    """
    Given an image whose pixels are all integer labels, this will fill any
    patches of the same label equal to or smaller than minpatchsize with either
    unique_neighbor = True : only fills a patch if its neighboring pixels all share the same label
    unique_neighbor = False : fills all patches smaller than minpatchsize with the mode of the neighboring pixel labels
    """    
    Ilab = Ilabeled.copy()
    Ind = np.isnan(Ilab)
    unique_hps = np.unique(Ilab)
    unique_hps = unique_hps[unique_hps != nodata] # -1 is a placeholder for NaNs
    no_smallies = []
    iteration = 0
    while no_smallies != set(unique_hps) and iteration < maxiter:
        iteration = iteration + 1
        print('iteration:{}'.format(iteration))
        no_smallies = set()
        for u in unique_hps:
                        
            # Make an image of just the unique value locations
            Iu = Ilab==u 
               
            # Get all the blobs of the unique value
            props = ['coords','area']
            rp, _ = iu.regionprops(Iu, props, connectivity=2)
                
            # Only investigate blobs that are below the threshold
            smallblobs = [i for i, x in enumerate(rp['area']) if x <= minpatchsize]
            
            if len(smallblobs) == 0:
                no_smallies.update([u])
                
            # For each small enough blob, check if its neighbors are all the same class
            done_sbs = set()
            for sb in smallblobs:
                coords = rp['coords'][sb]
                neighborvals = [iu.neighbor_vals(Ilab, c[1], c[0]) for c in coords]
                neighborvals = np.ndarray.flatten(np.array(neighborvals))
                
                # Exclude invalid neighbor values
                neighborvals = neighborvals[neighborvals != u] # Can't be the same as u
                neighborvals = neighborvals[np.isnan(neighborvals)== False] # Nans are at edges
                neighborvals = neighborvals[neighborvals != nodata] # -1 is nan in labeled image
                unique_neighbors = np.unique(neighborvals)   
                
                # It is possible that blobs may be enclosed by nodata (water), which
                # we don't want to use to fill the small blob. Keep track of these
                # so we can ensure that all the blobs have been corrected.
                if len(unique_neighbors) == 0:
                    done_sbs.update([sb])
                    continue
    
                if unique_neighbor is True:
                    if len(unique_neighbors) == 1:
                        Ilab[coords[:,0], coords[:,1]] = int(unique_neighbors[0])
                    else:
                        continue
                elif unique_neighbor is False:
                    fillval = scipy.stats.mode(neighborvals)[0]
                    Ilab[coords[:,0], coords[:,1]] = int(fillval)
                    
            # If the only remaining small blobs are nodata affected, that unique
            # value is complete
            if done_sbs == set(smallblobs):
                no_smallies.update([u])
    
    Ilab[Ind] = nodata
    
    return Ilab


def simplify_hpus(Ihpu, Iclasses, target_hpu_size, min_hpu_size, nodata):
    """
    Given an image where pixel values correspond to the HPU to which the
    pixel belongs, this attempts to merge smaller HPUs with neighboring ones
    of the same class such that no HPUs' areas are smaller than min_hpu_size.
    """       
    
    # Find all the too-small HPUs
    rp, Ilabeled = iu.regionprops(Ihpu, props=['area', 'label', 'coords'])
    
    do_idcs = np.where(rp['area']<(target_hpu_size/2))[0]
    
    for i in do_idcs:
        this_label = rp['label'][i]
        this_coords = rp['coords'][i]
        this_class = Iclasses[this_coords[0][0], this_coords[0][1]]
        this_area = rp['area'][i]
        # this_hpu_id = Ihpu[rp['coords'][i][0][0], rp['coords'][i][0][1]]
                
        # Get valid neighborhood values
        neighbor_labels = [iu.neighbor_vals(Ilabeled, c[1], c[0]) for c in this_coords]
        neighbor_labels = np.array(list(set(np.ndarray.flatten(np.array(neighbor_labels)))))
        
        # Exclude invalid neighbor values
        neighbor_labels = neighbor_labels[neighbor_labels != this_label] # Can't be the same hpu
        neighbor_labels = neighbor_labels[np.isnan(neighbor_labels)== False] # Nans are at edges
        neighbor_labels = neighbor_labels[neighbor_labels != nodata] # Don't want to set valid HPU to nodata
        
        # Get the classes and areas for each neighboring HPU
        neighs = pd.DataFrame(columns=['idx', 'label', 'area', 'class'])
        neighs['label'] = neighbor_labels
        neighs['idx'] = [np.where(rp['label']==nl)[0][0] for nl in neighs['label']] 
        neighs['area'] = rp['area'][neighs['idx'].values]
        neighs['class'] = [Iclasses[rp['coords'][ni][0][0], rp['coords'][ni][0][1]] for ni in neighs['idx']]
        
        # Select which neighbor to absorb this blob into
        neighs = neighs[neighs['class']==this_class] # must be the same class
        neighs = neighs[neighs['area'] > target_hpu_size/2] # we don't want to absorb into an HPU that is itself being absorbed
        
        if len(neighs) == 0: # no possible neighbors can absorb this one - this 
        # should never be the case if classes were computed correctly except 
        # where there is nodata
            continue
        elif len(neighs) > 1: # Choose the neighbor that will result in an HPU area closest to target_hpu_size
            neighs['area_discrepancy'] = np.abs(neighs['area'] + this_area - target_hpu_size) 
            neighs.sort_values(by='area_discrepancy', ascending=True)
        
        absorb_hpu_id = Ihpu[rp['coords'][neighs['idx'].values[0]][0][0], rp['coords'][neighs['idx'].values[0]][0][1]]
        Ihpu[this_coords[:,0], this_coords[:,1]] = absorb_hpu_id
        
    return Ihpu       
    

def polygonize_hpu(I, geotransform, proj_wkt, Imask=None):
    """
    Polygonizes HPUs using in-memory process (no need to write geotiff to disk).
    The resulting GeoDataFrame has an 'hpu_id' column that represents the
    value of the pixels comprising each polygon.

    Parameters
    ----------
    I : np.array
        Raster to polygonize.
    geotransform : tuple
        6-element GDAL GeoTransform
    proj_wkt : str
        Well-known-text representation of the CRS.
    Imask : np.array, optional
        Binary array where 1s are valid. Must be same shape as I.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Polygons of the rasterized image I.
    """
    # def make_valid(ob):        
    #     if ob.is_valid:
    #         return ob
    #     return geom_factory(lgeos.GEOSMakeValid(ob._geom))
        
    # Create in-memory GDAL data raster
    nrows, ncols = I.shape
    # Create a virtual raster
    vraster = gdal.GetDriverByName('MEM').Create(
        '', xsize=int(ncols), ysize=int(nrows), bands=1, eType=gdal.GDT_UInt32
    )
    vraster.SetGeoTransform(geotransform)
    vraster.SetProjection(proj_wkt)
    vraster.GetRasterBand(1).WriteArray(I)
    vband = vraster.GetRasterBand(1)
    vraster.FlushCache()
    
    # Create in-memory GDAL mask raster
    if Imask is not None:
        assert I.shape == Imask.shape
        Imask = np.array(Imask, dtype=bool)
        vmraster = gdal.GetDriverByName('MEM').Create(
            '', xsize=int(ncols), ysize=int(nrows), bands=1, eType=gdal.GDT_Byte)
        vmraster.SetGeoTransform(geotransform)
        vmraster.SetProjection(proj_wkt)
        vmraster.GetRasterBand(1).WriteArray(Imask)
        vmband = vmraster.GetRasterBand(1)
        vmraster.FlushCache()

    # Polygonize them
    driver = ogr.GetDriverByName("MEMORY")
    outDatasource = driver.CreateDataSource('')
    outLayer = outDatasource.CreateLayer("", srs=None)
    newField = ogr.FieldDefn('DN', ogr.OFTReal)
    outLayer.CreateField(newField)
    if Imask is None:
        _ = gdal.Polygonize(vband, None, outLayer, 0, ['8CONNECTED=8'], callback=None)
    else:
        _ = gdal.Polygonize(vband, vmband, outLayer, 0, ['8CONNECTED=8'], callback=None)

    pgons = []
    ids = []
    for feat in outLayer:
        f = json.loads(feat.ExportToJson())
        fg = shape(f['geometry'])
        if f['properties']['DN'] is not None:
            pgons.append(make_valid(fg))
            ids.append(int(f['properties']['DN']))

    gdf = gpd.GeoDataFrame(geometry=pgons, crs=CRS.from_epsg(4326))
    gdf['hpu_id'] = ids

    return gdf


def hpu_stats(do_stats, poly_gdf):
    """
    Computes all desired stats for each HPU. 
    which_stats: dictionary whose kyes correspond to those in paths and whose values
                 are the desired stats for each variable (look at rasterstats for
                 stat choices, but they're pretty intuitive.)
          paths: dictionary containing paths to the various rasters to be analyzed
    """  
    gdf = poly_gdf.copy()
    for i, r in enumerate(do_stats.keys()):
        rasterpath = do_stats[r][0]
        nodata = get_nodata_value(rasterpath)
        if nodata is None:
            nodata = -999
        import time
        t = time.time()
        df_for_merge = get_stats(rasterpath, poly_gdf, nodata=nodata, stats=do_stats[r][1], prefix=r)
        print(r, time.time() - t)
        gdf = pd.concat((gdf, df_for_merge), axis=1)
            
    return gdf
    

def get_stats(rastpath, poly_path, nodata=-999, stats='mean', prefix=''):
    """
    Given the path to the rasterized HPU polygons and a path to a raster 
    we want to compute statistics, this computes the stats in 'stats' and 
    returns a DataFrame containining all the stats for each HPU.
    
    Note that HPU polygons must be in the same coordinate reference system
    as the provided raster. In the case of HPUs, the rasters are all in
    EPSG:4326 and the HPUs are derived from these rasters, so they are also in 
    EPSG:4326.
    """    
    zstats = zonal_stats(poly_path, rastpath, stats=stats, nodata=nodata)  
    # Convert to DataFrame -- the ordering is important (and preserved)
    df = pd.DataFrame(zstats)
    # Must adjust column names to include the prefix as well
    new_names = {n:prefix + '_' + n for n in df.columns}
    df = df.rename(new_names, axis=1)
        
    return df
        
        
def get_nodata_value(tifpath):
    """
    Reads a geotiff's metadata to return the notdata value. This is converted
    to an int if the value is whole.
    """

    info = gdal.Info(tifpath)
    try:
        index = info.index('NoData Value')
    except:
        return None
    
    info = info[index:]  
    newline_index = info.index('\n')
    info = info[:newline_index]
    equal_index = info.index('=')
    ndv = float(info[equal_index+1:])
    
#    if ndv.is_integer() is True and ndv != -3.4*10**38:
#        ndv = int(ndv)
        
    return ndv


def areagrid(georaster_path): 
    """
    Must provide georaster in 4326 CRS
    """
    
    # Returns a matrix where each grid cell corresponds to actual area of that pixel in km^2
    
    rast_obj = gdal.Open(georaster_path)
    gt = rast_obj.GetGeoTransform()
 
    ul_lat = gt[3]    
    res_lon = gt[1]
    res_lat = gt[5]
    rows = rast_obj.RasterYSize
    cols = rast_obj.RasterXSize
    
    lr_lat = ul_lat + rows * res_lat

    # If CRS is not lat/lon, gotta transform coordinates    
    lats = np.linspace(ul_lat,lr_lat,rows+1)
       
    a = 6378137
    b = 6356752.3142
    
    # Degrees to radians
    lats = lats * np.pi/180
    
    # Intermediate vars
    e = np.sqrt(1-(b/a)**2)
    sinlats = np.sin(lats)
    zm = 1 - e * sinlats
    zp = 1 + e * sinlats
    
    # Distance between meridians
    #        q = np.diff(longs)/360
    q = res_lon/360
    
    # Compute areas for each latitude in square km
    areas_to_equator = np.pi * b**2 * ((2*np.arctanh(e*sinlats) / (2*e) + sinlats / (zp*zm))) / 10**6
    areas_between_lats = np.diff(areas_to_equator)
    areas_cells = np.abs(areas_between_lats) * q
    
    areagrid = np.transpose(np.tile(areas_cells, (cols,1)))
    
    return areagrid


def build_vrt(tilespath, clipper=None, extents=None, outputfile=None, nodataval=None, res=None, sampling='nearest', ftype='tif'):
    """
    Creates a text file for input to gdalbuildvrt, then builds vrt file with 
    same name. If output path is not specified, vrt is given the name of the 
    final folder in the path. 
    
    INPUTS: 
      tilespath - str:  the path to the file (or folder of files) to be clipped--
                        if tilespath contains an extension (e.g. .tif, .vrt), then
                        that file is used. Otherwise, a virtual raster will be 
                        built of all the files in the provided folder.
                        if filespath contains an extension (e.g. .tif, .vrt), 
                        filenames  of tiffs to be written to vrt. This list 
                        can be created by tifflist and should be in the same
                        folder
        extents - list: (optional) - the extents by which to crop the vrt. Extents
                        should be a 4 element list: [left, right, top, bottom] in
                        the ssame projection coordinates as the file(s) to be clipped
        clipper - str:  path to a georeferenced image, vrt, or shapefile that will be used
                        to clip
     outputfile - str:  path (including filename w/ext) to output the vrt. If 
                        none is provided, the vrt will be saved in the 'filespath'
                        path
            res - flt:  resolution of the output vrt (applied to both x and y directions)
       sampling - str:  resampling scheme (nearest, bilinear, cubic, cubicspline, lanczos, average, mode)
      nodataval - int:  (optional) - value to be masked as nodata
          ftype - str:  'tif' if buuilding from a list of tiffs, or 'vrt' if 
                        building from a vrt
    
    OUTPUTS:
        vrtname - str:  path+filname of the built virtual raster    
    """
    base, folder, file, ext = parse_path(tilespath)
    
    # Set output names  
    if outputfile is None:
        if clipper:
            cliptxt = '_clip'
        else:
            cliptxt = ''
        vrtname = os.path.join(base, folder, folder + cliptxt + '.vrt')
        vrttxtname = os.path.join(base, folder, folder + cliptxt + '.txt')
    else:
        vrtname = os.path.normpath(outputfile)
        vrttxtname = vrtname.replace('.vrt','.txt')
    
    # If a folder was given, make a list of all the text files
    if len(file) == 0: 
    
        filelist = []
        
        if ftype == 'tif':
            checktype = ('tif', 'tiff')
        elif ftype == 'hgt':
            checktype = ('hgt')
        elif ftype == 'vrt':
            checktype = ('vrt')
        else:
            raise TypeError('Unsupported filetype provided-must be tif, hgt, or vrt.')
      
        for f in os.listdir(tilespath):
            if f.lower().endswith(checktype): # ensure we're looking at a tif
                filelist.append(os.path.join(tilespath, f))
    else: 
        filelist = [tilespath] 
    
    if len(filelist) < 1:
        print('Supplied path for building vrt: {}'.format(filelist))
        raise RuntimeError('The path you supplied appears empty.')
                 
    # Clear out .txt and .vrt files if they already exist
    delete_file(vrttxtname)
    delete_file(vrtname)
    
    with open(vrttxtname, 'w') as tempfilelist:
        for f in filelist:
            tempfilelist.writelines('%s\n' %f)

    # Get extents of clipping raster
    if clipper:
        extents = raster_extents(clipper)

    # Build the vrt with input options
    callstring = ['gdalbuildvrt', '-overwrite',]
    
    if np.size(extents) == 4:
        stringadd = ['-te', str(extents[0]), str(extents[3]), str(extents[1]), str(extents[2])]
        for sa in stringadd:
            callstring.append(sa)
    
    if nodataval:
        stringadd = ["-srcnodata", str(nodataval)]
        for sa in stringadd:
            callstring.append(sa)
    
    if res:
        stringadd = ['-resolution', 'user', '-tr', str(res), str(res)]
        for sa in stringadd:
            callstring.append(sa)
        
    if sampling != 'nearest':
        stringadd = ['-r', sampling]
        for sa in stringadd:
            callstring.append(sa)
    
    stringadd = ['-input_file_list', vrttxtname, vrtname]
    for sa in stringadd:
        callstring.append(sa)
        
    # Make the call
    proc = subprocess.Popen(callstring, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=proc.communicate()

    # Check that vrt built successfully
    if len(stderr) > 3:
        raise RuntimeError('Virtual raster did not build sucessfully. Error: {}'.format(stderr))
    else:
        print(stdout)

    return vrtname


def get_raster_clipping_coords(bounds, gdobj):
    """
    Given a gdobj pointing to a raster and a list-like bounds (minx, miny, maxx, maxy),
    returns the row and col of the upper-leftmost pixel and the number
    of rows and columns to fetch. Also returns the GeoTransform of the
    clipped raster.
    
    Bounds will be clipped to the extents of the raster if they're beyond its
    limits.
    """    
    gt = gdobj.GetGeoTransform()
    
    # Get raster bounds
    rb_ulx, rb_uly = gt[0], gt[3]
    rb_lrx = rb_ulx + (gdobj.RasterXSize * gt[1])
    rb_lry = rb_uly + (gdobj.RasterYSize * gt[5])
    
    # Trim bounds to raster bounds if they're beyond
    warn = False
    if rb_ulx > bounds[0]: 
        bounds[0] = rb_ulx
        warn = True
    if rb_uly < bounds[3]:
        bounds[3] = rb_uly
        warn = True
    if rb_lrx < bounds[2]:
        bounds[2] = rb_lrx
        warn = True
    if rb_lry > bounds[1]:
        bounds[1] = rb_lry
        warn = True
    if warn is True:
        print('Provided bounds or bounding box was trimmed to the extents of the raster.')
    
    ul_c = floor(-(gt[0] - bounds[0]) / gt[1])
    ul_r = floor((gt[3] - bounds[3]) / abs(gt[5]))
    
    n_cols = ceil((bounds[2] - bounds[0]) / gt[1])
    n_rows = ceil((bounds[3] - bounds[1]) / abs(gt[5]))    
    
    gt_out = (gt[0] + (ul_c*gt[1]), gt[1], gt[2], gt[3] + (ul_r * gt[5]), gt[4], gt[5])
    
    return (ul_r, ul_c, n_cols, n_rows), gt_out



def parse_path(path):
    """
    Parses a file or folderpath into: base, folder (where folder is the 
    outermost subdirectory), filename, and extention. Filename and extension
    are empty if a directory is passed.
    """
    
    if path[0] != os.sep and platform.system() != 'Windows': # This is for non-windows...
        path = os.sep + path
    
    
    # Pull out extension and filename, if exist
    if '.' in path:
        extension = '.' + path.split('.')[-1]
        temp = path.replace(extension,'')
        filename = temp.split(os.sep)[-1]
        drive, temp = os.path.splitdrive(temp)
        path = os.path.join(*temp.split(os.sep)[:-1])
        path = drive + os.sep + path
    else:
        extension = ''
        filename = ''
    
    # Pull out most exterior folder
    folder = path.split(os.sep)[-1]
    
    # Pull out base
    drive, temp = os.path.splitdrive(path)
    base = os.path.join(*temp.split(os.sep)[:-1])
    base = drive + os.sep + base
    
    return base, folder, filename, extension


def delete_file(file):
    # Deletes a file. Input is file's location on disk (path + filename)
    try:
        os.remove(file)
    except OSError:
        pass


def raster_extents(raster_path):
    
    # Outputs extents as [xmin, xmax, ymin, ymax]                
    
    # Check if file is shapefile, else treat as raster
    fext = raster_path.split('.')[-1]
    if fext == 'shp' or fext == 'SHP':
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shapefile = driver.Open(raster_path, 0) # open as read-only
        layer = shapefile.GetLayer()
        ext = np.array(layer.GetExtent())
        extents = [ext[0], ext[1], ext[3], ext[2]]
    else:
        # Get the clipping raster lat/longs
        rast = gdal.Open(raster_path)
        cgt = rast.GetGeoTransform()
        clip_ULx = cgt[0]
        clip_ULy = cgt[3]
        clip_LRx = cgt[0] + cgt[1] * rast.RasterXSize
        clip_LRy = cgt[3] + cgt[5] * rast.RasterYSize
        extents = [clip_ULx, clip_LRx, clip_ULy, clip_LRy]
        
    return extents


def overlay_watersheds(hpus, basins, check_coverage=False):
    """
    Overlays HPUs on a GeoDataFrame of watersheds/basins and returns a dataframe
    grouped by watersheds that contains the HPUs and respective areas for each
    within each watershed. 

    Parameters
    ----------
    hpus : geopandas.GeoDataFrame
        Computed by the hpu class.
    basins : geopandas.GeoDataFrame
        At a minimum, needs two columns: the watershed geometries and an id
        column called 'id_gage'.

    Returns
    -------
    regrouped : pandas.DataFrame
        Contains three columns: id_gage, hpu_id (array), area_km2 (array). The
        ordering of the hpu_id and area_km2 arrays correspond. 

    """
    
    if check_coverage is True:
        # Ensure that all basins are completely covered by HPUs
        hpu_poly = unary_union(hpus.geometry.values)
        for g, bid in zip(basins.geometry.values, basins['id_gage'].values):
            if g.within(hpu_poly) is False:
                print('Warning: basin {} is not completely covered by HPUs.'.format(bid))
    
    # Compute basin areas (recompute since they're already provided by VotE)
    basins['basin_area_km2'] = [ru.area_4326(geom)[0] for geom in basins.geometry.values] 
    
    basins = basins[['id_gage', 'geometry', 'basin_area_km2']]
    intersected = gpd.overlay(hpus, basins, how="intersection")
    int_areas = []
    for g in intersected.geometry.values:
        if type(g) is MultiPolygon:
            this_area = 0
            for geom in g.geoms:    
                this_area = this_area + ru.area_4326(geom)[0]
        else:
            this_area = ru.area_4326(geom)[0]
        int_areas.append(this_area)           
    intersected['overlap_area_km2'] = int_areas
    intersected = intersected[['hpu_id', 'id_gage', 'area_sum', 'overlap_area_km2', 'basin_area_km2']]
    intersected.sort_values(by=['id_gage', 'overlap_area_km2'], inplace=True)

    return intersected


def segment_binary_im(all_coords, imshape, target_n_pix, initial_label=1):
    """
    Takes a binary image of imshape, with "on" pixel coordinates defined by 
    all_pixels and attempts to divide the binary image into regions of 
    size target_n_pix, giving each region a unique label starting with 
    initial_label. 
    
    Uses a breadth-first traversal algorithm to "grow" from initial points.
    An initial point is determined by the pixel that is farthest from the
    "centroid" of all pixels. Not actual centroid, simply the mean of all
    row, column coordinates.
    
    Parameters
    ----------
    all_pixels : set of tuples
        One entry per "on" pixel of the binary image.
    imshape : tuple OR list-like
        (number of rows, number of columns).
    target_n_pix : integer
        Desired size of regions to divide the binary image into. This algorithm
        does not guarantee these sizes exactly.
    initial_label : integer, optional
        The value to start with to apply labels to regions. The default is 1.

    Returns
    -------
    Iparent : np.array
        Array of imshape size where each pixel value is the region it belongs
        to. "Background" pixels (i.e. those that are "off" in the initial
        binary image) are labeled 0.
    
    label_id : int
        The highest label assigned to a region in Iparent 
        (i.e. Iparent.flatten().max()).

    """
    
    # Initialize traversal array
    Iparent = np.zeros(imshape, dtype=int)
    
    label_id = initial_label    
    # last_ncoords = len(all_coords)
    while all_coords:
        
        # update with each overall loop
        # Find the pixel farthest away from the centroid to initialize the traverse
        npcoords = np.array(list(all_coords))
        mean_r = int((np.max(npcoords[:,0]) - np.min(npcoords[:,0]))/2)
        mean_c = int((np.max(npcoords[:,1]) - np.min(npcoords[:,1]))/2)
        dists = np.sqrt((npcoords[:,0] - mean_r)**2 + (npcoords[:,1] - mean_c)**2)
        queue = [tuple(npcoords[np.argmax(dists)])]
                     
        Iparent[queue[0][0], queue[0][1]] = label_id
        all_coords.remove((queue[0][0], queue[0][1]))
           
        # Traverse
        ct = 1
        while queue and ct < target_n_pix:
            this_pt = queue.pop(0)
            neighs = iu.neighbor_idcs(this_pt[1], this_pt[0])
            for cs, rs in zip(neighs[0], neighs[1]):
                if cs < imshape[1] and rs < imshape[0]:
                    if (rs, cs) in all_coords:
                        queue.append((rs,cs))
                        Iparent[rs, cs] = label_id
                        all_coords.remove((rs,cs))
                        ct = ct + 1           

        label_id = label_id + 1   
    
    return Iparent, label_id


def create_hpus_from_classes(Iclasses, target_n_pix):
    """
    Given an initial image of HP classes (Iclasses), this will divide those
    classes into HPUs of approximately target_n_pix areas. 

    Parameters
    ----------
    Iclasses : numpy.array
        Image of HPU class labels for each pixel in the domain.
    target_n_pix : integer
        Target size for each HPU.

    Returns
    -------
    Iregions : numpy.array
        Same shape as Iclasses; each HPU is uniquely labeled.

    """
   
    rp, Ilabeled = iu.regionprops(Iclasses, props=['coords', 'area'])
    Iregions = np.ones(Iclasses.shape, dtype=int) * -1
    reg_label = 1
    for area, coords in zip(rp['area'], rp['coords']):
        if area <= target_n_pix * 1.5:
            Iregions[coords[:,0], coords[:,1]] = reg_label
            reg_label = reg_label + 1
            continue
        else:
            Isegmented, max_label = segment_binary_im(set([tuple(c) for c in coords]), Iclasses.shape, target_n_pix, initial_label=reg_label)
            Iregions[Isegmented>0] = Isegmented[Isegmented>0]
            reg_label = max_label + 1
            
    return Iregions



""" Graveyard """

#def fill_hp_holes(I, maxholesize):
#    """
#    This function has been replaced with simplify_hp. Choose unique_neghbor=True 
#    for the same behavior.
#    
#    Given an image whose pixels are all integer labels, this will cycle through
#    each label and fill any "holes" within that label up to maxholesize. I.e.
#    if label 2 has a small patch of 1's and 3's inside, this will overwrite the
#    1's and 3's with 2's.
#    """    
#    Ic = I.copy()
#
#    # We want to work from the least frequent to most frequent label, so find
#    # their frequencies
#    a = np.unique(I, return_counts=True)
#    forsort = np.array([a[0], a[1]]).T
#    forsort = forsort[forsort[:,1].argsort()]
#    # Remove -1 as it is nodata and set separately
#    forsort = forsort[forsort[:,0]!=-1,:]
#    unique_hps_sorted = forsort[:,0]
#           
#    for u in unique_hps_sorted:
#        
#        # Make an image of just the unique value locations
#        Iu = np.zeros(I.shape, dtype=np.bool)
#        Iu[Ic==u] = True
#        
#        # Fill all the holes up to maxholesize of the unique label image
#        Iu = iu.fill_holes(Iu, maxholesize=maxholesize)
#           
#        Ic[Iu==True] = u
#        
#    return Ic


# def polygonize_hpu(HPU_path, poly_path, path_merged=None):
#     """
#     Given the path to an HPU raster, this polygonizes it. Then it loads
#     the polygon file and recombines all HPU polygons of the same type.
    
#     merge_hpus : bool, If True, all HPUs of the same type will be merged
#                  into a Multipolygon.
#     """
#     # import pdb
#     # pdb.set_trace()
#     # mapping between gdal type and ogr field type
#     type_mapping = {gdal.GDT_Byte: ogr.OFTInteger,
#                     gdal.GDT_UInt16: ogr.OFTInteger,
#                     gdal.GDT_Int16: ogr.OFTInteger,
#                     gdal.GDT_UInt32: ogr.OFTInteger,
#                     gdal.GDT_Int32: ogr.OFTInteger,
#                     gdal.GDT_Float32: ogr.OFTReal,
#                     gdal.GDT_Float64: ogr.OFTReal,
#                     gdal.GDT_CInt16: ogr.OFTInteger,
#                     gdal.GDT_CInt32: ogr.OFTInteger,
#                     gdal.GDT_CFloat32: ogr.OFTReal,
#                     gdal.GDT_CFloat64: ogr.OFTReal}

#     # Initial polygonization with gdal.Polygonize()
#     ds = gdal.Open(HPU_path)
#     prj = ds.GetProjection()
#     srcband = ds.GetRasterBand(1)
#     dst_layername = "Shape"
#     drv = ogr.GetDriverByName("ESRI Shapefile")
#     if os.path.exists(poly_path):
#         drv.DeleteDataSource(poly_path)
#     dst_ds = drv.CreateDataSource(poly_path)
#     srs = osr.SpatialReference(wkt=prj)

#     dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)
#     raster_field = ogr.FieldDefn('hpu', type_mapping[srcband.DataType])
#     dst_layer.CreateField(raster_field)
#     gdal.Polygonize(srcband, srcband, dst_layer, 0, [], callback=None)
#     del HPU_path, ds, srcband, dst_ds, dst_layer
    
#     # Merge polygons of same HPU type
#     if path_merged is not None:
#         pgons = gpd.read_file(poly_path)
#         unique_ids = np.unique(pgons.hpu.values)
#         new_geoms = []
#         for uid in unique_ids:
#             polys_2_merge = pgons.geometry.values[pgons.hpu.values == uid]
#             new_geoms.append(cascaded_union(polys_2_merge))
        
#         merged_gdf = gpd.GeoDataFrame(geometry=new_geoms)
#         merged_gdf['hpu'] = unique_ids
#         merged_gdf.crs = pgons.crs
#         merged_gdf.to_file(path_merged)


