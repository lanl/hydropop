import os
import numpy as np
import pandas as pd
from osgeo import gdal
import geopandas as gpd
from scipy.cluster.vq import kmeans2
from skimage.graph import RAG
#
import ep_utils as eut
import rivgraph_ports as io


"""
The current HTHI has both large negative integers and nans as nodata.
"""

class epu():
    """
    The methods of this class are for generating and exporting ecopop Units.
    Functions are available for plotting and exporting as well.
    """
    
    def __init__(self, path_pop, path_hthi, path_water_raster=None, bounding=None, waterbody_thresh=None):
        """
        bounding can be either a 4-entry list-like of [W, S, E, N] coordinates or
        a GeoPandas-readable geofile (shapefile, etc.)
        
        waterbody_thresh : threshold in square kilometers above which waterbodies
                           will be masked. Water bodies smaller than this will 
                           be incorporated into epus.
        """
        # Quick check on inputs
        if waterbody_thresh is not None:
            if path_water_raster is None:
                raise KeyError('path_water_raster must be provided if waterbody_thresh is.')
        
        # Store some metadata
        self.paths = {'pop' : path_pop,
                      'hthi' : path_hthi,
                      'waterbodies' : path_water_raster}
        
        # Some georeferencing info--everything keys off hthi so make sure it's correct
        self.gdobj_hthi = gdal.Open(self.paths['hthi'])
        self.gdobj_pop = gdal.Open(self.paths['pop'])
        self.wkt = self.gdobj_hthi.GetProjection()
        self.gt = self.gdobj_hthi.GetGeoTransform()

        if bounding is not None:
            if type(bounding) is str:
                gdf = gpd.read_file(bounding)
                bounds = gdf.geometry.values.bounds[0]
            else:
                bounds = bounding
            rco, self.gt = eut.get_raster_clipping_coords(bounds, self.gdobj_hthi)
        else:
            rco = (0, 0, self.gdobj_hthi.RasterXSize, self.gdobj_hthi.RasterYSize)

                
        # Load the layers we'll use and handle masking/nodata
        self.I = {}
        self.I['pop'] = gdal.Open(self.paths['pop']).ReadAsArray(rco[1], rco[0], rco[2], rco[3])
        # Set population == 0 to 1e-10 so we can log-transform it
        self.I['pop'][self.I['pop']==0]= 1e-10
        self.I['pop'] = np.log10(self.I['pop']) # Population is log-10 transformed
        self.I['pop'][np.isnan(self.I['pop'])] = 0 # nodata are set to 0 in population - this could have unintended effects so be careful!

        self.I['hthi'] = gdal.Open(self.paths['hthi']).ReadAsArray(rco[1], rco[0], rco[2], rco[3])

        # Prepare mask
        # Mask is True for valid pixels
        self.I['mask'] = np.ones(self.I['hthi'].shape, dtype=bool)
        if waterbody_thresh is not None:
            wb = gdal.Open(self.paths['waterbodies']).ReadAsArray(rco[1], rco[0], rco[2], rco[3])   
            self.I['mask'][wb > waterbody_thresh] = False
        # Nodata in hthi should be masked
        self.I['mask'][~np.logical_and(self.I['hthi'] >= 0, self.I['hthi'] <= 1)] = False

        # Pre-mask the HTHI values        
        self.I['hthi'][~self.I['mask']] = np.nan
        
        # Some checking
        assert self.I['pop'].shape == self.I['hthi'].shape 
        assert self.I['pop'].shape == self.I['mask'].shape
    
        # Store valid pixel indices and values
        self.idcs = np.where(self.I['mask'])
        self.hi_vals = self.I['hthi'][self.idcs[0], self.idcs[1]]
        self.pop_vals = self.I['pop'][self.idcs[0], self.idcs[1]]
        
        
    def compute_ep_classes_kmeans(self, n_groups):
        """
        Creates an image where each pixel value is the HP group to which
        the pixel belongs. Pixels are grouped via a k-means clustering based
        on the (population, hab index) for each pixel. The number of groups
        must be specified and can be thought of as the number of regions
        in which the population, hab index space is divided into.
        
        n_groups is NOT the total number of ecopop units, but the number
        of HP unit types.
        
        self.centroids contains the centroid of the (population, hab index)
        "coordinates" of each group--there is no spatial information here.
        """
        coords = list(zip(self.hi_vals, self.pop_vals))
        self.centroids, labels = kmeans2(coords, n_groups, minit='points')
        self.I['epu_class'] = np.zeros(self.I['pop'].shape)
        self.I['epu_class'][self.idcs[0], self.idcs[1]] = labels
        
    
    def compute_ep_classes_ranges(self, breaks={'pop':[-100, -4, -2, -1, 0, 1, 100], 'hthi':[-0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1.1]}):
        """
        Divides the pop vs mhi space into ecopop units based on a supplied
        breaks dictionary that defines the boundaries along each axis.
        
        Populations of 0 were set to a very low number so as not to error in
        log-transformation. This should be accounted for when supplying breaks;
        i.e. make sure there's an interval that captures only this value.
        """
        self.n_ep_classes = (len(breaks['hthi'])-1) * (len(breaks['pop'])-1)
        popl, popu = breaks['pop'][:-1], breaks['pop'][1:]
        hthil, hthiu = breaks['hthi'][:-1], breaks['hthi'][1:]
        epclass_id = 1
        dmap = {}
        self.I['epu_class'] = np.zeros(self.I['hthi'].shape, dtype=int)            
        for pl, pup in zip(popl, popu):
            for hl, hu in zip(hthil, hthiu):
                dmap[epclass_id] = (pl, pup, hl, hu)
                self.I['epu_class'][np.logical_and(np.logical_and(self.I['hthi']>=hl, self.I['hthi']<hu),np.logical_and(self.I['pop']>=pl, self.I['pop']<pup))] = epclass_id
                epclass_id = epclass_id + 1
        
        # Set masked class values to 0
        # self.I['epu_class'][self.I['mask']] = 0
        self.class_map = dmap
        
        # Format the class map
        rows = ['HTHI_mean', 'pop_dens_mean', 'HTHI_min', 'HTHI_max', 'pop_dens_min', 'pop_dens_max']
        cmf = pd.DataFrame(index=rows, columns=sorted(list(self.class_map.keys())))
        for c in self.class_map:
            cmf.at['HTHI_mean', c] = (self.class_map[c][2] + self.class_map[c][3])/2
            cmf.at['pop_dens_mean', c] = (self.class_map[c][0] + self.class_map[c][1])/2
            cmf.at['HTHI_min', c] = self.class_map[c][2] 
            cmf.at['pop_dens_min', c] = self.class_map[c][0] 
            cmf.at['HTHI_max', c] = self.class_map[c][3]
            cmf.at['pop_dens_max', c] = self.class_map[c][1]
        self.class_map_formatted = cmf


    def simplify_epu_classes(self, min_class_size=4, nodata=0, maxiter=10, unique_neighbor=False):
        """
        Merges smaller epu classes into their neighbors. Uses an iterative 
        approach because class regions change if a neighboring class region
        is absorbed into it.
        unique_neighbor = True : 
        unique_neighbor = False : fills all patches smaller than minpatchsize with the mode of the neighboring pixel labels - this option will ensure that 


        Parameters
        ----------
        min_class_size : integer, optional
            Minimum area, in pixels, that a class size can have. The default is 4.
        nodata : integer, optional
            Class type 0 corresponds to nodata in the epu class code. Specifying
            this is necessary to avoid setting valid classes to nodata types.
            The default is 0.
        maxiter : integer, optional
            Maximum number of iteration to attempt to . The default is 10.
        unique_neighbor : boolean, optional
            If True, only merges a class region if its neighboring pixels all 
            share the same label (more conservative). This option will allow 
            patches smaller than min_class_size to persist. 
            If False, will merge ALL patches smaller than min_class_size using
            the mode of the neighboring pixel classes. This option will ensure
            that no class regions will be larger than min_class_size EXCEPT
            in cases of a class regions surrounded by nodata.
            The default is False.

        Returns
        -------
        Adds a 'epu_class_simplified' layer to the epu.I dictionary.

        """
        self.I['epu_class_simplified'] = eut.simplify_classes(self.I['epu_class'], 
                                                              minpatchsize=min_class_size, 
                                                              nodata=nodata, 
                                                              unique_neighbor=unique_neighbor, 
                                                              maxiter=maxiter)
    
    
    def compute_epus(self, target_epu_size, min_epu_size, nodata=0):
        """
        Divides the computed classes into regions of (approximately)
        target_epu_size pixels each. The resulting epu raster is then
        polygonized.
        
        target_size_pixels : target epu size in pixels
        minpatch is the smallest patch size allowed, in pixels
        unique_neighbor = True : only fills a patch if its neighboring pixels all share the same label (more conservative) - this option will allow patches larger than minpatch to persist.
        unique_neighbor = False : fills all patches smaller than minpatchsize with the mode of the neighboring pixel labels


        Parameters
        ----------
        target_epu_size : int
            The desired size of each epu, in pixels.
        min_epu_size : int
            The desired minimum size of each epu.
        nodata : int, optional
            Specify nodata class value. The epu class is designed to set these
            to 0. The default is 0.

        Returns
        -------
        Adds 'epu' and 'epu_simlified' layers to the epu.I dictionary.
        Adds a new attribute ('epus') to the class; this attribute is a
        polygonized version of self.I['epu_simplfied'].


        """
        # Use simplified version if computed
        if 'epu_class_simplified' not in self.I.keys():
            I = self.I['epu_class']
        else:
            I = self.I['epu_class_simplified']
            
        self.I['epu'] = eut.create_epus_from_classes(I, target_epu_size)
        
        # Simplify the computed epus
        self.I['epu_simplified'] = eut.simplify_epus(self.I['epu'], I, target_epu_size, min_epu_size, nodata)
        self.epus_all = eut.polygonize_epu(self.I['epu_simplified'], self.gt, self.wkt)
        self.epus = eut.polygonize_epu(self.I['epu_simplified'], self.gt, self.wkt, self.I['mask'])

    
    def compute_epu_stats(self, do_stats):
        """
        do_stats : dict
            keys are names of layers to compute stats for
            values are two-element lists of [path_to_raster, [stats to compute]]
        """
        # if 'area' in do_stats.keys():
        #     path_areagrid = do_stats['area'][0]
        #     if os.path.isfile(path_areagrid) is False:
        #         areagrid = eut.areagrid(do_stats['area'][2])
        #         io.write_geotiff(areagrid, self.gt, self.wkt, path_areagrid, dtype=gdal.GDT_Float32)
                
        self.epus = eut.epu_stats(do_stats, self.epus)
        
        # Add centroids of epu polygons
        centroids = [p.centroid.coords.xy for p in self.epus.geometry.values]
        self.epus['centroid_y'] = [c[1][0] for c in centroids]
        self.epus['centroid_x'] = [c[0][0] for c in centroids]
        
        # Update field names if necessary
        if 'epu_class_majority' in self.epus.keys():
            self.epus = self.epus.rename({'epu_class_majority':'epu_class'}, axis=1)
        
    def smooth_layers(self, layers, sigmas, write=False):
        """
        By default will smooth layers from the layers_norm dict. Call 
        eut.smooth_layer() directly if unnormalized layer smoothing is desired.
        sigma is the smoothing parameter. Higher sigma -> more smoothing.
        """
        for l, s in zip(layers, sigmas):
            self.layers_smooth[l] = eut.smooth_layer(self.layers_norm[l], sigma=s)
            
            if write is True:
                splitpath = os.path.splitext(self.paths[l])
                outpath = splitpath[0] + '_smooth' + splitpath[1]
                self.paths[l+'_smooth'] = outpath
                io.write_geotiff(self.layers_smooth[l], self.gt, self.wkt, self.paths[l+'_smooth'], dtype=gdal.GDT_Float32, nodata=np.nan)
        
    
    def watersheds(self, path, path_out=None):
        """
        Computes the fraction of watershed within each epu.
        
        path : str
            The path to the geopandas-readable watershed geometry file.
        
        path_out : str
            The path to write the watershed/epu dataframe. If None, nothing
            will be written but the dataframe will be stored as an object
            in the epu class.
            
        """
        gdf = gpd.read_file(path)
        self.watersheds = eut.overlay_watersheds(self.epus, gdf)
        
        if path_out is not None:
            self.watersheds.to_csv(path_out, index=False)
            
            
    def compute_adjacency(self, layer='epu_simplified'):
        """
        Computes the adjacency of a raster layer, typically 'epu_simplfied'.
        Must be run after computing epus if layer is not specified.

        Parameters
        ----------
        layer : str, optional
            The layer within self.I to compute adjaceny on. The default is 
            'epu_simplfied'.

        Returns
        -------
        adj_df : pandas.DataFrame
            The adjacency dataframe.

        """
        rag = RAG(self.I[layer], connectivity=2)
        adj_dict = {i: list(rag.neighbors(i)) for i in list(rag.nodes)}
        epu_ids = adj_dict.keys()
        adj_vals = []
        for hid in epu_ids:
            adj_vals.append(','.join([str(v) for v in adj_dict[hid]]))
        adj_df = pd.DataFrame({'epu_id':epu_ids, 'adjacency':adj_vals})
        
        return adj_df
        
    
    def export_raster(self, whichraster, path): 
        """
        Exports the geotiff and polygon versions of the ecopop Units. Check
        the paths dictionary for where these are exported.
        
        whichraster : str
            The key within self.I to export.
        path : str
            The path to export to.
        """     
        io.write_geotiff(self.I[whichraster], self.gt, self.wkt, path, dtype=gdal.GDT_UInt32)
        
        
        

    
        
        
        
