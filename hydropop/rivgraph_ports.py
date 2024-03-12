import numpy as np
from osgeo import gdal
from osgeo import osr


def neighbor_vals(I, c, r):
    """
    Returns the neighbor values in I of a specified pixel coordinate. Handles
    edge cases.

    Parameters
    ----------
    I : np.array
        Image to draw values from.
    c : int
        Column defining pixel to find neighbor values.
    r : int
        Row defining pixel to find neighbor values.

    Returns
    -------
    vals : np.array
        A flattened array of all the neighboring pixel values.

    """
    vals = np.empty((8, 1))
    vals[:] = np.NaN

    if c == 0:

        if r == 0:
            vals[4] = I[r, c+1]
            vals[6] = I[r+1, c]
            vals[7] = I[r+1, c+1]
        elif r == np.shape(I)[0]-1:
            vals[1] = I[r-1, c]
            vals[2] = I[r-1, c+1]
            vals[4] = I[r, c+1]
        else:
            vals[1] = I[r-1, c]
            vals[2] = I[r-1, c+1]
            vals[4] = I[r, c+1]
            vals[6] = I[r+1, c]
            vals[7] = I[r+1, c+1]

    elif c == I.shape[1]-1:

        if r == 0:
            vals[3] = I[r, c-1]
            vals[5] = I[r+1, c-1]
            vals[6] = I[r+1, c]
        elif r == I.shape[0]-1:
            vals[0] = I[r-1, c-1]
            vals[1] = I[r-1, c]
            vals[3] = I[r, c-1]
        else:
            vals[0] = I[r-1, c-1]
            vals[1] = I[r-1, c]
            vals[3] = I[r, c-1]
            vals[5] = I[r+1, c-1]
            vals[6] = I[r+1, c]

    elif r == 0:
        vals[3] = I[r, c-1]
        vals[4] = I[r, c+1]
        vals[5] = I[r+1, c-1]
        vals[6] = I[r+1, c]
        vals[7] = I[r+1, c+1]

    elif r == I.shape[0]-1:
        vals[0] = I[r-1, c-1]
        vals[1] = I[r-1, c]
        vals[2] = I[r-1, c+1]
        vals[3] = I[r, c-1]
        vals[4] = I[r, c+1]

    else:
        vals[0] = I[r-1, c-1]
        vals[1] = I[r-1, c]
        vals[2] = I[r-1, c+1]
        vals[3] = I[r, c-1]
        vals[4] = I[r, c+1]
        vals[5] = I[r+1, c-1]
        vals[6] = I[r+1, c]
        vals[7] = I[r+1, c+1]

    vals = np.ndarray.flatten(vals)

    return vals

def write_geotiff(raster, gt, wkt, path_export, dtype=gdal.GDT_UInt16,
                  options=['COMPRESS=LZW'], nbands=1, nodata=None,
                  color_table=None):
    """
    Writes a georeferenced raster to disk.

    Parameters
    ----------
    raster : np.array
        Image to be written. Shape is (nrows, ncols, nbands), although if only
        one band is present the shape can be just (nrows, ncols).
    gt : tuple
        GDAL geotransform for the raster. Often this can simply be copied from
        another geotiff via gdal.Open(path_to_geotiff).GetGeoTransform(). Can
        also be constructed following the gdal convention of
        (leftmost coordinate, pixel width, xskew, uppermost coordinate, pixel height, yskew).
        For non-rotated images, the skews will be zero.
    wkt : str
        Well-known text describing the coordinate reference system of the raster.
        Can be copied from another geotiff with gdal.Open(path_to_geotiff).GetProjection().
    path_export : str
        Path with extension of the geotiff to export.
    dtype : gdal.GDT_XXX, optional
        Gdal data type. Options for XXX include Byte, UInt16, UInt32, Int32,
        Float32, Float64 and complex types CInt16, Cint32, CFloat32 and CFloat64.
        If storing decimal data, use a Float type, binary data use Byte type.
        The default is gdal.GDT_UInt16 (non-float).
    options : list of strings, optional
        Options that can be fed to gdal dataset creator. See YYY for what
        can be specified by options.
        The default is ['COMPRESS=LZW'].
    nbands : int, optional
        Number of bands of the raster. The default is 1.
    nodata : numeric, optional
        Pixels with this value will be written as nodata. If None, no nodata
        value will be considered. The default is None.
    color_table : gdal.ColorTable, optional
        Color table to append to the geotiff. Can use colortable() function
        to create, or create a custom type with gdal.ColorTable().
        Note that color_tables can only be specified for Byte and UInt16 datatypes.
        The default is None.

    Returns
    -------
    None.

    """
    height = np.shape(raster)[0]
    width = np.shape(raster)[1]

    # Add empty dimension for single-band images
    if len(raster.shape) == 2:
        raster = np.expand_dims(raster, -1)

    # Prepare destination file
    driver = gdal.GetDriverByName("GTiff")
    if options != None:
        dest = driver.Create(path_export, width, height, nbands, dtype,
                             options)
    else:
        dest = driver.Create(path_export, width, height, nbands, dtype)

    # Write output raster
    for b in range(nbands):
        dest.GetRasterBand(b+1).WriteArray(raster[:, :, b])

        if nodata is not None:
            dest.GetRasterBand(b+1).SetNoDataValue(nodata)

        if color_table != None:
            dest.GetRasterBand(1).SetRasterColorTable(color_table)

    # Set transform and projection
    dest.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dest.SetProjection(srs.ExportToWkt())

    # Close and save output raster dataset
    dest = None

def neighbor_idcs(c, r):
    """
    Returns the column, row coordinats of all eight neighbors of a given
    column and row.

    Returns are ordered as

    [0 1 2
     3   4
     5 6 7]

    Parameters
    ----------
    c : int
        Column.
    r : int
        Row.

    Returns
    -------
    cidcs : list
        Columns of the eight neighbors.
    ridcs : TYPE
        Rows of the eight neighbors.

    """
    cidcs = [c-1, c, c+1, c-1, c+1, c-1, c, c+1]
    ridcs = [r-1, r-1, r-1, r, r, r+1, r+1, r+1]

    return cidcs, ridcs