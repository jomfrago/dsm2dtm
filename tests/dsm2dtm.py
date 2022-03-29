"""
dsm2dtm - Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)
Author: Naman Jain
        naman.jain@btech2015.iitgn.ac.in
        www.namanji.wixsite.com/naman/
"""

import os
import numpy as np
import rasterio
import argparse
from typing import Union
from pathlib import Path

try:
    import gdal
except:
    from osgeo import gdal


def downsample_raster(
    in_path: Union[str, Path], out_path: Union[str, Path], downsampling_factor: float
) -> None:
    """
    Generates a downsampled raster, downsampled by a factor of downsampling_factor.
    Args:
        in_path: Path to the input tiff which needs to be downsampled.
        out_path: Path where the downsampled tiff will be generated.
        downsampling_factor: Factor by which the input image will be downsampled.
    """
    gdal_raster = gdal.Open(str(in_path))
    width, height = gdal_raster.RasterXSize, gdal_raster.RasterYSize
    gdal.Translate(
        str(out_path),
        str(in_path),
        width=int((width // downsampling_factor)),
        height=int((height // downsampling_factor)),
        outputType=gdal.GDT_Float32,
    )


def resample_raster(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    target_height: int,
    target_width: int,
) -> None:
    """
    Resamples the input raster to match with the target shape.
    Args:
        in_path: Path to the input tiff which needs to be downsampled.
        out_path: Path where the downsampled tiff will be generated.
        target_width: Width of the resulting tiff image.
        target_height: Height of the resulting tiff image.
    """
    gdal.Translate(
        str(out_path),
        str(in_path),
        width=target_width,
        height=target_height,
        resampleAlg="bilinear",
        outputType=gdal.GDT_Float32,
    )


def generate_slope_raster(
    in_path: Union[str, Path], out_path: Union[str, Path]
) -> None:
    """
    Generates a slope raster from the input DEM raster.
    Args:
        in_path: path to the DEM raster
        out_path: path to the generated slope image
    """
    cmd = "gdaldem slope -alg ZevenbergenThorne {} {}".format(in_path, out_path)
    os.system(cmd)


def get_mean(raster_path: Union[str, Path], ignore_value: int = -9999.0) -> float:
    """
    Returns the mean pixel value of the input raster while ingorning the ignore_value
    Args:
        raster_path: Path to the input tiff file
        ignore_value: Pixel value that will be ignored in calculation of mean.
    Returns:
        Average Pixel value.
    """
    np_raster = np.array(gdal.Open(str(raster_path)).ReadAsArray())
    return np_raster[np_raster != ignore_value].mean()


def extract_dtm(
    dsm_path: Union[str, Path],
    ground_dem_path: Union[str, Path],
    non_ground_dem_path: Union[str, Path],
    radius: int,
    terrain_slope: float,
) -> None:
    """
    Generates a ground DEM and non-ground DEM raster from the input DSM raster.
    Args:
        dsm_path: Path to the input DSM raster.
        radius: Search radius of kernel (in unit: number of cells)
        terrain_slope: average slope value of the input DSM
        ground_dem_path: Path where ground DEM raster will be generated
        non_ground_dem_path: Path where non-ground DEM raster will be generated
    """
    cmd = "saga_cmd grid_filter 7 -INPUT {} -RADIUS {} -TERRAINSLOPE {} -GROUND {} -NONGROUND {}".format(
        dsm_path, radius, terrain_slope, ground_dem_path, non_ground_dem_path
    )
    os.system(cmd)


def remove_noise(
    ground_dem_path: Union[str, Path],
    out_path: Union[str, Path],
    ignore_value: float = -99999.0,
    std_factor: float = 1.5,
    no_data_value: float = -99999.0,
) -> None:
    """
    Repalce noisy pixels (high elevation data points like roofs, etc.) in ground DEM raster with No-data Value (-99999.0)
    Args:
        ground_dem_path: path to the generated ground DEM raster
        out_path: path to the filtered ground DEM raster
        ignore_value: Pixel value which will be ignored during calulations. Generally, this should be equal to no-data value
        std_factor: Standard deviation factor used while calculating threshold value.
        no_data_value: replacing value in the ground raster (useful for rendering in GIS visulaization softwares).

    """
    ground_np = np.array(gdal.Open(str(ground_dem_path)).ReadAsArray())
    std = ground_np[ground_np != ignore_value].std()
    mean = ground_np[ground_np != ignore_value].mean()
    threshold_value = mean + std_factor * std
    ground_np[ground_np >= threshold_value] = no_data_value
    save_array_as_geotif(ground_np, ground_dem_path, out_path)


def save_array_as_geotif(
    array: np.ndarray, source_tif_path: Union[str, Path], out_path: Union[str, Path]
) -> None:
    """
    Generates a geotiff raster from the input numpy array (height * width * depth)
    Args:
        array: numpy array to be saved as geotiff
        source_tif_path: path to the reference geotiff from which projection and geotransformation information will be extracted.
        out_path: Path where Geotiff raster will be generated
    """
    if len(array.shape) > 2:
        height, width, depth = array.shape
    else:
        height, width = array.shape
        depth = 1
    source_tif = gdal.Open(str(source_tif_path))
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(out_path, width, height, depth, gdal.GDT_Float32)
    if depth != 1:
        for i in range(depth):
            dataset.GetRasterBand(i + 1).WriteArray(array[:, :, i])
    else:
        dataset.GetRasterBand(1).WriteArray(array)
    geotrans = source_tif.GetGeoTransform()
    proj = source_tif.GetProjection()
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)
    dataset.FlushCache()
    dataset = None


def sdat_to_gtiff(
    sdat_raster_path: Union[str, Path], out_gtiff_path: Union[str, Path]
) -> None:
    """
    Generates a GeoTIFF corresponding to input sdat image.
    Args:
        sdat_raster_path: Path to the input sdat image.
        out_gtiff_path: Path where the geotiff will be generated.
    """
    gdal.Translate(
        str(out_gtiff_path),
        str(sdat_raster_path),
        format="GTiff",
    )


def close_gaps(in_path: str, out_path: str, threshold: float = 0.1) -> None:
    """
    Interpolates the holes (no data value) in the input raster.
    Input:
        in_path: {string} path to the input raster with holes
        threshold: {float} Tension Threshold
    Output:
        out_path: {string} path to the generated raster with closed holes.
    """
    cmd = "saga_cmd grid_tools 7 -INPUT {} -THRESHOLD {} -RESULT {}".format(
        in_path, threshold, out_path
    )
    os.system(cmd)


def smoothen_raster(in_path: str, out_path: str, radius: int = 2) -> None:
    """
    Applies gaussian filter to the input raster.
    Input:
        in_path: {string} path to the input raster
        radius: {int} kernel radius to be used for smoothing
    Output:
        out_path: {string} path to the generated smoothened raster
    """
    cmd = "saga_cmd grid_filter 1 -INPUT {} -RESULT {} -KERNEL_TYPE 0 -KERNEL_RADIUS {}".format(
        in_path, out_path, radius
    )
    os.system(cmd)


def subtract_rasters(
    rasterA_path: str, rasterB_path: str, out_path: str, no_data_value: int = -99999.0
) -> None:
    cmd = 'gdal_calc.py -A {} -B {} --outfile {} --NoDataValue={} --calc="A-B"'.format(
        rasterA_path, rasterB_path, out_path, no_data_value
    )
    os.system(cmd)


def replace_values(
    rasterA_path: str,
    rasterB_path: str,
    out_path: str,
    no_data_value: int = -99999.0,
    threshold: float = 0.98,
) -> None:
    """
    Replaces values in input rasterA with no_data_value where cell value >= threshold in rasterB
    Input:
        rasterA_path: {string} path to the input rasterA
        rasterB_path: {string} path to the input rasterB
    Output:
        out_path: {string} path to the generated raster
    """
    cmd = 'gdal_calc.py -A {} --NoDataValue={} -B {} --outfile {} --calc="{}*(B>={}) + (A)*(B<{})"'.format(
        rasterA_path,
        no_data_value,
        rasterB_path,
        out_path,
        no_data_value,
        threshold,
        threshold,
    )
    os.system(cmd)


def expand_holes_in_raster(
    in_path: str,
    search_window: int = 7,
    no_data_value: int = -99999.0,
    threshold: float = 50,
) -> np.ndarray:
    """
    Expands holes (cells with no_data_value) in the input raster.
    Input:
        in_path: {string} path to the input raster
        search_window: {int} kernel size to be used as window
        threshold: {float} threshold on percentage of cells with no_data_value
    Output:
        np_raster: {numpy array} Returns the modified input raster's array
    """
    np_raster = np.array(gdal.Open(in_path).ReadAsArray())
    height, width = np_raster.shape[0], np_raster.shape[1]
    for i in range(int((search_window - 1) / 2), width, 1):
        for j in range(int((search_window - 1) / 2), height, 1):
            window = np_raster[
                int(i - (search_window - 1) / 2) : int(i - (search_window - 1) / 2)
                + search_window,
                int(j - (search_window - 1) / 2) : int(j - (search_window - 1) / 2)
                + search_window,
            ]
            if (
                np.count_nonzero(window == no_data_value)
                >= (threshold * search_window**2) / 100
            ):
                try:
                    np_raster[i, j] = no_data_value
                except:
                    pass
    return np_raster


def get_raster_crs(raster_path: str) -> int:
    """
    Returns the CRS (Coordinate Reference System) of the raster
    Input:
        raster_path: {string} path to the source tif image
    """
    with rasterio.open(raster_path) as src:
        return int(src.crs.data["init"].split(":")[-1])


def get_raster_resolution(raster_path: Union[str, Path]) -> float:
    """
    Returns the X and Y resolution of the raster.
    Args:
        raster_path: Path to the source tiff image.
    Returns:
        X and Y resolution
    """
    src = gdal.Open(str(raster_path))
    raster_geotrans = src.GetGeoTransform()
    x_res = raster_geotrans[1]
    y_res = -raster_geotrans[5]
    return x_res, y_res


def get_res_and_downsample(dsm_path: str, temp_dir: str) -> str:
    # check DSM resolution. Downsample if DSM is of very high resolution to save processing time.
    x_res, y_res = get_raster_resolution(dsm_path)  # resolutions are in meters
    dsm_name = dsm_path.split("/")[-1].split(".")[0]
    dsm_crs = get_raster_crs(dsm_path)
    if dsm_crs != 4326:
        if x_res < 0.3 or y_res < 0.3:
            target_res = 0.3  # downsample to this resolution (in meters)
            downsampling_factor = target_res / gdal.Open(dsm_path).GetGeoTransform()[1]
            downsampled_dsm_path = os.path.join(temp_dir, dsm_name + "_ds.tif")
            # Dowmsampling DSM
            downsample_raster(dsm_path, downsampled_dsm_path, downsampling_factor)
            dsm_path = downsampled_dsm_path
    else:
        if x_res < 2.514e-06 or y_res < 2.514e-06:
            target_res = 2.514e-06  # downsample to this resolution (in degrees)
            downsampling_factor = target_res / gdal.Open(dsm_path).GetGeoTransform()[1]
            downsampled_dsm_path = os.path.join(temp_dir, dsm_name + "_ds.tif")
            # Dowmsampling DSM
            downsample_raster(dsm_path, downsampled_dsm_path, downsampling_factor)
            dsm_path = downsampled_dsm_path
    return dsm_path


def get_updated_params(dsm_path: str, search_radius: int, smoothen_radius: int) -> int:
    # search_radius and smoothen_radius are set wrt to 30cm DSM
    # returns updated parameters if DSM is of coarser resolution
    x_res, y_res = get_raster_resolution(dsm_path)  # resolutions are in meters
    dsm_crs = get_raster_crs(dsm_path)
    if dsm_crs != 4326:
        if x_res > 0.3 or y_res > 0.3:
            search_radius = int((min(x_res, y_res) * search_radius) / 0.3)
            smoothen_radius = int((min(x_res, y_res) * smoothen_radius) / 0.3)
    else:
        if x_res > 2.514e-06 or y_res > 2.514e-06:
            search_radius = int((min(x_res, y_res) * search_radius) / 2.514e-06)
            smoothen_radius = int((min(x_res, y_res) * smoothen_radius) / 2.514e-06)
    return search_radius, smoothen_radius


def main(
    dsm_path: str,
    out_dir: str,
    search_radius: int = 40,
    smoothen_radius: int = 45,
    dsm_replace_threshold_val: float = 0.98,
) -> str:
    # master function that calls all other functions
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)
    dsm_path = get_res_and_downsample(dsm_path, temp_dir)
    # get updated params wrt to DSM resolution
    search_radius, smoothen_radius = get_updated_params(
        dsm_path, search_radius, smoothen_radius
    )
    # Generate DTM
    # STEP 1: Generate slope raster from dsm to get average slope value
    dsm_name = dsm_path.split("/")[-1].split(".")[0]
    dsm_slp_path = os.path.join(temp_dir, dsm_name + "_slp.tif")
    generate_slope_raster(dsm_path, dsm_slp_path)
    avg_slp = int(get_mean(dsm_slp_path))
    # STEP 2: Split DSM into ground and non-ground surface rasters
    ground_dem_path = os.path.join(temp_dir, dsm_name + "_ground.sdat")
    non_ground_dem_path = os.path.join(temp_dir, dsm_name + "_non_ground.sdat")
    extract_dtm(
        dsm_path,
        ground_dem_path,
        non_ground_dem_path,
        search_radius,
        avg_slp,
    )
    # STEP 3: Applying Gaussian Filter on the generated ground raster (parameters: radius = 45, mode = Circle)
    smoothened_ground_path = os.path.join(temp_dir, dsm_name + "_ground_smth.sdat")
    smoothen_raster(ground_dem_path, smoothened_ground_path, smoothen_radius)
    # STEP 4: Generating a difference raster (ground DEM - smoothened ground DEM)
    diff_raster_path = os.path.join(temp_dir, dsm_name + "_ground_diff.sdat")
    subtract_rasters(ground_dem_path, smoothened_ground_path, diff_raster_path)
    # STEP 5: Thresholding on the difference raster to replace values in Ground DEM by no-data values (threshold = 0.98)
    thresholded_ground_path = os.path.join(
        temp_dir, dsm_name + "_ground_thresholded.sdat"
    )
    replace_values(
        ground_dem_path,
        diff_raster_path,
        thresholded_ground_path,
        threshold=dsm_replace_threshold_val,
    )
    # STEP 6: Removing noisy spikes from the generated DTM
    ground_dem_filtered_path = os.path.join(temp_dir, dsm_name + "_ground_filtered.tif")
    remove_noise(thresholded_ground_path, ground_dem_filtered_path)
    # STEP 7: Expanding holes in the thresholded ground raster
    bigger_holes_ground_path = os.path.join(
        temp_dir, dsm_name + "_ground_bigger_holes.sdat"
    )
    temp = expand_holes_in_raster(ground_dem_filtered_path)
    save_array_as_geotif(temp, ground_dem_filtered_path, bigger_holes_ground_path)
    # STEP 8: Close gaps in the DTM
    dtm_path = os.path.join(temp_dir, dsm_name + "_dtm.sdat")
    close_gaps(bigger_holes_ground_path, dtm_path)
    # STEP 9: Convert to GeoTiff
    dtm_array = gdal.Open(dtm_path).ReadAsArray()
    dtm_tif_path = os.path.join(out_dir, dsm_name + "_dtm.tif")
    # save_array_as_geotif(dtm_array, dsm_path, dtm_tif_path)
    sdat_to_gtiff(dtm_path, dtm_tif_path)
    return dtm_tif_path


# -----------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate DTM from DSM")
    parser.add_argument("--dsm", help="dsm path string")
    args = parser.parse_args()
    dsm_path = args.dsm
    out_dir = "generated_dtm"
    dtm_path = main(dsm_path, out_dir)
    print("######### DTM generated at: ", dtm_path)
