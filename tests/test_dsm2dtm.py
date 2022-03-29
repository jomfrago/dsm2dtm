import os
import shutil

import pytest
import numpy as np

try:
    import gdal
except:
    from osgeo import gdal
from fake_geo_images.fakegeoimages import FakeGeoImage
from rasterio.transform import from_bounds

import dsm2dtm


@pytest.fixture(scope="session")
def test_image():
    test_file, _ = FakeGeoImage(256, 256, 1, "int8").create(
        14,
        from_bounds(
            1370000.0,
            2500000.0,
            1370256.0,
            2500256.0,
            width=256,
            height=256,
        ),
        "test_image",
    )
    yield test_file.resolve()
    test_file.unlink()


@pytest.fixture(scope="session", autouse=True)
def generate_test_dir():
    os.makedirs("/tmp/temp_dsm2dtm", exist_ok=True)
    yield
    shutil.rmtree("/tmp/temp_dsm2dtm")


def test_downsample_raster(test_image):
    dsm2dtm.downsample_raster(test_image, "/tmp/temp_dsm2dtm/ds.tif", 2.0)
    assert os.path.isfile("/tmp/temp_dsm2dtm/ds.tif")
    out = gdal.Open("/tmp/temp_dsm2dtm/ds.tif").ReadAsArray()
    assert out.shape == (128, 128)


@pytest.mark.parametrize(
    "out_path, target_height, target_width",
    [
        ("/tmp/temp_dsm2dtm/resampled.tif", "500", "500"),
        ("/tmp/temp_dsm2dtm/resampled.tif", "100", "100"),
        ("/tmp/temp_dsm2dtm/resampled.tif", "100", "300"),
    ],
)
# 3 testing scenarios - upsample, downsample, and different height and width
def test_resample_raster(test_image, out_path, target_width, target_height):
    dsm2dtm.resample_raster(test_image, out_path, target_height, target_width)
    assert os.path.isfile(out_path)
    out = gdal.Open(out_path).ReadAsArray()
    assert out.shape == (int(target_height), int(target_width))


def test_generate_slope_raster(test_image):
    dsm2dtm.generate_slope_raster(test_image, "/tmp/temp_dsm2dtm/slope.tif")
    assert os.path.isfile("/tmp/temp_dsm2dtm/slope.tif")


def test_get_mean(test_image):
    assert dsm2dtm.get_mean(test_image) == 7.068572998046875


def test_get_mean_ignore_value(test_image):
    assert dsm2dtm.get_mean(test_image, 1) == 28.740112994350284


def test_extract_dtm(test_image):
    dsm2dtm.extract_dtm(
        test_image,
        "/tmp/temp_dsm2dtm/ground.sdat",
        "/tmp/temp_dsm2dtm/non_ground.sdat",
        5,
        4,
    )
    assert os.path.isfile("/tmp/temp_dsm2dtm/ground.sdat")
    assert os.path.isfile("/tmp/temp_dsm2dtm/non_ground.sdat")
    ground_array = gdal.Open("/tmp/temp_dsm2dtm/ground.sdat").ReadAsArray()
    assert ground_array.shape == (256, 256)
    assert ground_array.mean() == -21875.5234375
    non_ground_array = gdal.Open("/tmp/temp_dsm2dtm/non_ground.sdat").ReadAsArray()
    assert non_ground_array.shape == (256, 256)
    assert non_ground_array.mean() == -78116.3984375


def test_remove_noise(test_image):
    dsm2dtm.remove_noise(test_image, "/tmp/temp_dsm2dtm/noise.tif")
    assert os.path.isfile("/tmp/temp_dsm2dtm/noise.tif")
    assert gdal.Open("/tmp/temp_dsm2dtm/noise.tif").ReadAsArray().shape == (256, 256)


def test_save_array_as_geotif(test_image):
    test_array = np.arange(120).reshape(3,4,10)
    dsm2dtm.save_array_as_geotif(test_array, test_image, "/tmp/temp_dsm2dtm/saved_array.tif")
    assert os.path.isfile("/tmp/temp_dsm2dtm/saved_array.tif")
    assert gdal.Open("/tmp/temp_dsm2dtm/saved_array.tif").ReadAsArray().shape == (10, 3, 4)


def test_sdat_to_gtiff(test_image):
    # TODO: add assertions on whether the input and output array is same
    # TODO: test this on actual sdat file
    dsm2dtm.sdat_to_gtiff(test_image, "/tmp/temp_dsm2dtm/temp_sdat_to_gtiff.tif")
    assert os.path.isfile("/tmp/temp_dsm2dtm/temp_sdat_to_gtiff.tif")


def test_close_gaps():
    pass


def test_smoothen_raster():
    pass


def test_subtract_rasters():
    pass


def test_replace_values():
    pass


def test_expand_holes_in_raster():
    pass


def test_get_raster_crs(test_image):
    assert dsm2dtm.get_raster_crs(test_image) == 3857


def test_get_raster_resolution(test_image):
    assert dsm2dtm.get_raster_resolution(test_image) == (1.0, 1.0)


def test_get_res_and_downsample():
    pass


def test_get_updated_params():
    pass


def test_main():
    pass
