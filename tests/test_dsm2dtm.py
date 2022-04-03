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
def test_image1():
    test_file1, _ = FakeGeoImage(256, 256, 1, "int8").create(
        14,
        from_bounds(
            1370000.0,
            2500000.0,
            1370256.0,
            2500256.0,
            width=256,
            height=256,
        ),
        "test_image1",
    )
    yield test_file1.resolve()
    test_file1.unlink()


@pytest.fixture(scope="session")
def test_image2():
    test_file2, _ = FakeGeoImage(256, 256, 1, "int8").create(
        42,
        from_bounds(
            1370000.0,
            2500000.0,
            1370256.0,
            2500256.0,
            width=256,
            height=256,
        ),
        "test_image2",
    )
    yield test_file2.resolve()
    test_file2.unlink()


@pytest.fixture(scope="session")
def test_image1_high_res():
    test_file1_high_res, _ = FakeGeoImage(256, 256, 1, "int8").create(
        14,
        from_bounds(
            1370000.0,
            2500000.0,
            1370004.0,
            2500004.0,
            width=256,
            height=256,
        ),
        "test_image1_high_res",
    )
    yield test_file1_high_res.resolve()
    test_file1_high_res.unlink()


@pytest.fixture(scope="session")
def test_image1_crs4326():
    test_file1_crs4326, _ = FakeGeoImage(256, 256, 1, "int8", crs=4326).create(
        14,
        from_bounds(
            0.0,
            0.0,
            0.00000001,
            0.00000001,
            width=256,
            height=256,
        ),
        "test_image1_crs4326",
    )
    yield test_file1_crs4326.resolve()
    test_file1_crs4326.unlink()


@pytest.fixture(scope="session", autouse=True)
def generate_test_dir():
    os.makedirs("/tmp/temp_dsm2dtm", exist_ok=True)
    yield
    shutil.rmtree("/tmp/temp_dsm2dtm")


def test_downsample_raster(test_image1):
    dsm2dtm.downsample_raster(test_image1, "/tmp/temp_dsm2dtm/ds.tif", 2.0)
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
def test_resample_raster(test_image1, out_path, target_width, target_height):
    dsm2dtm.resample_raster(test_image1, out_path, target_height, target_width)
    assert os.path.isfile(out_path)
    out = gdal.Open(out_path).ReadAsArray()
    assert out.shape == (int(target_height), int(target_width))


def test_generate_slope_raster(test_image1):
    dsm2dtm.generate_slope_raster(test_image1, "/tmp/temp_dsm2dtm/slope.tif")
    assert os.path.isfile("/tmp/temp_dsm2dtm/slope.tif")


def test_get_mean(test_image1):
    assert dsm2dtm.get_mean(test_image1) == 7.068572998046875


def test_get_mean_ignore_value(test_image1):
    assert dsm2dtm.get_mean(test_image1, 1) == 28.740112994350284


def test_extract_dtm(test_image1):
    dsm2dtm.extract_dtm(
        test_image1,
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


def test_remove_noise(test_image1):
    dsm2dtm.remove_noise(test_image1, "/tmp/temp_dsm2dtm/noise.tif")
    assert os.path.isfile("/tmp/temp_dsm2dtm/noise.tif")
    assert gdal.Open("/tmp/temp_dsm2dtm/noise.tif").ReadAsArray().shape == (256, 256)


def test_save_array_as_geotif(test_image1):
    test_array = np.arange(120).reshape(3,4,10)
    dsm2dtm.save_array_as_geotif(test_array, test_image1, "/tmp/temp_dsm2dtm/saved_array.tif")
    assert os.path.isfile("/tmp/temp_dsm2dtm/saved_array.tif")
    assert gdal.Open("/tmp/temp_dsm2dtm/saved_array.tif").ReadAsArray().shape == (10, 3, 4)


def test_sdat_to_gtiff(test_image1):
    # First generate a .sdat file
    dsm2dtm.extract_dtm(
        test_image1,
        "/tmp/temp_dsm2dtm/ground.sdat",
        "/tmp/temp_dsm2dtm/non_ground.sdat",
        5,
        4,
    )
    assert os.path.isfile("/tmp/temp_dsm2dtm/ground.sdat")
    dsm2dtm.sdat_to_gtiff("/tmp/temp_dsm2dtm/ground.sdat", "/tmp/temp_dsm2dtm/temp_sdat_to_gtiff.tif")
    assert os.path.isfile("/tmp/temp_dsm2dtm/temp_sdat_to_gtiff.tif")
    sdat_array = gdal.Open("/tmp/temp_dsm2dtm/temp_sdat_to_gtiff.tif").ReadAsArray()
    assert sdat_array.shape == (256, 256)



def test_close_gaps(test_image1):
    dsm2dtm.close_gaps(test_image1, "/tmp/temp_dsm2dtm/closed_holes.sdat")
    assert os.path.isfile("/tmp/temp_dsm2dtm/closed_holes.sdat")
    closed_holes_array = gdal.Open("/tmp/temp_dsm2dtm/closed_holes.sdat").ReadAsArray()
    assert closed_holes_array.shape == (256, 256)


def test_smoothen_raster(test_image1):
    dsm2dtm.smoothen_raster(test_image1, "/tmp/temp_dsm2dtm/smoothened.sdat", 5)
    assert os.path.isfile("/tmp/temp_dsm2dtm/smoothened.sdat")
    smoothened_array = gdal.Open("/tmp/temp_dsm2dtm/smoothened.sdat").ReadAsArray()
    assert smoothened_array.shape == (256, 256)


def test_subtract_rasters(test_image1, test_image2):
    dsm2dtm.subtract_rasters(test_image1, test_image2, "/tmp/temp_dsm2dtm/subtracted.tif")
    assert os.path.isfile("/tmp/temp_dsm2dtm/subtracted.tif")
    subtracted_array = gdal.Open("/tmp/temp_dsm2dtm/subtracted.tif").ReadAsArray()
    assert subtracted_array.shape == (256, 256)
    assert subtracted_array.mean() == 4.2759552001953125


def test_replace_values(test_image1, test_image2):
    dsm2dtm.replace_values(test_image1, test_image2, "/tmp/temp_dsm2dtm/replaced.tif", -99999.0, 3.0)
    assert os.path.isfile("/tmp/temp_dsm2dtm/replaced.tif")
    replaced_array = gdal.Open("/tmp/temp_dsm2dtm/replaced.tif").ReadAsArray()
    assert replaced_array.shape == (256, 256)
    assert replaced_array.mean() == 3.769256591796875


def test_expand_holes_in_raster(test_image1):
    new_array = dsm2dtm.expand_holes_in_raster(test_image1, 5, 1, 30)
    assert new_array.shape == (256, 256)
    assert new_array.mean() == 1.2519989013671875


def test_get_raster_crs(test_image1):
    assert dsm2dtm.get_raster_crs(test_image1) == 3857


def test_get_raster_resolution(test_image1):
    assert dsm2dtm.get_raster_resolution(test_image1) == (1.0, 1.0)


@pytest.mark.parametrize(
    "test_image, out_height, out_width, array_mean_value",
    [
        ("test_image1", "256", "256", "7.068572998046875"),
        ("test_image1_high_res", "13", "13", "6.846153736114502"),
        ("test_image1_crs4326", "256", "256", "7.068572998046875"),
    ],
)
# 3 testing scenarios - different CRS and different resolutions
def test_get_res_and_downsample(test_image, out_height, out_width, array_mean_value, request):
    test_image = request.getfixturevalue(test_image)
    dsm_path = dsm2dtm.get_res_and_downsample(test_image, "/tmp/temp_dsm2dtm")
    assert os.path.isfile(dsm_path)
    dsm_array = gdal.Open(dsm_path).ReadAsArray()
    assert dsm_array.shape == (int(out_height), int(out_width))
    assert dsm_array.mean() == float(array_mean_value)

def test_get_updated_params():
    pass


def test_main():
    pass
