import rasterio
import numpy as np

def get_raster_band_and_profile(raster_filepath, band_index):
    with rasterio.open(raster_filepath) as dataset:
        profile = dataset.profile
        band = dataset.read(band_index)
    return band, profile

if __name__ == '__main__':

    nir_band_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\Reflectance\20221231_S2B_T29NPH_rdhz\20221231_S2B_T29NPH_rdhz_b8.tif"
    red_band_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\Reflectance\20221231_S2B_T29NPH_rdhz\20221231_S2B_T29NPH_rdhz_b4.tif"

    nir, profile = get_raster_band_and_profile(nir_band_filepath, 1)
    red, _ = get_raster_band_and_profile(red_band_filepath, 1)

    ndvi = (nir - red) / (nir + red)

    # And then change the band count to 1, set the
    # dtype to float32, and specify LZW compression.
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw')

    with rasterio.open('20221231_S2B_T29NPH_ndvi.tif', 'w', **profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)
