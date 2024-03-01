import rasterio
import numpy as np

with rasterio.open(r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH\20221231_S2B_T29NPH_ndfi.tif") as dataset:
    profile = dataset.profile
    ndfi = dataset.read(1)
    ndfi_mask = np.where(ndfi > 6500, 1, 0)

    # And then change the band count to 1, set the
    # dtype to uint8, and specify LZW compression.
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw')

    with rasterio.open('ndfi_binary_mask.tif', 'w', **profile) as dst:
        dst.write(ndfi_mask.astype(rasterio.uint8), 1)