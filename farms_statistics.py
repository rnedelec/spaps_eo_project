import rasterio
from rasterio.mask import mask
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import geopandas as gpd
import sys

import numpy as np

import matplotlib.pyplot as plt

sys.path.append(r"../spectralrao-monitoring")

import spectralrao

if __name__ == '__main__':
    down_sampling_factor = 10
    upscale_factor = 1 / down_sampling_factor

    path_to_pneo_tile_ned = r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo\000125930_1_1_STD_A\IMG_01_PNEO3_PMS-FS\IMG_PNEO3_202204031105489_PMS-FS_ORT_PWOI_000125930_1_1_F_1_NED_R2C1.JP2"
    path_to_pneo_tile_rgb = r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo\000125930_1_1_STD_A\IMG_01_PNEO3_PMS-FS\IMG_PNEO3_202204031105489_PMS-FS_ORT_PWOI_000125930_1_1_F_1_RGB_R2C1.JP2"

    with rasterio.open(path_to_pneo_tile_ned) as ned_dataset:
        data = ned_dataset.read(
            [1, 2, 3],
            out_shape=(
                ned_dataset.count,
                int(ned_dataset.height * upscale_factor),
                int(ned_dataset.width * upscale_factor)
                ),
            resampling=Resampling.average,
            )

        data_float = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Compute NDRE
    nir = data_float[0, :, :]
    red_edge = data_float[1, :, :]
    ndre = (nir - red_edge) / (nir + red_edge)

    plt.figure()
    plt.imshow(ndre)
    plt.colorbar()
    plt.show()

    # with rasterio.open("./NDRE_3m.tif") as ndre_dataset:
