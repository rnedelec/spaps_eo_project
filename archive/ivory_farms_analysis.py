import rasterio
from rasterio.mask import mask
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import geopandas as gpd
import sys

import numpy as np

import matplotlib.pyplot as plt

sys.path.append(r"../../../spectralrao-monitoring")

import spectralrao


def clip_data_to_farm(farm, dataset,  plot=False, down_sampling_factor=10):
    # Mask the down-sampled raster with the shapefile
    out_image, out_transform = rasterio.mask.mask(dataset=dataset, shapes=[farm["geometry"]],
                                                  crop=True, indexes=[1, 2, 3])
    out_meta = dataset.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open("../temp_masked.tif", "w", **out_meta) as dest:
        dest.write(out_image)

    with rasterio.open("../temp_masked.tif", "r") as dataset_croped:
        # Read and down sample image
        data = dataset_croped.read(
            [1, 2, 3],
            out_shape=(
                dataset_croped.count,
                int(dataset_croped.height * 1/down_sampling_factor),
                int(dataset_croped.width * 1/down_sampling_factor)
                ),
            resampling=Resampling.average,
            )

        # scale image transform
        transform = dataset_croped.transform * dataset_croped.transform.scale(
            (dataset_croped.width / data.shape[-1]),
            (dataset_croped.height / data.shape[-2])
            )

    data = np.where(data == 0, np.nan, data)  # Replace 0 with NaN
    data_float = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    if plot:
        plt.figure()
        rasterio.plot.show(data_float, transform=transform)
        plt.plot(*farm["geometry"].exterior.xy, color="red",
                 linewidth=3)  # Plot farm polygon
        plt.show()

    return data_float, transform


def get_data_around_farm(farm, dataset, plot=False, upscale_factor=1):
    left, bottom, right, top = farm["geometry"].bounds

    window = from_bounds(left, bottom, right, top, dataset.transform)

    # Read and down sample image
    data = dataset.read(
        [1, 2, 3],
        window=window,
        out_shape=(
            dataset.count,
            int(window.height * upscale_factor),
            int(window.width * upscale_factor)
            ),
        resampling=Resampling.average,
        )

    if data.shape[1] > 0:
        print(farm['Farm_ID'])

        # # scale image transform
        # transform = dataset.transform * dataset.transform.scale(
        #     (window.width / data.shape[-1]),
        #     (window.height / data.shape[-2])
        #     )

        # Define the necessary parameters for the new transform
        scale_x = window.width / data.shape[-1] * dataset.transform.a
        scale_y = window.height / data.shape[-2]  * dataset.transform.e
        rotation = 0  # Example rotation angle in degrees
        translate_x = left
        translate_y = top

        # Create a new affine transform using the Affine class
        transform = rasterio.Affine(scale_x, rotation, translate_x,
                                    rotation, scale_y, translate_y)

        # Convert uint16 to float
        data_float = (data - np.min(data)) / (np.max(data) - np.min(data))

        if plot:
            plt.figure()
            rasterio.plot.show(data_float, transform=transform)
            plt.plot(*farm["geometry"].exterior.xy, color="red",
                     linewidth=3)  # Plot farm polygon
            plt.show()
    else:
        return None, None

    return data_float, transform


if __name__ == '__main__':

    # path_to_pneo_tile_ned = r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo\000125930_1_1_STD_A\IMG_01_PNEO3_PMS-FS\IMG_PNEO3_202204031105489_PMS-FS_ORT_PWOI_000125930_1_1_F_1_NED_R2C1.JP2"
    # path_to_pneo_tile_rgb = r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo\000125930_1_1_STD_A\IMG_01_PNEO3_PMS-FS\IMG_PNEO3_202204031105489_PMS-FS_ORT_PWOI_000125930_1_1_F_1_RGB_R2C1.JP2"

    # path_to_pneo_tile_ned = r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo\000125932_1_1_STD_A\IMG_01_PNEO3_PMS-FS\IMG_PNEO3_202201091049566_PMS-FS_ORT_PWOI_000125932_1_1_F_1_NED_R2C1.JP2"
    # path_to_pneo_tile_rgb = r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo\000125932_1_1_STD_A\IMG_01_PNEO3_PMS-FS\IMG_PNEO3_202201091049566_PMS-FS_ORT_PWOI_000125932_1_1_F_1_RGB_R2C1.JP2"

    # path_to_pneo_tile_ned = r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo\WO_000125933_1_1_SAL23112714-1_ACQ_PNEO3_02801606514388\IMG_PNEO3_202304011046525_PMS-FS_ORT_PWOI_000125933_1_1_F_1_NED_R1C1.JP2"
    # path_to_pneo_tile_rgb = r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo\WO_000125933_1_1_SAL23112714-1_ACQ_PNEO3_02801606514388\IMG_PNEO3_202304011046525_PMS-FS_ORT_PWOI_000125933_1_1_F_1_RGB_R1C1.JP2"

    path_to_pneo_tile_ned = r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo\000125930_1_1_STD_A\IMG_01_PNEO3_PMS-FS\IMG_PNEO3_202204031105489_PMS-FS_ORT_PWOI_000125930_1_1_F_1_NED_R3C1.JP2"
    path_to_pneo_tile_rgb = r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo\000125930_1_1_STD_A\IMG_01_PNEO3_PMS-FS\IMG_PNEO3_202204031105489_PMS-FS_ORT_PWOI_000125930_1_1_F_1_RGB_R3C1.JP2"

    farms = gpd.read_file(r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ivory_farms_shapefile")

    # # Statistics for farms with few bareland
    # farms_low_bareland = farms[farms['BARELAND'] <= 0.1]
    # print(farms_low_bareland.count())
    # print(farms_low_bareland[['FULL_SUN', 'SHADED', 'NON_COCOA', 'BARELAND', 'NATURAL']].mean())

    # selected_farm_id = 221
    selected_farm_id = 137
    plot_rgb = True
    down_sampling_factor = 10
    upscale_factor = 1 / down_sampling_factor
    na_tolerance = 0.3
    ndre_threshold = 0.2
    window = 11

    with rasterio.open(path_to_pneo_tile_ned) as dataset_ned:
        # Inspect farms

        # Change projection
        farms.to_crs(dataset_ned.crs, inplace=True)

        # Select farm
        farm = farms[farms['Farm_ID'].astype('int') == selected_farm_id].iloc[0]

        # # Iterate over farms
        # for i, farm in farms_low_bareland.iterrows():
        #
        #     # Select cacao
        #     if farm['NON_COCOA'] > 0.1:
        #         continue
        #
        #     print(farm)

        data_ned, transform = clip_data_to_farm(farm, dataset_ned, plot=True, down_sampling_factor=down_sampling_factor)
        # data_ned, transform = get_data_around_farm(farm, dataset_ned, plot=True, upscale_factor=upscale_factor)

        if data_ned is None:
            raise ValueError("No intersection between farm polygon and raster")
            # continue

        # if plot_rgb:
        #     with rasterio.open(path_to_pneo_tile_rgb) as dataset_rgb:
        #         data_rgb, _ = get_data_around_farm(farm, dataset_rgb, plot=True, upscale_factor=upscale_factor)

        # Compute NDRE
        nir = data_ned[0, :, :]
        red_edge = data_ned[1, :, :]
        ndre = (nir - red_edge) / (nir + red_edge)

        plt.figure()
        plt.imshow(ndre)
        # rasterio.plot.show(ndre, transform=transform)
        # plt.plot(*farm["geometry"].exterior.xy, color="red", linewidth=3)  # Plot farm polygon
        plt.colorbar()
        plt.show()

        masked_ndre = np.where(ndre > ndre_threshold, ndre, np.nan)
        plt.figure()
        # plt.imshow(masked_ndre)
        # plt.colorbar()
        rasterio.plot.show(masked_ndre, transform=transform)
        plt.plot(*farm["geometry"].exterior.xy, color="red",
                 linewidth=3)  # Plot farm polygon
        plt.show()

        # rao = spectralrao.spectralrao(masked_ndre, ".", window=window, na_tolerance=na_tolerance)
        #
        # plt.figure()
        # plt.imshow(rao[0])
        # plt.colorbar()
        # plt.show()
        #
        # plt.figure()
        # rasterio.plot.show(rao[0], transform=transform)
        # plt.plot(*farm["geometry"].exterior.xy, color="red", linewidth=3)  # Plot farm polygon
        # # plt.colorbar()
        # plt.show()



            # img = reshape_as_image(data)

            #
            # plt.figure()
            # plt.imshow(img)
            # plt.plot(*farm["geometry"].exterior.xy)  # Plot farm polygon
            # plt.show()
            #
            # nir = img[:, :, 0]
            # red_edge = img[:, :, 1]
            # ndre = (nir - red_edge) / (nir + red_edge)
            #
            # plt.figure()
            # plt.imshow(ndre)
            # plt.show()

            # break




