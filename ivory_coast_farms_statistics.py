import sys
import json
from typing import Union

import shapely.errors

from spaps_eo_project import utils
import geopandas as gpd
import rasterio
import rasterio.mask
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import shape

sys.path.append(r"../spectralrao-monitoring")
import spectralrao


def compute_ndvi_and_rao_on_farm_by_id(b04_filepath,
                                       b08_filepath,
                                       farms,
                                       farm_id,
                                       ndvi_threshold,
                                       window,
                                       na_tolerance,
                                       plot=True):
    """Computes RAO's Q index from the given band rasters limited to one farm polygon

    The farm si selected through it's ID
    """
    #Select farm
    farm = farms[farms['Farm_ID'].astype('int') == farm_id].iloc[0]

    return compute_ndvi_and_rao_on_farm(
        b04_filepath, b08_filepath, farm, ndvi_threshold, window, na_tolerance,
        plot=plot
        )

def compute_ndvi_and_rao_on_farm(b04_filepath,
                                 b08_filepath,
                                 farm,
                                 ndvi_threshold,
                                 window,
                                 na_tolerance,
                                 plot=True):
    """Computes RAO's Q index from the given band rasters limited to one farm polygon.

    The farm is given directly as a shape object"""
    ndvi, transform = utils.compute_ndvi_from_sentinel(b04_filepath, b08_filepath,
                                                       farm["geometry"])
    masked_ndvi = np.where(ndvi > ndvi_threshold, ndvi, np.nan)

    if plot:
        plt.figure()
        plt.imshow(masked_ndvi)
        plt.colorbar()
        plt.show()

    rao = spectralrao.spectralrao(masked_ndvi, ".", window=window,
                                  na_tolerance=na_tolerance)

    if plot:
        plt.figure()
        plt.imshow(rao[0])
        plt.colorbar()
        plt.show()

    return ndvi, rao[0], transform

def get_raster_data_on_subregion(raster_filepath, subregion, plot=False):
    with rasterio.open(raster_filepath) as dataset:
        data, transform = rasterio.mask.mask(dataset=dataset,
                                             shapes=[subregion],
                                             crop=True, indexes=1)

    return data, transform


def get_subregion_from_geojson(shape_filepath):
    with open(shape_filepath) as f:
        features = json.load(f)["features"]
    return shape(features[0]["geometry"])


def write_rao_to_file(rao_array, transform, output_path, crs='EPSG:4326'):
    with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            dtype=rao_array.dtype,
            height=rao_array.shape[0],
            width=rao_array.shape[1],
            count=1,
            crs=crs,
            transform=transform,
            compress='lzw') as dataset:
        dataset.write(rao_array, indexes=1)


def process_subregion(
        mask_index_filepath: str,
        mask_threshold: Union[int, float],
        rao_input_index_filpath: str,
        subregion: Union[
            shapely.geometry.multipolygon.MultiPolygon, shapely.geometry.polygon.Polygon],
        window,
        na_tolerance,
        output_filepath=None,
        plot=False,
        ):
    mask_index, _ = get_raster_data_on_subregion(
        mask_index_filepath, subregion)
    rao_input_index, rao_input_index_transform = get_raster_data_on_subregion(
        rao_input_index_filpath, subregion)
    rao_input_index_masked = np.where(mask_index > mask_threshold, rao_input_index, np.nan)
    rao = spectralrao.spectralrao(rao_input_index_masked, ".",
                                  window=window,
                                  na_tolerance=na_tolerance)[0]
    if plot:
        plt.figure()
        plt.imshow(rao)
        plt.colorbar()
        plt.show()

    if output_filepath is not None:
        write_rao_to_file(rao, rao_input_index_transform, output_filepath)
    return rao


if __name__ == '__main__':
    # b04_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_B04_10m.jp2"
    # b08_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_B08_10m.jp2"
    b04_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20221231T105349_N0509_R051_T29NPH_20221231T133126.SAFE\GRANULE\L2A_T29NPH_A030393_20221231T110338\IMG_DATA\R10m\T29NPH_20221231T105349_B04_10m.jp2"
    b08_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20221231T105349_N0509_R051_T29NPH_20221231T133126.SAFE\GRANULE\L2A_T29NPH_A030393_20221231T110338\IMG_DATA\R10m\T29NPH_20221231T105349_B08_10m.jp2"
    ndfi_filepath = r"D:\Overland\export-ivorycoast2023-spot\20230108_S7_065N0080W\20230108_S7_065N0080W_ndfi.tif"
    lai_filepath = r"D:\Overland\export-ivorycoast2023-spot\20230108_S7_065N0080W\20230108_S7_065N0080W_lai.tif"
    # indicator_raster_filepath = r"D:\Overland\export-ivorycoast2023-spot\20230108_S7_065N0080W\20230108_S7_065N0080W_cshn.tif"
    indicator_raster_filepath = lai_filepath
    vegetation_mask_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\NDFI_vegetation_mask_int16.tif"

    # Get farm shapes
    farms = gpd.read_file(
        r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ivory_farms_shapefile"
        )

    # Change projection to be consistent with Sentinel 2 projection
    # farms.to_crs("EPSG:32629", inplace=True)

    # SPOT 7 projection
    # farms.to_crs("EPSG:4326")

    # SENTINEL 2 Projection


    # Filter farms to be processes
    filtered_farms = farms[
        (farms['BARELAND'] <= 0.2) & (farms['NON_COCOA'] <= 0.2) & (farms['NATURAL'] <= 0.2)
    ]

    # Methodology meta parameters
    window = 11
    na_tolerance = 0.2
    # ndvi_threshold = 0.4
    ndfi_threshold = 6500

    results = pd.DataFrame(columns=["ID", "area", "rao", "full_sun", "shaded", "no_full_sun", "nb_pixel_not_nan"])

    # Select farms subset
    # filtered_farms = filtered_farms[(filtered_farms['Farm_ID'].astype('int')).isin([80, 92, 106])]
    # filtered_farms = filtered_farms[
    #     (filtered_farms['Farm_ID'].astype('int')).isin([80])]

    for i, farm in filtered_farms.iterrows():
        if farm['SHADED'] + farm['FULL_SUN'] > 1:
            print(f"Problem with farm {farm['Farm_ID']}, SHADED and FULL_SUN sum > 1")
            continue

        print(f"Processing farm {i + 1} over {filtered_farms.shape[0]}")

        # Compute NDVI and RAO
        try:
            rao = process_subregion(
                mask_index_filepath=ndfi_filepath,
                mask_threshold=ndfi_threshold,
                rao_input_index_filpath=indicator_raster_filepath,
                subregion=farm["geometry"],
                window=window,
                na_tolerance=na_tolerance
                )
            # ndfi, _ = get_raster_data_on_subregion(
            #     ndfi_filepath, farm["geometry"])
            # spectral_index, index_transform = get_raster_data_on_subregion(indicator_raster_filepath, farm["geometry"])
            # # vegetation_mask, mask_transform = get_raster_data_on_subregion(vegetation_mask_filepath, farm["geometry"])
            # # masked_ndfi = np.where(ndfi > ndfi_threshold, ndfi, np.nan)
            # masked_index = np.where(ndfi > ndfi_threshold, spectral_index, np.nan)
            # rao = spectralrao.spectralrao(masked_index, ".",
            #                               window=window,
            #                               na_tolerance=na_tolerance)[0]
            # # ndvi, rao, transform = compute_ndvi_and_rao_on_farm(
            # #     b04_filepath, b08_filepath, farm, ndvi_threshold, window, na_tolerance,
            # #     plot=False
            # #     )
        except ValueError as e:
            print(f"Farm {farm['Farm_ID']} not processed")
            print(e)
            continue
        else:
            print(f"Farm {farm['Farm_ID']} has been successfully processed")

            # Store results for the current farm
            area = farm['geometry'].area
            results = pd.concat(
                [results,
                 pd.DataFrame(
                     [
                         [
                             farm['Farm_ID'],
                             area,
                             np.nanmean(rao),
                             farm['FULL_SUN'],
                             farm['SHADED'],
                             farm['SHADED'] + farm['NON_COCOA'] + farm['NATURAL'],
                             np.sum(~np.isnan(rao))
                             ]
                      ],
                     columns=results.columns
                     )
                 ]
                )

    # Plot relation between shaded parameter and RAO's Q
    plt.figure()
    sns.scatterplot(results, x='shaded', y='rao')
    plt.show()

    # If we want to check a specific farm. Execute this code
    # selected_farm_id = 92
    # farm = filtered_farms[filtered_farms['Farm_ID'].astype('int') == selected_farm_id].iloc[0]
    # subregion = farm["geometry"]
    # OR
    subregion = get_subregion_from_geojson(r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ROI\sub_aoi_agroforestry_vs_full_sun.geojson")
    # # ndvi, rao, transform = compute_ndvi_and_rao_on_farm_by_id(
    # #                 b04_filepath, b08_filepath, farms, selected_farm_id, ndvi_threshold, window, na_tolerance,
    # #                 plot=True
    # #                 )
    # # or
    #
    process_subregion(
        mask_index_filepath=ndfi_filepath,
        mask_threshold=ndfi_threshold,
        rao_input_index_filpath=indicator_raster_filepath,
        window=window,
        na_tolerance=na_tolerance,
        subregion=subregion,
        output_filepath="./rao_agroforestery_subregion_1_lai_ndfi_mask_w_11_na_0_2.tif",
        plot=True
        )

    subregion = get_subregion_from_geojson(
        r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ROI\sub_aoi_agroforestry_vs_full_sun_3_east_sentinel_proj.geojson")
    ndvi, rao, transform = compute_ndvi_and_rao_on_farm(b04_filepath,
                                                        b08_filepath,
                                                        {"geometry": subregion},
                                                        ndvi_threshold=0.4,
                                                        window=11,
                                                        na_tolerance=0.3,
                                                        plot=True)
    write_rao_to_file(rao, transform, "./rao_agroforestery_subregion_3_east_sentinel_ndvi_ndvi_mask_0_4_w_11_na_0_3.tif",
                      crs="EPSG:32629")
    # ndfi, _ = get_raster_data_on_subregion(
    #     ndfi_filepath, subregion)
    # spectral_index, index_transform = get_raster_data_on_subregion(
    #     indicator_raster_filepath, subregion)
    # masked_index = np.where(ndfi > ndfi_threshold, spectral_index, np.nan)
    # rao = spectralrao.spectralrao(masked_index, ".",
    #                               window=window,
    #                               na_tolerance=na_tolerance)[0]
    # plt.figure()
    # plt.imshow(rao)
    # plt.colorbar()
    # plt.show()
    #
    # write_rao_to_file(rao, index_transform, "./rao_agroforestry_subregion_2_east.tif")

    #
    # # To plot the RGB image on the same farm we can execute the following code
    # from spaps_eo_project import utils
    # true_colors_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_TCI_10m.jp2"
    # farm = farms[farms['Farm_ID'].astype('int') == selected_farm_id].iloc[0]
    # utils.plot_raster_on_subregion(true_colors_filepath, farm['geometry'])