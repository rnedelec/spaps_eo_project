import sys
import json
from typing import Union

import scipy
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

    The farm is selected through it's ID
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
        upper_threshold=False,
        output_filepath=None,
        plot=False,
        ):
    """Filters out bare soil pixels and computes the Rao's Q indicator on a given subregion

    Parameters
    ----------
    mask_index_filepath: str
        Path of the index raster to be used for soil filtering (vegetation mask)
    mask_threshold: int or float
        Threshold to be used for the vegetation mask
    rao_input_index_filepath: str
        Path of the index raster to be used as input of the Rao's Q index
    subregion: shapely Polygon or Multipolygon
        Subregion on which to compute Rao's Q index
    window: int
        Size of the window in Rao's Q computation
    na_tolerance: float
        NaN tolerance in Rao's Q computation
    upper_threshold: bool, optional
        True if the mask threshold is an upper threshold
    output_filepath: str, optional
        Path where the outputs Rao's Q will be stored
    plot: bool, optional
        If True the masked input index and the output will be plotted

    Returns
    -------
    numpy.array
        Rao's Q indicator computed on the given subregion
    """
    mask_index, _ = get_raster_data_on_subregion(
        mask_index_filepath, subregion)
    rao_input_index, rao_input_index_transform = get_raster_data_on_subregion(
        rao_input_index_filpath, subregion)
    if upper_threshold:
        rao_input_index_masked = np.where(
            (mask_index <= mask_threshold) & (mask_index > 0), rao_input_index, np.nan
            )
    else:
        rao_input_index_masked = np.where(
            mask_index > mask_threshold, rao_input_index, np.nan
            )
    rao = spectralrao.spectralrao(rao_input_index_masked, ".",
                                  window=window,
                                  na_tolerance=na_tolerance)[0]
    if plot:
        # Plot masked input
        plt.figure()
        plt.imshow(rao_input_index_masked)
        plt.colorbar()
        plt.show()

        # PLot RAO
        plt.figure()
        plt.imshow(rao)
        plt.colorbar()
        plt.show()

    if output_filepath is not None:
        write_rao_to_file(rao, rao_input_index_transform, output_filepath)
    return rao


if __name__ == '__main__':
    # # NPH Tile Sentinel 2
    # product_name = "sentinel_nph_overland"
    # scv_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH\20221231_S2B_T29NPH_scv.tif"
    # lai_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH\20221231_S2B_T29NPH_lai.tif"
    # cshn_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH\20221231_S2B_T29NPH_cshn.tif"
    # ndfi_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH\20221231_S2B_T29NPH_ndfi.tif"
    # # ndvi_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH\20221231_S2B_T29NPH_rdhz_ndvi_spaps_team.tif"
    # ndvi_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH\20221231_S2B_T29NPH_ndvi_2.tif"

    # NNH Tile Sentinel 2
    product_name = "sentinel_nnh_overland"
    scv_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NNH\20221231_S2B_T29NNH_scv.tif"
    lai_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NNH\20221231_S2B_T29NNH_lai.tif"
    cshn_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NNH\20221231_S2B_T29NNH_cshn.tif"
    ndfi_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NNH\20221231_S2B_T29NNH_ndfi.tif"

    # # SPOT 7
    # product_name = "spot_7"
    # scv_filepath = r"D:\Overland\export-ivorycoast2023-spot\20230108_S7_065N0080W\20230108_S7_065N0080W_scv.tif"
    # cshn_filepath = r"D:\Overland\export-ivorycoast2023-spot\20230108_S7_065N0080W\20230108_S7_065N0080W_cshn.tif"

    # Get farm shapes
    farms = gpd.read_file(
        r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ivory_farms_shapefile"
        )

    # Change projection to be consistent with Sentinel 2 projection
    # farms.to_crs("EPSG:32629", inplace=True)

    # SPOT 7 projection
    # farms.to_crs("EPSG:4326")

    # Filter farms to be processed
    farms_filter_threshold = 0.1
    filtered_farms = farms[
        (farms['NON_COCOA'] <= farms_filter_threshold) &
        (farms['NATURAL'] <= farms_filter_threshold)
    ]

    # Methodology meta parameters
    window = 5
    # window = 11
    na_tolerance = 0.3
    ndfi_threshold = 5700
    # scv_threshold = 1100
    scv_threshold = 850

    results = pd.DataFrame(
        columns=["ID", "area", "rao", "full_sun", "shaded",
                 "bareland", "natural", "non_cocoa",
                 "shaded_prop_in_no_bareland", "nb_pixel_not_nan"]
        )

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
                mask_index_filepath=scv_filepath,
                mask_threshold=scv_threshold,
                rao_input_index_filpath=cshn_filepath,
                subregion=farm["geometry"],
                upper_threshold=True,
                window=window,
                na_tolerance=na_tolerance
                )
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
                             farm['BARELAND'],
                             farm['NATURAL'],
                             farm['NON_COCOA'],
                             farm['SHADED'] / (1 - farm['BARELAND']),
                             np.sum(~np.isnan(rao))
                             ]
                      ],
                     columns=results.columns
                     )
                 ]
                )

    # Save results
    results.to_csv(
        f"{product_name}_cshn_mask_scv_{scv_threshold}_w_{window}_na_{str(na_tolerance).replace('.', '_')}"
        f"_farms_filtered_{str(farms_filter_threshold).replace('.', '_')}_no_bareland_filter_full_labels.csv"
        )

    # Filter data and evaluate the regression
    results_nan_filtered = results[~results['rao'].isna()]
    # results_filtered_2500_px = results_nan_filtered[results_nan_filtered['nb_pixel_not_nan'] > 2500]
    results_filtered_100_px = results_nan_filtered[
        results_nan_filtered['nb_pixel_not_nan'] > 100]

    # Pearson coefficient
    print("All results, shaded param : ")
    print(scipy.stats.pearsonr(results_nan_filtered['shaded'], results_nan_filtered['rao']))
    print("100 px filtered, shaded param : ")
    print(scipy.stats.pearsonr(results_filtered_100_px['shaded'],
                               results_filtered_100_px['rao']))

    print("All results, shaded no bareland param : ")
    print(scipy.stats.pearsonr(results_nan_filtered['shaded_prop_in_no_bareland'], results_nan_filtered['rao']))
    print("100 px filtered, shaded no bareland param : ")
    print(scipy.stats.pearsonr(results_filtered_100_px['shaded_prop_in_no_bareland'],
                               results_filtered_100_px['rao']))

    # Plot relation between shaded parameter and RAO's Q
    plt.figure()
    sns.regplot(results, x='shaded_prop_in_no_bareland', y='rao')
    plt.title("Regression for all farms")
    plt.show()

    plt.figure()
    sns.regplot(results_filtered_100_px, x='shaded_prop_in_no_bareland', y='rao')
    # plt.title("Regression for farms with more than 100 pixels in RAO index")
    plt.show()

    #
    # # To plot the RGB image on the same farm we can execute the following code
    # from spaps_eo_project import utils
    # true_colors_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_TCI_10m.jp2"
    # farm = farms[farms['Farm_ID'].astype('int') == selected_farm_id].iloc[0]
    # utils.plot_raster_on_subregion(true_colors_filepath, farm['geometry'])