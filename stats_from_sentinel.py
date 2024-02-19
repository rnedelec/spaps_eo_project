import sys
from spaps_eo_project import utils
import geopandas as gpd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(r"../../spectralrao-monitoring")
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


if __name__ == '__main__':
    b04_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_B04_10m.jp2"
    b08_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_B08_10m.jp2"

    # Get farm shapes
    farms = gpd.read_file(
        r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ivory_farms_shapefile")

    # Change projection to be consistent with Sentinel 2 projection
    farms.to_crs("EPSG:32629", inplace=True)

    # Filter farms to be processes
    filtered_farms = farms[
        (farms['BARELAND'] <= 0.1) & (farms['NON_COCOA'] <= 0.1) & (farms['NATURAL'] <= 0.1)
    ]

    # Methodology meta parameters
    window = 3
    na_tolerance = 0.3
    ndvi_threshold = 0.4

    results = pd.DataFrame(columns=["ID", "rao", "full_sun", "shaded", "no_full_sun"])
    for i, farm in filtered_farms.iterrows():
        if farm['SHADED'] + farm['FULL_SUN'] > 1:
            print(f"Problem with farm {farm['Farm_ID']}, SHADED and FULL_SUN sum > 1")

        # Compute NDVI and RAO
        try:
            ndvi, rao, transform = compute_ndvi_and_rao_on_farm(
                b04_filepath, b08_filepath, farm, ndvi_threshold, window, na_tolerance,
                plot=False
                )
        except ValueError as e:
            print(f"Farm {farm['Farm_ID']} not processed")
            print(e)
            continue
        else:
            print(f"Farm {farm['Farm_ID']} has been successfully processed")

            # Store results for the current farm
            results = pd.concat(
                [results,
                 pd.DataFrame(
                     [
                         [
                             farm['Farm_ID'],
                             np.nanmean(rao),
                             farm['FULL_SUN'],
                             farm['SHADED'],
                             farm['SHADED'] + farm['NON_COCOA'] + farm['NATURAL']
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
    # selected_farm_id = 165
    # ndvi, rao, transform = compute_ndvi_and_rao_on_farm_by_id(
    #                 b04_filepath, b08_filepath, farms, selected_farm_id, ndvi_threshold, window, na_tolerance,
    #                 plot=True
    #                 )

    # To plot the RGB image on the same farm we can execute the following code
    # from spaps_eo_project import utils
    # true_colors_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_TCI_10m.jp2"
    # farm = farms[farms['Farm_ID'].astype('int') == selected_farm_id].iloc[0]
    # utils.plot_raster_on_subregion(true_colors_filepath, farm['geometry'])