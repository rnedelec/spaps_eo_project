import sys
from spaps_eo_project import utils
import geopandas as gpd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(r"../spectralrao-monitoring")
import spectralrao

if __name__ == '__main__':
    b04_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_B04_10m.jp2"
    b08_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_B08_10m.jp2"

    farms = gpd.read_file(
        r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ivory_farms_shapefile")

    # # Statistics for farms with few bareland
    # filtered_farms = farms[(farms['BARELAND'] <= 0.1) & (farms['NON_COCOA'] <= 0.1)]

    # selected_farm_id = 221
    selected_farm_id = 165
    window = 3
    na_tolerance = 0.4
    ndvi_threshold = 0.43

    # Change projection
    farms.to_crs("EPSG:32629", inplace=True)

    # Select farm
    # farm = farms[farms['Farm_ID'].astype('int') == selected_farm_id].iloc[0]
    # ndvi, transform = utils.compute_ndvi_from_sentinel(b04_filepath, b08_filepath,
    #                                                    farm["geometry"])
    # masked_ndvi = np.where(ndvi > ndvi_threshold, ndvi, np.nan)
    # plt.figure()
    # plt.imshow(masked_ndvi)
    # plt.colorbar()
    # plt.show()
    #
    # rao = spectralrao.spectralrao(masked_ndvi, ".", window=window,
    #                               na_tolerance=na_tolerance)
    #
    # plt.figure()
    # plt.imshow(rao[0])
    # plt.colorbar()
    # plt.show()

    results = pd.DataFrame(columns=["ID", "rao", "full_sun", "shaded", "no_full_sun"])
    for i, farm in farms[(farms['BARELAND'] <= 0.1) & (farms['NON_COCOA'] <= 0.1) & (farms['NATURAL'] <= 0.1)].iterrows():
        if farm['SHADED'] + farm['FULL_SUN'] > 1:
            print(f"Problem with farm {farm['Farm_ID']}, SHADED and FULL_SUN sum > 1")
        # print(farm)
        # Compute NDVI
        try:
            ndvi, transform = utils.compute_ndvi_from_sentinel(b04_filepath, b08_filepath, farm["geometry"])
        except ValueError:
            print(f"Farm {farm['Farm_ID']} not processed")
            continue
        else:
            print(f"Farm {farm['Farm_ID']} has been successfully processed")
            # plt.figure()
            # plt.imshow(ndvi)
            # plt.colorbar()
            # plt.show()

            masked_ndvi = np.where(ndvi > ndvi_threshold, ndvi, np.nan)
            rao = spectralrao.spectralrao(masked_ndvi, ".", window=window,
                                          na_tolerance=na_tolerance)
            print(f"Mean RAO = {np.nanmean(rao[0])}")
            print(f"Farm : {farm[['Farm_ID', 'SHADED', 'FULL_SUN']]}")
            results = pd.concat(
                [results,
                 pd.DataFrame(
                     [
                         [
                             farm['Farm_ID'],
                             np.nanmean(rao[0]),
                             farm['FULL_SUN'],
                             farm['SHADED'],
                             farm['SHADED'] + farm['NON_COCOA'] + farm['NATURAL']
                             ]
                      ],
                     columns=results.columns
                     )
                 ]
                )
            # plt.figure()
            # plt.imshow(rao[0])
            # plt.colorbar()
            # plt.show()
            # break

    plt.figure()
    # sns.scatterplot(results[(results['full_sun'] > 0) & (results['shaded'] > 0.2)],
    #                 x='shaded', y='rao')
    sns.scatterplot(results, x='shaded', y='rao')
    plt.show()