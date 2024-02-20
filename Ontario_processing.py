import sys
import rasterio
import utils
import geopandas as gpd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spectralrao
import math
from typing import Tuple, List

def process_species_comp(comp_string: str) -> List[Tuple[str, str]]:
    """Deserialize species composition into a list of tuples"""
    n = 6  ## Nb of characters in line blocks
    blocks = [comp_string[i:i + n] for i in range(0, len(comp_string), n)]
    species_comp = []
    for block in blocks:
        species = block[:3].strip()
        cover_percentage = block[3:].strip()
        species_comp.append((species, cover_percentage))
    return species_comp

def compute_shannon(comp_string: str) -> float:
    """Computes Shannon index from deserialized species composition"""
    species_comp = process_species_comp(comp_string)
    shannon = 0
    for composition in species_comp:
        cover_percentage = composition[1]
        proportion = float(cover_percentage) / 100
        if proportion > 0:
            shannon -= proportion * math.log(proportion)
    return shannon

def compute_ndvi_and_rao_on_tree_by_id(b04_filepath,
                                       b08_filepath,
                                       data,
                                       tree_id,
                                       ndvi_threshold,
                                       window,
                                       na_tolerance,
                                       plot=True):
    """Computes RAO's Q index from the given band rasters limited to one tree polygon

    The tree is selected through its ID
    """
    #Select tree
    tree = data[data['FMFOBJID'].astype('int') == tree_id].iloc[0]

    return compute_ndvi_and_rao_on_tree(
        b04_filepath, b08_filepath, tree, ndvi_threshold, window, na_tolerance,
        plot=plot
        )

def compute_ndvi_and_rao_on_tree(b04_filepath,
                                 b08_filepath,
                                 tree,
                                 ndvi_threshold,
                                 window,
                                 na_tolerance,
                                 plot=True):
    """Computes RAO's Q index from the given band rasters limited to one tree polygon.

    The tree is given directly as a shape object"""
    ndvi, transform = utils.compute_ndvi_from_sentinel(b04_filepath, b08_filepath,
                                                       tree["geometry"])
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
    b04_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/S2B_MSIL1C_20240203T162459_N0510_R040_T17UMP_20240203T182319.SAFE/GRANULE/L1C_T17UMP_A036102_20240203T162501/IMG_DATA/T17UMP_20240203T162459_B04.jp2"
    b08_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/S2B_MSIL1C_20240203T162459_N0510_R040_T17UMP_20240203T182319.SAFE/GRANULE/L1C_T17UMP_A036102_20240203T162501/IMG_DATA/T17UMP_20240203T162459_B08.jp2"

    data = gpd.read_file(
        r"/Users/retif/Desktop/SUPAERO/Tutored Project/pp_FRI_FIMv2_Martel_Forest(509)_2015_2D.gdb",
        layer="MAR_FRI_2D")

    ## Inspecting species composition
    if 'SPCOMP' in data:
        species_keys = ['SPCOMP']
    elif 'OSPCOMP' and 'USPCOMP' in data:
        species_keys = ['OSPCOMP', 'USPCOMP']

    print("Columns in data GeoDataFrame:", data.columns)

    forest = data[data['POLYTYPE'] == 'FOR'].copy()

    ## Change projection to be consistent with Sentinel 2 projection (Ontario)
    with rasterio.open(b08_filepath) as src:
        crs_S2 = src.crs
    forest.to_crs(crs=crs_S2, inplace=True)

    calib_plots = forest[forest['SOURCE'] == 'PLOTVAR']

    # Methodology meta parameters
    window = 3
    na_tolerance = 0.3
    ndvi_threshold = 0.4

    results = pd.DataFrame(columns=["ID", "rao", "shannon"])
    for i, tree in calib_plots.iterrows():
        # Compute NDVI & RAO
        try:
            ndvi, rao, transform = compute_ndvi_and_rao_on_tree(
                b04_filepath, b08_filepath, tree, ndvi_threshold, window, na_tolerance,
                plot=False
            )
            if np.isnan(np.nanmean(rao)):
                print(f"Tree {tree['FMFOBJID']} has empty intersection, skipping...")
                continue
            forest = data[data['POLYTYPE'] == 'FOR'].copy()
            forest['Shannon'] = forest[species_keys[0]].map(compute_shannon)
        except ValueError as e:
            print(f"Tree {tree['FMFOBJID']} not processed")
            print(e)
            continue
        else:
            print(f"Tree {tree['FMFOBJID']} has been successfully processed")
            results = pd.concat(
                [results,
                 pd.DataFrame(
                     [
                         [tree['FMFOBJID'], np.nanmean(rao), forest['Shannon']]
                     ],
                     columns=results.columns
                 )]
            )

    # Plot relation between shannon and RAO's Q
    plt.figure()
    sns.scatterplot(results, x='shannon', y='rao')
    plt.show()

    # If we want to check a specific tree. Execute this code
    # selected_tree_id = 165
    # ndvi, rao, transform = compute_ndvi_and_rao_on_tree_by_id(
    #                 b04_filepath, b08_filepath, trees, selected_tree_id, ndvi_threshold, window, na_tolerance,
    #                 plot=True
    #                 )

    # To plot the RGB image on the same tree we can execute the following code
    # from spaps_eo_project import utils
    # true_colors_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_TCI_10m.jp2"
    # tree = trees[trees['tree_ID'].astype('int') == selected_tree_id].iloc[0]
    # utils.plot_raster_on_subregion(true_colors_filepath, tree['geometry'])