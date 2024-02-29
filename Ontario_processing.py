import sys
import rasterio
from scipy import stats
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
    tree = data[data['POLYID'].astype('int') == tree_id].iloc[0]

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
    b04_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/S2A_MSIL1C_20231213T163701_N0510_R083_T16TGT_20231213T182817.SAFE/GRANULE/L1C_T16TGT_A044267_20231213T163701/IMG_DATA/T16TGT_20231213T163701_B04.jp2"
    b08_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/S2A_MSIL1C_20231213T163701_N0510_R083_T16TGT_20231213T182817.SAFE/GRANULE/L1C_T16TGT_A044267_20231213T163701/IMG_DATA/T16TGT_20231213T163701_B08.jp2"

    data = gpd.read_file(
        r"/Users/retif/Desktop/SUPAERO/Tutored Project/pp_FRI_FIMv2_Martel_Forest(509)_2015_2D.gdb",
        layer="MAR_FRI_2D")

    ## Inspecting species composition
    if 'SPCOMP' in data:
        species_keys = ['SPCOMP']
    elif 'OSPCOMP' and 'USPCOMP' in data:
        species_keys = ['OSPCOMP', 'USPCOMP']

    forest = data[data['POLYTYPE'] == 'FOR'].copy()

    # Simplifier les géométries du GeoDataFrame
    forest['geometry'] = forest['geometry'].apply(lambda geom: geom.convex_hull)

    # Convertir MultiPolygons en Polygons
    forest['geometry'] = forest['geometry'].apply(lambda geom: geom.geoms[0] if geom.is_empty else geom)

    # Change projection
    new_epsg_code = 32616

    forest.to_crs(crs=new_epsg_code, inplace=True)
    print("CRS forest :", forest.crs)

    calib_plots = forest[forest['SOURCE'] == 'PLOTVAR']
    print("calib plots", calib_plots)
    # calib_plots = calib_plots.head(10)
    # print("calib plot subset", calib_plots)

    # Methodology meta parameters
    window = 7
    na_tolerance = 0.4
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
                print(f"Tree {tree['POLYID']} has empty intersection, skipping...")
                continue
            shannon_value = compute_shannon(tree[species_keys[0]])
        except ValueError as e:
            print(f"Tree {tree['POLYID']} not processed")
            print(e)
            continue
        else:
            print(f"Tree {tree['POLYID']} has been successfully processed")
            results = pd.concat(
                [results,
                 pd.DataFrame(
                     [
                         [tree['POLYID'], np.nanmean(rao), shannon_value]
                     ],
                     columns=results.columns
                 )]
            )

    # Plot relation between Shannon and RAO's Q
    plt.figure()
    sns.scatterplot(data=results, x='shannon', y='rao')

    # Fit a linear regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(results['shannon'], results['rao'])

    # Plot the regression line
    plt.plot(results['shannon'], intercept + slope * results['shannon'], color='red', label='Regression Line')

    # Calculate R-squared
    r_squared = r_value ** 2
    print(f'R-squared: {r_squared:.4f}')

    # Calculate Pearson correlation coefficient
    pearson_corr = results['shannon'].corr(results['rao'])
    print(f'Pearson correlation coefficient: {pearson_corr:.4f}')

    # Display the plot
    plt.text(0.1, 0.95, f'R-squared: {r_squared:.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.90, f'Pearson corr: {pearson_corr:.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.85, f'p-value: {p_value:.4f}', transform=plt.gca().transAxes)
    legend_text = f"Window={window}, NA Tolerance={na_tolerance}, NDVI Threshold={ndvi_threshold}"
    plt.legend([legend_text, 'Regression Line'])
    plt.show()

    ## DEBUG

    # Avant le changement de CRS
    print("CRS GeoDataFrame avant le changement :", data.crs)
    print("CRS B04 avant le changement :", rasterio.open(b04_filepath).crs)
    print("CRS B08 avant le changement :", rasterio.open(b08_filepath).crs)

    # Après le changement de CRS
    print("CRS GeoDataFrame après le changement :", forest.crs)
    print("CRS B04 après le changement :", rasterio.open(b04_filepath).crs)
    print("CRS B08 après le changement :", rasterio.open(b08_filepath).crs)

    # Vérification des coordonnées spatiales des formes géométriques
    print("Extent GeoDataFrame :", forest.total_bounds)
    print("Extent B04 :", rasterio.open(b04_filepath).bounds)
    print("Extent B08 :", rasterio.open(b08_filepath).bounds)

    print("Quelques lignes du GeoDataFrame :", forest.head())

    print("Géométrie de la première ligne du GeoDataFrame :", forest.geometry.iloc[0])

    print(results.head())

    # ## If we want to check a specific tree. Execute this code
    # selected_tree_id = '173205270-0747'
    # ndvi, rao, transform = compute_ndvi_and_rao_on_tree_by_id(
    #                 b04_filepath, b08_filepath, data, selected_tree_id, ndvi_threshold, window, na_tolerance,
    #                 plot=True
    #                 )
    #
    # ## To plot the RGB image on the same tree we can execute the following code
    # import utils
    # true_colors_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/S2A_MSIL1C_20231213T163701_N0510_R083_T16TGT_20231213T182817.SAFE/GRANULE/L1C_T16TGT_A044267_20231213T163701/IMG_DATA/T16TGT_20231213T163701_TCI.jp2"
    # tree = data[data['POLYID'].astype('int') == selected_tree_id].iloc[0]
    # utils.plot_raster_on_subregion(true_colors_filepath, tree['geometry'])