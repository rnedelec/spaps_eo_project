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
from typing import Tuple, List, Union
import shapely
import rasterio.mask
import json
from shapely.geometry import shape

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

def compute_ndvi_and_rao_on_calib_plot_by_id(red_filepath,
                                       nir_filepath,
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
    tree = data[data['POLYID'] == tree_id].iloc[0]

    return compute_ndvi_and_rao_on_calib_plot(
        red_filepath, nir_filepath, tree, ndvi_threshold, window, na_tolerance,
        plot=plot
        )

def compute_ndvi_and_rao_on_calib_plot(red_filepath,
                                 nir_filepath,
                                 tree,
                                 ndvi_threshold,
                                 window,
                                 na_tolerance,
                                 plot=True):
    """Computes RAO's Q index from the given band rasters limited to one tree polygon.

    The tree is given directly as a shape object"""
    ndvi, transform = utils.compute_ndvi_from_sentinel(red_filepath, nir_filepath,
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
        upper_threshold = False,
        output_filepath=None,
        plot=False,
        ):
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

if __name__ == '__main__':

    ## SPOT6
    # red_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario/Reflectance/20150712_S6_475N0835W_rdhz/20150712_S6_475N0835W_rdhz_b3.tif"
    # nir_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario/Reflectance/20150712_S6_475N0835W_rdhz/20150712_S6_475N0835W_rdhz_b4.tif"
    # ndfi_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario/20150712_S6_475N0835W/20150712_S6_475N0835W_ndfi.tif"
    # lai_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario/20150712_S6_475N0835W/20150712_S6_475N0835W_lai.tif"
    # cshn_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario/20150712_S6_475N0835W/20150712_S6_475N0835W_cshn.tif"
    #
    # mask_index_filepath = ndfi_filepath
    # indicator_raster_filepath = cshn_filepath

    ## LANDSAT8
    red_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario-landsat8/Reflectance/20170816_L8_021027_rdhz/20170816_L8_021027_rdhz_b4.tif"
    nir_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario-landsat8/Reflectance/20170816_L8_021027_rdhz/20170816_L8_021027_rdhz_b5.tif"
    ndfi_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario-landsat8/20170816_L8_021027/20170816_L8_021027_ndfi.tif"
    lai_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario-landsat8/20170816_L8_021027/20170816_L8_021027_lai.tif"
    ndvi_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario-landsat8/20170816_L8_021027/20170816_L8_021027_ndvi.tif"
    cshn_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario-landsat8/20170816_L8_021027/20170816_L8_021027_cshn.tif"
    scv_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/export-ontario-landsat8/20170816_L8_021027/20170816_L8_021027_scv.tif"

    mask_index_filepath = ndfi_filepath
    indicator_raster_filepath = cshn_filepath

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
  #  forest['geometry'] = forest['geometry'].apply(lambda geom: geom.convex_hull)

    # Convertir MultiPolygons en Polygons
    forest['geometry'] = forest['geometry'].apply(lambda geom: geom.geoms[0] if geom.is_empty else geom)

    # Change projection
    # new_epsg_code = 32616 ## Sentinel 2
    new_epsg_code = 4326 ## SPOT7 / LANDSAT8

    forest.to_crs(crs=new_epsg_code, inplace=True)
    print("CRS forest :", forest.crs)

    calib_plots = forest[forest['SOURCE'] == 'PLOTVAR']
    print("calib plots", calib_plots)
    print("Nombre de polygones dans calib plots:", calib_plots.shape[0])

    # Read the bounding box of the red file
    with rasterio.open(red_filepath) as red_ds:
        bounds = red_ds.bounds

    # Create a bounding box as a Shapely Polygon
    aoi_bbox = gpd.GeoDataFrame(geometry=[utils.bbox_to_polygon(bounds)], crs=red_ds.crs)

    # Select calib_plots within the bounding box
    calib_plots_within_aoi = gpd.sjoin(calib_plots, aoi_bbox, how='inner', op='intersects')
    print("Nombre de polygones dans calib plots within aoi:", calib_plots_within_aoi.shape[0])

    # Take a random sample of n calib_plots within the AOI
    calib_plots_subset = calib_plots_within_aoi.sample(n=30, random_state=42)

    print("Subset of calib plots within AOI:", calib_plots_subset)
    print("Nombre de polygones dans le subset de calib plots:", calib_plots_subset.shape[0])

    # Methodology meta parameters
    window = 3
    na_tolerance = 0.4
    ndvi_threshold = 0.82
    ndfi_threshold = 8750
    #ndfi_threshold = 6000
    scv_threshold = 550

    results = pd.DataFrame(columns=["ID", "rao", "shannon", "area"])
    flag = 0

    for i, calib_plot in calib_plots_subset.iterrows():
        flag += 1
        print(f"Calib plot number {flag} over {calib_plots_subset.shape[0]}")
        # Compute NDVI & RAO
        try:
            rao = process_subregion(
                mask_index_filepath=mask_index_filepath,
                mask_threshold=ndfi_threshold,
                rao_input_index_filpath=indicator_raster_filepath,
                subregion=calib_plot["geometry"],
                window=window,
                na_tolerance=na_tolerance
            )
            if np.isnan(np.nanmean(rao)):
                print(f"Calib_plot {calib_plot['POLYID']} has empty intersection, skipping...")
                continue
            shannon_value = compute_shannon(calib_plot[species_keys[0]])
        except ValueError as e:
            print(f"Calib_plot {calib_plot['POLYID']} not processed")
            print(e)
            continue
        else:
            print(f"Calib_plot {calib_plot['POLYID']} has been successfully processed")
            results = pd.concat(
                [results,
                 pd.DataFrame(
                     [
                         [calib_plot['POLYID'], np.nanmean(rao), shannon_value, calib_plot['AREA']]
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
    plt.text(0.1, 0.85, f'p-value: {p_value:.5f}', transform=plt.gca().transAxes)
    legend_text = f"Window={window}, NA Tolerance={na_tolerance}, NDVI Threshold={ndvi_threshold}"
    plt.legend([legend_text, 'Regression Line'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.show()

    # Plot relation between AREA and RAO's Q
    plt.figure()
    sns.scatterplot(data=results, x='area', y='rao')

    # Fit a linear regression line for AREA vs RAO
    slope_area, intercept_area, r_value_area, p_value_area, std_err_area = stats.linregress(results['area'],
                                                                                            results['rao'])
    # Plot the regression line for AREA vs RAO
    plt.plot(results['area'], intercept_area + slope_area * results['area'], color='blue', label='Area Regression Line')

    # Calculate R-squared for AREA vs RAO
    r_squared_area = r_value_area ** 2

    # Calculate Pearson correlation coefficient for AREA vs RAO
    pearson_corr_area = results['area'].corr(results['rao'])

    # Display the plot
    plt.title('Area vs RAO Regression')
    plt.legend()

    # Display additional information
    plt.text(0.1, 0.95, f'R-squared: {r_squared_area:.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.90, f'Pearson corr: {pearson_corr_area:.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.85, f'p-value: {p_value_area:.4f}', transform=plt.gca().transAxes)

    plt.show()

    # Print R-squared, Pearson correlation coefficient, and p-value for AREA vs RAO
    print(f'R-squared (Area vs RAO): {r_squared_area:.4f}')
    print(f'Pearson correlation coefficient (Area vs RAO): {pearson_corr_area:.4f}')
    print(f'p-value (Area vs RAO): {p_value_area:.4f}')

    # Plot relation between AREA and Shannon Index
    plt.figure()
    sns.scatterplot(data=results, x='area', y='shannon')

    # Fit a linear regression line for AREA vs Shannon Index
    slope_area_shannon, intercept_area_shannon, r_value_area_shannon, p_value_area_shannon, std_err_area_shannon = stats.linregress(
        results['area'], results['shannon'])

    # Plot the regression line for AREA vs Shannon Index
    plt.plot(results['area'], intercept_area_shannon + slope_area_shannon * results['area'], color='green',
             label='Area vs Shannon Regression Line')

    # Calculate R-squared for AREA vs Shannon Index
    r_squared_area_shannon = r_value_area_shannon ** 2

    # Calculate Pearson correlation coefficient for AREA vs Shannon Index
    pearson_corr_area_shannon = results['area'].corr(results['shannon'])

    # Display the plot
    plt.title('Area vs Shannon Index Regression')
    plt.legend()

    # Display additional information
    plt.text(0.1, 0.95, f'R-squared: {r_squared_area_shannon:.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.90, f'Pearson corr: {pearson_corr_area_shannon:.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.85, f'p-value: {p_value_area_shannon:.4f}', transform=plt.gca().transAxes)

    plt.show()

    # Print R-squared, Pearson correlation coefficient, and p-value for AREA vs Shannon Index
    print(f'R-squared (Area vs Shannon Index): {r_squared_area_shannon:.4f}')
    print(f'Pearson correlation coefficient (Area vs Shannon Index): {pearson_corr_area_shannon:.4f}')
    print(f'p-value (Area vs Shannon Index): {p_value_area_shannon:.4f}')

selected_calib_plot_id = '172905260-0177'
calib_plot = forest[forest['POLYID'] == selected_calib_plot_id].iloc[0]
subregion = calib_plot["geometry"]

rao = process_subregion(
    mask_index_filepath=ndfi_filepath,
    mask_threshold=ndfi_threshold,
    rao_input_index_filpath=indicator_raster_filepath,
    window=window,
    na_tolerance=na_tolerance,
    subregion=subregion,
    output_filepath=f"./calib_plot_0177_landsat_ndfi_{ndfi_threshold}_cshn_window_{window}_nan_tol_{na_tolerance}.tif",
    plot=True
)

  #  write_rao_to_file(rao, transform, "./rao_ontario_ndvi_ndvi_mask_0_82_w_11_na_0_4.tif",
  #                    crs="EPSG:4326")