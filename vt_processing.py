import os
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import math
from typing import Tuple, List


def process_species_comp(comp_string: str) -> List[Tuple[str, str]]:
    """Deserialize species composition into list of tuples"""
    n = 6  # Nb of characters in line blocs
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


if __name__ == '__main__':
    # data = gpd.read_file(os.environ["data_dir"] + r"\RM_FRI_SampleSet.gdb")
    # data = gpd.read_file(r"C:\Users\Renaud\SPAPS\Team project\RMF_Predictors_20m_SampleSet.gdb")
    data = gpd.read_file(
        r"C:\Users\Renaud\SPAPS\Team project\pp_FRI_FIMv2_Martel_Forest(509)_2015_2D.gdb",
        layer="MAR_FRI_2D")

    # Inspecting species composition
    if 'SPCOMP' in data:
        species_keys = ['SPCOMP']

    elif 'OSPCOMP' and 'USPCOMP' in data:
        species_keys = ['OSPCOMP', 'USPCOMP']

    # Printing first rows for forest polygons
    data[data['POLYTYPE'] == 'FOR'][species_keys].head()

    # Plotting compositions
    data[data['POLYTYPE'] == 'FOR'].plot(column=species_keys[0])
    plt.show()

    # Calibration plots
    data[data['SOURCE'] == 'PLOTVAR'].plot(column=species_keys[0])
    plt.show()

    # Compute Shannon on all forest polygons
    forest = data[data['POLYTYPE'] == 'FOR'].copy()
    forest['Shannon'] = forest[species_keys[0]].map(compute_shannon)
    forest.plot(column='Shannon', legend=True)
    plt.show()

    # Calibration Shannon
    forest[forest['SOURCE'] == 'PLOTVAR'].plot(column='Shannon', legend=True)
    plt.show()

    # Show Shannon on forest calibration plots subset
    calib_plots = forest[forest['SOURCE'] == 'PLOTVAR']
    calib_plots.plot(column='Shannon', legend=True)
    plt.show()

    # Build ROI and show
    polygon = box(2.9e5, 5.25e6, 3.1e5, 5.277e6)
    poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=forest.crs)

    ax = forest[forest['SOURCE'] == 'PLOTVAR'].plot(column='Shannon', legend=True)
    poly_gdf.boundary.plot(ax=ax, color="red")
    plt.show()

    ax = forest.plot(column='Shannon', legend=True)
    poly_gdf.boundary.plot(ax=ax, color="red")
    plt.show()

    # Export ROI
    poly_gdf.geometry.to_file("ontario_roi_reduced_3.geojson", driver="GeoJSON")
    # poly_gdf.geometry.to_file("ontario_roi_2.shp")

