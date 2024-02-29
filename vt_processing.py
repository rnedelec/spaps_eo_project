import os
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
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

if __name__ == '__main__':
    data = gpd.read_file(
        r"/Users/retif/Desktop/SUPAERO/Tutored Project/pp_FRI_FIMv2_Martel_Forest(509)_2015_2D.gdb",
        layer="MAR_FRI_2D")

    ## Inspecting species composition
    if 'SPCOMP' in data:
        species_keys = ['SPCOMP']
    elif 'OSPCOMP' and 'USPCOMP' in data:
        species_keys = ['OSPCOMP', 'USPCOMP']

    ## Plotting compositions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Species Composition and Shannon Index", fontsize=16)

    data[data['POLYTYPE'] == 'FOR'].plot(column=species_keys[0], ax=axes[0, 0])
    axes[0, 0].set_title("Species Composition - All Forest Polygons")

    ## Calibration plots
    data[data['SOURCE'] == 'PLOTVAR'].plot(column=species_keys[0], ax=axes[0, 1])
    axes[0, 1].set_title("Species Composition - Calibration Plots")

    ## Compute Shannon on all forest polygons
    forest = data[data['POLYTYPE'] == 'FOR'].copy()
    forest['Shannon'] = forest[species_keys[0]].map(compute_shannon)
    forest.plot(column='Shannon', legend=True, ax=axes[1, 0])
    axes[1, 0].set_title("Shannon Index - All Forest Polygons")

    ## Calibration Shannon
    forest[forest['SOURCE'] == 'PLOTVAR'].plot(column='Shannon', legend=True, ax=axes[1, 1])
    axes[1, 1].set_title("Shannon Index - Calibration Plots")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    ## Show Shannon on forest calibration plots subset
    calib_plots = forest[forest['SOURCE'] == 'PLOTVAR']
    calib_plots.plot(column='Shannon', legend=True)
    plt.show()

    ## Build ROI and show
    polygon = box(2.9e5, 5.25e6, 3.1e5, 5.277e6)
    poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=forest.crs)

    ax = forest[forest['SOURCE'] == 'PLOTVAR'].plot(column='Shannon', legend=True)
    poly_gdf.boundary.plot(ax=ax, color="red")
    plt.show()

    ax = forest.plot(column='Shannon', legend=True)
    poly_gdf.boundary.plot(ax=ax, color="red")
    plt.show()

    ## Export ROI
  #  output_shapefile_path = "output_polygon.shp"
  #  poly_gdf.to_file(output_shapefile_path)

