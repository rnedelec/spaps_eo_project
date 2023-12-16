import os
import geopandas as gpd
import matplotlib.pyplot as plt
import math
from typing import Tuple, List


def process_species_comp(comp_string: str) -> List[Tuple[str, str]]:
    """Deserialize species compositin into list of tuples"""
    n = 6  # Nb of characters in line blocs
    blocks = [comp_string[i:i + n] for i in range(0, len(comp_string), n)]
    species_comp = []
    for block in blocks:
        species = block[:3].strip()
        cover_percentage = block[3:].strip()
        species_comp.append((species, cover_percentage))
    return species_comp


def compute_shannon(comp_string: str) -> float:
    """Computes Shannon index from deserialized species comp"""
    species_comp = process_species_comp(comp_string)
    shannon = 0
    for composition in species_comp:
        cover_percentage = composition[1]
        proportion = float(cover_percentage) / 100
        shannon -= proportion * math.log(proportion)
    return shannon


if __name__ == '__main__':
    data = gpd.read_file(os.environ["data_dir"] + r"\RM_FRI_SampleSet.gdb")

    # Inspecting species composition
    # Pinting first rows for forest polygons
    data[data['POLYTYPE'] == 'FOR']['SPCOMP'].head()

    # Plotting compositions
    data[data['POLYTYPE'] == 'FOR'].plot(column='SPCOMP')
    plt.show()

    # Calibratin plots
    data[data['SOURCE'] == 'PLOTVAR'].plot(column='SPCOMP')
    plt.show()

    # Compute Shannon on all forest polygons
    forest = data[data['POLYTYPE'] == 'FOR'].copy()
    forest['Shannon'] = forest['SPCOMP'].map(compute_shannon)
    forest.plot(column='Shannon', legend=True)
    plt.show()

    # Show Shannon on forest calibration plots subset
    calib_plots = forest[forest['SOURCE'] == 'PLOTVAR']
    calib_plots.plot(column='Shannon', legend=True)
    plt.show()