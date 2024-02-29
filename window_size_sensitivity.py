import geopandas as gpd
import numpy as np
from spaps_eo_project import ivory_coast_farms_statistics as eo_tools
import matplotlib
matplotlib.use('Qt5Agg')

ndfi_filepath = r"D:\Overland\export-ivorycoast2023-spot\20230108_S7_065N0080W\20230108_S7_065N0080W_ndfi.tif"

# Get farm shapes
farms = gpd.read_file(
    r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ivory_farms_shapefile"
    )

# SPOT 7 projection
farms.to_crs("EPSG:4326")

# Select farm
selected_farm_id = 6
farm = farms[farms['Farm_ID'].astype('int') == selected_farm_id].iloc[0]
subregion = farm["geometry"]

results = []
for window in [5, 9, 11, 13, 15]:
    rao = eo_tools.process_subregion(
        mask_index_filepath=ndfi_filepath,
        mask_threshold=6500,
        rao_input_index_filpath=ndfi_filepath,
        subregion=subregion,
        window=window,
        na_tolerance=0.2,
        output_filepath=f"./farm_6_spot7_ndfi_window_{window}.tif",
        plot=True
        )
    results.append([window, np.nanmean(rao)])
