import os
from spaps_eo_project import ivory_coast_farms_statistics as eo_tools
import geopandas as gpd


if __name__ == '__main__':
    indicator_files_root_dir = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH"
    # ndfi_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH\20221231_S2B_T29NPH_ndfi.tif"
    # ndvi_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH\20221231_S2B_T29NPH_rdhz_ndvi_spaps_team.tif"

    product_name = "sentinel_overland"

    # Subregion
    # subregion_name = "6_east"
    # subregion_path = f"C:\\Users\\Renaud\\Dropbox\\SPAPS\\Team project\\ROI\\" \
    #                  f"sub_aoi_agroforestry_vs_full_sun_{subregion_name}.geojson"
    # subregion = eo_tools.get_subregion_from_geojson(subregion_path)
    farms = gpd.read_file(
        r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ivory_farms_shapefile"
        )

    selected_farm_id = 33
    subregion = farms[farms['Farm_ID'].astype('int') == selected_farm_id].iloc[0]["geometry"]
    subregion_name = f"farm_{selected_farm_id}"

    # Input spectral index
    indicator_name = "lai"
    indicator_filepath = os.path.join(
        indicator_files_root_dir, f"20221231_S2B_T29NPH_{indicator_name}.tif"
        )

    # Mask input
    mask_name = "scv"
    mask_filepath = os.path.join(indicator_files_root_dir, f"20221231_S2B_T29NPH_{mask_name}.tif")

    # Mask threshold
    ndfi_threshold = 5700
    scv_threshold = 850
    mask_threshold = scv_threshold

    # RAO parameters
    window = 5
    na_tolerance = 0.3

    # Process
    rao = eo_tools.process_subregion(
        mask_index_filepath=mask_filepath,
        mask_threshold=mask_threshold,
        rao_input_index_filpath=indicator_filepath,
        window=window,
        na_tolerance=na_tolerance,
        subregion=subregion,
        upper_threshold=True,
        output_filepath=f"./rao_subregion_{subregion_name}_{product_name}_{indicator_name}_"
                        f"{mask_name}_mask_{scv_threshold}"
                        f"w_{window}_"
                        f"na_{str(na_tolerance).replace('.', '_')}.tif",
        plot=True
        )
