from spaps_eo_project import ivory_coast_farms_statistics as eo_tools


if __name__ == '__main__':
    ndfi_filepath = r"D:\Overland\export-ivorycoast2023-sentinel2-bundle\20221231_S2B_T29NPH\20221231_S2B_T29NPH_ndfi.tif"
    subregion = eo_tools.get_subregion_from_geojson(
        r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ROI\sub_aoi_agroforestry_vs_full_sun_3_east.geojson"
        )

    product_name = "sentinel_overland"
    subregion_name = "3_east"
    indicator_name = "ndfi"
    mask_name = "ndfi"

    window = 11
    ndfi_threshold = 6500
    na_tolerance = 0.2


    eo_tools.process_subregion(
        mask_index_filepath=ndfi_filepath,
        mask_threshold=ndfi_threshold,
        rao_input_index_filpath=ndfi_filepath,
        window=window,
        na_tolerance=na_tolerance,
        subregion=subregion,
        output_filepath=f"./rao_subregion_{subregion_name}_{product_name}_{indicator_name}_"
                        f"{mask_name}_mask_"
                        f"w_{window}_"
                        f"na_{str(na_tolerance).replace('.', '_')}.tif",
        plot=True
        )