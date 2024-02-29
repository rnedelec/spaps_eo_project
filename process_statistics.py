import geopandas as gpd
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

farms = gpd.read_file(
        r"C:\Users\Renaud\Dropbox\SPAPS\Team project\ivory_farms_shapefile"
        )

results = pd.read_csv("rao_ndfi_ndfi_mask_6500_window_11_na_tol_0_2.csv")

farms_to_drop_from_results = []
for i, result in results.iterrows():
    if (farms[farms["Farm_ID"].astype(int) == int(result['ID'])]["BARELAND"].values[0] > 0.1
        or farms[farms["Farm_ID"].astype(int) == int(result['ID'])]["NON_COCOA"].values[0] > 0.1
        or farms[farms["Farm_ID"].astype(int) == int(result['ID'])]["NATURAL"].values[0] > 0.1):
        # print(farms[farms["Farm_ID"].astype(int) == int(result['ID'])][["BARELAND", "NON_COCOA", "NATURAL"]])
        farms_to_drop_from_results.append(i)

results_filtered = results.drop(farms_to_drop_from_results)
df_more_than_2500_pix = results_filtered[results_filtered["nb_pixel_not_nan"] > 2500]
df_more_than_2500_pix_unfiltered = results[results["nb_pixel_not_nan"] > 2500]

sns.scatterplot(results, x="shaded", y="rao")
sns.scatterplot(results_filtered, x="shaded", y="rao")
sns.scatterplot(df_more_than_2500_pix, x="shaded", y="rao")

plt.figure()
sns.scatterplot(df_more_than_2500_pix, x="shaded", y="rao")

df_more_than_2500_cleaned = df_more_than_2500_pix[~df_more_than_2500_pix['rao'].isnull()]
df_more_than_2500_pix_unfiltered_cleaned = (
    df_more_than_2500_pix_unfiltered[~df_more_than_2500_pix_unfiltered['rao'].isnull()]
)
scipy.stats.pearsonr(
    df_more_than_2500_cleaned["shaded"], df_more_than_2500_cleaned['rao']
    )

