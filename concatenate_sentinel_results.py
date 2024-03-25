import scipy
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

indicator_template_path = (
    r"C:\Users\Renaud\SPAPS\Team project\Code\eo_project"
    r"\sentinel_{}_overland_cshn_mask_scv_850_w_5_na_0_3"
    r"_farms_filtered_0_1_no_bareland_filter_full_labels.csv"
)
nph_results_filepath = indicator_template_path.format("nph")
nnh_results_filepath = indicator_template_path.format("nnh")

nph_results = pd.read_csv(nph_results_filepath)
nnh_results = pd.read_csv(nnh_results_filepath)

results = pd.concat([nph_results, nnh_results])
results.drop_duplicates(subset=['ID'], inplace=True)

# Filter data and evaluate the regression
results_nan_filtered = results[~results['rao'].isna()]
# results_filtered_2500_px = results_nan_filtered[results_nan_filtered['nb_pixel_not_nan'] > 2500]
results_filtered_100_px = results_nan_filtered[
    results_nan_filtered['nb_pixel_not_nan'] > 100]

# Pearson coefficient
print(scipy.stats.pearsonr(results_nan_filtered['shaded'], results_nan_filtered['rao']))
print(
    scipy.stats.pearsonr(results_filtered_100_px['shaded'], results_filtered_100_px['rao']))

# Plot relation between shaded parameter and RAO's Q
plt.figure()
sns.regplot(results, x='shaded', y='rao')
plt.title("Regression for all farms")
plt.show()

plt.figure()
sns.regplot(results_filtered_100_px, x='shaded_prop_in_no_bareland', y='rao')
# plt.title("Regression for farms with more than 100 pixels in RAO index")
plt.show()


