import numpy as np
import scipy
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from spaps_eo_project import ivory_coast_farms_statistics as eo_tools


def compute_shannon(plot_data):
    shannon = 0
    return shannon

raster_filpath = ""
shape_filepath = ""
field_data_filepath = ""

# Import data
plots = gpd.read_file(
    shape_filepath
    )
field_data = pd.read_csv(field_data_filepath)

results = pd.DataFrame(columns=["field_shannon", "mean_rao"])

# Main loop
for i, plot in plots.iterrows():
    plot_shannon = compute_shannon(field_data[plot['ID']])
    plot_rao = eo_tools.get_raster_data_on_subregion(raster_filpath, plot["geometry"])
    results = pd.concat(
        [results,
         pd.DataFrame(
             [[plot_shannon, np.nanmean(plot_rao)]], columns=results.columns
             )
         ]
        )

# Plot graph
plt.figure()
sns.regplot(results, x='field_shannon', y='mean_rao')
plt.show()

# Regression parameters
pearson, pvalue = scipy.stats.pearsonr(results['field_shannon'], results['mean_rao'])

