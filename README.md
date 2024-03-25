# SPAPS EO Poject 2023/2024

This repository gathers some code used during the earth observation project of the 
advanced Master Space APplications and Services of ISAE Supaero for the 2023/2024 session.

## Dependencies

The code makes use of the `spectralrao-monitoring package`. The code uses https://github.com/rnedelec/spectralrao-monitoring.
which is a fork of https://github.com/AndreaTassi23/spectralrao-monitoring with marginal modifications :

* A modification of the inputs to be able to use directly numpy matrices and not rasters as input
* A correction of what seems a mistake : missing absolute value in the 1D distance between pixel values
* Some prints removing

## Content of the repository

Some utilitary scripts for raster processing or simple data manipulation:

* `compute_and_store_ndvi_raster`
* `concatenate_sentinel_results`
* `make_binary_mask`
* `ndvi_mask_threhsold`
* `utils`

The scripts implementing the data processing in the Ivory Coast and Ontario study sites.
The whole process of computing the mean Rao's Q biodiversity index 
on field plots and computing the Pearson correlation coefficient is implmented in those scripts
(reusable parts of the code has been copied and not imported from one module to the other)

* `ivory_coast_farms_statistics`
* `ontario_processing`

A small script for computing the Rao's Q on a single AOI

* `process_single_subregion`

A small script used for window size sensitivity study

* `window_size_sensitivity`

A scipt used for processing of Ontario field data at the beginning of the project

* `vt_processing` : reused in `ontario_processing`




