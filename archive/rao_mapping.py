import sys
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_raster, reshape_as_image
import numpy as np

import skimage as ski

sys.path.append(r"../spectralrao-monitoring")

import spectralrao

src = rasterio.open(r"C:\Users\Renaud\SPAPS\Team project\Ivory\PNeo_processed\farm_00_cliped_ned.tif")

raster = src.read()
img_16_bit = reshape_as_image(raster)

nir = img_16_bit[:, :, 0]
red_edge = img_16_bit[:, :, 1]
ndre = (nir - red_edge) / (nir + red_edge)

# img_rescaled = ski.exposure.rescale_intensity(img_16_bit, in_range='uint16')

# img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
# img_8_bit = ski.util.img_as_ubyte(img_16_bit)
# img_rescaled = ski.exposure.rescale_intensity(img_float, in_range=(0, 1))
# plt.imshow(img_8_bit)
# plt.show()

rao = spectralrao.spectralrao(ndre[:300, 400:], ".", window=9)
plt.imshow(rao[0])
plt.colorbar()
plt.show()
