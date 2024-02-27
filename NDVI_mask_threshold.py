import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show

def calculate_ndvi(b04_data, b08_data):
    ndvi = (b08_data - b04_data) / (b08_data + b04_data + 1e-8)
    return ndvi

def create_ndvi_mask(ndvi_data, threshold):
    mask = np.where(ndvi_data > threshold, 1, 0)
    return mask

# File paths
b04_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/S2A_MSIL1C_20231213T163701_N0510_R083_T16TGT_20231213T182817.SAFE/GRANULE/L1C_T16TGT_A044267_20231213T163701/IMG_DATA/T16TGT_20231213T163701_B04.jp2"
b08_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/S2A_MSIL1C_20231213T163701_N0510_R083_T16TGT_20231213T182817.SAFE/GRANULE/L1C_T16TGT_A044267_20231213T163701/IMG_DATA/T16TGT_20231213T163701_B08.jp2"
rgb_filepath = r"/Users/retif/Desktop/SUPAERO/Tutored Project/S2A_MSIL1C_20231213T163701_N0510_R083_T16TGT_20231213T182817.SAFE/GRANULE/L1C_T16TGT_A044267_20231213T163701/IMG_DATA/T16TGT_20231213T163701_TCI.jp2"

# Read RGB and band data using rasterio
with rasterio.open(b04_filepath) as b04_ds, rasterio.open(b08_filepath) as b08_ds, rasterio.open(rgb_filepath) as rgb_ds:
    b04_data = b04_ds.read(1)
    b08_data = b08_ds.read(1)
    rgb_data = rgb_ds.read([1, 2, 3])

# Calculate NDVI
ndvi = calculate_ndvi(b04_data, b08_data)

# Set NDVI threshold
threshold = 0.4

# Create NDVI mask
ndvi_mask = create_ndvi_mask(ndvi, threshold)

# Visualize NDVI, RGB, and NDVI mask
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Visualize NDVI
axes[0].imshow(ndvi, cmap='viridis')
axes[0].set_title('NDVI')
axes[0].axis('off')

# Visualize RGB image
show(rgb_data, ax=axes[1], title='RGB Image')

# Visualize NDVI mask on RGB image
rgb_with_mask = np.copy(rgb_data)
rgb_with_mask[0, ndvi_mask == 1] = 255  # Set red channel to max for masked pixels
axes[2].imshow(rgb_with_mask.transpose(1, 2, 0))
axes[2].set_title('RGB with NDVI Mask')
axes[2].axis('off')

plt.show()
