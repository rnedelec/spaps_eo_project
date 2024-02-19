import rasterio.plot
import rasterio.mask
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt

def compute_ndvi_from_sentinel(b04_filepath, b08_filepath, subregion=None):
    with rasterio.open(b04_filepath) as dataset_b04:
        if subregion is not None:
            red, transform = rasterio.mask.mask(dataset=dataset_b04,
                                                shapes=[subregion],
                                                crop=True, indexes=1)
        else:
            red = dataset_b04.read(1)
            transform = dataset_b04.transform

    with rasterio.open(b08_filepath) as dataset_b08:
        if subregion is not None:
            nir, transform = rasterio.mask.mask(dataset=dataset_b08,
                                                shapes=[subregion],
                                                crop=True, indexes=1)
        else:
            nir = dataset_b08.read(1)
            transform = dataset_b08.transform

        red[red > 10000] = 10000
        nir[nir > 10000] = 10000
        red[red < 0] = 0
        nir[nir < 0] = 0

        ndvi = (nir - red) / (nir + red)

    return ndvi, transform


def plot_from_1d_raster(raster_filepath, subregion=None):
    with rasterio.open(raster_filepath) as dataset:
        if subregion is not None:
            data, transform = rasterio.mask.mask(dataset=dataset,
                                                 shapes=[subregion],
                                                 crop=True)
        else:
            data = dataset.read()
            transform = dataset.transform

    plt.figure()
    plt.imshow(reshape_as_image(data))
    plt.colorbar()
    plt.show()



if __name__ == '__main__':
    b04_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_B04_10m.jp2"
    b08_filepath = r"C:\Users\Renaud\SPAPS\Team project\Ivory\Sentinel\S2B_MSIL2A_20230110T105329_N0509_R051_T29NNH_20230110T133644.SAFE\GRANULE\L2A_T29NNH_A030536_20230110T110319\IMG_DATA\R10m\T29NNH_20230110T105329_B08_10m.jp2"

    ndvi, transform = compute_ndvi_from_sentinel(b04_filepath, b08_filepath)

    # rasterio.plot.show(ndvi)
    plt.imshow(ndvi)
    plt.colorbar()
    plt.show()

