
# This script is used to split the raw input raster image
import os.path

import pandas as pd
import rasterio

inf = r"/home/mirandalv/Documents/github/geo_AnimalDetection/dataset/raw_data/droneData/bird_30m_wgs84.tif"
outf = r"/home/mirandalv/Documents/github/geo_AnimalDetection/dataset/test.tif"
dim = 64

img_raw = rasterio.open(inf)
array = img_raw.read(1)
height, width = array.shape
xmin, ymin, xmax, ymax = img_raw.bounds


# get the coordinates of the top left corner
# start_lon = xmin
# start_lat = ymax

# print(xmin, ymin, xmax, ymax)
# r, c = img_raw.index(lon, lat)
r = 0
c = 0
# win = ((r - dim / 2, r + dim / 2), (c - dim / 2, c + dim / 2))


def get_startpoint(cols, rows, dim):

    num_cols = int(cols/dim)
    num_rows = int(rows/dim)

    windows = [(row*dim, col*dim) for row in range(num_rows) for col in range(num_cols)]

    return windows



corners = get_startpoint(width, height, dim)

for corner in corners:
    r, c = corner
    fname = str(r)+str(c)+'.tif'
    fpath = os.path.join(os.path.abspath("."), fname)
    window = rasterio.windows.Window(r, c, dim, dim)

#window = rasterio.windows.Window(r, c, dim, dim)

    out_meta = img_raw.meta
    out_meta.update({
                    "width": dim,
                    "height": dim,
                    "transform": rasterio.windows.transform(window, img_raw.transform),
                    "nodata": None
                })

    data = img_raw.read(window=window)
    with rasterio.open(fpath, 'w', **out_meta) as dst:
        dst.write(data)






