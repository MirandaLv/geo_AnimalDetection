
import rasterio
import os
import numpy as np
import fiona
from shapely.geometry import Polygon, shape
import geopandas as gpd
import pandas as pd
import shapely
import json


class DataProcessing:

    def __init__(self, ROOT_DIR, IMAGE_PATH, POLY_PATH):

        self.ImagePath = IMAGE_PATH
        self.PolyPath = POLY_PATH
        self.ROOT_DIR = ROOT_DIR
        # self.polys = gpd.read_file(POLY_PATH)

    def create_meta_df(self, FolderPath):

        img_paths = [os.path.join(FolderPath, f) for f in os.listdir(FolderPath)]
        polys = gpd.read_file(self.PolyPath)
        outmeta = {'img_path': [], 'poly_ids': []}

        for img_path in img_paths:

            try:
                img = rasterio.open(img_path)
            except:
                print(img_path)
                raise

            bound_xmin, bound_ymin, bound_xmax, bound_ymax = img.bounds

            polys['centroid_x'] = polys['geometry'].centroid.x
            polys['centroid_y'] = polys['geometry'].centroid.y

            polys['inorout'] = polys.apply(lambda z: 'in' if bound_xmin < z['centroid_x'] < bound_xmax
                                 and bound_ymin < z['centroid_y'] < bound_ymax else 'out', axis=1)

            poly_geos = polys[polys['inorout'] == 'in']['geometry'].tolist()
            poly_ids = polys[polys['inorout'] == 'in']['img_id'].tolist()


        #df = pd.DataFrame.from_dict(outmeta)
        #df.to_csv(os.path.join(self.ROOT_DIR, 'out_meta.csv'), encoding='utf-8', sep=',')



    def generate_json(self):

        image_dict = {"image": '', "annotations": []}
        label_dict = {"label": '', "coordinates": {}}
        # coord_dict = {"x": float, "y": float, "width": int, "height": int}

        coord_dict = {"geometry": self.polys['geometry']}

        label_dict["label"] = 'bird'
        label_dict["coordinates"] = coord_dict

        # image_dict["image"] = self.image_name
        image_dict["annotations"].append(label_dict)

        # annotations.append(image_dict)



    def get_patches(self, dim, Out_PATH):

        try:
            poly_shapes = fiona.open(self.PolyPath)
        except:
            print(self.PolyPath)
            raise

        try:
            img_raw = rasterio.open(self.ImagePath)
        except:
            print(self.ImagePath)
            raise
        out_meta = img_raw.meta

        for feature in poly_shapes:

            img_id = feature['properties']['img_id']
            img_out = os.path.join(Out_PATH, str(img_id)+'.tif')
            poly = shape(feature['geometry'])

            lon = poly.centroid.x
            lat = poly.centroid.y

            r, c = img_raw.index(lon, lat)
            win = ((r - dim / 2, r + dim / 2), (c - dim / 2, c + dim / 2))
            # window = rasterio.windows.Window(lon - dim//2, lat - dim//2, dim, dim)

            try:
                data = img_raw.read(window=win)
            except:
                print(win)
                raise

            out_meta.update({
                "width": dim,
                "height": dim,
                "transform": rasterio.windows.transform(win, img_raw.transform),
                "nodata": None
            })

            with rasterio.open(img_out, 'w', **out_meta) as dst:
                dst.write(data)














