
import rasterio
import os
import numpy as np
import fiona
from shapely.geometry import Polygon, shape
import geopandas as gpd
import pandas as pd
import shapely
import json
import random


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

            self.generate_annotation(img_path, poly_geos)

            outmeta['img_path'].append(img_path)
            outmeta['poly_ids'].append(poly_ids)

        #df = pd.DataFrame.from_dict(outmeta)
        #df.to_csv(os.path.join(self.ROOT_DIR, 'out_meta.csv'), encoding='utf-8', sep=',')


    def generate_annotation(self, single_image_path, polygeos):

        try:
            img = rasterio.open(single_image_path)
            width = img.width
            height = img.height
        except:
            print(img)
            raise

        basename = os.path.basename(single_image_path)
        annotationfile = '/'.join(os.path.dirname(single_image_path).split('/')[:-1])+ '/annotations/' + basename.split('.')[0] + '.json'

        image_dict = {"image_path": single_image_path, "image_name": basename,"annotations": [], "width": width, "height": height}


        for idx in range(len(polygeos)):

            # assert isinstance(polygeos[idx], shapely.geometry.polygon.Polygon)

            label_dict = {"label": 'bird'}
            regions = dict()

            minx, miny, maxx, maxy = polygeos[idx].bounds

            lur, luc = img.index(minx, maxy)
            brr, brc = img.index(maxx, miny)

            # win = ((r - dim / 2, r + dim / 2), (c - dim / 2, c + dim / 2))

            regions['minx'] = lur
            regions['maxy'] = luc
            regions['maxx'] = brr
            regions['miny'] = brc

            label_dict['region'] = regions
            image_dict['annotations'].append(label_dict)

        with open(annotationfile, 'w') as js:
            json.dump(image_dict, js)


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



    def prepare_train_val(self, annote_path, agg_path, train_csv):

        assert os.path.isdir(annote_path)
        allfiles = [os.path.join(annote_path, name) for name in os.listdir(annote_path) if os.path.isfile(os.path.join(annote_path, name))]

        df = pd.DataFrame(columns=['image_path', 'image_name', 'label', 'maxx', 'minx', 'maxy', 'miny'])
        for file in allfiles:
            f = open(file)
            data = json.load(f)

            temp_dict = dict()
            minxls = list()
            maxxls = list()
            minyls = list()
            maxyls = list()

            c_obj = len(data['annotations'])
            for c in range(0,c_obj):
                minxls.append(data['annotations'][c]['region']['minx'])
                maxxls.append(data['annotations'][c]['region']['maxx'])
                minyls.append(data['annotations'][c]['region']['miny'])
                maxyls.append(data['annotations'][c]['region']['maxy'])

            labelist = [data['annotations'][0]['label']] * c_obj
            img_paths = [data['image_path']] * c_obj
            img_names = [data['image_name']] * c_obj

            temp_dict['minx'] = minxls
            temp_dict['maxx'] = maxxls
            temp_dict['miny'] = maxyls
            temp_dict['maxy'] = minyls
            temp_dict['label'] = labelist
            temp_dict['image_path'] = img_paths
            temp_dict['image_name'] = img_names

            df = df.append(pd.DataFrame.from_dict(temp_dict))

        self.get_train(df, train_csv, train_rate=0.05)

        df.to_csv(agg_path, encoding='utf-8', sep=',', index=False)


    def get_train(self, df, train_csv, train_rate=0.05):

        img_count = df['image_path'].nunique()
        ctrain= int(img_count * train_rate)

        img_train = random.sample(range(0,img_count), ctrain)
        df['id'] = df.apply(lambda x: int(x['image_name'].split('.')[0]), axis=1)
        train_df = df[df['id'].isin(img_train)]

        train_df.to_csv(train_csv, encoding='utf-8', sep=',', index=False)
        txt_path = train_csv.split('.')[0] + '.txt'

        self.csv2txt(train_csv, txt_path)



    def csv2txt(self, fcsv, txt_path):

        df = pd.read_csv(fcsv, encoding='utf-8', sep=',')
        data = pd.DataFrame()
        data['format'] = df['image_path']

        """
        # as the images are in train_images folder, add train_images before the image name
        for i in range(data.shape[0]):
            data['format'][i] = 'train_images/' + data['format'][i]
        """
        # add xmin, ymin, xmax, ymax and class as per the format required
        for i in range(data.shape[0]):
            data['format'][i] = data['format'][i] + ',' + str(df['minx'][i]) + ',' + str(
                df['miny'][i]) + ',' + str(df['maxx'][i]) + ',' + str(df['maxy'][i]) + ',' + \
                                df['label'][i]

        data.to_csv(txt_path, header=None, index=None, sep=' ')





























