

import os
import sys
#import tensorflow_hub as hub
import pandas as pd
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
#ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)
import geopandas as gpd
import DataProcessing as DP
import json
import skimage


img = os.path.join(ROOT_DIR, "dataset/raw_data/droneData/bird_30m_wgs84.tif")
polys = os.path.join(ROOT_DIR, "dataset/raw_data/digitizedData/bird_poly.geojson")
out = os.path.join(ROOT_DIR, "dataset/processing_data/clipped")
annotation = os.path.join(ROOT_DIR, "dataset/processing_data/annotations")
agg_annote = os.path.join(ROOT_DIR, "dataset/processing_data/agg.csv")
agg_train = os.path.join(ROOT_DIR, "dataset/processing_data/train_annotation.csv")

#detector = hub.Module("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")
#print(detector)

# df = pd.read_csv(os.path.join(ROOT_DIR, "out_meta.csv"), encoding='utf-8', sep=',')
a = DP.DataProcessing(ROOT_DIR, img, polys)

a.get_patches(256, out)
a.create_meta_df(out)

a.prepare_train_val(annotation, agg_annote, agg_train)

def create_json(path, jsonpath):

    files = [os.path.join(path, f) for f in os.listdir(path)]
    file_dict = dict()

    for file in files:
        f = open(file)
        img_id = os.path.basename(file).split('.')[0]
        file_dict[img_id] = json.load(f)

    with open(jsonpath, 'w') as js:
        json.dump(file_dict, js)

#create_json("/home/mirandalv/Documents/github/ObjectDetection/dataset/annotation_val", "/home/mirandalv/Documents/github/ObjectDetection/dataset/val/annotation.json")

#jsontest = r"/home/mirandalv/Documents/github/ObjectDetection/dataset/annotation/1.json"
#json.load(jsontest)


#a.prepare_train_val(annote_path, train_random=0.2, val_random=0.2)
