
from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import tensorflow as tf
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras_frcnn import roi_helpers
import pandas as pd


sys.setrecursionlimit(40000)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

parser = OptionParser()

parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")


(options, args) = parser.parse_args()

config_output_filename = options.config_filename
with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

print(C.model_path)