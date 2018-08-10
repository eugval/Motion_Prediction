import os
import sys
ROOT_DIR = os.path.abspath("../")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))
sys.path.append(os.path.join(ROOT_DIR,"deprecated"))
import matplotlib
matplotlib.use('Agg')
import pickle
import torch
from experiments.training_tracker import TrainingTracker
from experiments.load_data import DataFromH5py, ResizeSample , ToTensor
from experiments.evaluation_metrics import IoUMetric, DistanceViaMean
from torchvision import transforms
import cv2
import numpy as np
from preprocessing.get_stats import get_histogram, get_histogram_same_plot
from experiments.model import Unet, UnetShallow, SpatialUnet,  SpatialUnet0, SpatialUnet2, SpatialUNetOnFeatures, UNetOnFeatures
import matplotlib.pyplot as plt
from deprecated.experiment import main_func
import json
import colorsys
import random
import h5py



data_names =[("Football1and2",1)]


for data_name, number in data_names :

    model = UnetShallow

    model_name = "UnetShallow_MI_{}_{}".format(data_name, number)

    param_file = os.path.join(MODEL_PATH, "{}/param_holder.pickle".format(model_name))

    ##### BACKWARD COMPATIBILITY ########
    params = pickle.load(open(param_file, "rb"))

    if ('label_type' not in params):
        params['label_type'] = 'future_mask'

    if ('model_inputs' not in params):
        params['model_inputs'] = [params['number_of_inputs']]

    if ('dataset_file' not in params):
        params['dataset_file'] = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name, data_name))

    if ('model_history_file' not in params):
        params['model_history_file'] = os.path.join(MODEL_PATH, "{}/{}_history.pickle".format(model_name, model_name))

    if ('model_folder' not in params):
        params['model_folder'] = os.path.join(MODEL_PATH, "{}/".format(model_name))

    if ('model_file' not in params):
        params['model_file'] = os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name, model_name))

    if ('baselines_file' not in params):
        params['baselines_file'] = os.path.join(PROCESSED_PATH, "{}/{}_metrics_to_beat.pickle".format(data_name, data_name))

    if ('intermediate_loss' not in params):
        params['intermediate_loss'] = False

    if ('many_times' not in params):
        params['many_times'] = False

    pickle.dump(params, open(param_file, "wb"))
    ###################################

