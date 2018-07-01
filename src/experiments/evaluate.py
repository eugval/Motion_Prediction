import os
import sys
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")


sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))
sys.path.append(os.path.join(ROOT_DIR,"data_eval"))


import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import pickle

from torchvision import transforms
from experiments.model import   SimpleUNet
from experiments.history_tracking import DistanceViaMean, DistanceViaMode
from experiments.load_data import DataFromH5py, ResizeSample , ToTensor

matplotlib.use('Agg')

from matplotlib.offsetbox import AnchoredText

device = torch.device("cpu")


data_names =["Football1", "Crossing1","Light1"]

for data_name in data_names:
    print("Doing {}".format(data_name))

    model_name = "Mask_only_Simple_Unet_{}".format(data_name)
    model_file = os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))
    model_folder = os.path.join(MODEL_PATH, "{}/".format(model_name,model_name))

    dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name,data_name))
    idx_sets_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(data_name,data_name))





    model = SimpleUNet(12)
    model.load_state_dict(torch.load(model_file))

    model.to(device)

    idx_sets = pickle.load(open(idx_sets_file, "rb"))
    dataset = DataFromH5py(dataset_file,idx_sets, transform = transforms.Compose([
                                                       ResizeSample(),
                                                       ToTensor()
                                                      ]))


    idx =  np.random.randint(len(dataset))

    sample = dataset[idx]
    input = sample['input'].float().to(device)
    while (len(input.size()) < 4):
        input = input.unsqueeze(0)

    output = model(input)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    initial_dims = (dataset.initial_dims[1], dataset.initial_dims[0])
    #initial_dims = dataset.initial_dims
    output_initial_dims = cv2.resize(output,initial_dims )



    label = sample['label']


    sample_raw = dataset.get_raw(idx)
    input_raw = sample_raw['input']
    label_raw = sample_raw['label']


    centroid_list = []
    true_centroid = sample_raw['future_centroid']
    centroid_list.append((true_centroid[1],true_centroid[0]))


    distance_via_mean_calc = DistanceViaMean()
    distance_via_mode_calc = DistanceViaMode()
    centroid_via_mean = distance_via_mean_calc.get_centroid(output_initial_dims)
    centroid_list.append((centroid_via_mean[1],centroid_via_mean[0]))
    distance_via_mean = distance_via_mean_calc.get_metric(output_initial_dims,true_centroid)
    centroid_via_mode = distance_via_mode_calc.get_centroid(output_initial_dims)
    centroid_list.append((centroid_via_mode[1],centroid_via_mode[0]))
    distance_via_mode = distance_via_mode_calc.get_metric(output_initial_dims,true_centroid)


    plt.figure(figsize=(15,15))
    number_of_plots = 11

    plt.subplot2grid((4,3),(0,0))
    plt.imshow(input_raw[0])
    plt.title("RGB at time t")
    plt.subplot2grid((4,3),(0,1))
    plt.imshow(input_raw[3])
    plt.title("mask at  t")


    plt.subplot2grid((4,3),(0,2))
    plt.imshow(label_raw)
    plt.title("raw label")
    plt.scatter(*zip(centroid_list[0]), marker = '+')

    plt.subplot2grid((4,3),(1,0))
    plt.imshow(label)
    plt.title("resized label")

    plt.subplot2grid((4,3),(1,1))
    plt.imshow(output)
    plt.title("direct output")
    plt.subplot2grid((4,3),(1,2))
    plt.imshow(output_initial_dims)
    plt.title("resized output")


    plt.subplot2grid((4,3),(2,0), rowspan=2, colspan=3  )
    plt.scatter(*zip(*centroid_list))
    plt.imshow(output_initial_dims)
    plt.annotate(
        'true centroid:\ncoords: {}'.format([ round(elem, 2) for elem in true_centroid ] ),
        xy=(true_centroid[1],true_centroid[0]), xytext=(20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))



    centroid_via_mean =[ np.round(centroid_via_mean[i]) for i in range(2) ]


    plt.annotate(
        'centroid mean:\ncoords: {}\ndist: {}'.format(centroid_via_mean,round(distance_via_mean,2)),
        xy=(centroid_via_mean[1],centroid_via_mean[0]), xytext=(20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.annotate(
        'centroid mode:\n coords: {}, \n dist: {}'.format([ round(elem, 2) for elem in centroid_via_mode ], round(distance_via_mode,2)),
        xy=(centroid_via_mode[1], centroid_via_mode[0]), xytext=(20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))




    plt.title("centroids")

    plt.tight_layout()
    plt.savefig(model_folder+"img")
    print("done")