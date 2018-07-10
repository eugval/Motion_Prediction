import os
import sys
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")
import matplotlib
matplotlib.use('Agg')

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))
sys.path.append(os.path.join(ROOT_DIR,"deprecated"))


import torch
import numpy as np
import matplotlib.pyplot as plt

import cv2
import pickle

from torchvision import transforms
from experiments.model import   SimpleUNet, Unet
from experiments.evaluation_metrics import DistanceViaMean, DistanceViaMode
from experiments.load_data import DataFromH5py, ResizeSample , ToTensor



from matplotlib.offsetbox import AnchoredText

device = torch.device("cpu")

input_type = 'images'
input_types = ['masks', 'images']

data_name ="Crossing2"
trials = 5
input_num = 12
purpose = 'val'

for trial in range(trials):
    print("Doing {}".format(data_name))

    model_name = "Unet_MI_2ndGen_{}".format(data_name)
    model_file = os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))
    model_folder = os.path.join(MODEL_PATH, "{}/".format(model_name,model_name))

    dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name,data_name))
    idx_sets_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(data_name,data_name))


    save_path = os.path.join(model_folder,'qualitative_{}_{}'.format(purpose,trial))


    model = Unet(input_num)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))

    model.to(device)

    idx_sets = pickle.load(open(idx_sets_file, "rb"))
    dataset = DataFromH5py(dataset_file,idx_sets,input_type = input_types,purpose =purpose, transform = transforms.Compose([
                                                       ResizeSample(),
                                                       ToTensor()
                                                      ]))


    idx =  np.random.randint(len(dataset))

    sample = dataset[idx]
    input = sample['input'].float().to(device)
    while (len(input.size()) < 4):
        input = input.unsqueeze(0)

    output = model.eval_forward(input)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    initial_dims = (dataset.initial_dims[1], dataset.initial_dims[0])
    #initial_dims = dataset.initial_dims
    output_initial_dims = cv2.resize(output,initial_dims )
    output_after_thresh = output > 0.5



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

    output_initial_dims = output_initial_dims>0.5



    if(input_type == 'masks'):
        plt.figure(figsize=(15,15))
        number_of_plots = 11





        plt.subplot2grid((4,3),(0,0))
        plt.imshow(input_raw[2])
        plt.title("mask at  t-4")

        plt.subplot2grid((4,3),(0,1))
        plt.imshow(input_raw[1])
        plt.title("mask at  t-2")

        plt.subplot2grid((4,3),(0,2))
        plt.imshow(input_raw[0])
        plt.title("Mask time t")

        plt.subplot2grid((4, 3), (1, 0))
        plt.imshow(label_raw)
        plt.title("raw label (t+10)")
        plt.scatter(*zip(centroid_list[0]), marker='+')




        plt.subplot2grid((4,3),(1,1))
        plt.imshow(output)
        plt.title("direct output")

        plt.subplot2grid((4,3),(1,2))
        plt.imshow(output_initial_dims)
        plt.title("Thresholded and resized output at 0.5")

        plt.subplot2grid((4,3),(2,0), rowspan=2, colspan=3  )
        plt.scatter(*zip(*centroid_list))
        plt.imshow(output_initial_dims)
        plt.annotate(
            'true centroid' ,
            xy=(true_centroid[1],true_centroid[0]), xytext=(20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))



        centroid_via_mean =[ np.round(centroid_via_mean[i]) for i in range(2) ]


        plt.annotate(
            'centroid mean',
            xy=(centroid_via_mean[1],centroid_via_mean[0]), xytext=(-20, -20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))




        plt.title("centroids")

        plt.tight_layout()
        plt.savefig(save_path)
        print("done")
    else:


        plt.figure(figsize=(15, 15))
        number_of_plots = 11

        plt.subplot2grid((4, 3), (0, 0))
        plt.imshow(input_raw[0])
        plt.title("RGB at time t")
        plt.subplot2grid((4, 3), (0, 1))
        plt.imshow(input_raw[3])
        plt.title("mask at  t")

        plt.subplot2grid((4, 3), (0, 2))
        plt.imshow(label_raw)
        plt.title("raw label")
        plt.scatter(*zip(centroid_list[0]), marker='+')

        plt.subplot2grid((4, 3), (1, 0))
        plt.imshow(label)
        plt.title("resized label")





        plt.subplot2grid((4,3),(1,1))
        plt.imshow(output)
        plt.title("direct output")

        plt.subplot2grid((4,3),(1,2))
        plt.imshow(output_initial_dims)
        plt.title("Thresholded and resized output at 0.5")


        plt.subplot2grid((4, 3), (2, 0), rowspan=2, colspan=3)
        plt.scatter(*zip(*centroid_list))
        plt.imshow(output_initial_dims)
        plt.annotate(
            'true centroid:\ncoords: {}'.format([round(elem, 2) for elem in true_centroid]),
            xy=(true_centroid[1], true_centroid[0]), xytext=(-20, -20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        centroid_via_mean = [np.round(centroid_via_mean[i]) for i in range(2)]

        plt.annotate(
            'centroid mean:\ncoords: {}\ndist: {}'.format(centroid_via_mean, round(distance_via_mean, 2)),
            xy=(centroid_via_mean[1], centroid_via_mean[0]), xytext=(20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


        plt.title("centroids")

        plt.tight_layout()
        plt.savefig(save_path)
        print("done")