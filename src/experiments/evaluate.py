import os
import sys
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")


sys.path.append(os.path.join(ROOT_DIR,"experiments"))


import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

from torchvision import transforms
from experiments.model import   SimpleUNet
from experiments.history_tracking import CentroidCalculator
from experiments.load_data import DataFromH5py, ResizeSample , ToTensor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



data_name = 'Football1_sm5'


model_name = "Mask_only_Simple_Unet_{}".format(data_name)
model_file = os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))


dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name,data_name))
idx_sets_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(data_name,data_name))





model = SimpleUNet(12)
model.load_state_dict(torch.load(model_file))

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

centroid_calculator = CentroidCalculator()
centroid_via_mean = centroid_calculator.get_centroid_via_mean(output_initial_dims)
centroid_list.append((centroid_via_mean[1],centroid_via_mean[0]))
centroid_via_mode = centroid_calculator.get_centroid_via_mode(output_initial_dims)
centroid_list.append((centroid_via_mode[1],centroid_via_mode[0]))

plt.figure()
number_of_plots = 11

plt.subplot(3, 4, 1)
plt.imshow(input_raw[0])
plt.title("time t")
plt.subplot(3, 4, 2)
plt.imshow(input_raw[1])
plt.title("time t+1")
plt.subplot(3, 4, 3)
plt.imshow(input_raw[2])
plt.title("time t+2")
plt.subplot(3, 4, 4)
plt.imshow(input_raw[3])
plt.title("mask at  t")
plt.subplot(3, 4, 5)
plt.imshow(input_raw[4])
plt.title("mask at  t+1")
plt.subplot(3, 4, 6)
plt.imshow(input_raw[5])
plt.title("mask at  t+2")

plt.subplot(3, 4, 7)
plt.imshow(label_raw)
plt.title("raw label")
plt.scatter(*zip(centroid_list[0]), marker = '+')

plt.subplot(3, 4, 8)
plt.imshow(label)
plt.title("resized label")

plt.subplot(3, 4, 9)
plt.imshow(output)
plt.title("direct output")
plt.subplot(3, 4, 10)
plt.imshow(output_initial_dims)
plt.title("resized output")
plt.subplot(3, 4, 11)
plt.scatter(*zip(*centroid_list))
plt.imshow(output_initial_dims)
plt.annotate(
    'true centroid:\ncoords: {}\ndist: {}'.format(true_centroid,true_centroid),
    xy=(true_centroid[1],true_centroid[0]), xytext=(-20, 20),
    textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


plt.annotate(
    'centroid mean:\ncoords: {}\ndist: {}'.format(centroid_via_mean,centroid_via_mean),
    xy=(centroid_via_mean[1],centroid_via_mean[0]), xytext=(20, 20),
    textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.annotate(
    'centroid mode:\n coords: {}, \n dist: {}'.format(centroid_via_mode, centroid_via_mode),
    xy=centroid_via_mode, xytext=(-20, -20),
    textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.title("centroids")

plt.show()