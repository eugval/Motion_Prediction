
import os
import sys
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")


sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))
sys.path.append(os.path.join(ROOT_DIR,"experiments"))

import torch.optim as optim
import torch.nn as nn


from experiments.model import   SimpleUNet
from experiments.model import TrainingTracker
from experiments.load_data import DataFromH5py, ResizeInput, ToTensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle


num_epochs = 10

data_file_name = "football1_sm5"
dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_file_name,data_file_name))
set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(data_file_name,data_file_name))

net = SimpleUNet()
print(net)


set_idx = pickle.load( open(set_idx_file, "rb" ) )


dataset = DataFromH5py(dataset_file,set_idx, transform = transforms.Compose([
                                               ResizeInput(),
                                               ToTensor()
                                           ]))

criterion = nn.MSELoss(size_average = False)
optimizer = optim.RMSprop(net.parameters(), lr=0.001)

dataloader = DataLoader(dataset, batch_size=32,
                        shuffle=True)

tracker = TrainingTracker()


for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = []
    running_distance = []
    for i, data in enumerate(dataloader):
        # get the inputs
        inputs = data['input'].float()
        labels = data['label'].float()
        labels = labels.unsqueeze(1)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        l2 = loss.item()
        c_true = data['future_centroid']

        #TODO: make it so I can do it for epochs
        tracker.add_distance(outputs,c_true,"mean")
        tracker.add_distance(outputs, c_true, "mode")
        tracker.add_loss(l2)

        if i % 1 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f mean distance : %.3f mode distance : %.3f' %
                  (epoch + 1, i + 1, tracker.running_loss[i],tracker.running_mean_distance[i], tracker.running_mode_distance[i]))


#tracker.plot_metrics("steps","values","plot")

def plot_metrics(metric):
    plt.plot(np.arange(len(metric)), metric)
    plt.show()


plot_metrics(tracker.running_mean_distance)
plot_metrics(tracker.running_mode_distance)
plot_metrics(tracker.running_loss)