import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt


class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)

        self.deconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv6 = nn.Conv2d(32, 16, 3, padding=1)

        self.conv7 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv8 = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x2 = F.relu(x2)

        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x3 = F.relu(x3)

        x4 = self.deconv1(x3)
        x4 = torch.cat((x4, x2), 1)
        x4 = self.conv4(x4)
        x4 = F.relu(x4)

        x5 = self.deconv2(x4)
        x5 = torch.cat((x5, x1), 1)
        x5 = self.conv5(x5)
        x5 = F.relu(x5)

        x6 = self.conv6(x5)
        x6 = F.relu(x6)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        return x8



#TODO: make it work with general metrics
class TrainingTracker(object):
    def __init__(self):
        self.running_loss = []
        self.running_mean_distance = []
        self.running_mode_distance = []

    def add_distance(self, batch_outputs, batch_true_centroids, mode):
        if(mode == "mean"):
            self.running_mean_distance.append(self.get_mean_distance(batch_outputs,batch_true_centroids, mode))

        elif(mode=="mode"):
            self.running_mode_distance.append(self.get_mean_distance(batch_outputs,batch_true_centroids,mode))
        else:
            raise ValueError("Can only do mean and mode distances")

    def add_loss(self, loss):
        self.running_loss.append(loss)

    def plot_metrics(self, x_label, y_label, title):
        plt.plot(np.arange(len(self.running_loss)), self.running_loss, label = 'running_loss' )
        plt.plot(np.arange(len(self.running_mean_distance)), self.running_mean_distance, label='running_mean_dist')
        plt.plot(np.arange(len(self.running_mode_distance)), self.running_mode_distance, label='running_mode_dist')
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.title(title)

        plt.legend()

        plt.show()



    def get_mean_distance(self,batch_outputs,batch_c_true, mode = "mean"):

        np_pred = batch_outputs.detach().numpy()
        np_pred = np.squeeze(np_pred, axis =1)
        np_c_true = batch_c_true.numpy()

        if(mode == "mean"):
            batch_c_pred =np.stack([self.get_mean_centroid(np_pred[i,:,:]) for i in range(np_pred.shape[0])])

        if(mode =='mode'):
            batch_c_pred = np.stack([self.get_mode_centroid(np_pred[i, :, :]) for i in range(np_pred.shape[0])])

        dist = np.linalg.norm(batch_c_pred - np_c_true, axis=1)

        return np.mean(dist)


    def get_mean_centroid(self, grid):
        grid_dims = grid.shape
        x = np.broadcast_to(np.arange(grid_dims[0]).reshape(grid_dims[0], 1), (grid_dims[0], grid_dims[1]))
        y = np.broadcast_to(np.arange(grid_dims[1]), (grid_dims[0], grid_dims[1]))
        d = np.dstack((x, y))
        positions = np.reshape(d, (-1, 2))
        w = np.zeros(positions.shape[0])

        for i, v in enumerate(positions):
            w[i]= grid[tuple(v)]

        mean_x = np.average(positions[:,0], weights = w)
        mean_y = np.average(positions[:,1], weights = w)

        return np.array([mean_x,mean_y])

    def get_mode_centroid(self, grid):

        return np.unravel_index(np.argmax(grid), grid.shape)
