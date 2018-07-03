import os
import sys
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")


sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))
sys.path.append(os.path.join(ROOT_DIR,"deprecated"))
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))

from preprocessing.tracking import iou

import numpy as np
import cv2
import matplotlib.pyplot as plt


class CentroidCalculator(object):
    def get_centroid_via_mean(self, grid):
        grid_dims = grid.shape

        x = np.broadcast_to(np.arange(grid_dims[0]).reshape(grid_dims[0], 1), (grid_dims[0], grid_dims[1]))
        y = np.broadcast_to(np.arange(grid_dims[1]), (grid_dims[0], grid_dims[1]))
        d = np.dstack((x, y))

        positions = np.reshape(d, (-1, 2))
        w = [grid[tuple(v)] for v in positions]

        mean_x = np.average(positions[:, 0], weights=w)
        mean_y = np.average(positions[:, 1], weights=w)

        return np.array([mean_x,mean_y])

    def get_centroid_via_mode(self,grid):
        return np.unravel_index(np.argmax(grid), grid.shape)


class DistanceViaMean(object):
    name = 'dist_via_mean'

    def __init__(self):
        self.centroid_calculator = CentroidCalculator()

    def get_centroid(self, grid):
        return self.centroid_calculator.get_centroid_via_mean(grid)

    def get_metric(self, grid, centroid):
        centroid_pred = self.get_centroid(grid)

        dist = np.linalg.norm(centroid - centroid_pred)

        return dist

    def evaluate(self, model, dataloader,  device):

        num_examples = len(dataloader)
        tot_dist = 0.0

        for i, data in enumerate(dataloader):
            inputs = data['input'].float().to(device)
            centroids = data['future_centroid']

            outputs = model(inputs)

            outputs = outputs.detach().cpu().numpy()
            outputs = np.squeeze(outputs)


            for i in range(outputs.shape[0]):
                output = outputs[i,:,:]
                intial_h = dataloader.dataset.initial_dims[0]
                initial_w = dataloader.dataset.initial_dims[1]
                output = cv2.resize(output, (initial_w, intial_h))
                tot_dist += self.get_metric(output,centroids[i])

        return tot_dist / float(num_examples)




class MaskToMeasures(object):
    def get_bbox_from_mask(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

    def get_centroid_from_mask(self,mask):
        rmin, rmax, cmin, cmax = self.get_bbox_from_mask(mask)
        return int(rmin + (rmax-rmin)/2), int(cmin + (cmax-cmin)/2)

#TODO vectorize this
class IoUMetric(object):
    name = 'iou'
    types = ['bbox', 'mask']

    def __init__(self, type):
        if(type not in  self.types):
            raise ValueError("IoU only works with masks and bounding boxes")
        self.type = type

        if(type == 'bbox'):
            self.boxMaker = MaskToMeasures()

    def get_metric(self, pred_mask, true_mask):
        if(self.type == 'bbox'):
            p_rmin, p_rmax, p_cmin, p_cmax = self.boxMaker.get_bbox_from_mask(pred_mask)
            t_rmin, t_rmax, t_cmin, t_cmax = self.boxMaker.get_bbox_from_mask(true_mask)

            return iou([p_cmin, p_rmin, p_cmax, p_rmax], [t_cmin, t_rmin, t_cmax, t_rmax])
        elif(self.type == 'mask'):
            if(not pred_mask.dtype == 'bool' or true_mask.dtype =='bool'):
                pred_mask = pred_mask.astype('bool')
                true_mask = true_mask.astype('bool')
            intersection = pred_mask*true_mask
            union = pred_mask + true_mask

            return intersection.sum()/float(union.sum())

    def evaluate(self, model, dataloader,  device):
        num_examples = len(dataloader)
        tot_iou = 0.0

        for i, data in enumerate(dataloader):
            inputs = data['input'].float().to(device)
            labels = data['label'].float().to(device)

            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy()
            outputs = np.squeeze(outputs)

            labels = labels.detach().cpu().numpy()

            for i in range(outputs.shape[0]):
                    output = outputs[i,:,:]
                    label = labels[i,:,:]
                    tot_iou += self.get_metric(output,label)

            return tot_iou / float(num_examples)






class DistanceViaMode(object):
    name = 'dist_via_mode'

    def __init__(self):
        self.centroid_calculator = CentroidCalculator()

    def get_centroid(self, grid):
        return self.centroid_calculator.get_centroid_via_mode(grid)

    def get_metric(self, grid, centroid):
        centroid_pred = self.get_centroid(grid)
        dist = np.linalg.norm(centroid - centroid_pred)
        return dist

    def evaluate(self, model, dataset, device, eval_percent=1):
        num_examples = int(len(dataset)*eval_percent)
        indices = np.random.permutation(num_examples)

        tot_dist = 0.0
        for idx in indices:
            sample = dataset[idx]
            input = sample['input'].float().to(device)
            while (len(input.size()) < 4):
                input= input.unsqueeze(0)

            centroid = sample['future_centroid']

            output = model(input)

            output = output.detach().cpu().numpy()
            output = np.squeeze(output)
            intial_h = dataset.initial_dims[0]
            initial_w = dataset.initial_dims[1]
            output = cv2.resize(output, (initial_w, intial_h))



            tot_dist += self.get_metric(output,centroid)

        return tot_dist / float(num_examples)




class LossMetric(object):
    name = 'loss'

    def evaluate(self,model, criterion, dataloader, device):

        num_examples = len(dataloader)
        tot_loss = 0.0

        for i, data in enumerate(dataloader):
            inputs = data['input'].float().to(device)
            labels = data['label'].float().to(device)
            labels = labels.unsqueeze(1)

            outputs = model(inputs)

            tot_loss += criterion(outputs, labels).item()
        return tot_loss / float(num_examples)





class TrainingTracker(object):
    def __init__(self):
        self.metrics = {}


    def add(self, value, name):
        if name in self.metrics:
            self.metrics[name].append(value)
        else:
            self.metrics[name] = [value]


    def plot_metric(self, metric_name,  y_label= False, title = False , log = False, save_path = False):
        metric = self.metrics[metric_name]
        plt.plot(np.arange(len(metric)), metric)

        plt.xlabel("Epochs")

        if y_label :
            plt.ylabel(y_label)
        else:
            plt.ylabel(metric_name)

        if title:
            plt.title(title)
        else:
            plt.title('{} vs training time'.format(metric_name))

        if(log):
            plt.yscale('log')

        if(save_path):
            plt.savefig(save_path)

        else:
            plt.show()


    def plot_all(self, save_path = False):
        plt.figure(figsize=(20, 15))
        number_of_plots = len(self.metrics.keys())

        i = 1
        for k, v in self.metrics.items():
            plt.subplot(int(number_of_plots/2), 2, i)
            plt.plot(v)
            plt.xlabel("Epochs")
            plt.ylabel(k)
            plt.title(k + " vs Epoch Number")
            i += 1

        plt.tight_layout()

        if (save_path):
            plt.savefig(save_path)
        else:
            plt.show()
