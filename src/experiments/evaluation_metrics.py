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
import torch

class MaskToMeasures(object):
    def get_bbox_from_mask(self, mask):
        if(np.sum(mask) < 3):
            return 0,0,0,0
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

    def get_centroid_from_mask(self,mask):
        rmin, rmax, cmin, cmax = self.get_bbox_from_mask(mask)
        return int(rmin + (rmax-rmin)/2), int(cmin + (cmax-cmin)/2)


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

        return np.array([mean_y,mean_x])

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

        num_examples = len(dataloader.dataset)
        tot_dist = 0.0

        for i, data in enumerate(dataloader):
            inputs = data['input'].float().to(device)
            centroids = data['future_centroid'].detach().numpy()

            if(hasattr(model, 'eval_forward')):
                outputs = model.eval_forward(inputs)
            else:
                outputs = model(inputs)

            outputs = torch.squeeze(outputs,1)
            outputs = outputs.detach().cpu().numpy()

            for i in range(outputs.shape[0]):
                output = outputs[i,:,:]
                intial_h = dataloader.dataset.initial_dims[0]
                initial_w = dataloader.dataset.initial_dims[1]
                output = cv2.resize(output, (initial_w, intial_h), interpolation=cv2.INTER_NEAREST)
                tot_dist += self.get_metric(output,centroids[i])

        return tot_dist / float(num_examples)


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


    def evaluate(self, model, dataloader,  device):

        num_examples = len(dataloader.dataset)
        tot_dist = 0.0

        for i, data in enumerate(dataloader):
            inputs = data['input'].float().to(device)
            centroids = data['future_centroid'].detach().numpy()

            if(hasattr(model, 'eval_forward')):
                outputs = model.eval_forward(inputs)
            else:
                outputs = model(inputs)


            outputs = torch.squeeze(outputs,1)
            outputs = outputs.detach().cpu().numpy()

            for i in range(outputs.shape[0]):
                output = outputs[i,:,:]
                intial_h = dataloader.dataset.initial_dims[0]
                initial_w = dataloader.dataset.initial_dims[1]
                output = cv2.resize(output, (initial_w, intial_h), interpolation=cv2.INTER_NEAREST)
                tot_dist += self.get_metric(output,centroids[i])

        return tot_dist / float(num_examples)



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

    def get_metric(self, pred_masks, true_masks, verbose =1):
        if(self.type == 'bbox'):
            ious = []
            for i in range(pred_masks.shape[0]):
                pred_mask = pred_masks[i,:,:]
                true_mask = true_masks[i,:,:]

                #if(not np.any(true_mask) and not  np.any(pred_mask)):
                 #   continue
                #else:
                p_rmin, p_rmax, p_cmin, p_cmax = self.boxMaker.get_bbox_from_mask(pred_mask)
                t_rmin, t_rmax, t_cmin, t_cmax = self.boxMaker.get_bbox_from_mask(true_mask)

                ious.append(iou([p_cmin, p_rmin, p_cmax, p_rmax], [t_cmin, t_rmin, t_cmax, t_rmax]))
                #####DEBUG - REMOVE ####
                if(verbose>0):
                    if(np.isnan(iou([p_cmin, p_rmin, p_cmax, p_rmax], [t_cmin, t_rmin, t_cmax, t_rmax]))):
                        print('nan in bbox iou')
                        print('pred and true bbxes:')
                        print([p_cmin, p_rmin, p_cmax, p_rmax], [t_cmin, t_rmin, t_cmax, t_rmax])
                        print('predicted mask:{}'.format(pred_mask.sum(-1).sum(-1)))
                        print('true mask: {}'.format(true_mask.sum(-1).sum(-1)))
                #############################

            return sum(ious)
        elif(self.type == 'mask'):
            if(not pred_masks.dtype == 'bool' or not true_masks.dtype =='bool'):
                pred_masks = pred_masks.astype('bool')
                true_masks = true_masks.astype('bool')
            intersection = pred_masks*true_masks
            union = pred_masks + true_masks

            #####DEBUG - REMOVE ####
            if(verbose>0):
                if(np.isnan((intersection.sum(-1).sum(-1)/union.sum(-1).sum(-1).astype('float')).sum())):
                    print('mask ious nan')
                    print('intesection:')
                    print(intersection.sum(-1).sum(-1))
                    print('union:')
                    print(union.sum(-1).sum(-1))
                    print('predicted masks:{}'.format(pred_masks.sum(-1).sum(-1)))
                    print('true masks: {}'.format(true_masks.sum(-1).sum(-1)))
            ####################
            return (intersection.sum(-1).sum(-1)/union.sum(-1).sum(-1).astype('float')).sum()

    def evaluate(self, model, dataloader,  device, threshold = 0.5):
        num_examples = len(dataloader.dataset)
        tot_iou = 0.0

        for i, data in enumerate(dataloader):
            inputs = data['input'].float().to(device)
            labels = data['label'].float().to(device)

            if(hasattr(model, 'eval_forward')):
                outputs = model.eval_forward(inputs)
            else:
                outputs = model(inputs)

            outputs = torch.squeeze(outputs, 1)
            outputs = outputs.detach().cpu().numpy()


            labels = labels.detach().cpu().numpy()

            outputs = outputs>threshold
            labels = labels.astype('bool')


            tot_iou += self.get_metric(outputs,labels)

        return tot_iou / float(num_examples)



class LossMetric(object):
    name = 'loss'

    def evaluate(self,model, criterion, dataloader, device):
        tot_loss = 0.0
        count = 0
        for i, data in enumerate(dataloader):
            inputs = data['input'].float().to(device)
            labels = data['label'].float().to(device)
            labels = labels.unsqueeze(1)

            outputs = model(inputs)
            count+=1

            tot_loss += criterion(outputs, labels).item()
        return tot_loss / count

