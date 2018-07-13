import os
import sys
ROOT_DIR = os.path.abspath("../")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))
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
from preprocessing.get_stats import get_histogram
from experiments.model import Unet, UnetShallow



class ModelEvaluator(object):
    def __init__(self, model, weights_file, tracker_file, param_file, save_folder, dataset_file,
                 reload_tracker = False, cpu_only = False):
        if(cpu_only):
            self.device = 'cpu'
            self.device_string = 'cpu'
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.device_string = "cuda:0"
        self.params = pickle.load(open(param_file, "rb"))
        self.training_tracker = pickle.load(open(tracker_file, "rb"))
        if(reload_tracker):
           tracker2 = TrainingTracker(self.training_tracker.iterations_per_epoch)
           tracker2.metrics = self.training_tracker.metrics
           tracker2.saved_epoch = self.training_tracker.saved_epoch
           tracker2.finished = self.training_tracker.finished
           self.training_tracker = tracker2

        self.save_folder = save_folder
        self.model = model(self.params['number_of_inputs'])
        self.model.load_state_dict(torch.load(weights_file, map_location = self.device_string))
        self.model.to(self.device)

        self.dataset_file = dataset_file

        self.performance_statistics = {}
        self.performance_arrays = {}
        self.problematic_datapoints = []


    def get_performace_stats(self, set):
        dataset = DataFromH5py(self.dataset_file,self.params['idx_sets'],input_type = self.params['input_types'],
                                    purpose = set, label_type = 'future_mask',
                                    transform = transforms.Compose([
                                        ResizeSample(height= self.params['resize_height'], width = self.params['resize_width']),
                                        ToTensor()
                                    ])) #self.params['label_type']


        iou_bbox = IoUMetric(type = 'bbox')
        iou_mask = IoUMetric(type = 'mask')
        distance_via_mean = DistanceViaMean()

        iou_bboxes = []
        iou_masks = []
        distances_via_mean = []

        len_data = len(dataset)
        for i in range(len_data):
            sample = dataset[i]
            raw_sample = dataset.get_raw(i)

            input = sample['input'].float().to(self.device)

            while (len(input.size()) < 4):
                input = input.unsqueeze(0)

            with torch.no_grad():
                if(hasattr(self.model, 'eval_forward')):
                    output = self.model.eval_forward(input)
                else:
                    output = self.model(input)

                output = output.detach().cpu().numpy()
                output = np.squeeze(output)
                initial_dims = (dataset.initial_dims[1], dataset.initial_dims[0])
                output_initial_dims = cv2.resize(output, initial_dims)
                output_initial_dims_exp = output_initial_dims
                while (len(output_initial_dims_exp.shape) < 3):
                    output_initial_dims_exp = np.expand_dims(output_initial_dims_exp,0)

                output_after_thresh = output_initial_dims_exp > 0.5
                label_raw = raw_sample['label']

                while (len(label_raw.shape) < 3):
                    label_raw = np.expand_dims(label_raw,0)

                future_centroid =  sample['future_centroid']

                iou_bboxes.append(iou_bbox.get_metric(output_after_thresh,label_raw))
                iou_masks.append(iou_mask.get_metric(output_after_thresh,label_raw))

                distances_via_mean.append(distance_via_mean.get_metric(output_initial_dims, future_centroid))

        self.performance_arrays['{}_iou_bboxes'.format(set)] = iou_bboxes
        self.performance_arrays['{}_iou_masks'.format(set)] = iou_masks
        self.performance_arrays['{}_distances_via_mean'.format(set)] = distances_via_mean

        self.performance_statistics['{}_iou_bboxes'.format(set)] = (np.mean(iou_bboxes),np.std(iou_bboxes))
        self.performance_statistics['{}_iou_masks'.format(set)] = (np.mean(iou_masks), np.std(iou_masks))
        self.performance_statistics['{}_distances_via_mean'.format(set)] = (np.mean(distances_via_mean),
                                                                            np.std(distances_via_mean))


        return self.performance_statistics, self.performance_arrays


    def save_stats(self):
        model_name = self.params['model_name']
        stats_file = os.path.join(MODEL_PATH, "{}/{}_eval_stats.pickle".format(model_name, model_name))
        pickle.dump(self.performance_statistics, open(stats_file, "wb"))

    def plot_performance_histograms(self):
        model_name = self.params['model_name']
        get_histogram(self.performance_arrays, model_name,self.save_folder)


#TODO: put print statements and dont forget to flush
if __name__=='__main__':
    data_names = ['Football2_1person','Football1and2']
    for data_name in data_names:
        print('dealing with {}'.format(data_name))
        sys.stdout.flush()
        model = Unet
        model_name = "Unet_M_3ndGen_{}".format(data_name)

        model_history_file = os.path.join(MODEL_PATH, "{}/{}_history.pickle".format(model_name, model_name))
        model_file = os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))
        param_file = os.path.join(MODEL_PATH, "{}/param_holder.pickle".format(model_name))
        model_folder = os.path.join(MODEL_PATH, "{}/".format(model_name))
        dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name, data_name))

        evaluator = ModelEvaluator(model,model_file,model_history_file,param_file,model_folder,dataset_file, cpu_only=False)

        print('getting performance')
        sys.stdout.flush()
        evaluator.get_performace_stats('val')
        print('saving stats')
        sys.stdout.flush()
        evaluator.save_stats()

        print('saving histograms')
        sys.stdout.flush()
        evaluator.plot_performance_histograms()
        print('done')














