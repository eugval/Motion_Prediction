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
import matplotlib.pyplot as plt




class ModelEvaluator(object):
    def __init__(self, model,  param_file, reload_tracker = True, cpu_only = False):
        if(cpu_only):
            self.device = 'cpu'
            self.device_string = 'cpu'
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.device_string = "cuda:0"

        self.params = pickle.load(open(param_file, "rb"))

        self.training_tracker = pickle.load(open(self.params['model_history_file'], "rb"))
        if(reload_tracker):
           tracker2 = TrainingTracker(self.training_tracker.iterations_per_epoch)
           tracker2.metrics = self.training_tracker.metrics
           tracker2.saved_epoch = self.training_tracker.saved_epoch
           tracker2.finished = self.training_tracker.finished
           self.training_tracker = tracker2

        self.save_folder = self.params['model_folder']
        self.model = model(self.params['number_of_inputs'])
        self.model.load_state_dict(torch.load(self.params['model_file'], map_location = self.device_string))
        self.model.to(self.device)

        self.dataset_file = self.params['dataset_file']

        self.performance_statistics = {}
        self.performance_arrays = {}
        self.problematic_datapoints = []


    def get_performace_stats(self, set):
        dataset = DataFromH5py(self.dataset_file,self.params['idx_sets'],input_type = self.params['input_types'],
                                    purpose = set, label_type = self.params['label_type'],
                                    transform = transforms.Compose([
                                        ResizeSample(height= self.params['resize_height'], width = self.params['resize_width']),
                                        ToTensor()
                                    ]))


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
            output, output_initial_dims, output_after_thresh, label_raw, future_centroid = self.__perform_inference__(sample,raw_sample, dataset.initial_dims)

            with torch.no_grad():
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

    def plot_training_progression(self):
        plot_save_file = os.path.join(self.save_folder, "{}_training_plots.png".format(self.params['model_name']))
        self.training_tracker.plot_all(plot_save_file)

    def plot_qualitative_vis(self, trials, set, verbose = 1):

        dataset = DataFromH5py(self.dataset_file, self.params['idx_sets'], input_type=self.params['input_types'],
                               purpose=set, label_type='future_mask',
                               transform=transforms.Compose([
                                   ResizeSample(height=self.params['resize_height'], width=self.params['resize_width']),
                                   ToTensor()
                               ]))

        distance_via_mean = DistanceViaMean()

        for trial in trials:
            if (verbose > 0): print("Qualitative plot () for {} set of {}".format(trial,set,self.params['model_name'] ))
            save_path = os.path.join(self.save_folder, "Qualitative_plot_{}_on_{}_set.png".format(trial,set))

            idx = np.random.randint(len(dataset))
            sample = dataset[idx]
            raw_sample = dataset.get_raw(idx)

            output, output_initial_dims, output_after_thresh, label_raw, future_centroid = self.__perform_inference__(sample,raw_sample, dataset.initial_dims)
            predicted_centroid = distance_via_mean.get_centroid(output_initial_dims)

            centroid_list = [(future_centroid[1],future_centroid[0]), (np.round(predicted_centroid[1]),np.round(predicted_centroid[0]))]

            input_images = raw_sample['input_images']
            input_masks = raw_sample['input_masks']
            resized_input = np.squeeze(sample['input'].detach().cpu().numpy())
            resized_label = np.squeeze(sample['label'])

            delta_outputs = np.dstack([output_after_thresh>0.5, label_raw, np.zeros(label_raw.shape)])

            plt.figure(figsize=(15, 15))

            plt.subplot2grid((4, 3), (0, 0))
            plt.imshow(input_masks[0])
            plt.title("Mask at time t. Input also includes {} previous masks.".format(len(input_masks)-1))

            plt.subplot2grid((4, 3), (0, 1))
            if(len(input_images)>0):
                plt.imshow(input_images[0])
                plt.title("Image at time t. Input also includes {} previous images.".format(len(input_images) - 1))
            else:
                plt.imshow(resized_input[0])
                plt.title("Resized Mask at time t. Input also includes {} previous masks.".format(len(input_masks) - 1))

            plt.subplot2grid((4, 3), (0, 2))
            plt.imshow(label_raw)
            plt.title("Raw label at time t + {}".format(dataset.future_time))
            plt.scatter(*zip(centroid_list[0]), marker='+')

            plt.subplot2grid((4, 3), (1, 0))
            plt.imshow(resized_label)
            plt.title("Resized label at time t + {}".format(dataset.future_time))

            plt.subplot2grid((4, 3), (1, 1))
            plt.imshow(output)
            plt.title("Direct model output")

            plt.subplot2grid((4, 3), (1, 2))
            plt.imshow(output_initial_dims > 0.5)
            plt.title("Resized output and thresholded at 0.5")

            plt.subplot2grid((4, 3), (2, 0), rowspan=2, colspan=3)
            plt.scatter(*zip(*centroid_list))
            plt.imshow(delta_outputs)

            plt.annotate(
                'true centroid',
                xy=(centroid_list[0][0], centroid_list[0][1]), xytext=(20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            plt.annotate(
                'centroid mean',
                xy=(centroid_list[1][0], centroid_list[1][1]), xytext=(-20, -20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            plt.title("Difference between prediction and label along with the corresponding centroids")

            plt.tight_layout()
            plt.savefig(save_path)
            if(verbose>0): print("Done")


    def __perform_inference__(self,sample, raw_sample, dims):
        input = sample['input'].float().to(self.device)

        while (len(input.size()) < 4):
            input = input.unsqueeze(0)

        with torch.no_grad():
            if (hasattr(self.model, 'eval_forward')):
                output = self.model.eval_forward(input)
            else:
                output = self.model(input)

            output = output.detach().cpu().numpy()
            output = np.squeeze(output)

            initial_dims = (dims[1], dims[0])
            output_initial_dims = cv2.resize(output, initial_dims)
            output_initial_dims_exp = output_initial_dims
            while (len(output_initial_dims_exp.shape) < 3):
                output_initial_dims_exp = np.expand_dims(output_initial_dims_exp, 0)

            output_after_thresh = output_initial_dims_exp > 0.5
            label_raw = raw_sample['label']

            while (len(label_raw.shape) < 3):
                label_raw = np.expand_dims(label_raw, 0)

            future_centroid = sample['future_centroid']

            return output, output_initial_dims, output_after_thresh, label_raw, future_centroid


if __name__=='__main__':
    data_names = ['Football2_1person','Football1and2']
    for data_name in data_names:
        print('dealing with {}'.format(data_name))
        sys.stdout.flush()

        model = Unet
        model_name = "Unet_M_3ndGen_{}".format(data_name)

        param_file = os.path.join(MODEL_PATH, "{}/param_holder.pickle".format(model_name))


        ##### BACKWARD COMPATIBILITY ########
        params = pickle.load(open(param_file, "rb"))

        if('label_type' not in params):
            params['label_type'] = 'future_mask'

        if('dataset_file' not in params):
           params['dataset_file'] = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name, data_name))

        if('model_history_file' not in params):
            params['model_history_file'] = os.path.join(MODEL_PATH, "{}/{}_history.pickle".format(model_name, model_name))

        if ('model_folder' not in params):
            params['model_folder'] =  os.path.join(MODEL_PATH, "{}/".format(model_name))

        if ('model_file' not in params):
            params['model_file'] =  os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))

        pickle.dump(params, open(param_file, "wb"))
        ###################################


        evaluator = ModelEvaluator(model, param_file, cpu_only=False)


        print('getting performance')
        sys.stdout.flush()
        evaluator.get_performace_stats('val')
        print('saving stats')
        sys.stdout.flush()
        evaluator.save_stats()

        print('saving histograms')
        sys.stdout.flush()
        evaluator.plot_performance_histograms()

        print('Making training Plots')
        sys.stdout.flush()
        evaluator.plot_training_progression()

        print('Making Qualitative Plots')
        sys.stdout.flush()
        evaluator.plot_qualitative_vis(8,'train')
        evaluator.plot_qualitative_vis(8, 'val')

        print('FINISHED')














