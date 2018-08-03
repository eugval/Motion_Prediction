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
           if(not hasattr(self.training_tracker,"baselines") or not self.training_tracker.baselines):
               tracker2.add_baselines(self.params['baselines_file'])

           self.training_tracker = tracker2

        self.intermediate_loss = self.params['intermediate_loss']
        self.save_folder = self.params['model_folder']

        model_inputs = self.params['model_inputs']
        if(isinstance(model_inputs, dict)):
            self.model =  model(**model_inputs)
        elif(isinstance(model_inputs,list)):
            self.model = model(*model_inputs)
        else:
            self.model = model(model_inputs)


        self.model.load_state_dict(torch.load(self.params['model_file'], map_location = self.device_string))
        self.model.to(self.device)
        self.model.eval()

        self.dataset_file = self.params['dataset_file']

        self.performance_statistics = {}
        self.performance_arrays = {}
        self.problematic_datapoints = []



    def get_performace_stats(self, set):
        dataset = DataFromH5py(self.dataset_file,self.params['idx_sets'],input_type = self.params['input_types'],
                                    purpose = set, label_type = self.params['label_type'], only_one_mask=self.params['only_one_mask'],
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


        average_prediction_input_iou = 0.0
        average_label_input_iou = 0.0

        iou_bboxes_on_no_movement = []
        iou_masks_on_no_movement = []
        distance_via_mean_on_no_movement = []


        iou_bboxes_on_moderate_movement = []
        iou_masks_on_moderate_movement = []
        distance_via_mean_on_moderate_movement = []

        iou_bboxes_on_high_movement = []
        iou_masks_on_high_movement = []
        distance_via_mean_on_high_movement = []





        len_data = len(dataset)
        for i in range(len_data):
            sample = dataset[i]
            raw_sample = dataset.get_raw(i)
            output, output_initial_dims, output_after_thresh, label_raw, future_centroid = self.__perform_inference__(sample,
                                                                                                                      raw_sample,
                                                                                                                      dataset.initial_dims)

            with torch.no_grad():
                iou_bbox_inference = iou_bbox.get_metric(output_after_thresh,label_raw)
                iou_mask_inference = iou_mask.get_metric(output_after_thresh,label_raw)
                dist_inference = distance_via_mean.get_metric(output_initial_dims, future_centroid)

                raw_input_mask = raw_sample['input_masks'][0]
                while (len(raw_input_mask.shape) < 3):
                    raw_input_mask = np.expand_dims(raw_input_mask, 0)

                iou_bbox_input_label = iou_bbox.get_metric(raw_input_mask,label_raw)
                iou_bbox_input_pred = iou_bbox.get_metric(raw_input_mask,output_after_thresh)

            #Add full statistics
            iou_bboxes.append(iou_bbox_inference)
            iou_masks.append(iou_mask_inference)
            distances_via_mean.append(dist_inference)

            #Add point stats
            average_prediction_input_iou += iou_bbox_input_pred
            average_label_input_iou += iou_bbox_input_label

            #Add stratified stats
            if(iou_bbox_input_label > 0.9):
                iou_bboxes_on_no_movement.append(iou_bbox_inference)
                iou_masks_on_no_movement.append(iou_mask_inference)
                distance_via_mean_on_no_movement.append(dist_inference)

            elif(iou_bbox_input_label < 0.6 and iou_bbox_input_label > 0.4):
                iou_bboxes_on_moderate_movement.append(iou_bbox_inference)
                iou_masks_on_moderate_movement.append(iou_mask_inference)
                distance_via_mean_on_moderate_movement.append(dist_inference)

            elif(iou_bbox_input_label == 0) :
                iou_bboxes_on_high_movement.append(iou_bbox_inference)
                iou_masks_on_high_movement.append(iou_mask_inference)
                distance_via_mean_on_high_movement.append(dist_inference)



        average_prediction_input_iou =  average_prediction_input_iou/len_data
        average_label_input_iou = average_label_input_iou/len_data



        self.performance_arrays['{}_iou_bboxes'.format(set)] = iou_bboxes
        self.performance_arrays['{}_iou_masks'.format(set)] = iou_masks
        self.performance_arrays['{}_distances_via_mean'.format(set)] = distances_via_mean


        if(iou_bboxes_on_no_movement):
            self.performance_arrays['{}_iou_bboxes_on_no_movement'.format(set)] = iou_bboxes_on_no_movement
            self.performance_arrays['{}_iou_masks_on_no_movement'.format(set)] = iou_masks_on_no_movement

        if(iou_bboxes_on_moderate_movement):
            self.performance_arrays['{}_iou_bboxes_on_moderate_movement'.format(set)] = iou_bboxes_on_moderate_movement
            self.performance_arrays['{}_iou_masks_on_moderate_movement'.format(set)] = iou_masks_on_moderate_movement

        if(iou_bboxes_on_high_movement):
            self.performance_arrays['{}_iou_bboxes_on_high_movement'.format(set)] = iou_bboxes_on_high_movement
            self.performance_arrays['{}_iou_masks_on_high_movement'.format(set)] = iou_masks_on_high_movement



        self.performance_statistics['{}_iou_bboxes'.format(set)] = (np.mean(iou_bboxes),np.std(iou_bboxes))
        self.performance_statistics['{}_iou_masks'.format(set)] = (np.mean(iou_masks), np.std(iou_masks))
        self.performance_statistics['{}_distances_via_mean'.format(set)] = (np.mean(distances_via_mean),
                                                                            np.std(distances_via_mean))

        if(iou_bboxes_on_no_movement):
            self.performance_statistics['{}_iou_bboxes_on_no_movement'.format(set)] = (np.mean(iou_bboxes_on_no_movement),np.std(iou_bboxes_on_no_movement))
            self.performance_statistics['{}_iou_masks_on_no_movement'.format(set)] = (np.mean(iou_masks_on_no_movement), np.std(iou_masks_on_no_movement))
            self.performance_statistics['{}_distances_via_mean_on_no_movement'.format(set)] = (np.mean(distance_via_mean_on_no_movement),
                                                                                np.std(distance_via_mean_on_no_movement))
        if(iou_bboxes_on_moderate_movement):
            self.performance_statistics['{}_iou_bboxes_on_moderate_movement'.format(set)] = (np.mean(iou_bboxes_on_moderate_movement),np.std(iou_bboxes_on_moderate_movement))
            self.performance_statistics['{}_iou_masks_on_moderate_movement'.format(set)] = (np.mean(iou_masks_on_moderate_movement), np.std(iou_masks_on_moderate_movement))
            self.performance_statistics['{}_distances_via_mean_on_moderate_movement'.format(set)] = (np.mean(distance_via_mean_on_moderate_movement),
                                                                            np.std(distance_via_mean_on_moderate_movement))
        if(iou_bboxes_on_high_movement):
            self.performance_statistics['{}_iou_bboxes_on_high_movement'.format(set)] = (np.mean(iou_bboxes_on_high_movement),np.std(iou_bboxes_on_high_movement))
            self.performance_statistics['{}_iou_masks_on_high_movement'.format(set)] = (np.mean(iou_masks_on_high_movement), np.std(iou_masks_on_high_movement))
            self.performance_statistics['{}_distances_via_mean_on_high_movement'.format(set)] = (np.mean(distance_via_mean_on_high_movement),
                                                                            np.std(distance_via_mean_on_high_movement))



        self.performance_statistics['final_saved_epoch'] = self.training_tracker.saved_epoch
        self.performance_statistics['{}_average_label_input_iou'.format(set)] = average_label_input_iou
        self.performance_statistics['{}_average_prediction_input_iou'.format(set)] = average_prediction_input_iou



        return self.performance_statistics, self.performance_arrays


    def save_stats(self):
        model_name = self.params['model_name']
        stats_file = os.path.join(MODEL_PATH, "{}/{}_eval_stats.pickle".format(model_name, model_name))
        stats_text_file = os.path.join(MODEL_PATH, "{}/{}_eval_stats.json".format(model_name, model_name))
        pickle.dump(self.performance_statistics, open(stats_file, "wb"))
        with open(stats_text_file, 'w') as file:
            file.write(json.dumps(self.performance_statistics))

    def plot_performance_histograms(self, agregate = True, set = 'val', tpe = 'bbox'):
        model_name = self.params['model_name']
        if(not agregate):
            get_histogram(self.performance_arrays, model_name,self.save_folder)
        else:
            arrays = self.performance_arrays
            agregates = {}
            for k,v in arrays.items():
                if('movement' in k and set in k and tpe in k):
                    if('high' in k):
                        agregates['high movement'] = v
                    elif('moderate'in k):
                        agregates['moderate movement'] = v
                    elif('no' in k):
                        agregates['no movment'] = v
                elif(set in k and tpe in k):
                    get_histogram({k:v},model_name, self.save_folder)

            if(agregates):
                get_histogram_same_plot(agregates, '{} distribution on different movement scales on the {} set'.format(tpe,set), self.save_folder)



    def plot_training_progression(self):
        plot_save_file = os.path.join(self.save_folder, "{}_training_plots.png".format(self.params['model_name']))
        self.training_tracker.plot_all(plot_save_file)

    def plot_qualitative_vis(self, trials, set, verbose = 1):

        dataset = DataFromH5py(self.dataset_file, self.params['idx_sets'], input_type=self.params['input_types'],
                               purpose=set, label_type='future_mask',only_one_mask=self.params['only_one_mask'],
                               transform=transforms.Compose([
                                   ResizeSample(height=self.params['resize_height'], width=self.params['resize_width']),
                                   ToTensor()
                               ]))

        distance_via_mean = DistanceViaMean()

        for trial in range(trials):
            if (verbose > 0): print("Qualitative plot {} for {} set of {}".format(trial,set,self.params['model_name'] ))
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
            label_raw = np.squeeze(label_raw)

            delta_outputs = np.dstack([output_initial_dims>0.5, label_raw, input_masks[0]/255])


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
            plt.imshow(output_initial_dims>0.5 )
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
            plt.close()
            if(verbose>0): print("Done")



    def plot_qualitative_vis_2(self, trials, set, verbose = 1):
        dataset = DataFromH5py(self.dataset_file, self.params['idx_sets'], input_type=self.params['input_types'],
                               purpose=set, label_type='future_mask',only_one_mask=self.params['only_one_mask'],
                               transform=transforms.Compose([
                                   ResizeSample(height=self.params['resize_height'], width=self.params['resize_width']),
                                   ToTensor()
                               ]))
        for trial in range(trials):
            if (verbose > 0): print("Prediction visualisation {} for {} set of {}".format(trial,set,self.params['model_name'] ))
            save_path = os.path.join(self.save_folder, "Qualitative_prediction_{}_on_{}_set.png".format(trial,set))

            corresponding_points = []
            dataset_indices = []

            while len(corresponding_points) < 2:
                corresponding_points = []
                dataset_indices = []
                idx = np.random.randint(len(dataset))
                datapoint_idx = dataset.get_datapoint_index(idx)

                dataset_indices.append(idx)
                corresponding_points.append(datapoint_idx)

                datapoint_key = 'datapoint{}'.format(datapoint_idx)
                frame_idx_reference = dataset.f[datapoint_key]['origin_frame'].value[0]

                for i in range(len(dataset)):
                    datapoint_idx = dataset.get_datapoint_index(i)
                    if(datapoint_idx in corresponding_points):
                        continue
                    datapoint_key = 'datapoint{}'.format(datapoint_idx)

                    if(dataset.f[datapoint_key]['origin_frame'].value[0] == frame_idx_reference):
                        dataset_indices.append(i)
                        corresponding_points.append(datapoint_idx)


            output_masks = []

            for idx in dataset_indices:
                sample = dataset[idx]
                raw_sample = dataset.get_raw(idx)

                output, output_initial_dims, output_after_thresh, label_raw, future_centroid = self.__perform_inference__(sample,raw_sample, dataset.initial_dims)

                output_masks.append(output_after_thresh)


            image = dataset.f['datapoint{}'.format(corresponding_points[0])]['images'].value[:,:,:,0]

            colors = self.__random_colors__(len(output_masks))

            for i in range(len(output_masks)):
                mask = output_masks[i]
                color = colors[i]
                image = self.__apply_mask__(image,mask,color)

            plt.figure(figsize=(15, 15))
            plt.imshow(image)

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            if(verbose>0): print("Done")

    #TODO: FINISH THIS
    def full_video_prediction(self, set = 'val',verbose =1):
        if (verbose > 0): print("Video Prediction visualisation for {} set of {}".format(set,self.params['model_name'] ))

        save_folder = os.path.join(self.save_folder,'{}_video_prediction'.format(set))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        dataset = DataFromH5py(self.dataset_file, self.params['idx_sets'], input_type=self.params['input_types'],
                               purpose=set, label_type='future_mask',only_one_mask=self.params['only_one_mask'],
                               transform=transforms.Compose([
                                   ResizeSample(height=self.params['resize_height'], width=self.params['resize_width']),
                                   ToTensor()
                               ]))

        frame_masks = {}

        for i in range(len(dataset)):
            datapoint_idx = dataset.get_datapoint_index(i)
            datapoint_key = 'datapoint{}'.format(datapoint_idx)

            origin_frame_idx = dataset.f[datapoint_key]['origin_frame'].value[0]
            origin_frame = 'frame{}'.format(origin_frame_idx)

            sample = dataset[i]
            raw_sample = dataset.get_raw(i)
            output, output_initial_dims, output_after_thresh, label_raw, future_centroid = self.__perform_inference__(sample,raw_sample, dataset.initial_dims)

            if origin_frame in frame_masks:
                frame_masks[origin_frame].append(output_after_thresh)
            else :
                frame_masks[origin_frame] = [output_after_thresh]

            frame_data = self.dataset_file.replace('_dataset','_resized')

            f = h5py.File(frame_data,'r')

            color= self.__one_color__()

            for k,v in frame_masks.items():
                image = f[k]['image'].value



                for i in range(len(v)):
                    mask = v[i]
                    image = self.__apply_mask__(image,mask,color)

                _, ax = plt.subplots(1)
                ax.axis('off')
                ax.imshow(image)
                ax.margins(0)

                height, width = image.shape[:2]
                ax.set_ylim(height, 0)
                ax.set_xlim(0, width)

                save_path = os.path.join(save_folder, "{}.png".format(k))
                plt.savefig(save_path, bbox_inches='tight',  pad_inches = 0)
                plt.close()
        if(verbose>0): print("Done")






    def __random_colors__(self,N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def __one_color__(self, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = (1 / 2, 1, brightness)
        color =  colorsys.hsv_to_rgb(*hsv)
        return color


    def __apply_mask__(self,image, mask, color):

        alpha = 0.4
        for c in range(3):
            image[:, :, c] =  image[:, :, c] * (1 - alpha*mask) + alpha* mask * color[c] * 255
        return image

    def __perform_inference__(self,sample, raw_sample, dims):
        input = sample['input'].float().to(self.device)

        while (len(input.size()) < 4):
            input = input.unsqueeze(0)

        with torch.no_grad():
            if (hasattr(self.model, 'eval_forward')):
                output = self.model.eval_forward(input)
            else:
                output = self.model(input)


            # ########   DEBUG  ###########
            # output = torch.squeeze(output, 1)
            # output = output.detach().cpu().numpy()
            # output_after_thresh = output > 0.5
            # output = np.squeeze(output)
            #
            # output_initial_dims = output
            #
            # label_raw = sample['label'].detach().cpu().numpy()
            #
            # while (len(label_raw.shape) < 3):
            #     label_raw = np.expand_dims(label_raw, 0)
            # label_raw.astype('bool')
            #
            ##############################################

            if(self.intermediate_loss):
                output = output[0]

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
    data_names = [('Crossing1', 4 )]# ('Crossing1', 1),('Football2_1person',1) ('Football1and2', 2)
    for data_name, number in data_names:
        print('dealing with {} {}'.format(data_name, number))
        sys.stdout.flush()


        evaluate_perf = True
        make_histograms = True
        make_training_plots = True
        make_qual_plots = True
        make_qual_plots_2 = True
        vid_pred = False


        model = SpatialUnet2
        m = model(12)
        print(m)
        model_name = "SpatialUnet2_MI_{}_{}".format(data_name, number)


        param_file = os.path.join(MODEL_PATH, "{}/param_holder.pickle".format(model_name))


        ##### BACKWARD COMPATIBILITY ########
        params = pickle.load(open(param_file, "rb"))

        if('label_type' not in params):
            params['label_type'] = 'future_mask'

        if('model_inputs' not in params):
            params['model_inputs'] = [params['number_of_inputs']]

        if('dataset_file' not in params):
           params['dataset_file'] = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name, data_name))

        if('model_history_file' not in params):
            params['model_history_file'] = os.path.join(MODEL_PATH, "{}/{}_history.pickle".format(model_name, model_name))

        if ('model_folder' not in params):
            params['model_folder'] =  os.path.join(MODEL_PATH, "{}/".format(model_name))

        if ('model_file' not in params):
            params['model_file'] =  os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))

        if ('baselines_file' not in params):
            params['baselines_file'] =  os.path.join(PROCESSED_PATH, "{}/{}_metrics_to_beat.pickle".format(data_name,data_name))

        if('intermediate_loss' not in params):
            params['intermediate_loss'] = False

        pickle.dump(params, open(param_file, "wb"))
        ###################################


        evaluator = ModelEvaluator(model, param_file, cpu_only = False)

        print(evaluator.device)


        if(evaluate_perf):
            print('getting performance')
            sys.stdout.flush()
            evaluator.get_performace_stats('val')
            print('saving stats')
            sys.stdout.flush()
            evaluator.save_stats()

        if(make_histograms):
            print('saving histograms')
            sys.stdout.flush()
            evaluator.plot_performance_histograms(tpe ='bbox')
            evaluator.plot_performance_histograms(tpe ='masks')
            evaluator.plot_performance_histograms(tpe ='distances')


        if(make_training_plots):
            print('Making training Plots')
            sys.stdout.flush()
            evaluator.plot_training_progression()

        if(make_qual_plots):
            print('Making Qualitative Plots')
            sys.stdout.flush()
            #evaluator.plot_qualitative_vis(5,'train')
            evaluator.plot_qualitative_vis(10, 'val')

        if(make_qual_plots_2):
            print('Making Qualitative Plots')
            sys.stdout.flush()
            #evaluator.plot_qualitative_vis(5,'train')
            evaluator.plot_qualitative_vis_2(5, 'val')

        if(vid_pred):
            evaluator.full_video_prediction('val')

        print('FINISHED')

#main_func()














