import os
import sys
ROOT_DIR = os.path.abspath("../")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))
sys.path.append(os.path.join(ROOT_DIR,"deprecated"))

import pickle
import matplotlib.pyplot as plt
import numpy as np



class TrainingTracker(object):
    def __init__(self, iterations_per_epoch):
        self.metrics = {}
        self.iterations_per_epoch=iterations_per_epoch
        self.saved_epoch = 0
        self.finished = False
        self.baselines = False

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
            x_ticks = np.arange(len(v)) * self.iterations_per_epoch
            plt.plot(x_ticks, v)
            if('iou' in k):
                plt.ylim(0, 1)



            plt.xlabel("Iterations (no of batches with {} batches per epoch)".format(self.iterations_per_epoch))
            plt.ylabel(k)
            plt.title(k + " vs number of iterations")

            plt.axvline(x=self.saved_epoch *self.iterations_per_epoch, ymin=0.0, ymax=1.0, color='b')

            if('iou_bbox' in k):
                plt.axhline(y=self.baselines['mean_iou_bbox'], xmin=0.0, xmax=1.0, color='r')
            elif('iou_val' in k):
                plt.axhline(y=self.baselines['mean_iou_mask'], xmin=0.0, xmax=1.0, color='r')
            elif('dist' in k):
                plt.axhline(y=self.baselines['mean_dist'], xmin=0.0, xmax=1.0, color='r')

            i += 1

        plt.tight_layout()

        if (save_path):
            plt.savefig(save_path)
        else:
            plt.show()

    def record_saving(self):
        self.saved_epoch += 1

    def record_finished_training(self):
        self.finished = True

    def add_baselines(self, baseline_file):
        self.baselines = pickle.load(open(baseline_file, "rb"))



if __name__ == '__main__':

    save = True
    data_name = 'Football2_1person'
    model_name = "Unet_B_3ndGen_{}".format(data_name)

    model_folder = os.path.join(MODEL_PATH, "{}/".format(model_name))
    model_file = os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))
    model_history_file = os.path.join(MODEL_PATH, "{}/{}_history.pickle".format(model_name,model_name))
    plot_save_file = os.path.join(MODEL_PATH,"{}/{}_training_plots.png".format(model_name,model_name))

    training_tracker = pickle.load(open(model_history_file, "rb"))

    tracker2 = TrainingTracker(training_tracker.iterations_per_epoch)
    tracker2.metrics = training_tracker.metrics



    if(save ==True):
        tracker2.plot_all(plot_save_file)
    else:
        tracker2.plot_all()
