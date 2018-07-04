import os
ROOT_DIR = os.path.abspath("../")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")


import pickle
import matplotlib.pyplot as plt
import numpy as np



class TrainingTracker(object):
    def __init__(self, iterations_per_epoch):
        self.metrics = {}
        self.iterations_per_epoch=iterations_per_epoch

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


if __name__ == '__name__':

    save = False

    data_name = 'Football1_sm'
    model_name = "Unet_{}".format(data_name)

    model_folder = os.path.join(MODEL_PATH, "{}/".format(model_name))
    model_file = os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))
    model_history_file = os.path.join(MODEL_PATH, "{}/{}_history.pickle".format(model_name,model_name))
    plot_save_file = os.path.join(MODEL_PATH,"{}/{}_training_plots.png".format(model_name,model_name))

    training_tracker = pickle.load(open(model_history_file, "rb"))

    if(save ==True):
        training_tracker.plot_all(plot_save_file)
    else:
        training_tracker.plot_all()
