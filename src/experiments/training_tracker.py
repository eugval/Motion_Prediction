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
            x_ticks = np.arange(len(v)) * self.iterations_per_epoch
            plt.plot(x_ticks, v)
            plt.xlabel("Iterations (no of batches)")
            plt.ylabel(k)
            plt.title(k + " vs number of iterations")

            i += 1

        plt.tight_layout()

        if (save_path):
            plt.savefig(save_path)
        else:
            plt.show()


if __name__ == '__main__':

    save = True

    data_name = 'Football2_1person'
    model_name = "Unet_{}".format(data_name)

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
