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

    def evaluate(self, model, dataset,  device, eval_percent=1):
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

    def evaluate(self,model, criterion, dataset, device, eval_percent=1):
        num_examples = int(len(dataset)*eval_percent)
        indices = np.random.permutation(num_examples)

        tot_loss = 0.0
        for idx in indices:
            sample = dataset[idx]

            input = sample['input'].float().to(device)
            label = sample['label'].float().to(device)
            while (len(input.size()) < 4):
                input= input.unsqueeze(0)

            while(len(label.size())<4):
                label = label.unsqueeze(0)

            output = model(input)

            tot_loss += criterion(output,label).item()


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
