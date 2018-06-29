import numpy as np
import cv2

class DistanceViaMean(object):
    name = 'dist_via_mean'
    def get_metric(self, grid, centroid):
        grid_dims = grid.shape

        x = np.broadcast_to(np.arange(grid_dims[0]).reshape(grid_dims[0], 1), (grid_dims[0], grid_dims[1]))
        y = np.broadcast_to(np.arange(grid_dims[1]), (grid_dims[0], grid_dims[1]))
        d = np.dstack((x, y))

        positions = np.reshape(d, (-1, 2))
        w = [grid[tuple(v)] for v in positions]

        mean_x = np.average(positions[:,0], weights = w)
        mean_y = np.average(positions[:,1], weights = w)

        dist = np.linalg.norm(centroid - np.array([mean_x,mean_y]))

        return dist

    def evaluate(self, model, dataset,  device, eval_percent=1):
        num_examples = int(len(dataset)*eval_percent)
        indices = np.random.permutation(num_examples)

        tot_dist = 0.0
        for idx in indices:
            sample = dataset[idx]
            input = sample['input'].float().to(device)
            centroid = sample['future_centroid']

            output = model(input)

            output = output.detach().cpu().numpy()
            output = np.squeeze(output)
            output = cv2.resize(output, dataset.initial_dims)

            centroid = centroid.numpy()
            tot_dist += self.get_metric(output,centroid)

        return tot_dist / float(num_examples)


class DistanceViaMode(object):
    name = 'dist_via_mode'
    def get_metric(self, grid, centroid):
        mode_coords = np.unravel_index(np.argmax(grid), grid.shape)
        dist = np.linalg.norm(centroid - mode_coords)
        return dist

    def evaluate(self, model, dataset, device, eval_percent=1):
        num_examples = int(len(dataset)*eval_percent)
        indices = np.random.permutation(num_examples)

        tot_dist = 0.0
        for idx in indices:
            sample = dataset[idx]
            input = sample['input'].float().to(device)
            centroid = sample['future_centroid']

            output = model(input)

            output = output.detach().cpu().numpy()
            output = np.squeeze(output)
            output = cv2.resize(output, dataset.initial_dims)


            centroid = centroid.numpy()
            tot_dist += self.get_metric(output,centroid)

        return tot_dist / float(num_examples)




class LossMetric(object):
    name = 'loss'

    def evaluate(self,model, criterion, dataset, device, eval_percent):
        num_examples = int(len(dataset)*eval_percent)
        indices = np.random.permutation(num_examples)

        tot_loss = 0.0
        for idx in indices:
            sample = dataset[idx]
            input = sample['input'].float().to(device)
            label = sample['label'].float().to(device)

            output = model(input)

            tot_loss += criterion(output,label).item()


        return tot_loss / float(num_examples)









#
#
# class TrainingTracker(object):
#     def __init__(self,metrics_list):
#         self.metrics = {}
#         for metric in metrics_list:
#             self.metrics[metric]=[]
#
#
#
#
#
#
#
#
# class TrainingTracker(object):
#
#     def __init__(self, metrics_list,set_list, batch_size, num_datapoints ):
#
#         self.metrics = {}
#         for metric in metrics_list:
#             self.metrics[metric]= []
#
#
#     def add(self, value,  metric_name):
#         self.metrics[metric_name].append(value)
#
#     def plot_all(self):
#         for name in self.metrics.keys():
#             self.plot_metric(name)
#
#     def plot_metric(self, metric_name, units = '', epochs = False, log_scale = False ):
#
#
#
#
# #TODO: make it work with general metrics
# class TrainingTracker(object):
#     def __init__(self):
#         self.running_loss = []
#         self.running_mean_distance = []
#         self.running_mode_distance = []
#
#     def add_distance(self, batch_outputs, batch_true_centroids, mode):
#         if(mode == "mean"):
#             self.running_mean_distance.append(self.get_mean_distance(batch_outputs,batch_true_centroids, mode))
#
#         elif(mode=="mode"):
#             self.running_mode_distance.append(self.get_mean_distance(batch_outputs,batch_true_centroids,mode))
#         else:
#             raise ValueError("Can only do mean and mode distances")
#
#     def add_loss(self, loss):
#         self.running_loss.append(loss)
#
#     def plot_metrics(self, x_label, y_label, title):
#         plt.plot(np.arange(len(self.running_loss)), self.running_loss, label = 'running_loss' )
#         plt.plot(np.arange(len(self.running_mean_distance)), self.running_mean_distance, label='running_mean_dist')
#         plt.plot(np.arange(len(self.running_mode_distance)), self.running_mode_distance, label='running_mode_dist')
#         plt.xlabel(x_label)
#         plt.ylabel(y_label)
#
#         plt.title(title)
#
#         plt.legend()
#
#         plt.show()
#
#
# #TODO: fix the resizing before the image calc
#     def get_mean_distance(self,batch_outputs,batch_c_true, mode = "mean"):
#
#         np_pred = batch_outputs.detach().cpu().numpy()
#         np_pred = np.squeeze(np_pred, axis =1)
#         np_c_true = batch_c_true.numpy()
#
#         if(mode == "mean"):
#             batch_c_pred =np.stack([self.get_mean_centroid(np_pred[i,:,:]) for i in range(np_pred.shape[0])])
#
#         if(mode =='mode'):
#             batch_c_pred = np.stack([self.get_mode_centroid(np_pred[i, :, :]) for i in range(np_pred.shape[0])])
#
#         dist = np.linalg.norm(batch_c_pred - np_c_true, axis=1)
#
#         return np.mean(dist)
#
#
#     def get_mean_centroid(self, grid):
#         grid_dims = grid.shape
#         x = np.broadcast_to(np.arange(grid_dims[0]).reshape(grid_dims[0], 1), (grid_dims[0], grid_dims[1]))
#         y = np.broadcast_to(np.arange(grid_dims[1]), (grid_dims[0], grid_dims[1]))
#         d = np.dstack((x, y))
#         positions = np.reshape(d, (-1, 2))
#         w = np.zeros(positions.shape[0])
#
#         for i, v in enumerate(positions):
#             w[i]= grid[tuple(v)]
#
#         mean_x = np.average(positions[:,0], weights = w)
#         mean_y = np.average(positions[:,1], weights = w)
#
#         return np.array([mean_x,mean_y])
#
#     def get_mode_centroid(self, grid):
#
#         return np.unravel_index(np.argmax(grid), grid.shape)
