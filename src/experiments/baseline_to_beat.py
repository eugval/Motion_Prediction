import os
import sys
import pickle
import numpy as np
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))
sys.path.append(os.path.join(ROOT_DIR,"deprecated"))
import h5py
from experiments.evaluation_metrics import DistanceViaMean, IoUMetric
import json

def baselines_to_beat(data_file, savepath = False):
    f = h5py.File(data_file,'r')

    datapoints = f['datapoints'].value[0]
    iou_bbox = IoUMetric(type = 'bbox')
    iou_mask = IoUMetric(type = 'mask')
    distance_via_mean = DistanceViaMean()

    mean_iou_bbox = 0
    mean_iou_mask = 0
    mean_dist = 0


    for i in range(datapoints):
        datapoint = 'datapoint{}'.format(i)
        v = f[datapoint]
        input_mask = v['masks'].value[:,:,0]
        future_mask = v['future_mask'].value
        future_centroid = v['future_centroid'].value

        mean_iou_bbox += iou_bbox.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0))
        mean_iou_mask += iou_mask.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0))
        mean_dist += distance_via_mean.get_metric(input_mask,future_centroid)

    if(savepath):
        metrics_to_beat = {'mean_iou_bbox':mean_iou_bbox/datapoints,
                         'mean_iou_mask':mean_iou_mask/datapoints,
                         'mean_dist':mean_dist/datapoints}
        pickle.dump(metrics_to_beat, open(savepath, "wb"))



    return  mean_iou_bbox/datapoints , mean_iou_mask/datapoints, mean_dist/datapoints



if __name__ == '__main__':

    #names = ['Football1and2', 'Football2_1person', 'Crossing1', 'Crossing2']
    names = ['Crossing1and2']
    generate = True
    verify = True
    save_json = True
    for name in names:
        data_file = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name, name))
        class_filtered_file = os.path.join(PROCESSED_PATH, "{}/{}_cls_filtered.hdf5".format(name, name))

        tracked_file = os.path.join(PROCESSED_PATH, "{}/{}_tracked.hdf5".format(name, name))
        tracked_file_c = os.path.join(PROCESSED_PATH, "{}/{}_tracked_c.hdf5".format(name, name))
        resized_file = os.path.join(PROCESSED_PATH, "{}/{}_resized.hdf5".format(name, name))
        dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name, name))
        set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(name, name))
        input_vis_file =os.path.join(PROCESSED_PATH, "{}/{}_input_vis.png".format(name, name))
        label_vis_file = os.path.join(PROCESSED_PATH, "{}/{}_label_vis.png".format(name, name))
        metrics_to_beat_file =  os.path.join(PROCESSED_PATH, "{}/{}_metrics_to_beat.pickle".format(name, name))
        metrics_to_beat_file_json =  os.path.join(PROCESSED_PATH, "{}/{}_metrics_to_beat.json".format(name, name))

        target_folder = os.path.join(PROCESSED_PATH, "{}/mask_images/".format(name))
        target_folder_consolidated = os.path.join(PROCESSED_PATH, "{}/tracked_images_consolidated/".format(name))
        target_folder_gauss = os.path.join(PROCESSED_PATH, "{}/tracked_images_gauss/".format(name))


        print('doing {}'.format(name))
        if(generate):
            mean_iou_bbox, mean_iou_mask, mean_dist = baselines_to_beat(dataset_file,metrics_to_beat_file)

            print('The mean iou bbox is {}'.format(mean_iou_bbox))
            print('The mean iou MASK is {}'.format(mean_iou_mask))
            print('The mean  dist is {}'.format(mean_dist))

        if(verify):
            stats = pickle.load(open(metrics_to_beat_file, "rb"))
            print(stats)
            if(save_json):
                with open(metrics_to_beat_file_json, 'w') as file:
                    file.write(json.dumps(stats))





