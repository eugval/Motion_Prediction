import os
import sys
import matplotlib
matplotlib.use('Agg')
import pickle
import h5py
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))

PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")

from experiments.evaluation_metrics import IoUMetric
import numpy as np


def get_data_stats(data_file, save_path = False):
    f = h5py.File(data_file, "r")

    no_data = f['datapoints'].value[0]
    iou_bbox = IoUMetric(type = 'bbox')
    iou_mask = IoUMetric(type = 'mask')

    distances = []
    ious_mask = []
    ious_bbox = []
    for i in range(no_data):
        datapoint = 'datapoint{}'.format(i)

        centroid = f[datapoint]['centroids'].value[0]
        future_centroid = f[datapoint]['future_centroid'].value
        input_mask = f[datapoint]['masks'].value[:,:,0]
        future_mask = f[datapoint]['future_mask'].value

        distances.append(np.linalg.norm(future_centroid-centroid))
        ious_bbox.append(iou_bbox.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0)))
        ious_mask.append(iou_mask.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0)))



    displacement_stats = {'dist_mean': np.mean(distances),
                          'dis_std': np.std(distances),
                            'dist': distances,
                          'iou_mask_mean':np.mean(ious_mask),
                          'iou_mask_sdt':np.std(ious_mask),
                          'iou_mask':ious_mask,
                          'iou_bbox_mean':np.mean(ious_bbox),
                          'iou_bbox_std':np.std(ious_bbox),
                          'iou_bbox':ious_bbox,
                          }

    if(save_path):
        pickle.dump(displacement_stats, open(save_path, "wb"))

    return displacement_stats



def get_data_stats_with_idx_sets(data_file,idx_sets_file, save_path = False):
    f = h5py.File(data_file, "r")

    iou_bbox = IoUMetric(type = 'bbox')
    iou_mask = IoUMetric(type = 'mask')

    distances = []
    ious_mask = []
    ious_bbox = []






    idx_sets = pickle.load( open(idx_sets_file, "rb" ) )

    for k, v  in idx_sets.items():
        for i, idx in enumerate(v):
            datapoint = 'datapoint{}'.format(idx)

            centroid = f[datapoint]['centroids'].value[0]
            future_centroid = f[datapoint]['future_centroid'].value
            input_mask = f[datapoint]['masks'].value[:,:,0]
            future_mask = f[datapoint]['future_mask'].value

            distances.append(np.linalg.norm(future_centroid-centroid))
            ious_bbox.append(iou_bbox.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0)))
            ious_mask.append(iou_mask.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0)))



    displacement_stats = {'dist_mean': np.mean(distances),
                          'dis_std': np.std(distances),
                            'dist': distances,
                          'iou_mask_mean':np.mean(ious_mask),
                          'iou_mask_sdt':np.std(ious_mask),
                          'iou_mask':ious_mask,
                          'iou_bbox_mean':np.mean(ious_bbox),
                          'iou_bbox_std':np.std(ious_bbox),
                          'iou_bbox':ious_bbox,
                          }

    if(save_path):
        pickle.dump(displacement_stats, open(save_path, "wb"))

    return displacement_stats



def get_histogram(stats, data_name, save_folder, suffix = ''):
    for k, v in stats.items():
        if(type(v)==list):
            plt.figure()
            plt.hist(v, bins=20, range=(min(v),max(v)))
            plt.title('{} {} distribution'.format(data_name,k))
            if('iou' in k):
                plt.xlim(0, 1)
            save_path = os.path.join(save_folder, '{}_{}_ditribution{}.png'.format(data_name, k,suffix))
            plt.savefig(save_path)
            plt.close()






if __name__ =='__main__'  :
    data_names = ['Football1and2', 'Crossing1' ] # 'Football2_1person' 'Football1and2', 'Crossing1','Crossing2'

    generate = False
    verify = False
    genreate_with_idx = True


    for name in data_names:
        data_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name,name))
        save_path = os.path.join(PROCESSED_PATH, "{}/{}_stats.pickle".format(name,name))
        save_folder = os.path.join(PROCESSED_PATH, "{}/".format(name))
        set_idx_file_high_movement = os.path.join(PROCESSED_PATH, "{}/{}_sets_high_movement.pickle".format(name, name))
        save_path_high_movement = os.path.join(PROCESSED_PATH, "{}/{}_stats_high_movement.pickle".format(name,name))

        if(generate):
            print('doing {}'.format(name))
            sys.stdout.flush()
            stats = get_data_stats(data_file, save_path)
            print('dist_mean:{}'.format(stats['dist_mean']))
            print('dis_std:{}'.format(stats['dis_std']))
            print('iou_mask_mean:{}'.format(stats['iou_mask_mean']))
            print('iou_mask_sdt:{}'.format(stats['iou_mask_sdt']))
            print('iou_bbox_mean:{}'.format(stats['iou_bbox_mean']))
            print('iou_bbox_std:{}'.format(stats['iou_bbox_std']))
            sys.stdout.flush()



            get_histogram(stats,name,save_folder)


        if(genreate_with_idx):
            print('doing {} with the higj movement idx set'.format(name))
            sys.stdout.flush()
            stats = get_data_stats_with_idx_sets(data_file,set_idx_file_high_movement, save_path = save_path_high_movement)
            print('dist_mean:{}'.format(stats['dist_mean']))
            print('dis_std:{}'.format(stats['dis_std']))
            print('iou_mask_mean:{}'.format(stats['iou_mask_mean']))
            print('iou_mask_sdt:{}'.format(stats['iou_mask_sdt']))
            print('iou_bbox_mean:{}'.format(stats['iou_bbox_mean']))
            print('iou_bbox_std:{}'.format(stats['iou_bbox_std']))
            sys.stdout.flush()

            get_histogram(stats,name,save_folder, suffix ='_high_movement')



        if(verify):
            print("looking at {}".format(name))
            stats = pickle.load(open(save_path, "rb"))
            print(stats)














