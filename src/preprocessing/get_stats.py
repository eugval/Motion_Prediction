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
                          'iou_mask_mean': np.mean(ious_mask),
                          'iou_mask_sdt': np.std(ious_mask),
                          'iou_mask': ious_mask,
                          'iou_bbox_mean': np.mean(ious_bbox),
                          'iou_bbox_std': np.std(ious_bbox),
                          'iou_bbox': ious_bbox,
                          }

    if(save_path):
        pickle.dump(displacement_stats, open(save_path, "wb"))

    return displacement_stats



def get_data_stats_with_idx_sets(data_file,idx_sets_file, save_path = False):
    f = h5py.File(data_file, "r")

    iou_bbox = IoUMetric(type = 'bbox')
    iou_mask = IoUMetric(type = 'mask')


    displacement_stats = {}


    idx_sets = pickle.load( open(idx_sets_file, "rb" ) )

    for k, v  in idx_sets.items():
        distances = []
        ious_mask = []
        ious_bbox = []
        for i, idx in enumerate(v):
            datapoint = 'datapoint{}'.format(idx)

            centroid = f[datapoint]['centroids'].value[0]
            future_centroid = f[datapoint]['future_centroid'].value
            input_mask = f[datapoint]['masks'].value[:,:,0]
            future_mask = f[datapoint]['future_mask'].value

            distances.append(np.linalg.norm(future_centroid-centroid))
            ious_bbox.append(iou_bbox.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0)))
            ious_mask.append(iou_mask.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0)))

        displacement_stats[k] = {'dist_mean': np.mean(distances),
                                 'dis_std': np.std(distances),
                                 'dist': distances,
                                 'iou_mask_mean': np.mean(ious_mask),
                                 'iou_mask_sdt': np.std(ious_mask),
                                 'iou_mask': ious_mask,
                                 'iou_bbox_mean': np.mean(ious_bbox),
                                 'iou_bbox_std': np.std(ious_bbox),
                                 'iou_bbox': ious_bbox,
                                 }


    if(save_path):
        pickle.dump(displacement_stats, open(save_path, "wb"))

    return displacement_stats



def get_histogram(stats, data_name, save_folder, suffix = ''):
    for k, v in stats.items():
        if(type(v)==list):
            plt.figure()
            plt.hist(v, bins=20, range=(min(v),max(v)))
            if('iou' in k):
                plt.xlim(0, 1)
            save_path = os.path.join(save_folder, '{}_{}_ditribution{}.png'.format(data_name, k,suffix))
            plt.savefig(save_path)
            plt.close()


def get_histogram_same_plot(stats, title, save_folder, nomal_x_lim = False):
    plt.figure()
    for k, v in stats.items():
        if(type(v)==list):
            plt.hist(v, bins=20, range=(min(v),max(v)), alpha=0.3, label='{}'.format(k))

            if(nomal_x_lim or 'iou' in k or 'mask' in k or 'bbox' in k  ):
                plt.xlim(0, 1)
    plt.legend(loc='upper right')
    save_path = os.path.join(save_folder, '{}.png'.format(title))
    plt.savefig(save_path)
    plt.close()


def get_subfig_historgram(stats, name, save_folder):

    plt.figure(figsize=(16,4))
    if(name == 'Crossing1'):
        plt.suptitle('CRS statistics', fontsize=18 )#fontweight='bold'
    elif(name == 'Crossing1_lt'):
        plt.suptitle('CRSLT statistics', fontsize=18)
    elif(name == 'Crossing1and2_lt'):
        plt.suptitle('LCRSLT statistics', fontsize=18)
    elif(name == 'Crossing1and2'):
        plt.suptitle('LCRS statistics', fontsize=18)
    elif(name== 'Football1and2'):
        plt.suptitle('FBL statisics', fontsize=18)
    elif(name== 'Football1and2_lt'):
        plt.suptitle('FBLLT statistics', fontsize=18)


    count = 1
    for metric_type, stat in stats.items():
        nomal_x_lim = True
        if('Dis' in metric_type or 'dis' in metric_type):
            nomal_x_lim = False

        plt.subplot(1, 3, count)
        for k, v in stat.items():
            if(k == 'val'):
                leg = 'test'
            else:
                leg = k

            if(type(v)==list):
                plt.hist(v, bins=20, range=(min(v),max(v)), alpha=0.3, label='{}'.format(leg))

                if(nomal_x_lim or 'iou' in k or 'mask' in k or 'bbox' in k  ):
                    plt.xlim(0, 1)

        if(count==1):
            plt.ylabel('Number of examples')


        plt.legend(loc='upper right')
        plt.title(metric_type)
        count +=1
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(save_folder, '{}_all_stats.png'.format(name))
    plt.savefig(save_path)
    plt.close()




if __name__ =='__main__'  :
    data_names = ['Crossing1', 'Crossing1_lt', 'Crossing1and2', 'Crossing1and2_lt','Football1and2', 'Football1and2_lt' ] # 'Football2_1person' 'Football1and2', 'Crossing1','Crossing2'

    generate = False
    plot = True
    genreate_with_mvnt = False
    plot_with_mvnt = False

    inspect = False


    for name in data_names:
        data_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name,name))
        save_path = os.path.join(PROCESSED_PATH, "{}/{}_stats.pickle".format(name,name))
        save_folder = os.path.join(PROCESSED_PATH, "{}/".format(name))
        set_idx_file_high_movement = os.path.join(PROCESSED_PATH, "{}/{}_sets_high_movement.pickle".format(name, name))
        set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(name, name))
        save_path_high_movement = os.path.join(PROCESSED_PATH, "{}/{}_stats_high_movement.pickle".format(name,name))


        if(generate):
            print('doing {}'.format(name))
            sys.stdout.flush()
            stats = get_data_stats_with_idx_sets(data_file,set_idx_file, save_path = save_path)

            for set in ['train','val']:
                print('{}_dist_mean:{}'.format(set,stats[set]['dist_mean']))
                print('{}_dis_std:{}'.format(set,stats[set]['dis_std']))
                print('{}_iou_mask_mean:{}'.format(set,stats[set]['iou_mask_mean']))
                print('{}_iou_mask_sdt:{}'.format(set,stats[set]['iou_mask_sdt']))
                print('{}_iou_bbox_mean:{}'.format(set,stats[set]['iou_bbox_mean']))
                print('{}_iou_bbox_std:{}'.format(set,stats[set]['iou_bbox_std']))
            sys.stdout.flush()

        if(genreate_with_mvnt):
            print('doing {} with the high movement idx set'.format(name))
            sys.stdout.flush()
            stats = get_data_stats_with_idx_sets(data_file,set_idx_file_high_movement, save_path = save_path_high_movement)

            for set in ['train','val']:
                print('{}_dist_mean:{}'.format(set,stats[set]['dist_mean']))
                print('{}_dis_std:{}'.format(set,stats[set]['dis_std']))
                print('{}_iou_mask_mean:{}'.format(set,stats[set]['iou_mask_mean']))
                print('{}_iou_mask_sdt:{}'.format(set,stats[set]['iou_mask_sdt']))
                print('{}_iou_bbox_mean:{}'.format(set,stats[set]['iou_bbox_mean']))
                print('{}_iou_bbox_std:{}'.format(set,stats[set]['iou_bbox_std']))
            sys.stdout.flush()




        if(plot):
            print("looking at {}".format(name))
            stats = pickle.load(open(save_path, "rb"))
            print(stats.keys())

            iou_bbox = {'train': stats['train']['iou_bbox'],
                        'val': stats['val']['iou_bbox']}

            iou_mask = {'train': stats['train']['iou_mask'],
                        'val': stats['val']['iou_mask']}

            dist = {'train': stats['train']['dist'],
                        'val': stats['val']['dist']}


            all_stats ={'centroid displacement': dist, 'mIoU': iou_mask, 'bbIoU': iou_bbox}

            get_subfig_historgram(all_stats,name,save_folder)
            #get_histogram_same_plot(iou_bbox,'iou_bbox_distributions',save_folder, True)
            #get_histogram_same_plot(iou_mask,'iou_mask_distributions',save_folder, True)
            #get_histogram_same_plot(dist,'centroid_distance_distributions',save_folder)



        if(plot_with_mvnt):
            print("looking at {}".format(name))
            stats = pickle.load(open(save_path_high_movement, "rb"))
            print(stats.keys())

            stats_retuned = {'train_bbox': stats['train']['iou_bbox'],
                        'val_bbox': stats['val']['iou_bbox'],
                       'train_mask': stats['train']['iou_mask'],
                        'val_mask': stats['val']['iou_mask']}

            get_histogram_same_plot(stats_retuned,'high-movement trucated distributions',save_folder, True)


        if(inspect):
            print("looking at {}".format(name))
            stats = pickle.load(open(save_path_high_movement, "rb"))
            for k,v in stats.items():
                print(k)
                for k,v in v.items():
                    if(not type(v) is list and not isinstance(v, np.ndarray)):
                        print("{} : {}".format(k,v))












