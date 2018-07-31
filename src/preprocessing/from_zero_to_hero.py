import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))
import matplotlib
matplotlib.use('Agg')

import h5py
from preprocessing.detection import run_mask_rcnn, generate_save_dir_path , generate_image_dir_path

from preprocessing.tracking import  track, consolidate_indices, visualise_tracks, iou_track
from preprocessing.discard import  score_and_pos_discard, class_and_size_discard

from preprocessing.make_gaussians import make_gaussian_masks, visualise_gaussians, add_centroids

from preprocessing.make_dataset import make_dataset
from preprocessing.manipulate_dataset import MakeDataSplits, MakeDataSplitsWithMerge
from preprocessing.resize_data import resize_data
from preprocessing.get_stats import get_data_stats, get_histogram, get_data_stats_with_idx_sets, get_histogram_same_plot
import time
import pickle



os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import os
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

names = [ ("Football1and2_lt",2) ] # Football1and2, Football2_1person , Crossing1, Crossing2, Football1and2_lt
detecting = False
discarding = False
tracking = False
resizing = False
calculate_centroids = False
make_gaussians = False
dataset = False
mk_idx = False
mk_idx_merge = False
mk_stats = True
mask_vis = False
gauss_vis = False

future_time = 10
sparse_sampling = 5
merged_data = False


start_time = time.time()


for name, config in names:
    print("dealing with {} ...".format(name))
    sys.stdout.flush()
    #Directory to fetch the raw images
    raw_dir_name = name

    #Directory to save the processed data
    processed_dir_name = name
    processed_dir = generate_save_dir_path(processed_dir_name)

    if(detecting):
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)


    #Files and folders in the processed directory
    detected_file = "{}.hdf5".format(processed_dir_name)
    data_file = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name,name))
    class_filtered_file = os.path.join(PROCESSED_PATH, "{}/{}_cls_filtered.hdf5".format(name, name))
    tracked_file = os.path.join(PROCESSED_PATH, "{}/{}_tracked.hdf5".format(name,name))
    resized_file = os.path.join(PROCESSED_PATH, "{}/{}_resized.hdf5".format(name,name))
    dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name,name))
    set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(name,name))
    target_folder_consolidated = os.path.join(PROCESSED_PATH, "{}/tracked_images_consolidated/".format(name))
    target_folder_gauss = os.path.join(PROCESSED_PATH, "{}/tracked_images_gauss/".format(name))
    stats_file = os.path.join(PROCESSED_PATH, "{}/{}_stats.pickle".format(name,name))
    target_folder = os.path.join(PROCESSED_PATH, "{}/".format(name))

    if (detecting):
        print("Detecting...")
        sys.stdout.flush()
        run_mask_rcnn(detected_file,
                      processed_dir,
                      generate_image_dir_path(raw_dir_name), save_images=True)

        print("--- %s seconds elapsed ---" % (time.time() - start_time))

    if(discarding):
        print("Discarding Masks...")
        sys.stdout.flush()
        if(config == 1):
            class_and_size_discard(data_file,class_filtered_file, masks_to_keep=['car'], small_threshold = 50, global_stats=False)
            score_and_pos_discard(class_filtered_file, tracked_file, [('car', 0.85)])
        elif(config==2):
            class_and_size_discard(data_file, class_filtered_file,masks_to_keep=['person'] )
            score_and_pos_discard(class_filtered_file, tracked_file, [('person', 0.99)], positions={'y_min': 150})

        print("--- %s seconds elapsed ---" % (time.time() - start_time))

    if(tracking):
        print("Tracking...")
        sys.stdout.flush()
        if (config == -1):
            track(tracked_file)

            print("--- %s seconds elapsed ---" % (time.time() - start_time))

            print("Tracking reverse...")
            track(tracked_file, reverse=True)

            print("--- %s seconds elapsed ---" % (time.time() - start_time))

            print("Consolidating...")
            consolidate_indices(tracked_file)
            print("--- %s seconds elapsed ---" % (time.time() - start_time))


        if(config==2 or config==1):
            iou_track(tracked_file)
            print("--- %s seconds elapsed ---" % (time.time() - start_time))


    if(resizing):
        print("Resizing...")
        sys.stdout.flush()
        resize_data(tracked_file, resized_file, 256, 512, maintain_ratio=True)
    else:
        if not os.path.exists(resized_file):
            resized_file = tracked_file
    print("--- %s seconds elapsed ---" % (time.time() - start_time))


    if(calculate_centroids):
        print("Calculating centroids...")
        sys.stdout.flush()
        add_centroids(resized_file, f = None,  method= 'masks')
        print("--- %s seconds elapsed ---" % (time.time() - start_time))#
        sys.stdout.flush()

    if(make_gaussians):
        print("Making gaussian masks...")
        sys.stdout.flush()

        make_gaussian_masks(resized_file)
        print("--- %s seconds elapsed ---" % (time.time() - start_time))#

    if(dataset):
        print("Making the dataset...")
        sys.stdout.flush()
        make_dataset(resized_file, dataset_file, future_time= future_time, sparse_sampling = sparse_sampling, merged_data = merged_data)
        print("--- %s seconds elapsed ---" % (time.time() - start_time))

    if(mk_idx):
        print("Making the dataset train test split...")
        sys.stdout.flush()
        data_splitter = MakeDataSplits(dataset_file, resized_file)
        data_splitter.make_frame_split('test', 0)
        data_splitter.make_frame_split('val', 0.1,save_path=set_idx_file)
        print("--- %s seconds elapsed ---" % (time.time() - start_time))
    if(mk_idx_merge):
        print("Making the dataset train test split for merged dataset...")
        sys.stdout.flush()
        data_splitter = MakeDataSplitsWithMerge(dataset_file, resized_file)
        data_splitter.make_frame_split('test', 0)
        data_splitter.make_frame_split('val', 0.1,save_path=set_idx_file)
        print("--- %s seconds elapsed ---" % (time.time() - start_time))


    if(mk_stats):
        print("Making Stats...")
        sys.stdout.flush()
        get_data_stats_with_idx_sets(dataset_file,set_idx_file, save_path = stats_file)
        stats = pickle.load(open(stats_file, "rb"))
        iou_bbox = {'train': stats['train']['iou_bbox'],
                        'val': stats['val']['iou_bbox']}

        iou_mask = {'train': stats['train']['iou_mask'],
                    'val': stats['val']['iou_mask']}

        dist = {'train': stats['train']['dist'],
                        'val': stats['val']['dist']}

        get_histogram_same_plot(iou_bbox,'iou_bbox_distributions',target_folder, True)
        get_histogram_same_plot(iou_mask,'iou_mask_distributions',target_folder, True)
        get_histogram_same_plot(dist,'centroid_distance_distributions',target_folder)
        get_histogram(stats,name,target_folder)
        print("--- %s seconds elapsed ---" % (time.time() - start_time))

    if(mask_vis):
        print("Making Visualisation...")
        sys.stdout.flush()
        visualise_tracks(resized_file, target_folder_consolidated, id_idx = 0)
        print("--- %s seconds elapsed ---" % (time.time() - start_time))


    if(gauss_vis):
        print("Making Gaussian Visualisation...")
        sys.stdout.flush()
        visualise_gaussians(resized_file,target_folder_gauss)
        print("--- %s seconds elapsed ---" % (time.time() - start_time))
