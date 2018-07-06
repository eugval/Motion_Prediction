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

from preprocessing.make_gaussians import make_gaussian_masks, visualise_gaussians

from preprocessing.make_dataset import make_dataset
from preprocessing.manipulate_dataset import make_train_test_split,make_val_set
from preprocessing.resize_data import resize_data
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import os
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

names = [ ("Crossing1",1)]
detecting = False
discarding = False
tracking = False
resizing = False
make_gaussians = False
dataset = True
make_idx = True
mask_vis = False
gauss_vis = False


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
            score_and_pos_discard(class_filtered_file, tracked_file, [('car', 0.8)])
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

    if(make_gaussians):
        print("Making gaussian masks...")
        sys.stdout.flush()

        make_gaussian_masks(resized_file)
        print("--- %s seconds elapsed ---" % (time.time() - start_time))#

    if(dataset):
        print("Making the dataset...")
        sys.stdout.flush()
        make_dataset(resized_file, dataset_file)
        print("--- %s seconds elapsed ---" % (time.time() - start_time))

        f = h5py.File(dataset_file, "r")
        dataset_size = f['datapoints'].value[0]
        print(dataset_size)
        f.close()
        idx_sets = make_train_test_split(dataset_size, 0.1)
        make_val_set(idx_sets,0.1,save_path = set_idx_file)


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
