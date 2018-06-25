import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))
import matplotlib
matplotlib.use('Agg')

from preprocessing.detection import run_mask_rcnn, generate_save_dir_path , generate_image_dir_path

from preprocessing.tracking import  track, consolidate_indices, visualise_tracks
from preprocessing.discard import  score_and_pos_discard, class_and_size_discard

from preprocessing.make_gaussians import make_gaussian_masks, visualise_gaussians

from preprocessing.make_dataset import make_dataset
from preprocessing.resize_data import resize_data
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import os
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

#names = ["Light1", "Light2", "Crossing1", "Crossing2"]
names = ["30SLight1"]

start_time = time.time()
#Run the MaskRcnn
for name in names:
    print("dealing with {} ...".format(name))
    #Directory to fetch the raw images
    raw_dir_name = name

    #Directory to save the processed data
    processed_dir_name = name
    processed_dir = generate_save_dir_path(processed_dir_name)

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)


    #Files and folders in the processed directory
    detected_file = "{}.hdf5".format(processed_dir_name)
    data_file = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name,name))
    tracked_file = os.path.join(PROCESSED_PATH, "{}/{}_tracked.hdf5".format(name,name))
    resized_file = os.path.join(PROCESSED_PATH, "{}/{}_resized.hdf5".format(name,name))
    dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name,name))
    target_folder_consolidated = os.path.join(PROCESSED_PATH, "{}/tracked_images_consolidated/".format(name))
    target_folder_gauss = os.path.join(PROCESSED_PATH, "{}/tracked_images_gauss/".format(name))

    print("Detecting...")
    run_mask_rcnn(detected_file,
                  processed_dir,
                  generate_image_dir_path(raw_dir_name), save_images=True)

    print("--- %s seconds elapsed ---" % (time.time() - start_time))

    print("Discarding Masks...")
    class_and_size_discard(data_file,tracked_file, small_threshold = 20)

    print("--- %s seconds elapsed ---" % (time.time() - start_time))

    print("Tracking...")
    track(tracked_file)

    print("--- %s seconds elapsed ---" % (time.time() - start_time))

    print("Tracking reverse...")
    track(tracked_file, reverse=True)

    print("--- %s seconds elapsed ---" % (time.time() - start_time))

    print("Consolidating...")
    consolidate_indices(tracked_file)

    print("--- %s seconds elapsed ---" % (time.time() - start_time))

    print("Resizing...")
    resize_data(tracked_file, resized_file, 256, 314, maintain_ratio=True)

    print("--- %s seconds elapsed ---" % (time.time() - start_time))

    print("Making gaussian masks...")
    make_gaussian_masks(resized_file)
    print("--- %s seconds elapsed ---" % (time.time() - start_time))

    print("Making the dataset...")
    make_dataset(resized_file, dataset_file)
    print("--- %s seconds elapsed ---" % (time.time() - start_time))

    print("Making Visualisation...")
    visualise_tracks(resized_file, target_folder_consolidated, id_idx = 0)
    print("--- %s seconds elapsed ---" % (time.time() - start_time))

    print("Making Gaussian Visualisation...")
    visualise_gaussians(resized_file,target_folder_gauss)
    print("--- %s seconds elapsed ---" % (time.time() - start_time))