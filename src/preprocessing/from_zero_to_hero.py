from preprocessing.detection import run_mask_rcnn, generate_save_dir_path , generate_image_dir_path

from preprocessing.tracking import discard_masks, track, consolidate_indices, visualise_tracks

from preprocessing.make_gaussians import make_gaussian_masks, visualise_gaussians

from preprocessing.make_dataset import make_dataset
from preprocessing.resize_data import resize_data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import os
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

names = ["Light1", "Light2", "Crossing1", "Crossing2"]

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

    print("Discarding Masks...")
    discard_masks(data_file,tracked_file, small_threshold = 20)

    print("Tracking...")
    track(tracked_file)

    print("Tracking reverse...")
    track(tracked_file, reverse=True)

    print("Consolidating...")
    consolidate_indices(tracked_file)

    print("Resizing...")
    resize_data(tracked_file, resized_file, 256, 314, maintain_ratio=True)

    print("Making gaussian masks...")
    make_gaussian_masks(resized_file)

    print("Making the dataset...")
    make_dataset(resized_file, dataset_file)

    print("Making Visualisation...")
    visualise_tracks(resized_file, target_folder_consolidated, id_idx = 0)

    print("Making Gaussian Visualisation...")
    visualise_gaussians(resized_file,target_folder_gauss)