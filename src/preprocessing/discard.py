import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))

import h5py
import numpy as np
from preprocessing.utils import find_start_count


def class_and_score(frame_obj, scores_with_ids ,mask_id):
    class_id = frame_obj['class_ids'][mask_id]
    if(class_id not in scores_with_ids):
        return True
    elif(frame_obj['scores'][mask_id] > scores_with_ids[class_id]):
        return True

    return False

def correct_pos(frame_obj, positions ,mask_id):
    if(positions ==-1):
        return True

    if('x_min' in  positions and frame_obj['rois'].value[mask_id,1]< positions['x_min']):
        return False
    elif ('x_max' in positions and frame_obj['rois'].value[mask_id, 1] > positions['x_max']):
        return False
    elif ('y_min' in positions and frame_obj['rois'].value[mask_id, 2] < positions['y_min']):
        return False
    elif('y_max' in positions and frame_obj['rois'].value[mask_id, 2] > positions['y_max']):
        return False
    else:
        return True


def score_and_pos_discard(data_file,target_file,scores,positions =-1, verbose=0):
    f = h5py.File(data_file, "r")
    cls_names = f["class_names"]

    scores_with_ids = {}
    for score_tup in scores:
        scores_with_ids[np.where(cls_names.value == np.string_(score_tup[0]))[0][0]] = score_tup[1]

    if verbose == 1: print("The min scores and ids are {}".format(scores_with_ids))

    start_count = find_start_count(list(f.keys()))

    frame_indices = range(start_count, f['frame_number'].value[0])


    f2 = h5py.File(target_file, "w")
    f2.create_dataset("class_names", data=f["class_names"])
    f2.create_dataset("tracks_n", data=np.array([0]))
    f2.create_dataset("frame_number", data=f["frame_number"])

    for i in frame_indices:
        frame = "frame{}".format(i)
        relevant_masks = []
        for mask_id in range(f[frame]["rois"].shape[0]):
            if (class_and_score(f[frame], scores_with_ids ,mask_id) and correct_pos(f[frame], positions ,mask_id)):
                relevant_masks.append(mask_id)

        f2.create_dataset("{}/class_ids".format(frame), data=f[frame]['class_ids'].value[relevant_masks])
        f2.create_dataset("{}/image".format(frame), data=f[frame]["image"].value)
        f2.create_dataset("{}/rois".format(frame), data=f[frame]["rois"].value[relevant_masks])
        f2.create_dataset("{}/scores".format(frame), data=f[frame]["scores"].value[relevant_masks])
        f2.create_dataset("{}/masks".format(frame), data=f[frame]["masks"].value[:, :, relevant_masks])


    f.close()
    f2.close()




def get_mask_stats(file,class_filtered_indices, threshold, verbose =0):
    mask_list = []
    start_count = find_start_count(list(file.keys()))

    frame_indices = range(start_count, file['frame_number'].value[0])

    for i in frame_indices:
        frame = "frame{}".format(i)

        for mask_idx in class_filtered_indices[i]:
            mask_area = np.sum(file[frame]["masks"].value[:, :, mask_idx])
            mask_list.append(mask_area)
    if verbose == 2: print(mask_list)
    return np.mean(mask_list), np.std(mask_list), np.percentile(mask_list,threshold)



def class_and_size_discard(data_file,target_file, masks_to_keep = ['car','truck'], small_threshold = -1, verbose=0):
    f = h5py.File(data_file, "r")
    cls_names = f["class_names"]

    mask_bstrings = [np.string_(j) for j in masks_to_keep]
    mask_ids = [np.where(cls_names.value == name)[0][0] for name in mask_bstrings]

    if verbose == 1: print("The mask_ids to look for {}".format(mask_ids))

    start_count = find_start_count(list(f.keys()))

    frame_indices = range(start_count, f['frame_number'].value[0])

    class_filtered_indices = []
    if(start_count == 1):
        class_filtered_indices.append(None)

    for i in frame_indices :
        frame = "frame{}".format(i)
        class_filtered_indices.append(np.where(np.isin(f[frame]['class_ids'].value, mask_ids))[0])

    if (small_threshold > 0):
        mean, std, percentile = get_mask_stats(f,class_filtered_indices, small_threshold, verbose)
        if verbose == 1: print("mean area : {}, std aread : {}, percentile {}".format(mean, std, percentile))

    f2 = h5py.File(target_file, "w")
    f2.create_dataset("class_names", data=f["class_names"])
    f2.create_dataset("tracks_n", data=np.array([0]))
    f2.create_dataset("frame_number", data=f["frame_number"])

    for i in frame_indices:
        frame = "frame{}".format(i)
        if (small_threshold > 0):
            relevant_masks = []
            for mask_id in class_filtered_indices[i]:
                if (np.sum(f[frame]["masks"].value[:, :, mask_id]) > percentile):
                    relevant_masks.append(mask_id)
                elif(verbose==1):
                    print("Discarding mask with area {}".format(np.sum(f[frame]["masks"].value[:, :, mask_id])))
        else:
            relevant_masks = class_filtered_indices[i]

        f2.create_dataset("{}/class_ids".format(frame), data=f[frame]['class_ids'].value[relevant_masks])
        f2.create_dataset("{}/image".format(frame), data=f[frame]["image"].value)
        f2.create_dataset("{}/rois".format(frame), data=f[frame]["rois"].value[relevant_masks])
        f2.create_dataset("{}/scores".format(frame), data=f[frame]["scores"].value[relevant_masks])
        f2.create_dataset("{}/masks".format(frame), data=f[frame]["masks"].value[:, :, relevant_masks])


    f.close()
    f2.close()









if __name__=='__main__':
    # Path to the processed and raw folders in the data
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
    RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

    name = "football1_sm5"

    data_file = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name, name))
    class_filtered_file = os.path.join(PROCESSED_PATH, "{}/{}_cls_filtered.hdf5".format(name, name))

    tracked_file = os.path.join(PROCESSED_PATH, "{}/{}_tracked.hdf5".format(name, name))
    resized_file = os.path.join(PROCESSED_PATH, "{}/{}_resized.hdf5".format(name, name))
    dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name, name))
    target_folder = os.path.join(PROCESSED_PATH, "{}/mask_images/".format(name))
    target_folder_consolidated = os.path.join(PROCESSED_PATH, "{}/tracked_images_consolidated/".format(name))
    target_folder_gauss = os.path.join(PROCESSED_PATH, "{}/tracked_images_gauss/".format(name))

    class_and_size_discard(data_file, class_filtered_file, masks_to_keep=['person'])
    score_and_pos_discard(class_filtered_file, tracked_file, [('person', 0.98)], positions = {'y_min':150})