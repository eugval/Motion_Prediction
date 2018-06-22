
import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))

import h5py
import numpy as np
from Mask_RCNN.mrcnn import visualize
from preprocessing.utils import make_colormap,find_start_count
from sort.sort import Sort, KalmanBoxTracker

from scipy.spatial import distance
from shutil import copyfile



def find_nn(query, array):
    nn = 0
    min_dist = float('inf')
    for i in range(array.shape[0]):
        dist = distance.euclidean(query, array[i, :])
        if (dist < min_dist):
            nn = i
            min_dist = dist

    return nn



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


def discard_masks(data_file,target_file, masks_to_keep = ['car','truck'], small_threshold = -1, verbose=0):
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








def track(data_file, reverse= False, verbose = 0):
    if(verbose==1):
        print("Opening File...")

    f = h5py.File(data_file, "r+")
    mot_tracker = Sort()
    tracks_n = f["tracks_n"].value[0]

    start_count = find_start_count(list(f.keys()))

    if(not reverse):
        frame_indices = range(start_count, f['frame_number'].value[0])
    else:
        frame_indices = reversed(range(start_count, f['frame_number'].value[0]))

    if(verbose==1):
        print("Starting loop...")
    for i in frame_indices :
        frame = "frame{}".format(i)

        bbox_handle = f[frame]['rois']
        detection = bbox_handle.value

        scores = f[frame]['scores'].value
        number_of_masks = scores.shape[0]

        detection_with_scores = np.hstack((detection, np.reshape(scores, (-1, 1))))
        if(verbose== 1):
            print("detections with scores:")
            print(detection_with_scores)

        track_bbs_ids = mot_tracker.update(detection_with_scores)

        if(verbose== 1):
            print("tracked bbs:")
            print(track_bbs_ids)

        # Associate the track_BBs with the original bbs
        # for each of the track bbs
        # find the nearest neighbour in the original detections
        # associate the ID with the index of the original detection

        index_array = np.zeros(number_of_masks)

        if verbose==1 : print("number of masks {}".format(number_of_masks))

        for track in track_bbs_ids:
            nn_index = find_nn(track[:-1], detection)
            index_array[nn_index] = track[-1]

        if(verbose==1):
            print("The index array is")
            print(index_array)

        max_idx = np.amax(index_array) if number_of_masks > 0 else 0
        if(max_idx> tracks_n):
            tracks_n = max_idx

        ID_dataset_key = "{}/IDs".format(frame)

        if(ID_dataset_key in f):
            f[ID_dataset_key][:,1]= index_array
        else:
            f.create_dataset(ID_dataset_key,(index_array.shape[0],2))
            f[ID_dataset_key][:, 0] = index_array

    f["tracks_n"][0] = tracks_n

    KalmanBoxTracker.count = 0

    f.close()



def recursive_update(f,change_from,change_to, col_idx,i):
    frame = "frame{}".format(i)
    IDs = f[frame]['IDs'].value

    if(np.any(IDs[:,col_idx]==change_from)):
        idx = np.where(IDs[:, col_idx] == change_from)
        f[frame]['IDs'][idx[0][0],col_idx]=change_to


def consolidate_indices(data_file, target_file= None, verbose = 0):
    if(target_file==None):
        target_file=data_file
    else:
        if verbose==1 : print("Creating target file...")
        #assert not os.path.exists(target_file)
        copyfile(data_file, target_file)

    f = h5py.File(target_file, "r+")
    tracks_n = f["tracks_n"].value


    start_count = find_start_count(list(f.keys()))
    for i in range(start_count, f['frame_number'].value[0]-1) :
        frame1 = "frame{}".format(i)
        frame2 = "frame{}".format(i+1)




        IDs_1= f[frame1]['IDs'].value


        IDs_2 = f[frame2]['IDs'].value


        for pair_index,id_pair_1 in enumerate(IDs_1):
            if(np.any(id_pair_1 == 0)):
                id_pair_1[id_pair_1 == 0] = tracks_n+1
                f[frame1]['IDs'][pair_index,:]= id_pair_1
                tracks_n+=1


            if(np.any(np.all(id_pair_1==IDs_2,axis=1))):
                continue
            elif(np.any(id_pair_1[0]==IDs_2[:,0])): #and not id_pair_1[0]==0
                idx= np.where(IDs_2[:,0]==id_pair_1[0])
                change_to = id_pair_1[1]
                change_from = IDs_2[idx[0][0],1]

                if(change_from==0):
                    f[frame2]['IDs'][idx[0][0],1]=change_to
                else:
                    recursive_update(f,change_from, change_to,1 ,i+1)
            elif (np.any(id_pair_1[1] == IDs_2[:, 1]) ): #and not id_pair_1[1] == 0
                idx= np.where(IDs_2[:, 1] == id_pair_1[1])
                change_to = id_pair_1[0]
                change_from = IDs_2[idx[0][0], 0]

                if (change_from == 0):
                    f[frame2]['IDs'][idx[0][0], 0] = change_to
                else:
                    recursive_update(f, change_from, change_to, 0, i + 1)

    f["tracks_n"][0] = tracks_n
    f.close()







def visualise_tracks(data_file, target_folder, id_idx = 0):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    f=h5py.File(data_file, "r")
    #Get a colormap for the ID
    tracks_n = f["tracks_n"].value[0]
    colormap = make_colormap(int(tracks_n+1))


    #Save visualisations
    #TODO: Get rid of the rest of the padding on the saved visualisations

    start_count = find_start_count(list(f.keys()))

    for i in range(start_count, f['frame_number'].value[0]):
        frame = "frame{}".format(i)
        r = f[frame]
        image = r['image'].value
        class_names = f["class_names"]
        IDs = r['IDs'].value[:,id_idx]

        colors = [colormap[int(j)] for j in IDs]

        save_path = os.path.join(target_folder,"{}.jpg".format(frame))
        visualize.save_instances(image, save_path, r['rois'], r['masks'], r['class_ids'],
                                  class_names, IDs, colors=colors)


    f.close()






if __name__ == "__main__":
    import argparse



    trck = True
    vis = False
    consolidate = False

    # Path to the processed and raw folders in the data
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
    RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

    data_file = os.path.join(PROCESSED_PATH, "30SLight1_Test2/30SLight1_Test2.hdf5")
    target_file = os.path.join(PROCESSED_PATH, "30SLight1_Test2/30SLight1_Test2_Fewer_Masks.hdf5")
    consolidated_file = os.path.join(PROCESSED_PATH, "30SLight1_Test2/30SLight1_Test2_Fewer_Masks_Consolidated.hdf5")
    target_folder = os.path.join(PROCESSED_PATH, "30SLight1_Test2/tracked_images_gauss/")

    if(trck):
        print("Discarding other masks...")

        discard_masks(data_file,target_file, small_threshold=20)

        print("tracking...")

        track(target_file,verbose=0)

        print("tracking reverse...")

        track(target_file,reverse=True, verbose=0)

    if(vis):
        print("Creating the tracked visualisation...")
        target_folder = os.path.join(PROCESSED_PATH, "30SLight2/tracked_images/")


        visualise_tracks(target_file,target_folder)

    if(consolidate):
        print("Doing index consolidation...")
        consolidate_indices(target_file,  verbose =0)





