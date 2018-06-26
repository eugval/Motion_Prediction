
import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))

import h5py
import numpy as np
from Mask_RCNN.mrcnn import visualize
from preprocessing.utils import make_colormap,find_start_count
from preprocessing.discard import  class_and_size_discard
from sort.sort import Sort, KalmanBoxTracker

from scipy.spatial import distance
from shutil import copyfile
from sklearn.utils.linear_assignment_ import linear_assignment



def find_nn(query, array):
    nn = 0
    min_dist = float('inf')
    for i in range(array.shape[0]):
        dist = distance.euclidean(query, array[i, :])
        if (dist < min_dist):
            nn = i
            min_dist = dist

    return nn



def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)



def iou_track(data_file):
    f = h5py.File(data_file, "r+")
    start_count = find_start_count(list(f.keys()))
    first_frame = "frame{}".format(start_count)

    track_n = f[first_frame]['masks'].shape[2]

    index_array = np.arange(1,track_n+1)
    index_array = np.stack((index_array,index_array),axis =1)
    f.create_dataset("{}/IDs".format(first_frame), data=index_array)

    for i in range(start_count+1, f['frame_number'].value[0]):

        frame = "frame{}".format(i)
        previous_frame = "frame{}".format(i-1)

        previous_mask_n = f[previous_frame]['masks'].shape[2]
        current_mask_n = f[frame]['masks'].shape[2]

        index_array = np.zeros((current_mask_n,2))


        ious = np.zeros((current_mask_n, previous_mask_n))


        for mask_id in range(current_mask_n):
            current_box = f[frame]['rois'].value[mask_id,:]
            ious[mask_id,:] = np.array([iou(f[previous_frame]['rois'].value[previous_id,:], current_box)  for previous_id in range(f[previous_frame]['rois'].shape[0])])

        assignments = linear_assignment(-ious)

        assigned_ids = []
        for assignment in assignments:
            assigned_ids.append(assignment[0])
            if (ious[assignment[0], assignment[1]] > 0):
                index_array[assignment[0], :] = f[previous_frame]['IDs'].value[assignment[1], 0]
            else:
                track_n += 1
                index_array[assignment[0], :] = track_n


        if (len(assignments) < ious.shape[0]):
            missing_ids = [i for i in range(current_mask_n) if i not in assigned_ids]
            for missing_id in missing_ids:
                track_n += 1
                index_array[missing_id, :] = track_n






        f.create_dataset("{}/IDs".format(frame), data=index_array)

    f["tracks_n"][0] = track_n

    f.close()











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





def recursive_update(f,change_from,change_to, col_idx, i, start_count):
    max_idx = f['frame_number'].value[0] + start_count -1


    for j in range(i, f['frame_number'].value[0]):
        if(i > max_idx):
            break

        frame = "frame{}".format(j)
        IDs = f[frame]['IDs'].value


        if(np.any(IDs[:,col_idx]==change_from)):
            idx = np.where(IDs[:, col_idx] == change_from)
            f[frame]['IDs'][idx[0][0],col_idx]=change_to
        else:
            break



def full_recursive_update(f,change_from,change_to, col_idx, i, start_count):
    max_idx = f['frame_number'].value[0] + start_count -1
    if(i > max_idx):
        return

    frame = "frame{}".format(i)
    IDs = f[frame]['IDs'].value


    if(np.any(IDs[:,col_idx]==change_from)):
        idx = np.where(IDs[:, col_idx] == change_from)
        f[frame]['IDs'][idx[0][0],col_idx]=change_to
        recursive_update(f,change_from,change_to,col_idx,i+1,start_count)


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
                    recursive_update(f,change_from, change_to,1 ,i+1, start_count)
            elif (np.any(id_pair_1[1] == IDs_2[:, 1]) ): #and not id_pair_1[1] == 0
                idx= np.where(IDs_2[:, 1] == id_pair_1[1])
                change_to = id_pair_1[0]
                change_from = IDs_2[idx[0][0], 0]

                if (change_from == 0):
                    f[frame2]['IDs'][idx[0][0], 0] = change_to
                else:
                    recursive_update(f, change_from, change_to, 0, i + 1,start_count)

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
    #TODO: Fix the caption problem

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



    trck = False
    vis = False
    consolidate = False

    iou_trck = True

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

    if(trck):
        print("tracking...")

        track(tracked_file,verbose=0)

        print("tracking reverse...")

        track(tracked_file,reverse=True, verbose=0)

    if(vis):
        print("Creating the tracked visualisation...")
        target_folder = os.path.join(PROCESSED_PATH, "30SLight2/tracked_images/")


        visualise_tracks(tracked_file,target_folder)

    if(consolidate):
        print("Doing index consolidation...")
        consolidated_file = os.path.join(PROCESSED_PATH, "{}/{}_tracked_c.hdf5".format(name, name))
        consolidate_indices(tracked_file,consolidated_file,  verbose =0)



    if(iou_trck):
        iou_track(tracked_file)

