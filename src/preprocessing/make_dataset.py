import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))

import h5py
from preprocessing.utils import find_start_count
import numpy as np


import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")
import pickle

def get_idx_from_id(idx,file,frame_num,id_idx=0):
    frame = 'frame{}'.format(frame_num)
    return np.where(file[frame]['IDs'].value[:,id_idx]==idx)[0][0]




def add_datapoint(f,f2,id,i,timestep,number_inputs,future_time, datapoint_index):

    #Add the frame that corresponds to that datapoint at time t
    f2.create_dataset("datapoint{}/origin_frame".format(datapoint_index), data=[i])

    #Add the mask id that corresponds to that datapoint
    f2.create_dataset('datapoint{}/mask_id'.format(datapoint_index), data=[id])

    images = []
    masks = []
    delta_masks = []
    centroids = []
    bboxes = []

    #Grap the  inputs
    for j in range(number_inputs):

        #Grab the frame for a previous time input
        frame_number = i - int(j*timestep)
        origin_frame = 'frame{}'.format(frame_number)
        origin = f[origin_frame]

        #Grab the image from that frame
        images.append(origin['image'].value)

        #Gran the mask of the id for the datapoint
        mask_idx = get_idx_from_id(id,f,frame_number)
        mask = origin['masks'].value[:,:,mask_idx]
        masks.append(mask)

        #Get the centroid of the mask
        centroids.append(origin['centroids'].value[mask_idx, :])

        #Get the bbox of the mask
        bboxes.append(origin['rois'].value[mask_idx,:])

        #Get tge delta_masks
        delta_masks.append(masks[0].astype(int)- mask.astype(int))

    #Grab the future frame
    frame_number = i+int(future_time)
    mask_idx = get_idx_from_id(id,f,frame_number)
    use_gaussian = False
    if('gaussians' in f['frame{}'.format(frame_number)]):
        use_gaussian = True
        gaussian_mask = f['frame{}'.format(frame_number)]['gaussians'].value[:,:,mask_idx]
    future_mask =  f['frame{}'.format(frame_number)]['masks'].value[:,:,mask_idx]
    future_centroid =  f['frame{}'.format(frame_number)]['centroids'].value[mask_idx,:]
    future_bbox =  f['frame{}'.format(frame_number)]['rois'].value[mask_idx,:]

    #prepare the data as np arrays for hdf5
    images = np.stack(images, axis=3)
    masks = np.dstack(masks)
    delta_masks = np.dstack(delta_masks)
    centroids = np.stack(centroids,axis = 0)
    bboxes = np.stack(bboxes,axis = 0)

    f2.create_dataset('datapoint{}/images'.format(datapoint_index),data=images)
    f2.create_dataset('datapoint{}/masks'.format(datapoint_index),data=masks)
    f2.create_dataset('datapoint{}/centroids'.format(datapoint_index), data=centroids)
    f2.create_dataset('datapoint{}/bboxes'.format(datapoint_index), data=bboxes)
    f2.create_dataset('datapoint{}/delta_masks'.format(datapoint_index),data=delta_masks)
    f2.create_dataset('datapoint{}/future_mask'.format(datapoint_index), data=future_mask)
    f2.create_dataset('datapoint{}/future_centroid'.format(datapoint_index), data=future_centroid)
    f2.create_dataset('datapoint{}/future_bbox'.format(datapoint_index), data=future_bbox)
    if(use_gaussian):
        f2.create_dataset('datapoint{}/gaussian_mask'.format(datapoint_index),data=gaussian_mask)



def check_id_consistency(f,id,frame_idx, timestep, number_inputs, future_time):
    future_frame = frame_idx + future_time
    #If the id is not in the future frame return False
    if (id  not in f['frame{}'.format(future_frame)]['IDs'].value[:, 0]):
        return False

    #If the complementary previous times do not have that ID return False
    for j in range(number_inputs):
        frame_number = frame_idx - int(j * timestep)
        if(id  not in f['frame{}'.format(frame_number)]['IDs'].value[:, 0]):
            return False

    return True


def make_dataset(data_file, target_file, timestep=2, number_inputs=3 , future_time=10):
    f = h5py.File(data_file, "r")
    f2= h5py.File(target_file, "w")
    f2.create_dataset("origin_file", data = [np.string_(data_file)] )
    f2.create_dataset("timestep", data= [timestep])
    f2.create_dataset("number_inputs", data=[number_inputs])
    f2.create_dataset("future_time", data=[future_time])


    max_frames =  f['frame_number'].value[0]
    start_count = find_start_count(list(f.keys()))

    frame_indices = range(start_count,max_frames)

    datapoint_index = 0
    for i in frame_indices:
        frame = "frame{}".format(i)

        id_list = []
        for id, id_2 in f[frame]['IDs']:

            #Check that the frame at t - timestep*(num_inputs-1) exists
            if(i-int(timestep*(number_inputs-1))<0):
                continue

            #Check that the future frame exists
            future_frame = i + future_time
            if(future_frame > max_frames-1):
                break

            #Add the ids of that frame that have previous and future frames with the same id
            if (check_id_consistency(f,id,i,timestep,number_inputs,future_time)):
                id_list.append(id)

        #Add the datapoint
        if(len(id_list)>0):
            for id in id_list:
                add_datapoint(f,f2,id,i,timestep,number_inputs,future_time, datapoint_index)
                datapoint_index+=1

    f2.create_dataset("datapoints", data = [datapoint_index])







def visualise_dataset_inputs(data_file,target_file = False, idx = 1 ):
    f = h5py.File(data_file, "r")

    if( not type(idx == int)):
        idx_sets = pickle.load(open(idx, "rb"))

        idx = np.random.randint(len(idx_sets['train']))

    key = 'datapoint{}'.format(idx)

    datapoint = f[key]

    number_of_inputs = f['number_inputs'].value[0]
    timestep = f['timestep'].value[0]

    f, axarr = plt.subplots(number_of_inputs, 3, figsize=(27,15))


    for i in range(number_of_inputs):
        axarr[i, 0].imshow(datapoint['images'].value[:,:,:,i])
        axarr[i, 0].set_title("RGB from frame t+{}".format(i*timestep))

        axarr[i,1].imshow(datapoint['masks'].value[:,:,i])
        centroid = (datapoint['centroids'].value[i,1], datapoint['centroids'].value[i,0])
        axarr[i, 1].scatter(*zip(centroid), marker='+')
        axarr[i, 1].set_title("Mask and centroid from frame t+{}".format(i*timestep))

        axarr[i,2].imshow(datapoint['delta_masks'].value[:,:,i])
        axarr[i, 2].set_title("Mask at t+{} - Mask at t".format(i*timestep))

    f.tight_layout()
    if(target_file):

        f.savefig(target_file)
    else:
        f.show()



def visualise_dataset_labels(data_file,target_file=False,idx=1, new_gen = True):
    f = h5py.File(data_file, "r")

    if (not type(idx == int)):
        idx_sets = pickle.load(open(idx, "rb"))

        idx = np.random.randint(len(idx_sets['train']))

    key = 'datapoint{}'.format(idx)

    datapoint = f[key]

    future_time = f['future_time'].value[0]
    if(not new_gen):
        future_time = future_time - 1


    f, axarr = plt.subplots(2 ,figsize=(10, 10))

    centroid = (datapoint['future_centroid'].value[1], datapoint['future_centroid'].value[0])

    axarr[0].imshow(datapoint['future_mask'].value[:, :])
    axarr[0].set_title("Mask and centroid from frame t+{}".format(future_time))
    axarr[0].scatter(*zip(centroid), marker='+')

    axarr[1].imshow(datapoint['gaussian_mask'].value[: ,:])
    axarr[1].scatter(*zip(centroid), marker='+')
    axarr[1].set_title("Gaussian and centroid from frame t+{}".format(future_time))

    f.tight_layout()
    if (target_file):

        f.savefig(target_file)
    else:
        f.show()









#TODO: Abstact away all these files in a module
if __name__ == "__main__":
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")


    make_dataset= False
    visualise_dataset = True

    # Path to the processed and raw folders in the data
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
    RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

    names = [  ("Football1",2), ("30SLight1",1),("Crossing1",1),("Light1",1), ("Light2",1),("Crossing2",1), ("Football2",2),("Football1_sm",2)]

    #names = [('Football1_sm5',1)]
    for name, config in names:
        data_file = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name, name))
        class_filtered_file = os.path.join(PROCESSED_PATH, "{}/{}_cls_filtered.hdf5".format(name, name))

        tracked_file = os.path.join(PROCESSED_PATH, "{}/{}_tracked.hdf5".format(name, name))
        tracked_file_c = os.path.join(PROCESSED_PATH, "{}/{}_tracked_c.hdf5".format(name, name))
        resized_file = os.path.join(PROCESSED_PATH, "{}/{}_resized.hdf5".format(name, name))
        dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name, name))
        set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(name, name))
        input_vis_file =os.path.join(PROCESSED_PATH, "{}/{}_input_vis.png".format(name, name))
        label_vis_file = os.path.join(PROCESSED_PATH, "{}/{}_label_vis.png".format(name, name))

        target_folder = os.path.join(PROCESSED_PATH, "{}/mask_images/".format(name))
        target_folder_consolidated = os.path.join(PROCESSED_PATH, "{}/tracked_images_consolidated/".format(name))
        target_folder_gauss = os.path.join(PROCESSED_PATH, "{}/tracked_images_gauss/".format(name))


        if(make_dataset):
            make_dataset(resized_file, dataset_file)

        if(visualise_dataset):
            visualise_dataset_inputs(dataset_file,input_vis_file )
            visualise_dataset_labels(dataset_file, label_vis_file, new_gen=False)

        print("finished {}".format(name))