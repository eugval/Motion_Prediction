import h5py
from preprocessing.utils import find_start_count
import numpy as np
import os
ROOT_DIR = os.path.abspath("../")

def get_idx_from_id(idx,file,frame_num,id_idx=0):
    frame = 'frame{}'.format(frame_num)
    return np.where(file[frame]['IDs'].value[:,id_idx]==idx)[0][0]




def add_datapoint(f,f2,id,i,timestep,number_inputs,future_time, datapoint_index):
    f2.create_dataset("datapoint{}/origin_frame".format(datapoint_index), data=[i])

    f2.create_dataset('datapoint{}/mask_id'.format(datapoint_index), data=[id])
    images = []
    masks = []
    delta_masks = []

    for j in range(number_inputs):

        frame_number = i + int(j*timestep)
        origin_frame = 'frame{}'.format(frame_number)
        origin = f[origin_frame]

        images.append(origin['image'].value)

        mask_idx = get_idx_from_id(id,f,frame_number)
        mask = origin['masks'].value[:,:,mask_idx]
        masks.append(mask)
        delta_masks.append(mask.astype(int) - masks[0].astype(int))

    frame_number= i+int(future_time-1)
    mask_idx = get_idx_from_id(id,f,frame_number)
    gaussian_mask = f['frame{}'.format(frame_number)]['gaussians'].value[:,:,mask_idx]
    future_mask =  f['frame{}'.format(frame_number)]['masks'].value[:,:,mask_idx]

    #images = np.dstack(images)
    images = np.stack(images, axis=3)
    masks = np.dstack(masks)
    delta_masks=np.dstack(delta_masks)

    f2.create_dataset('datapoint{}/images'.format(datapoint_index),data=images)
    f2.create_dataset('datapoint{}/masks'.format(datapoint_index),data=masks)
    f2.create_dataset('datapoint{}/delta_masks'.format(datapoint_index),data=delta_masks)
    f2.create_dataset('datapoint{}/future_mask'.format(datapoint_index), data=future_mask)
    f2.create_dataset('datapoint{}/gaussian_mask'.format(datapoint_index),data=gaussian_mask)



def check_id_consistency(f,id,frame_idx, timestep, number_inputs, future_time):
    future_frame = frame_idx + future_time - 1
    if (id  not in f['frame{}'.format(future_frame)]['IDs'].value[:, 0]):
        return False

    for j in range(number_inputs):
        frame_number = frame_idx + int(j * timestep)
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

    assert number_inputs*timestep< future_time-1

    max_frames =  f['frame_number'].value[0]
    start_count = find_start_count(list(f.keys()))

    frame_indices = range(start_count,max_frames)

    datapoint_index = 0
    for i in frame_indices:
        frame = "frame{}".format(i)

        id_list = []
        for id, id_2 in f[frame]['IDs']:

            future_frame = i + future_time-1
            if(future_frame > max_frames-1):
                break


            if (check_id_consistency(f,id,i,timestep,number_inputs,future_time)):
                id_list.append(id)


        if(len(id_list)>0):
            for id in id_list:
                add_datapoint(f,f2,id,i,timestep,number_inputs,future_time, datapoint_index)
                datapoint_index+=1

    f2.create_dataset("datapoints", data = [datapoint_index])



if __name__ == "__main__":
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")


    data_file = os.path.join(PROCESSED_PATH, "30SLight1_Test2/30SLight1_Test2_Fewer_Masks.hdf5")
    target_file = os.path.join(PROCESSED_PATH, "30SLight1_Test2/30SLight1_Test2_dataset.hdf5")

    make_dataset(data_file,target_file)