import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))

import h5py
from preprocessing.utils import find_start_count
import numpy as np

ROOT_DIR = os.path.abspath("../")
import pickle

def get_idx_from_id(idx,file,frame_num,id_idx=0):
    frame = 'frame{}'.format(frame_num)
    return np.where(file[frame]['IDs'].value[:,id_idx]==idx)[0][0]




def add_datapoint(f,f2,id,i,timestep,number_inputs,future_time, datapoint_index):
    f2.create_dataset("datapoint{}/origin_frame".format(datapoint_index), data=[i])

    f2.create_dataset('datapoint{}/mask_id'.format(datapoint_index), data=[id])
    images = []
    masks = []
    delta_masks = []
    centroids = []

    for j in range(number_inputs):

        frame_number = i + int(j*timestep)
        origin_frame = 'frame{}'.format(frame_number)
        origin = f[origin_frame]

        images.append(origin['image'].value)


        mask_idx = get_idx_from_id(id,f,frame_number)
        mask = origin['masks'].value[:,:,mask_idx]

        masks.append(mask)
        delta_masks.append(mask.astype(int) - masks[0].astype(int))
        centroids.append(origin['centroids'].value[mask_idx, :])

    frame_number= i+int(future_time-1)
    mask_idx = get_idx_from_id(id,f,frame_number)
    gaussian_mask = f['frame{}'.format(frame_number)]['gaussians'].value[:,:,mask_idx]
    future_mask =  f['frame{}'.format(frame_number)]['masks'].value[:,:,mask_idx]
    future_centroid =  f['frame{}'.format(frame_number)]['centroids'].value[mask_idx,:]

    #images = np.dstack(images)
    images = np.stack(images, axis=3)
    masks = np.dstack(masks)
    delta_masks=np.dstack(delta_masks)
    centroids = np.stack(centroids,axis = 0)

    f2.create_dataset('datapoint{}/images'.format(datapoint_index),data=images)
    f2.create_dataset('datapoint{}/masks'.format(datapoint_index),data=masks)
    f2.create_dataset('datapoint{}/centroids'.format(datapoint_index), data=centroids)
    f2.create_dataset('datapoint{}/delta_masks'.format(datapoint_index),data=delta_masks)
    f2.create_dataset('datapoint{}/future_mask'.format(datapoint_index), data=future_mask)
    f2.create_dataset('datapoint{}/future_centroid'.format(datapoint_index), data=future_centroid)
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

            future_frame = i + future_time #Was -1 for first gen of data
            if(future_frame > max_frames-1):
                break


            if (check_id_consistency(f,id,i,timestep,number_inputs,future_time)):
                id_list.append(id)


        if(len(id_list)>0):
            for id in id_list:
                add_datapoint(f,f2,id,i,timestep,number_inputs,future_time, datapoint_index)
                datapoint_index+=1

    f2.create_dataset("datapoints", data = [datapoint_index])






def make_train_test_split(dataset_size, test_frac, save_path = False):
    dataset_idx = np.random.permutation(dataset_size)

    testset_size = int(test_frac * dataset_size)

    test_idx, train_idx = dataset_idx[:testset_size], dataset_idx[testset_size:]

    idx_sets = {'test': test_idx, 'train': train_idx, 'val': np.array([])}

    if(save_path):
        pickle.dump(idx_sets, open(save_path, "wb"))


    return idx_sets


def make_val_set(idx_sets, val_frac, save_path = False):
    full_train = np.concatenate([idx_sets['train'],idx_sets['val']]).astype('int')

    np.random.shuffle(full_train)

    val_size = int(val_frac * full_train.shape[0])

    val_idx, train_idx = full_train[:val_size], full_train[val_size:]

    new_set = {'test': idx_sets['test'], 'train': train_idx, 'val': val_idx}
    if (save_path):
        pickle.dump(new_set, open(save_path, "wb"))

    return new_set





#TODO: Finish that funciton
def convert_to_folder_structure(data_file, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    f = h5py.File(data_file, "r")

    #TODO: Make a non-hardcoded implementation

    metadata = {'datapoints': f['datapoints'].value[0],
                'future_time': f['future_time'].value[0],
                'number_inputs':f['number_inputs'].value[0],
                'origin_file':f['origin_file'].value[0],
                'timestep':f['timestep'].value[0] }


    for i in range(f['datapoints'].value[0]):
        pass





#TODO: Abstact away all these files in a module
if __name__ == "__main__":
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")

    make_split = True
    make_dataset= False

    # Path to the processed and raw folders in the data
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
    RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

    names = [  ("Football1",2), ("30SLight1",1),("Crossing1",1),("Light1",1), ("Light2",1),("Crossing2",1), ("Football2",2),("Football1_sm",2)]

    for name, config in names:
        data_file = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name, name))
        class_filtered_file = os.path.join(PROCESSED_PATH, "{}/{}_cls_filtered.hdf5".format(name, name))

        tracked_file = os.path.join(PROCESSED_PATH, "{}/{}_tracked.hdf5".format(name, name))
        tracked_file_c = os.path.join(PROCESSED_PATH, "{}/{}_tracked_c.hdf5".format(name, name))
        resized_file = os.path.join(PROCESSED_PATH, "{}/{}_resized.hdf5".format(name, name))
        dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name, name))
        set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(name, name))

        target_folder = os.path.join(PROCESSED_PATH, "{}/mask_images/".format(name))
        target_folder_consolidated = os.path.join(PROCESSED_PATH, "{}/tracked_images_consolidated/".format(name))
        target_folder_gauss = os.path.join(PROCESSED_PATH, "{}/tracked_images_gauss/".format(name))


        if(make_dataset):
            make_dataset(resized_file, dataset_file)

        if(make_split):
            f = h5py.File(dataset_file, "r")
            dataset_size = f['datapoints'].value[0]
            f.close()
            idx_sets = make_train_test_split(dataset_size, 0.1)
            idx_sets = make_val_set(idx_sets, 0.1, save_path=set_idx_file)

        print("finished {}".format(name))