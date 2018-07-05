import os
import sys
import pickle
import h5py
import numpy as np
from preprocessing.utils import find_start_count

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))


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










def merge_datasets(dataset1,dataset2,new_dataset):
    f1 = h5py.File(dataset1, "r")
    f2 = h5py.File(dataset2, "r")
    f_new = h5py.File(new_dataset, "w")



    total_frames = f1['frame_number'].value[0] +f2['frame_number'].value[0]
    f_new.create_dataset("class_names", data = f1['class_names'])
    f_new.create_dataset("frame_number", data = [total_frames])

    count = 0

    start_count = find_start_count(list(f1.keys()))
    frame_indices = range(start_count, f1['frame_number'].value[0])
    for i in frame_indices:
        frame = 'frame{}'.format(i)

        for k, v in f[frame].items():
            f_new.create_dataset("frame{}/{}".format(count,k), data=v)
            count+=1

    assert set(f1.keys()) == set(f_new.keys())
    assert set(f1['frame4'].keys()) == set(f_new['frame4'].keys())

    start_count = find_start_count(list(f2.keys()))
    frame_indices = range(start_count, f2['frame_number'].value[0])
    for i in frame_indices:
        frame = 'frame{}'.format(i)

        for k, v in f[frame].items():
            f_new.create_dataset("frame{}/{}".format(count,k), data=v)
            count+=1

    assert total_frames -1 == count
    assert set(f2['frame{}'.format(f2['frame_number'].value[0] - 5)])==set(f_new['frame{}'.format(total_frames -5)])


    f1.close()
    f2.close()
    f_new.close()







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




if __name__=='__main__':
    make_split = False

    # Path to the processed folders in the data
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")

    names = [  ("Football1",2), ("30SLight1",1),("Crossing1",1),("Light1",1), ("Light2",1),("Crossing2",1), ("Football2",2),("Football1_sm",2)]

    #names = [('Football1_sm5',1)]
    for name, config in names:
        dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name, name))
        set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(name, name))

        if(make_split):
            f = h5py.File(dataset_file, "r")
            dataset_size = f['datapoints'].value[0]
            f.close()
            idx_sets = make_train_test_split(dataset_size, 0.1)
            idx_sets = make_val_set(idx_sets, 0.1, save_path=set_idx_file)
