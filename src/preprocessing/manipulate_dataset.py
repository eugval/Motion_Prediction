import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))
import pickle
import h5py
import numpy as np
from preprocessing.utils import find_start_count





class MakeDataSplits(object):
    def __init__(self, dataset_file_path, frames_file_path):
        self.idx_sets = {'train':np.array([]),
                         'test': np.array([]),
                         'val': np.array([]),
                         }

        self.split_method = 'frame'

        self.dataset_file_path = dataset_file_path
        self.frames_file_path = frames_file_path

        f_dset = h5py.File(self.dataset_file_path, 'r')
        self.test_split_datapoint = f_dset['datapoints'].value[0] - 1
        f_dset.close()


    def save_idx_sets(self, save_path):
        pickle.dump(self.idx_sets, open(save_path, "wb"))

    def get_idx_sets(self):
        return self.idx_sets

    def reset(self):
        self.idx_sets = {'train': np.array([]),
                         'test': np.array([]),
                         'val': np.array([]),
                         }

        f_dset = h5py.File(self.dataset_file_path, 'r')
        self.test_split_datapoint = f_dset['datapoints'].value[0] - 1
        f_dset.close()


    def make_frame_split(self, split_type, frac,
                         save_path = False, discard_input_overlap = False ): #discard input overlap not tested

        if(self.split_method != 'frame'):
            self.split_method = 'frame'
            self.reset()

        #Open the file containing the datapoints and frames
        f_dset = h5py.File(self.dataset_file_path, 'r')
        f_fset = h5py.File(self.frames_file_path, 'r')

        timestep = f_dset['timestep'].value[0]
        number_of_inputs = f_dset['number_inputs'].value[0]
        dataset_size = f_dset['datapoints'].value[0]



        if(split_type == 'test'):
            #reset the test split datapoint, reset the validation set and set the iteration to the whole dataset
            split_datapoint = dataset_size - int(frac * dataset_size) - 1
            self.test_split_datapoint = split_datapoint

            self.idx_sets['val'] = np.array([])


            iteration = range(f_dset['datapoints'].value[0])

        elif(split_type == 'val'):
            #Use the previous training and validation sets as the iteration
            iteration = np.concatenate([self.idx_sets['train'], self.idx_sets['val']]).astype('int')
            #set the splitting datapoint
            split_datapoint = self.test_split_datapoint - int(frac * dataset_size)



        if(split_datapoint < 0):
            splitting_frame = -1
        else:
            splitting_frame = f_dset['datapoint{}'.format(split_datapoint)]['origin_frame'].value[0]

        split_idx = []
        train_idx = []
        for i in iteration:
            #fetch datapoint
            datapoint = 'datapoint{}'.format(i)
            origin_frame = f_dset[datapoint]['origin_frame'].value[0]
            #If it comes from a frame higher than the split
            if origin_frame > splitting_frame:
                #put it on the split set
                split_idx.append(i)
            elif (discard_input_overlap and origin_frame >= (splitting_frame - int(timestep * number_of_inputs))):
                #If we want a buffer to avoid overlapping inputs discard some frames
                continue
            else:
                #dd the rest to the training set
                train_idx.append(i)

        #Updtate the split sets
        self.idx_sets['train'] = np.array(train_idx)
        self.idx_sets[split_type] = np.array(split_idx)

        #save and return
        if(save_path):
            self.save_idx_sets(save_path)

        return self.idx_sets

    def make_random_split(self,  split_type,  frac, save_path = False):
        if (self.split_method != 'random'):
            self.split_method = 'random'
            self.reset()

        f_dset = h5py.File(self.dataset_file_path, 'r')
        dataset_size = f_dset['datapoints'].value[0]
        split_size = int(frac * dataset_size)

        if(split_type == 'test'):
            self.idx_sets['val'] = np.array([])
            dataset_idx = np.random.permutation(dataset_size)
        elif(split_type == 'val'):
            dataset_idx = np.concatenate([self.idx_sets['train'], self.idx_sets['val']]).astype('int')
            np.random.shuffle(dataset_idx)

        split_idx, train_idx = dataset_idx[:split_size], dataset_idx[split_size:]

        self.idx_sets['train'] = np.array(train_idx)
        self.idx_sets[split_type] = np.array(split_idx)

        if (save_path):
            self.save_idx_sets(save_path)

        return self.idx_sets








def merge_data(dataset1,dataset2,new_dataset_folder, new_dataset_name, verbose = 0):
    if verbose > 0 : print("Starting data merge...")
    #If the new folder does not exist, create if
    if not os.path.exists(new_dataset_folder):
        os.makedirs(new_dataset_folder)

    #get the path of the new dataset
    new_dataset = os.path.join(new_dataset_folder,'{}.hdf5'.format(new_dataset_name))

    #open the files
    f1 = h5py.File(dataset1, "r")
    f2 = h5py.File(dataset2, "r")
    f_new = h5py.File(new_dataset, "w")

    if verbose > 0: print("Making metadata...")
    #Put the metadata into the new file
    total_frames = f1['frame_number'].value[0] + f2['frame_number'].value[0]
    f_new.create_dataset("class_names", data = f1['class_names'])
    f_new.create_dataset("frame_number", data = [total_frames])

    count = 0

    if verbose > 0: print("Copying over frames from {}".format(dataset1))

    #Iterate over the frames of the the first video
    start_count = find_start_count(list(f1.keys()))
    frame_indices = range(start_count, f1['frame_number'].value[0])
    for i in frame_indices:
        frame = 'frame{}'.format(i)

        for k, v in f1[frame].items():
            f_new.create_dataset("frame{}/{}".format(count,k), data=v)
        count+=1

    if verbose > 0: print("Asserting...")
    #Check all the keys are correct
    assert set(f1.keys()) == set(f_new.keys())
    assert set(f1['frame4'].keys()) == set(f_new['frame4'].keys())

    #Check that some of the values are correct
    rand_indices = np.random.randint(0,f1['frame_number'].value[0]-1,5)
    for idx in rand_indices:
        frame = 'frame{}'.format(idx)
        for k,v in  f_new[frame].items():
            assert np.all(v.value == f1[frame][k].value)

    if verbose > 0: print("Copying over frames from {}".format(dataset2))
    #Iterate over the frames of the second video
    start_count = find_start_count(list(f2.keys()))
    frame_indices = range(start_count, f2['frame_number'].value[0])
    for i in frame_indices:
        frame = 'frame{}'.format(i)
        for k, v in f2[frame].items():
            f_new.create_dataset("frame{}/{}".format(count,k), data=v)
        count+=1

    if verbose > 0: print("Asserting...")
    #Check that the keys are correct
    assert total_frames  == count
    assert set(f2['frame{}'.format(f2['frame_number'].value[0] - 5)])==set(f_new['frame{}'.format(total_frames -5)])

    #Check that some of the values are correct
    rand_indices = np.random.randint(0, f2['frame_number'].value[0] - 1, 5)
    for idx in rand_indices:
        frame_old = 'frame{}'.format(idx)
        frame_new = 'frame{}'.format(idx + f1['frame_number'].value[0])
        for k, v in f_new[frame_new].items():
            assert np.all(v.value == f2[frame_old][k].value)


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
    merge = True

    # Path to the processed folders in the data
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")

    names = [  ("Football1",2), ("30SLight1",1),("Crossing1",1),("Light1",1), ("Light2",1),("Crossing2",1), ("Football2",2),("Football1_sm",2)]

    #names = [('Football1_sm5',1)]
    for name, config in names:
        resized_file = os.path.join(PROCESSED_PATH, "{}/{}_resized.hdf5".format(name,name))
        dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name, name))
        set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(name, name))

        if(make_split):
            data_splitter = MakeDataSplits()
            data_splitter.make_frame_split(dataset_file,resized_file,'test', 0)
            data_splitter.make_frame_split(dataset_file, resized_file, 'val', 0.1, set_idx_file)


    if(merge):
        name1 = "Football1"
        name2 = "Football2"
        new_name = "Football1and2"

        data_file1 = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name1, name1))
        data_file2 = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name2, name2))

        new_folder = os.path.join(PROCESSED_PATH, "{}/".format(new_name))

        merge_data(data_file1, data_file2, new_folder, new_name, verbose =1)
