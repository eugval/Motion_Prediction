import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"preprocessing"))
sys.path.append(os.path.join(ROOT_DIR,"experiments"))

from experiments.evaluation_metrics import IoUMetric
import pickle
import h5py
import numpy as np
from preprocessing.utils import find_start_count
import time




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

        f_dset.close()

        return self.idx_sets



    def discard_based_on_iou(self,dataset_file, high_thresh =1 , low_thresh=0, idx_sets_file = None, save_path=None):

        f = h5py.File(dataset_file,'r')
        if(idx_sets_file is not None):
            idx_sets = pickle.load( open(idx_sets_file, "rb" ) )
            self.idx_sets = idx_sets

        iou_bbox_calculator = IoUMetric(type = 'bbox')
        iou_mask_calculator = IoUMetric(type = 'mask')
        for k,v  in self.idx_sets.items():
            indices_to_delete = []
            for i, idx in enumerate(v):
                datapoint = 'datapoint{}'.format(idx)
                input_mask = f[datapoint]['masks'].value[:,:,0]
                future_mask = f[datapoint]['future_mask'].value
                iou_bbox = iou_bbox_calculator.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0))
                iou_mask = iou_mask_calculator.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0))


                if(iou_bbox > high_thresh or iou_mask >high_thresh):
                    indices_to_delete.append(i)
                elif(iou_bbox <  low_thresh or iou_mask < low_thresh):
                    indices_to_delete.append(i)
            self.idx_sets[k]=np.delete(self.idx_sets[k], indices_to_delete)

        if(save_path is not None):
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

        f_dset.close()

        return self.idx_setsclass














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
    f_new.create_dataset("transition_frame", data = [f1['frame_number'].value[0]])

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







class MakeDataSplitsWithMerge(object):
    def __init__(self, dataset_file_path, frames_file_path):
        self.idx_sets = {'train':np.array([]),
                         'test': np.array([]),
                         'val': np.array([]),
                         }

        self.split_method = 'frame'

        self.dataset_file_path = dataset_file_path
        self.frames_file_path = frames_file_path

        f_dset = h5py.File(self.dataset_file_path, 'r')
        self.test_split_datapoint_1 = f_dset['datapoints'].value[0] - 1
        self.test_split_datapoint_2 = f_dset['datapoints'].value[0] - 1
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
        self.test_split_datapoint_1 = f_dset['datapoints'].value[0] - 1
        self.test_split_datapoint_2 = f_dset['datapoints'].value[0] - 1
        f_dset.close()


    def make_frame_split(self, split_type, frac,
        save_path = False, discard_input_overlap = False ): #discard input overlap not tested

        if(self.split_method != 'frame_with_merge'):
            self.split_method = 'frame_with_merge'
            self.reset()

        #Open the file containing the datapoints and frames
        f_dset = h5py.File(self.dataset_file_path, 'r')

        timestep = f_dset['timestep'].value[0]
        number_of_inputs = f_dset['number_inputs'].value[0]
        dataset_size = f_dset['datapoints'].value[0]
        transition_frame = f_dset['transition_frame'].value[0]
        transition_datapoint = f_dset['transision_datapoint'].value[0]



        if(split_type == 'test'):
            #reset the test split datapoint, reset the validation set and set the iteration to the whole dataset
            split_datapoint_1 = dataset_size - int(frac/2 * dataset_size) - 1
            split_datapoint_2 = transition_datapoint - int(frac/2 * dataset_size)
            self.test_split_datapoint_1 = split_datapoint_1
            self.test_split_datapoint_2 = split_datapoint_2

            self.idx_sets['val'] = np.array([])


            iteration = range(f_dset['datapoints'].value[0])

        elif(split_type == 'val'):
            #Use the previous training and validation sets as the iteration
            iteration = np.concatenate([self.idx_sets['train'], self.idx_sets['val']]).astype('int')
            #set the splitting datapoint
            split_datapoint_1 = self.test_split_datapoint_1 - int(frac/2 * dataset_size)
            split_datapoint_2 = self.test_split_datapoint_2 - int(frac/2 * dataset_size)



        # Do the end split
        if(split_datapoint_1 < transition_datapoint):
            raise ValueError('Split datapoint {} less than transition datapoint {}'.format(split_datapoint_1 , transition_datapoint))
        else:
            splitting_frame_1 = f_dset['datapoint{}'.format(split_datapoint_1)]['origin_frame'].value[0]
            if(splitting_frame_1< transition_frame):
                raise ValueError('Split frame {} less than transition frame {}'.format(splitting_frame_1 , transition_frame))
        # Do the end split
        if(split_datapoint_2 < 0):
            raise ValueError('Split datapoint {} less than 0'.format(split_datapoint_2 , transition_datapoint))
        else:
            splitting_frame_2 = f_dset['datapoint{}'.format(split_datapoint_2)]['origin_frame'].value[0]
            if(splitting_frame_2< 0):
                raise ValueError('Split frame {} less than 0'.format(splitting_frame_2 , transition_frame))


        if(splitting_frame_2>(splitting_frame_1 - int(timestep * number_of_inputs)) ):
            raise ValueError('Split frame2 bigger than  split frame 1 - inputs')


        split_idx = []
        train_idx = []
        for i in iteration:
            #fetch datapoint
            datapoint = 'datapoint{}'.format(i)
            origin_frame = f_dset[datapoint]['origin_frame'].value[0]
            #If it comes from a frame higher than the split
            if origin_frame > splitting_frame_1:
                #put it on the split set
                split_idx.append(i)
            elif (discard_input_overlap and origin_frame >= (splitting_frame_1 - int(timestep * number_of_inputs))):
                #If we want a buffer to avoid overlapping inputs discard some frames
                continue

            elif(origin_frame > splitting_frame_2 and origin_frame < transition_frame):
                #put it on the split set
                split_idx.append(i)
            elif (discard_input_overlap and origin_frame >= (splitting_frame_2 - int(timestep * number_of_inputs))):
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

        f_dset.close()

        return self.idx_sets



    def discard_based_on_iou(self,dataset_file, high_thresh =1 , low_thresh=0, idx_sets_file = None, save_path=None):

        f = h5py.File(dataset_file,'r')
        if(idx_sets_file is not None):
            idx_sets = pickle.load( open(idx_sets_file, "rb" ) )
            self.idx_sets = idx_sets

        iou_bbox_calculator = IoUMetric(type = 'bbox')
        iou_mask_calculator = IoUMetric(type = 'mask')
        for k,v  in self.idx_sets.items():
            indices_to_delete = []
            for i, idx in enumerate(v):
                datapoint = 'datapoint{}'.format(idx)
                input_mask = f[datapoint]['masks'].value[:,:,0]
                future_mask = f[datapoint]['future_mask'].value
                iou_bbox = iou_bbox_calculator.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0))
                iou_mask = iou_mask_calculator.get_metric(np.expand_dims(input_mask,0),np.expand_dims(future_mask,0))


                if(iou_bbox > high_thresh or iou_mask >high_thresh):
                    indices_to_delete.append(i)
                elif(iou_bbox <  low_thresh or iou_mask < low_thresh):
                    indices_to_delete.append(i)
            self.idx_sets[k]=np.delete(self.idx_sets[k], indices_to_delete)

        if(save_path is not None):
             self.save_idx_sets(save_path)

        return self.idx_sets



if __name__=='__main__':


    print("starting")
    sys.stdout.flush()
    start_time = time.time()
    make_split = False
    merge = False
    discard = True

    # Path to the processed folders in the data
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")

    if (make_split):
        names = [("Crossing1",1),("Football1and2",2)]

        for name, config in names:
            print("Doing {} .... ".format(name))
            sys.stdout.flush()
            resized_file = os.path.join(PROCESSED_PATH, "{}/{}_resized.hdf5".format(name,name))
            dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name, name))
            set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(name, name))


            data_splitter = MakeDataSplits(dataset_file, resized_file)
            data_splitter.make_frame_split('test', 0)
            data_splitter.make_frame_split( 'val', 0.1, save_path = set_idx_file)

            print("finished")
            print("--- %s seconds elapsed ---" % (time.time() - start_time))



    if(discard):
        names = [("Football1and2_lt",2)]

        for name, config in names:
            print("Doing {} .... ".format(name))
            sys.stdout.flush()

            resized_file = os.path.join(PROCESSED_PATH, "{}/{}_resized.hdf5".format(name,name))
            dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(name, name))
            set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(name, name))
            set_idx_file_high_movement = os.path.join(PROCESSED_PATH, "{}/{}_sets_high_movement.pickle".format(name, name))

            data_splitter = MakeDataSplits(dataset_file, resized_file)
            data_splitter.discard_based_on_iou(dataset_file, high_thresh = 0.6, idx_sets_file = set_idx_file, save_path=set_idx_file_high_movement)
            print("finished")
            print("--- %s seconds elapsed ---" % (time.time() - start_time))



    if(merge):
        name1 = "Crossing1"
        name2 = "Crossing2"
        new_name = "Crossing1and2"

        data_file1 = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name1, name1))
        data_file2 = os.path.join(PROCESSED_PATH, "{}/{}.hdf5".format(name2, name2))

        new_folder = os.path.join(PROCESSED_PATH, "{}/".format(new_name))

        merge_data(data_file1, data_file2, new_folder, new_name, verbose =1)
