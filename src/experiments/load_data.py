from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2
import torch
import math
from preprocessing.utils import incorporate_ratio
from preprocessing.utils import visualise_image


class DataFromH5py(Dataset):
    def __init__(self, file_path, idx_sets, purpose, input_type, label_type,  only_one_mask = False, other_sample_entries = ["future_centroid"],transform=None):
        #Data and data manipulations
        self.f = h5py.File(file_path, "r")
        self.transform = transform
        self.future_time = self.f['future_time'].value[0]
        self.number_of_inputs = self.f['number_inputs'].value[0]
        self.timestep = self.f['timestep'].value[0]

        #Train / Test / Val splits
        self.purpose = purpose #train /test /val
        self.idx_sets = idx_sets
        self.len = idx_sets[purpose].shape[0]

        #Sample parameters
        self.label_type = label_type
        self.input_type = input_type
        self.other_sample_entries = other_sample_entries
        self.only_one_mask = only_one_mask

        #Data general parameters
        self.initial_dims = (self.f['datapoint1']['images'].shape[0], self.f['datapoint1']['images'].shape[1])

        if(hasattr(transform, 'h' ) and hasattr(transform , 'w')):
            self.resized_dims =(transform.h, transform.w)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        datapoint_idx = self.idx_sets[self.purpose][idx]

        frame = "datapoint{}".format(datapoint_idx)

        #TODO: maybe use a different technique than reshaping
        inputs_images =  []
        inputs_masks = []
        for in_type in self.input_type:
            inp = self.f[frame][in_type].value
            if(len(inp.shape)==4):
                for i in range(inp.shape[3]):
                    inputs_images.append(inp[:,:,:,i])
            elif(len(inp.shape)==3):
                #Length 3 inputs are masks, so convert them to integers and max them out
                if(self.only_one_mask):
                    inputs_masks.append(inp[:,:,0].astype(int) * 255)
                else:
                    inputs_masks.append(inp.astype(int)*255)
            elif (len(inp.shape) == 2):
                # Length 2 inputs are bboxes, so convert them to masks
                for i in range(inp.shape[0]):
                    bbox_mask = np.zeros(self.initial_dims)
                    ymin, xmin, ymax, xmax = inp[i,:]
                    bbox_mask[ymin:ymax,xmin:xmax]=1
                    inputs_masks.append(bbox_mask)
                    if (self.only_one_mask):
                        break

            else:
                raise ValueError("Inputs can have 2, 3 or 4 dimentions")

        inputs = inputs_images+inputs_masks
        inputs = np.dstack(inputs)

        if(self.label_type == 'future_bbox'):
            label = np.zeros(self.initial_dims)
            ymin, xmin, ymax, xmax = self.f[frame][self.label_type].value
            label[ymin:ymax,xmin:xmax]=1
        else:
            label = self.f[frame][self.label_type].value

        sample = {'input': inputs.astype(np.float), 'label': label.astype(np.float)}

        for key in self.other_sample_entries:
            sample[key] = self.f[frame][key].value

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_raw(self, idx):
        datapoint_idx = self.idx_sets[self.purpose][idx]

        frame = "datapoint{}".format(datapoint_idx)

        inputs_images =  []
        inputs_masks = []
        for in_type in self.input_type:
            inp = self.f[frame][in_type].value
            if(len(inp.shape)==4):
                for i in range(inp.shape[3]):
                    inputs_images.append(inp[:,:,:,i])
            elif(len(inp.shape)==3):
                #Length 3 inputs are masks, so convert them to integers and max them out
                if(self.only_one_mask):
                    inputs_masks.append(inp[:,:,0].astype(int) * 255)
                else:
                    for i in range(inp.shape[2]):
                        inputs_masks.append(inp[:,:,i].astype(int)*255)
            elif (len(inp.shape) == 2):
                # Length 2 inputs are bboxes, so convert them to masks
                for i in range(inp.shape[0]):
                    bbox_mask = np.zeros(self.initial_dims)
                    ymin, xmin, ymax, xmax = inp[i,:]
                    bbox_mask[ymin:ymax,xmin:xmax]=1
                    inputs_masks.append(bbox_mask)
                    if (self.only_one_mask):
                        break

            else:
                raise ValueError("Inputs can have 2, 3 or 4 dimentions")


        inputs = inputs_images+inputs_masks

        if(self.label_type == 'future_bbox'):
            label = np.zeros(self.initial_dims)
            ymin, xmin, ymax, xmax = self.f[frame][self.label_type].value
            label[ymin:ymax,xmin:xmax]=1
        else:
            label = self.f[frame][self.label_type].value

        sample = {'input': inputs, 'label': label}

        for key in self.other_sample_entries:
            sample[key] = self.f[frame][key].value

        return sample

    def get_datapoint_index(self,idx):
        datapoint_idx = self.idx_sets[self.purpose][idx]
        return datapoint_idx



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        new_sample = sample
        input = sample['input']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input = input.transpose((2, 0, 1))
        new_sample['input'] = torch.from_numpy(input / 255)
        new_sample['label'] = torch.from_numpy(sample['label'])
        return new_sample


class ResizeSample(object):
    def __init__(self, height = 128, width = 256):
        self.h = int(height)
        self.w = int(width)


    def __call__(self,sample):
        input, label = sample['input'], sample['label']

        new_sample = sample

        input = cv2.resize(input, (self.w, self.h))
        label = cv2.resize(label, (self.w, self.h),interpolation=cv2.INTER_NEAREST)

        new_sample['input'] = input
        new_sample['label'] = label

        return new_sample

class RandomCropWithAspectRatio(object):
    def __init__(self, max_crop = 10):
        self.max_crop = max_crop

    def __call__(self, sample):

        input, label = sample['input'], sample['label']
        new_sample = sample

        initial_h, initial_w = input.shape[0], input.shape[1]

        crop_amount = np.random.randint(self.max_crop)
        new_w = initial_w - crop_amount
        ratio  = new_w/initial_w
        new_h = int(round(ratio*initial_h))

        min_h = initial_h - new_h
        min_w = initial_w - new_w

        crop_type = np.random.randint(3)
        if(crop_type == 0):
            input = input[min_h:new_h,min_w:new_w,:]
            label =  label[min_h:new_h, min_w:new_w]
        elif(crop_type == 1):
            input = input[:new_h, :new_w, :]
            label = label[:new_h, :new_w]
        elif(crop_type == 2):
            input = input[min_h:, min_w:, :]
            label = label[min_h:, min_w:]


        new_sample['input'] = input
        new_sample['label'] = label

        return new_sample

class RandomHorizontalFlip(object):
    def __init__(self, chance = 0.5):
        self.chance = chance

    def __call__(self, sample):
        input, label = sample['input'], sample['label']
        new_sample = sample

        do_it = np.random.uniform()
        if (do_it >= self.chance):
            return sample

        input = cv2.flip(input, 1)
        label  = cv2.flip(label, 1)

        new_sample['input'] = input
        new_sample['label'] = label

        return new_sample

class RandomRotation(object):
    def __init__(self, rotation_range = 2):
        self.rotation_range = rotation_range
    def __call__(self,sample):
        input, label = sample['input'], sample['label']
        new_sample = sample

        rows, cols = input.shape[0], input.shape[1]

        rotation_angle = np.random.uniform(-self.rotation_range, self.rotation_range)


        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle , 1)
        input = cv2.warpAffine(input, M, (cols, rows))
        label = cv2.warpAffine(label, M, (cols, rows))


        new_sample['input'] = input
        new_sample['label'] = label

        return new_sample



class RandomNoise(object):
    def __init__(self, noise_range = 50, chance = 0.3):
        self.noise_range = noise_range
        self.chance = chance

    def __call__(self,sample):
        input, label = sample['input'], sample['label']
        new_sample = sample

        rows, cols = input.shape[0], input.shape[1]

        do_it = np.random.uniform()
        if(do_it >= self.chance):
            return sample

        for inp_idx in range(input.shape[-1]):
            if(np.unique(input[:,:, inp_idx]).shape[0]==2):
                continue

            rand_array = np.random.randint(-self.noise_range,self.noise_range,(rows,cols))
            input[:,:,inp_idx]+=rand_array
            input[:, :, inp_idx] = np.minimum(input[:,:,inp_idx], 255)
            input[:, :, inp_idx] = np.maximum(input[:,:,inp_idx], 0)

        new_sample['input'] = input

        return new_sample



if __name__=='__main__':
    import pickle
    import os
    ROOT_DIR = os.path.abspath("../")
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
    from torchvision import transforms


    data_file_name = "Crossing1_sm"
    dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_file_name,data_file_name))
    idx_sets_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(data_file_name,data_file_name))

    input_types = ['masks', 'images']
    idx_sets = pickle.load(open(idx_sets_file, "rb"))

    train_set = DataFromH5py(dataset_file, idx_sets, input_type=input_types, transform=RandomCropWithAspectRatio())

    sample = train_set[4]
    raw_sample = train_set.get_raw(4)

    sample_inputs = sample['input']
    sample_label = sample['label']

    raw_sample_inputs = raw_sample['input']
    raw_sample_label = raw_sample['label']

    # for i in range(len(train_set)):
    #     print(i)
    #     sample = train_set[i]
    #     print(sample)
    #     print(sample['input'].size(),sample['label'].size())