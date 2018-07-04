from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2
import torch

class DataFromH5py(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path, idx_sets, purpose ='train', input_type = ["masks"], label_type = "future_mask", other_sample_entries = ["future_centroid"],transform=None):
        #Data and data manipulations
        self.f = h5py.File(file_path, "r")
        self.transform = transform

        #Train / Test / Val splits
        self.purpose = purpose #train /test /val
        self.idx_sets = idx_sets
        self.len = idx_sets[purpose].shape[0]

        #Sample parameters
        self.label_type = label_type
        self.input_type = input_type
        self.other_sample_entries = other_sample_entries

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
        inputs =  []
        for in_type in self.input_type:
            inp = self.f[frame][in_type].value
            if(len(inp.shape)==4):
                for i in range(inp.shape[3]):
                    inputs.append(inp[:,:,:,i])
            elif(len(inp.shape)==3):
                #Length 3 inputs are masks, so convert them to integers and max them out
                inputs.append(inp.astype(int)*255)
            else:
                raise ValueError("Inputs can have 3 or 4 dimentions")

        inputs = np.dstack(inputs)
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

        inputs = []
        for in_type in self.input_type:
            inp = self.f[frame][in_type].value
            if (len(inp.shape) == 4):
                for i in range(inp.shape[3]):
                    inputs.append(inp[:, :, :, i])
            elif (len(inp.shape) == 3):
                for i in range(inp.shape[2]):
                    inputs.append(inp[:, :, i])
            else:
                raise ValueError("Inputs can have 3 or 4 dimentions")

        label = self.f[frame][self.label_type].value

        sample = {'input': inputs, 'label': label}

        for key in self.other_sample_entries:
            sample[key] = self.f[frame][key].value

        return sample




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
    def __init__(self, height= 128, width = 256):
        self.h = int(height)
        self.w = int(width)


    def __call__(self,sample):
        input, label = sample['input'], sample['label']

        new_sample = sample

        input = cv2.resize(input, (self.w, self.h))
        label = cv2.resize(label, (self.w, self.h))

        new_sample['input'] = input
        new_sample['label'] = label

        return new_sample





if __name__=='__main__':
    import pickle
    import os
    ROOT_DIR = os.path.abspath("../")
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
    from torchvision import transforms


    data_file_name = "Football1"
    dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_file_name,data_file_name))
    set_idx_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(data_file_name,data_file_name))

    set_idx = pickle.load( open(set_idx_file, "rb" ) )


    dataset = DataFromH5py(dataset_file,set_idx, transform = transforms.Compose([
                                               ResizeSample(),
                                               ToTensor()
                                           ]))

    for i in range(len(dataset)):
        print(i)
        sample = dataset[i]
        print(sample)
        print(sample['input'].size(),sample['label'].size())