from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2

class DataFromH5py(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path, set_idx,  input_type = ["images","masks"], label_type ='future_mask', other_sample_entries = ["future_centroid"],transform=None):
        self.f = h5py.File(file_path, "r")
        self.transform = transform
        self.set_idx = set_idx
        self.train_len = set_idx['train'].shape[0]
        self.label_type = label_type
        self.input_type = input_type
        self.other_sample_entries = other_sample_entries
        self.initial_dims = (self.f['datapoint1']['images'].shape[0], self.f['datapoint1']['images'].shape[1])

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        datapoint_idx = self.set_idx['train'][idx]

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




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        new_sample = sample
        input = sample['input']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input = input.transpose((2, 0, 1))
        new_sample['input'] = input / 255
        return new_sample


class ResizeInput(object):
    def __init__(self, height= 128, width = 256):
        self.h = int(height)
        self.w = int(width)


    def __call__(self,sample):
        input, label = sample['input'], sample['label']

        new_sample = sample

        input = cv2.resize(input, (self.w,self.h))
        label = cv2.resize(label, (self.w, self.h))

        new_sample['input'] = input
        new_sample['label'] = label

        return new_sample





