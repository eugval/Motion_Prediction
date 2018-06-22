import h5py
from preprocessing.utils import find_start_count
import cv2
import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import numpy as np


def resize_image(image, height, width):
    make_bool = False
    if(image.dtype == bool):
        image = image.astype(np.uint8)
        make_bool = True
    res = cv2.resize(image, (int(width),int( height)))

    if(make_bool):
        res= res.astype(bool)

    return res



def incorporate_ratio(initial_dims,max_height, max_width):
    initial_height = initial_dims[0]
    initial_width = initial_dims[1]
    ratio = min(max_height/initial_height, max_width/initial_width)

    new_h = ratio*initial_height
    new_w = ratio * initial_width

    return int(new_h), int(new_w), ratio


def crop_images(image_dir, x_min =0 , x_max=None,y_min=0, y_max=None):
    file_names = next(os.walk(image_dir))[2]

    if(x_min == 0 and x_max == None and y_min ==0 and y_max == None ):
        return

    for file_name in file_names:
        if (file_name == ".DS_Store"):
            continue

        img = cv2.imread(file_name)
        if(x_max == None):
            x_max = img.shape[0]
        if(y_max == None):
            y_max = img.shape[1]

        crop_img = img[x_min:x_max, y_min:y_max]
        cv2.imwrite(file_name,crop_img)


def resize_data(data_file,target_file, height, width, maintain_ratio = True):
    f = h5py.File(data_file, "r")
    f2 = h5py.File(target_file, "w")



    #TODO: change this hardcoding to automatic search
    f2.create_dataset("class_names", data=f["class_names"])
    f2.create_dataset("frame_number", data=f["frame_number"])
    if ('tracks_n' in f):
        f2.create_dataset("tracks_n", data=f['tracks_n'])


    initial_dims = f['frame1']['image'].value.shape


    if(maintain_ratio):
        height, width, ratio = incorporate_ratio(initial_dims, height,width)
        ratio_h = ratio
        ratio_w = ratio
    else:
        ratio_h = height/initial_dims[0]
        ratio_w = width/initial_dims[1]



    start_count = find_start_count(list(f.keys()))

    frame_indices = range(start_count, f['frame_number'].value[0])

    for i in frame_indices:
        frame = "frame{}".format(i)
        for key in f[frame].keys():
            if(len(f[frame][key].shape)>2 and f[frame][key].shape[0] == initial_dims[0] and  f[frame][key].shape[1]==initial_dims[1] and f[frame][key].shape[2]>0 ):
                resized_image = resize_image(f[frame][key].value, height, width)
                if(len(resized_image.shape)<3):
                   resized_image =  resized_image[:,:, np.newaxis]
                f2.create_dataset("{}/{}".format(frame,key), data=resized_image)
            elif(key == 'rois'):
                old_rois = f[frame][key].value
                new_rois = np.zeros(old_rois.shape)
                new_rois[:,(0,2)] = old_rois[:,(0,2)]*ratio_w
                new_rois[:,(1,3)] = old_rois[:,(1,3)]*ratio_h
                f2.create_dataset("{}/{}".format(frame, key), data=new_rois.astype(int))

            else:
                f2.create_dataset("{}/{}".format(frame, key), data=f[frame][key].value)
    f.close()
    f2.close()


def scale_data(data_file, target_file, scale):
    f = h5py.File(data_file, "r")

    initial_dims = f['frame1']['image'].value.shape

    height = int(initial_dims[0] / scale)
    width = int(initial_dims[0] / scale)
    f.close()
    resize_data(data_file, target_file, height, width, maintain_ratio=True)



if __name__=='__main__':
    # Path to the processed and raw folders in the data
    PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
    RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

    data_file = os.path.join(PROCESSED_PATH, "30SLight1_Test2/30SLight1_Test2_Fewer_Masks.hdf5")
    target_file = os.path.join(PROCESSED_PATH, "30SLight1_Test2/30SLight1_Test2_resized.hdf5")

    resize_data(data_file, target_file, 314, 258)