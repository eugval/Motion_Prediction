#Script to Run Mask-RCNN on a sequence of frame images
#Needs to be run from the src/preprocessing folder for the paths to be correct

#Imports and Global paths
import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"Mask_RCNN"))

from Mask_RCNN.mrcnn import model as modellib
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.samples.coco import coco
import skimage.io
import h5py
import numpy as np
import random

#Loads the pre-trained weights of coco
COCO_MODEL_PATH = os.path.join(ROOT_DIR,"Mask_RCNN/mask_rcnn_coco.h5")

#Path to the processed and raw folders in the data
PROCESSED_PATH = os.path.join(ROOT_DIR,"../data/processed/")
RAW_PATH = os.path.join(ROOT_DIR,"../data/raw/")

#Directory that saves logs of the Mask_RCNN
MODEL_DIR = os.path.join(ROOT_DIR, "Mask_RCNN/logs")




# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


#TODO: Get rid of these as they will make the script dependent on the folder which is being run
def generate_image_dir_path(image_dir):
    return os.path.join(RAW_PATH, image_dir)

def generate_save_dir_path(target_dir):
    return os.path.join(PROCESSED_PATH,target_dir)



def create_model(gpu_num = 1, images_per_gpu = 1, detection_confidence = 0.5, detection_nms_threshold=0.3):
    '''
    Creates the Mask-RCNN model trained on COCO.
    :param gpu_num: The number of GPUs to use
    :param images_per_gpu: The number of images to process per GPU.
    :return: the loaded model.
    '''
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = gpu_num
        IMAGES_PER_GPU = images_per_gpu
        DETECTION_MIN_CONFIDENCE = detection_confidence
        DETECTION_NMS_THRESHOLD = detection_nms_threshold

    config = InferenceConfig()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir= MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model


def run_mask_rcnn(save_name, target_dir, image_dir, model = None, one = False, display = False, save_images = False, detection_confidence= 0.5, detection_nms_threshold=0.3):
    '''

    :param save_name: The name of the h5py data file to create.
    :param target_dir: The directory to save all the generated data
    :param image_dir: The directory where the images are
    :param model: use a mask-rcnn model with external configs
    :param one: only run for one test image
    :param display: Show the generated images with masks
    :param save_images: save the generated images with masks
    :return: N/A
    '''

    #TODO: Allow for batch processing of images

    file_names = next(os.walk(image_dir))[2]

    if (one):
        file_name = random.choice(file_names)
        while (file_name == ".DS_store"):
            file_name= random.choice(file_names)
        file_names = [file_name]


    #Build the model
    if(model == None):
        model = create_model(detection_confidence= detection_confidence, detection_nms_threshold=0.3)

    if(not one):
        save_path = os.path.join(target_dir,save_name)

        f = h5py.File(save_path, "w")
        f.create_dataset("class_names", data=[np.string_(j) for j in class_names])


    i=0
    for file_name in file_names:
        if (file_name == ".DS_Store"):
            continue

        image = skimage.io.imread(os.path.join(image_dir, file_name))

        # Run detection
        results = model.detect([image], verbose=0)

        r = results[0]

        if(not one):
            f.create_dataset("frame{}/image".format(i), data=image)

            for k, v in r.items():
                f.create_dataset("frame{}/{}".format(i, k), data=v)

        if(save_images):
            save_dir_images = os.path.join(target_dir, "images/")
            if not os.path.exists(save_dir_images):
                os.makedirs(save_dir_images)

            save_path_images = os.path.join(target_dir, "images/frame{}.jpg".format(i))
            visualize.save_instances(image, save_path_images, r['rois'], r['masks'], r['class_ids'],
                                      class_names, r['scores'])

        if (display):
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'])
        i += 1

    number_of_frames = len(f.keys())-1
    f.create_dataset("frame_number", data=[number_of_frames])
    if(not one):
        f.close()






if __name__=="__main__":



    # The name of the directory to save the processed data
    target_dir_name = "football1_sm5"

    #The name of the hdf5 file to save the data
    save_name = "{}.hdf5".format(target_dir_name)

    #The name of the directory to fetch the data
    image_dir_name = "football1_sm"

    save_dir = generate_save_dir_path(target_dir_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    detection_confidence = 0.5
    detection_nms_threshold = 0.3

    run_mask_rcnn(save_name,
                  save_dir,
                  generate_image_dir_path(image_dir_name), save_images = True, detection_confidence=detection_confidence ,detection_nms_threshold=detection_nms_threshold)





