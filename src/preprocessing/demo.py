
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[48]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import h5py

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from new_functions import visualize2

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "../../../Data/rraw/30SLight2") #images


# In[17]:


print(ROOT_DIR)
print(IMAGE_DIR)


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[3]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[4]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[43]:


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


# ## Run Object Detection

# In[9]:


# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=0)

# Visualize results
r = results[0]
print(image.shape)
print(r.keys())
#print(results)

save_path = os.path.join(ROOT_DIR, "../../Data/")

f = h5py.File("{}mytestfile.hdf5".format(save_path), "w")
for k,v in r.items():
    print(k)
    f.create_dataset(k, data = v)
    
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                           # class_names, r['scores'])


#save_instances(image,save_path, r['rois'], r['masks'], r['class_ids'], 
        #                    class_names, r['scores'])


# In[32]:


import colorsys
def make_colormap(N, bright = True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors


# In[39]:


##### VISUALISE ONE WITH ID #########



save_path = os.path.join(ROOT_DIR, "../../../Data/")
col = make_colormap(500)


f = h5py.File("{}30SLight1_Tracked_Fewer.hdf5".format(save_path), "r")

for i in range(1,len(list(f.keys()))):
    frame = "frame{}".format(i)
    r = f[frame]
    image =r['image'].value


    #for k,v in r.items():
      #  print(k)
      #  f.create_dataset(k, data = v)

        
    #visualize2.save_instances(image, r['rois'].value, r['masks'].value, r['class_ids'].value, 
      #                         class_names, r['ID'].value,IDs= r['ID'].value, colormap = col )

    save_path2 = os.path.join(save_path, "FullSegmentation/30STracked/{}.jpg".format(frame))
    visualize2.save_instances(image,save_path2, r['rois'], r['masks'], r['class_ids'], 
                             class_names, r['ID'].value,IDs= r['ID'].value, colormap = col )


# In[37]:


f.close()


# In[ ]:


f = h5py.File("{}mytestfile.hdf5".format(save_path), "r")


# In[ ]:


f.keys()


# In[ ]:


dset = f['rois']


# In[ ]:


dset.value
f.close()


# In[49]:


print(IMAGE_DIR)


# In[50]:


# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]


# In[51]:


print(file_names)


# In[ ]:


for file_name in file_names :
    print(file_name)


# In[41]:


N_FRAMES = len(file_names)
IM_DIM = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[0])).shape
print(IM_DIM)


# In[55]:


i=1


#/
##Class IDs
##Frame
###Image
###r
#class_names_arr = np.array(class_names)
#print(type(class_names_arr))
#print(class_names_arr)
save_path = os.path.join(ROOT_DIR, "../../../Data/")
f = h5py.File("{}30SLight2.hdf5".format(save_path), "w")
f.create_dataset("class_names", data = [np.string_(j) for j in class_names])


for file_name in file_names:
    if(file_name ==".DS_Store"):
        continue
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
    f.create_dataset("frame{}/image".format(i), data = image)
    # Run detection
    results = model.detect([image], verbose=0)

    # Visualize results
    r = results[0]

    for k,v in r.items():
       
        f.create_dataset("frame{}/{}".format(i,k), data = v)
    
    save_path = os.path.join(ROOT_DIR, "../../../Data/FullSegmentation/30SLight2/{}".format(file_name))
    visualize2.save_instances(image,save_path, r['rois'], r['masks'], r['class_ids'], 
                        class_names, r['scores'])
    
    i+=1
    if(i%20 == 0 ):
        print(i)


# In[52]:


f.close()


# In[ ]:


f.keys()


# In[ ]:



print(np.shape(r['rois']))
print(np.shape(r['class_ids']))
print(np.shape(r['scores']))
print(np.shape(r['masks']))
print(r)

