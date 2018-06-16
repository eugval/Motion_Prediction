
# coding: utf-8

# In[1]:


import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import h5py
import numpy as np

from sort.sort import Sort


# In[2]:


data_import_path = '../../Data/30SLight1.hdf5'

f = h5py.File(data_import_path, "r")


# In[3]:


print(len(list(f.keys())))


# In[4]:


cls_names =f["class_names"]
print(cls_names.value)
print(np.where(cls_names.value == b'car' )[0][0])
print(np.where(cls_names.value == b'traffic light' )[0][0])
print(np.where(cls_names.value == b'bus' )[0][0])
np.where(cls_names.value == b'truck' )[0][0]


# In[5]:


list(f['frame1'].keys())


# In[6]:


f['frame1']['class_ids'].value.shape


# In[7]:


f['frame1']['image'].value.shape


# In[8]:


f['frame1']['masks'].value.shape


# In[9]:


f['frame1']['scores'].value.shape


# In[10]:


f['frame1']['rois'].value


# In[11]:


print(f['frame1']['class_ids'].value)
print(np.isin(f['frame1']['class_ids'].value,[3,8]))
np.where(np.isin(f['frame1']['class_ids'].value,[3,8]))


# In[12]:


f.close()


# In[57]:


f2.close()


# In[59]:


mot_tracker = Sort()
new_r = {}



save_path = os.path.join(ROOT_DIR, "../Data/")
f2 = h5py.File("{}30SLight1_Tracked_Fewer.hdf5".format(save_path), "r+")


# In[26]:


list(f2.keys())


# In[31]:


relevant_masks = np.where(np.isin(f2["frame1"]['class_ids'].value,[3,8]))[0]
print(relevant_masks)
print(f2["frame1"]['class_ids'].value)
f2["frame1"]['class_ids'].value[relevant_masks]


# In[46]:


f = h5py.File("{}30SLight1_Fewer.hdf5".format(save_path), "w")
f.create_dataset("class_names",data= f2["class_names"])
for key in list(f2.keys()):
    if (key == 'class_names'):
        continue
    relevant_masks = np.where(np.isin(f2[key]['class_ids'].value,[3,8,10]))[0]
    f.create_dataset("{}/class_ids".format(key), data =f2[key]['class_ids'].value[relevant_masks])
    f.create_dataset("{}/image".format(key), data = f2[key]["image"].value)
    f.create_dataset("{}/rois".format(key), data = f2[key]["rois"].value[relevant_masks])
    f.create_dataset("{}/scores".format(key), data = f2[key]["scores"].value[relevant_masks])
    f.create_dataset("{}/masks".format(key), data = f2[key]["masks"].value[:,:,relevant_masks])


# In[56]:


f.close()


# In[48]:


f = h5py.File("{}30SLight1_Fewer.hdf5".format(save_path), "r")


# In[ ]:


def get_mask_area(mask):
    return np.sum(mask)


# In[49]:


mask_list = []
for key in list(f2.keys()):
    if (key == 'class_names'):
        continue
        
    for mask_idx in f2[key]["masks"].value.shape[2]:
        mask_area = get_mask_area(f2[key]["masks"].value[:,:,mask_idx])
        mask_list.append(mask_area)
        
    


# In[54]:


f["frame1"]['masks'].value.shape


# In[60]:


from scipy.spatial import distance

def find_nn(query, array):
    nn = 0
    min_dist = float('inf')
    for i in range(array.shape[0]):
        dist = distance.euclidean(query,array[i,:])
        if ( dist < min_dist):
            nn = i 
            min_dist = dist
    
    return nn
    


# In[61]:


for i in range(1,len(list(f2.keys()))):
    key = "frame{}".format(i)
    #relevant_masks = np.where(np.isin(f2[key]['class_ids'].value,[3,6,8,10]))[0]
    bbox_handle = f2[key]['rois']

    
    print("Handling key"+key)
    #print("the relevant masks are at")
    #print(relevant_masks)
    
    detection = bbox_handle.value
    print("the detection is")
    print(detection)
    
    scores = f2[key]['scores'].value
    print("the scores are")
    print(scores)
    number_of_masks= scores.shape[0]
    
    detection_with_scores = np.hstack((detection,np.reshape(scores,(-1,1)) ))
    
    
   
    
    print("the detection with scores is")
    #print(detection_with_scores)
    
    track_bbs_ids = mot_tracker.update(detection_with_scores)

    print("the tracked bbs")
    print (track_bbs_ids)
    
    #Associate the track_BBs with the original bbs
    # for each of the track bbs
    # find the nearest neighbour in the original detections
    #associate the ID with the index of the original detection
    
    index_array = np.zeros(number_of_masks)
    for track in track_bbs_ids :
        nn_index = find_nn(track[:-1], detection)
        print(nn_index)
        index_array[nn_index] = track[-1]
        
    
    print("The index array is")
    print(index_array)
        
    f2.create_dataset("{}/ID".format(key), data = index_array)
 
        
    


# In[62]:


f2.close()


# In[17]:


f2 = h5py.File("{}30SLight1_Tracked.hdf5".format(save_path), "r")


# In[21]:


f2["frame3"]["ID"].value


# In[ ]:


"""for i in range(1,len(list(f2.keys()))+1):
    key = "frame{}".format(i)
    if (key=="class_names"):
        continue
    relevant_masks = np.where(np.isin(f2[key]['class_ids'].value,[3,8]))[0]
    bbox_handle = f2[key]['rois']
    
    print("Handling key"+key)
    print("the relevant masks are at")
    print(relevant_masks)
    
    detection = bbox_handle.value[relevant_masks,:]
    print("the detection is")
    print(detection)
    
    scores = f2[key]['scores'].value[relevant_masks]
    print("the scores are")
    print(scores)
    
    detection_with_scores = np.hstack((detection,np.reshape(scores,(-1,1)) ))
    
    
   
    
    print("the detection with scores is")
    #print(detection_with_scores)
    
    track_bbs_ids = mot_tracker.update(detection_with_scores)
    print("the tracked bbs")
    
        
    print (track_bbs_ids)
    
    
    #for mask_id in relevant_masks[0]:

        
        #bbox_handle[mask_id,:] = track_bbs_ids
        
     #   i+=1
        
    """


# In[41]:


f2["frame1"]["rois"]


# In[14]:


f2.close()

