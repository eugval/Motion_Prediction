import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import h5py

from sort.sort import Sort




def track_in_class(data_import_path, track_class):
    # create instance of SORT
    mot_tracker = Sort()

    # get detections
    f = h5py.File(data_import_path, "r")



    # update SORT
    track_bbs_ids = mot_tracker.update(detections)

    # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
    ...


