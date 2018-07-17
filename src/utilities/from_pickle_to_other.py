import os
import sys
import json
import pickle
import numpy as np

ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))
sys.path.append(os.path.join(ROOT_DIR,"deprecated"))


def from_pickle_to_json(pickle_file,json_file):
    obj = pickle.load( open(pickle_file, "rb" ) )

    to_delete = []
    for k, v in obj.items():
        if(isinstance(v, dict) or isinstance(v, np.ndarray) or  isinstance(v, np.int64)):
            to_delete.append(k)

    for k in to_delete:
        del obj[k]

    with open(json_file, 'w') as file:
            file.write(json.dumps(obj))



if __name__ == '__main__':
    data_names = [ 'Football2_1person']  #'Football2_1person' 'Football1and2', 'Crossing1','Crossing2'

    for data_name in data_names:
        model_name = "UnetShallow_M_1rstGen_{}".format(data_name)
        model_folder = os.path.join(MODEL_PATH, "{}/".format(model_name))
        param_holder_file = os.path.join(model_folder, "param_holder.pickle")
        param_holder_json = os.path.join(model_folder, "param_holder.json")

        from_pickle_to_json(param_holder_file,param_holder_json)