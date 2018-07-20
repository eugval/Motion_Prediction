import os
import sys
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")
from shutil import copyfile

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))
sys.path.append(os.path.join(ROOT_DIR,"deprecated"))

import torch

from deprecated.experiment import main_func

import experiments.train as train1
import experiments.train2 as train2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # For test change here
#device = 'cpu'
print(device)


data_names = ['Football1and2']  #'Football2_1person' 'Football1and2', 'Crossing1','Crossing2' 'Football1_sm'

train1.train_func(data_names, device)
train2.train_func(data_names, device)

main_func()