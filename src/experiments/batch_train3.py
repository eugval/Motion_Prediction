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
import experiments.train3 as train3
import experiments.train4 as train4
import experiments.train5 as train5
import experiments.train6 as train6
import experiments.train7 as train7
import experiments.train8 as train8
import experiments.train9 as train9

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # For test change here
#device = 'cpu'
print(device)


data_names = ['Football1and2']
train1.train_func(data_names,device)

data_names = ['Crossing1']
train3.train_func(data_names,device)



main_func()