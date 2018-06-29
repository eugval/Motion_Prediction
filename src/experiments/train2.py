#https://www.youtube.com/watch?v=YcTCIMKeiNQ
import os
import sys
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")


sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))

import torch
import pickle
import torch.optim as optim
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import  DataLoader
from experiments.model import   SimpleUNet
from experiments.history_tracking import DistanceViaMean, DistanceViaMode, LossMetric ,TrainingTracker


from experiments.load_data import DataFromH5py, ResizeSample , ToTensor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)





###### PARAMETERS #######
data_name = "Football1"
model_name = "Mask_only_Simple_Unet"
num_epochs = 2
batch_size = 4
learning_rate = 0.01
eval_percent = 0.1








dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name,data_name))
idx_sets_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(data_name,data_name))





###### Grab the data #####

idx_sets = pickle.load( open(idx_sets_file, "rb" ) )

train_set = DataFromH5py(dataset_file,idx_sets, transform = transforms.Compose([
                                               ResizeSample(),
                                               ToTensor()
                                              ]))

val_set = DataFromH5py(dataset_file, idx_sets, purpose ='val',  transform = transforms.Compose([
                                                                   ResizeSample(),
                                                                   ToTensor()
                                                                  ]))

train_dataloader = DataLoader(train_set, batch_size= batch_size,
                        shuffle=True)



###### Define the Model ####
model = SimpleUNet(12)
model.to(device)
print(model)


##### Define the Loss/ Optimiser #####
criterion = nn.MSELoss(size_average = False)
optimizer = optim.RMSprop(model.parameters(), lr = learning_rate)


#### Instantiate history tracking objects ####
loss_metric = LossMetric()
distance_via_mean = DistanceViaMean()
distance_via_mode = DistanceViaMode()
tracker = TrainingTracker()


#### Start the training ####

for epoch in range(num_epochs):
    for i, data in enumerate(train_dataloader):
        inputs = data['input'].float().to(device)
        labels = data['label'].float().to(device)
        labels = labels.unsqueeze(1)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    #Evaluate Model
    #Evaluate on Trainnig set
    train_loss =  loss_metric.evaluate(model, criterion, train_set, device, eval_percent)
    train_dist_mean = distance_via_mean.evaluate(model,train_set, device, eval_percent)
    train_dist_mode = distance_via_mode.evaluate(model,train_set, device, eval_percent)

    tracker.add(train_loss,'train_loss')
    tracker.add(train_dist_mean,'train_dist_mean')
    tracker.add(train_dist_mode, 'train_dist_mode')

    #Evaluate on Valisation Set
    val_loss = loss_metric.evaluate(model, criterion, val_set, device)
    val_dist_mean = distance_via_mean.evaluate(model,val_set, device)
    val_dist_mode = distance_via_mode.evaluate(model,val_set, device)

    tracker.add(val_loss,'val_loss')
    tracker.add(val_dist_mean,'val_dist_mean')
    tracker.add(val_dist_mode, 'val_dist_mode')




















