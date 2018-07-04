#https://www.youtube.com/watch?v=YcTCIMKeiNQ
import os
import sys
ROOT_DIR = os.path.abspath("../")
PROCESSED_PATH = os.path.join(ROOT_DIR, "../data/processed/")
MODEL_PATH = os.path.join(ROOT_DIR,"../models/")


sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,"experiments"))
sys.path.append(os.path.join(ROOT_DIR,"deprecated"))

import torch
import pickle
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import  DataLoader

from experiments.model import   SimpleUNet, Unet
from experiments.evaluation_metrics import DistanceViaMean, DistanceViaMode, LossMetric , IoUMetric
from experiments.training_tracker import  TrainingTracker
from experiments.load_data import DataFromH5py, ResizeSample , ToTensor
from deprecated.experiment import main_func

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)


#data_names = ['Football1', 'Crossing1', 'Light1', 'Football2', 'Crossing2' ]
data_names = ['Football2_1person']


for data_name in data_names:
    ###### PARAMETERS #######
    model_name = "Unet_{}".format(data_name)
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.01
    eval_percent = 0.1



    #Retrieving files
    dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name,data_name))
    idx_sets_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(data_name,data_name))

    #Saving files, folders
    model_folder = os.path.join(MODEL_PATH, "{}/".format(model_name))
    model_file = os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))
    model_history_file = os.path.join(MODEL_PATH, "{}/{}_history.pickle".format(model_name,model_name))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)


    start_time = time.time()

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

    #Make a dataset with a subset of the training examples for evaluation
    idx_set_eval = {'train': np.random.choice(idx_sets['train'], int(len(train_set)*eval_percent), replace=False)}
    eval_train_set = DataFromH5py(dataset_file,idx_set_eval, transform = transforms.Compose([
                                                   ResizeSample(),
                                                   ToTensor()
                                                  ]))


    train_dataloader = DataLoader(train_set, batch_size= batch_size,
                            shuffle=True)

    val_dataloader = DataLoader(val_set, batch_size= 128,
                            shuffle=True)

    train_eval_dataloader = DataLoader(eval_train_set, batch_size= 128,
                            shuffle=True)




    ###### Define the Model ####
    model = Unet(3)
    model.to(device)
    print(model)


    ##### Define the Loss/ Optimiser #####
    criterion = nn.BCEWithLogitsLoss(size_average = True)
    optimizer = optim.RMSprop(model.parameters(), lr = learning_rate)


    #### Instantiate history tracking objects ####
    tracker = TrainingTracker(int(len(train_set)/batch_size))
    loss_metric = LossMetric()
    iou_bbox = IoUMetric(type = 'bbox')
    iou_mask = IoUMetric(type = 'mask')


    print("Ready to train")
    sys.stdout.flush()
    #### Start the training ####

    for epoch in range(num_epochs):
        print("Training Epoch {}".format(epoch))
        print("--- %s seconds elapsed ---" % (time.time() - start_time))
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

        with torch.no_grad():
            print("Finished training epoch {}".format(epoch))
            print("--- %s seconds elapsed ---" % (time.time() - start_time))
            print("Evaluating...")
            sys.stdout.flush()

            #Evaluate Model
            #Evaluate on Trainnig set
            train_loss =  loss_metric.evaluate(model, criterion, train_eval_dataloader, device)
            train_iou_bbox = iou_bbox.evaluate(model,train_eval_dataloader, device )
            train_iou_mask = iou_mask.evaluate(model,train_eval_dataloader, device )

            tracker.add(train_loss,'train_loss')
            tracker.add(train_iou_bbox,'train_iou_bbox')
            tracker.add(train_iou_mask, 'train_iou_mask')

            #Evaluate on Valisation Set
            val_loss = loss_metric.evaluate(model, criterion, val_dataloader, device)
            val_iou_bbox = iou_bbox.evaluate(model,val_dataloader, device )
            val_iou_mask = iou_mask.evaluate(model,val_dataloader, device )

            tracker.add(val_loss,'val_loss')
            tracker.add(val_iou_bbox,'val_iou_bbox')
            tracker.add(val_iou_mask, 'val_iou_mask')

            print('Train loss: {}'.format(train_loss))
            print('Train iou bbox: {}'.format(train_iou_bbox))
            print('Train iou mask: {}'.format(train_iou_mask))
            print('Val loss: {}'.format(val_loss))
            print('Val iou bbox: {}'.format(val_iou_bbox))
            print('Val iou mask: {}'.format(val_iou_mask))



        print("Finished Evaluating Epoch {}".format(epoch))
        print("--- %s seconds elapsed ---" % (time.time() - start_time))

        print("Finished Training, saving model and  training tracker...")
        sys.stdout.flush()
        torch.save(model.state_dict(), model_file)

    pickle.dump(tracker, open(model_history_file, "wb"))




















