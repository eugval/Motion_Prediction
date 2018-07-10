#https://www.youtube.com/watch?v=YcTCIMKeiNQ
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
from experiments.load_data import DataFromH5py, ResizeSample , ToTensor, RandomCropWithAspectRatio, RandomHorizontalFlip, RandomNoise, RandomRotation
from experiments.early_stopper import EarlyStopper
from deprecated.experiment import main_func

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)


data_names = ['Crossing1' ] # 'Football2_1person' 'Football1and2', 'Crossing1','Crossing2'


for data_name in data_names:
    print("Dealing with {}".format(data_name))
    sys.stdout.flush()

    ###### PARAMETERS #######
    #inputs and model params
    model_name = "Unet_MI_3ndGen_{}".format(data_name)
    only_one_mask = False
    input_types = ['masks', 'images']
    number_of_inputs = 12 # 3 RGB images + 3 masks


    #training params
    num_epochs = 40
    batch_size = 32
    learning_rate = 0.01
    eval_percent = 0.1
    patience = 6
    use_loss_for_early_stopping = True

    #data manipulation/augmentation params
    resize_height = 64
    resize_width = 2*resize_height

    random_crop = True
    random_horizontal_fip = True
    random_rotation = True
    random_noise = False

    #evaluation params
    eval_batch_size = 128

    #hyperparam holder for disk saving
    param_holder ={'data_name': data_name,
                   'model_name': model_name,
                   'num_epochs': num_epochs,
                   'batch_size': batch_size,
                   'learning_rate': learning_rate,
                   'eval_percent': eval_percent,
                   'only_one_mask': only_one_mask,
                   'patience': patience,
                   'input_types': input_types,
                   'number_of_inputs': number_of_inputs,
                   'eval_batch_size': eval_batch_size,
                   'use_loss_for_early_stopping':use_loss_for_early_stopping,
                   'resize_height':resize_height,
                   'resize_width':resize_width,
                   'random_crop':random_crop,
                   'random_horizontal_fip': random_horizontal_fip,
                   'random_rotation': random_rotation,
                   'random_noise': random_noise
                   }



    #Retrieving file paths
    dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name,data_name))
    idx_sets_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(data_name,data_name))

    #Saving files, folder paths
    model_folder = os.path.join(MODEL_PATH, "{}/".format(model_name))
    model_file = os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))
    model_history_file = os.path.join(MODEL_PATH, "{}/{}_history.pickle".format(model_name,model_name))
    param_holder_file = os.path.join(model_folder, "param_holder.pickle")

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)


    start_time = time.time()

    ###### Grab the data ######

    #Get the train/val/test split
    idx_sets = pickle.load( open(idx_sets_file, "rb" ) )
    param_holder['idx_sets' ] = idx_sets

    #Add the data augmentation
    input_transforms = []

    if(random_horizontal_fip):
        input_transforms.append(RandomHorizontalFlip())
    if(random_rotation):
        input_transforms.append(RandomRotation())
    if(random_crop):
        input_transforms.append(RandomCropWithAspectRatio())
    if(random_noise):
        input_transforms.append(RandomNoise())

    input_transforms.append(ResizeSample(height= resize_height, width = resize_width))
    input_transforms.append(ToTensor())

    train_set = DataFromH5py(dataset_file,idx_sets,input_type = input_types,only_one_mask=only_one_mask,
                             transform = transforms.Compose(input_transforms))

    param_holder['future_time'] = train_set.future_time
    param_holder['input_frame_num'] = train_set.number_of_inputs
    param_holder['input_timestep'] = train_set.timestep


    val_set = DataFromH5py(dataset_file, idx_sets, input_type = input_types,
                           only_one_mask=only_one_mask, purpose ='val',  transform = transforms.Compose(input_transforms))

    #Make a dataset with a subset of the training examples for evaluation
    idx_set_eval = {'train': np.random.choice(idx_sets['train'], int(len(train_set)*eval_percent), replace=False)}
    eval_train_set = DataFromH5py(dataset_file,idx_set_eval,
                                  only_one_mask=only_one_mask,input_type = input_types,transform = transforms.Compose(input_transforms))

    param_holder['idx_set_eval'] = idx_set_eval

    train_dataloader = DataLoader(train_set, batch_size= batch_size,
                            shuffle=True)

    val_dataloader = DataLoader(val_set, batch_size= eval_batch_size,
                            shuffle=True)

    train_eval_dataloader = DataLoader(eval_train_set, batch_size= eval_batch_size,
                            shuffle=True)



    ### Save the param holder ####
    pickle.dump(param_holder, open(param_holder_file, "wb"))

    ###### Define the Model ####
    model = Unet(number_of_inputs)
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
    distance_via_mean = DistanceViaMean()


    ##### Instantiate Early Stopping Object ######
    early_stopper = EarlyStopper(patience, seek_decrease = use_loss_for_early_stopping)

    print("Ready to train")
    sys.stdout.flush()
    #### Start the training ####

    for epoch in range(num_epochs):
        if(not early_stopper.continue_training()):
            print("Early Stopping Triggered, Breaking out of the training loop")
            print("--- %s seconds elapsed ---" % (time.time() - start_time))
            break

        print("Training Epoch {}".format(epoch))
        print("--- %s seconds elapsed ---" % (time.time() - start_time))
        sys.stdout.flush()

        once = True
        data_load_time =  (time.time() - start_time)
        backprop_time = 0

        for i, data in enumerate(train_dataloader):
            inputs = data['input'].float().to(device)
            labels = data['label'].float().to(device)
            labels = labels.unsqueeze(1)


            data_load_time = (time.time() - start_time) - data_load_time
            backprop_time = (time.time() - start_time)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            backprop_time = (time.time() - start_time) - backprop_time

        with torch.no_grad():
            print("Finished training epoch {}".format(epoch))
            print("--- %s seconds elapsed ---" % (time.time() - start_time))
            print("data_load_time {}".format(data_load_time))
            print("backprop time {}".format(backprop_time))
            print("Evaluating...")
            sys.stdout.flush()

            #Evaluate Model
            #Evaluate on Trainnig set
            train_loss =  loss_metric.evaluate(model, criterion, train_eval_dataloader, device)
            train_iou_bbox = iou_bbox.evaluate(model,train_eval_dataloader, device )
            train_iou_mask = iou_mask.evaluate(model,train_eval_dataloader, device )
            train_dist = distance_via_mean.evaluate(model,train_eval_dataloader,device)

            tracker.add(train_loss,'train_loss')
            tracker.add(train_iou_bbox,'train_iou_bbox')
            tracker.add(train_iou_mask, 'train_iou_mask')
            tracker.add(train_dist, 'train_dist')

            #Evaluate on Valisation Set
            val_loss = loss_metric.evaluate(model, criterion, val_dataloader, device)
            val_iou_bbox = iou_bbox.evaluate(model,val_dataloader, device )
            val_iou_mask = iou_mask.evaluate(model,val_dataloader, device )
            val_dist = distance_via_mean.evaluate(model,val_dataloader,device)

            tracker.add(val_loss,'val_loss')
            tracker.add(val_iou_bbox,'val_iou_bbox')
            tracker.add(val_iou_mask, 'val_iou_mask')
            tracker.add(val_dist, 'val_dist')

            print('Train loss: {}'.format(train_loss))
            print('Train iou bbox: {}'.format(train_iou_bbox))
            print('Train iou mask: {}'.format(train_iou_mask))
            print('Train centroid distance: {}'.format(train_dist))
            print('Val loss: {}'.format(val_loss))
            print('Val iou bbox: {}'.format(val_iou_bbox))
            print('Val iou mask: {}'.format(val_iou_mask))
            print('Val centroid distance: {}'.format(val_dist))

            print("Finished Evaluating Epoch {}".format(epoch))
            print("--- %s seconds elapsed ---" % (time.time() - start_time))
            sys.stdout.flush()

        if(not use_loss_for_early_stopping):
            save_model = early_stopper.checkpoint(val_iou_mask)
        else:
            save_model = early_stopper.checkpoint(val_loss)


        sys.stdout.flush()
        if(save_model):
            tracker.record_saving()
            torch.save(model.state_dict(), model_file)

        print('Saving training tracker....')
        pickle.dump(tracker, open(model_history_file, "wb"))

    tracker.record_finished_training()
    pickle.dump(tracker, open(model_history_file, "wb"))
    print('FINISHED ALL')

    main_func()



















