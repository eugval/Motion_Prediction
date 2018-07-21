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
import json
import datetime

from experiments.model import   SimpleUNet, Unet, UnetShallow, SpatialUnet
from experiments.evaluation_metrics import DistanceViaMean, DistanceViaMode, LossMetric , IoUMetric
from experiments.training_tracker import  TrainingTracker
from experiments.load_data import DataFromH5py, ResizeSample , ToTensor, RandomCropWithAspectRatio, RandomHorizontalFlip, RandomNoise, RandomRotation
from experiments.early_stopper import EarlyStopper, SmoothedEarlyStopper
from experiments.custom_losses import IoULoss, DistanceLoss, DistancePlusIoU
from deprecated.experiment import main_func

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_func(data_names, device):
    for data_name in data_names:
        print("Dealing with {}".format(data_name))
        sys.stdout.flush()

        ###### PARAMETERS #######
        descriptive_text = '''
        Spatial Unet with double convolution, iou plus distance loss, depth 4 apart from bottleneck depth 3, 
        dropout 0.5, early stopping at 0.4, 100 epochs
         '''


        #inputs, label and model params
        model = SpatialUnet
        model_name = "SpatialUnet_MI_{}_2".format(data_name) # For test change here
        only_one_mask = False
        input_types = ['images', 'masks']
        label_type = 'future_mask'
        number_of_inputs = 12

        model_inputs = [number_of_inputs]

        #training params
        loss_used = 'iou_plus_dist' # 'iou_plus_dist' 'iou' 'dist'
        optimiser_used = 'adam'
        momentum = 0.9
        num_epochs = 100
        batch_size = 32    # For test change here
        learning_rate = 0.001
        eval_percent = 0.1
        patience = 4
        use_loss_for_early_stopping = True
        use_smoothed_early_stopping = True
        early_stopper_weight_factor = 0.4


        #Data selection params
        high_movement_bias = False

        #data manipulation/augmentation params
        resize_height =  128
        resize_width = 2*resize_height

        random_crop = True
        crop_order = 15

        random_horizontal_flip = True

        random_rotation = True
        max_rotation_angle = 3

        random_noise = False

        #evaluation params
        eval_batch_size = 128

        #Retrieving file paths
        if(high_movement_bias):
            idx_sets_file = os.path.join(PROCESSED_PATH, "{}/{}_sets_high_movement.pickle".format(data_name,data_name))
        else:
            idx_sets_file = os.path.join(PROCESSED_PATH, "{}/{}_sets.pickle".format(data_name,data_name))

        dataset_file = os.path.join(PROCESSED_PATH, "{}/{}_dataset.hdf5".format(data_name,data_name))
        baselines_file = os.path.join(PROCESSED_PATH, "{}/{}_metrics_to_beat.pickle".format(data_name,data_name))

        #Saving files, folder paths
        model_folder = os.path.join(MODEL_PATH, "{}/".format(model_name))
        model_file = os.path.join(MODEL_PATH, "{}/{}.pkl".format(model_name,model_name))
        model_history_file = os.path.join(MODEL_PATH, "{}/{}_history.pickle".format(model_name,model_name))
        param_holder_file = os.path.join(model_folder, "param_holder.pickle")
        param_holder_json = os.path.join(model_folder, "param_holder.json")

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # hyperparam holder for disk saving
        param_holder = {'descriptive_text': descriptive_text,
                        'data_name': data_name,
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
                        'use_loss_for_early_stopping': use_loss_for_early_stopping,
                        'resize_height': resize_height,
                        'resize_width': resize_width,
                        'random_crop': random_crop,
                        'random_horizontal_flip': random_horizontal_flip,
                        'random_rotation': random_rotation,
                        'random_noise': random_noise,
                        'max_rotation_angle': max_rotation_angle,
                        'crop_order': crop_order,
                        'use_smoothed_early_stopping': use_smoothed_early_stopping,
                        'early_stopper_weight_factor': early_stopper_weight_factor,
                        'label_type': label_type,
                        'loss_used': loss_used,
                        'dataset_file': dataset_file,
                        'model_file' : model_file,
                        'model_history_file': model_history_file,
                        'idx_sets_file':idx_sets_file,
                        'model_folder': model_folder,
                        'baselines_file': baselines_file,
                        'optimiser_used':optimiser_used,
                        'momentum':momentum,
                        'datetime': str(datetime.datetime.now()),
                        'high_movement_bias': high_movement_bias
                        }

        for k,v in param_holder.items():
            print('{} : {}'.format(k,v))
        sys.stdout.flush()


        with open(param_holder_json, 'w') as file:
            file.write(json.dumps(param_holder))


        start_time = time.time()

        ###### Grab the data ######

        #Get the train/val/test split
        idx_sets = pickle.load( open(idx_sets_file, "rb" ) )
        param_holder['idx_sets' ] = idx_sets

        #Add the data augmentation
        input_transforms = []
        eval_transforms = []

        if(random_horizontal_flip):
            input_transforms.append(RandomHorizontalFlip())
        if(random_rotation):
            input_transforms.append(RandomRotation(rotation_range = max_rotation_angle))
        if(random_crop):
            input_transforms.append(RandomCropWithAspectRatio( max_crop = crop_order))
        if(random_noise):
            input_transforms.append(RandomNoise())

        input_transforms.append(ResizeSample(height= resize_height, width = resize_width))
        input_transforms.append(ToTensor())

        eval_transforms.append(ResizeSample(height= resize_height, width = resize_width))
        eval_transforms.append(ToTensor())

        train_set = DataFromH5py(dataset_file,idx_sets,purpose = 'train', input_type = input_types,
                                 label_type= label_type, only_one_mask=only_one_mask,
                                 transform = transforms.Compose(input_transforms))

        param_holder['future_time'] = train_set.future_time
        param_holder['input_frame_num'] = train_set.number_of_inputs
        param_holder['input_timestep'] = train_set.timestep

        val_set = DataFromH5py(dataset_file, idx_sets, purpose ='val', input_type = input_types, label_type= label_type,
                               only_one_mask=only_one_mask,   transform = transforms.Compose(eval_transforms))

        #Make a dataset with a subset of the training examples for evaluation
        idx_set_eval = {'train': np.random.choice(idx_sets['train'], int(len(train_set)*eval_percent), replace=False)}
        eval_train_set = DataFromH5py(dataset_file,idx_set_eval, purpose = 'train',input_type = input_types,
                                      label_type= label_type, only_one_mask=only_one_mask,
                                      transform = transforms.Compose(eval_transforms))

        param_holder['idx_set_eval'] = idx_set_eval

        train_dataloader = DataLoader(train_set, batch_size= batch_size,
                                shuffle=True)

        val_dataloader = DataLoader(val_set, batch_size= eval_batch_size)

        train_eval_dataloader = DataLoader(eval_train_set, batch_size= eval_batch_size)



        ### Save the param holder ####
        pickle.dump(param_holder, open(param_holder_file, "wb"))

        ###### Define the Model ####
        model = model(*model_inputs)
        model.to(device)
        print(model)


        ##### Define the Loss/ Optimiser #####
        if(loss_used =='bce'):
            criterion = nn.BCEWithLogitsLoss(size_average = True)
        elif(loss_used == 'iou'):
            criterion = IoULoss(device = device)
        elif(loss_used == 'dist'):
            criterion = DistanceLoss(device = device)
        elif(loss_used == 'iou_plus_dist'):
            criterion = DistancePlusIoU(device = device)


        if(optimiser_used == 'rmsprop'):
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
        elif(optimiser_used == 'adam'):
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
        elif(optimiser_used == 'sgd'):
            torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate, momentum = momentum)


        #### Instantiate history tracking objects ####
        tracker = TrainingTracker(int(len(train_set)/batch_size))
        loss_metric = LossMetric()
        iou_bbox = IoUMetric(type = 'bbox')
        iou_mask = IoUMetric(type = 'mask')
        distance_via_mean = DistanceViaMean()


        ##### Instantiate Early Stopping Object ######
        if(use_smoothed_early_stopping):
            early_stopper = SmoothedEarlyStopper(patience, seek_decrease = use_loss_for_early_stopping)
        else:
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


            data_load_time =  (time.time() - start_time)
            backprop_time = 0

            for i, data in enumerate(train_dataloader):
                inputs = data['input'].float().to(device)
                labels = data['label'].float().to(device)
                labels = labels.unsqueeze(1)

                future_centroids = data['future_centroid'].float().to(device)


                data_load_time = (time.time() - start_time) - data_load_time
                backprop_time = (time.time() - start_time)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                #TODO: add the sigmoid in the Iou Loss so that I dont have to have it here
                if(loss_used == 'iou' ):
                    outputs = model.eval_forward(inputs)
                    loss = criterion(outputs, labels)
                elif(loss_used == 'dist'):
                     outputs = model.eval_forward(inputs)
                     loss = criterion(outputs, future_centroids)
                elif( loss_used == 'iou_plus_dist'):
                    outputs = model.eval_forward(inputs)
                    loss = criterion(outputs, labels, future_centroids)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)


                loss.backward()
                optimizer.step()

                #break

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

                model.eval()

                train_loss =  loss_metric.evaluate(model, criterion, loss_used, train_eval_dataloader, device)
                train_iou_bbox = iou_bbox.evaluate(model,train_eval_dataloader, device )
                train_iou_mask = iou_mask.evaluate(model,train_eval_dataloader, device )
                train_dist = distance_via_mean.evaluate(model,train_eval_dataloader,device)

                tracker.add(train_loss,'train_loss')
                tracker.add(train_iou_bbox,'train_iou_bbox')
                tracker.add(train_iou_mask, 'train_iou_mask')
                tracker.add(train_dist, 'train_dist')

                #Evaluate on Valisation Set
                val_loss = loss_metric.evaluate(model, criterion,loss_used, val_dataloader, device)
                val_iou_bbox = iou_bbox.evaluate(model,val_dataloader, device )
                val_iou_mask = iou_mask.evaluate(model,val_dataloader, device )
                val_dist = distance_via_mean.evaluate(model,val_dataloader,device)

                tracker.add(val_loss,'val_loss')
                tracker.add(val_iou_bbox,'val_iou_bbox')
                tracker.add(val_iou_mask, 'val_iou_mask')
                tracker.add(val_dist, 'val_dist')

                model.train()
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

            print('saving model? {}'.format(save_model))
            if(save_model):
                print('recording saved model for plotting..')
                tracker.record_saving(epoch)
                torch.save(model.state_dict(), model_file)

            print('Saving training tracker....')
            pickle.dump(tracker, open(model_history_file, "wb"))

            print("Epoch {} Done".format(epoch))
            sys.stdout.flush()

        print("Record finished training...")
        tracker.record_finished_training()
        pickle.dump(tracker, open(model_history_file, "wb"))

    print('FINISHED ALL')












if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # For test change here
    #device = 'cpu'
    print(device)


    data_names = ['Football1and2']  #'Football2_1person' 'Football1and2', 'Crossing1','Crossing2' 'Football1_sm'    # For test change here


    train_func(data_names, device)

    main_func()
