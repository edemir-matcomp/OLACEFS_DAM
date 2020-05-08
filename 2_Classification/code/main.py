from __future__ import print_function 
from __future__ import division
import os, time, copy, random
import numpy as np
import scipy.stats as st
import scipy
import argparse
from pprint import pprint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from skimage import io

# Deep learning framework
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter

# Personal Network
import network_factory
import data_loader
import metrics

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Detect if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix seed's for reproducibility
random.seed(42)
torch.manual_seed(42)


def train_model(model, dataloaders, criterion, optimizer, scheduler, path_tb, num_epochs, writer, id_fold, is_inception=False):

    num_classes = dataloaders['train'].dataset.num_classes
    mean, std = dataloaders['train'].dataset.mean, dataloaders['train'].dataset.std

    since = time.time()

    val_acc_history = []
    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    counter_early_stop_epochs = 0
    best_acc = 0.0
    epochs_early_stop = 200 

    for epoch in range(num_epochs):
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
        
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            incorrect = torch.tensor([])
            incorrect_path = []
            cm_total = np.zeros((num_classes, num_classes))
            num_iter_per_epoch = len(dataloaders[phase])
            
            # Iterate over data.
            for iteration, (inputs, labels, inputs_path) in enumerate(dataloaders[phase]):
            
                iteration_step = num_iter_per_epoch*epoch + iteration
                        
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        #scheduler.step()
                    
                    
                    elif phase == 'val':
                        mask_incorrect = preds != labels.data
                        
                        
                        for idx, k in enumerate(list(mask_incorrect)):
                            if k:
                                incorrect_path.append(inputs_path[idx])
                        
                        
                        '''
                        #incorrect_path = list(inputs_path)[mask_incorrect]
                        incorrect = torch.cat((incorrect, inputs[mask_incorrect].cpu()),0)
                        '''

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                cm_train = confusion_matrix(labels.data.cpu(), preds.cpu(), labels=range(num_classes))
                cm_total += cm_train
                
                mean_acc = metrics.balanced_accuracy_score_from_cm(cm_train)
                f1_score = metrics.f1_score_from_cm(cm_train)
                
                if phase == 'train':
                
                    # Tensorboard writer
                    writer.add_scalar('{0}_model/{1}_mean_acc_{0}'.format(phase, id_fold), mean_acc, iteration_step)
                    writer.add_scalar('{0}_model/{1}_f1_score_{0}'.format(phase, id_fold), f1_score, iteration_step)
                    writer.add_scalar('{0}_model/{1}_loss_{0}'.format(phase, id_fold), loss.item(), iteration_step)
                
                    print('{} Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Acc: {:.4f} CM: \n{}'.format(phase,
                            epoch, num_epochs, 
                            iteration, len(dataloaders[phase]),
                            loss.item(), mean_acc, cm_train))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = metrics.balanced_accuracy_score_from_cm(cm_total)
            epoch_f1 = metrics.f1_score_from_cm(cm_total)
            
            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} CM: \n{}'.format(phase, epoch_loss, epoch_acc, epoch_f1, cm_total))
            
            if phase == 'val':
            
                counter_early_stop_epochs += 1
                
                # Tensorboard writer
                writer.add_scalar('{0}_model/{1}_mean_acc_{0}'.format(phase, id_fold), epoch_acc, epoch)
                writer.add_scalar('{0}_model/{1}_f1_score_{0}'.format(phase, id_fold), epoch_f1, epoch)
                writer.add_scalar('{0}_model/{1}_loss_{0}'.format(phase, id_fold), epoch_loss, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                counter_early_stop_epochs = 0
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_cm = cm_total
                best_f1 = epoch_f1
                
                '''
                # Inverse Normalization
                rev_inputs = incorrect.new(*incorrect.size())
                print(incorrect.size(1))
                for c in range(incorrect.size(1)):
                    rev_inputs[:, c, :, :] = incorrect[:, c, :, :] * std[c] + mean[c]
                    #rev_inputs[:, c, :, :] *= 255. / rev_inputs[:, c, :, :].max() 
                
                # Create a grid of incorrect samples
                #grid = torchvision.utils.make_grid(incorrect)
                grid = torchvision.utils.make_grid(rev_inputs)
                writer.add_image('images', grid, epoch)
                '''
                
                best_incorrect = incorrect_path
                
        if (counter_early_stop_epochs >= epochs_early_stop):
            print ('Stopping training because validation loss did not improve in ' + str(epochs_early_stop) + ' consecutive epochs.')
            break
                
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val F1: {:4f}'.format(best_f1))
    print('Best CM:\n{0}'.format(best_cm))
    
    results = (best_acc, best_f1, best_cm, best_incorrect)

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    writer.close()
    
    return model, results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Size of batch')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--channels', type=str, default=3, help='Number of channels')
    parser.add_argument('--method', type=str, default='vgg', help='resnet, alexnet, vgg, squeezenet, densenet, inception')
    parser.add_argument('--finetune', type=int, default=1, help='True:1 , False: 0')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay SGD [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]')
    parser.add_argument('--momentum', type=float, default=0.01, help='Momentum SGD')
    parser.add_argument('--path_tb', type=str, default='', help='Path for tensorboard')
    parser.add_argument('--dataset_path', type=str, default='', help='Path for dataset')
    parser.add_argument('--path_classes', type=int, default=0, help='Path file with classes')
    FLAGS, unparsed = parser.parse_known_args()
    
    # Print input arguments
    print(FLAGS)
    
    # Parse to variables
    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs
    channels = FLAGS.channels
    method = FLAGS.method
    finetune = FLAGS.finetune
    feature_extract = finetune
    learning_rate = FLAGS.learning_rate
    weight_decay = FLAGS.weight_decay
    momentum = FLAGS.momentum
    path_tb = FLAGS.path_tb
    dataset_path = FLAGS.dataset_path
    path_classes = FLAGS.path_classes
    
    # Summary for TensorboardX
    run_id, flags_info = random.randint(1,100000), path_tb.split('/')[-1]
    logdir = path_tb #os.path.join(os.path.dirname(path_tb), os.path.basename(path_tb)+'_event/'+flags_info+'/'+str(run_id))
    writer = SummaryWriter(log_dir=logdir)
    
    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    dataset_info = data_loader.get_dataset_base(dataset_path, channels, path_classes)
     
    # Create a Stratified Shuffle Split Outside loop to keep same folds over experiments
    result_fold = {}
    list_indices = []
    np.random.seed(0)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    for id_fold, (train_idx, test_idx) in enumerate(sss.split(np.zeros(dataset_info['len']), dataset_info['list_labels'])):
    
        #indices = {'train': train_idx, 'val': test_idx}
        list_indices.append({'train': train_idx, 'val': test_idx})
        
    for id_fold in range(len(list_indices)):
    
        indices = list_indices[id_fold] 
        
        dataloaders_dict = {}
        
        print(indices)
        
        
        ###TODO
        # Assign image size regarding to model
        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if finetune else (None, None)
        input_size = 299 if method == 'inception' else 224
        #if finetune:
        #    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        #else:
        #    mean, std = None, None
                
        # Resize to match architecture models
        #if method == 'inception':
        #    input_size = 299
        #else:
        #    input_size = 224
                
        #TODO
        
        
        for x in ['train', 'val']:
        
            # New dataset using just subsampling
            tmp_dataset = data_loader.Remote_Sensing_Loader(dataset_info, indices[x])

            # Create a sampler by samples weights            
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights=tmp_dataset.samples_weights,
                num_samples=tmp_dataset.len)#, replacement=True)
                
            
            if x == 'train':
                dataloaders_dict[x] = torch.utils.data.DataLoader(tmp_dataset, 
                                                                  batch_size=batch_size, 
                                                                  sampler=sampler, #torch.utils.data.SubsetRandomSampler(indices[x]), 
                                                                  num_workers=8)
            else:
                dataloaders_dict[x] = torch.utils.data.DataLoader(tmp_dataset, 
                                                                  batch_size=batch_size,
                                                                  num_workers=8)
                                     
                        
            if not finetune:                             
                if x == 'train':
                    mean, std, transform = data_loader.get_transforms(dataloaders_dict[x], x, input_size, mean=None, std=None)
                else:
                    _, _, transform = data_loader.get_transforms(dataloaders_dict[x], x, input_size, mean=mean, std=std)
            else:
                _, _, transform = data_loader.get_transforms(dataloaders_dict[x], x, input_size, mean=mean, std=std)
                
                                                              
            dataloaders_dict[x].dataset.transform = transform
            
            dataloaders_dict[x].dataset.mean = mean
            dataloaders_dict[x].dataset.std = std

        num_classes = dataloaders_dict['train'].dataset.num_classes

        print("Datasets and Dataloaders created")

        # Initialize the model for this run
        model_ft, _ = network_factory.initialize_model(method, num_classes, dataset_info['channels'], feature_extract=finetune, use_pretrained=finetune)

        # Print the model we just instantiated
        print(model_ft)

        # Send the model to GPU
        model_ft = model_ft.to(device)

        params_to_update = model_ft.parameters()
        
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        
        # Observe that all parameters are being optimized
        optimizer_type = 'adam'
        if optimizer_type == 'sgd':
            optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(dataloaders_dict['train'].dataset.weight_class).float().cuda())
        
        #lambda1 = lambda epoch: learning_rate / (1. + epoch*1e-7)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_ft, lr_lambda=lambda1)
        scheduler = None

        # Train and evaluate
        model_ft, res = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler, path_tb, num_epochs, writer, id_fold, is_inception=(method=="inception"))
        
        result_fold[id_fold] = res
        
        #saving best models
        torch.save(model_ft, logdir+'/best_model_{}.pt'.format(id_fold))
        print(mean)
        print(std)
        #model = torch.load('filename.pt')
        
    
    mean_res = [result_fold[res][0] for res in result_fold]
    CI_mean = st.t.interval(0.95, len(mean_res)-1, loc=np.mean(mean_res), scale=st.sem(mean_res))

    pprint(result_fold)
    print(mean_res)
    print('{} {} {}'.format(np.mean(mean_res), np.mean(mean_res) - CI_mean[0] , CI_mean))

