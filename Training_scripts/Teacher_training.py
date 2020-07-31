import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import time
import torch.nn.functional as F
import torch.utils.data
from sklearn import metrics 
import os
import sys, getopt
sys.path.append('../utils/')
from Models import TeacherModel
from ImbalancedDatasetSampler import ImbalancedDatasetSampler
import argparse

parser = argparse.ArgumentParser(description='TMA patches extractor')
parser.add_argument('-d','--DATASET', type=str, default='train', help='dataset to annotate patches (train, valid, test)')
parser.add_argument('-a','--APPROACH', type=str, default='ssl', help='teacher/student approach: ssl (semi-supervised), swsl (semi-weakly supervised)')
parser.add_argument('-w','--N_EXP_weak', type=int, default=0, help='number experiment to finetune (only semi-weakly supervised)')
parser.add_argument('-s','--N_EXP_strong', type=int, default=0, help='number experiment to save')
parser.add_argument('-e','--EPOCHS', type=int, default=15, help='epochs of training')
parser.add_argument('-b','--BATCH_SIZE', type=int, default=32, help='batch size')

args = parser.parse_args()

print( torch.cuda.current_device())
print( torch.cuda.device_count())

assert not(args.APPROACH=='ssl' and args.DATASET=='weak')

def create_dir(directory):
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except OSError:
            print ("Creation of the directory %s failed" % directory)
        else:
            print ("Successfully created the directory %s " % directory) 

if (args.APPROACH=='ssl'):
    approach = 'Semi_Supervised'
elif (args.APPROACH=='swsl'):
    approach = 'Semi_Weakly_Supervised'

if (args.DATASET=='strong'):
    csv_folder = 'LOCAL/PATH/../Teacher_Student_models/csv_files/Strongly_Annotated_data/'
    model_labels_approach = 'strong_labels_training' 
    N_EXP = args.N_EXP_strong   

elif(args.DATASET=='weak'):
    csv_folder = 'LOCAL/PATH/../Teacher_Student_models/csv_files/Weakly_Annotated_Data/'
    model_labels_approach = 'weak_labels_training' 
    N_EXP = args.N_EXP_weak

models_folder = 'LOCAL/PATH/../Teacher_Student_models/models_weights/'

models_path = models_folder+approach+'/'
create_dir(models_path)

models_path = models_path+'Teacher_model/'
create_dir(models_path)

models_path = models_path+model_labels_approach+'/'
create_dir(models_path)

models_path = models_path+'N_EXP_'+str(N_EXP)+'/'
create_dir(models_path) 

checkpoint_path = models_path+'checkpoints/'
create_dir(checkpoint_path)

model_weights = models_path+'teacher_model.pt'

train_csv = csv_folder+'train_patches.csv'
valid_csv = csv_folder+'valid_patches.csv'
test_csv = csv_folder+'test_patches.csv'

train_dataset = pd.read_csv(train_csv,header=None).values
valid_dataset = pd.read_csv(valid_csv,header=None).values
test_dataset = pd.read_csv(test_csv,header=None).values


if (args.DATASET=='weak' or (args.DATASET=='strong' and args.APPROACH=='ssl')):
    model = TeacherNetwork()

elif (args.DATASET=='strong' and args.APPROACH=='swsl'):
    model_to_finetune = models_folder+approach+'/Teacher_model/weak_labels_training/N_EXP_'+args.N_EXP_weak+'/teacher_model.pt'
    model = torch.load(model_to_finetune)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Parameters
batch_size = args.BATCH_SIZE
params_train = {'batch_size': batch_size,
          #'shuffle': True,
          'sampler': ImbalancedDatasetSampler(train_dataset),
          'num_workers': 32}

batch_size_test = 500
params_valid = {'batch_size': batch_size_test,
          'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(train_dataset),
          'num_workers': 32}

params_test = {'batch_size': batch_size_test,
          'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(test_dataset),
          'num_workers': 32}

num_epochs = args.EPOCHS

from Data_Generator import Dataset_patches
#CREATE GENERATORS
#train
training_set = Dataset_patches(train_dataset[:,0], train_dataset[:,1],'train')
training_generator = data.DataLoader(training_set, **params_train)

validation_set = Dataset_patches(valid_dataset[:,0], valid_dataset[:,1],'valid')
validation_generator = data.DataLoader(validation_set, **params_valid)

testing_set = Dataset_patches(test_dataset[:,0], test_dataset[:,1],'test')
testing_generator = data.DataLoader(testing_set, **params_test)

if (torch.cuda.device_count() > 1):
    model = torch.nn.DataParallel(model)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

class_sample_count = np.unique(train_dataset[:,1], return_counts=True)[1]
weight = class_sample_count / len(train_dataset[:,1])
samples_weight = torch.from_numpy(weight).type(torch.FloatTensor)

import torch.optim as optim
criterion = torch.nn.CrossEntropyLoss(weight=samples_weight.to(device))
optimizer = optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
model.to(device)

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return metrics.accuracy_score(y_true, y_pred)

def kappa_score(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return  metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')

def evaluate_validation_set(generator):
    #accumulator for validation set
    y_pred_val = []
    y_true_val = []

    valid_loss = 0.0

    with torch.no_grad():
        j = 0
        for inputs,labels in generator:
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            try:
                outputs = model(inputs)
            except:
                outputs = model.module(inputs)

            loss = criterion(outputs, labels)
            outputs = F.softmax(outputs)

            valid_loss = valid_loss + ((1 / (j+1)) * (loss.item() - valid_loss)) 
            
            #accumulate values
            outputs_np = outputs.cpu().data.numpy()
            labels_np = labels.cpu().data.numpy()
            outputs_np = np.argmax(outputs_np, axis=1)
            
            y_pred_val = np.append(y_pred_val,outputs_np)
            y_true_val = np.append(y_true_val,labels_np)

            j = j+1

        acc_valid = metrics.accuracy_score(y_true=y_true_val, y_pred=y_pred_val)
        kappa_valid = metrics.cohen_kappa_score(y1=y_true_val,y2=y_pred_val, weights='quadratic')
        
    return kappa_valid, valid_loss
# In[35]:

best_loss_valid = 100000.0

for epoch in range(num_epochs): 
    
    #loss functions outputs and network
    train_loss = 0.0
    
    #accuracy for the outputs
    acc = 0.0
    
    is_best = False
    
    i = 0
    
    model.train()
    
    for inputs,labels in training_generator:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        try:
            outputs = model(inputs)
        except:
            outputs = model.module(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        train_loss = train_loss + ((1 / (i+1)) * (loss.item() - train_loss))   
        outputs = F.softmax(outputs)
        #accumulate values
        outputs_np = outputs.cpu().data.numpy()
        labels_np = labels.cpu().data.numpy()
        outputs_np = np.argmax(outputs_np, axis=1)
        
        acc_primary_batch = accuracy(outputs_np, labels_np)
        acc = (acc*i+acc_primary_batch)/(i+1)
        
        print("epoch "+str(epoch)+ " train loss: " + str(train_loss) + " acc_train: " + str(acc))

        i = i+1
        
    model.eval()

    print("epoch "+str(epoch)+ " train loss: " + str(train_loss) + " acc_train: " + str(acc))
    
    print("evaluating validation")
    kappa_valid, valid_loss = evaluate_validation_set(validation_generator)
    
        
    print('[%d] valid loss TCGA: %.4f, kappa_valid: %.4f' %
          (epoch, valid_loss, kappa_valid))         
    
    
    if (best_loss_valid>valid_loss):
        print ("=> Saving a new best model")
        print("previous best loss: " + str(best_loss_valid) + ", new best loss function: " + str(valid_loss))
        best_loss_valid = valid_loss
        torch.save(model, model_weights)


print('Finished Training')
