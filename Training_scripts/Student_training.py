import sys, getopt
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
sys.path.append('../utils/')
from Models import StudentModel
from ImbalancedDatasetSampler import ImbalancedDatasetSampler
import argparse

parser = argparse.ArgumentParser(description='TMA patches extractor')
parser.add_argument('-v','--VARIANT', type=str, default='train', help='student training variant to use (I,II,III,baseline)')
parser.add_argument('-a','--APPROACH', type=str, default='ssl', help='teacher/student approach: ssl (semi-supervised), swsl (semi-weakly supervised)')
parser.add_argument('-s','--SUBSET', type=int, default='19', help='subset of pseudo-labels to use 19=1000, 0=20000 pseudo labels per class')
parser.add_argument('-n','--N_EXP', type=int, default=0, help='number experiment')
parser.add_argument('-p','--PRE_TRAINED', type=int, default=0, help='student to finetune (only variant II)')
parser.add_argument('-e','--EPOCHS', type=int, default=15, help='epochs of training')
parser.add_argument('-b','--BATCH_SIZE', type=int, default=32, help='batch size')

args = parser.parse_args()

print( torch.cuda.current_device())
print( torch.cuda.device_count())


def create_dir(models_path):
    if not os.path.isdir(models_path):
        try:
            os.mkdir(models_path)
        except OSError:
            print ("Creation of the directory %s failed" % models_path)
        else:
            print ("Successfully created the directory %s " % models_path)   

if (args.APPROACH=='ssl'):
    approach = 'Semi_Supervised'
elif (APPROACH=='swsl'):
    approach = 'Semi_Weakly_Supervised'

models_folder = '/LOCAL/PATH/../Teacher_Student_models/models_weights/'
create_dir(models_folder)

models_path = models_folder+approach+'/'
create_dir(models_path)

models_path = models_path+'Student_model/'
create_dir(models_path)

if (args.VARIANT!='baseline'):
    models_path = models_path+'student_variant_'+args.VARIANT+'/'
    create_dir(models_path)

    models_path = models_path+'perc_'+str(args.SUBSET)+'/'
    create_dir(models_path)
else:
    models_path = models_path+'fully_supervised/'
    create_dir(models_path)

models_path = models_path+'N_EXP_'+str(args.N_EXP)+'/'
create_dir(models_path)

checkpoint_path = models_path+'checkpoints/'
create_dir(checkpoint_path)

model_weights = models_path+'student_model.pt'
model_weights_strong = models_path+'student_model_strong.pt'
model_weights_pseudo = models_path+'student_model_pseudo.pt'


strong_dir = '/LOCAL/PATH/../Teacher_Student_models/csv_files/Strongly_Annotated_data/'
pseudo_dir = '/LOCAL/PATH/../Teacher_Student_models/csv_files/Pseudo_Labeled_Data/'+approach+'/'

train_csv_strong = strong_dir+'train_patches.csv'
valid_csv_strong = strong_dir+'valid_patches.csv'
test_csv_strong = strong_dir+'test_patches.csv'

train_csv_pseudo = pseudo_dir+'train/csv_densely_semi_annotated_subset_'+str(args.SUBSET)+'.csv'
valid_csv_pseudo = pseudo_dir+'valid/csv_densely_semi_annotated_subset_2.csv'
test_csv_pseudo = pseudo_dir+'test/csv_densely_semi_annotated_subset_2.csv'

if (args.VARIANT=='I'):
    train_dataset = pd.read_csv(train_csv_pseudo,header=None).values

elif (args.VARIANT=='II'):
    train_dataset = pd.read_csv(train_csv_strong,header=None).values

elif (args.VARIANT=='III'):
    train_dataset_strong = pd.read_csv(train_csv_strong,header=None).values
    train_dataset_pseudo = pd.read_csv(train_csv_pseudo,header=None).values
    train_dataset = np.append(train_dataset_strong,train_dataset_pseudo,axis=0)

elif (args.VARIANT=='baseline'):
    train_dataset = pd.read_csv(train_csv_strong,header=None).values


valid_dataset_strong = pd.read_csv(valid_csv_strong,header=None).values
valid_dataset_pseudo = pd.read_csv(valid_csv_pseudo,header=None).values

test_dataset_strong = pd.read_csv(test_csv_strong,header=None).values
test_dataset_pseudo = pd.read_csv(test_csv_pseudo,header=None).values


if (args.VARIANT=='II'):
    model_to_finetune = models_folder+approach+'/Student_model/student_variant_I/'+'perc_'+str(args.SUBSET)+'/'+'N_EXP_'+str(args.PRE_TRAINED)+'/student_model.pt'
    model = torch.load(model_to_finetune)
else:
    model = StudentModel()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


# Parameters
batch_size = args.BATCH_SIZE
num_workers = 32
params_train = {'batch_size': batch_size,
          #'shuffle': True,
          'sampler': ImbalancedDatasetSampler(train_dataset),
          'num_workers': num_workers}

params_valid = {'batch_size': 200,
          'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(valid_dataset),
          'num_workers': num_workers}

params_test = {'batch_size': 200,
          'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(test_dataset),
          'num_workers': num_workers}

num_epochs = args.EPOCHS

from Data_Generator import Dataset_patches
#load data
training_set = Dataset_patches(train_dataset[:,0], train_dataset[:,1],'train')
training_generator = data.DataLoader(training_set, **params_train)

validation_set = Dataset_patches(valid_dataset_strong[:,0], valid_dataset_strong[:,1],'valid')
validation_generator_strong = data.DataLoader(validation_set, **params_valid)

validation_set = Dataset_patches(valid_dataset_pseudo[:,0], valid_dataset_pseudo[:,1],'valid')
validation_generator_pseudo = data.DataLoader(validation_set, **params_valid)

testing_set = Dataset_patches(test_dataset_strong[:,0], test_dataset_strong[:,1],'test')
testing_generator_strong = data.DataLoader(testing_set, **params_test)

testing_set = Dataset_patches(test_dataset_pseudo[:,0], test_dataset_pseudo[:,1],'test')
testing_generator_pseudo = data.DataLoader(testing_set, **params_test)

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
#for avoiding propagation of fake benign class
#weight[0]=0
samples_weight = torch.from_numpy(weight).type(torch.FloatTensor)

import torch.optim as optim
criterion = torch.nn.CrossEntropyLoss(weight=samples_weight.to(device))
optimizer = optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
model.to(device)


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return metrics.accuracy_score(y_true, y_pred)


# In[32]:


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

best_loss_valid_strong = 100000.0
best_loss_valid_pseudo = 100000.0

best_loss_valid_strong_only = 100000.0
best_loss_valid_pseudo_only = 100000.0

kappa_patches_strong = []

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
        
        #print("epoch "+str(epoch)+ " train loss: " + str(train_loss) + " acc_train: " + str(acc))

        i = i+1
        
    model.eval()

    print("epoch "+str(epoch)+ " train loss: " + str(train_loss) + " acc_train: " + str(acc))
    
    print("evaluating validation")
    print("evaluate strong")
    kappa_valid_strong, valid_loss_strong = evaluate_validation_set(validation_generator_strong)
    print("evaluate pseudo")
    kappa_valid_pseudo, valid_loss_pseudo = evaluate_validation_set(validation_generator_pseudo)
    
        
    print('[%d] valid loss pseudo: %.4f, valid loss strong: %.4f, kappa_valid_pseudo: %.4f, kappa_valid_strong: %.4f' %
          (epoch + 1, valid_loss_pseudo, valid_loss_strong, kappa_valid_pseudo, kappa_valid_strong))         

   
    if (best_loss_valid_strong>valid_loss_strong and best_loss_valid_pseudo>valid_loss_pseudo):
        print ("=> Saving a new best model")
        print("previous loss pseudo: " + str(best_loss_valid_pseudo) + ", new loss function pseudo: " + str(valid_loss_pseudo))
        print("previous loss strong: " + str(best_loss_valid_strong) + ", new loss function strong: " + str(valid_loss_strong))
        best_loss_valid_strong = valid_loss_strong
        best_loss_valid_pseudo = valid_loss_pseudo
        torch.save(model, model_weights)

    if (best_loss_valid_strong_only>valid_loss_strong):
        print ("=> Saving a new best model strong only")
        print("previous loss strong: " + str(best_loss_valid_strong_only) + ", new loss function strong: " + str(valid_loss_strong))
        best_loss_valid_strong_only = valid_loss_strong
        torch.save(model, model_weights_strong)

    if (best_loss_valid_pseudo_only>valid_loss_pseudo):
        print ("=> Saving a new best model pseudo only")
        print("previous loss pseudo: " + str(best_loss_valid_pseudo_only) + ", new loss function strong: " + str(valid_loss_pseudo))
        best_loss_valid_pseudo_only = valid_loss_pseudo
        torch.save(model, model_weights_pseudo)

print('Finished Training')