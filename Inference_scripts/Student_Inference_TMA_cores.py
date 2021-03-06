from PIL import Image
from torchvision import transforms
import torch
from torch.utils import data
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import json
import copy
from sklearn import metrics
import sys, getopt
import os
import glob
import random
import collections
import time
from tqdm import tqdm
import torch.nn as nn
sys.path.append('../utils/')
from Models import StudentModel
from Data_Generator import Dataset_WSI
import argparse

import argparse

print( torch.cuda.current_device())
print( torch.cuda.device_count())

parser = argparse.ArgumentParser(description='TMA patches extractor')
parser.add_argument('-v','--VARIANT', type=str, default='train', help='student training variant to use (I,II,III,baseline)')
parser.add_argument('-a','--APPROACH', type=str, default='ssl', help='teacher/student approach: ssl (semi-supervised), swsl (semi-weakly supervised)')
parser.add_argument('-s','--SUBSET', type=int, default='19', help='subset of pseudo-labels to use 19=1000, 0=20000 pseudo labels per class')
parser.add_argument('-n','--N_EXP', type=int, default=0, help='number experiment')
parser.add_argument('-b','--BATCH_SIZE', type=int, default=32, help='batch size')
parser.add_argument('-t','--THRESHOLD', type=int, default=500, help='patches to select')

args = parser.parse_args()


THRESHOLD = args.THRESHOLD

MEDICAL_SOURCE = 'TCGA'
#MEDICAL_SOURCE = 'panda'

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

models_folder = 'LOCAL/PATH/../Teacher_Student_models/models_weights/'
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

model = torch.load(model_weights)

#load testing data
test_dir = 'LOCAL/PATH/../Teacher_Student_models/WSI_patches/test_densely/'

csv_test = test_dir+'csv_test_densely.csv'
data_test = pd.read_csv(csv_test,header=None).values

data_test_paths = data_test[:,0]
data_test_labels = data_test[:,1:]
data_test_labels = data_test_labels.astype('int64')
print(data_test.shape)
print(data_test_paths.shape)
print(data_test_labels.shape)

array_probs = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


#DATA NORMALIZATION
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#model.eval()
model.to(device)

def create_dir(directory):
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except OSError:
            print ("Creation of the directory %s failed" % directory)


batch_size = args.BATCH_SIZE
num_workers = 2
params_test = {'batch_size': batch_size,
          #'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(test_dataset),
          'num_workers': num_workers}

def find_first_two(array):
    x = np.copy(array)
    max_1 = np.argmax(x)
    max_val1 = x[max_1]
    x[max_1]=-1
    max_2 = np.argmax(x)
    max_val2 = x[max_2]
    """
    if(max_1==0 or max_2==0):
        max_1 = max(max_1,max_2)
        max_2 = max(max_1,max_2)
    """
    if (max_val1>(2*max_val2)):
        max_2 = max_1
    
    return max_1,max_2

def assign_group(a, b, survival_groups=False):
    # if both cancer and benign tissue are predicted
    # ignore benign tissue for reporting, as pathologists do
    if (a > 0) and (b == 0):
        b = a
    if (b > 0) and (a == 0):
        a = b

    if not survival_groups:
        return a + b
    else:
        # get the actual Gleason pattern (range 3-5)
        a += 2
        b += 2
        if a+b <= 6:
            return 1
        elif a+b == 7:
            return 2
        else:
            return 3
    
def gleason_summary_wsum(y_pred, survival_groups=False, thres=None):
    gleason_scores = y_pred.copy()
    gleason_scores /= np.sum(gleason_scores)
    # remove outlier predictions
    if thres is not None:
        gleason_scores[gleason_scores < thres] = 0
    # and assign overall grade
    idx = np.argsort(gleason_scores)[::-1]
    primary_class = idx[0]
    secondary_class = idx[1] if gleason_scores[idx[1]] > 0 else idx[0]
    return primary_class, secondary_class
    #return assign_group(primary_class, secondary_class, survival_groups)

def majority_voting(array):
    majority = [0,0,0,0]
    for i in range(array.shape[0]):
        #print(prob)
        idx = np.argmax(array[i])
        majority[idx] = majority[idx]+1
    #majority[0]=0
    pgp, sgp = find_first_two(majority)
    return pgp, sgp, majority


def load_and_evaluate(list_f,elems,directory_histograms):
    testing_set = Dataset_WSI(list_f)
    testing_generator = data.DataLoader(testing_set, **params_test)
    array_probs = []

    local_filenames = []

    with torch.no_grad():
        j = 0
        for inputs, filenames in testing_generator:
            inputs = inputs.to(device)
            # zero the parameter gradients
            #optimizer.zero_grad()

            # forward + backward + optimize
            try:
                outputs = model(inputs)
            except:
                outputs = model.module(inputs)
            probs = F.softmax(outputs)
            #print(probs)

            #accumulate values
            probs = probs.cpu().data.numpy()
            array_probs = np.append(array_probs,probs)
            local_filenames = np.append(local_filenames,filenames)

    array_probs = np.reshape(array_probs,(elems,4))
    #array_probs = np.squeeze(array_probs)
 
    #majority voting
    pgp,sgp, histogram = majority_voting(array_probs)
    y_preds.append([pgp,sgp])

    #add pgp,sgp to y_preds


def gleason_score(primary,secondary):
    
    array = []
    """
    for i in range(len(primary)):
        a = primary[i]
        b = secondary[i]
        
        if (a > 0) and (b == 0):
            b = a
        if (b > 0) and (a == 0):
            a = b

        sum_gs = a+b
        
        if (a==0 and b==0):
            gs = 0
        elif (a==1 and b==1):
            gs = 1
        elif (a==1 and b==2):
            gs = 2
        elif (a==2 and b==1):
            gs = 3
        elif (sum_gs==4):
            gs = 4
        elif (sum_gs>4):
            gs = 5

        array.append(gs)
    """  

    for i in range(len(primary)):
        a = primary[i]
        b = secondary[i]
        
        if (a > 0) and (b == 0):
            b = a
        if (b > 0) and (a == 0):
            a = b

        gs = a+b
        array.append(gs)

    return array

def predict_metrics(y_pred,y_true,metric):
    if(metric=='primary'):
        #primary gleason pattern
        y_true = y_true[:,0]
        y_pred = y_pred[:,0]
        k_score_primary = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
        
        print("k_score_primary " + str(k_score_primary))
        
        return k_score_primary
        
    elif(metric=='secondary'):
        #secondary gleason pattern
        y_true = y_true[:,1]
        y_pred = y_pred[:,1]
        k_score_secondary = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
        
        print("k_score_secondary " + str(k_score_secondary))
        
        return k_score_secondary
        
    else:
        #gleason score
        #y_true = y_true[:,0]+y_true[:,1]
        #y_pred = y_pred[:,0]+y_pred[:,1]

        y_true = gleason_score(y_true[:,0],y_true[:,1])
        y_pred = gleason_score(y_pred[:,0],y_pred[:,1])
        
        #print(y_pred)
        k_score_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
                
        print("k_score_score " + str(k_score_score))
        
        return k_score_score

y_preds = []

filenames_array = []
histo_preds = []
histo_true = []

for p in data_test_paths:
    d = os.path.split(p)[1]
    directory = test_dir+d
    #csv_file = directory+'/'+d+'_densely.csv'
    csv_file = directory+'/'+d+'_densely_sorted_br_patches.csv'

    directory_histograms = directory+'/histograms/'

    create_dir(directory_histograms)

    local_csv = pd.read_csv(csv_file,header=None).values[:THRESHOLD,0]

    load_and_evaluate(local_csv,len(local_csv),directory_histograms)

    filenames_array.append(d)

#METRICS

y_preds = np.array(y_preds)
y_true = data_test_labels

y_preds = y_preds.astype('int64')
"""
for i in range(len(y_preds)):
    print(y_preds[i],y_true[i])    
"""

kappa_score_primary = predict_metrics(y_preds,y_true,'primary')
kappa_score_secondary = predict_metrics(y_preds,y_true,'secondary')
kappa_score_score = predict_metrics(y_preds,y_true,'score')


kappa_score_best_PGP_filename = checkpoint_path+'kappa_score_PGP.csv'
kappa_score_best_SGP_filename = checkpoint_path+'kappa_score_SGP.csv'
kappa_score_best_GS_filename = checkpoint_path+'kappa_score_GS.csv'

kappas = [kappa_score_primary]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_best_PGP_filename, df.values, fmt='%s',delimiter=',')

kappas = [kappa_score_secondary]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_best_SGP_filename, df.values, fmt='%s',delimiter=',')

kappas = [kappa_score_score]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_best_GS_filename, df.values, fmt='%s',delimiter=',')