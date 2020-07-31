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
from Data_Generator import Dataset_patches
import argparse

print( torch.cuda.current_device())
print( torch.cuda.device_count())

parser = argparse.ArgumentParser(description='TMA patches extractor')
parser.add_argument('-d','--DATASET', type=str, default='strong', help='dataset to use: pseudo or strong')
parser.add_argument('-v','--VARIANT', type=str, default='train', help='student training variant to use (I,II,III,baseline)')
parser.add_argument('-a','--APPROACH', type=str, default='ssl', help='teacher/student approach: ssl (semi-supervised), swsl (semi-weakly supervised)')
parser.add_argument('-s','--SUBSET', type=int, default='19', help='subset of pseudo-labels to use 19=1000, 0=20000 pseudo labels per class')
parser.add_argument('-n','--N_EXP', type=int, default=0, help='number experiment')
parser.add_argument('-b','--BATCH_SIZE', type=int, default=32, help='batch size')

args = parser.parse_args()

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

strong_dir = 'LOCAL/PATH/../Teacher_Student_models/csv_files/Strongly_Annotated_data/'
pseudo_dir = 'LOCAL/PATH/../Teacher_Student_models/csv_files/Pseudo_Labeled_Data/'+approach+'/'

if(args.DATASET=='strong'):
    test_csv = strong_dir+'test_patches.csv'
elif (args.DATASET=='pseudo'):
    test_csv = pseudo_dir+'test/csv_densely_semi_annotated_subset_2.csv'

test_dataset = pd.read_csv(test_csv,header=None).values


model = torch.load(model_weights)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

params_test = {'batch_size': args.BATCH_SIZE,
          'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(test_dataset),
          'num_workers': 32}


testing_set = Dataset_patches(test_dataset[:,0], test_dataset[:,1],'test')
testing_generator_strong = data.DataLoader(testing_set, **params_test)

if (torch.cuda.device_count() > 1):
    model = torch.nn.DataParallel(model)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

#SAVE LOSS FUNCTION DATA
if (args.DATASET=='strong'):
    kappa_score_general_filename = checkpoint_path+'kappa_score_strong.csv'
elif(args.DATASET=='pseudo'):
    kappa_score_general_filename = checkpoint_path+'kappa_score_pseudo.csv'


print("TESTING")


y_pred = []
y_true = []

model.eval()

with torch.no_grad():
    j = 0
    for inputs,labels in testing_generator_strong:
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        #optimizer.zero_grad()

        # forward + backward + optimize
        try:
            outputs = model(inputs)
        except:
            outputs = model.module(inputs)
        outputs = F.softmax(outputs)
        #loss = criterion(outputs, labels)
        #loss.backward()
        #optimizer.step()

        #accumulate values
        outputs_np = outputs.cpu().data.numpy()
        labels_np = labels.cpu().data.numpy()
        outputs_np = np.argmax(outputs_np, axis=1)
        y_pred = np.append(y_pred,outputs_np)
        y_true = np.append(y_true,labels_np)

#k-score
k_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
print("k_score " + str(k_score))
#confusion matrix
confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
print("confusion_matrix ")
print(str(confusion_matrix))
#confusion matrix normalized
np.set_printoptions(precision=2)
cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(str(cm_normalized))

kappas = [k_score]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_general_filename, df.values, fmt='%s',delimiter=',')