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
parser.add_argument('-d','--DATASET', type=str, default='strong', help='dataset to test: weak, strong')
parser.add_argument('-a','--APPROACH', type=str, default='ssl', help='teacher/student approach: ssl (semi-supervised), swsl (semi-weakly supervised)')
parser.add_argument('-n','--N_EXP', type=int, default=0, help='number experiment to test')
parser.add_argument('-b','--BATCH_SIZE', type=int, default=32, help='batch size')
parser.add_argument('-t','--THRESHOLD', type=int, default=500, help='patches to select')

args = parser.parse_args()

THRESHOLD = args.THRESHOLD

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
  
elif(args.DATASET=='weak'):
    csv_folder = 'LOCAL/PATH/../Teacher_Student_models/csv_files/Weakly_Annotated_Data/'
    model_labels_approach = 'weak_labels_training' 

models_folder = 'LOCAL/PATH/../Teacher_Student_models/models_weights/'

models_path = models_folder+approach+'/'
create_dir(models_path)

models_path = models_path+'Teacher_model/'
create_dir(models_path)

models_path = models_path+model_labels_approach+'/'
create_dir(models_path)

models_path = models_path+'N_EXP_'+str(args.N_EXP)+'/'
create_dir(models_path) 

checkpoint_path = models_path+'checkpoints/'
create_dir(checkpoint_path)

model_weights = models_path+'teacher_model.pt'

model = torch.load(model_weights)

test_csv = csv_folder+'test_patches.csv'

test_dataset = pd.read_csv(test_csv,header=None).values


from torchvision import transforms
#DATA AUGMENTATION
prob = 0.5
pipeline_transform = A.Compose([
    A.VerticalFlip(p=prob),
    A.HorizontalFlip(p=prob),
    A.RandomRotate90(p=prob),
    A.ElasticTransform(alpha=0.1,p=prob),
    A.HueSaturationValue(hue_shift_limit=(-9),sat_shift_limit=25,val_shift_limit=10,p=prob),
    ])

#DATA NORMALIZATION
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

batch_size_test = args.BATCH_SIZE

params_test = {'batch_size': batch_size_test,
          'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(test_dataset),
          'num_workers': 32}

#CREATE GENERATORS
from Data_Generator import Dataset_patches
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

model.to(device)

print("testing on TMA")

y_pred = []
y_true = []

model.eval()


with torch.no_grad():
    j = 0
    for inputs,labels in testing_generator:
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


#SAVE LOSS FUNCTION DATA
kappa_score_filename = checkpoint_path+'kappa_score.csv'

#kappa_score_GSs_filename
File = {'val':[k_score]}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_filename, df.values, fmt='%s',delimiter=',')
