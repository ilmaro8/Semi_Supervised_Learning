import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='TMA patches extractor')
parser.add_argument('-d','--DATASET', type=str, default='train', help='dataset to extract patches (train, valid, test)')
parser.add_argument('-t','--THRESHOLD', type=int, default='500', help='amount of patches to select')

args = parser.parse_args()

DATASET = args.DATASET

file_dir = 'LOCAL/PATH/../Teacher_Student_models/WSI_patches/'+DATASET+'_densely'

general_csv_filename = file_dir+'csv_'+DATASET+'_densely.csv'

csv_general = pd.read_csv(general_csv_filename,header=None).values

filenames_WSI = csv_general[:,0]
labels_WSI = csv_general[:,1]

filenames = []
labels = []

THRESHOLD = args.THRESHOLD

for i in range(len(filenames_WSI)):
    d = os.path.split(filenames_WSI[i])[1]
    directory = file_dir+d
    csv_file = directory+'/'+d+'_densely_sorted_br_patches.csv'

    local_csv = pd.read_csv(csv_file,header=None).values
    
    for j in range(len(local_csv[:THRESHOLD])):
        filenames.append(local_csv[j,0])
        labels.append(labels_WSI[i])
    
    for j in range(10):
        filenames.append(local_csv[-j,0])
        labels.append(0)

filename_output = 'LOCAL/PATH/../Teacher_Student_models/csv_files/Weakly_Annotated_Data/'+args.DATASET+'_patches.csv'

File = {'filename':filenames,'labels':labels}
df = pd.DataFrame(File,columns=['filename','labels'])
np.savetxt(filename_output, df.values, fmt='%s',delimiter=',')


