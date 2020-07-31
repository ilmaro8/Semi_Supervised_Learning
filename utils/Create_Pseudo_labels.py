import numpy as np
import pandas as pd
import sys, os, getopt
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='TMA patches extractor')
parser.add_argument('-a','--APPROACH', type=str, default='ssl', help='teacher/student approach: ssl (semi-supervised), swsl (semi-weakly supervised)')
parser.add_argument('-d','--DATASET', type=str, default='train', help='dataset to extract patches (train, valid, test)')

args = parser.parse_args()

patch_dir = 'LOCAL/PATH/../Teacher_Student_models/Weakly_Annotated_patches/'+args.DATASET+'_densely/'
csv_filename = patch_dir+'csv_'+args.DATASET+'_densely.csv'

csv_data = pd.read_csv(csv_filename,header=None).values

filenames = []
gps = []
dicts = []

if (args.APPROACH=='ssl'):
    approach = 'Semi_Supervised'
elif (args.APPROACH=='swsl'):
    approach = 'Semi_Weakly_Supervised'

def create_dict(csv_local):
    #br_dict = []
    for i in range(len(csv_local)):
        br={'filename':csv_local[i,0],'pgp':csv_local[i,1],'sgp':csv_local[i,2],'p_b':csv_local[i,3],'p_gp3':csv_local[i,4],'p_gp4':csv_local[i,5],'p_gp5':csv_local[i,6]}
        dicts.append(br)
    #return br_dict

def find_csv_patches(line_csv):
    dir_name = line_csv[0]
    pgp = line_csv[1]
    sgp = line_csv[2]
    
    name_dir = os.path.normpath(dir_name).split(os.sep)[-1]
    #local_csv_path = patch_dir+name_dir+'/'+name_dir+'_'+NUM_PATCHES_str+'_probs.csv'
    local_csv_path = patch_dir+name_dir+'/'+name_dir+'_densely_probs.csv'
    csv_local = pd.read_csv(local_csv_path,header=None).values
    return csv_local, pgp, sgp


def get_key(x):
    if (x==1):
        k = 'p_gp3'
    elif (x==2):
        k = 'p_gp4'
    elif (x==3):
        k = 'p_gp5'
    else:
        k = 'p_b'
    return k

def sort_dict(array_dict, pattern, threshold):
    x = get_key(pattern)
    new_array = np.array(sorted(array_dict, key=lambda k: k[x],reverse=True))
    p_max = 1.0
    i = 0
    while (i<len(new_array) and p_max>threshold):
        p_max = new_array[i][x]
        
        filenames.append(new_array[i]['filename'])
        gps.append(pattern)
        
        i = i+1
    #print("pattern: " + str(pattern))
    #plot_images(new_array[:10])


def sort_dict(array_dict, pattern):
    x = get_key(pattern)
    new_array = np.array(sorted(array_dict, key=lambda k: k[x],reverse=True))
    return new_array


def analyze_csv(csv_file):
    for l in csv_file:
        #print(l)
        try:
            csv_local, pgp, sgp = find_csv_patches(l)
            create_dict(csv_local)
            #extract_patches(array_dict, pgp, sgp)
            #dicts.append(array_dict)
        except(FileNotFoundError, IOError):
            #print("Wrong file or file path")
            pass

analyze_csv(csv_data)

dicts = np.asarray(dicts).flatten()
print(dicts.shape)

def lower_bound(dicts,pattern):
    i = 0
    x = get_key(pattern)
    #print(x)
    threshold = 0.5
    print(x)
    sorted_dicts = np.array(sorted(dicts, key=lambda k: k[x],reverse=True))
    while(sorted_dicts[i][x]>threshold and i<len(sorted_dicts)):
        i = i+1
    return i


# In[17]:


def sort_and_extract(pattern,amount):
    x = get_key(pattern)
    new_array = np.array(sorted(dicts, key=lambda k: k[x],reverse=True))
    i = 0
    while (i<amount):
        
        filenames.append(new_array[i]['filename'])
        gps.append(pattern)
        
        i = i+1

# In[18]:

#SORTED
#print(dicts)
MAX_AMOUNT_BENIGN = lower_bound(dicts,0)
MAX_AMOUNT_GP3 = lower_bound(dicts,1)
MAX_AMOUNT_GP4 = lower_bound(dicts,2)
MAX_AMOUNT_GP5 = lower_bound(dicts,3)

print(MAX_AMOUNT_BENIGN,MAX_AMOUNT_GP3,MAX_AMOUNT_GP4,MAX_AMOUNT_GP5)

sort_and_extract(0,MAX_AMOUNT_BENIGN)
sort_and_extract(1,MAX_AMOUNT_GP3)
sort_and_extract(2,MAX_AMOUNT_GP4)
sort_and_extract(3,MAX_AMOUNT_GP5)


unique, counts = np.unique(gps, return_counts=True)
print(dict(zip(unique, counts)))

#save file without probabilities
new_csv_filename = patch_dir+'csv_'+args.DATASET+'_densely_semi_annotated_fixed.csv'

File = {'filename':filenames,'gleason_pattern':gps}
df = pd.DataFrame(File,columns=['filename','gleason_pattern'])
np.savetxt(new_csv_filename, df.values, fmt='%s',delimiter=',')

print("NEW CSV SAVED")

print("CREATE SUBSETS")

# CREATE SUBSETS
csv_data = pd.read_csv(new_csv_filename,header=None).values

labels = [0,1,2,3]
filename_label = []

for l in labels:
    f_labels = []
    for j in range(len(csv_data)):
        if (csv_data[j,1]==l):
            f_labels.append(csv_data[j,0])
    filename_label.append(f_labels)

def create_dir(models_path):
    if not os.path.isdir(models_path):
        try:
            os.mkdir(models_path)
        except OSError:
            print ("Creation of the directory %s failed" % models_path)
        else:
            print ("Successfully created the directory %s " % models_path) 

def save_csv(new_filenames,new_labels,pow_two):

    csv_dir = 'LOCAL/PATH/../Teacher_Student_models/csv_files/Pseudo_Labeled_Data/'+approach+'/'+args.DATASET+'/'
    create_dir(patch_dir)

    #new_csv_filename = patch_dir+'csv_'+NUM_PATCHES_str+'_semi_annotated_subset_'+str(pow_two)+'.csv'
    new_csv_filename = csv_dir+'csv_densely_semi_annotated_subset_'+str(pow_two)+'.csv'
    
    File = {'filename':new_filenames,'gleason_pattern':new_labels}
    df = pd.DataFrame(File,columns=['filename','gleason_pattern'])
    np.savetxt(new_csv_filename, df.values, fmt='%s',delimiter=',')

def sort_both(seed):
    s = np.arange(seed)
    np.random.shuffle(s)
    return s

def divide_and_save(filename_label,pow_two,max_amount):
    new_filenames = []
    new_labels = []
    num_samples = max_amount
    for l in labels:
        #num_samples = int(max/pow(2,pow_two))
        i = 0
        while(i<num_samples and i<len(filename_label[l])):
        #for i in range(num_samples):
            new_labels.append(l)
            new_filenames.append(filename_label[l][i])
            i = i+1
       
        
    new_filenames = np.asarray(new_filenames).flatten()
    new_labels = np.asarray(new_labels)
    save_csv(new_filenames,new_labels,pow_two)

if (args.DATASET=='train'):
    selected_samples = 20000
elif (args.DATASET=='valid'):
    selected_samples = 5000
elif (args.DATASET=='test'):
    selected_samples = 4000

#split in subsets random
if (args.DATASET=='train'):
    num_subsets = 20
else:
    num_subsets = 5

#split in subsets sorted
for i in range(num_subsets):
    divide_and_save(filename_label,i,selected_samples)
    selected_samples = selected_samples-1000

s = sort_both(len(csv_data))
csv_data = csv_data[s]

labels = [0,1,2,3]
filename_label = []

for l in labels:
    f_labels = []
    for j in range(len(csv_data)):
        if (csv_data[j,1]==l):
            f_labels.append(csv_data[j,0])
    filename_label.append(f_labels)

