import sys, os, getopt
from PIL import Image
import numpy as np
import pandas as pd 
from collections import Counter
from matplotlib import pyplot as plt
from skimage import io
from PIL import Image
import threading
import time
import collections
from torchvision import transforms
import torch
from torch.utils import data
import torch.nn.functional as F
sys.path.append('../utils/')
from Models import TeacherModel
import argparse

parser = argparse.ArgumentParser(description='TMA patches extractor')
parser.add_argument('-d','--DATASET', type=str, default='train', help='dataset to annotate patches (train, valid, test)')
parser.add_argument('-n','--N_EXP', type=int, default=0, help='number experiment to use for annotating data')
parser.add_argument('-a','--APPROACH', type=str, default='ssl', help='teacher/student approach: ssl (semi-supervised), swsl (semi-weakly supervised)')

args = parser.parse_args()


patch_dir = 'LOCAL/PATH/../Teacher_Student_models/WSI_patches/'+args.DATASET+'_densely/'
csv_filename = patch_dir+'csv_'+args.DATASET+'_densely.csv'

models_dir = 'LOCAL/PATH/../Teacher_Student_models/models_weights/'

csv_data = pd.read_csv(csv_filename,header=None).values

if (args.APPROACH=='ssl'):
	models_weights = models_dir+'Semi_Supervised/Teacher_model/strong_labels_training/N_EXP_'+str(args.N_EXP)+'/teacher_model.pt'
	suffix = '_densely_probs_semi.csv'

elif (args.APPROACH=='swsl'):
	models_weights = models_dir+'Semi_Weakly_Supervised/Teacher_model/strong_labels_training/N_EXP_'+str(args.N_EXP)+'/teacher_model.pt'
	suffix = '_densely_probs_semi_weakly.csv'


#DATA NORMALIZATION
preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Dataset_WSI(data.Dataset):

	def __init__(self, list_IDs, pgps, sgps, filename):

		self.filename = filename
		self.list_IDs = list_IDs
		self.pgps = pgps
		self.sgps = sgps
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):

		# Select sample
		ID = self.list_IDs[index]

		# Load data and get label
		X = Image.open(ID)
		X = np.asarray(X)

		pgp = self.pgps[index]
		sgp = self.sgps[index]

		#data transformation
		input_tensor = preprocess(X)
				
		return input_tensor, pgp, sgp, ID, index


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Parameters
batch_size = 500
params_gen = {'batch_size': batch_size,
		  #'shuffle': True,
		  #'sampler': ImbalancedDatasetSampler(train_dataset),
		  'num_workers': 32}

model = torch.load(models_weights)
model.eval()

model.to(device)  

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

def find_csv_patches(line_csv):
	dir_name = line_csv[0]
	pgp = line_csv[1]
	sgp = line_csv[2]


	name_dir = os.path.normpath(dir_name).split(os.sep)[-1]
	local_csv_path = patch_dir+name_dir+'/'+name_dir+'_paths_densely.csv'
	csv_local = pd.read_csv(local_csv_path,header=None).values[:,0]
	return csv_local, pgp, sgp, name_dir

def evaluation_loop(generator,filename_output):

	#print("analyzing " + str(filename_output))

	filenames = []
	p_b = []
	p_gp3 = []
	p_gp4 = []
	p_gp5 = []
	pgps = []
	sgps = []
	idxs = []

	with torch.no_grad():
		i = 0
		start_time = time.time()
		for inputs, pgp, sgp, filename, idx in generator:
			#print("analyzed " + str(i*batch_size) + " files")
			inputs = inputs.to(device)
			# zero the parameter gradients
			#optimizer.zero_grad()

			# forward + backward + optimize
			try:
				outputs = model(inputs)
			except:
				outputs = model.module(inputs)
			outputs = F.softmax(outputs)

			#accumulate values
			outputs_np = outputs.cpu().data.numpy()
			p_b = np.append(p_b,outputs_np[:,0])
			p_gp3 = np.append(p_gp3,outputs_np[:,1])
			p_gp4 = np.append(p_gp4,outputs_np[:,2])
			p_gp5 = np.append(p_gp5,outputs_np[:,3])
			filenames = np.append(filenames,filename)
			pgps = np.append(pgps,pgp)
			sgps = np.append(sgps,sgp)
			idxs = np.append(idxs,idx)

		elapsed_time = time.time() - start_time
		#print("elapsed_time " + str(elapsed_time))

	#create new_csv
	File = {'filename':filenames,'primary_GP':pgps.astype(int),'secondary_GP':sgps.astype(int),'p_b':p_b,'p_gp3':p_gp3,'p_gp4':p_gp4,'p_gp5':p_gp5}
	df = pd.DataFrame(File,columns=['filename','primary_GP','secondary_GP','p_b','p_gp3','p_gp4','p_gp5'])
	
	#print(file_path)
	np.savetxt(filename_output, df.values, fmt='%s',delimiter=',')

def analyze_csv(csv_file):

	for l in csv_file:
		csv_local, pgp, sgp, filename_csv = find_csv_patches(l)
		pgp_array = np.full(len(csv_local),pgp)
		sgp_array = np.full(len(csv_local),sgp)
		
		#print(filename_csv)

		#create data generator
		dataset = Dataset_WSI(csv_local, pgp_array, sgp_array,filename_csv)
		generator = data.DataLoader(dataset, **params_gen)
		
		filename_output = patch_dir+filename_csv+'/'+filename_csv+suffix
		evaluation_loop(generator,filename_output)

analyze_csv(csv_data)