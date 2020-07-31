from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import collections
import time
from skimage import io
import sys, os, getopt
import pandas as pd
import threading
from skimage import exposure
import argparse

parser = argparse.ArgumentParser(description='TMA patches extractor')
parser.add_argument('-d','--DATASET', type=str, default='train', help='dataset to extract patches (train, valid, test)')
parser.add_argument('-s','--SIZE_P', type=int, default=750, help='size of the tiles to extract (before resize)')
parser.add_argument('-n','--number_patches', type=int, default=30, help='number of patches to extract')
parser.add_argument('-t','--THREADS', type=int, default=10, help='number of threads')
parser.add_argument('-p','--PERCENTAGE', type=float, default=0.6, help='minimum percentage of tissue in a tile')


args = parser.parse_args()

DATASET_NAME = args.DATASET
THREAD_NUMBER = args.THREADS
NUMBER_PATCHES = args.number_patches
PATCH_SIZE = args.SIZE_P
THRESHOLD = args.PERCENTAGE
NEW_PATCH_SIZE = 224
SIZE_TMA = 3100

ALL_TMA_NAME = 'PATH/TMA/IMAGES/'

TMA_output_DIR = 'LOCAL/PATH/../Teacher_Student_models/Strongly_Annotated_patches/'
OUTPUT_DIR = TMA_output_DIR+'/'+DATASET_NAME+'/'

def create_dir(models_path):
    if not os.path.isdir(models_path):
        try:
            os.mkdir(models_path)
        except OSError:
            print ("Creation of the directory %s failed" % models_path)
        else:
            print ("Successfully created the directory %s " % models_path)

create_dir(OUTPUT_DIR)

CSV_IMAGES_DIR = 'LOCAL/PATH/../Teacher_Student_models/csv_files/List_Strongly_Annotated_Images/'

CSV_OUTPUT = 'LOCAL/PATH/../Teacher_Student_models/csv_files/Strongly_Annotated_data/'

filename_images = CSV_IMAGES_DIR+DATASET_NAME+'_images.csv'

MASK_DIRECTORY = 'LOCAL/PATH/../Teacher_Student_models/Images_masks/TMA/'+DATASET_NAME+'/'

csv_data = pd.read_csv(filename_images,header=None).values.flatten()

filenames = []
labels = []

lockTrain = threading.Lock()

def create_csv(nome_patch,labels,fname):
	File = {'filename':nome_patch,'gleason_pattern':labels}
	df = pd.DataFrame(File,columns=['filename','gleason_pattern'])
	#print(file_path)
	np.savetxt(fname, df.values, fmt='%s',delimiter=',')

def check_patch(patch,threshold):
	b = False
	window_size = PATCH_SIZE
	tot_pxl = window_size*window_size
	unique, counts = np.unique(patch, return_counts=True)
	elems = dict(zip(unique, counts))
	i = np.argmax(counts)
	if (unique[i]!=4 and (counts[i]/tot_pxl)>threshold):
		b = True
	
	return b,unique[i]

def generate_glimpses_coords():
	
	x_boundary = SIZE_TMA-PATCH_SIZE-1
	y_boundary = SIZE_TMA-PATCH_SIZE-1

	#generate number
	x=np.random.randint(low=0,high=x_boundary)
	y=np.random.randint(low=0,high=y_boundary)
	x1 = x+PATCH_SIZE
	y1 = y+PATCH_SIZE

	return x,y,x1,y1

def explore_list(list_files):

	for f in list_files:

		prefix = f[:4]

		filename = ALL_TMA_NAME+f
		print(filename)
			#open TMA image
		tma = Image.open(filename)
		tma_array = np.asarray(tma)

		file_name = f[:-4]

			#find mask directory
		if (os.path.isfile(MASK_DIRECTORY+'mask_'+file_name+'.png')):
			filename_mask = MASK_DIRECTORY+'mask_'+file_name+'.png'
		elif (os.path.isfile(MASK_DIRECTORY+'mask1_'+file_name+'.png')):
			filename_mask = MASK_DIRECTORY+'mask1_'+file_name+'.png'
		elif (os.path.isfile(MASK_DIRECTORY+'mask2_'+file_name+'.png')):
			filename_mask = MASK_DIRECTORY+'mask2_'+file_name+'.png'

			#open mask file
		tma_mask = Image.open(filename_mask)
		tma_array_mask = np.asarray(tma_mask)

		i = 0
		cont = 0
		CONT_MAX = 10000

		start_time = time.time()
		#print("analyzing " + str(filename))
		while(i<NUMBER_PATCHES and cont<CONT_MAX):

			x_coords, y_coords, x1_coords, y1_coords = generate_glimpses_coords()
			mask_patch = tma_array_mask[y_coords:y1_coords,x_coords:x1_coords]
			flag, label = check_patch(mask_patch,THRESHOLD)

			if(flag):
				fname_patch = OUTPUT_DIR+file_name+'_'+str(i)+'.jpg'

				tma_patch = tma_array[y_coords:y1_coords,x_coords:x1_coords,:]

				new_im = Image.fromarray(tma_patch)
				#y_up, x_up, y_down, x_down = random_crop(new_im)
				#new_im = new_im.crop((y_up, x_up, y_down, x_down))
				new_im = new_im.resize((NEW_PATCH_SIZE,NEW_PATCH_SIZE))
				new_im = np.asarray(new_im)

				if (exposure.is_low_contrast(new_im)==False):
					io.imsave(fname_patch, new_im)
						
					lockTrain.acquire()
					filenames.append(fname_patch)
					labels.append(label)
					lockTrain.release()

				i = i+1
				#cont = 0
			else:
				cont = cont+1


		elapsed_time = time.time() - start_time
		print("done in " + str(elapsed_time))


def chunker_list(seq, size):
    return (seq[i::size] for i in range(size))

lists_files = list(chunker_list(csv_data,THREAD_NUMBER))
print("lists_files " + str(type(lists_files)))

threads = []
for i in range(THREAD_NUMBER):
	t = threading.Thread(target=explore_list,args=[lists_files[i]])
	threads.append(t)

for t in threads:
	t.start()
for t in threads:
	t.join()

print("DONE")

create_csv(filenames,labels,CSV_OUTPUT+DATASET_NAME+'_patches.csv')
