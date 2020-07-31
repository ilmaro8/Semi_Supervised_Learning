import sys, os, getopt
import openslide
from PIL import Image
import numpy as np
import pandas as pd 
from skimage import io
import threading
import time
import albumentations as A
import time
from torchvision import transforms
from skimage import exposure
import json
import cv2
import argparse

np.random.seed(0)

parser = argparse.ArgumentParser(description='WSI densely patches extractor')
parser.add_argument('-d','--DATASET', type=str, default='train', help='dataset to extract patches (train, valid, test)')
parser.add_argument('-s','--SIZE_P', type=int, default=750, help='size of the tiles')
parser.add_argument('-t','--THREADS', type=int, default=10, help='number of threads')
parser.add_argument('-p','--PERCENTAGE', type=float, default=0.6, help='minimum percentage of tissue in a tile')

args = parser.parse_args()

def create_dir(models_path):
	if not os.path.isdir(models_path):
		try:
			os.mkdir(models_path)
		except OSError:
			print ("Creation of the directory %s failed" % models_path)
		else:
			print ("Successfully created the directory %s " % models_path)

LIST_FILE = 'LOCAL/PATH/../Teacher_Student_models/csv_files/Weakly_Annotated_Data/'+args.DATASET+'_WSI.csv' 

PATH_INPUT_MASKS = 'LOCAL/PATH/../Teacher_Student_models/Images_masks/WSI/'+args.DATASET+'/'

PATH_OUTPUT = 'LOCAL/PATH/../Teacher_Student_models/Weakly_Annotated_patches/'
create_dir(PATH_OUTPUT)
PATH_OUTPUT = PATH_OUTPUT + args.DATASET+'_densely/'

GENERAL_TXT_PATH = PATH_OUTPUT+'csv_'+args.DATASET+'_densely.csv'

THREAD_NUMBER = args.THREADS
lockList = threading.Lock()
lockGeneralFile = threading.Lock() 

def create_output_imgs(img,fname):
	#save file
	new_patch_size = 224
	img = img.resize((new_patch_size,new_patch_size))
	img = np.asarray(img)
	#io.imsave(fname, img)
	print("file " + str(fname) + " saved")

def read_file_csv(fname):
	df = pd.read_csv(fname, sep=',', header=None)
	print("list data")
	#return df['filename'],df['primary_GG'],df['secondary_GG']
	return df[0].values,df[1].values,df[2].values


def write_general_csv(fname,arrays):
	File = {'filename':arrays[0],'primary_GG':arrays[1],'secondary_GG':arrays[2]}
	df = pd.DataFrame(File,columns=['filename','primary_GG','secondary_GG'])
	#print(file_path)
	np.savetxt(fname, df.values, fmt='%s',delimiter=',')

def check_background(glimpse,threshold,GLIMPSE_SIZE_SELECTED_LEVEL,MAGNIFICATION_RATIO):
	b = False

	window_size = int(GLIMPSE_SIZE_SELECTED_LEVEL/MAGNIFICATION_RATIO)
	tot_pxl = window_size*window_size
	white_pxl = np.count_nonzero(glimpse)
	score = white_pxl/tot_pxl
	if (score>=threshold):
		b=True
	return b

def write_coords_local_file(fname,arrays):
		#select path
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	filename_path = output_dir+fname+'_coords_densely.csv'
		#create file
	File = {'filename':arrays[0],'level':arrays[1],'x_top':arrays[2],'y_top':arrays[3],'factor':arrays[4]}
	df = pd.DataFrame(File,columns=['filename','level','x_top','y_top','factor'])
		#save file
	np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

def write_paths_local_file(fname,listnames):
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	filename_path = output_dir+fname+'_paths_densely.csv'
		#create file
	File = {'filenames':listnames}
	df = pd.DataFrame(File,columns=['filenames'])
		#save file
	np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

def write_paths_br_file(fname,listnames,brs):
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	filename_path = output_dir+fname+'_densely_sorted_br_patches.csv'

		#sort by BR
	zipped_lists = zip(brs, listnames)
	sorted_pairs = sorted(zipped_lists,reverse=True)

	tuples = zip(*sorted_pairs)
	brs, listnames = [ list(tuple) for tuple in tuples]

		#create file
	File = {'filenames':listnames,'br':brs}
	df = pd.DataFrame(File,columns=['filenames','br'])
		#save file
	np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

def multi_one_hot_enc(current_labels):
	labels = [0,0,0,0,0]
	for i in range(len(current_labels)):
		labels[current_labels[i]]=1
	return labels

def whitish_img(img):
	THRESHOLD_WHITE = 200
	b = True
	if (np.mean(img) > THRESHOLD_WHITE):
		b = False
	return b

def find_central_and_analyze(wsi_np):
	pixel_stride = 50
	THRESHOLD = 0.5
	b = False
	
	half = int(wsi_np.shape[1]/2)
	h1 = half-pixel_stride
	h2 = half+pixel_stride
	
	central_section = wsi_np[:,h1:h2]
	
	tot_surface = 2*pixel_stride*wsi_np.shape[0]
	
	unique, counts = np.unique(central_section, return_counts=True)
	
	if (counts[0]==tot_surface):
		b=True
	elif (counts[1]<THRESHOLD*tot_surface):
		b=True
	return b

def left_or_right(img):
	half = int(img.shape[1]/2)
	left_img = img[:,:half]

	right_img = img[:,half:]

	unique, counts_left = np.unique(left_img, return_counts=True)
	unique, counts_right = np.unique(right_img, return_counts=True)


	b = None

	if (len(counts_left)<len(counts_right)):
		b = 'right'
	elif(len(counts_left)>len(counts_right)):
		b = 'left'
	else:
		if (counts_left[-1]>counts_right[-1]):
			b = 'left'
		else:
			b = 'right'

	return b

def BR(img):
	np_img = np.asarray(img)
	np_img.shape
	
	r = np_img[:,:,0].astype('uint16')
	g = np_img[:,:,1].astype('uint16')
	b = np_img[:,:,2].astype('uint16')
	
	br_mat = 100.*b/(1+g+r)*(256./(1+g+r+b))
	
	#equalisation and median
	br_mat = br_mat.astype('uint16')
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	br_val = clahe.apply(br_mat)
	
	br = np.median(br_val)
	
	return br

#estrae glimpse e salva metadati relativi al glimpse
def analyze_file(filename,gleason_score,patch_score):

	global filename_list_general, gleason_scores_general, patch_scores_general, SIZE_P

	patches = []

	new_patch_size = 224
		#load file
	try:
		file = openslide.open_slide(filename)
	except:
		file = openslide.OpenSlide(filename)
	level = 0

		#load mask
	fname = os.path.split(filename)[-1]
		#check if exists
	fname_mask = PATH_INPUT_MASKS+fname+'/'+fname+'_mask_use.png' 

	array_dict = []

		#select level with highest magnification
	MAGNIFICATION_SELECTED = 40
	SELECTED_LEVEL = int(float(file.properties['aperio.AppMag']))
	MASK_LEVEL = 1.25
	MAGNIFICATION_RATIO = SELECTED_LEVEL/MASK_LEVEL
	WINDOW_40X = args.SIZE_P
	GLIMPSE_SIZE_SELECTED_LEVEL = int(WINDOW_40X*SELECTED_LEVEL/MAGNIFICATION_SELECTED)
	GLIMPSE_SIZE_1x = int(GLIMPSE_SIZE_SELECTED_LEVEL/MAGNIFICATION_RATIO)
	STRIDE_SIZE_1X = 0
	TILE_SIZE_1X = GLIMPSE_SIZE_1x+STRIDE_SIZE_1X
	PIXEL_THRESH = args.PERCENTAGE

	output_dir = PATH_OUTPUT+fname

	#if (os.path.isfile(fname_mask) and os.path.isdir(output_dir)):
	if (os.path.isfile(fname_mask)):
			#creates directory
		output_dir = PATH_OUTPUT+fname
		create_dir(output_dir)

			#create CSV file structure (local)
		filename_list = []
		level_list = []
		x_list = []
		y_list = []
		factor_list = []
		BRs = []

		img = Image.open(fname_mask)
		img = np.asarray(img)

			#if it is a biopsy and it includes similar slices, the patches are selected only from some of them
		tile_x = int(img.shape[1]/TILE_SIZE_1X)
		tile_y = int(img.shape[0]/TILE_SIZE_1X)
			
		n_image = 0
		threshold = PIXEL_THRESH

		for i in range(tile_y):
			for j in range(tile_x):
				y_ini = int(TILE_SIZE_1X*i)
				x_ini = int(TILE_SIZE_1X*j)

				glimpse = img[y_ini:y_ini+GLIMPSE_SIZE_1x,x_ini:x_ini+GLIMPSE_SIZE_1x]

				check_flag = check_background(glimpse,threshold,GLIMPSE_SIZE_SELECTED_LEVEL,MAGNIFICATION_RATIO)
				
				if(check_flag):

					fname_patch = output_dir+'/'+fname+'_'+str(n_image)+'.png'
						#change to magnification 40x
					x_coords_0 = int(x_ini*MAGNIFICATION_RATIO)
					y_coords_0 = int(y_ini*MAGNIFICATION_RATIO)
						
					file_40x = file.read_region((x_coords_0,y_coords_0),level,(GLIMPSE_SIZE_SELECTED_LEVEL,GLIMPSE_SIZE_SELECTED_LEVEL))
					file_40x = file_40x.convert("RGB")

					br = BR(file_40x)
					
					new_patch_size = 224
					save_im = file_40x.resize((new_patch_size,new_patch_size))
					save_im = np.asarray(save_im)	

					#if (whitish_img(save_im) and exposure.is_low_contrast(save_im)==False):
					if (exposure.is_low_contrast(save_im)==False):

						io.imsave(fname_patch, save_im)
						
						#add to arrays (local)
						filename_list.append(fname_patch)
						level_list.append(level)
						x_list.append(x_coords_0)
						y_list.append(y_coords_0)
						factor_list.append(file.level_downsamples[level])
						BRs.append(br)
						n_image = n_image+1
						#save the image
						#create_output_imgs(file_10x,fname)
					else:
						print("low_contrast " + str(output_dir))
		
			#add to general arrays
		if (n_image!=0):
			lockGeneralFile.acquire()
			filename_list_general.append(output_dir)
			gleason_scores_general.append(gleason_score)


			print("len filename " + str(len(filename_list_general)) + "; WSI done: " + filename)
			print("extracted " + str(n_image) + " patches")

			lockGeneralFile.release()

			write_coords_local_file(fname,[filename_list,level_list,x_list,y_list,factor_list])
			write_paths_local_file(fname,filename_list)
			write_paths_br_file(fname,filename_list,BRs)
		else:
			print("ZERO OCCURRENCIES " + str(output_dir))
			print(right_side)

	else:
		print("no mask")

def explore_list(list_dirs,primary_gleason_patterns,secondary_gleason_patterns):
	global list_dicts, n
	
	
	#print(threadname + str(" started"))


	for i in range(len(list_dirs)):
		analyze_file(list_dirs[i],primary_gleason_patterns[i],secondary_gleason_patterns[i])
	#print(threadname + str(" finished"))


#list of lists fname-bool
def create_list_dicts(filenames,gs,ps):
	n_list = []
	for (f,g,p)in zip(filenames,gs,ps):
		dic = {"filename":f,"primary_GG":g,"secondary_GG":p,"state":False}
		n_list.append(dic)
	return n_list

def chunker_list(seq, size):
		return (seq[i::size] for i in range(size))

def main():
	#create output dir if not exists
	start_time = time.time()
	global list_dicts, n, filename_list_general, gleason_scores_general, patch_scores_general


		#create CSV file structure (global)
	filename_list_general = []
	gleason_scores_general = []
	patch_scores_general = []

	n = 0
		#create dir output
	if not os.path.exists(PATH_OUTPUT):
		print("create_output " + str(PATH_OUTPUT))
		os.makedirs(PATH_OUTPUT)

	list_dirs, primary_gleason_patterns, secondary_gleason_patterns = read_file_csv(LIST_FILE)
	
	def chunker_list(seq, size):
		return (seq[i::size] for i in range(size))

	list_dirs = list(chunker_list(list_dirs,THREAD_NUMBER))
	primary_gleason_patterns = list(chunker_list(primary_gleason_patterns,THREAD_NUMBER))
	secondary_gleason_patterns = list(chunker_list(secondary_gleason_patterns,THREAD_NUMBER))

	threads = []
	for i in range(THREAD_NUMBER):
		t = threading.Thread(target=explore_list,args=(list_dirs[i],primary_gleason_patterns[i],secondary_gleason_patterns[i]))
		threads.append(t)

	for t in threads:
		t.start()
		#time.sleep(60)

	for t in threads:
		t.join()

	try:
		write_general_csv(GENERAL_TXT_PATH,[filename_list_general,gleason_scores_general,patch_scores_general])
	except:
		pass
	
		#prepare data
	
	elapsed_time = time.time() - start_time
	print("elapsed time " + str(elapsed_time))

if __name__ == "__main__":
	main()
