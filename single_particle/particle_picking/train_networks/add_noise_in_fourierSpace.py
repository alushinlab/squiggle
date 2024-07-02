#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# import of python packages
print('Beginning to import packages...')
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import random
from tqdm import tqdm
print('Packages finished importing. Data will now be loaded')
################################################################################
################################################################################
# This program will read all of the files in the actin projections directory
# and it will read in the pink noise boxes and add them in Fourier space. Then
# it will take the inverse fourier transform and save that image

# Load file names of all projection images
folder = '/mnt/data0/neural_network_training_sets/'
ctf_projections_folder = folder + 'squig_proj_ctfModulated/squig_proj_ctfModulated/'
pnk_noise_box_folder = folder + 'squig_pnk_noise_boxes/'
output_dir = folder + 'squig_proj_for_denseNN/'
################################################################################
# method to import synthetic data from files
def import_synth_data(noise_folder,box_length, NUM_IMGS_MIN, NUM_IMGS_MAX):
	noise_holder = []; name_holder = []
	print('Loading files from ' + noise_folder)
	for i in tqdm(range(NUM_IMGS_MIN, NUM_IMGS_MAX)):
		file_name = 'actin_rotated%05d.mrc'%i
		noise_data = None;
		with mrcfile.open(noise_folder + file_name) as mrc:
			if(mrc.data.shape == (box_length,box_length)):
				noise_data = mrc.data
		
		if(not np.isnan(noise_data).any()): #doesn't have a nan
			noise_holder.append(noise_data)
			name_holder.append(file_name)
		
		else: # i.e. if mrc.data does have an nan, skip it and print a statement
			print('Training image number %d has at least one nan value. Skipping this image.'%i)
	
	return np.asarray(noise_holder), name_holder

def add_in_fourier_space(ctf_proj, pnk_boxes,j, file_names):
	for i in range(0, len(ctf_proj)):
		added_img = np.fft.ifft2(np.fft.fft2(np.random.gamma(0.09/0.04, 0.04)*ctf_proj[i]) + np.fft.fft2(pnk_boxes[i])).real
		#added_img = np.fft.ifft2(np.fft.fft2(np.random.normal(0.20, 0.05)*ctf_proj[i]) + np.fft.fft2(pnk_boxes[i])).real
		added_img = (added_img - np.average(added_img)) / np.std(added_img)
		#with mrcfile.new(output_dir + 'actin_rotated%05d.mrc'%(i+10000*j), overwrite=True) as mrc:
		with mrcfile.new(output_dir + file_names[i+10000*j], overwrite=True) as mrc:
			mrc.set_data(added_img.astype('float16'))


#gauss = np.random.normal(0.2, 0.05, 10000)		
#gamma1 = np.random.gamma(0.13/0.05, 0.05, 10000)
#gamma2 = np.random.gamma(0.09/0.04, 0.04, 10000)
#_=plt.hist(gamma1, bins=np.arange(0,1.0,0.01),alpha=0.5)
#_=plt.hist(gamma2, bins=np.arange(0,1.0,0.01),alpha=0.5)
#_=plt.hist(gauss, bins=np.arange(0,1.0,0.01),alpha=0.5)
#plt.show()

################################################################################
ctf_projections, file_names = import_synth_data(ctf_projections_folder, 128, 0, 800000) # change the 9000 to 800000
print('Finished loading CTF modulated projections. Will now begin adding pink noise in Fourier space...')

for i in range(0, len(ctf_projections)/10000): # change 5000 to 10000
	print('Loading ' + 'pnk_noise_boxes_10k_%03d.mrcs'%i)
	with mrcfile.open(pnk_noise_box_folder + 'pnk_noise_boxes_10k_%03d.mrcs'%i) as mrc:
		pnk_boxes_10k = mrc.data
	
	add_in_fourier_space(ctf_projections[i*10000:(i+1)*10000], pnk_boxes_10k, i, file_names)




