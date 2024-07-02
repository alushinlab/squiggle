#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
print('Loading python packages...')
################################################################################
# import of python packages
import numpy as np
import matplotlib.pyplot as plt
import keras
import mrcfile
import random
from tqdm import tqdm
from keras import layers
from keras.models import Model
import tensorflow as tf
#from EMAN2 import *
from scipy import interpolate; from scipy.ndimage import filters
from skimage.morphology import skeletonize_3d; import scipy
import keras.backend as K
from helper_multiple_whole_micrograph_preds import *
import glob
import os; import sys
################################################################################
INDEX = int(sys.argv[1]) #1
TOTAL_GPUS = int(sys.argv[2])#4
################################################################################
print('Python packages loaded. Setting CUDA environment...')
os.environ["CUDA_VISIBLE_DEVICES"]= str(INDEX-1)#"2"
################################################################################
################################################################################
# load trained Fully Convolutional Network for semantic segmentation
################################################################################
print('Loading neural network models'); print('')
model_path = './trained_networks/FCN_semantic_segmentation_squig_60k_15kCarbon_lowerDefocus.h5'#'./trained_networks/FCN_semantic_segmentation_squig_30k_moreNoiseCarbon_lowerDefocus.h5'#'./trained_networks/FCN_semantic_segmentation_squig_30k.h5'#'./trained_networks/FCN_semantic_segmentation_3.h5'
#model_path = './trained_networks/FCN_semantic_segmentation_squig_30k_moreNoiseCarbon_lowerDefocus.h5'
FCN = keras.models.load_model(model_path)
#model_path = './trained_networks/800000training_CCC9887.h5'
#autoencoder_three = keras.models.load_model(model_path, custom_objects={'CCC':CCC})
#model_path = './trained_networks/multidimensional_regression_4dim_curvOK.h5'
#curv_measurement = keras.models.load_model(model_path, custom_objects={'custom_activation':custom_activation, 'custom_loss':custom_loss})
################################################################################
print(''); print('');
print('Neural network models loaded. Starting predictions')
# Load one test image for histogram matching
hist_match_dir = ''#'/mnt/data0/neural_network_training_sets/noise_proj4/'
with mrcfile.open(hist_match_dir + 'actin_rotated%05d.mrc'%1) as mrc:
	hist_matcher = mrc.data

file_names = sorted(glob.glob('../Micrographs_bin4/*bin4.mrc'))#[29:]
file_names = file_names#[7380:8498]#file_names[int(len(file_names)*(1.0/TOTAL_GPUS)*(INDEX-1)):int(len(file_names)*(1.0/TOTAL_GPUS)*INDEX)]
print('Predicting on files from ' + file_names[0] + ' to ' + file_names[-1])
for i in tqdm(range(0, len(file_names))):
#for i in tqdm(range(1003, 1103)):
	################################################################################
	# Load real micrographs

i=656
real_data_dir = ''#'whole_micrographs/'
big_micrograph_name = file_names[i]#'beta_actin_Au_0757_noDW_bin4.mrc'
with mrcfile.open(real_data_dir + big_micrograph_name) as mrc:
	real_data = mrc.data

plt.imshow(real_data, cmap=plt.cm.gray)
plt.show()

################################################################################
# Divide up the whole micrograph to feed to the FCN for semantic segmentation
import skimage.morphology as filt_stitch_back
from skimage.morphology import disk, remove_small_objects
increment = 32
extractions = slice_up_micrograph(real_data, increment, hist_matcher)
preds = FCN.predict(np.expand_dims(extractions, axis=-1))[:,:,:,1]
stitch_back = stitch_back_seg(real_data.shape, preds, increment)

#dilated = filt_stitch_back.dilation(stitch_back, disk(8))
skel_pruned = skeletonize_and_prune_nubs(stitch_back, 16, 8, 0.9)
E_img = filters.convolve(skel_pruned,[[1,1,1],[1,10,1],[1,1,1]])
E_img[E_img != 11] = 0
E_img = filters.convolve(E_img,disk(50))
E_img[E_img > 1] = 1
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(real_data, cmap=plt.cm.gray)
ax[0,1].imshow(stitch_back, cmap=plt.cm.gray)
ax[1,0].imshow(stitch_back>0.9, cmap=plt.cm.gray)
ax[1,1].imshow(E_img)
plt.show()


#skel_pruned = remove_small_objects(skel_pruned, 1000)
E_img = filters.convolve(skel_pruned,[[1,1,1],[1,10,1],[1,1,1]])
E_img[E_img != 11] = 0
E_img = filters.convolve(E_img,disk(50))
E_img[E_img > 1] = 1

dilated = filt_stitch_back.dilation(stitch_back+0.6*(E_img*stitch_back), disk(8))
skel_pruned = skeletonize_and_prune_nubs(dilated, 32, 8, 0.3)
#skel_pruned = remove_small_objects(skel_pruned, 500)
E_img = filters.convolve(skel_pruned,[[1,1,1],[1,10,1],[1,1,1]])
E_img[E_img != 11] = 0
E_img = filters.convolve(E_img,disk(50))
E_img[E_img > 1] = 1
	
	#dilated = dilated+0.7*(E_img*dilated)
	#dilated[dilated > 1] = 1
	#skel_pruned = skeletonize_and_prune_nubs(dilated, 32, 8, 0.4)
	#fig, ax = plt.subplots(2,2)
	#ax[0,0].imshow(stitch_back, cmap=plt.cm.gray, origin='lower')
	#ax[0,1].imshow(skel_pruned, cmap=plt.cm.gray, origin='lower')
	#ax[1,0].imshow(dilated, cmap=plt.cm.gray, origin='lower')
	#ax[1,1].imshow(dilated>0.4, cmap=plt.cm.gray, origin='lower')
	#plt.show()
	
	################################################################################
	################################################################################
	# define actin filaments for whole micrograph. 
	triple_pt_box_size = 16
	num_to_prune = 8
	skel_pruned = skeletonize_and_prune_nubs(dilated, triple_pt_box_size, num_to_prune, 0.3)
	# Now plot end points to see results
	E_img = filters.convolve(skel_pruned,[[1,1,1],[1,10,1],[1,1,1]])
	end_pts = np.asarray(np.argwhere(E_img==11))
	################################################################################
	# Now that we have a clean, image-wide skeleton; do instance segmentation
	filaments = define_filaments_from_skelPruned(skel_pruned, end_pts, 50)
	# Plot whole micrograph with segmented pixels
	real_data = -1.0*real_data
	real_data = (real_data - np.mean(real_data)) / np.std(real_data)
	_=plt.imshow(-1.0*real_data, cmap=plt.cm.gray)
	colors = iter(plt.cm.rainbow(np.linspace(0,1,len(filaments))))
	rs = []; curvatures = []
	for j in range(0, len(filaments)):
		_=plt.scatter(filaments[j][:,1], filaments[j][:,0], s=0.1, alpha=0.2, color=next(colors))
		curvature, r = spline_curv_estimate(filaments[j], len(filaments[j]), 3, 12, 50) # sampling(usually10), buff, min_threshold
		if(curvature != [] and len(curvature) > 20):
			r_curv_range = r[np.argwhere(np.logical_and(-10e20<curvature, curvature<10e20))][:,0,:]
			kept_curvs = curvature[np.argwhere(np.logical_and(-10e20<curvature, curvature<10e20))]
			plt.scatter(r_curv_range[:,1], r_curv_range[:,0], c='red',s=5, alpha=0.5)
			rs.append(np.concatenate((r_curv_range,kept_curvs),axis=-1)); curvatures.append(curvature)
	
	plt.tight_layout()
	plt.savefig('pngs/'+big_micrograph_name[:-4].split('/')[-1]+'.png', dpi=600)
	plt.clf()
	################################################################################
	# Extract and predict rotation
	step_size_px = 1
	rs_full = []; extractions_full = []
	for j in range(0, len(rs)):
		if(rs[j].shape[0] != 0 and rs[j].shape[0]>1):
			rot_angles = np.zeros((rs[j].shape[0], 2))
			extraction = extract_filaments(real_data, hist_matcher, rs[j], step_size_px)
			xyrot = np.concatenate((rs[j], rot_angles), axis=1)
			rs_full.append(xyrot)
			extractions_full.append(extraction)
	
	
	curv_preds = []
	for j in range(0, len(rs_full)):
		measured_curv = 0#curv_measurement.predict(np.expand_dims(extractions_full[j], axis=-1))
		#rs_full[j][:,2] = measured_curv[:,1] # rotation
		rs_full[j][:,3] = 0#measured_curv[:,0] #curvature
		curv_preds.append(measured_curv)
	
	################################################################################
	# Prepare star file
	header = '# RELION; version 3.0\n\ndata_\n\nloop_ \n_rlnCoordinateX #1 \n_rlnCoordinateY #2 \n_rlnClassNumber #3 \n_rlnAutopickFigureOfMerit #4 \n_rlnHelicalTubeID #5 \n_rlnAngleTiltPrior #6 \n_rlnAnglePsiPrior #7 \n_rlnHelicalTrackLength #8 \n_rlnAnglePsiFlipRatio #9 \n'
	star_file = header
	for j in range(0, len(rs_full)):
		for k in range(0, len(rs_full[j])):
			if(rs_full[j][k][3] <275):
				star_file = star_file + starify(rs_full[j][k][1]*4.0,rs_full[j][k][0]*4.0,1, rs_full[j][k][2], j+1, 90.0, 0, 0.0, 0.5)
	
	star_file_name = 'starFiles/'+(big_micrograph_name[:-4].split('/')[-1]).split('_bin4')[0]
	with open(star_file_name+'.star', "w") as text_file:
		text_file.write(star_file)
	







