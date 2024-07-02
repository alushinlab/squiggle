#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
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
from scipy import interpolate; from scipy.ndimage import filters
import scipy
from skimage.morphology import skeletonize_3d; import scipy
################################################################################
# method to import synthetic data from files
def import_synth_data(noise_folder, noNoise_folder, box_length, NUM_IMGS_MIN, NUM_IMGS_MAX):
	noise_holder = []; noNoise_holder = []
	print('Loading files from ' + noise_folder)
	for i in tqdm(range(NUM_IMGS_MIN, NUM_IMGS_MAX)):
		file_name = 'actin_rotated%05d.mrc'%i
		noise_data = None; noNoise_data = None
		with mrcfile.open(noise_folder + file_name) as mrc:
			if(mrc.data.shape == (box_length,box_length)):
				noise_data = mrc.data
		file_name = 'actin_rotated%05d.mrcs'%i
		with mrcfile.open(noNoise_folder + file_name) as mrc:
			if(mrc.data.shape == (5,box_length,box_length)):
				noNoise_data = mrc.data
				
		if(not np.isnan(noise_data).any() and not np.isnan(noNoise_data).any()): #doesn't have a nan
			noise_holder.append(noise_data)
			noNoise_holder.append(noNoise_data)
		
		else: # i.e. if mrc.data does have an nan, skip it and print a statement
			print('Training image number %d has at least one nan value. Skipping this image.'%i)
	
	return noise_holder, noNoise_holder

################################################################################
#https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
import keras.backend as K
def CCC(y_pred, y_true):
	x = y_true
	y = y_pred
	mx=K.mean(x)
	my=K.mean(y)
	xm, ym = x-mx, y-my
	r_num = K.sum(tf.multiply(xm,ym))
	r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
	r = r_num / r_den
	return -1*r

################################################################################
################################################################################
################################################################################
def hist_match(source, template):
	"""
	Adjust the pixel values of a grayscale image such that its histogram
	matches that of a target image
	Arguments:
	-----------
		source: np.ndarray
		Image to transform; the histogram is computed over the flattened array
		template: np.ndarray
		Template image; can have different dimensions to source
	Returns:
	-----------
		matched: np.ndarray
					The transformed output image
	"""
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()
	# get the set of unique pixel values and their corresponding indices and counts
	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)
	
	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]
	
	# interpolate linearly to find the pixel values in the template image
	# that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
	return interp_t_values[bin_idx].reshape(oldshape)

################################################################################
def get_circle_center(y):
	y_center = y[:,-2:]
	radius_of_curv = 1.0/y[:,0]*100000/4.12
	radius_of_curv = tf.convert_to_tensor(radius_of_curv)
	radius_of_curv = K.repeat(K.expand_dims(radius_of_curv, axis=-1), 2)
	rotation = y[:,1]*np.pi/180.0
	R = [[K.cos(rotation), -1.0*K.sin(rotation)],[K.sin(rotation), K.cos(rotation)]]
	R = tf.convert_to_tensor(R)
	R = K.permute_dimensions(R, (2,0,1))
	unit_x = K.variable([[1.0],[0.0]])
	disp = K.dot(K.cast(R, K.floatx()), K.cast(unit_x,K.floatx()))
	disp = tf.multiply(K.cast(disp, K.floatx()), K.cast(radius_of_curv,K.floatx()))
	return disp + K.expand_dims(y_center, axis=-1)

def custom_loss(y_pred, y_true):
	MSE = K.mean(K.square(y_pred - y_true), axis=-1)
	true_center = get_circle_center(y_true)
	pred_center = get_circle_center(y_pred)
	delta_center = K.transpose(tf.norm((true_center - pred_center), axis=-2))[0]
	d = delta_center
	r = K.cast(tf.convert_to_tensor(y_true[:,0]), K.floatx())
	R = K.cast(tf.convert_to_tensor(y_pred[:,0]), K.floatx())
	r = 1.0/(r+0.3)*100000/4.12
	R = 1.0/(R+0.3)*100000/4.12
	term_1 = K.square(r)*tf.acos(K.maximum(K.minimum((K.square(d)+K.square(r)-K.square(R))/(2*d*r+K.epsilon()),1.0), -1.0))
	term_2 = K.square(R)*tf.acos(K.maximum(K.minimum((K.square(d)+K.square(R)-K.square(r))/(2*d*R+K.epsilon()),1.0), -1.0))
	term_3 = -0.5*K.sqrt((r+R-d)*(d+r-R)*(d-r+R)*(d+r+R))
	
	intersection = term_1 + term_2 + term_3
	union = np.pi*K.square(r) + np.pi*K.square(R) - intersection
	IoU = intersection / (union+K.epsilon())
	return MSE-100.0**IoU

def custom_activation(x):
	activ = keras.activations.linear(x)[:,1:]
	non_lin = K.expand_dims(keras.activations.relu(x)[:,0], axis=-1)+0.31
	return K.concatenate((non_lin, activ), axis=-1)

################################################################################
def slice_up_micrograph(real_data, increment, hist_matcher):
	extractions = []
	for i in range(0, real_data.shape[0]-128, increment):
		for j in range(0, real_data.shape[1]-128, increment):
			extraction = -1.0*real_data[i:i+128,j:j+128]
			extraction = (extraction - np.mean(extraction)) / np.std(extraction)
			extraction = hist_match(extraction, hist_matcher)
			extraction = (extraction - np.mean(extraction)) / np.std(extraction)
			extractions.append(extraction)
	
	extractions = np.asarray(extractions)
	return extractions

# same as above, but more efficient and without histogram matching
def slice_up_micrograph(real_data, increment, hist_matcher):
	extractions = []
	for i in range(0, real_data.shape[0]-128, increment):
		for j in range(0, real_data.shape[1]-128, increment):
			extraction = -1.0*real_data[i:i+128,j:j+128]
			extraction = hist_match(extraction, hist_matcher)
			extractions.append(extraction)
	extractions = np.moveaxis(np.asarray(extractions), 0,-1)
	extractions = (extractions - np.mean(extractions, axis=(0,1))) / np.std(extractions, axis=(0,1))
	return np.moveaxis(extractions, -1, 0)


def stitch_back_seg(shape, preds, increment):
	stitch_back = np.zeros((shape))
	cntr=0
	for i in range(0, stitch_back.shape[0]-128, increment):
		for j in range(0, stitch_back.shape[1]-128, increment):
			stitch_back[i:i+128, j:j+128] = np.max(np.stack((preds[cntr], stitch_back[i:i+128, j:j+128])),axis=0)
			cntr=cntr+1
	return stitch_back

def prune(skel_pruned, i):
	if(i==0):
		return skel_pruned
	else:
		E_img = filters.convolve(skel_pruned,[[1,1,1],[1,10,1],[1,1,1]])
		skel_pruned[E_img ==11] = 0
		prune(skel_pruned, i-1)
	return skel_pruned

def find_end_pt_partners(skel_pruned, end_pts, i):
	start = [end_pts[i][0],end_pts[i][1]]
	filament = [start]
	window = skel_pruned[filament[-1][0]-1:filament[-1][0]+2,filament[-1][1]-1:filament[-1][1]+2].copy()
	window[1,1] = 0
	new_pt = (np.argwhere(window==1)-1)[0]
	filament.append(filament[-1]+new_pt)
	while(not((filament[-1] == end_pts).all(axis=1).any()) or len(filament)<2):
		prev_pt = -1*new_pt+1
		window = skel_pruned[filament[-1][0]-1:filament[-1][0]+2,filament[-1][1]-1:filament[-1][1]+2].copy()
		window[1,1] = 0
		window[prev_pt[0],prev_pt[1]] = 0
		new_pt = (np.argwhere(window==1)-1)[0]
		filament.append(filament[-1]+new_pt)
	return np.asarray(filament)

def skeletonize_and_prune_nubs(stitch_back, intersect_removal_box_size, prune_length, skel_threshold):
	# First, skeletonize the image
	sem_seg_img = stitch_back.copy()
	big_mask = np.ones((sem_seg_img.shape))
	big_mask[64:big_mask.shape[0]-64,64:big_mask.shape[1]-64] = 0
	big_mask = -1*big_mask +1
	sem_seg_img = np.multiply(big_mask, sem_seg_img)
	skel = skeletonize_3d(sem_seg_img.astype('float32')>skel_threshold)
	skel[skel>0] = 1
	# Then, prune the image by trimming back end points (to remove nubs)
	skel_pruned = prune(skel.copy(), prune_length)
	# Follow up pruning with another round of skeletonize to remove weird triple pts
	skel_pruned = skeletonize_3d(skel_pruned.astype('float32')>skel_threshold-0.05)
	skel_pruned[skel_pruned>0] = 1
	
	# Now, remove three-way and four-way intersections
	E_img = filters.convolve(skel_pruned,[[1,1,1],[1,10,1],[1,1,1]])
	triple_pts = np.asarray(np.argwhere(E_img==13))
	quadruple_pts = np.asarray(np.argwhere(E_img==14))
	for i in range(0, len(triple_pts)):
		x = triple_pts[i][0]
		y = triple_pts[i][1]
		skel_pruned[x-intersect_removal_box_size:x+intersect_removal_box_size,y-intersect_removal_box_size:y+intersect_removal_box_size] = 0
	return skel_pruned

def define_filaments_from_skelPruned(skel_pruned, end_pts, shortest_fil_len):
	filaments = []
	for i in range(0, len(end_pts)):
		filaments.append(find_end_pt_partners(skel_pruned, end_pts, i))
	
	# The above method will result in two tracks per filament. To get only one track
	# per filament, remove the duplicates
	filaments_noDups = []
	for i in range(0, len(filaments)):
		add_this_fil = True
		for j in range(0, len(filaments_noDups)):
			if(((filaments[i][0] == filaments_noDups[j][-1])).all() or len(filaments[i]) < shortest_fil_len):
				add_this_fil = False
		if(add_this_fil):
			filaments_noDups.append(filaments[i])
	
	filaments = filaments_noDups
	return filaments

def extract_filaments(real_data, hist_matcher, filament, increment):
	extractions = []
	for i in range(0, len(filament), increment):
			x,y = max(64,filament[i][0].astype('int')), max(64,filament[i][1].astype('int'))
			extraction = -1.0*real_data[x-64:x+64,y-64:y+64]
			extraction = (extraction - np.mean(extraction)) / np.std(extraction)
			extraction = hist_match(extraction, hist_matcher)
			extraction = (extraction - np.mean(extraction)) / np.std(extraction)
			extractions.append(extraction)
	return np.asarray(extractions)

def spline_curv_estimate(filament, smooth_param, sampling, buff, min_threshold):
	try:
		tck, u = interpolate.splprep([filament[:,0],filament[:,1]], s=smooth_param) # 1-d parametric spline
		unew = np.linspace(0, 1.0 ,len(filament)/sampling)
		r = interpolate.splev(unew, tck)
		r = np.asarray(r).T
		r_prime = interpolate.splev(unew, tck, der=1)
		r_prime = np.asarray(r_prime).T
		r_double_prime = interpolate.splev(unew, tck, der=2)
		r_double_prime = np.asarray(r_double_prime).T
	except:
		return [],[]
	curvature_num = np.multiply(r_prime[:,1],r_double_prime[:,0]) - np.multiply(r_prime[:,0],r_double_prime[:,1])
	curvature_denom = (r_prime[:,1]**2 + r_prime[:,0]**2)**1.5
	curvature = 1.0/((curvature_num) / curvature_denom) # Try removing absolute value of curvature
	#curvature = 1.0/(np.abs(curvature_num) / curvature_denom)
	#curvature[curvature < min_threshold] = 10000
	return curvature[buff:-buff], r[buff:-buff]


def starify(*args):
	return (''.join((('%.3f'%i).rjust(13))  if not isinstance(i,int) else ('%d'%i).rjust(13) for i in args) + ' \n')[1:]

def isolate_one_filament(extracted_filament, FCN):
	sem_seg_img = FCN.predict(np.expand_dims(extracted_filament, axis=-1))[:,:,:,1]
	for i in range(0, len(sem_seg_img)):
		skel = skeletonize_3d(sem_seg_img[i].astype('float32')>0.9)
		skel[skel>0] = 1
		# Then, prune the image by trimming back end points (to remove nubs)
		skel_pruned = prune(skel.copy(), 4)
		# Follow up pruning with another round of skeletonize to remove weird triple pts
		skel_pruned = skeletonize_3d(skel_pruned.astype('float32')>0.9)
		skel_pruned[skel_pruned>0] = 1
		skel = skel_pruned
		E_img = filters.convolve(skel,[[1,1,1],[1,10,1],[1,1,1]])
		end_pts = np.asarray(np.argwhere(E_img==11))
		if((len(end_pts) != 1) or (len(end_pts) != 2)):
			extracted_filament[i] = np.ones((128,128))*200
		#elif((len(end_pts) == 1) or  (len(end_pts) == 2)):
		#	continue
	return extracted_filament










