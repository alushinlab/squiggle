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
# prune end points that are false positive end points
def prune(pot_end_pts, triple_pts):
	idxs_to_prune = []
	for i in range(0, len(pot_end_pts)):
		for j in range(0, len(triple_pts)):
			if(np.linalg.norm(pot_end_pts[i] - triple_pts[j]) < 7):
				idxs_to_prune.append(i)
	
	end_pts = np.delete(pot_end_pts, idxs_to_prune, axis=0)
	return end_pts

################################################################################
def eval_spline_energy(x,y,E_img):
	tck, u = interpolate.splprep([x,y], s=0) # 1-d parametric spline
	unew = np.linspace(0, 1.0 ,50)
	r = interpolate.splev(unew, tck)
	r = np.asarray(r).T
	r[r[:,0]>127] = 127; r[r[:,1]>127] = 127; r[r[:,0]<0] = 0; r[r[:,1]<0] = 0
	r_prime = interpolate.splev(unew, tck, der=1)
	r_prime = np.asarray(r_prime).T
	r_double_prime = interpolate.splev(unew, tck, der=2)
	r_double_prime = np.asarray(r_double_prime).T
	elasticity = np.sum(np.linalg.norm(r_prime, axis=1)**2)
	stiffness = np.sum(np.linalg.norm(r_double_prime, axis=1)**2)
	img_energy = np.sum(E_img[r[:,1].astype(int), r[:,0].astype(int)])
	alpha, beta, gamma = -0.01, -0.01, 1000000.0
	energy = alpha * elasticity + beta * stiffness + gamma * img_energy
	internal_energy = alpha* elasticity + beta* stiffness
	return r, energy, internal_energy

################################################################################
def spline_energies(pt1, pt2, E_img):
	cntl_pts = 6
	x_orig = np.linspace(pt1[1]+0.01, pt2[1], cntl_pts) # add 0.001 in case of duplicate x or y coords
	y_orig = np.linspace(pt1[0]+0.01, pt2[0], cntl_pts)
	r_orig, energy_orig, internal_energy_orig=eval_spline_energy(x_orig, y_orig, E_img)
	best_x = x_orig
	best_y = y_orig
	best_energy = energy_orig
	best_r = r_orig
	best_internal_energy = internal_energy_orig
	for i in range(0, 1000):
		del_x = np.random.randint(-1,2,size=(cntl_pts)); del_x[0] = 0; del_x[-1] = 0
		del_y = np.random.randint(-1,2,size=(cntl_pts)); del_y[0] = 0; del_y[-1] = 0
		x = best_x + del_x
		y = best_y + del_y
		x[x>127] = 127; y[y>127] = 127; x[x<0] = 0; y[y<0] = 0
		r, energy, internal_energy = eval_spline_energy(x,y, E_img)
		if(energy > best_energy):
			best_energy = energy
			best_x = x
			best_y = y
			best_r = r
			best_internal_energy = internal_energy
	
	return best_internal_energy, best_r

################################################################################
def segment_image(sem_seg_img, end_pts, orig_img):
	if(len(end_pts) == 0):
		return np.zeros((128,128))
	
	elif(len(end_pts) == 2 or len(end_pts)==1):
		return orig_img
	
	elif(len(end_pts) == 4):
		E_img = filters.gaussian_filter(sem_seg_img,1)
		# construct dictionary of possible matchups
		combos = {0:[(0,1),(2,3)], 1:[(0,2),(1,3)], 2:[(0,3),(1,2)]}
		combo_energies = np.zeros(len(combos)); r_holder = []
		for i in range(0, len(combos)):
			pt1 = end_pts[combos[i][0][0]]; pt2 = end_pts[combos[i][0][1]]
			energy_0, best_r0 = spline_energies(pt1, pt2, E_img)
			
			pt1 = end_pts[combos[i][1][0]]; pt2 = end_pts[combos[i][1][1]]
			energy_1, best_r1 = spline_energies(pt1, pt2, E_img)
			combo_energies[i] = energy_0 + energy_1
			r_holder.append([best_r0, best_r1])
		
		best_r = np.asarray(r_holder[np.argmax(combo_energies)])	
		plt.imshow(E_img)
		plt.scatter(best_r[0,:,0], best_r[0,:,1])
		plt.scatter(best_r[1,:,0], best_r[1,:,1])
		plt.show()
		mask = np.zeros((4,128,128))
		xA = np.mgrid[0:128:1, 0:128:1].reshape(2, -1).T.reshape((128*128,2))
		mask[0] = np.min(scipy.spatial.distance.cdist(xA,best_r[0]), axis=-1).reshape((128,128))<10
		mask[1] = np.min(scipy.spatial.distance.cdist(xA,best_r[1]), axis=-1).reshape((128,128))<10
		A_intersect_B = (mask[0] + mask[1])
		A_intersect_B[A_intersect_B == 1] = 0
		A_intersect_B[A_intersect_B == 2] = 1
		A_minus_AintersectB = mask[0] - A_intersect_B
		mask[2] = -1*A_minus_AintersectB.T+1
		mask[2] = np.multiply(mask[2], orig_img)
		
		B_minus_AintersectB = mask[1] - A_intersect_B
		mask[3] = -1*B_minus_AintersectB.T+1
		mask[3] = np.multiply(mask[3], orig_img)
		
		non_actin = (-1*((mask[0] + mask[1]) - A_intersect_B).T +1)
		non_actin = np.multiply(non_actin, orig_img)
		non_actin_flattened = non_actin.flatten()
		noise_dist = np.delete(non_actin_flattened, np.where(non_actin_flattened == 0))
		noise_dist = np.random.permutation(np.repeat(noise_dist, 4))[:128*128].reshape((128,128))
		
		fill_in_noise_A = np.multiply(A_minus_AintersectB, noise_dist).T
		fill_in_noise_B = np.multiply(B_minus_AintersectB, noise_dist).T
		mask[2] = mask[2] + fill_in_noise_A
		mask[3] = mask[3] + fill_in_noise_B
		
		mask[2] = (mask[2] - np.mean(mask[2])) / np.std(mask[2])
		mask[3] = (mask[3] - np.mean(mask[3])) / np.std(mask[3])
		return mask
	
	elif(len(end_pts) == 6):
		mask = three_filaments(sem_seg_img, end_pts, orig_img)
		return mask
	
	else:
		print('The program identified an odd number of end points in this image')
		print('A blank frame will be returned for now')
		return np.zeros((128,128))

################################################################################
def three_filaments(sem_seg_img, end_pts, orig_img):
	E_img = filters.gaussian_filter(sem_seg_img,1)
	# construct dictionary of possible matchups
	combos = { 0:[(0,1),(2,3),(4,5)],  1:[(0,1),(2,4),(3,5)],  2:[(0,1),(2,5),(3,4)],#
				  3:[(0,2),(1,3),(4,5)],  4:[(0,2),(1,4),(3,5)],  5:[(0,2),(1,5),(3,4)],
				  6:[(0,3),(1,2),(4,5)],  7:[(0,3),(1,4),(2,5)],  8:[(0,3),(1,5),(2,4)],
				  9:[(0,4),(1,2),(3,5)], 10:[(0,4),(1,3),(2,5)], 11:[(0,4),(1,5),(2,3)],
				 12:[(0,5),(1,2),(3,4)], 13:[(0,5),(1,3),(2,4)], 14:[(0,5),(1,4),(2,3)]	
				}
	combo_energies = np.zeros(len(combos)); r_holder = []
	for i in range(0, len(combos)):
		pt1 = end_pts[combos[i][0][0]]; pt2 = end_pts[combos[i][0][1]]
		energy_0, best_r0 = spline_energies(pt1, pt2, E_img)
		
		pt1 = end_pts[combos[i][1][0]]; pt2 = end_pts[combos[i][1][1]]
		energy_1, best_r1 = spline_energies(pt1, pt2, E_img)
		
		pt1 = end_pts[combos[i][2][0]]; pt2 = end_pts[combos[i][2][1]]
		energy_2, best_r2 = spline_energies(pt1, pt2, E_img)
		combo_energies[i] = energy_0 + energy_1 + energy_2
		r_holder.append([best_r0, best_r1, best_r2])
	
	best_r = np.asarray(r_holder[np.argmax(combo_energies)])	
	plt.imshow(E_img)
	plt.scatter(best_r[0,:,0], best_r[0,:,1])
	plt.scatter(best_r[1,:,0], best_r[1,:,1])
	plt.scatter(best_r[2,:,0], best_r[2,:,1])
	plt.show()
	mask = np.zeros((3,128,128))
	xA = np.mgrid[0:128:1, 0:128:1].reshape(2, -1).T.reshape((128*128,2))
	A = np.min(scipy.spatial.distance.cdist(xA,best_r[0]), axis=-1).reshape((128,128))<10
	B = np.min(scipy.spatial.distance.cdist(xA,best_r[1]), axis=-1).reshape((128,128))<10
	C = np.min(scipy.spatial.distance.cdist(xA,best_r[2]), axis=-1).reshape((128,128))<10
	
	mask[0] = np.multiply((-1*(B+C)+1).T, orig_img)
	mask[1] = np.multiply((-1*(A+C)+1).T, orig_img)
	mask[2] = np.multiply((-1*(A+B)+1).T, orig_img)
	
	actin_present = (A + B + C)
	actin_present[actin_present > 0] = 1
	non_actin = (-1*(actin_present).T +1)
	non_actin = np.multiply(non_actin, orig_img)
	non_actin_flattened = non_actin.flatten()
	noise_dist = np.delete(non_actin_flattened, np.where(non_actin_flattened == 0))
	noise_dist = np.random.permutation(np.repeat(noise_dist, 4))[:128*128].reshape((128,128))
	
	fill_in_noise_A = np.multiply(B+C, noise_dist).T
	fill_in_noise_B = np.multiply(A+C, noise_dist).T
	fill_in_noise_C = np.multiply(A+B, noise_dist).T
	mask[0] = mask[0] + fill_in_noise_A
	mask[1] = mask[1] + fill_in_noise_B
	mask[2] = mask[2] + fill_in_noise_C
	
	mask[0] = (mask[0] - np.mean(mask[0])) / np.std(mask[0])
	mask[1] = (mask[1] - np.mean(mask[1])) / np.std(mask[1])
	mask[2] = (mask[2] - np.mean(mask[2])) / np.std(mask[2])
	return mask







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


















