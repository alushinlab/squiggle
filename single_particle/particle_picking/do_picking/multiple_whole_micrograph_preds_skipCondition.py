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
import skimage.morphology as filt_stitch_back
from skimage.morphology import disk, remove_small_objects
import glob
import os; import sys; import threading
try:
    import thread
except ImportError:
    import _thread as thread
################################################################################
INDEX = int(sys.argv[1]) #1
TOTAL_GPUS = int(sys.argv[2])#4
################################################################################
print('Python packages loaded. Setting CUDA environment...')
os.environ["CUDA_VISIBLE_DEVICES"]= str(INDEX-1)#"2"
################################################################################
def quit_function(fn_name):
	# print to stderr, unbuffered in Python 2.
	#print('{0} took too long'.format(fn_name), file=sys.stderr)
	#sys.stderr.flush() # Python 3 stderr is likely buffered.
	thread.interrupt_main() # raises KeyboardInterrupt

def exit_after(s):
	'''
	use as decorator to exit process if 
	function takes longer than s seconds
	'''
	def outer(fn):
		def inner(*args, **kwargs):
			timer = threading.Timer(s, quit_function, args=[fn.__name__])
			timer.start()
			try:
				result = fn(*args, **kwargs)
			finally:
				timer.cancel()
			return result
		return inner
	return outer


# Define picking function
@exit_after(70)
def run_pick_on_micrograph(file_name):
	#############################################################################
	# Load real micrographs
	real_data_dir = ''#'whole_micrographs/'
	big_micrograph_name = file_name#'beta_actin_Au_0757_noDW_bin4.mrc'
	with mrcfile.open(real_data_dir + big_micrograph_name) as mrc:
		real_data = mrc.data
	
	#############################################################################
	# Divide up the whole micrograph to feed to the FCN for semantic segmentation
	increment = 32
	extractions = slice_up_micrograph(real_data, increment, hist_matcher)
	preds = FCN.predict(np.expand_dims(extractions, axis=-1))[:,:,:,1]
	stitch_back = stitch_back_seg(real_data.shape, preds, increment)
	
	#dilated = filt_stitch_back.dilation(stitch_back, disk(8))
	#skel_pruned = skeletonize_and_prune_nubs(stitch_back, 16, 8, 0.9)
	#E_img = filters.convolve(skel_pruned,[[1,1,1],[1,10,1],[1,1,1]])
	#E_img[E_img != 11] = 0
	#E_img = filters.convolve(E_img,disk(20))
	#E_img[E_img > 1] = 1
	
	#dilated = filt_stitch_back.dilation(stitch_back+0.6*(E_img*stitch_back), disk(8))
	#skel_pruned = skeletonize_and_prune_nubs(dilated, 16, 8, 0.9)
	#E_img = filters.convolve(skel_pruned,[[1,1,1],[1,10,1],[1,1,1]])
	#E_img[E_img != 11] = 0
	#E_img = filters.convolve(E_img,disk(50))
	#E_img[E_img > 1] = 1
	
	#dilated = dilated+0.7*(E_img*dilated)
	#dilated[dilated > 1] = 1
	
	################################################################################
	################################################################################
	# define actin filaments for whole micrograph. 
	triple_pt_box_size = 16
	num_to_prune = 8
	skel_pruned = skeletonize_and_prune_nubs(stitch_back, triple_pt_box_size, num_to_prune, 0.9)
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
	
################################################################################
# load trained Fully Convolutional Network for semantic segmentation
################################################################################
print('Loading neural network models'); print('')
model_path = './trained_networks/FCN_semantic_segmentation_squig_60kb_15kCarbon_lowerDefocus.h5'
FCN = keras.models.load_model(model_path)
################################################################################
print(''); print('');
print('Neural network models loaded. Starting predictions')
# Load one test image for histogram matching
hist_match_dir = ''#'/mnt/data0/neural_network_training_sets/noise_proj4/'
with mrcfile.open(hist_match_dir + 'actin_rotated%05d.mrc'%1) as mrc:
	hist_matcher = mrc.data

# check file names in pngs and Micrographs_bin4 directories
pngs = sorted(glob.glob('pngs/*.png'))
micrographs_bin4 = sorted(glob.glob('../Micrographs_bin4/*.mrc'))
pngs_trimmed = []
for i in range(0, len(pngs)):
	pngs_trimmed.append(pngs[i][5:-4])

skipped_file_names = []
for i in range(0, len(micrographs_bin4)):
	if(micrographs_bin4[i][20:-4] not in set(pngs_trimmed)):
		skipped_file_names.append(micrographs_bin4[i])

file_names = skipped_file_names[int(len(skipped_file_names)*(1.0/TOTAL_GPUS)*(INDEX-1)):int(len(skipped_file_names)*(1.0/TOTAL_GPUS)*INDEX)]
# Do picks!
print('Predicting on files from ' + file_names[0] + ' to ' + file_names[-1])
for i in tqdm(range(0, len(file_names))):
	try:
		run_pick_on_micrograph(file_names[i])
	except KeyboardInterrupt:
		print('Script hung on micrograph: ' + file_names[i] + ' ...')
		print('Proceeding to next micrograph.')
		

