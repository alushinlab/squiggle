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
from EMAN2 import *
from scipy import interpolate; from scipy.ndimage import filters
from skimage.morphology import skeletonize_3d; import scipy
from instance_segmentation_helper_methods import prune, eval_spline_energy, spline_energies, segment_image, import_synth_data,CCC
################################################################################
# Load both synthetic and real data
################################################################################
# Load synthetic dataset network was trained on
folder = '/mnt/data0/neural_network_training_sets/'
noise_folder = folder + 'multiple_filaments_noise/'
noNoise_folder = folder + 'multiple_filaments_noNoise/'

train, target = import_synth_data(noise_folder, noNoise_folder, 128, 0, 100)
train = np.asarray(train, dtype='float16'); target = np.asarray(target,dtype='float16')
target = np.array((target[:,0], np.max(target[:,1:], axis=1)))
#add extra dimension at end because only one color channel
train = np.expand_dims(train, axis=-1)
target = np.moveaxis(target, 0, -1)

# Load real data set
real_data_dir = '/mnt/data1/Matt/computer_vision/VAE_squiggle/synthetic_data/real_data/'
with mrcfile.open(real_data_dir + 'extractions_bent/beta_actin_Au_0015.mrcs') as mrc:
	real_data = mrc.data

real_data_shrunk = np.zeros((len(real_data),128,128))
for i in range(0, len(real_data)):
	eman2_format = EMNumPy.numpy2em(real_data[i])
	eman2_format.process_inplace('math.meanshrink', {'n':4})
	temp = EMNumPy.em2numpy(eman2_format)
	real_data_shrunk[i] = (temp - np.mean(temp)) / np.std(temp)

real_data = real_data_shrunk

################################################################################
# load trained Fully Convolutional Network for semantic segmentation
################################################################################
model_path = './FCN_semantic_segmentation.h5'
FCN = keras.models.load_model(model_path)
model_path = '../train_neural_network/750000training_CCC9749.h5'
autoencoder_three = keras.models.load_model(model_path, custom_objects={'CCC':CCC})

################################################################################
# Synthetic Data
check_num = 20
prediction = FCN.predict(np.expand_dims(train[check_num].astype('float16'), axis=0))[0]
plt.imshow(prediction[:,:,1].astype('float32'), cmap=plt.cm.gray)
plt.show()


#skel, distance = skimage.morphology.medial_axis(prediction[:,:,1].astype('float32')>0.8, return_distance=True)
skel = skeletonize_3d(prediction[:,:,1].astype('float32')>0.78)
skel[skel>0] = 1
fig, ax = plt.subplots(2,3)
ax[0,0].imshow(prediction[:,:,0].astype('float32'), cmap=plt.cm.gray)
ax[0,1].imshow(prediction[:,:,1].astype('float32'), cmap=plt.cm.gray)
ax[0,2].imshow(skel, cmap=plt.cm.magma)
ax[1,0].imshow(train[check_num,:,:,0].astype('float32'), cmap=plt.cm.gray)
ax[1,1].imshow(target[check_num,:,:,1].astype('float32'), cmap=plt.cm.gray)
end_detection = scipy.ndimage.filters.convolve(skel, np.array([[1,1,1],[1,10,1],[1,1,1]]),mode='constant')
ax[1,2].imshow(end_detection, cmap=plt.cm.magma)

# Try open active contours
# first identify potential end points
end_detection = scipy.ndimage.filters.convolve(skel, np.array([[1,1,1],[1,10,1],[1,1,1]]),mode='constant')
end_detection[end_detection != 11] = 0
pot_end_pts = np.argwhere(end_detection)
# then trim by removing those too close to three-way junctions
end_detection = scipy.ndimage.filters.convolve(skel, np.array([[1,1,1],[1,10,1],[1,1,1]]),mode='constant')
end_detection[end_detection != 13] = 0
triple_pts = np.argwhere(end_detection)
end_pts = prune(pot_end_pts, triple_pts)
ax[1,2].scatter(end_pts[:,1], end_pts[:,0])
ax[1,2].scatter(triple_pts[:,1], triple_pts[:,0])
plt.show()

################################################################################
################################################################################
masks = segment_image(prediction[:,:,1], end_pts, train[check_num,:,:,0])
if(len(end_pts) == 2):
	fig, ax = plt.subplots(1,2)
	ax[0].imshow(train[check_num,:,:,0].astype('float32'), cmap=plt.cm.gray)
	prediction_1 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks,axis=-1),axis=0))[0,:,:,0]
	ax[1].imshow(prediction_1.astype('float32'), cmap=plt.cm.gray)
	plt.show()
elif(len(end_pts) == 4):
	fig, ax = plt.subplots(2,3)
	ax[0,0].imshow(train[check_num,:,:,0].astype('float32'), cmap=plt.cm.gray)
	ax[0,1].imshow(masks[2], cmap=plt.cm.gray)
	ax[0,2].imshow(masks[3], cmap=plt.cm.gray)
	prediction_1 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[2],axis=-1),axis=0))[0,:,:,0]
	prediction_2 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[3],axis=-1),axis=0))[0,:,:,0]
	ax[1,0].imshow(np.max([prediction_1, prediction_2],axis=0).astype('float32'), cmap=plt.cm.gray)
	ax[1,1].imshow(prediction_1.astype('float32'), cmap=plt.cm.gray)
	ax[1,2].imshow(prediction_2.astype('float32'), cmap=plt.cm.gray)
	plt.show()
elif(len(end_pts) == 6):
	fig, ax = plt.subplots(2,4)
	ax[0,0].imshow(train[check_num,:,:,0].astype('float32'), cmap=plt.cm.gray)
	ax[0,1].imshow(masks[0], cmap=plt.cm.gray)
	ax[0,2].imshow(masks[1], cmap=plt.cm.gray)
	ax[0,3].imshow(masks[2], cmap=plt.cm.gray)
	prediction_1 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[0],axis=-1),axis=0))[0,:,:,0]
	prediction_2 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[1],axis=-1),axis=0))[0,:,:,0]
	prediction_3 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[2],axis=-1),axis=0))[0,:,:,0]
	ax[1,0].imshow(np.max([prediction_1, prediction_2, prediction_3],axis=0).astype('float32'), cmap=plt.cm.gray)
	ax[1,1].imshow(prediction_1.astype('float32'), cmap=plt.cm.gray)
	ax[1,2].imshow(prediction_2.astype('float32'), cmap=plt.cm.gray)
	ax[1,3].imshow(prediction_3.astype('float32'), cmap=plt.cm.gray)
	plt.show()





################################################################################
# Real Data
check_num = 37
real_img = np.expand_dims(np.expand_dims(real_data[check_num], axis=0), axis=-1).astype('float16')
prediction = FCN.predict(real_img)[0]
plt.imshow(prediction[:,:,1].astype('float32'), cmap=plt.cm.gray)
plt.show()

#skel, distance = skimage.morphology.medial_axis(prediction[:,:,1].astype('float32')>0.8, return_distance=True)
skel = skeletonize_3d(prediction[:,:,1].astype('float32')>0.78)
skel[skel>0] = 1
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(prediction[:,:,0].astype('float32'), cmap=plt.cm.gray)
ax[0,1].imshow(prediction[:,:,1].astype('float32'), cmap=plt.cm.gray)
ax[1,0].imshow(skel, cmap=plt.cm.magma)
end_detection = scipy.ndimage.filters.convolve(skel, np.array([[1,1,1],[1,10,1],[1,1,1]]),mode='constant')
ax[1,1].imshow(end_detection, cmap=plt.cm.magma)

# first identify potential end points
end_detection = scipy.ndimage.filters.convolve(skel, np.array([[1,1,1],[1,10,1],[1,1,1]]),mode='constant')
end_detection[end_detection != 11] = 0
pot_end_pts = np.argwhere(end_detection)
# then trim by removing those too close to three-way junctions
end_detection = scipy.ndimage.filters.convolve(skel, np.array([[1,1,1],[1,10,1],[1,1,1]]),mode='constant')
end_detection[end_detection != 13] = 0
triple_pts = np.argwhere(end_detection)
end_pts = prune(pot_end_pts, triple_pts)
ax[1,1].scatter(end_pts[:,1], end_pts[:,0])
ax[1,1].scatter(triple_pts[:,1], triple_pts[:,0])
plt.show()

################################################################################
################################################################################
masks = segment_image(prediction[:,:,1], end_pts, real_img[0,:,:,0])
if(len(end_pts) == 2 or len(end_pts) == 1):
	fig, ax = plt.subplots(1,2)
	ax[0].imshow(real_img[0,:,:,0].astype('float32'), cmap=plt.cm.gray)
	prediction_1 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks,axis=-1),axis=0))[0,:,:,0]
	ax[1].imshow(prediction_1.astype('float32'), cmap=plt.cm.gray)
	plt.show()
elif(len(end_pts) == 4):
	fig, ax = plt.subplots(2,3)
	ax[0,0].imshow(real_img[0,:,:,0].astype('float32'), cmap=plt.cm.gray)
	ax[0,1].imshow(masks[2], cmap=plt.cm.gray)
	ax[0,2].imshow(masks[3], cmap=plt.cm.gray)
	prediction_1 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[2],axis=-1),axis=0))[0,:,:,0]
	prediction_2 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[3],axis=-1),axis=0))[0,:,:,0]
	ax[1,0].imshow(np.max([prediction_1, prediction_2],axis=0).astype('float32'), cmap=plt.cm.gray)
	ax[1,1].imshow(prediction_1.astype('float32'), cmap=plt.cm.gray)
	ax[1,2].imshow(prediction_2.astype('float32'), cmap=plt.cm.gray)
	plt.show()
elif(len(end_pts) == 6):
	fig, ax = plt.subplots(2,4)
	ax[0,0].imshow(real_img[0,:,:,0].astype('float32'), cmap=plt.cm.gray)
	ax[0,1].imshow(masks[0], cmap=plt.cm.gray)
	ax[0,2].imshow(masks[1], cmap=plt.cm.gray)
	ax[0,3].imshow(masks[2], cmap=plt.cm.gray)
	prediction_1 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[0],axis=-1),axis=0))[0,:,:,0]
	prediction_2 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[1],axis=-1),axis=0))[0,:,:,0]
	prediction_3 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[2],axis=-1),axis=0))[0,:,:,0]
	ax[1,0].imshow(np.max([prediction_1, prediction_2, prediction_3],axis=0).astype('float32'), cmap=plt.cm.gray)
	ax[1,1].imshow(prediction_1.astype('float32'), cmap=plt.cm.gray)
	ax[1,2].imshow(prediction_2.astype('float32'), cmap=plt.cm.gray)
	ax[1,3].imshow(prediction_3.astype('float32'), cmap=plt.cm.gray)
	plt.show()

################################################################################
# Try to do full scale with whole real_data stack
real_data_orig = np.zeros((len(real_data),128,128)); real_data_predictions = np.zeros((len(real_data),128,128))
for i in tqdm(range(0, len(real_data))):
	check_num = i
	real_img = np.expand_dims(np.expand_dims(real_data[check_num], axis=0), axis=-1).astype('float16')
	prediction = FCN.predict(real_img)[0]
	
	skel = skeletonize_3d(prediction[:,:,1].astype('float32')>0.78)
	skel[skel>0] = 1
	
	# first identify potential end points
	end_detection = scipy.ndimage.filters.convolve(skel, np.array([[1,1,1],[1,10,1],[1,1,1]]),mode='constant')
	end_detection[end_detection != 11] = 0
	pot_end_pts = np.argwhere(end_detection)
	# then trim by removing those too close to three-way junctions
	end_detection = scipy.ndimage.filters.convolve(skel, np.array([[1,1,1],[1,10,1],[1,1,1]]),mode='constant')
	end_detection[end_detection != 13] = 0
	triple_pts = np.argwhere(end_detection)
	end_pts = prune(pot_end_pts, triple_pts)
	
	masks = segment_image(prediction[:,:,1], end_pts, real_img[0,:,:,0])
	real_data_orig[i] = real_img[0,:,:,0].astype('float32')
	if(len(end_pts) == 2 or len(end_pts) ==1):
		prediction_1 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks,axis=-1),axis=0))[0,:,:,0]
		real_data_predictions[i] = prediction_1
	elif(len(end_pts) == 4):
		prediction_1 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[2],axis=-1),axis=0))[0,:,:,0]
		prediction_2 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[3],axis=-1),axis=0))[0,:,:,0]
		real_data_predictions[i] = np.max([prediction_1, prediction_2],axis=0).astype('float32')
	elif(len(end_pts) == 6):
		prediction_1 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[0],axis=-1),axis=0))[0,:,:,0]
		prediction_2 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[1],axis=-1),axis=0))[0,:,:,0]
		prediction_3 = autoencoder_three.predict(np.expand_dims(np.expand_dims(masks[2],axis=-1),axis=0))[0,:,:,0]
		real_data_predictions[i] = np.max([prediction_1, prediction_2, prediction_3],axis=0).astype('float32')


with mrcfile.new('real_data_preds.mrc', overwrite=True) as mrc:
	mrc.set_data(real_data_predictions.astype('float32'))

with mrcfile.new('real_data.mrc', overwrite=True) as mrc:
	mrc.set_data(real_data_orig.astype('float32'))


################################################################################
# Try to full scale it with whole micrograph







