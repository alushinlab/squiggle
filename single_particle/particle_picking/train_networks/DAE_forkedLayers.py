#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
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
################################################################################
import os
from os import path
os.environ["CUDA_VISIBLE_DEVICES"]="0"
################################################################################
# method to import synthetic data from files
def import_synth_data(noise_folder, noNoise_folder, box_length, NUM_IMGS_MIN, NUM_IMGS_MAX):
	noise_holder = []; noNoise_holder = []
	print('Loading files from ' + noise_folder)
	for i in tqdm(range(NUM_IMGS_MIN, NUM_IMGS_MAX)):
		file_name = 'actin_rotated%05d.mrc'%i
		noise_data = None; noNoise_data = None
		if(path.exists(noise_folder + file_name)):
			with mrcfile.open(noise_folder + file_name) as mrc:
				if(mrc.data.shape == (box_length,box_length)):
					noise_data = mrc.data
			with mrcfile.open(noNoise_folder + file_name) as mrc:
				if(mrc.data.shape == (box_length,box_length)):
					noNoise_data = mrc.data
			
			if(not np.isnan(noise_data).any() and not np.isnan(noNoise_data).any()): #doesn't have a nan
				noise_holder.append(noise_data.astype('float16'))
				noNoise_holder.append(noNoise_data.astype('float16'))
			
			else: # i.e. if mrc.data does have an nan, skip it and print a statement
				print('Training image number %d has at least one nan value. Skipping this image.'%i)
	
	return noise_holder, noNoise_holder

################################################################################
#https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
import keras.backend as K
def custom_loss(weights, outputs):
	def contractive_loss(y_pred, y_true):
		lam = 1e-2
		#print(len(autoencoder.layers))
		mse = K.mean(K.square(y_true - y_pred), axis=1)
		W = K.variable(value=weights)  # N x N_hidden
		W = K.transpose(W)  # N_hidden x N
		h = outputs
		dh = h * (1 - h)  # N_batch x N_hidden
		# N_batch x N_hidden * N_hidden x 1 = N_batch x 1
		contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
		return mse + contractive
	
	return contractive_loss

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
#folder = '/mnt/data1/Matt/computer_vision/VAE_squiggle/data_storage_bent_actin128/'
folder = '/mnt/data0/neural_network_training_sets/'
noise_folder = folder + 'squig_proj_pinkNoise/'#'noise_proj4/'
noNoise_folder = folder + 'squig_proj_noNoise/squig_proj_noNoise/'#'noNoise_proj4/'

train, target = import_synth_data(noise_folder, noNoise_folder, 128, 0, 780000)
train = np.asarray(train, dtype='float16'); target = np.asarray(target,dtype='float16')

#add extra dimension at end because only one color channel
train = np.expand_dims(train, axis=-1)
target = np.expand_dims(target, axis=-1)

FRAC_VAL = int(train.shape[0] * 0.1)
val_train = train[:FRAC_VAL]
val_target = target[:FRAC_VAL]
train = train[FRAC_VAL:]
target = target[FRAC_VAL:]
print('All files loaded and parsed into training and validation sets.')
print('Beginning training')

################################################################################
######### The data should be imported; now build the model #####################
################################################################################
# Define the model
def create_model_dense(training_data, full_training, lr):
	# Instantiate the model
	input_img = layers.Input(shape=(training_data.shape[1:]))
	# Make layers
	x = layers.Conv2D(10, kernel_size=(3,3), padding='same', activation='relu',trainable=full_training)(input_img) #1
	x = layers.Conv2D(16, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#3
	x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#5
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)#4
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#7
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#11
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#15
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#15
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#13
	x = layers.Conv2D(12, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#15
	x = layers.Flatten()(x) #16
	
	x = layers.Dense(512, activation='relu')(x) #17
	x = layers.Dropout(0.02)(x) #18
	x = layers.Dense(384, activation='relu')(x) #17
	x = layers.Dropout(0.02)(x) #18
	x = layers.Dense(256, activation='relu')(x) #21
	x = layers.Dropout(0.0)(x) #22
	x = layers.Dense(384, activation='relu')(x)#25
	x = layers.Dropout(0.02)(x)#26
	x = layers.Dense(512, activation='relu')(x)#25
	x = layers.Dropout(0.02)(x)#26
	x = layers.Dense(49152, activation='relu')(x)#27
	x = layers.Reshape((64,64,12))(x)#28
	
	x = layers.Conv2D(64, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#30
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#32
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#30
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#30
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#34
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#38
	x = layers.UpSampling2D((2,2))(x)#37
	x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#38
	x = layers.Conv2D(16, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#38
	x = layers.Conv2D(10, kernel_size=(3,3), activation='relu', padding='same')(x)
	decoded = layers.Conv2D(1, (1,1), activation='linear', padding='same',trainable=full_training)(x)#40
	
	# optimizer
	adam = keras.optimizers.Adam(lr=lr)
	# Compile model
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer=adam, loss=CCC, metrics=['mse'])
	autoencoder.summary()
	return autoencoder, x

################################################################################
# Handle model
def train_model(train_data, train_target):
	autoencoder, encoder = create_model_dense(train_data,True, 0.00005)
	es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=7, restore_best_weights=True)
	history = autoencoder.fit(x=train_data, y=train_target, epochs=200, batch_size=16, verbose=1, validation_data = (val_train[1:], val_target[1:]), callbacks=[es])
	return [autoencoder, history, encoder]

def continue_training(train_data, train_target, epochs, autoencoder_model):
	es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2, restore_best_weights=True)
	history = autoencoder_model.fit(x=train_data, y=train_target, epochs=epochs, batch_size=16, verbose=1, validation_data = (val_train[1:], val_target[1:]), callbacks=[es])
	return [autoencoder_model, history]

################################################################################
# train the three models. First do greedy training of outer layers then inner layers
# then train full model
autoencoder_three, history_three, encoder_three = train_model(train,target)

# continue training, if needed
#autoencoder_longer_train, history_longer_train = continue_training(train[:100000], target[:100000], 2, autoencoder_three)

# save the final model
model_save_name = './10000training_bentActin4squig_CCC9808_2.h5'
print('Model finished training.\nSaving model as ' + model_save_name)
autoencoder_three.save(model_save_name)


plot_history(history_three)

################################################################################
################################################################################
# check conv-dense autoencoder
check_num = 29
cm = plt.get_cmap('gray')#plt.cm.greens
predict_conv = autoencoder_three.predict(np.expand_dims(train[check_num].astype('float16'), axis=0))[0,:,:,0]
predict_dense = -1.0*autoencoder_three.predict(np.expand_dims(train[check_num].astype('float16'), axis=0))[0,:,:,0]
fig,ax = plt.subplots(2,2); _=ax[0,0].imshow(train[check_num,:,:,0].astype('float32'), cmap=cm); _=ax[0,1].imshow(target[check_num,:,:,0].astype('float32'), cmap=cm); _=ax[1,0].imshow(predict_conv.astype('float32'), cmap=cm);_=ax[1,1].imshow(predict_dense,cmap=cm);  #plt.show(block=False)

#encoder_model = Model(autoencoder_three.input, autoencoder_three.layers[21].output)
#encoded_pred = encoder_model.predict(np.expand_dims(train[check_num].astype('float16'), axis=0))[0]
#ax[0,2].plot(encoded_pred);
plt.show()



with mrcfile.new('noisy_training%02d.mrc'%i, overwrite=True) as mrc:
	mrc.set_data(train[check_num][:,:,0].astype('float32'))

with mrcfile.new('ground_truth%02d.mrc'%i, overwrite=True) as mrc:
	mrc.set_data(target[check_num][:,:,0].astype('float32'))

with mrcfile.new('predicted_synth%02d.mrc'%i, overwrite=True) as mrc:
	mrc.set_data(predict_dense.astype('float32'))


################################################################################
################################################################################
# if you want to see learning curves, plot this
def plot_history(history):
	p1, = plt.plot(history.history['loss']); p2, = plt.plot(history.history['val_loss']); 
	plt.title('Loss'); plt.ylim(ymin=-1); 
	plt.legend((p1,p2), ('Training Loss', 'Validation Loss'), loc='upper right', shadow=True)
	plt.show()

################################################################################
