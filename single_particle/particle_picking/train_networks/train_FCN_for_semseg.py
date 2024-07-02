#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# import of python packages
print('Beginning to import packages...')
import numpy as np
import matplotlib.pyplot as plt
import keras
import mrcfile
import random
from tqdm import tqdm
from keras import layers
from keras.models import Model
import tensorflow as tf; import keras.backend as K
from scipy import interpolate; from scipy.ndimage import filters
from instance_segmentation_helper_methods import import_synth_data
from instance_segmentation_helper_methods import CCC
print('Packages finished importing. Data will now be loaded')
################################################################################
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
################################################################################
################################################################################
folder = '/mnt/data0/neural_network_training_sets/'
noise_folder = folder + 'noise_forSquig/'
noNoise_folder = folder + 'noNoise_forSquig/'

train, target = import_synth_data(noise_folder, noNoise_folder, 128, 0, 30000)
train = np.asarray(train, dtype='float16'); target = np.asarray(target,dtype='float16')

target = np.array((target[:,0], np.max(target[:,1:], axis=1)))
#add extra dimension at end because only one color channel
train = np.expand_dims(train, axis=-1)
target = np.moveaxis(target, 0, -1)

FRAC_VAL = int(train.shape[0] * 0.1)
val_train = train[:FRAC_VAL]
val_target = target[:FRAC_VAL]
train = train[FRAC_VAL:]
target = target[FRAC_VAL:]
print('All files loaded and parsed into training and validation sets.')
print('Beginning training')

################################################################################
######### The data should be imported; now create the model ####################
################################################################################
# Import the encoding layers of the DAE model
model_path = './800000training_CCC9887.h5'
autoencoder_three = keras.models.load_model(model_path, custom_objects={'CCC':CCC})

# Create new model. With first 15 layers the same as autoencoder_three
# Instantiate the model
trainable = False
input_img = layers.Input(shape=(train.shape[1:]))
# Make layers
full_training = False
x = layers.Conv2D(10, kernel_size=(3,3), padding='same', activation='relu',trainable=full_training)(input_img) #1
x = layers.Conv2D(10, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#2
x = layers.Conv2D(10, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#3
x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)#4
x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#5
x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#6
x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#7
x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#8
x = layers.Conv2D(12, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#9

# start Fully Convolutional Network portion
fcn_input64 = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(x)#15
fcn_64 = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(fcn_input64)
x = layers.MaxPooling2D(pool_size=(4,4), padding='same')(fcn_64)
x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(x)
x = layers.Dropout(0.05)(x) #3
x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(x)
x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(x)
fcn_16 = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(4,4), padding='same')(fcn_16)
x = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(x)
x = layers.Dropout(0.05)(x) #3
x = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(x)
fcn_4 = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(x)

fcn_4 = layers.UpSampling2D((16,16))(fcn_4)
fcn_4 = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(fcn_4)
fcn_16 = layers.UpSampling2D((4,4))(fcn_16)
fcn_16 = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(fcn_16)
fcn_64 = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(fcn_64)

x = layers.Concatenate(axis=-1)([fcn_4, fcn_16])
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(256, kernel_size=(1,1), activation='relu', padding='same')(x)
x = layers.Conv2D(2, kernel_size=(1,1), activation='sigmoid', padding='same')(x)

adam = keras.optimizers.Adam(lr=0.001)
FCN = Model(input_img, x)
FCN.compile(optimizer=adam, loss='binary_crossentropy', metrics=['mse'])
FCN.summary()
for i in range(0, 9):
	FCN.layers[i].set_weights(autoencoder_three.layers[i].get_weights())

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3, restore_best_weights=True)
history = FCN.fit(x=train, y=target, epochs=20, batch_size=32, verbose=1, validation_data = (val_train[1:], val_target[1:]), callbacks=[es])
model_save_name = './FCN_semantic_segmentation_squig.h5'
print('Model finished training.\nSaving model as ' + model_save_name)
FCN.save(model_save_name)
print('Model saved. Exiting.')

################################################################################
################################################################################

