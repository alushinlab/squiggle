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
box_num = 60000 # Number of boxes to make in total
################################################################################
# Take fft of a stack
def pw_spectra(stack):
	pw = []
	for i in range(0, len(stack)):
		pw.append(np.fft.fftshift(np.fft.fft2(stack[i])))
	return np.asarray(pw)

# Given a stack of white boxes, first, compute their DFFTs, then multiply by the 
# amplitude multiplier
def generate_pink_noise_box(white_boxes, amp_mult, amp_mult_std, amp_mult_rand):
	print('Calculating DFFTs of white noise boxes')
	white_box_spectra = pw_spectra(white_boxes)
	pink_noise_boxes = []
	print('Multiplying each white noise box FFT by the average power spectrum derived from real data.')
	for i in tqdm(range(0, len(white_box_spectra))):
		pnk_pw = np.multiply(amp_mult+amp_mult_rand[i]*amp_mult_std, white_box_spectra[i])
		pnk_pw_shift = np.fft.ifftshift(pnk_pw)
		pink_noise_box = np.fft.ifft2(pnk_pw_shift).real
		pink_noise_boxes.append((pink_noise_box - np.average(pink_noise_box)) / np.std(pink_noise_box))
	return np.asarray(pink_noise_boxes)

################################################################################
# Import data
print('Importing data...')
hole_picks_path = './range_of_defocus_holes.mrcs'
with mrcfile.open(hole_picks_path) as mrc:
	hole_picks = mrc.data

hole_picks = hole_picks.copy()
for i in range(0, len(hole_picks)):
	hole_picks[i] = (hole_picks[i] - np.mean(hole_picks[i]))/np.std(hole_picks[i])

extracted_noise_boxes = hole_picks.copy()
print('Data finished importing')

################################################################################
# Calculate average power spectrum of real data
print('Calculating DFFTs of real data')
pw_real_data = pw_spectra(extracted_noise_boxes)

print('Calculating average power spectrum of real data')
amp_mult = np.average(np.abs(pw_real_data),axis=0)
amp_mult_std = np.std(np.abs(pw_real_data),axis=0)
# Randomly increase or decrease the amount of noise
rands = np.random.normal(0,1,box_num)

fig, ax = plt.subplots(1,3)
ax[0].imshow(amp_mult, cmap=plt.cm.gray)
ax[1].imshow(amp_mult - amp_mult_std, cmap=plt.cm.gray)
ax[2].imshow(amp_mult + amp_mult_std, cmap=plt.cm.gray)
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].imshow(amp_mult, cmap=plt.cm.gray, vmin=amp_mult.min(), vmax=amp_mult.max())
ax[1].imshow(amp_mult_std, cmap=plt.cm.gray, vmin=amp_mult.min(), vmax=amp_mult.max())
plt.show()

with mrcfile.new('/mnt/data1/ayala/master_squig/wExampleNetworkInputs/pwrSpectrum.mrc', overwrite=True) as mrc:
	mrc.set_data(amp_mult.astype('float16'))

with mrcfile.new('/mnt/data1/ayala/master_squig/wExampleNetworkInputs/pwrSpectrumSTD.mrc', overwrite=True) as mrc:
	mrc.set_data(amp_mult_std.astype('float16'))



import sys
sys.exit()
################################################################################
################################################################################
# Generate synthetic noise boxes
# To prevent running out of memory, only load 10000 at a time
if box_num < 20000:
	print('Generating white noise boxes')
	white_noise_boxes = np.random.normal(0,0.25,[box_num,128,128])
	pnk_boxes = generate_pink_noise_box(white_noise_boxes, amp_mult, amp_mult_std, rands)

	outputDir = '/mnt/data0/neural_network_training_sets/pink_boxes_moreSTD/'
	print('Saving pink noise boxes as .mrcs file')
	with mrcfile.new(outputDir + 'pnk_noise_boxes.mrcs', overwrite=True) as mrc:
		mrc.set_data(pnk_boxes.astype('float16'))

else:
	for j in range(0, box_num/10000):
		print('Generating white noise boxes')
		white_noise_boxes = np.random.normal(0,1.0,[10000,128,128])
		pnk_boxes = generate_pink_noise_box(white_noise_boxes, amp_mult, amp_mult_std, rands[j*10000:(j+1)*10000])
	
		outputDir = '/mnt/data0/neural_network_training_sets/pink_boxes_moreSTD/'
		print('Saving pink noise boxes as .mrcs file')
		with mrcfile.new(outputDir + 'pnk_noise_boxes_10k_%03d.mrcs'%j, overwrite=True) as mrc:
			mrc.set_data(pnk_boxes.astype('float16'))



fig, ax = plt.subplots(2,3)
ax[0,0].imshow(extracted_noise_boxes[0], cmap=plt.cm.gray)
ax[0,1].imshow(extracted_noise_boxes[1], cmap=plt.cm.gray)
ax[0,2].imshow(extracted_noise_boxes[2], cmap=plt.cm.gray)
ax[1,0].imshow(pnk_boxes[0], cmap=plt.cm.gray)
ax[1,1].imshow(pnk_boxes[1], cmap=plt.cm.gray)
ax[1,2].imshow(pnk_boxes[2], cmap=plt.cm.gray)
plt.show()












semSegGTpath = '/mnt/data1/ayala/master_squig/wExampleNetworkInputs/actin_rotated00023.mrcs'
with mrcfile.open(semSegGTpath) as mrc:
	semSegGT = mrc.data

semSegGT_oneChannel = np.max(semSegGT[1:], axis=0)

with mrcfile.new('/mnt/data1/ayala/master_squig/wExampleNetworkInputs/semSegGT.mrc', overwrite=True) as mrc:
	mrc.set_data(semSegGT_oneChannel.astype('float16'))



