#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# import of python packages
import mrcfile
from tqdm import tqdm
import os
import subprocess
import scipy
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
################################################################################
def compute_sum(file_names):
	sums = []
	for i in tqdm(range(0, len(file_names))):
		with mrcfile.open(file_names[i]) as mrc:
			masked_vol = mrc.data
		
		sums.append(np.sum(masked_vol))
	
	sums = np.asarray(sums)
	return sums

def compute_sum_thresh(file_names, thresh):
	sums = []
	for i in tqdm(range(0, len(file_names))):
		with mrcfile.open(file_names[i]) as mrc:
			masked_vol = mrc.data
		
		sums.append(np.sum(masked_vol>thresh))
	
	sums = np.asarray(sums)*1.0
	return sums

def normalize(values):
	values_minus_bg = values - np.min(values)
	values_norm = values_minus_bg / np.max(values_minus_bg)
	return values_norm

################################################################################
file_names = sorted(glob.glob('../measurements/acat_3dva/acatVols_segmented/frame_0/*.mrc'))[2:-2]
sums = compute_sum(file_names)
sums_norm = normalize(sums)

sums_cumul = compute_sum_thresh(file_names, 0.75)
sums_norm_cumul = normalize(sums_cumul)

fig, ax = plt.subplots(1,2)
ax[0].plot(np.arange(0,len(sums_norm),2), sums_norm[::2], marker='o')
ax[0].plot(np.arange(1,len(sums_norm),2), sums_norm[1::2], marker='o')
ax[1].plot(np.arange(0,len(sums_norm),2), sums_norm_cumul[::2], marker='o')
ax[1].plot(np.arange(1,len(sums_norm),2), sums_norm_cumul[1::2], marker='o')
ax[0].set_xlabel('Subunit index'); ax[1].set_xlabel('Subunit index')
ax[0].set_ylabel('Occupancy (sum)'); ax[1].set_ylabel('Occupancy (binarized sum)')
plt.show()

################################################################################
################################################################################
all_file_names = []
for i in range(0, 20):
	file_names = sorted(glob.glob('../measurements/acat_3dva/acatVols_segmented/frame_%s/*.mrc'%str(i)))[2:-2]
	all_file_names.append(file_names)

all_sums = []
for i in range(0, 20):
	sums = compute_sum_thresh(all_file_names[i], 0.75)
	all_sums.append(sums)

all_sums = np.asarray(all_sums)
all_sums_flat = all_sums.flatten()
all_sums_norm = normalize(all_sums_flat)
all_sums_reshape = np.reshape(all_sums_norm, (20,21))

fig, ax = plt.subplots(1,2)
ax[0].plot(np.arange(0,len(sums_norm),2), all_sums_reshape[-1][::2], marker='o')
ax[0].plot(np.arange(1,len(sums_norm),2), all_sums_reshape[-1][1::2], marker='o')
ax[1].plot(np.arange(0,len(sums_norm),2), sums_norm_cumul[::2], marker='o')
ax[1].plot(np.arange(1,len(sums_norm),2), sums_norm_cumul[1::2], marker='o')
ax[0].set_xlabel('Subunit index'); ax[1].set_xlabel('Subunit index')
ax[0].set_ylabel('Occupancy (sum)'); ax[1].set_ylabel('Occupancy (binarized sum)')
plt.show()

################################################################################
# Do exporting
#occupancy = np.append(twists, '')
import pandas as pd
for i in range(0, len(all_sums_reshape)):
	dummy = np.append(all_sums_reshape[i], '')
	df_occupancy = pd.DataFrame(np.asarray([np.repeat(dummy[::2],2), np.repeat(dummy[1::2],2)]).T, columns=['occupancy pf2', 'occupancy pf1'])
	df_occupancy['occupancy pf2'].iloc[1::2] = ''
	df_occupancy['occupancy pf1'].iloc[::2] = ''
	df_occupancy.insert(0,'idx',range(-2,len(df_occupancy)-2))
	df_occupancy = df_occupancy.set_index('idx')
	df_occupancy.to_excel('occupancy_frame'+ str(i).zfill(2) + '.xlsx')
















