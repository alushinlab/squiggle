#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import mrcfile
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
print('Imports finished. Beginning script...')
################################################################################
def pw_spectra(stack):
	pw = []
	for i in tqdm(range(0, len(stack))):
		temp = np.fft.fftn(stack[i])
		pw.append(np.fft.fftshift(temp))
		#pw.append(np.fft.fftshift(np.fft.fftn(stack[i])))

	return np.asarray(pw)

def compute_avg_spectrum_per_micrograph(file_name, NUM_RANDOM_SAMPLES):
    tomo = []
    print('Opening tomo: %s'%file_name)
    output_ID = file_name.split('/')[-1][:-4]
    print(output_ID)
    with mrcfile.open(file_name, 'r') as mrc:
        tomo = mrc.data
    
    zdim, ydim, xdim = tomo.shape[0], tomo.shape[1], tomo.shape[2]
    # Load in 1000 random samples
    randsX = np.random.randint(int(xdim*0.25),int(xdim*0.75),NUM_RANDOM_SAMPLES)
    randsY = np.random.randint(int(ydim*0.25),int(ydim*0.75),NUM_RANDOM_SAMPLES)
    randsZ = np.random.randint(half_box_size,zdim-half_box_size,NUM_RANDOM_SAMPLES) # np.random.randint(int(zdim*0.25),int(zdim*0.75),NUM_RANDOM_SAMPLES)

    print('Cropping micrograph sections...')
    for i in tqdm(range(0, NUM_RANDOM_SAMPLES)):
        box = tomo[randsZ[i]-half_box_size:randsZ[i]+half_box_size, 
            randsY[i]-half_box_size:randsY[i]+half_box_size, 
            randsX[i]-half_box_size:randsX[i]+half_box_size]
		
        box = -1.0* box
        #box = np.rot90(box, k=1, axes=(1,2))
        box = (box - np.mean(box)) / np.std(box)
        with mrcfile.new('/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/chunking_tomos/%s_chunk_%s.mrc'%(str(output_ID),str(i).zfill(5)), overwrite=True) as mrc:
            mrc.set_data(np.asarray(box).astype('float32'))



################################################################################
#file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/*.mrc'))
file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/orig_tomos_bin4/combined_tomos/*.mrc'))
#bin2
#box_size = 96
#bin3
box_size = 48
half_box_size = int(box_size / 2)

for i in range(0, len(file_names)):
    compute_avg_spectrum_per_micrograph(file_names[i], 500)


