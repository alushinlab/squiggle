#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# imports
import argparse; import sys
parser = argparse.ArgumentParser('Generate noisy and noiseless 2D projections of randomly oriented and translated MRC files.')
parser.add_argument('--input_mrc_dir', type=str, help='input directory containing MRC files to be rotated, translated, CTF-convolved, and projected')
parser.add_argument('--output_noise_dir', type=str, help='output directory to store noisy 2D projections')
parser.add_argument('--output_noiseless_dir', type=str, help='output directory to store noiseless 2D projections')
parser.add_argument('--numProjs', type=int, help='total number of projections to make')
parser.add_argument('--nProcs', type=int, help='total number of parallel threads to launch')
args = parser.parse_args()
print('')
if(args.input_mrc_dir == None or args.output_noise_dir == None or args.output_noiseless_dir == None):
	print('Please enter an input_mrc_dir, AND an output_noise_dir, AND an output_noiseless_dir')
	sys.exit('The preferred input style may be found with ./projection_generator.py -h')

if(args.numProjs == None):
	sys.exit('Please enter the number of projection images you would like with the --numProjs flag')

if(args.nProcs == None):
	print('No process number specified, using one thread')
	nProcs = 1
else:
	nProcs = args.nProcs

if(args.numProjs % nProcs != 0):
	print('The numProjs that you specified was not a multiple of nProcs.')
	print('Instead of %d 2D projections, this program will generate %d 2D projections'%(args.numProjs, args.numProjs/nProcs*nProcs))

folder = args.input_mrc_dir
noNoise_outputDir = args.output_noiseless_dir
noise_outputDir = args.output_noise_dir
TOTAL_NUM_TO_MAKE = args.numProjs

if(folder[-1] != '/'): folder = folder + '/'
if(noNoise_outputDir[-1] != '/'): noNoise_outputDir = noNoise_outputDir + '/'
if(noise_outputDir[-1] != '/'): noise_outputDir = noise_outputDir + '/'
print('The program will now generate %d 2D projections'%(args.numProjs/args.nProcs*args.nProcs))
################################################################################
# import of python packages
import numpy as np
from EMAN2 import *; from sparx import *; import mrcfile
import json; import glob
from multiprocessing import Pool
import os; from tqdm import tqdm
################################################################################
################################################################################
# import data
print('Opening MRC files')
actin_orig = []
file_names = sorted(os.listdir(folder))
for file_name in tqdm(file_names):
	if(file_name[-4:] == '.mrc' and file_name[:10] == 'bent_actin'):
		actin_orig.append(EMData(folder+file_name))
print('Opened %d MRC files'%(len(actin_orig)))
print('Beginning to generate projection images')
################################################################################
# image sizes
box_len = 256; BL = 64
################################################################################
def launch_parallel_process(thread_idx):
	index=num_per_proc*thread_idx
	for i in tqdm(range(0,num_per_proc)):
		local_random_state = np.random.RandomState(None)
		# First: randomly pick one of the actin mrc files that were loaded into actin_orig
		r0 = local_random_state.randint(0,len(actin_orig))
		r0_name = file_names[r0]
		rotated_actin = actin_orig[r0].copy() 
		# Rotation angles: azimuth, alt, phi, then Translations: tx, ty,tz
		r1, r2, r3 = int(local_random_state.random_sample()*360),90,int(local_random_state.random_sample()*360)
		r4, r5, r6 = local_random_state.normal(0, 20), local_random_state.normal(0, 20), 0
		t = Transform()
		t.set_params({'type':'eman','az':r1, 'alt':r2, 'phi':r3, 'tx':r4, 'ty':r5, 'tz':r6})
		rotated_actin.transform(t) # apply rotation and translation
		proj_eman = rotated_actin.project('standard',Transform()) # project
		proj_np = EMNumPy.em2numpy(proj_eman)
		center = 128
		# Save the target image
		target = proj_np[center-BL:center+BL, center-BL:center+BL]
		target = (target - np.mean(target)) / np.std(target) # normalize
		with mrcfile.new(noNoise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
			mrc.set_data(target.astype('float16'))
		
		# Generate noisy image
		r7 = local_random_state.uniform(1.0, 4.0) #defocus
		r8 = max(local_random_state.normal(0.030, 0.010),0) # noise amplitude
		target_eman = EMNumPy.numpy2em(target)
		target_eman.process_inplace('math.simulatectf',{'ampcont':7.0,'apix':4.32,'bfactor':0.0,'cs':0.01,'defocus':r7,'purectf':False,'voltage':300.0})
		target_noise = EMNumPy.em2numpy(target_eman)
		target_noise = (target_noise - np.mean(target_noise)) / np.std(target_noise) # normalize
		with mrcfile.new(noise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
			mrc.set_data(target_noise.astype('float16'))
		
		# Write text file with all random values and which actin model (curvature and phi) was chosen
		params = {'actin_name':r0_name,'actin_num':r0, 'alpha':r1, 'beta':r2, 'gamma':r3, 'tx':r4, 'ty':r5, 'tz':r6, 'defocus':r7,'noiseamp':r8,'iteration':i+num_per_proc*thread_idx}
		with open(noise_outputDir+'params_%02d.json'%(num_per_proc*thread_idx), 'a') as fp:
			data_to_write = json.dumps(params)
			fp.write(data_to_write + '\n')

################################################################################
# run in parallel
num_per_proc = TOTAL_NUM_TO_MAKE / nProcs
if __name__ == '__main__':
	p=Pool(nProcs)
	p.map(launch_parallel_process, range(0, nProcs))

################################################################################
# Now all files are written, combine all json files into one master json file
read_files = glob.glob(noise_outputDir+'params_*.json')
output_list = []
for f in read_files:
	for line in open(f, 'r'):
		output_list.append(json.loads(line))

#sort the json dictionaries based on iteration number
output_list = sorted(output_list, key=lambda i: i['iteration'])
for line in output_list:
	with open(noise_outputDir+'master_params.json', 'a') as fp:
		data_to_write = json.dumps(line)
		fp.write(data_to_write + '\n')

print('All projection images generated.')
