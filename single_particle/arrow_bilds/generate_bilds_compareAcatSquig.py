#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import string
from prody import *
################################################################################
def load_pdb(file_name):
	# Use ProDy to import PDB file
	p = parsePDB(file_name, subset='calpha')
	chids = sorted(set(p.getChids()))
	chains = []
	for chain_idx in chids:
		chains.append(p.select('chain ' + chain_idx).copy())
	
	# get the coordinates for each atom of each actin subunit
	coords = []
	for i in range(0,len(chains)):
		coords.append(chains[i].getCoords())
	
	# make each helix into a [num_chains x num_atoms_per_actin x 3] array
	coords = np.asarray(coords)
	return coords

# save coordinates as bild file
def save_bild_strand(sd_coords, dv, scalar, strand, o):
	tip = sd_coords + scalar*dv
	zStep = 60
	out=open(o, 'w')
	for i in range(0, sd_coords.shape[0]):
		for j in range(0, sd_coords.shape[1]):
			#write out marker entries for each residue pair
			if(i%2 == 0):
				if(strand == i%2):
					out.write('.transparency 0.0\n') # 1.0 is fully transparent, 0.0 is opaque
				#else:
				#	out.write('.transparency 0.5\n') # 1.0 is fully transparent, 0.0 is opaque
				out.write('.color 0.25 0.5 0.75\n') #close to steel blue
			else:
				if(strand == i%2):
					out.write('.transparency 0.0\n') # 1.0 is fully transparent, 0.0 is opaque
				#else:
				#	out.write('.transparency 0.3\n') # 1.0 is fully transparent, 0.0 is opaque
				out.write('.color 0.68 0.85 0.9\n') #light blue
			
			if(strand == i%2):
			   out.write(".sphere %.5f %.5f %.5f %.5f \n"%(sd_coords[i][j][0], sd_coords[i][j][1], sd_coords[i][j][2]+zStep*i, 4))
		
		#out.write('.color %.4f %.4f %.4f\n'%(0.5,0.5,0.5))
		if(strand == i%2):
			out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(sd_coords[i][1][0], sd_coords[i][1][1], sd_coords[i][1][2]+zStep*i, sd_coords[i][0][0], sd_coords[i][0][1], sd_coords[i][0][2]+zStep*i, 1.5))
			out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(sd_coords[i][0][0], sd_coords[i][0][1], sd_coords[i][0][2]+zStep*i, sd_coords[i][2][0], sd_coords[i][2][1], sd_coords[i][2][2]+zStep*i, 1.5))
			out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(sd_coords[i][2][0], sd_coords[i][2][1], sd_coords[i][2][2]+zStep*i, sd_coords[i][3][0], sd_coords[i][3][1], sd_coords[i][3][2]+zStep*i, 1.5))
	
	out.close()
	out=open(o[:-5]+'_arrows.bild', 'w')
	out.write('.transparency 0.0\n') # 1.0 is fully transparent, 0.0 is opaque
	arrow_colors = ['1.0 0.08 0.58', #SD1 yellow
						 '0.0 1.0 0.5', #SD2 0.0,0.5,1.0 cyan and green mix
						 '1.0 1.0 0.0', #SD3  yellow and deep pink blend 1.0,0.08,0.58
						 '0.0 0.0 1.0']   #SD4 0.0,0.0,1.0 blue
	for i in range(0, sd_coords.shape[0]):
		if(i%2 == strand):
			for j in range(0, sd_coords.shape[1]):
			   out.write('.color '+ arrow_colors[j]+'\n')
			   out.write(".arrow %.5f %.5f %.5f %.5f %.5f %.5f 2.0 4.0 0.75\n"%(sd_coords[i][j][0], sd_coords[i][j][1], sd_coords[i][j][2]+zStep*i, tip[i][j][0],tip[i][j][1],tip[i][j][2]+zStep*i))
	
	#write final line of xml file, is constant	
	out.close()	


def save_bild_each_staple(sd_coords, dv, scalar, strand, o):
	tip = sd_coords + scalar*dv
	for i in range(0, sd_coords.shape[0]):
		out=open(o+'_'+str(i).zfill(3)+'.bild', 'w')
		if(i==3):
			for j in range(0, sd_coords.shape[1]):
				#write out marker entries for each residue pair
				out.write('.transparency 0.5\n') # 1.0 is fully transparent, 0.0 is opaque
				out.write('.color 0.6 0.6 0.6\n') #close to steel blue 0.25 0.5 0.75
				out.write(".sphere %.5f %.5f %.5f %.5f \n"%(sd_coords[i][j][0], sd_coords[i][j][1], sd_coords[i][j][2], 4))
			
			#out.write('.color %.4f %.4f %.4f\n'%(0.5,0.5,0.5))
			out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(sd_coords[i][1][0], sd_coords[i][1][1], sd_coords[i][1][2], sd_coords[i][0][0], sd_coords[i][0][1], sd_coords[i][0][2], 1.5))
			out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(sd_coords[i][0][0], sd_coords[i][0][1], sd_coords[i][0][2], sd_coords[i][2][0], sd_coords[i][2][1], sd_coords[i][2][2], 1.5))
			out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(sd_coords[i][2][0], sd_coords[i][2][1], sd_coords[i][2][2], sd_coords[i][3][0], sd_coords[i][3][1], sd_coords[i][3][2], 1.5))
		
		
		out.write('.transparency 0.0\n') # 1.0 is fully transparent, 0.0 is opaque
		arrow_colors = ['1.0 0.08 0.58', #SD1 yellow
						 '0.0 1.0 0.5', #SD2 0.0,0.5,1.0 cyan and green mix
						 '1.0 1.0 0.0', #SD3  yellow and deep pink blend 1.0,0.08,0.58
						 '0.28 0.51 0.71']   #SD4 0.0,0.0,1.0 blue
		for j in range(0, sd_coords.shape[1]):
			if(j ==1 or j ==2):
				out.write('.color '+ arrow_colors[j]+'\n')
				out.write(".arrow %.5f %.5f %.5f %.5f %.5f %.5f 0.5 1.0 0.85\n"%(sd_coords[i][j][0], sd_coords[i][j][1], sd_coords[i][j][2], tip[i][j][0],tip[i][j][1],tip[i][j][2]))
	   
	   #write final line of xml file, is constant	
		out.close()	




################################################################################
bent_file_name = './aligned_to_chainM_skinny/avgsquig_alignedStrProtomers.pdb'
output_diff_vects1 = './aligned_to_chainM_skinny/avgsquig_diff_vects.npy'
output_diff_vects2 = './aligned_to_chainM_skinny/acat_diff_vects.npy'

# First load the pdb
bent = load_pdb(bent_file_name)
straight_file_name = './afterIsolde_pdbs/beforeIsolde_chainM.pdb'
straight = load_pdb(straight_file_name)

# Next compute subdomain centroids
# define subdomain amino acids
# oda 2019 and voth and pollard 2020; use this one
SD1 = np.concatenate((np.arange(0,28), np.arange(65,140), np.arange(333,370)))
#SD2 = np.arange(28,65)
SD2 = np.concatenate([np.arange(35,38),np.arange(55,68)]) - 7 # SD core only
SD3 = np.concatenate((np.arange(140,176), np.arange(265,333)))
SD4 = np.arange(176,265)

subdomain_struct = [SD1, SD2, SD3, SD4]

centroids = []
for i in range(0, len(straight)):
	SD1_centroid = straight[i][SD1].mean(axis=0)
	SD2_centroid = straight[i][SD2].mean(axis=0)
	SD3_centroid = straight[i][SD3].mean(axis=0)
	SD4_centroid = straight[i][SD4].mean(axis=0)
	centroid_i = [SD1_centroid, SD2_centroid, SD3_centroid, SD4_centroid]
	centroids.append(centroid_i)

centroids = np.repeat(np.asarray(centroids), 25, axis=0)

diff_vects = np.load(output_diff_vects1)
diff_vects = np.reshape(diff_vects, (25,371,3))
diff_vects_SD1 = np.expand_dims(diff_vects[:,SD1].mean(axis=1), axis=1)
diff_vects_SD2 = np.expand_dims(diff_vects[:,SD2].mean(axis=1), axis=1)
diff_vects_SD3 = np.expand_dims(diff_vects[:,SD3].mean(axis=1), axis=1)
diff_vects_SD4 = np.expand_dims(diff_vects[:,SD4].mean(axis=1), axis=1)
diff_vects_SDs = np.concatenate((diff_vects_SD1, diff_vects_SD2, diff_vects_SD3, diff_vects_SD4), axis=1)
#diff_vects_SDs = np.reshape(diff_vects_SDs, (25,4,3))

# Get average vector in each condition:
diff_vects_SDs = diff_vects_SDs[2:-2]
avg_vectors1 = np.mean(diff_vects_SDs, axis=0)

diff_vects = np.load(output_diff_vects2)
diff_vects = np.reshape(diff_vects, (25,371,3))
diff_vects_SD1 = np.expand_dims(diff_vects[:,SD1].mean(axis=1), axis=1)
diff_vects_SD2 = np.expand_dims(diff_vects[:,SD2].mean(axis=1), axis=1)
diff_vects_SD3 = np.expand_dims(diff_vects[:,SD3].mean(axis=1), axis=1)
diff_vects_SD4 = np.expand_dims(diff_vects[:,SD4].mean(axis=1), axis=1)
diff_vects_SDs = np.concatenate((diff_vects_SD1, diff_vects_SD2, diff_vects_SD3, diff_vects_SD4), axis=1)
#diff_vects_SDs = np.reshape(diff_vects_SDs, (25,4,3))

# Get average vector in each condition:
diff_vects_SDs = diff_vects_SDs[2:-2]
avg_vectors2 = np.mean(diff_vects_SDs, axis=0)

avg_vectors = avg_vectors2- avg_vectors1

def save_bild_average_staple(sd_coords, dv, scalar, strand, o):
	tip = sd_coords + scalar*dv
	for i in range(0, sd_coords.shape[0]):
		out=open(o+'_'+str(i).zfill(3)+'.bild', 'w')
		for j in range(0, sd_coords.shape[1]):
			#write out marker entries for each residue pair
			out.write('.transparency 0.5\n') # 1.0 is fully transparent, 0.0 is opaque
			out.write('.color 0.50 0.50 0.50\n') #close to steel blue 0.25 0.5 0.75
			out.write(".sphere %.5f %.5f %.5f %.5f \n"%(sd_coords[i][j][0], sd_coords[i][j][1], sd_coords[i][j][2], 4))
		
		#out.write('.color %.4f %.4f %.4f\n'%(0.5,0.5,0.5))
		out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(sd_coords[i][1][0], sd_coords[i][1][1], sd_coords[i][1][2], sd_coords[i][0][0], sd_coords[i][0][1], sd_coords[i][0][2], 1.5))
		out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(sd_coords[i][0][0], sd_coords[i][0][1], sd_coords[i][0][2], sd_coords[i][2][0], sd_coords[i][2][1], sd_coords[i][2][2], 1.5))
		out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(sd_coords[i][2][0], sd_coords[i][2][1], sd_coords[i][2][2], sd_coords[i][3][0], sd_coords[i][3][1], sd_coords[i][3][2], 1.5))
		
		out.write('.transparency 0.0\n') # 1.0 is fully transparent, 0.0 is opaque
		arrow_colors = ['1.0 0.08 0.58', #SD1 yellow
						 '0.0 1.0 0.5', #SD2 0.0,0.5,1.0 cyan and green mix
						 '1.0 1.0 0.0', #SD3  yellow and deep pink blend 1.0,0.08,0.58
						 '0.28 0.51 0.71']   #SD4 0.0,0.0,1.0 blue
		out.close()
		out=open(o+'_'+str(i).zfill(3)+'_arr.bild', 'w')
		print(sd_coords.shape)
		for j in range(0, sd_coords.shape[1]):
		   out.write('.color '+ arrow_colors[j]+'\n')
		   out.write(".arrow %.5f %.5f %.5f %.5f %.5f %.5f 1.0 2.0 0.75\n"%(sd_coords[i][j][0], sd_coords[i][j][1], sd_coords[i][j][2], tip[i][j][0],tip[i][j][1],tip[i][j][2]))
	   
		out.close()	

save_bild_average_staple(np.expand_dims(np.mean(centroids[2:-2],axis=0),axis=0), np.expand_dims(avg_vectors,axis=0), 15, 0, './temp_output')


























