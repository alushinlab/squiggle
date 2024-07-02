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
	print(chids)
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

################################################################################
'''bent_file_name = './aligned_to_chainM/avgsquig_alignedStrProtomers.pdb'
output_diff_vects = './aligned_to_chainM/avgsquig_diff_vects.npy'

'''
bent_file_name = './aligned_to_chainM/acat_alignedStrProtomers.pdb'
output_diff_vects = './aligned_to_chainM/acat_diff_vects.npy'

'''
bent_file_name = './aligned_to_chainM/ctrlJ126_alignedStrProtomers.pdb'
output_diff_vects = './aligned_to_chainM/ctrlJ126_diff_vects.npy'

bent_file_name = './aligned_to_chainM/ctrlJ131_alignedStrProtomers.pdb'
output_diff_vects = './aligned_to_chainM/ctrlJ131_diff_vects.npy'
'''
# First load the pdb
bent = load_pdb(bent_file_name)

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
for i in range(0, len(bent)):
	SD1_centroid = bent[i][SD1].mean(axis=0)
	SD2_centroid = bent[i][SD2].mean(axis=0)
	SD3_centroid = bent[i][SD3].mean(axis=0)
	SD4_centroid = bent[i][SD4].mean(axis=0)
	centroid_i = [SD1_centroid, SD2_centroid, SD3_centroid, SD4_centroid]
	centroids.append(centroid_i)

centroids = np.asarray(centroids)

diff_vects = np.load(output_diff_vects)
diff_vects = np.reshape(diff_vects, (25,371,3))
diff_vects_SD1 = np.expand_dims(diff_vects[:,SD1].mean(axis=1), axis=1)
diff_vects_SD2 = np.expand_dims(diff_vects[:,SD2].mean(axis=1), axis=1)
diff_vects_SD3 = np.expand_dims(diff_vects[:,SD3].mean(axis=1), axis=1)
diff_vects_SD4 = np.expand_dims(diff_vects[:,SD4].mean(axis=1), axis=1)
diff_vects_SDs = np.concatenate((diff_vects_SD1, diff_vects_SD2, diff_vects_SD3, diff_vects_SD4), axis=1)
#diff_vects_SDs = np.reshape(diff_vects_SDs, (25,4,3))


################################################################################
# Do rose plots
straight_file_name = './afterIsolde_pdbs/beforeIsolde_chainM.pdb'
straight = load_pdb(straight_file_name)

straight_centroids = []
for i in range(0, len(straight)):
	SD1_centroid = straight[i][SD1].mean(axis=0)
	SD2_centroid = straight[i][SD2].mean(axis=0)
	SD3_centroid = straight[i][SD3].mean(axis=0)
	SD4_centroid = straight[i][SD4].mean(axis=0)
	centroid_i = [SD1_centroid, SD2_centroid, SD3_centroid, SD4_centroid]
	straight_centroids.append(centroid_i)

straight_centroids = np.asarray(straight_centroids)[0]

# computes rejection vector (orthogonal component to projection vector of a onto b)
# if a1 is the projection of a onto b, a2 = a - a1, this returns a1
def compute_vector_rejection(a,b):
	return a - (np.dot(a,b)/(np.dot(b,b)))*b

SD1_to_SD2_vect = straight_centroids[1] - straight_centroids[0]
SD1_to_SD3_vect = straight_centroids[2] - straight_centroids[0]

rejection_vector = -1.0*compute_vector_rejection(SD1_to_SD3_vect,SD1_to_SD2_vect)

y = SD1_to_SD2_vect / np.linalg.norm(SD1_to_SD2_vect)
x = rejection_vector / np.linalg.norm(rejection_vector)
z = np.cross(y,x) # might need to negate this

new_basis_diff_vects_SDs = []
for i in range(0, diff_vects_SDs.shape[0]):
	temp = []
	for j in range(0, diff_vects_SDs.shape[1]):
		vec_new = np.linalg.inv(np.column_stack((x,y,z))).dot(diff_vects_SDs[i,j])
		temp.append(vec_new)
	
	new_basis_diff_vects_SDs.append(temp)

new_basis_diff_vects_SDs = np.asarray(new_basis_diff_vects_SDs)
the_vects = new_basis_diff_vects_SDs[2:-2]

# Do SDs 1,4
fig = plt.figure(figsize=(8,8))
ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=False, frameon=False)
for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,0,0],the_vects[i,0,1], width=0.06, head_width=0.18,lw=0.5, alpha=0.5, length_includes_head=True, facecolor='magenta')

for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,3,0],the_vects[i,3,1], width=0.06, head_width=0.18,lw=0.5, alpha=0.5, length_includes_head=True, facecolor=(0.28,0.51,0.81))


ax3.grid(True)
limit = 2.75
ax3.set_rmax(limit)
ax2.set_ylim(-1.0*limit,limit)
ax2.set_xlim(-1.0*limit,limit)
ax3.set_rticks([0.5,1.5,2.5])
ax3.set_xticklabels('')
ax3.set_yticklabels('')
ax2.grid(False)
ax2.axis('off')
plt.show()


fig = plt.figure(figsize=(8,8))
ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=False, frameon=False)
for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,0,0],the_vects[i,0,2], width=0.06, head_width=0.18,lw=0.5, alpha=0.5, length_includes_head=True, facecolor='magenta')

for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,3,0],the_vects[i,3,2], width=0.06, head_width=0.18,lw=0.5, alpha=0.5, length_includes_head=True, facecolor=(0.28,0.51,0.81))


ax3.grid(True)
limit = 2.75
ax3.set_rmax(limit)
ax2.set_ylim(-1.0*limit,limit)
ax2.set_xlim(-1.0*limit,limit)
ax3.set_rticks([0.5,1.5,2.5])
ax3.set_xticklabels('')
ax3.set_yticklabels('')
ax2.grid(False)
ax2.axis('off')
plt.show()


# Do SDs 2,3
fig = plt.figure(figsize=(8,8))
ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=False, frameon=False)
for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,1,0],the_vects[i,1,1], width=0.06, head_width=0.18,lw=0.5, alpha=0.5, length_includes_head=True, facecolor='green')

for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,2,0],the_vects[i,2,1], width=0.06, head_width=0.18,lw=0.5, alpha=0.5, length_includes_head=True, facecolor='yellow')


ax3.grid(True)
limit = 2.75
ax3.set_rmax(limit)
ax2.set_ylim(-1.0*limit,limit)
ax2.set_xlim(-1.0*limit,limit)
ax3.set_rticks([0.5,1.5,2.5])
ax3.set_xticklabels('')
ax3.set_yticklabels('')
ax2.grid(False)
ax2.axis('off')
plt.show()


fig = plt.figure(figsize=(8,8))
ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=False, frameon=False)
for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,1,0],the_vects[i,1,2], width=0.06, head_width=0.18,lw=0.5, alpha=0.3, length_includes_head=True, facecolor='green')

for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,2,0],the_vects[i,2,2], width=0.06, head_width=0.18,lw=0.5, alpha=0.3, length_includes_head=True, facecolor='yellow')


ax3.grid(True)
limit = 2.75
ax3.set_rmax(limit)
ax2.set_ylim(-1.0*limit,limit)
ax2.set_xlim(-1.0*limit,limit)
ax3.set_rticks([0.5,1.5,2.5])
ax3.set_xticklabels('')
ax3.set_yticklabels('')
ax2.grid(False)
ax2.axis('off')
plt.show()




y = SD1_to_SD2_vect / np.linalg.norm(SD1_to_SD2_vect)
x = rejection_vector / np.linalg.norm(rejection_vector)
z = np.cross(y,x) # might need to negate this

def save_orthonormal_basis(centroid, x,y,z, o):
	out=open(o, 'w')
	out.write('.color %.4f %.4f %.4f\n'%(0.5,0.5,0.5))
	out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(centroid[0], centroid[1], centroid[2], centroid[0]+20*x[0], centroid[1]+20*x[1], centroid[2]+20*x[2], 1.5))
	out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(centroid[0], centroid[1], centroid[2], centroid[0]+20*y[0], centroid[1]+20*y[1], centroid[2]+20*y[2], 1.5))
	out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(centroid[0], centroid[1], centroid[2], centroid[0]+20*z[0], centroid[1]+20*z[1], centroid[2]+20*z[2], 1.5))
	out.close()


save_orthonormal_basis(straight_centroids[0], x,y,z, 'orthonormal.bild')


# Save acat SD2 arrow magnitudes, from top plane
vectors = np.asarray([the_vects[:,1,0], the_vects[:,1,2]]).T

angles = np.degrees(np.arctan2(vectors[:,1], vectors[:,0]))+360

# occupancies, bottom of the filament is the beginning of the list
occupancies = np.asarray([0.001302, 0.346354, 0.003906, 0.708333, 0.000000, 0.886719, 0.000000, 0.753906, 
	0.024740, 0.298177, 0.207031, 0.000000, 0.248698, 0.000000, 0.678385,0.000000, 
	0.899740, 0.000000, 1.000000, 0.002604, 0.733073])

plt.scatter(angles, occupancies[::-1])
plt.xlim(180,270);
plt.show()


np.savetxt('SD2angles_vs_occupancies.txt', [angles, occupancies[::-1]], delimiter=',')






