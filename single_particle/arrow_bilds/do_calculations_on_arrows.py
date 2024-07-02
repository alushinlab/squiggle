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

################################################################################
bent_file_name = './aligned_to_chainM/avgsquig_alignedStrProtomers.pdb'
output_diff_vects = './aligned_to_chainM/avgsquig_diff_vects.npy'

'''
bent_file_name = './aligned_to_chainM/acat_alignedStrProtomers.pdb'
output_diff_vects = './aligned_to_chainM/acat_diff_vects.npy'
'''

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
# plot SD2 magnitude
strand1 = diff_vects_SDs[2:-2][::2]
strand2 = diff_vects_SDs[2:-2][1::2]

plt.plot(np.arange(0,21)[::2],np.linalg.norm(strand1[:,1], axis=-1))
plt.plot(np.arange(0,21)[1::2],np.linalg.norm(strand2[:,1], axis=-1))
plt.show()

# plot SD1+SD4 magnitudes
strand1 = diff_vects_SDs[2:-2][::2]
strand2 = diff_vects_SDs[2:-2][1::2]

plt.plot(np.arange(0,21)[::2],np.linalg.norm(strand1[:,0], axis=-1)+np.linalg.norm(strand1[:,3], axis=-1))
plt.plot(np.arange(0,21)[1::2],np.linalg.norm(strand2[:,0], axis=-1)+np.linalg.norm(strand2[:,3], axis=-1))
plt.show()

# plot SD4 magnitudes
strand1 = diff_vects_SDs[2:-2][::2]
strand2 = diff_vects_SDs[2:-2][1::2]

plt.plot(np.arange(0,21)[::2],np.linalg.norm(strand1[:,3], axis=-1))
plt.plot(np.arange(0,21)[1::2],np.linalg.norm(strand2[:,3], axis=-1))
plt.show()




# compression SD1_4 index
strand1 = diff_vects_SDs[2:-2][::2]
strand2 = diff_vects_SDs[2:-2][1::2]

strand1_compression = np.einsum('ij,ij->i',strand1[:,0], strand1[:,3])
strand2_compression = np.einsum('ij,ij->i',strand2[:,0], strand2[:,3])


plt.plot(np.arange(0,21)[::2],strand1_compression)
plt.plot(np.arange(0,21)[1::2],strand2_compression)
plt.show()

################################################################################
# distance between SD1 and SD4
strand1 = centroids[2:-2][::2]
strand2 = centroids[2:-2][1::2]

plt.plot(np.arange(0,21)[::2],np.linalg.norm(strand1[:,0]-strand1[:,3], axis=-1))
plt.plot(np.arange(0,21)[1::2],np.linalg.norm(strand2[:,0]-strand2[:,3], axis=-1))
plt.show()



# dihedral between SD2-1-3-4
dihedrals = []
for i in range(0, len(centroids)):
   dihedrals.append(compute_dihedral([centroids[i][1], centroids[i][0], centroids[i][2], centroids[i][3]]))

dihedrals = np.asarray(dihedrals)
strand2 = dihedrals[2:-2][::2]
strand1 = dihedrals[2:-2][1::2]


plt.plot(np.arange(0,21)[::2],strand2)
plt.plot(np.arange(0,21)[1::2],strand1)
plt.show()


def compute_dihedral(p):
	p0=p[0]; p1=p[1]; p2=p[2]; p3=p[3]
	b0 = -1.0*(p1-p0)
	b1 = p2-p1
	b2 = p3-p2
	#normalize b1
	b1 = b1 / np.linalg.norm(b1)
	
	# v = projection of b0 onto plane perpendicular to p1 = b0 minus component aligning with b1
	v = b0 - np.dot(b0,b1)*b1
	w = b2 - np.dot(b2,b1)*b1
	x = np.dot(v,w)
	y = np.dot(np.cross(b1,v),w)
	return np.degrees(np.arctan2(y,x))


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

'''
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
bars = ax.bar(np.arctan2(the_vects[:,0,1],the_vects[:,0,0]),np.sqrt(the_vects[:,0,1]**2+the_vects[:,0,0]**2), width=np.pi/20, bottom=0.0, alpha=0.5, color='pink')
#plt.show()

#fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
bars = ax.bar(np.arctan2(the_vects[:,3,1],the_vects[:,3,0]),np.sqrt(the_vects[:,3,1]**2+the_vects[:,3,0]**2), width=np.pi/20, bottom=0.0, alpha=0.25)
plt.show()


labels = ['xy', 'xz', 'yz']
idxs = [[0,1], [0,2], [1,2]]
for i in range(0,4):
	for j in range(0,3):
		fig = plt.figure(figsize=(8,8))
		ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
		bars = ax.bar(np.arctan2(the_vects[:,i,idxs[j][0]],the_vects[:,i,idxs[j][1]]),np.sqrt(the_vects[:,i,idxs[j][0]]**2+the_vects[:,i,idxs[j][1]]**2), width=np.pi/20, bottom=0.0, alpha=0.5)
		ax.set_title('Cntrl J131; plane defined by %s; SD_%s'%(str(labels[j]), str(i)))
		plt.show()

'''


# Do SDs 1,4
fig = plt.figure(figsize=(8,8))
ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=False, frameon=False)
for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,0,0],the_vects[i,0,1], width=0.06, head_width=0.18,lw=0.5, alpha=0.2, length_includes_head=True, facecolor='magenta')

for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,3,0],the_vects[i,3,1], width=0.06, head_width=0.18,lw=0.5, alpha=0.2, length_includes_head=True, facecolor=(0.28,0.51,0.81))


ax3.grid(True)
limit = 2.75
ax3.set_rmax(limit)
ax2.set_ylim(-1.0*limit,limit)
ax2.set_xlim(-1.0*limit,limit)
ax2.grid(False)
ax2.axis('off')
plt.show()


fig = plt.figure(figsize=(8,8))
ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=False, frameon=False)
for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,0,0],the_vects[i,0,2], width=0.06, head_width=0.18,lw=0.5, alpha=0.2, length_includes_head=True, facecolor='magenta')

for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,3,0],the_vects[i,3,2], width=0.06, head_width=0.18,lw=0.5, alpha=0.2, length_includes_head=True, facecolor=(0.28,0.51,0.81))


ax3.grid(True)
limit = 2.75
ax3.set_rmax(limit)
ax2.set_ylim(-1.0*limit,limit)
ax2.set_xlim(-1.0*limit,limit)
ax2.grid(False)
ax2.axis('off')
plt.show()




# Do SDs 2,3
fig = plt.figure(figsize=(8,8))
ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=False, frameon=False)
for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,1,0],the_vects[i,1,1], width=0.06, head_width=0.18,lw=0.5, alpha=0.3, length_includes_head=True, facecolor='green')

for i in range(0, len(the_vects)):
	bars = ax2.arrow(0,0,the_vects[i,2,0],the_vects[i,2,1], width=0.06, head_width=0.18,lw=0.5, alpha=0.3, length_includes_head=True, facecolor='yellow')


ax3.grid(True)
limit = 2.75
ax3.set_rmax(limit)
ax2.set_ylim(-1.0*limit,limit)
ax2.set_xlim(-1.0*limit,limit)
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
ax2.grid(False)
ax2.axis('off')
plt.show()














