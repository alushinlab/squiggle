#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from prody import *
from scipy.interpolate import CubicSpline
import string; import sys
################################################################################
################################################################################
mid_file_name = '../pdbs/noATPcontrol/startingpdbs/str25subunits.pdb'
################################################################################
################################################################################
def define_axis_spline_curve(x,y,z,res=0.1):
	cs_x = CubicSpline(np.arange(0,len(x)), x, bc_type='natural')
	x_spline = cs_x(np.arange(-1, len(x), res))
	cs_y = CubicSpline(np.arange(0,len(y)), y, bc_type='natural')
	y_spline = cs_y(np.arange(-1, len(y), res))
	cs_z = CubicSpline(np.arange(0,len(z)), z, bc_type='natural')
	z_spline = cs_z(np.arange(-1, len(z), res))
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

def deriv_axis_spline_curve(x,y,z,res=0.1, order=1):
	cs_x = CubicSpline(np.arange(0,len(x)), x, bc_type='natural')
	x_spline = cs_x(np.arange(-1, len(x), res),order)
	cs_y = CubicSpline(np.arange(0,len(y)), y, bc_type='natural')
	y_spline = cs_y(np.arange(-1, len(y), res),order)
	cs_z = CubicSpline(np.arange(0,len(z)), z, bc_type='natural')
	z_spline = cs_z(np.arange(-1, len(z), res),order)
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

def get_coord_of_min_dist(pt, axis):
	pt_repeat = np.repeat(np.expand_dims(pt, axis=0), len(axis), axis=0)
	dists = np.linalg.norm(pt_repeat - axis, axis=1)
	min_dist_arg = np.argmin(dists)
	vect_to_center = axis[min_dist_arg] - pt
	vect_to_center = vect_to_center / np.linalg.norm(vect_to_center) * 15.76719916835962
	return vect_to_center

# get the argument along the axis spline that corresponds to the subunit
def get_h_of_min_dist(pt, axis):
	pt_repeat = np.repeat(np.expand_dims(pt, axis=0), len(axis), axis=0)
	dists = np.linalg.norm(pt_repeat - axis, axis=1)
	min_dist_arg = np.argmin(dists)
	return min_dist_arg

# Get phis by finding optimal angle based on distance to centroid of rotated vector
def get_phis_optAng(axis_pt, true_pos, rot_matrix):
	vect_to_rotate_to = axis_pt - true_pos#np.matmul(rot_matrix.T, true_pos - axis_pt)
	tang_vect = np.array([15.76719916835962,0,0])#np.matmul(rot_matrix, np.array([15.76719916835962,0,0]))
	best_dist = 10000
	samp_rate = np.radians(90)
	best_alpha = np.radians(0); 
	it_cnt = 0
	# gradient descent to get best phi
	while(best_dist > 0.000001 and samp_rate > np.radians(0.000005) and it_cnt < 100000):
		alpha = best_alpha
		decrease_sample_rate = False
		for i in range(-1,2):
			rotated_vect = rotate_yaw(alpha+i*samp_rate, tang_vect)
			new_vect = np.matmul(rot_matrix, rotated_vect) #+ axis_pt
			temp_dist = np.linalg.norm(new_vect - vect_to_rotate_to)
			if(temp_dist <= best_dist):
				best_dist = temp_dist
				best_alpha = alpha+i*samp_rate
				if(i == 0):
					decrease_sample_rate = True
		if(decrease_sample_rate):
			samp_rate = samp_rate / 2.0
		it_cnt = it_cnt + 1
	print(best_dist, np.linalg.norm(new_vect), np.linalg.norm(vect_to_rotate_to))
	print(vect_to_rotate_to, new_vect)
	return best_alpha, best_dist

def get_R_from_a_to_b(a, b):
	v = np.cross(a,b)
	skew_symm_cp = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
	a_b_dot = np.dot(a,b)
	R = np.eye(3) + skew_symm_cp + np.matmul(skew_symm_cp, skew_symm_cp) * 1.0/(1.0+a_b_dot)
	return R

################################################################################
def matt_parsePDB(file_name):
	with open(file_name, 'r') as f:
		lines = f.readlines()
	
	chains, coords = [] ,[]
	chainNum = 0
	for i in range(0, len(lines)):
		if(lines[i][:3]  == 'TER'):
			chainNum = chainNum + 1
		if(lines[i][:4] == 'ATOM' and lines[i][13:15] == 'CA'):
			chain,x,y,z = lines[i][21] + str(chainNum), float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])
			chains.append(chain); coords.append((x,y,z))
			#print lines[i][21] + str(chainNum),
	
	chains = np.asarray(chains)
	coords = np.asarray(coords)
	chain_idxs = list(set(chains))
	coords_per_chain = []
	for i in range(0, len(chain_idxs)):
		coords_per_chain.append(coords[np.argwhere(chains == chain_idxs[i])][:,0])
	
	return np.asarray(coords_per_chain)

################################################################################
################################################################################
# Load in PDB file and get chains
output_file_name = mid_file_name[:-4] + '.bild'
print('This program will generate a .bild file from the input PDB: ' + mid_file_name)
print('The output will be named: ' + output_file_name)
def load_coords(file_name):
	# make each helix into a [num_chains x num_atoms_per_actin x 3] array
	coords = matt_parsePDB(file_name)
	centroids = np.average(coords, axis=1)
	
	# sort centroids by z
	z_index = np.argsort(centroids[:,2])
	centroids = centroids[z_index]
	return centroids

centroids = load_coords(mid_file_name)

#centroids = centroids - np.average(centroids, axis=0)
avgs = avgs = (centroids[:-1] + centroids[1:]) / 2.0
#plt.plot(np.linalg.norm(centroids[:-1] - centroids[1:], axis=-1), marker='o')
#plt.show()

# plot the centroids in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(avgs[:,0], avgs[:,1], avgs[:,2], c='red')
_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()

# Save centroids as bild file
with open(output_file_name, 'w') as output:
   for i in range(0,len(centroids)):
   	if(i%2 == 0):
   		output.write('.color 0.000 0.000 1.000\n')
   	else:
   		output.write('.color 0.000 1.000 1.000\n')
   	
   	output.write('.sphere %.4f %.4f %.4f 20\n'%(centroids[i,0],centroids[i,1],centroids[i,2]))


sys.exit()






























################################################################################
################################################################################
# Now that the data is loaded in, define an initial axis
orig_axis = define_axis_spline_curve(avgs[:,0], avgs[:,1], avgs[:,2])
center_pointers = np.zeros((centroids.shape))
for i in range(0, len(centroids)):
	center_pointers[i] = get_coord_of_min_dist(centroids[i], orig_axis)

new_avgs = ((center_pointers+centroids)[:-1] + avgs + (center_pointers+centroids)[1:])/3.0

"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(orig_axis[:,0], orig_axis[:,1], orig_axis[:,2], c='red')
_=ax.scatter(avgs[:,0], avgs[:,1], avgs[:,2], c='blue', s=100)
_=ax.scatter(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], c='orange', s=100)
for i in range(0, len(center_pointers)):
	a=np.array([centroids[i], center_pointers[i] + centroids[i]])
	_=plt.plot(a[:,0],a[:,1],a[:,2], c='purple')

_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()
"""
################################################################################
# Iteratively update the central axis estimate
prev_avgs = new_avgs.copy()
for i in tqdm(range(0, 500)):
	temp_axis = define_axis_spline_curve(prev_avgs[:,0], prev_avgs[:,1], prev_avgs[:,2], 0.01)
	center_pointers = np.zeros((centroids.shape))
	for j in range(0, len(centroids)):
		center_pointers[j] = get_coord_of_min_dist(centroids[j], temp_axis)
	new_avgs = ((center_pointers+centroids)[:-1] + prev_avgs + (center_pointers+centroids)[1:])/3.0
	prev_avgs = new_avgs.copy()

# Final axis, for viewing. When doing more calculations, sample more finely
final_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.05)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(final_axis[:,0], final_axis[:,1], final_axis[:,2], c='red')
_=ax.scatter(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], c='orange', s=100)
for i in range(0, len(center_pointers)):
	a=np.array([centroids[i], center_pointers[i] + centroids[i]])
	_=plt.plot(a[:,0],a[:,1],a[:,2], c='purple')

_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()

# Plot, x(t), y(t), z(t)
fig, ax = plt.subplots(1,3)
ax[0].plot(final_axis[:,0])
ax[1].plot(final_axis[:,1])
ax[2].plot(final_axis[:,2])
plt.show()


################################################################################
# get distance between steps along central axis curve
final_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001)
time_steps = np.linalg.norm(final_axis[:-1] - final_axis[1:], axis=1)

# Now that we have the central axis as final_axis, get rise and twist
# Plot h'(t) 
hs = []
for i in range(0, len(centroids)):
	hs.append(get_h_of_min_dist(centroids[i], final_axis))

hs = np.asarray(hs)

from scipy.interpolate import CubicSpline
cs = CubicSpline(np.arange(0,len(hs)), hs, bc_type='natural')
h_of_t = cs(np.arange(0, len(hs), 0.001))[:-1] * time_steps/1.03
h_of_t_prime = cs(np.arange(0, len(hs), 0.001), 1)[:-1] / 1.03 * time_steps 
plt.plot(h_of_t_prime) # resize b/c pixel size is 1.03A/px
plt.scatter(hs, h_of_t_prime[hs])
plt.ylim(26.5,28.5)
plt.show()









