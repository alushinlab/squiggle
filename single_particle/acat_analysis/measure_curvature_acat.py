#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import string; import sys
from scipy.optimize import leastsq
import pandas as pd
################################################################################
from scipy.interpolate import UnivariateSpline
def define_axis_spline_curve(x,y,z,res=0.1):
	cs_x = UnivariateSpline(np.arange(0,len(x)), x, s=len(x)*0.1)
	x_spline = cs_x(np.arange(-1, len(x), res))
	cs_y = UnivariateSpline(np.arange(0,len(y)), y, s=len(y)*0.1)
	y_spline = cs_y(np.arange(-1, len(y), res))
	cs_z = UnivariateSpline(np.arange(0,len(z)), z, s=len(z)*0.1)
	z_spline = cs_z(np.arange(-1, len(z), res))
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

def deriv_axis_spline_curve(x,y,z,res=0.1,order=1):
	cs_x = UnivariateSpline(np.arange(0,len(x)), x, s=len(x)*0.1)
	x_spline = cs_x(np.arange(-1, len(x), res),order)
	cs_y = UnivariateSpline(np.arange(0,len(y)), y, s=len(y)*0.1)
	y_spline = cs_y(np.arange(-1, len(y), res),order)
	cs_z = UnivariateSpline(np.arange(0,len(z)), z, s=len(z)*0.1)
	z_spline = cs_z(np.arange(-1, len(z), res),order)
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

################################################################################
# Curvature measures
def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

def get_k_roll_average(new_avgs, win):
	# define r'(t)
	final_axis_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 1)
	# define r"(t) for later
	final_axis_second_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 1, 2)
	
	# Now, get curvature of the central axis
	r_prime = final_axis_deriv.copy()
	r_double_prime = final_axis_second_deriv.copy()
	k = np.linalg.norm(np.cross(r_prime, r_double_prime), axis=-1) / (np.linalg.norm(r_prime, axis=-1)**3)*10000
	
	roll_window = win
	k_roll = moving_average(k,roll_window)
	return k, k_roll

pc1_curv_holder = []
#frames = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
frames = [0,9,19]
for i in tqdm(range(0, len(frames))):
	file_name = 'fitAcat_frame_%s'%(str(frames[i]).zfill(3))
	centerSpline = pd.read_excel('../measurements/acat_3dva/updated_fitting_method/stitched_final/'+file_name+'_centerSplineSampling.xlsx', delimiter=',')
	centerSpline = centerSpline.to_numpy()[:,1:]
	k, k_roll = get_k_roll_average(centerSpline, 1000)
	k = k[24000:44000]
	k_roll = k_roll[24000:44000]
	pc1_curv_holder.append(k_roll)#k[24000:43000])

pc1_curv_holder = np.asarray(pc1_curv_holder)

for i in range(0,len(pc1_curv_holder)):
	plt.plot(pc1_curv_holder[i][::1000])

plt.show()

np.mean(pc1_curv_holder, axis=-1)
plt.plot(np.mean(pc1_curv_holder, axis=-1))
plt.show()

# Do for average one
file_name = 'job3_lp10_rigidfit'
centerSpline = pd.read_excel('../measurements/acat_upToFrame4/'+file_name+'_centerSplineSampling.xlsx', delimiter=',')
centerSpline = centerSpline.to_numpy()[:,1:]
k, k_roll, k_roll_mean = get_k_roll_average(centerSpline, 1000)



df_pc1 = pd.DataFrame(pc1_curv_holder[:,::1000].T*10000, columns=['pc1_frame000','pc1_frame001','pc1_frame002','pc1_frame003','pc1_frame004','pc1_frame005','pc1_frame006','pc1_frame007','pc1_frame008','pc1_frame009','pc1_frame010','pc1_frame011','pc1_frame012','pc1_frame013','pc1_frame014','pc1_frame015','pc1_frame016','pc1_frame017','pc1_frame018','pc1_frame019'])
#df_pc2 = pd.DataFrame(pc2_curv_holder[:,::1000].T*10000, columns=['pc2_frame000','pc2_frame004','pc2_frame009','pc2_frame014','pc2_frame019'])
df_avg = pd.DataFrame(k_roll[::1000].T*10000, columns=['avg_curvature'])

df_pc1.to_excel('../measurements/acat_3dva/updated_fitting_method/stitched/curvature_measure_3DVA_comp1_19frames.xlsx')
#df_pc2.to_excel('../measurements/squig_3dva/curvature_measure_3DVA_comp2_5frames.xlsx')
df_avg.to_excel('../measurements/acat_3dva/curvature_measure_upToFrame4.xlsx')


################################################################################
# Unused functions
def get_tau_roll_average(new_avgs, win):
	# define r'(t)
	final_axis_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 1)
	# define r"(t) for later
	final_axis_second_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 1, 2)
	# define r'''(t) 
	final_axis_third_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 1, 3)
	
	# Now, get curvature of the central axis
	r_prime = final_axis_deriv.copy()
	r_double_prime = final_axis_second_deriv.copy()
	r_triple_prime = final_axis_third_deriv.copy()
	
	tau = np.einsum('ij,ij->i',np.cross(r_prime, r_double_prime),r_triple_prime) / (np.linalg.norm(np.cross(r_prime,r_double_prime), axis=-1)**2)
	
	roll_window = win
	tau_roll = moving_average(tau,roll_window)
	return tau, tau_roll[24000:45000], tau_roll[24000:45000].mean()


pc2_curv_holder = []
frames = [0,4,9,14,19]
for i in range(0, len(frames)):
	file_name = 'masterSquigJ38_comp2Frame%s'%(str(frames[i]))
	centerSpline = pd.read_excel('../measurements/squig_3dva/stitched/'+file_name+'_centerSplineSampling.xlsx', delimiter=',')
	centerSpline = centerSpline.to_numpy()[:,1:]
	k, k_roll, k_roll_mean = get_k_roll_average(centerSpline, 1000)
	pc2_curv_holder.append(k_roll)#k[24000:43000])

pc2_curv_holder = np.asarray(pc2_curv_holder)

for i in range(0, len(pc2_curv_holder)):
	plt.plot(pc2_curv_holder[i][::1000])

plt.show()

np.mean(pc2_curv_holder*10000, axis=-1)

