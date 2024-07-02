#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
print('Loading python packages...')
################################################################################
# import of python packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import scipy.stats
import os; import sys
################################################################################
################################################################################
############################### Estimate Energies ##############################
################################################################################
################################################################################
ATP_combo_file_name = sorted(glob.glob('./squig_curvMeasures/combined_squig_and_nonsquig_distribution_long.csv'))[0]
noATP_combo_file_name = sorted(glob.glob('./noATPcntl_curvMeasures/combined_noATPcntl_squig_and_nonsquig_distribution_long.csv'))[0]

ATP_squig_file_name = sorted(glob.glob('./squig_curvMeasures/squig_curv_distribution_long.csv'))[0]
noATP_squig_file_name = sorted(glob.glob('./noATPcntl_curvMeasures/noATPcntl_squig_curv_distribution_long.csv'))[0]

ATP_nonsquig_file_name = sorted(glob.glob('./squig_curvMeasures/nonsquig_curv_distribution_long.csv'))[0]
noATP_nonsquig_file_name = sorted(glob.glob('./noATPcntl_curvMeasures/noATPcntl_nonsquig_curv_distribution_long.csv'))[0]

ATP_combo = np.abs(np.loadtxt(ATP_combo_file_name))*1000
noATP_combo = np.abs(np.loadtxt(noATP_combo_file_name))*1000
ATP_squig = np.abs(np.loadtxt(ATP_squig_file_name))*1000
noATP_squig = np.abs(np.loadtxt(noATP_squig_file_name))*1000
ATP_nonsquig = np.abs(np.loadtxt(ATP_nonsquig_file_name))*1000
noATP_nonsquig = np.abs(np.loadtxt(noATP_nonsquig_file_name))*1000


'''fig, ax = plt.subplots(3,2)
ax[0,0].hist(ATP_combo, bins=np.linspace(0,15,50), density=True)
ax[0,1].hist(noATP_combo, bins=np.linspace(0,15,50), density=True)
ax[1,0].hist(ATP_squig, bins=np.linspace(0,15,50), density=True)
ax[1,1].hist(noATP_squig, bins=np.linspace(0,15,50), density=True)
ax[2,0].hist(ATP_nonsquig, bins=np.linspace(0,15,50), density=True)
ax[2,1].hist(noATP_nonsquig, bins=np.linspace(0,15,50), density=True)
ax[0,0].set_ylim(0,0.6); ax[0,1].set_ylim(0,0.6); ax[1,0].set_ylim(0,0.6); ax[1,1].set_ylim(0,0.6); ax[2,0].set_ylim(0,0.6); ax[2,1].set_ylim(0,0.6)
ax[0,0].set_xlim(0,15); ax[0,1].set_xlim(0,15); ax[1,0].set_xlim(0,15); ax[1,1].set_xlim(0,15); ax[2,0].set_xlim(0,15); ax[2,1].set_xlim(0,15)
plt.show()
'''

fig, ax = plt.subplots(1,3)
_=ax[0].hist(ATP_combo, bins=np.linspace(0,15,50), density=True, alpha=0.6)
_=ax[0].hist(noATP_combo, bins=np.linspace(0,15,50), density=True, alpha=0.6)
_=ax[1].hist(ATP_squig, bins=np.linspace(0,15,50), density=True, alpha=0.6)
_=ax[1].hist(noATP_squig, bins=np.linspace(0,15,50), density=True, alpha=0.6)
_=ax[2].hist(ATP_nonsquig, bins=np.linspace(0,15,50), density=True, alpha=0.6)
_=ax[2].hist(noATP_nonsquig, bins=np.linspace(0,15,50), density=True, alpha=0.6)
_=ax[0].set_ylim(0,0.6); ax[1].set_ylim(0,0.6); ax[2].set_ylim(0,0.6); 
_=ax[0].set_xlim(0,15); ax[1].set_xlim(0,15); ax[2].set_xlim(0,15)
plt.show()


from scipy import integrate 
from scipy.interpolate import interp1d
def p_of_kappa(Lp, L, kappa):
	return np.exp(-0.5*Lp*L*kappa*kappa)

def p_of_kappa_2(Lp, L, kappa, a):
	return np.exp(a*-0.5*Lp*L*kappa*kappa)

def compute_loss(data, alpha_value, data2):
	x_axis = np.linspace(0.0, 15, 100)
	y_axis = p_of_kappa_2(9, 0.0500, x_axis, alpha_value) #assume Lp=9
	y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)
	
	f = interp1d(x_axis, y_axis_norm*y_axis)
	interpolated = f(data2)
	
	difference = data - interpolated
	return np.sum(difference * difference) / len(data)

def fit_curve(data):
	# Fit alpha value in front for each one for ADP
	holder = plt.hist(data, bins=np.linspace(0,15,61),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
	data_to_match = [(holder[1][1:] + holder[1][:-1])/2, holder[0]]
	plt.clf()
	
	losses = []
	for i in range(0, 10000):
		loss = compute_loss(data_to_match[1], np.linspace(0.1,6.0,10000)[i], data_to_match[0]) #values of alpha to sample between 0.1 and 8.0
		losses.append(loss)
	
	ALPHA = np.linspace(0.1,6,10000)[np.argmin(losses)] #values of alpha to sample between 0.1 and 8.0
	plt.plot(losses)
	plt.show()
	return ALPHA

alphas = [fit_curve(ATP_combo), fit_curve(noATP_combo), fit_curve(ATP_squig), fit_curve(noATP_squig), fit_curve(ATP_nonsquig), fit_curve(noATP_nonsquig)]

x_axis = np.linspace(0, 15, 61)
y_axes = []
y_axes.append(p_of_kappa(9, 0.0500, x_axis)) #assume Lp=9
y_axes.append(p_of_kappa_2(9, 0.0500, x_axis,alphas[0])) #assume Lp=9
y_axes.append(p_of_kappa_2(9, 0.0500, x_axis, alphas[1])) #assume Lp=9
y_axes.append(p_of_kappa(9, 0.0500, x_axis)) #assume Lp=9
y_axes.append(p_of_kappa_2(9, 0.0500, x_axis,alphas[2])) #assume Lp=9
y_axes.append(p_of_kappa_2(9, 0.0500, x_axis, alphas[3])) #assume Lp=9
y_axes.append(p_of_kappa(9, 0.0500, x_axis)) #assume Lp=9
y_axes.append(p_of_kappa_2(9, 0.0500, x_axis,alphas[4])) #assume Lp=9
y_axes.append(p_of_kappa_2(9, 0.0500, x_axis, alphas[5])) #assume Lp=9
y_axes = np.asarray(y_axes)


y_axes_norm = []
for i in range(0, 9):
	y_axes_norm.append(1.0/integrate.simps(y_axes[i], x_axis))

y_axes_norm = np.asarray(y_axes_norm)


#fig.set_size_inches(3,3)
# width of each bin is 0.25, so multiply each bar's y-axis by 25 (height*.25*100%) to get percentage
ATP_color = '#4682B4'
noATP_color = 'tomato'
max_x = 12
fig, ax = plt.subplots(1,3)
_=ax[0].hist(ATP_combo, bins=np.linspace(0,15,61), density=True, color=ATP_color, ec='black', alpha=0.6)
_=ax[0].hist(noATP_combo, bins=np.linspace(0,15,61), density=True, color=noATP_color, ec='black', alpha=0.6)
_=ax[1].hist(ATP_squig, bins=np.linspace(0,15,61), density=True, color=ATP_color, ec='black', alpha=0.6)
_=ax[1].hist(noATP_squig, bins=np.linspace(0,15,61), density=True, color=noATP_color, ec='black', alpha=0.6)
_=ax[2].hist(ATP_nonsquig, bins=np.linspace(0,15,61), density=True, color=ATP_color, ec='black', alpha=0.6)
_=ax[2].hist(noATP_nonsquig, bins=np.linspace(0,15,61), density=True, color=noATP_color, ec='black', alpha=0.6)
_=ax[0].set_ylim(0,1.25); ax[1].set_ylim(0,1.25); ax[2].set_ylim(0,1.25); 
_=ax[0].set_xlim(0,max_x); ax[1].set_xlim(0,max_x); ax[2].set_xlim(0,max_x)
_=ax[0].plot(x_axis, y_axes_norm[0]*y_axes[0], c='dimgray',linewidth=4)
_=ax[0].plot(x_axis, y_axes_norm[1]*y_axes[1], c=ATP_color,linewidth=4)
_=ax[0].plot(x_axis, y_axes_norm[2]*y_axes[2], c=noATP_color,linewidth=4)
_=ax[1].plot(x_axis, y_axes_norm[3]*y_axes[3], c='dimgray',linewidth=4)
_=ax[1].plot(x_axis, y_axes_norm[4]*y_axes[4], c=ATP_color,linewidth=4)
_=ax[1].plot(x_axis, y_axes_norm[5]*y_axes[5], c=noATP_color,linewidth=4)
_=ax[2].plot(x_axis, y_axes_norm[6]*y_axes[6], c='dimgray',linewidth=4)
_=ax[2].plot(x_axis, y_axes_norm[7]*y_axes[7], c=ATP_color,linewidth=4)
_=ax[2].plot(x_axis, y_axes_norm[8]*y_axes[8], c=noATP_color,linewidth=4)
plt.show()




# All filament segments
ATP_color = 'magenta'
noATP_color = 'deepskyblue'
max_x = 12
fig, ax = plt.subplots(1,1)
fig.set_size_inches(5,5)
_=ax.hist(ATP_combo, bins=np.linspace(0,15,61), density=True, color=ATP_color, ec='black', alpha=0.6)
_=ax.hist(noATP_combo, bins=np.linspace(0,15,61), density=True, color=noATP_color, ec='black', alpha=0.6)
_=ax.set_ylim(0,1.25);_=ax.set_xlim(0,max_x);
_=ax.plot(x_axis, y_axes_norm[0]*y_axes[0], c='dimgray',linewidth=4)
_=ax.plot(x_axis, y_axes_norm[1]*y_axes[1], c=ATP_color,linewidth=4)
_=ax.plot(x_axis, y_axes_norm[2]*y_axes[2], c=noATP_color,linewidth=4)
plt.show()

max_x = 12
fig, ax = plt.subplots(1,1)
fig.set_size_inches(5,5)
_=ax.hist(ATP_squig, bins=np.linspace(0,15,61), density=True, color=ATP_color, ec='black', alpha=0.6)
_=ax.hist(noATP_squig, bins=np.linspace(0,15,61), density=True, color=noATP_color, ec='black', alpha=0.6)
_=ax.set_ylim(0,1.25);_=ax.set_xlim(0,max_x);
_=ax.plot(x_axis, y_axes_norm[3]*y_axes[3], c='dimgray',linewidth=4)
_=ax.plot(x_axis, y_axes_norm[4]*y_axes[4], c=ATP_color,linewidth=4)
_=ax.plot(x_axis, y_axes_norm[5]*y_axes[5], c=noATP_color,linewidth=4)
plt.show()

max_x = 12
fig, ax = plt.subplots(1,1)
fig.set_size_inches(5,5)
_=ax.hist(ATP_nonsquig, bins=np.linspace(0,15,61), density=True, color=ATP_color, ec='black', alpha=0.6)
_=ax.hist(noATP_nonsquig, bins=np.linspace(0,15,61), density=True, color=noATP_color, ec='black', alpha=0.6)
_=ax.set_ylim(0,1.25);_=ax.set_xlim(0,max_x);
_=ax.plot(x_axis, y_axes_norm[6]*y_axes[6], c='dimgray',linewidth=4)
_=ax.plot(x_axis, y_axes_norm[7]*y_axes[7], c=ATP_color,linewidth=4)
_=ax.plot(x_axis, y_axes_norm[8]*y_axes[8], c=noATP_color,linewidth=4)
plt.show()




#Zoom ins
max_x = 12
fig, ax = plt.subplots(1,1)
fig.set_size_inches(5,5)
_=ax.hist(ATP_combo, bins=np.linspace(0,15,61), density=True, color=ATP_color, ec='black', alpha=0.6)
_=ax.hist(noATP_combo, bins=np.linspace(0,15,61), density=True, color=noATP_color, ec='black', alpha=0.6)
_=ax.set_ylim(0,0.15);_=ax.set_xlim(3,7);
_=ax.plot(x_axis, y_axes_norm[0]*y_axes[0], c='dimgray',linewidth=4)
_=ax.plot(x_axis, y_axes_norm[1]*y_axes[1], c=ATP_color,linewidth=4)
_=ax.plot(x_axis, y_axes_norm[2]*y_axes[2], c=noATP_color,linewidth=4)
plt.show()

max_x = 12
fig, ax = plt.subplots(1,1)
fig.set_size_inches(5,5)
_=ax.hist(ATP_squig, bins=np.linspace(0,15,61), density=True, color=ATP_color, ec='black', alpha=0.6)
_=ax.hist(noATP_squig, bins=np.linspace(0,15,61), density=True, color=noATP_color, ec='black', alpha=0.6)
_=ax.set_ylim(0,0.15);_=ax.set_xlim(3,7);
_=ax.plot(x_axis, y_axes_norm[3]*y_axes[3], c='dimgray',linewidth=4)
_=ax.plot(x_axis, y_axes_norm[4]*y_axes[4], c=ATP_color,linewidth=4)
_=ax.plot(x_axis, y_axes_norm[5]*y_axes[5], c=noATP_color,linewidth=4)
plt.show()


max_x = 12
fig, ax = plt.subplots(1,1)
fig.set_size_inches(5,5)
_=ax.hist(ATP_nonsquig, bins=np.linspace(0,15,61), density=True, color=ATP_color, ec='black', alpha=0.6)
_=ax.hist(noATP_nonsquig, bins=np.linspace(0,15,61), density=True, color=noATP_color, ec='black', alpha=0.6)
_=ax.set_ylim(0,0.15);_=ax.set_xlim(3,7);
_=ax.plot(x_axis, y_axes_norm[6]*y_axes[6], c='dimgray',linewidth=4)
_=ax.plot(x_axis, y_axes_norm[7]*y_axes[7], c=ATP_color,linewidth=4)
_=ax.plot(x_axis, y_axes_norm[8]*y_axes[8], c=noATP_color,linewidth=4)
plt.show()






