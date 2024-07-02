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
# Frequency comparisons
atp_dataset = [2849,43790]
noatp_dataset = [613, 16007]
acat_dataset = [3288, 85186]

obs = np.array([atp_dataset, noatp_dataset])
chi2_stat, p_val, dof, ex = scipy.stats.chi2_contingency(obs)

print('Chi-squared test of independence of variables in a contingency table')
print('Test between noATP and ATP squiggle:')
print('Chi-squared statistic: ' + str(chi2_stat))
print('Degrees of freedom: ' + str(dof))
print('P-value: ' + str(p_val))
print('')

obs = np.array([atp_dataset, acat_dataset])
chi2_stat, p_val, dof, ex = scipy.stats.chi2_contingency(obs)
print('Chi-squared test of independence of variables in a contingency table')
print('Test between ATP and acat squiggle:')
print('Chi-squared statistic: ' + str(chi2_stat))
print('Degrees of freedom: ' + str(dof))
print('P-value: ' + str(p_val))
print('')


obs = np.array([noatp_dataset, acat_dataset])
chi2_stat, p_val, dof, ex = scipy.stats.chi2_contingency(obs)
print('Chi-squared test of independence of variables in a contingency table')
print('Test between noATP and acat squiggle:')
print('Chi-squared statistic: ' + str(chi2_stat))
print('Degrees of freedom: ' + str(dof))
print('P-value: ' + str(p_val))
print('')


################################################################################
# Frequency comparisons
squig_combo_file_name = sorted(glob.glob('./squig_curvMeasures/combined_squig_and_nonsquig_distribution.csv'))[0]
acat_combo_file_name = sorted(glob.glob('./acat_curvMeasures/combined_acat_squig_and_nonsquig_distribution.csv'))[0]
noATP_combo_file_name = sorted(glob.glob('./noATPcntl_curvMeasures/combined_noATPcntl_squig_and_nonsquig_distribution.csv'))[0]

squig_combo = np.loadtxt(squig_combo_file_name)
acat_combo =  np.loadtxt(acat_combo_file_name)
noATP_combo = np.loadtxt(noATP_combo_file_name)


fig, ax = plt.subplots(1,3)
ax[0].hist(np.abs(squig_combo)*1000, bins=np.linspace(0,15,50), density=True)
ax[1].hist(np.abs(acat_combo)*1000, bins=np.linspace(0,15,50), density=True)
ax[2].hist(np.abs(noATP_combo)*1000, bins=np.linspace(0,15,50), density=True)
ax[0].set_ylim(0,0.6); ax[1].set_ylim(0,0.6); ax[2].set_ylim(0,0.6)
ax[0].set_xlim(0,15); ax[1].set_xlim(0,15); ax[2].set_xlim(0,15)
plt.show()


statistic, pvalue = scipy.stats.ks_2samp(np.abs(squig_combo),np.abs(noATP_combo))
print('KS test to test whether two datasets are drawn from the same distribution')
print('Test between noATP and +ATP:')
print('KS statistic: ' + str(statistic))
print('P-value: ' + str(pvalue))
print('')



statistic, pvalue = scipy.stats.ks_2samp(np.abs(squig_combo),np.abs(acat_combo))
print('KS test to test whether two datasets are drawn from the same distribution')
print('Test between ATP and acat squiggle:')
print('KS statistic: ' + str(statistic))
print('P-value: ' + str(pvalue))
print('')


statistic, pvalue = scipy.stats.ks_2samp(np.abs(acat_combo),np.abs(noATP_combo))
print('KS test to test whether two datasets are drawn from the same distribution')
print('Test between noATP and acat squiggle:')
print('KS statistic: ' + str(statistic))
print('P-value: ' + str(pvalue))
print('')

################################################################################
################################################################################
############################### Estimate Energies ##############################
################################################################################
################################################################################
squig_combo_file_name = sorted(glob.glob('./squig_curvMeasures/combined_squig_and_nonsquig_distribution.csv'))[0]
acat_combo_file_name = sorted(glob.glob('./acat_curvMeasures/combined_acat_squig_and_nonsquig_distribution.csv'))[0]
noATP_combo_file_name = sorted(glob.glob('./noATPcntl_curvMeasures/combined_noATPcntl_squig_and_nonsquig_distribution.csv'))[0]

squig_combo = np.abs(np.loadtxt(squig_combo_file_name))*1000
acat_combo =  np.abs(np.loadtxt(acat_combo_file_name))*1000
noATP_combo = np.abs(np.loadtxt(noATP_combo_file_name))*1000


fig, ax = plt.subplots(1,3)
ax[0].hist(squig_combo, bins=np.linspace(0,15,50), density=True)
ax[1].hist(acat_combo, bins=np.linspace(0,15,50), density=True)
ax[2].hist(noATP_combo, bins=np.linspace(0,15,50), density=True)
ax[0].set_ylim(0,0.6); ax[1].set_ylim(0,0.6); ax[2].set_ylim(0,0.6)
ax[0].set_xlim(0,15); ax[1].set_xlim(0,15); ax[2].set_xlim(0,15)
plt.show()


from scipy import integrate 
from scipy.interpolate import interp1d
def p_of_kappa(Lp, L, kappa):
	return np.exp(-0.5*Lp*L*kappa*kappa)

def p_of_kappa_2(Lp, L, kappa, a):
	return np.exp(a*-0.5*Lp*L*kappa*kappa)

def compute_loss(data, alpha_value):
	x_axis = np.linspace(0.0, 15, 100)
	y_axis = p_of_kappa_2(9, 0.0500, x_axis, alpha_value)
	y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)
	
	f = interp1d(x_axis, y_axis_norm*y_axis)
	interpolated = f(data_to_match[0][:40])
	
	difference = data[:40] - interpolated
	return np.sum(difference * difference) / 74

# Fit alpha value in front for each one for ADP
holder = plt.hist(squig_combo, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
data_to_match = [(holder[1][1:] + holder[1][:-1])/2, holder[0]]
plt.clf()

losses = []
for i in range(0, 500):
	loss = compute_loss(data_to_match[1], np.linspace(0.5,8.0,500)[i])
	losses.append(loss)

ALPHA = np.linspace(0.5,8,500)[np.argmin(losses)]
plt.plot(losses)
plt.show()

# Alpha value for squig is 3.82
x_axis = np.linspace(0, 15, 100)
y_axis = p_of_kappa(ALPHA, 0.0500, x_axis)
#y_axis = p_of_kappa_2(9, 0.0500, x_axis, 0.8003)
y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)

fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.hist(squig_combo, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)

_=ax.set_xlim(0,15)
_=ax.set_ylim(0,1.0)
plt.plot(x_axis, y_axis_norm*y_axis,color='#4682B4')
plt.plot(x_axis, 1.0/integrate.simps(p_of_kappa(8, 0.0500, x_axis), x_axis) * p_of_kappa(8, 0.0500, x_axis),color='gray')
plt.show()












