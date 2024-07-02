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
from scipy.optimize import least_squares
print('Finished loading packages. \nMeasuring persistence lengths...')
################################################################################
################################################################################
################################################################################
# Function to take an array of measured curvatures and convert it into an array 
# where curvature values above a certain threshold are turned to 1, and values
# below a threshold get a value of -1. All other values get 0
def idxs_above_below_thresh(curvs, lowThresh, highThresh):
	curvs_bool_pos = curvs.copy()
	curvs_bool_neg = curvs.copy()
	curvs_bool_pos[curvs_bool_pos > highThresh] = 1
	curvs_bool_pos[curvs_bool_pos <= highThresh] = 0
	curvs_bool_neg[curvs_bool_neg < lowThresh] = -1
	curvs_bool_neg[curvs_bool_neg >= lowThresh] = 0
	return curvs_bool_pos + curvs_bool_neg

# Star file operations from Greg and Ayala
def Split_header(name):
	#find the last line that starts with _, this is the end of the header
	#also find star file column labels
	f=open(name,'r')
	lines=f.readlines()
	f.close()
	labels=[]
	for line in lines:
		if line.startswith('_'):
			labels.append(line)
			endheader=lines.index(line)
	#now split the list into 2
	endheader+=1
	print "Header ends at line %s"%str(endheader)
	header=lines[0:endheader]
	body=lines[endheader:]
	return header,body,labels

## Create new Micrograph file with new CTF values
def newStar(header,newBody,newStarFileName):
    print str(newStarFileName)
    bod = '\t'.join([str(x) for x in newBody])
    #print(bod)
    #print header
    #print ''.join(header)
    with open(newStarFileName, 'w') as newstar:
        newstar.write(''.join(header))
        for bod in newBody:
        	newstar.write(''.join(bod))
		newstar.write('')

def calc_wlc_length(points, kT, lp_guess=10.0):
    """Estimate the persistence length and contour length of a filament given a set of 2D points"""
    # Calculate the tangent vectors at each point along the filament
    tangents = np.diff(points, axis=0)
    # Calculate the lengths of the tangent vectors
    lengths = np.sqrt(np.sum(tangents**2, axis=1))
    # Normalize the tangent vectors
    tangents /= lengths[:, np.newaxis]
    # Calculate the cosine of the angle between tangent vectors
    cos_angles = np.dot(tangents[:-1], tangents[1:].T)
    
    def residuals(lp):
	 	 expected_cos_angles = np.exp(-lengths[:-1] / lp)
	 	 return cos_angles - expected_cos_angles
    # Use nonlinear least-squares to estimate the persistence length
    lp_fit = least_squares(residuals, lp_guess).x[0]
    # Calculate the contour length of the filament
    contour_length = np.sum(lengths)
    # Calculate the persistence length from the fitted value
    persistence_length = lp_fit * contour_length
    return persistence_length, contour_length

def calc_wlc_length2(points, kT):
    """Estimate the persistence length and contour length of a filament given a set of 2D points"""
    # Calculate the tangent vectors at each point along the filament
    tangents = np.diff(points, axis=0)
    # Calculate the lengths of the tangent vectors
    lengths = np.sqrt(np.sum(tangents**2, axis=1))
    # Normalize the tangent vectors
    tangents /= lengths[:, np.newaxis]
    # Calculate the cosine of the angle between tangent vectors
    cos_angles = np.dot(tangents[:-1], tangents[1:].T)
    # Calculate the Kratky-Porod persistence length
    persistence_length = -np.mean(lengths[:-1] * np.log(cos_angles)) / kT
    # Calculate the contour length of the filament
    contour_length = np.sum(lengths)
    return persistence_length, contour_length

def calc_pers_length_SPRING(points):
	filament = points*1.08
	end_to_end_length = 1e-10 * np.linalg.norm(filament[0] - filament[-1])
	contour_length = 1e-10 * np.sum(np.linalg.norm(filament[:-1] - filament[1:], axis=1))
	
	ratio = end_to_end_length / contour_length
	if(np.isnan(ratio)):
		return -1.0*1e6
	if ratio <= 0.4:#1 / np.sqrt(2):
		pers_length = -1.0
	elif 0.4 < ratio and ratio < 1.0:
		pers_length = 1 / ((-np.log(2 * (end_to_end_length / contour_length) ** 2 - 1) / contour_length))
		pers_length = np.min([1.0, pers_length])
	elif ratio >= 1.0:
		pers_length = 1.0
	
	return pers_length*1e6



################################################################################
# Read in list of star files to be processed
file_names = sorted(glob.glob('./squiggle_picks/*starFiles/*[!squigPick].star'))

squigIdxs = []
LP_list = []
squigHash = {}
total_squigs = 0
total_nonsquigs = 0
all_curv_measures_squig = []
all_curv_measures_nonsquig = []
for i in tqdm(range(len(file_names))):#8300,8400):
	#print(i)
	num_lines = sum(1 for line in open(file_names[i]))#'../starFiles/dualmotor_ATPsquig_0%d_noDW.star'%i))
	if(num_lines > 14): # 14 is what should be used for all filaments
		vals = np.loadtxt(file_names[i], skiprows=14, usecols=(0,1,3,4))#'../starFiles/dualmotor_ATPsquig_0%d_noDW.star'%i,skiprows=14,usecols=(0,1,3,4))
		for j in set(vals[:,3]):# vals[:,3} is the list of filament IDs
			thisFilament = vals[vals[:,3]==j]
			curv_measures = 1.0/thisFilament[:,2]
			curv_measures_bool = idxs_above_below_thresh(curv_measures, -0.0015, 0.0015)
			#print(thisFilament.shape)
			pos_cnt=0; neg_cnt = 0; it = 1; isSquig_posCurv = False; isSquig_negCurv = False
			while((not isSquig_posCurv or not isSquig_negCurv) and it < len(curv_measures_bool)):
				#print(it, pos_cnt, neg_cnt, isSquig_posCurv, isSquig_negCurv)
				if(curv_measures_bool[it] == 1):
					pos_cnt = pos_cnt + 1
				if(curv_measures_bool[it] == -1):
					neg_cnt = neg_cnt + 1
				if(curv_measures_bool[it] == 0 and curv_measures_bool[it-1] == 1):
					pos_cnt = 0
				if(curv_measures_bool[it] == 0 and curv_measures_bool[it-1] == -1):
					neg_cnt = 0
				if (pos_cnt >= 15):
					isSquig_posCurv = True
				if (neg_cnt >= 15): 
					isSquig_negCurv = True
				it = it+1
			
			if(isSquig_posCurv and isSquig_negCurv): # is a squiggle
				squigIdxs.append([file_names[i],j])
				if(file_names[i] not in squigHash.keys()):
					squigHash[file_names[i]] = []
				squigHash[file_names[i]].append(j)
				#print('Micrograph ' + file_names[i] + ' filament number ' + str(j) + ' is a squiggle')
				total_squigs = total_squigs + 1
				all_curv_measures_squig.append(curv_measures)
			else: # is not a squiggle
				total_nonsquigs = total_nonsquigs + 1
				all_curv_measures_nonsquig.append(curv_measures)
			
			LP_list.append(calc_pers_length_SPRING(thisFilament[:,:2]))

LP_list = np.asarray(LP_list)
retained_LP = LP_list[LP_list > 0.0]
retained_LP = retained_LP[retained_LP < 15000]
np.median(retained_LP)


plt.hist(retained_LP, bins=np.arange(-2,50))
#plt.xscale('log')
plt.show()


squig_curvs_flat = np.asarray([item for sublist in all_curv_measures_squig for item in sublist])
nonsquig_curvs_flat = np.asarray([item for sublist in all_curv_measures_nonsquig for item in sublist])



def calc_wlc_length(points, kT, lp_guess=10.0):
    """Estimate the persistence length and contour length of a filament given a set of 2D points"""
    # Calculate the tangent vectors at each point along the filament
    tangents = np.diff(points, axis=0)
    # Calculate the lengths of the tangent vectors
    lengths = np.sqrt(np.sum(tangents**2, axis=1))
    # Normalize the tangent vectors
    tangents /= lengths[:, np.newaxis]
    # Calculate the cosine of the angle between tangent vectors
    cos_angles = np.dot(tangents[:-1], tangents[1:].T)
    
    def residuals(lp):
	 	 expected_cos_angles = np.exp(-lengths[:-1] / lp)
	 	 return cos_angles - expected_cos_angles
    # Use nonlinear least-squares to estimate the persistence length
    lp_fit = least_squares(residuals, lp_guess).x[0]
    # Calculate the contour length of the filament
    contour_length = np.sum(lengths)
    # Calculate the persistence length from the fitted value
    persistence_length = lp_fit * contour_length
    return persistence_length, contour_length


filament = np.asarray([[3591.561, 379.759],[ 3588.032,391.356],[3584.883, 403.143],[3582.093, 415.106], [3579.636, 427.236],
	[3577.491, 439.519], [3575.633, 451.944], [3574.039, 464.499],[3572.686, 477.173],[3571.55 , 489.954],[3570.608, 502.829],
	[3569.838, 515.788],[3569.214, 528.818],[3568.715, 541.908],[3568.316, 555.046],[3567.995, 568.221],[3567.727, 581.419],
	[3567.49, 594.631],[3567.261, 607.843],[3567.016, 621.044],[3566.731, 634.223],[3566.383, 647.368],[3565.949, 660.467],
	[3565.406, 673.508],[3564.73, 686.479]])



"""Estimate the persistence length and contour length of a filament given a set of 2D points"""
tangents = np.diff(filament, axis=0)
lengths = np.sqrt(np.sum(tangents**2, axis=1))
tangents /= lengths[:, np.newaxis]
cos_angles = np.dot(tangents[:-1], tangents[1:].T)
    
def residuals(lp):
	expected_cos_angles = np.exp(-lengths[:-1] / lp)
	return cos_angles.ravel() - expected_cos_angles
lp_fit = least_squares(residuals, 10).x[0]
contour_length = np.sum(lengths)
persistence_length = lp_fit * contour_length



return persistence_length, contour_length

filament = filament*1.03
end_to_end_length = 1e-10 * np.linalg.norm(filament[0] - filament[-1])
contour_length = 1e-10 * np.sum(np.linalg.norm(filament[:-1] - filament[1:], axis=1))

ratio = end_to_end_length / contour_length
if ratio <= 1 / np.sqrt(2):
	pers_length = 0.0
elif 1 / np.sqrt(2) < ratio and ratio < 1.0:
	pers_length = 1 / ((-np.log(2 * (end_to_end_length / contour_length) ** 2 - 1) / contour_length))
	pers_length = np.min([1.0, pers_length])
elif ratio >= 1.0:
	pers_length = 1.0



return pers_length










