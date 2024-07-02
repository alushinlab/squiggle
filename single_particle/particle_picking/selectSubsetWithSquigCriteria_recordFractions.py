#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
print('Loading python packages...')
################################################################################
# import of python packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os; import sys
print('Finished loading packages. \nComputing squiggle filament numbers...')
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

################################################################################
# Read in list of star files to be processed
file_names = sorted(glob.glob('../starFiles/*.star'))
print(len(file_names))

squigIdxs = []
squigHash = {}
total_squigs = 0
total_nonsquigs = 0
all_curv_measures_squig = []
all_curv_measures_nonsquig = []
for i in range(len(file_names)):#8300,8400):
	#print(i)
	num_lines = sum(1 for line in open(file_names[i]))#'../starFiles/dualmotor_ATPsquig_0%d_noDW.star'%i))
	if(num_lines > 14):
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


squig_curvs_flat = np.asarray([item for sublist in all_curv_measures_squig for item in sublist])
nonsquig_curvs_flat = np.asarray([item for sublist in all_curv_measures_nonsquig for item in sublist])

total_squigs
total_nonsquigs
total_filaments = total_squigs+total_nonsquigs

fig, ax = plt.subplots(1,2)
ax[0].hist(squig_curvs_flat, bins=np.linspace(-0.25,0.25,100), density=True)
ax[1].hist(nonsquig_curvs_flat, bins=np.linspace(-0.25,0.25,100), density=True)
ax[0].set_ylim(0,0.5)
ax[1].set_ylim(0,0.5)
plt.show()

print('The number of total filaments is: ' + str(total_filaments))
print('The number of those filaments that are squiggles is: ' + str(total_squigs) + '\t '+str(100.0*total_squigs/total_filaments) + '%')
print('The number of those filaments that are not squiggles is: ' + str(total_nonsquigs) + '\t '+str(100.0*total_nonsquigs/total_filaments) + '%')

print('Saving the curvature distributions...')
np.savetxt('squig_curv_distribution.csv', squig_curvs_flat, delimiter=',')
np.savetxt('nonsquig_curv_distribution.csv', nonsquig_curvs_flat, delimiter=',')
with open('squig_fraction.txt', "w") as text_file:
	text_file.write('The number of total filaments is: ' + str(total_filaments)+'\n')
	text_file.write('The number of those filaments that are squiggles is: ' + str(total_squigs) + '\t '+str(100.0*total_squigs/total_filaments) + '%'+'\n')
	text_file.write('The number of those filaments that are not squiggles is: ' + str(total_nonsquigs) + '\t '+str(100.0*total_nonsquigs/total_filaments) + '%'+'\n')

print('Finished.')






