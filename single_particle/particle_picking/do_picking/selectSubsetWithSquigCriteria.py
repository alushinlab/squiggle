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

squigIdxs = []
squigHash = {}
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
			
			if(isSquig_posCurv and isSquig_negCurv):
				squigIdxs.append([file_names[i],j])
				if(file_names[i] not in squigHash.keys()):
					squigHash[file_names[i]] = []
				squigHash[file_names[i]].append(j)
				print('Micrograph ' + file_names[i] + ' filament number ' + str(j) + ' is a squiggle')

# Just do a quick double-check that all the values easily added to squigIdxs were
# added to the easier to parse squigHash
numKeys = 0
for key in squigHash.keys():
	numKeys = numKeys + len(squigHash[key])

numKeys
len(squigIdxs)

for key in squigHash.keys():
	squigHash[key] = np.asarray(squigHash[key]).astype('int')

for key in squigHash.keys():
	newStarName = key[:-5] + '_squigPick.star'
	TubeIds_toKeep = set(squigHash[key])
	header, old_body, labels = Split_header(key)
	new_body = []
	# Keep only one in every four initial picks
	for i in range(0, len(old_body)):
		if((i % 3) == 0):
			new_body.append(old_body[i])
	old_body = new_body
	new_body = []
	for i in range(0, len(old_body)):
		if(int(float(old_body[i].split()[4])) in TubeIds_toKeep):
			new_body.append(old_body[i])
	newStar(header, new_body, newStarName)
	













i=8319
vals = np.loadtxt('../starFiles/dualmotor_ATPsquig_0%d_noDW.star'%i,skiprows=14,usecols=(0,1,3,4))
vals = vals[vals[:,3] == 2]
plt.scatter(vals[:,0], vals[:,1],c=1.0/vals[:,2],cmap=plt.cm.coolwarm)
plt.ylim(0,4092)
plt.xlim(0,5760)
plt.show()


