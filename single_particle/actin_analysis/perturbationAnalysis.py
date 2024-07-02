#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
import chimera
from chimera import runCommand as rc
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import string; import sys
from string import ascii_uppercase
import Midas
from Midas.midas_text import makeCommand
from chimera import runCommand
from random import randint
from VolumeViewer import volume_list
from ColorZone import split_volume_by_color_zone
################################################################################
#os.chdir('pdbs/PC1')
simNum = 1
model='final_squiggle_rigidfit.pdb'
cryomap='final_squiggle_map.mrc'
subs = 25 
#rc('open ../realphabetized_fit_pdbs_3dva/masterSquigJ38_'+index+'_final.pdb')
#print('open ../final_pdb_rigidfit/'+model+'.pdb')
rc('open /mnt/data1/ayala/final_squiggle_paper/squiggle_dataset/final_pdb_rigidfit/'+model+'.pdb')


#for letter in ascii_uppercase:
#	sel = rc('select #0:.'+letter)
#	print(sel)
#	centroid = rc('define centroid')
#	mod_centroid = rc(
rc('split #0')

for i in range (1,subs+1):
	[rand1,rand2,rand3] = [randint(-1.5,1.5) for p in range(0,3)]
	sel = rc('select #0.'+str(i))
	#centroid = rc('define centroid')
	rc('move '+str(rand1)+','+str(rand2) +','+str(rand3)+' coord #0.'+str(i)+' mod #0.'+str(i))
	#mod_centroid = rc('define centroid')
	#rc('cs')
rc('select')
rc('rainbow model')

rc('open /mnt/data1/ayala/final_squiggle_paper/squiggle_dataset/final_map/'+cryomap+'.mrc')
colorchains = ''
for c in range (1,subs):
	colorchains= colorchains+'0.'+str(c)+','

	
rc('scolor #1 zone #0.'+colorchains[-1]+' range 15')

for v in volume_list():
	split_volume_by_color_zone(v)

for chain in range(1,subs+1):
	rc('fitmap #0.'+str(chain)+' #'+ str(chain+2)+' moveWholeMolecules false')

rc('combine #0.'+colorchains[-1]+' name simNum'+str(simNum))

modName = 'squig_avg_sim'+str(simNum)
modnum1 = str(subs+3)
modnum2 = str(subs+4)
modnum3 = str(subs+5)

rightorient = str(subs+6)
rightorientname = 'masterSquig_J36_final.pdb'
rc('write format pdb relative #1 #'+modnum1+' '+modName+'.pdb')
################################################################################
#make 3 copies of model for stitching

rc('combine #'+modnum1+' modelId #'+modnum2+' name simNum2')
rc('combine #'+modnum2+' modelId #'+modnum3+' name simNum3')

rc('open /mnt/data1/ayala/final_squiggle_paper/measure_all/pdbs/squig_avg/'+rightOrientname+'.pdb')
rc('matchmaker #'+modnum1+' #'+rightorient+' pairing ss')


rc('matchmaker #'+modnum1+':.A:.B:.C #'+modnum2+':.W:.X:.Y pairing ss')
rc('delete #'+modnum1+':.A:.B:.C')
rc('matchmaker #'+modnum2+':.A:.B:.C #'+modnum3+':.W:.X:.Y pairing ss')
rc('delete #'+modnum3+':.W:.X:.Y')
	
rc('write #'+modnum1+' squig_avg_sim1_final_A.pdb')
rc('write #'+modnum2+' squig_avg_sim1_final_B.pdb')
rc('write #'+modnum3+' squig_avg_sim1_final_C.pdb')

rc('stop now')


