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
frame = 'frame19'
#map_file_name = '../pdbs/acat_upToFrame4/job3_lp10_map.mrc'
map_file_name = '../pdbs/acat_3dva/frame_019.mrc'
#model_file_name = '../pdbs/acat_upToFrame4/job3_lp10_rigidfit_acat.pdb'
model_file_name = '../pdbs/acat_3dva/acat_3dva_'+frame+'.pdb'
subs = 25 
rc('open ' + map_file_name)
rc('open ' + model_file_name)

colorchains = ''
for i in range(0,26):
	colorchains= colorchains+':.'+ascii_uppercase[i]

print(colorchains)

rc('color green #1'+colorchains)
rc('scolor #0 zone #1'+colorchains + ' range 8')
for v in volume_list():
	split_volume_by_color_zone(v)

rc('vop subtract #0 #3')
rc('delete #1'+colorchains)
rc('rainbow chain #1')
rc('scolor #4 zone #1 range 20')
rc('close #0,2,3')
rc('volume #4 level 0.75 step 1') 

for v in volume_list():
	split_volume_by_color_zone(v)

rc('volume #* level 0.75 step 1') 
j=0
for i in range(27,4,-1):
	print('volume #' + str(i) + ' save ../measurements/acat_3dva/acatVols_segmented/'+frame +'_' + str(j).zfill(2) +'.mrc')
	rc('volume #' + str(i) + ' save ../measurements/acat_3dva/acatVols_segmented/'+frame +'_' + str(j).zfill(2) +'.mrc')
	j = j +1 

rc('volume #' + str(3) + ' save ../measurements/acat_3dva/acatVols_segmented/'+frame +'_' + str(j).zfill(2) +'.mrc')
j = j +1 
rc('volume #' + str(2) + ' save ../measurements/acat_3dva/acatVols_segmented/'+frame +'_' + str(j).zfill(2) +'.mrc')
j = j +1 


'''

#for letter in ascii_uppercase:
#	sel = rc('select #0:.'+letter)
#	print(sel)
#	centroid = rc('define centroid')
#	mod_centroid = rc(

for i in range (1,subs+1):
	#[rand1,rand2,rand3] = [randint(-7,7) for p in range(0,3)]
	sel = rc('select #0.'+str(i))
	#centroid = rc('define centroid')
	#rc('move '+str(rand1)+','+str(rand2) +','+str(rand3)+' coord #0.'+str(i)+' mod #0.'+str(i))
	#mod_centroid = rc('define centroid')
	#rc('cs')
rc('select')
rc('rainbow model')

rc('open ./maps/'+cryomap+'.mrc')
colorchains = ''
for c in range (1,subs+1):
	colorchains= colorchains+'0.'+str(c)+','

	
rc('scolor #1 zone #0.'+colorchains[-1]+' range 15')

for v in volume_list():
	split_volume_by_color_zone(v)

for chain in range(1,subs+1):
	rc('fitmap #0.'+str(chain)+' #'+ str(chain+2)+' moveWholeMolecules false')

rc('combine #0.'+colorchains[-1]+' name simNum'+str(simNum))

modName = 'fromJ64_actinonly_frame18'
modnum1 = str(subs+3)
modnum2 = str(subs+4)
modnum3 = str(subs+5)
rc('write format pdb relative #1 #'+modnum1+' '+modName+'.pdb')
################################################################################
#make 3 copies of model for stitching

#rc('combine #'+modnum1+' modelId #'+modnum2+' name simNum2')
#rc('combine #'+modnum2+' modelId #'+modnum3+' name simNum3')

#rc('matchmaker #'+modnum1+':.A:.B:.C #'+modnum2+':.W:.X:.Y pairing ss')
#rc('delete #'+modnum1+':.A:.B:.C')
#rc('matchmaker #'+modnum2+':.A:.B:.C #'+modnum3+':.W:.X:.Y pairing ss')
#rc('delete #'+modnum3+':.W:.X:.Y')
	
#rc('write #'+modnum1+' comp2bin3_final_A.pdb')
#rc('write #'+modnum2+' comp2bin3_final_B.pdb')
#rc('write #'+modnum3+' comp2bin3_final_C.pdb')

rc('stop now')

'''
