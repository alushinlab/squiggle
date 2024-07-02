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
model='acat_allsubs'
acatMod = 'squig_actinref'
print('open ./'+model+'.pdb')
subs = 25

print('open ./'+model+'.pdb')
rc('open ./'+model+'.pdb')

for i in range (1,subs+1):
	rc('open ./ref_actin/'+acatMod+'.pdb')

rc('split #0')

for i in range (1,subs+1):
	rc('matchmaker #0.'+str(i)+' #'+str(i)+':.A pairing ss')

modName = 'acat_allsquigsubs'

rc('combine #1-'+str(subs)+' name '+modName)
rc('write #'+str(subs+1)+' '+modName+'.pdb')

rc('stop now')


## for middle acat one only
#for i in range (1,subs-1):
	#rc('mm #0.'+str(i)+',0.'+str(i+2)+' #'+str(i)+':.A:.B pairing ss iterate false') #for middle acat
	#rc('delete #'+str(i)+':.A:.B')	



