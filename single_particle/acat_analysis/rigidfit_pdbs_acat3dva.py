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
index=19
subs = 25 
# For the fitting, start with the model from the higher-resolution upToFrame4 map
# and fit that model into frame 2's map, chain by chain.
# For each subsequent map, start with an adjacent frame's model that is known.
#rc('open ../pdbs/acat_upToFrame4/ayala_fit/acat_fit.pdb') # path to uptoFrame4 model, use only for frame 2
rc('open ../pdbs/acat_3dva/updated_fitting_method/fitAcat_frame_'+str(index-1).zfill(3)+'.pdb') # path to adjacent frame's model
rc('open ../pdbs/acat_3dva/frame_'+str(index).zfill(3)+'.mrc')
rc('split #0')

for chain in range(1,subs+1):
	rc('fitmap #0.'+str(chain)+' #1 moveWholeMolecules true resolution 11 metric correlation ')

rc('combine #0.*')

rc('write format pdb relative #1 #2 ../pdbs/acat_3dva/updated_fitting_method/fitAcat_frame_'+str(index).zfill(3)+'.pdb')
################################################################################

rc('stop now')


