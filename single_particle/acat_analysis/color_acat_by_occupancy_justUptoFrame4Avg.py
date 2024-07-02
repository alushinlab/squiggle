##!/home/alus_soft/matt_EMAN2/bin/python
####################################################################################################
import os
#from chimerax import runCommand as rc
from chimerax.core.commands import run as rc
import numpy as np
import matplotlib.pyplot as plt
####################################################################################################

def color_by_occupancy(pdb_file_name, acat_map, idx):
	# Perform chimera operations:
	# Open files
	rc(session, 'open ' + pdb_file_name)
	rc(session, 'open ' + acat_map)
	
	rc(session, 'turn x 90')
	rc(session, 'volume #%s level 0.0195'%str(2*idx+2))
	
	#rc(session, 'select #%s:860-end'%str(2*idx+1))
	#rc(session, 'delete sel')
	
	
	rc(session, 'sel #%s/A,C,E,G,I,K,M,O,Q,S,U,W,Y,a'%str(2*idx+1))
	rc(session, 'color sel light blue')
	rc(session, 'sel #%s/B,D,F,H,J,L,N,P,R,T,V,X,b,x'%str(2*idx+1))
	rc(session, 'color sel blue')
	
	occ_vals = np.loadtxt('./occupancy_txt/upToFrame4_occ.txt')[::-1]
	cm = plt.get_cmap('Reds')
	cm(0.0)
	
	#lowercase = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
	lowercase = ['c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
	
	for i in range(0, len(occ_vals)):
		rgb_val = cm(occ_vals[i])
		print(rgb_val)
		rc(session, 'color #%s/%s green'%(str(2*idx+1), lowercase[i]))
		print('color #%s/%s green'%(str(2*idx+1), lowercase[i]))
		print('color modify #%s/%s red '%(str(2*idx+1), lowercase[i]) + str(100.0*rgb_val[0]) + ' green ' + str(100.0*rgb_val[1]) + ' blue ' + str(100.0*rgb_val[2]))
		rc(session, 'color modify #%s/%s red '%(str(2*idx+1), lowercase[i]) + str(100.0*rgb_val[0]))
		rc(session, 'color modify #%s/%s green '%(str(2*idx+1), lowercase[i]) + str(100.0*rgb_val[1]))
		rc(session, 'color modify #%s/%s blue '%(str(2*idx+1), lowercase[i]) + str(100.0*rgb_val[2]))
		#rc(session, 'color modify #1/%s blue '%(lowercase[i]) + str(100.0*rgb_val[2]))
		
	
	rc(session, 'sel #%s/y,z'%str(2*idx+1))
	rc(session, 'delete sel')
	
	rc(session, 'sel #%s'%str(2*idx+1))
	rc(session, 'color zone #%s near sel distance 15'%str(2*idx+2))
	rc(session, 'sel #1000')
	rc(session, 'hide #!* models')
	#rc(session, 'color zone #2 near sel distance 10')


for i in range(0,1):
	pdb_file = '/mnt/data1/ayala/final_squiggle_paper/measure_all/pdbs/acat_upToFrame4/ayala_fit/add_acats/acat_allsquigsubs.pdb'
	mrc_file = '/mnt/data1/ayala/final_squiggle_paper/measure_all/pdbs/acat_upToFrame4/job3_lp10_map.mrc'
	color_by_occupancy(pdb_file, mrc_file, i)


rc(session, 'volume #* step 1 level 0.75')
rc(session, 'set bgColor white; lighting full; material dull; graphics silhouettes true')
rc(session, 'show #!2 models')
rc(session, 'view')



