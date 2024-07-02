import chimera
from chimera import runCommand as rc
import os
################################################################################
#os.chdir('pdbs/PC1')

def stitch_actin_filaments(model_number):
	for i in range(0, 3):
		rc('open ../pdbs/acat_3dva/acat_ActinOnly_3dva_frame%s.pdb'%str(model_number))

	rc('matchmaker #0:.A:.B:.C #1:.W:.X:.Y pairing ss')
	#rc('delete #1:.W:.X:.Y')
	rc('delete #0:.A:.B:.C')
	rc('matchmaker #1:.A:.B:.C #2:.W:.X:.Y pairing ss')
	rc('delete #2:.W:.X:.Y')

	#rc('matchmaker #0:.X:.Y:.Z #3:.A:.B:.C pairing ss')
	#rc('delete #3:.A:.B:.C')
	#rc('matchmaker #3:.X:.Y:.Z #4:.A:.B:.C pairing ss')
	#rc('delete #4:.A:.B:.C')

	#rc('combine #0,1,2 close true')
	rc('write #0 ../pdbs/acat_3dva/stitched/acat_ActinOnly_3dva_frame%s_final_A.pdb'%str(model_number))
	rc('write #1 ../pdbs/acat_3dva/stitched/acat_ActinOnly_3dva_frame%s_final_B.pdb'%str(model_number))
	rc('write #2 ../pdbs/acat_3dva/stitched/acat_ActinOnly_3dva_frame%s_final_C.pdb'%str(model_number))
	#rc('close #3')
	rc('close #0,1,2')

for i in range(0, 20):
	stitch_actin_filaments(i)

rc('stop now')





