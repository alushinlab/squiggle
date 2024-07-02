import chimera
from chimera import runCommand as rc
import os
################################################################################
#os.chdir('pdbs/PC1')

def stitch_actin_filaments(model_number):
	for i in range(0, 5):
		rc('open /mnt/data1/ayala/final_squiggle_paper/measure_all/pdbs/in_plane_bent/ADP_cryodrgn_isolde_frame009_alignedP.pdb ')

	rc('matchmaker #0:.A:.B:.C #1:.N:.O:.P pairing ss')
	#rc('delete #1:.W:.X:.Y')
	rc('delete #0:.A:.B:.C')
	rc('matchmaker #1:.A:.B:.C #2:.N:.O:.P pairing ss')
	rc('delete #2:.N:.O:.P')

	rc('matchmaker #0:.N:.O:.P #3:.A:.B:.C pairing ss')
	rc('delete #3:.A:.B:.C')
	rc('matchmaker #2:.A:.B:.C #4:.N:.O:.P pairing ss')
	rc('delete #4:.N:.O:.P')

	#rc('combine #0,1,2 close true')
	rc('write #0 ADP_cryodrgn_isolde_frame009_alignedP_final_A.pdb')
	rc('write #1 ADP_cryodrgn_isolde_frame009_alignedP_final_B.pdb')
	rc('write #2 ADP_cryodrgn_isolde_frame009_alignedP_final_C.pdb')
	rc('write #3 ADP_cryodrgn_isolde_frame009_alignedP_final_D.pdb')
	rc('write #4 ADP_cryodrgn_isolde_frame009_alignedP_final_E.pdb')
	#rc('close #3')

for i in range(0, 1):
	stitch_actin_filaments('%03d'%i)

rc('stop now')





