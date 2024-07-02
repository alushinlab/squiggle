import chimera
from chimera import runCommand as rc
import os
################################################################################
#os.chdir('pdbs/PC1')

def stitch_actin_filaments(model_number):
	for i in range(0, 3):
		rc('open /mnt/data1/ayala/final_squiggle_paper/measure_all/pdbs/noATPcontrol/ayala_fits/str25_J131_newfit.pdb ')

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
	rc('write #0 str25_J131_ayalanewfit_final_A.pdb')
	rc('write #1 str25_J131_ayalanewfit_final_B.pdb')
	rc('write #2 str25_J131_ayalanewfit_final_C.pdb')
	#rc('close #3')

for i in range(0, 1):
	stitch_actin_filaments('%03d'%i)

rc('stop now')





