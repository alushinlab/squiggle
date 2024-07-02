import chimera
from chimera import runCommand as rc
import os
################################################################################
#os.chdir('pdbs/PC1')

def stitch_actin_filaments(model_number):
	for i in range(0, 5):
		rc('open ../pdbs/squig_3dva/masterSquigJ38_comp2Frame9.pdb')

	rc('matchmaker #0:.A:.B:.C #1:.W:.X:.Y pairing ss')
	rc('delete #1:.W:.X:.Y')
	#rc('delete #0:.A:.B:.C')
	rc('matchmaker #1:.A:.B:.C #2:.W:.X:.Y pairing ss')
	rc('delete #2:.W:.X:.Y')

	rc('matchmaker #2:.A:.B:.C #3:.W:.X:.Y pairing ss')
	rc('delete #3:.W:.X:.Y')
	rc('matchmaker #3:.A:.B:.C #4:.W:.X:.Y pairing ss')
	rc('delete #4:.W:.X:.Y')
	#rc('matchmaker #0:.W:.X:.Y #3:.A:.B:.C pairing ss')
	#rc('delete #3:.A:.B:.C')
	#rc('matchmaker #3:.W:.X:.Y #4:.A:.B:.C pairing ss')
	#rc('delete #4:.A:.B:.C')

	#rc('combine #0,1,2 close true')
	rc('write #0 squig_3dva_5stitch_comp2Frame9_final_A.pdb')
	rc('write #1 squig_3dva_5stitch_comp2Frame9_final_B.pdb')
	rc('write #2 squig_3dva_5stitch_comp2Frame9_final_C.pdb')
	rc('write #3 squig_3dva_5stitch_comp2Frame9_final_D.pdb')
	rc('write #4 squig_3dva_5stitch_comp2Frame9_final_E.pdb')
	#rc('close #3')

for i in range(0, 1):
	stitch_actin_filaments('%03d'%i)

rc('stop now')





