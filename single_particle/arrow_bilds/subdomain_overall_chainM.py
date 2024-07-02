#!/home/greg/Programs/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from prody import *
import string
import sys
################################################################################
straight_file_name = './afterIsolde_pdbs/beforeIsolde_chainM.pdb'

bent_file_name = './afterIsolde_pdbs/avg_squigft_afterIsolde_noH.pdb'
output_pdb_name = './aligned_to_chainM/avgsquig_alignedStrProtomers.pdb'
output_diff_vects = './aligned_to_chainM/avgsquig_diff_vects.npy'

'''
bent_file_name = './afterIsolde_pdbs/acat_upToFrame4fit_afterIsolde_noH.pdb'
output_pdb_name = './aligned_to_chainM/acat_alignedStrProtomers.pdb'
output_diff_vects = './aligned_to_chainM/acat_diff_vects.npy'

bent_file_name = './afterIsolde_pdbs/ctrl_lp10_J126fit_afterIsolde_noH.pdb'
output_pdb_name = './aligned_to_chainM/ctrlJ126_alignedStrProtomers.pdb'
output_diff_vects = './aligned_to_chainM/ctrlJ126_diff_vects.npy'

bent_file_name = './afterIsolde_pdbs/ctrl_lp10_J131fit_afterIsolde_noH.pdb'
output_pdb_name = './aligned_to_chainM/ctrlJ131_alignedStrProtomers.pdb'
output_diff_vects = './aligned_to_chainM/ctrlJ131_diff_vects.npy'
'''

################################################################################
def load_pdb(file_name):
	# Use ProDy to import PDB file
	p = parsePDB(file_name, subset='calpha')
	chids = sorted(set(p.getChids()))
	chains = []
	for chain_idx in chids:
		chains.append(p.select('chain ' + chain_idx).copy())
	
	# get the coordinates for each atom of each actin subunit
	coords = []
	for i in range(0,len(chains)):
		coords.append(chains[i].getCoords())
	
	# make each helix into a [num_chains x num_atoms_per_actin x 3] array
	coords = np.asarray(coords)
	return coords

################################################################################
straight = load_pdb(straight_file_name)
bent = load_pdb(bent_file_name)

################################################################################
# generate aligned chain
def compute_alignedChains(ref, chains):
	aligned_chains = []
	#trans_matrix = calcTransformation(ref[4], chains[4]) # align whole chain
	for i in range(0, len(chains)):
		trans_matrix = calcTransformation(chains[i], ref) # align whole chain
		aligned_chain = applyTransformation(trans_matrix, chains[i])
		aligned_chains.append(aligned_chain)
	
	return np.asarray(aligned_chains)


################################################################################
aligned_chains = compute_alignedChains(straight[0], bent)

# save result
p = parsePDB(straight_file_name, subset='calpha') #protein to map these shear energies to
new_p = p.setCoords(aligned_chains)
writePDB(output_pdb_name, p)

loaded_p = parsePDB(output_pdb_name)
aligned_str_ref = loaded_p.getCoordsets()

################################################################################
diff_vects = aligned_str_ref - straight[0]
scalar = 1.0
bent_scaled = straight[0] + scalar*diff_vects

np.save(output_diff_vects, diff_vects)

sys.exit()


