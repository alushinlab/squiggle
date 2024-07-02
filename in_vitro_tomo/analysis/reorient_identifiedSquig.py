#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import mrcfile
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
print('Imports finished. Beginning script...')
################################################################################
def generate_bild(data, color):
    o= base_name+'side_by_side/tension_17_boxCorners.bild'
    out=open(o, 'w')
    out.write('.transparency 0.0\n') # 1.0 is fully transparent, 0.0 is opaque
    for j in range(0, len(data)):
            #write out marker entries for each residue pair
            out.write('.color %.5f %.5f %.5f\n'%(color[0], color[1], color[2]))
            out.write(".sphere %.5f %.5f %.5f %.5f \n"%(data[j][0], data[j][1], data[j][2], 6))
    #write final line of xml file, is constant	
    out.close()

def load_cmm_data(file_name):
    text_holder = np.genfromtxt(file_name, delimiter='\"', dtype=str, skip_header=1, skip_footer=1)
    cmm_data = text_holder[:,[3,5,7]]
    return cmm_data

################################################################################
base_name = '/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/denoise_tomos_adv/fine_sampling_squig_tomos/'
# bin2
noisy_file_name = base_name + 'starFiles/tension_17.mrc'#sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/squiggle_cell_tomos/orig_tomos/*.mrc'))
# bin2 denoised
denoised_file_name = base_name + 'denoised/tension_17.mrc'#sorted(glob.glob(base_name + 'denoised/tension_17*'))

# bin2 denoised
trace_file_name = base_name + 'manual_traces/tension_16/tension_17_fil1_smoothed.cmm'#sorted(glob.glob(base_name + 'denoised/tension_17*'))

with mrcfile.open(denoised_file_name, 'r') as mrc:
    tomo = mrc.data

with mrcfile.open(noisy_file_name, 'r') as mrc:
    noisy_tomo = mrc.data

squig_coords = load_cmm_data(trace_file_name)
# do PCA on coords
from sklearn.decomposition import PCA
# squig_coords is your numpy array of shape [100, 3]
pca = PCA(n_components=3)
pca.fit(squig_coords)
transformed_coords = pca.transform(squig_coords)

# Find min and max along each principal component axis
min_bounds = transformed_coords.min(axis=0)
max_bounds = transformed_coords.max(axis=0)

corners_pca = np.array([[min_bounds[0], min_bounds[1], min_bounds[2]],
                        [min_bounds[0], min_bounds[1], max_bounds[2]],
                        [min_bounds[0], max_bounds[1], min_bounds[2]],
                        [min_bounds[0], max_bounds[1], max_bounds[2]],
                        [max_bounds[0], min_bounds[1], min_bounds[2]],
                        [max_bounds[0], min_bounds[1], max_bounds[2]],
                        [max_bounds[0], max_bounds[1], min_bounds[2]],
                        [max_bounds[0], max_bounds[1], max_bounds[2]]])

def pca_to_original(points, pca):
    return pca.inverse_transform(points)

# Transform the corners back to the original space
corners_original = pca_to_original(corners_pca, pca)
rectangular_prism = corners_original
generate_bild(rectangular_prism, (1,1,1))

# Find the minimum and maximum bounds of the corners
min_corner = np.min(corners_original, axis=0) - 10
max_corner = np.max(corners_original, axis=0) + 10
min_corner_clipped = min_corner#np.clip(min_corner, 0, np.array(tomo.shape) - 1)
max_corner_clipped = max_corner#np.clip(max_corner, 0, np.array(tomo.shape) - 1)

# Extract the sub-volume
extracted_sub_volume = tomo[
    int(min_corner_clipped[2]):int(max_corner_clipped[2]),
    int(min_corner_clipped[1]):int(max_corner_clipped[1]),
    int(min_corner_clipped[0]):int(max_corner_clipped[0])
]

extracted_sub_volume_noisy = noisy_tomo[
    int(min_corner_clipped[2]):int(max_corner_clipped[2]),
    int(min_corner_clipped[1]):int(max_corner_clipped[1]),
    int(min_corner_clipped[0]):int(max_corner_clipped[0])
]


with mrcfile.new(base_name+'side_by_side/tension_17_fil1.mrc', overwrite=True) as mrc:
    mrc.set_data(np.asarray(extracted_sub_volume).astype('float32'))
with mrcfile.new(base_name+'side_by_side/tension_17_fil1_noisy.mrc', overwrite=True) as mrc:
    mrc.set_data(np.asarray(extracted_sub_volume_noisy).astype('float32'))




print('hi"')
# Use these bounds to extract the relevant part of your 3D image
# Note: You might need to cast the bounds to integer and ensure they are within the valid range of your image dimensions
extracted_image = tomo[int(max_bounds_rotated[2]):int(min_bounds_rotated[2]),
                           int(min_bounds_rotated[1]):int(max_bounds_rotated[1]),
                           int(max_bounds_rotated[0]):int(min_bounds_rotated[0])]


#Picks
trim_coords = [[950, 1500, 290, 530], #488
               ]

print(len(trim_coords))
#700 for bin2, 450 for bin3
for i in range(0, len(file_names)):
    trim_tomogram(file_names[i], [trim_coords[i][0]-700, trim_coords[i][0]+700, trim_coords[i][1]-700, trim_coords[i][1]+700,trim_coords[i][2], trim_coords[i][3]])
    #trim_tomogram(file_names[i], [trim_coords[i][0]-450, trim_coords[i][0]+450, trim_coords[i][1]-450, trim_coords[i][1]+450,trim_coords[i][2], trim_coords[i][3]])



from scipy.ndimage import affine_transform
# Create an affine transformation matrix
affine_matrix = np.eye(4)
affine_matrix[:3, :3] = pca.components_  # rotation part
affine_matrix[:3, 3] = -np.dot(pca.mean_, pca.components_)  # translation part, if necessary

# Apply the affine transformation
# This aligns the principal components with the axes of the volume
rotated_volume = affine_transform(tomo, affine_matrix[:3,:3],offset=affine_matrix[:3,3],output_shape=tomo.shape,mode='constant',cval=0, order=3)
min_bounds_clipped = np.clip(min_bounds, 0, np.array(rotated_volume.shape) - 1)
max_bounds_clipped = np.clip(max_bounds, 0, np.array(rotated_volume.shape) - 1)

