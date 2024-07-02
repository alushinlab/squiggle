#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from scipy.stats import kstest
print('Imports finished. Beginning script...')
################################################################################
# save coordinates as bild file
def generate_bild_fil(o, coords):
    out=open(o, 'w')
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    out.write('.transparency 0.0\n') # 1.0 is fully transparent, 0.0 is opaque
    for j in range(0, len(coords)):
        #write out marker entries for each residue pair
        out.write('.color %.5f %.5f %.5f\n'%(colors[j]))
        out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(coords[j][0][0].astype(float), coords[j][0][1].astype(float), coords[j][0][2].astype(float),
                                                            coords[j][1][0].astype(float), coords[j][1][1].astype(float), coords[j][1][2].astype(float),6))
    #write final line of xml file, is constant	
    out.close()	

def generate_bild_plane(o, corners, normal, normal_proj):
    out=open(o[:-5]+'_plane.bild', 'w')
    out.write('.transparency 0.0\n') # 1.0 is fully transparent, 0.0 is opaque
    #write out marker entries for each residue pair
    out.write('.color %.5f %.5f %.5f\n'%(1,1,0))
    out.write(".polygon %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(corners[0][0].astype(float), corners[0][1].astype(float), corners[0][2].astype(float),
                                                            corners[1][0].astype(float), corners[1][1].astype(float), corners[1][2].astype(float),
                                                            corners[2][0].astype(float), corners[2][1].astype(float), corners[2][2].astype(float)))
    out.write(".polygon %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(
                                                            corners[1][0].astype(float), corners[1][1].astype(float), corners[1][2].astype(float),
                                                            corners[2][0].astype(float), corners[2][1].astype(float), corners[2][2].astype(float),
                                                            corners[3][0].astype(float), corners[3][1].astype(float), corners[3][2].astype(float)))

    # Write the normal vector
    out.write('.color %.5f %.5f %.5f\n' % (1,0,1))
    out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n" % (
          normal[0][0], normal[0][1], normal[0][2],
          normal[1][0], normal[1][1], normal[1][2], 6))
        
    # Write the projected normal vector
    out.write('.color %.5f %.5f %.5f\n' % (0,1,1))
    out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n" % (
            normal_proj[0][0], normal_proj[0][1], normal_proj[0][2],
            normal_proj[1][0], normal_proj[1][1], normal_proj[1][2], 6))
    
    out.close()	

def load_cmm_data(file_name):
    text_holder = np.genfromtxt(file_name, delimiter='\"', dtype=str, skip_header=1, skip_footer=1)
    cmm_data = text_holder[:,[3,5,7]]
    return cmm_data

def project_onto_plane(v, axis):
    # Project v onto axis
    projection = np.dot(v, axis) / np.dot(axis, axis) * axis
    # Subtract the projection from v to get the component orthogonal to axis
    return v - projection

# Calculate the angle between two vectors
def calculate_angle(v1, v2):
    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
    return 90-min(angle, 180-angle)




def analyze_fil_orientation(output_name, pca, picks):
    pc1, pc2, pc3 = pca.components_
    mean = pca.mean_
    pc1_axis = [mean-pc1*150,mean+pc1*150]
    pc2_axis = [mean-pc2*100,mean+pc2*100]
    pc3_axis = [mean-pc3*50,mean+pc3*50]
    coords = [pc1_axis, pc2_axis, pc3_axis]
    generate_bild_fil(output_name, coords)

    picks_pca = PCA(n_components=3)
    transformed = picks_pca.fit_transform(picks)
    picks_pc1, picks_pc2, picks_pc3 = picks_pca.components_
    picks_mean = picks_pca.mean_
    corner1 = picks_mean-picks_pc1*500-picks_pc2*500
    corner2 = picks_mean+picks_pc1*500-picks_pc2*500
    corner3 = picks_mean-picks_pc1*500+picks_pc2*500
    corner4 = picks_mean+picks_pc1*500+picks_pc2*500
    corners = [corner1, corner2, corner3, corner4]

    normal_proj = project_onto_plane(picks_pc3, pc1)
    #normalize vectors
    normal_proj = normal_proj / np.linalg.norm(normal_proj)
    pc2 = pc2 / np.linalg.norm(pc2)
    pc2 = pc2 / np.linalg.norm(pc3)

    angle_pc2 = calculate_angle(normal_proj, pc2)
    angle_pc3 = calculate_angle(normal_proj, pc3)
    print(f"Rotation angle for PC2: {angle_pc2} degrees")
    print(f"Rotation angle for PC3: {angle_pc3} degrees")
    
    generate_bild_plane(output_name, corners, [picks_mean, picks_mean + picks_pc3 * 100], [picks_mean, picks_mean + normal_proj * 100])
    return angle_pc2

def plot_angles(o, angles_rad):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(1)  # Reverses the direction of the angle
    ax.set_theta_zero_location('E')  # Sets the zero location to the top (North)
    ax.set_ylim(0, 1.0)  # Sets the radial limits from 0 to 1

    # Plot the data points
    r = np.ones_like(angles_rad)  # Radius set to 1 for quarter circle plot
    for angle in angles_rad[:10]: # compression
        ax.arrow(angle, 0,0,1,width=0.05,head_width=0.10, lw=0.5, length_includes_head=True,alpha=1.0,fc='gray', ec='black')
    for angle in angles_rad[10:]: # tension
        ax.arrow(angle, 0,0,1,width=0.05,head_width=0.10, lw=0.5, length_includes_head=True,alpha=1.0,fc='gray', ec='black')
    
    #ax.scatter(angles_rad, r)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    # Plot a quarter circle for reference
    theta = np.linspace(0, np.pi/2, 100)
    ax.plot(theta, np.ones_like(theta), linestyle='dashed')
    ax.grid(True)
    ax.set_rticks([1.0])
    #ax.set_xticklabels('')
    ax.set_yticklabels('')
    # Display the plot
    plt.savefig(o)
    plt.savefig(o[:-4]+'.svg')


################################################################################
base_name = '/rugpfs/fs0/cem/store/mreynolds/scripts/in_development/denoising_squig_tomos/temp/npy/'
output_base_name = '/rugpfs/fs0/cem/store/mreynolds/scripts/in_development/denoising_squig_tomos/temp/bild/'

input_file_names = sorted(glob.glob(base_name+'*.pkl'))
input_pick_names = sorted(glob.glob(base_name.replace('npy','bild')+'*.cmm'))

loaded_data = []
i = 0
pc2_angles = []
for input_file_name in input_file_names:
    with open(input_file_name, 'rb') as f:
        pca = pickle.load(f)
    with open(input_pick_names[i]):
        picks = load_cmm_data(input_pick_names[i])
    
    output_name = input_file_name.replace('npy', 'bild').replace('.pkl', '.bild')
    pc2_angles.append(analyze_fil_orientation(output_name, pca, picks))
    i = i+1
    print('hi')

pc2_angles = np.asarray(pc2_angles)
angles_rad = np.deg2rad(pc2_angles)

plot_angles(output_base_name+'quarter_circle.png', angles_rad)
plot_angles(output_base_name+'quarter_circle_compression.png', angles_rad[:10])
plot_angles(output_base_name+'quarter_circle_tension.png', angles_rad[10:])


d, p_value = kstest(np.asarray(pc2_angles) / 90.0, 'uniform')  # Normalize the data between 0 and 1
print(f"K-S test statistic: {d}, p-value: {p_value}")

if p_value > 0.05:
    print("The data is uniformly distributed (fail to reject H0).")
else:
    print("The data is not uniformly distributed (reject H0).")



print('hi')