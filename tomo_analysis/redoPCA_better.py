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
from sklearn.decomposition import PCA
import sys
from matplotlib.patches import Ellipse
from skimage.measure import EllipseModel
print('Imports finished. Beginning script...')
################################################################################
def load_cmm_data(file_name):
    text_holder = np.genfromtxt(file_name, delimiter='\"', dtype=str, skip_header=1, skip_footer=1)
    cmm_data = text_holder[:,[3,5,7]].astype(np.float32)
    return cmm_data

def raster_along_fil(squig_coords, step_size, contour_length):
    # Calculate the contour length of the filament
    cumulative_distances = np.zeros(len(squig_coords))
    for i in range(1, len(squig_coords)):
        cumulative_distances[i] = cumulative_distances[i-1] + np.linalg.norm(squig_coords[i] - squig_coords[i-1])
    # Initialize the list to hold segments
    segments = []
    start_idx = 0
    while True:
        # Find the end index where the cumulative distance exceeds the contour length
        end_idx = np.searchsorted(cumulative_distances, cumulative_distances[start_idx] + contour_length)
        if end_idx >= len(squig_coords): # Break if we've reached or exceeded the length of the filament
            break
        # Append the segment
        segments.append(squig_coords[start_idx:end_idx])
        # Update the start index by finding the new start index after stepping forward by step_size_nm
        start_idx = np.searchsorted(cumulative_distances, cumulative_distances[start_idx] + step_size)
        # Break if we don't have enough points left for another full segment
        if start_idx >= len(squig_coords) - 1:
            break
    return segments

def sample_ellipse_points(ell_params, num_samples=100):
    xc, yc, a, b, theta = ell_params
    t = np.linspace(0, 2 * np.pi, num_samples)
    x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    return np.vstack((x, y)).T

def point_to_ellipse_distance(px, py, ell_params, num_samples=100):
    ellipse_points = sample_ellipse_points(ell_params, num_samples)
    distances = np.sqrt((ellipse_points[:, 0] - px)**2 + (ellipse_points[:, 1] - py)**2)
    return np.min(distances)

def fit_ellipse(segment):
    # Fit an ellipse to the segment
    ell = EllipseModel()
    if ell.estimate(segment):
        # Extract the parameters
        xc, yc, a, b, theta = ell.params
        sample_ellipse_points((xc, yc, a, b, theta), 50)
        distances = np.array([point_to_ellipse_distance(x, y, (xc, yc, a, b, theta)) for x, y in segment])
        # Calculate the score as the product of the axes lengths
        # This is a placeholder for your scoring function
        size_score = a * b
        score = size_score / np.mean(distances)
        return ell.params, score
    else:
        return (0, 0, 0, 0, 0), 0

def make_pca_scatterplot(transformed_coords, output_file_name):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.cm.winter
    transformed_coords = transformed_coords[np.argsort(transformed_coords[:, 0])]
    # Project onto PC1 vs PC2
    #axs[0].plot(transformed_coords[:, 1], transformed_coords[:, 0], linewidth=5, color=cmap(i/len(transformed_coords))
    for j in range(1, len(transformed_coords)):
            axs[0].plot(transformed_coords[j-1:j+1, 1], transformed_coords[j-1:j+1, 0], color=cmap(j / len(transformed_coords)), linewidth=8)
            axs[1].plot(transformed_coords[j-1:j+1, 2], transformed_coords[j-1:j+1, 0], color=cmap(j / len(transformed_coords)), linewidth=8)
            axs[2].plot(transformed_coords[j-1:j+1, 1], transformed_coords[j-1:j+1, 2], color=cmap(j / len(transformed_coords)), linewidth=8)
            axs[0].set_aspect(0.5)
            axs[1].set_aspect(0.5)
            axs[2].set_aspect(1)
    #axs[0].set_aspect('equal')
    axs[0].set_xlabel('PC2')
    axs[0].set_ylabel('PC1')
    axs[0].set_xlim(-40,40)
    axs[0].set_ylim(-200,200)
    axs[0].set_title('Projection onto PC1 vs PC2')
    # Project onto PC1 vs PC3
    #axs[1].plot(transformed_coords[:, 2], transformed_coords[:, 0], linewidth=5)
    #axs[1].set_aspect('equal')
    axs[1].set_xlabel('PC3')
    axs[1].set_ylabel('PC1')
    axs[1].set_title('Projection onto PC1 vs PC3')
    axs[1].set_xlim(-40,40)
    axs[1].set_ylim(-200,200)
    # Project onto PC2 vs PC3
    #axs[2].plot(transformed_coords[:, 1], transformed_coords[:, 2], linewidth=5)
    axs[2].set_xlabel('PC2')
    axs[2].set_ylabel('PC3')
    axs[2].set_xlim(-20,20)
    axs[2].set_ylim(-20,20)
    ell = EllipseModel()
    ell.estimate(np.column_stack([transformed_coords[:, 1], transformed_coords[:, 2]]))
    xc, yc, a, b, theta = ell.params
    # Create the matplotlib Ellipse object with the parameters
    ellipse = Ellipse(xy=(xc, yc), width=2*a, height=2*b, angle=np.degrees(theta),
                      edgecolor='r', facecolor='none',alpha=0.8,zorder=3)
    #note_text = f"a: {2*a:.2f}, b: {2*b:.2f}"
    #axs[2].text(0.05, 0.95, note_text, transform=axs[2].transAxes, fontsize=12,
    #            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axs[2].set_title('Projection onto PC2 vs PC3: '+f"a: {2.0*a:.2f}, b: {2.0*b:.2f}")
    axs[2].add_patch(ellipse)
    axs[2].set_aspect('equal')
    for ax in axs:
        # Transparent line at x=0, behind the plot
        ax.axvline(x=0, color='grey', linestyle='--', alpha=0.5, zorder=0)
        # Transparent line at y=0, behind the plot
        ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5, zorder=0)
        
    plt.tight_layout()
    plt.savefig(output_file_name)
    plt.savefig(output_file_name[:-4]+'.svg',format='svg')
    plt.show()

################################################################################
base_name = '/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/denoise_tomos_adv/fine_sampling_squig_tomos/'
#trace_file_name = base_name + 'manual_traces/tension_16/tension_17_fil1_smoothed.cmm'#sorted(glob.glob(base_name + 'denoised/tension_17*'))
trace_file_names = sorted(glob.glob(base_name + 'manual_traces/*/*smoothed.cmm'))
output_base_name = '/rugpfs/fs0/cem/store/mreynolds/scripts/in_development/denoising_squig_tomos/temp/'

for i in (46):#tqdm(range(0, len(trace_file_names))):
    squig_coords = load_cmm_data(trace_file_names[i])
    squig_coords = np.asarray(squig_coords)*9.6/10 # convert to nm (9.6A/px)
    squig_segments = raster_along_fil(squig_coords, 10, 350)
    if(len(squig_segments) <1):
        continue

    best_score = 0
    best_squig_idx = -1
    best_params = None
    for j in range(0, len(squig_segments)):
        # do PCA on coords
        pca = PCA(n_components=3)
        pca.fit(squig_segments[j])
        transformed_coords = pca.transform(squig_segments[j])
        params, score = fit_ellipse(transformed_coords[:,[1,2]])
        if score > best_score:
            best_score = score
            best_squig_idx = j
            best_params = params

    if best_squig_idx != -1:
        pca.fit(squig_segments[best_squig_idx])
        transformed_coords = pca.transform(squig_segments[best_squig_idx])
        output_dirs = output_base_name+trace_file_names[i].split('/')[-2]
        os.makedirs(output_dirs, exist_ok=True)
        make_pca_scatterplot(transformed_coords, output_dirs + '/PCA_proj%s.png'%(str(i).zfill(3)))




sys.exit()
