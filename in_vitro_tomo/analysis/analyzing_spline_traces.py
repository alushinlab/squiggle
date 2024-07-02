#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import mrcfile
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
print('Imports finished. Beginning script...')
################################################################################
def load_cmm_data(file_name):
    text_holder = np.genfromtxt(file_name, delimiter='\"', dtype=str, skip_header=1, skip_footer=1)
    cmm_data = text_holder[:,[3,5,7]]
    return cmm_data

def generate_distinct_colors(n):
    """Generate a list of distinct RGB colors."""
    np.random.seed(42)  # Setting seed for reproducibility
    colors = np.random.rand(n, 3)
    return colors

def save_to_cmm(file_name, points, color):
    """Save points to a .cmm file with the specified color."""
    with open(file_name, 'w') as file:
        file.write('<marker_set name="{}">\n'.format(os.path.basename(file_name).split('.')[0]))
        for idx, (z, y, x) in enumerate(points):
            file.write('<marker id="{}" x="{}" y="{}" z="{}" r="{}" g="{}" b="{}" radius="5"/>\n'.format(
                idx + 1, z, y, x, color[0], color[1], color[2]))
        file.write('</marker_set>\n')


from skimage.restoration import denoise_bilateral
def bilateral_smoothing(data, sigma_color, sigma_spatial):
    """
    Apply bilateral filtering to smooth data while preserving edges.
    sigma_color: Controls how much influence similar intensity values have.
    sigma_spatial: Controls how much influence nearby pixels have.
    """
    smoothed_data = np.zeros_like(data)
    for dim in range(data.shape[1]):  # Apply filtering for each dimension
        smoothed_data[:, dim] = denoise_bilateral(data[:, dim], sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    return smoothed_data

def create_spline(initial_points, step_size, doSmoothing):
    if(doSmoothing):
        #initial_points = bilateral_smoothing(initial_points, 0.1, 100)
        s=150
    else:
        s=0
    # Calculate the cumulative distance along the points
    cumulative_dist = np.cumsum(np.sqrt(np.sum(np.diff(initial_points, axis=0)**2, axis=1)))
    cumulative_dist = np.insert(cumulative_dist, 0, 0)

    # Parameterize points by their cumulative distance
    x_values = cumulative_dist

    # Create a univariate spline for each dimension (x, y, z)
    splines = [UnivariateSpline(x_values, initial_points[:, dim], s=s) for dim in range(3)]

    # Generate evenly spaced arc lengths
    total_arc_length = x_values[-1]
    target_arc_lengths = np.arange(0, total_arc_length, step_size)

    # Evaluate the splines at the computed arc lengths
    curve_points = np.vstack([spline(target_arc_lengths) for spline in splines]).T
    return curve_points


################################################################################
# Load in traces
print('Loading in manual traces...')
manual_traces = []
cmm_file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/denoise_tomos_adv/fine_sampling_squig_tomos/manual_traces/tension_*/*.cmm'))
for i in range(0, len(cmm_file_names)):
    manual_traces.append(load_cmm_data(cmm_file_names[i]).astype(float))

colors = generate_distinct_colors(len(manual_traces))
splines = []
for i in range(0, len(manual_traces)):
    this_spline = create_spline(manual_traces[i], 0.96, False)
    this_spline = create_spline(manual_traces[i], 0.96, True)
    splines.append(this_spline)
    save_to_cmm(cmm_file_names[i][:-4]+'_smoothed.cmm', this_spline, colors[i])






















