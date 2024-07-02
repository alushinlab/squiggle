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
from scipy.ndimage import gaussian_filter
import sys
print('Imports finished. Beginning script...')
################################################################################
def load_cmm_data(file_name):
    text_holder = np.genfromtxt(file_name, delimiter='\"', dtype=str, skip_header=1, skip_footer=1)
    cmm_data = text_holder[:,[3,5,7]]
    return cmm_data

def create_spline(initial_points, eval_points, s):
    # Generate a range of x values corresponding to the number of points
    x_values = np.arange(len(initial_points))

    # Create a univariate spline for each dimension (x, y, z)
    splines = []
    for dim in range(3):  # x, y, z
        y_values = initial_points[:, dim]
        spline = UnivariateSpline(x_values, y_values, s=s)
        splines.append(spline)

    # Compute the arc length of the spline
    arc_lengths = [0]
    for i in range(1, len(x_values)):
        delta_s = np.sqrt(sum((splines[dim](x_values[i]) - splines[dim](x_values[i-1]))**2 for dim in range(3)))
        arc_lengths.append(arc_lengths[-1] + delta_s)

    total_arc_length = arc_lengths[-1]
    target_arc_lengths = np.linspace(0, total_arc_length, eval_points)

    # Now find the x values that correspond to these target arc lengths
    x_eval = np.interp(target_arc_lengths, arc_lengths, x_values)
    # Evaluate the splines at the computed x values
    curve_points = np.vstack([spline(x_eval) for spline in splines]).T
    return curve_points, total_arc_length

def create_volume_from_spline(curve_points, radius, map_shape):
    volume = np.zeros(map_shape)
    for point in curve_points:
        # Create a mask for points within the sphere/cylinder
        z, y, x = np.ogrid[-radius: radius+1, -radius: radius+1, -radius: radius+1]
        mask = x**2 + y**2 + z**2 <= radius**2

        # Calculate bounds for placing mask onto the volume
        z_start, z_end = int(max(0, point[2] - radius)), int(min(map_shape[0], point[2] + radius + 1))
        y_start, y_end = int(max(0, point[1] - radius)), int(min(map_shape[1], point[1] + radius + 1))
        x_start, x_end = int(max(0, point[0] - radius)), int(min(map_shape[2], point[0] + radius + 1))

        # Calculate bounds for the mask
        mz_start, mz_end = int(max(0, radius - point[2])), int(min(2*radius + 1, map_shape[0] + radius - point[2]))
        my_start, my_end = int(max(0, radius - point[1])), int(min(2*radius + 1, map_shape[1] + radius - point[1]))
        mx_start, mx_end = int(max(0, radius - point[0])), int(min(2*radius + 1, map_shape[2] + radius - point[0]))

        # Assign values using the mask
        volume[z_start:z_end, y_start:y_end, x_start:x_end][mask[mz_start:mz_end, my_start:my_end, mx_start:mx_end]] = 1

    volume = gaussian_filter(volume, sigma=1)  # Adjust sigma as needed
    return volume

def create_volume_from_spline(curve_points, radius, map_shape):
    volume = np.zeros(map_shape)
    for point in curve_points:
        # Create a mask for points within the sphere/cylinder
        z, y, x = np.ogrid[-radius: radius+1, -radius: radius+1, -radius: radius+1]
        mask = x**2 + y**2 + z**2 <= radius**2

        # Calculate bounds for placing mask onto the volume
        z_start, z_end = int(max(0, point[2] - radius)), int(min(map_shape[0], point[2] + radius + 1))
        y_start, y_end = int(max(0, point[1] - radius)), int(min(map_shape[1], point[1] + radius + 1))
        x_start, x_end = int(max(0, point[0] - radius)), int(min(map_shape[2], point[0] + radius + 1))

        # Calculate bounds for the mask
        mz_start = int(radius - (point[2] - z_start))
        mz_end = int(mz_start + (z_end - z_start))

        my_start = int(radius - (point[1] - y_start))
        my_end = int(my_start + (y_end - y_start))

        mx_start = int(radius - (point[0] - x_start))
        mx_end = int(mx_start + (x_end - x_start))

        # Clip mask and volume slices to ensure they have the same shape
        mask_region = mask[mz_start:mz_end, my_start:my_end, mx_start:mx_end]
        volume_region = volume[z_start:z_end, y_start:y_end, x_start:x_end]

        min_depth = min(mask_region.shape[0], volume_region.shape[0])
        min_height = min(mask_region.shape[1], volume_region.shape[1])
        min_width = min(mask_region.shape[2], volume_region.shape[2])

        # Assign values using the clipped mask and volume regions
        volume_region[:min_depth, :min_height, :min_width][mask_region[:min_depth, :min_height, :min_width]] = 1

    volume = gaussian_filter(volume, sigma=1)  # Adjust sigma as needed
    return volume

def compute_CCC(vol1, vol2, mask_vol):
    # Apply the mask
    vol1_masked = vol1 * mask_vol
    vol2_masked = vol2 * mask_vol
    
    # Compute means
    mean_vol1 = np.sum(vol1_masked) / np.sum(mask_vol)
    mean_vol2 = np.sum(vol2_masked) / np.sum(mask_vol)
    
    # Compute the CCC
    numerator = np.sum((vol1_masked - mean_vol1) * (vol2_masked - mean_vol2))
    denominator = np.sqrt(np.sum((vol1_masked - mean_vol1)**2) * np.sum((vol2_masked - mean_vol2)**2))
    return numerator / denominator

# Procedure to adjust control points
def adjust_control_points(control_points, gradient, step_size):
    new_control_points = np.copy(control_points)
    for i in range(len(control_points)):
        # Fetch the control point
        point = control_points[i]
        # Ensure the coordinates are within the image volume
        coords = np.clip(point.astype(int), 0, np.array(denoised_map.shape)-1)
        # Look up the gradient at this point
        grad_at_point = gradient[:, coords[2], coords[1], coords[0]]
        # Move the point in the direction of the gradient
        new_control_points[i] += 1.0*grad_at_point * step_size
    return new_control_points

def compute_spline_and_volume(data_points, grad_map, denoised_map, file_name):
	map_shape = denoised_map.shape
	print('Creating uniformly sampled spline...')
	_, arc_length = create_spline(data_points, len(data_points), DEFAULT_S)
	print(arc_length)
	NUM_POINTS = int(arc_length / 7.6)
	print('Using ' + str(NUM_POINTS) + ' points for the calculation.')
	curve_points, _ = create_spline(data_points, NUM_POINTS, DEFAULT_S)
	print('Creating mask for masked CCC calculation...')
	masked_volume = create_volume_from_spline(curve_points, 15, map_shape) > 0.9
	#with mrcfile.new('/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/denoise_tomos_adv/fine_sampling_squig_tomos/manual_traces/compression_03/fil%s_mask.mrc'%str(i+1).zfill(2), overwrite=True) as mrc:
	#	mrc.set_data(masked_volume.astype('float32'))

	# Iterate until convergence (or maximum number of iterations)
	print('Now performing optimization...')
	currBest_control_points, _ = create_spline(curve_points, NUM_POINTS, DEFAULT_S)
	orig_volume = create_volume_from_spline(currBest_control_points, 5, map_shape)
	currBest_CCC = compute_CCC(orig_volume, denoised_map, masked_volume)
	print('The starting CCC is: ' + str(currBest_CCC))
	max_iterations = 100
	step_size = 100
	no_improvement_iterations = 0
	MAX_NO_IMPROVEMENT= 10
	for iteration in tqdm(range(max_iterations)):
		new_control_points = adjust_control_points(currBest_control_points, grad_map, step_size)
		new_spline, _ = create_spline(new_control_points, NUM_POINTS, DEFAULT_S)
		new_volume = create_volume_from_spline(new_spline, 5, map_shape)
		new_CCC = compute_CCC(new_volume, denoised_map, masked_volume)
		if(new_CCC > currBest_CCC):
			currBest_control_points = new_control_points
			currBest_CCC = new_CCC
			step_size *= 1.0
			no_improvement_iterations = 0
		else:
			step_size *= 0.5
			no_improvement_iterations += 1
		if(no_improvement_iterations >= MAX_NO_IMPROVEMENT):
			break
		if(step_size < 0.1):
			break
		
	print('The final CCC is: ' + str(currBest_CCC))
	curve_points_optimized, _ = create_spline(currBest_control_points[1:-1], NUM_POINTS*3, DEFAULT_S)
	volume_optimized = create_volume_from_spline(curve_points_optimized, 5, map_shape)

	with mrcfile.new('/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/denoise_tomos_adv/fine_sampling_squig_tomos/manual_traces/'+file_name+'/fil%s_spheres_optimized.mrc'%str(i+1).zfill(2), overwrite=True) as mrc:
		mrc.set_data(volume_optimized.astype('float32'))
    
	return volume_optimized, curve_points_optimized

################################################################################
base_file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/denoise_tomos_adv/fine_sampling_squig_tomos/manual_traces/compression_10*'))
for i in range(0, len(base_file_names)):
    file_name = base_file_names[i].split('/')[-1]
    # Load in traces
    print('Loading in manual traces...')
    cmm_file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/denoise_tomos_adv/fine_sampling_squig_tomos/manual_traces/'+file_name+'/*.cmm'))
    manual_traces = []
    for i in range(0, len(cmm_file_names)):
        manual_traces.append(load_cmm_data(cmm_file_names[i]).astype(float))

    print('Manual traces loaded.')

    # Load in image data; denoised tomogram
    print('Loading in denoised map...')
    denoised_file_name = '/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/denoise_tomos_adv/fine_sampling_squig_tomos/denoised//'+file_name+'.mrc'
    with mrcfile.open(denoised_file_name) as mrc:
        denoised_map = mrc.data.astype('float32')
    print('Denoised map loaded.')

    # Compute the gradient of the image
    print('Computing the gradient for step direction...')
    gradient = np.array(np.gradient(denoised_map))
    filtered_gradient = np.zeros_like(gradient)
    for i in range(3):
        filtered_gradient[i] = gaussian_filter(gradient[i], sigma=3)
    print('Finished computing gradient.')

    DEFAULT_S = 10 #0.5 works and gives pretty good results
    for i in range(0, len(manual_traces)):
        compute_spline_and_volume(manual_traces[i], filtered_gradient, denoised_map, file_name)





















