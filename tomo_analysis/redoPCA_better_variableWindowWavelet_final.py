#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import pywt
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.gridspec as gridspec
import pickle
print('Imports finished. Beginning script...')
################################################################################
def load_cmm_data(file_name):
    text_holder = np.genfromtxt(file_name, delimiter='\"', dtype=str, skip_header=1, skip_footer=1)
    cmm_data = text_holder[:,[3,5,7]].astype(np.float32)
    return cmm_data

def plot_wavelet_coeffs(coeffs, scales, title, filename):
    plt.imshow(np.abs(coeffs), extent=[0, len(coeffs[0]), scales[0], scales[-1]], aspect='auto', 
               origin='lower', cmap='jet')
    plt.colorbar(label='Magnitude of Coefficients')
    plt.title(title)
    plt.xlabel('Position along filament')
    plt.ylabel('Scale')
    plt.savefig(filename)
    plt.close()

def plot_wavelet_coeffs_together(coeffs_curvature, coeffs_torsion, scales, output_base_name, plot_file_name):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Curvature Coefficients
    cax1 = axs[0].imshow(np.abs(coeffs_curvature), extent=[0, len(coeffs_curvature[0]), scales[0], scales[-1]], aspect='auto', 
                          origin='lower', cmap='jet')
    fig.colorbar(cax1, ax=axs[0], label='Magnitude of Coefficients')
    axs[0].set_title('Curvature Wavelet Coefficients')
    axs[0].set_ylabel('Scale')

    # Torsion Coefficients
    cax2 = axs[1].imshow(np.abs(coeffs_torsion), extent=[0, len(coeffs_torsion[0]), scales[0], scales[-1]], aspect='auto', 
                          origin='lower', cmap='jet')
    fig.colorbar(cax2, ax=axs[1], label='Magnitude of Coefficients')
    axs[1].set_title('Torsion Wavelet Coefficients')
    axs[1].set_xlabel('Position along filament')
    axs[1].set_ylabel('Scale')

    plt.tight_layout()
    plt.savefig(f"{output_base_name}{plot_file_name}_wavelet_coefficients.png")
    plt.close()

def wavelet_analysis_curvature_torsion(curvature, torsion, wavelet='morl', max_scale=128):
    scales = np.arange(1, max_scale)
    coeffs_curvature, _ = pywt.cwt(curvature, scales, wavelet)
    coeffs_torsion, _ = pywt.cwt(torsion, scales, wavelet)
    return coeffs_curvature, coeffs_torsion

def compute_curvature_torsion(coords):
    # Fit a spline to the 3D coordinates of the filament.
    tck, u = splprep(coords.T, s=0)
    u_fine = np.linspace(0, 1, coords.shape[0])

    first_derivatives = splev(u_fine, tck, der=1)
    second_derivatives = splev(u_fine, tck, der=2)
    third_derivatives = splev(u_fine, tck, der=3)

    # Initialize arrays to hold the curvature and torsion values
    curvature = np.zeros(u_fine.shape[0])
    torsion = np.zeros(u_fine.shape[0])

    # Calculate curvature and torsion
    for i in range(u_fine.shape[0]):
        dx, dy, dz = first_derivatives[0][i], first_derivatives[1][i], first_derivatives[2][i]
        ddx, ddy, ddz = second_derivatives[0][i], second_derivatives[1][i], second_derivatives[2][i]
        dddx, dddy, dddz = third_derivatives[0][i], third_derivatives[1][i], third_derivatives[2][i]
        
        # Tangent vector
        T = np.array([dx, dy, dz])
        T_magnitude = np.linalg.norm(T)
        T = T / T_magnitude
        
        # Normal vector
        N = np.cross(T, np.array([ddx, ddy, ddz]))
        N_magnitude = np.linalg.norm(N)
        N = N / N_magnitude
        
        # Binormal vector
        B = np.cross(T, N)
        
        # Curvature (κ)
        curvature[i] = N_magnitude / T_magnitude**2
        
        # Torsion (τ)
        torsion[i] = np.dot(np.cross(T, N), np.array([dddx, dddy, dddz])) / N_magnitude**2
    return curvature, torsion

def apply_pca_and_plot(segment, output_base_name, plot_file_name, color):
    pca = PCA(n_components=3)
    transformed = pca.fit_transform(segment)
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 4)
    axs = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2:4])]
    # PC1 vs PC2
    axs[0].scatter(transformed[:, 1], transformed[:, 0], alpha=0.7, c=color)
    axs[0].set_xlabel('PC2')
    axs[0].set_ylabel('PC1')
    axs[0].set_title('PC1 vs PC2')
    axs[0].set_xlim(-30,30)
    axs[0].set_ylim(-200,200)

    # PC1 vs PC3
    axs[1].scatter(transformed[:, 2], transformed[:, 0], alpha=0.7, c=color)
    axs[1].set_xlabel('PC3')
    axs[1].set_ylabel('PC1')
    axs[1].set_title('PC1 vs PC3')
    axs[1].set_xlim(-30,30)
    axs[1].set_ylim(-200,200)

    # PC2 vs PC3
    axs[2].scatter(transformed[:, 2], transformed[:, 1], alpha=0.7, c=color)
    axs[2].set_xlabel('PC3')
    axs[2].set_ylabel('PC2')
    axs[2].set_title('PC2 vs PC3')
    axs[2].set_xlim(-20,20)
    axs[2].set_ylim(-20,20)

    plt.tight_layout()
    plt.savefig(f"{output_base_name}{plot_file_name}_PCA_subplots.png")
    plt.close()

def index_of_min_derivative_within_window(pc1, pc2, window_size=75):
    # Find the current center of PC1 based on its range
    center_value_pc1 = (np.max(pc1) + np.min(pc1)) / 2
    # Find the index closest to the center value of PC1
    center_index_pc1 = np.argmin(np.abs(pc1 - center_value_pc1))
    # Define the window around this center index
    window_start = max(center_index_pc1 - window_size, 0)
    window_end = min(center_index_pc1 + window_size, len(pc1))

    # Calculate the derivative within the window
    derivative = np.gradient(pc2[window_start:window_end], pc1[window_start:window_end])
    # Find the index of the minimum absolute derivative within this window
    min_derivative_index = window_start + np.argmin(np.abs(derivative))
    return min_derivative_index, derivative, window_start

def apply_pca_and_plot_centered(segment, output_base_name, plot_file_name, fig, axs, color='gray'):
    pca = PCA(n_components=3)
    transformed = pca.fit_transform(segment)
    min_derivative_index, derivatives, window_start = index_of_min_derivative_within_window(transformed[:, 0], transformed[:, 1])

    # Check if the second derivative at that point is negative
    second_derivative = np.gradient(derivatives)[min_derivative_index - window_start]  # Local index in derivatives array
    p = plot_file_name+'.pkl'
    with open(output_base_name[:-8]+'npy/'+p, 'wb') as f:
        pickle.dump(pca, f)
    #np.save(output_base_name[:-8]+'npy/'+p,pca)
    print(second_derivative)
    if (second_derivative > 0.0 and 'compression' in p): 
        print('here')
        # Mirror PC2 and PC3 if the second derivative is negative
        transformed[:, 1] = -transformed[:, 1]
        transformed[:, 2] = -transformed[:, 2]
    
    if (second_derivative < 0.0 and 'tension' in p): 
        print('here')
        # Mirror PC2 and PC3 if the second derivative is negative
        transformed[:, 1] = -transformed[:, 1]
        transformed[:, 2] = -transformed[:, 2]

    if ('tension_07_fil4' in p) or ('compression_06_fil1' in p) or ('compression_10_fil2' in p):
        print('hello')
        transformed[:, 1] = -transformed[:, 1]
        transformed[:, 2] = -transformed[:, 2]

    if ('tension_07_fil4' in p) or ('tension_06_fil1' in p) or ('compression_06_fil3' in p) or ('compression_08_fil9' in p) or ('compression_09_fil5' in p) or ('compression_09_fil6' in p) or ('compression_10_fil2' in p): #or 'compression_08_fil9' in p:
        transformed[:,0] = -transformed[:,0]
    transformed[:, 0] -= transformed[min_derivative_index, 0]
    #transformed[:, 1] -= transformed[min_derivative_index, 1]
    # PC1 vs PC2
    axs[0].plot(transformed[:, 1], transformed[:, 0], alpha=0.1, linewidth=4, c=color)
    axs[0].set_xlabel('PC2')
    axs[0].set_ylabel('PC1')
    axs[0].set_title('PC1 vs PC2')
    axs[0].set_xlim(-20,20)
    axs[0].set_ylim(-150,150)

    # PC1 vs PC3
    axs[1].plot(transformed[:, 2], transformed[:, 0], alpha=0.1, linewidth=4, c=color)
    axs[1].set_xlabel('PC3')
    axs[1].set_ylabel('PC1')
    axs[1].set_title('PC1 vs PC3')
    axs[1].set_xlim(-20,20)
    axs[1].set_ylim(-150,150)

    # PC2 vs PC3
    axs[2].plot(transformed[:, 2], transformed[:, 1], alpha=0.1, linewidth=4, c=color)
    axs[2].set_xlabel('PC3')
    axs[2].set_ylabel('PC2')
    axs[2].set_title('PC2 vs PC3')
    axs[2].set_xlim(-20,20)
    axs[2].set_ylim(-20,20)
    return transformed

def plot_curvature_torsion_functions(curvature, torsion, output_base_name, plot_file_name):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot Curvature
    axs[0].plot(curvature, label='Curvature', color='blue')
    axs[0].set_title('Curvature along the Filament')
    axs[0].set_xlabel('Position along filament')
    axs[0].set_ylabel('Curvature')
    axs[0].legend()
    axs[0].set_ylim(0.0,0.015)

    # Plot Torsion
    axs[1].plot(torsion, label='Torsion', color='red')
    axs[1].set_title('Torsion along the Filament')
    axs[1].set_xlabel('Position along filament')
    axs[1].set_ylabel('Torsion')
    axs[1].legend()
    axs[1].set_ylim(-3,3)

    plt.tight_layout()
    plt.savefig(f"{output_base_name}{plot_file_name}_curvature_torsion_functions.png")
    plt.close()

def interpolate_segment(segment, pc1_range=(-300, 300)):
    # Extract PC1, PC2, and PC3
    pc1, pc2, pc3 = segment[:,0], segment[:,1], segment[:,2]
    
    # Create an interpolation function for PC2 and PC3
    interp_pc2 = interp1d(pc1, pc2, bounds_error=False, fill_value=np.nan)
    interp_pc3 = interp1d(pc1, pc3, bounds_error=False, fill_value=np.nan)

    # Define integer steps within the range of PC1 in the segment
    steps = np.arange(max(np.floor(min(pc1)), pc1_range[0]), min(np.ceil(max(pc1)), pc1_range[1]) + 1)

    # Interpolate PC2 and PC3 at these steps
    pc2_interpolated = interp_pc2(steps)
    pc3_interpolated = interp_pc3(steps)

    return steps, pc2_interpolated, pc3_interpolated

def aggregate_interpolations(segments):
    pc1_range = (-300, 300)
    values_at_steps = {i: [] for i in range(pc1_range[0], pc1_range[1]+1)}

    # Interpolate each segment and collect values
    for segment in segments:
        steps, pc2s, pc3s = interpolate_segment(segment, pc1_range)
        for i, pc2, pc3 in zip(steps, pc2s, pc3s):
            if not np.isnan(pc2) and not np.isnan(pc3):  # Check for NaN to avoid adding non-interpolated points
                values_at_steps[i].append((pc2, pc3))
    
    # Compute means and standard deviations
    means = {}
    std_devs = {}
    for step in values_at_steps:
        if values_at_steps[step]:
            pc2s, pc3s = zip(*values_at_steps[step])
            means[step] = (np.mean(pc2s), np.mean(pc3s))
            std_devs[step] = (np.std(pc2s), np.std(pc3s))

    mean_plotting_points = []
    std_plotting_points = []
    for key in means.keys():
        mean_plotting_points.append([key, means[key][0], means[key][1]])
        std_plotting_points.append([key, std_devs[key][0], std_devs[key][1]])

    return np.asarray(mean_plotting_points), np.asarray(std_plotting_points)

def simple_peak_detect(y):
    # Calculate the derivative
    dy = np.diff(y)
    # Identify zero-crossings
    signs = np.sign(dy)
    zero_crossings = np.where(np.diff(signs))[0]
    return zero_crossings + 1

def choose_smallest_derivative(y, exclude_region=None, exclude_radius=15):
    dy = np.abs(np.diff(y))
    
    if exclude_region is not None:
        exclude_indices = np.arange(max(0, exclude_region - exclude_radius), min(len(dy), exclude_region + exclude_radius))
        dy[exclude_indices] = np.inf

    min_derivative_index = np.argmin(dy)
    return min_derivative_index + 1

def analyze_segment(segment, skip_peaks=0, skip_peaks_pc3=0):
    # Assuming segment is an array with rows as [PC1, PC2, PC3]
    pc1, pc2, pc3 = segment[:, 0], segment[:, 1], segment[:, 2]
    zero_index = np.argmin(np.abs(pc1))
    # Find peaks in PC2
    peaks_pc2 = simple_peak_detect(pc2)
    mask = np.abs(peaks_pc2 - zero_index) <= 15
    if np.any(mask):
        peaks_pc2 = np.delete(peaks_pc2, np.where(mask))

    if len(peaks_pc2) > 0:
        closest_peak_index_pc2 = peaks_pc2[np.argsort(np.abs(pc1[peaks_pc2] - pc1[zero_index]))[skip_peaks]]
        wavelength_pc2 = 2 * np.abs(pc1[closest_peak_index_pc2] - pc1[zero_index])
        amplitude_pc2 = np.abs(pc2[closest_peak_index_pc2] - pc2[zero_index])
        pc1_peak_pc2 = pc1[closest_peak_index_pc2]
    else:
        peaks_pc2 = simple_peak_detect(pc2)
        smallest_derivative_index = choose_smallest_derivative(pc2,peaks_pc2)
        wavelength_pc2 = 2 * np.abs(pc1[smallest_derivative_index] - pc1[zero_index])
        amplitude_pc2 = np.abs(pc2[smallest_derivative_index] - pc2[zero_index])
        pc1_peak_pc2 = pc1[smallest_derivative_index]
    
    peaks_pc3 = simple_peak_detect(pc3)
    if len(peaks_pc3) > 1:
            sorted_indices = np.argsort(np.abs(pc1[peaks_pc3] - pc1[zero_index]))
            closest_peaks_pc3 = peaks_pc3[sorted_indices][:2]
            wavelengths_pc3 = 2*np.abs(pc1[closest_peaks_pc3[0]] - pc1[closest_peaks_pc3[1]])
            amplitudes_pc3 = np.abs(pc3[closest_peaks_pc3[0]] - pc3[closest_peaks_pc3[1]])
            pc1_peaks_pc3 = pc1[closest_peaks_pc3]
    else:
            wavelengths_pc3 = None
            amplitudes_pc3 = None
            pc1_peaks_pc3 = []

    if len(pc1_peaks_pc3) >0 and wavelength_pc2 != None and wavelengths_pc3 != None:
        fractional_offset = np.mean(np.abs(pc1_peaks_pc3)) / np.mean([wavelength_pc2, wavelengths_pc3])
    else:
        fractional_offset = None

    if len(pc1_peaks_pc3) >0 and wavelength_pc2 != None and wavelengths_pc3 != None:
        offset = np.mean(np.abs(pc1_peaks_pc3))
    else:
        offset = None

    return {
            'Wavelength PC2': wavelength_pc2,
            'Amplitude PC2': amplitude_pc2,
            'PC1 Peak PC2': pc1_peak_pc2,
            'Wavelength PC3': wavelengths_pc3,
            'Amplitude PC3': amplitudes_pc3,
            'PC1 Peaks PC3': pc1_peaks_pc3,
            'Fractional offset': fractional_offset,
            'absolute offset': offset
        }


################################################################################
base_name = '/rugpfs/fs0/cem/store/mreynolds/invitro_squig_tomos/denoise_tomos_adv/fine_sampling_squig_tomos/'
#trace_file_name = base_name + 'manual_traces/tension_16/tension_17_fil1_smoothed.cmm'#sorted(glob.glob(base_name + 'denoised/tension_17*'))
trace_file_names = sorted(glob.glob(base_name + 'manual_traces/*/*smoothed.cmm'))
output_base_name = '/rugpfs/fs0/cem/store/mreynolds/scripts/in_development/denoising_squig_tomos/temp/wavelet/'

selected_segments_tension = []
cntr=0
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(1, 4)
axs = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2:4])]
axs[0].hlines(0,-150,150, colors='gray', linestyles='dashed'); axs[0].vlines(0,-150,150, colors='gray', linestyles='dashed')
axs[1].hlines(0,-150,150, colors='gray', linestyles='dashed'); axs[1].vlines(0,-150,150, colors='gray', linestyles='dashed')
axs[2].hlines(0,-150,150, colors='gray', linestyles='dashed'); axs[2].vlines(0,-150,150, colors='gray', linestyles='dashed')
for i in (46,47,49,52,53,56,59):
    squig_coords = load_cmm_data(trace_file_names[i])
    plot_file_name = trace_file_names[i].split('/')[-1][:-4]
    squig_coords *= 9.6 / 10  # Convert to nm

    # Applying wavelet transform to each dimension
    curvature, torsion = compute_curvature_torsion(squig_coords)
    coeffs_curvature, coeffs_torsion = wavelet_analysis_curvature_torsion(curvature, torsion)
    plot_wavelet_coeffs_together(coeffs_curvature, coeffs_torsion, np.arange(1, 128), output_base_name, plot_file_name)
    plot_curvature_torsion_functions(curvature, torsion, output_base_name, plot_file_name)

    boundaries = [[75,500],[75,450],[25,300],[0,450],[100,500],[50,400],[0,350]]#150, 425
    segment_of_interest = squig_coords[boundaries[cntr][0]:boundaries[cntr][1]]
    #segment_of_interest = squig_coords[boundaries[0][0]:boundaries[0][1]]
    apply_pca_and_plot(segment_of_interest, output_base_name, plot_file_name, 'purple')
    transformed_segment = apply_pca_and_plot_centered(segment_of_interest, output_base_name, plot_file_name, fig, axs, 'purple')
    selected_segments_tension.append(transformed_segment)
    cntr= cntr+1

means, std_devs = aggregate_interpolations(selected_segments_tension)
axs[0].plot(means[:,1][118:388], means[:,0][118:388], linewidth=8, c='purple')
axs[1].plot(means[:,2][118:388], means[:,0][118:388], linewidth=8, c='purple')
axs[2].plot(means[:,2][118:388], means[:,1][118:388], linewidth=8, c='purple')#[103:403]
plt.tight_layout()
plt.savefig(f"{output_base_name}tension_overlaid_PCA_subplots_centered.png")
plt.savefig(f"{output_base_name}tension_overlaid_PCA_subplots_centered.svg")
plt.close()

tension_stats = []
for i in range(0, len(selected_segments_tension)):
   tension_stats.append(analyze_segment(selected_segments_tension[i]))

tension_avg_stats = analyze_segment(means,2)

selected_segments_compression = []
cntr=0
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(1, 4)
axs = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2:4])]
axs[0].hlines(0,-150,150, colors='gray', linestyles='dashed'); axs[0].vlines(0,-150,150, colors='gray', linestyles='dashed')
axs[1].hlines(0,-150,150, colors='gray', linestyles='dashed'); axs[1].vlines(0,-150,150, colors='gray', linestyles='dashed')
axs[2].hlines(0,-150,150, colors='gray', linestyles='dashed'); axs[2].vlines(0,-150,150, colors='gray', linestyles='dashed')
for i in (3,7,9,10,23,25,30,35,36,38):#TODO Finish from 10 to 42
    squig_coords = load_cmm_data(trace_file_names[i])
    plot_file_name = trace_file_names[i].split('/')[-1][:-4]
    squig_coords *= 9.6 / 10  # Convert to nm

    # Applying wavelet transform to each dimension
    curvature, torsion = compute_curvature_torsion(squig_coords)
    coeffs_curvature, coeffs_torsion = wavelet_analysis_curvature_torsion(curvature, torsion)
    plot_wavelet_coeffs_together(coeffs_curvature, coeffs_torsion, np.arange(1, 128), output_base_name, plot_file_name)
    plot_curvature_torsion_functions(curvature, torsion, output_base_name, plot_file_name)

    boundaries = [[450,800],[420,850],[350,750],[50,550],[500,850],[1000,1450],[0,400],[550,1050],[0,375],[100,450]]#150, 425
    segment_of_interest = squig_coords[boundaries[cntr][0]:boundaries[cntr][1]]
    #segment_of_interest = squig_coords[boundaries[0][0]:boundaries[0][1]]
    apply_pca_and_plot(segment_of_interest, output_base_name, plot_file_name, 'green')
    transformed_segment = apply_pca_and_plot_centered(segment_of_interest, output_base_name, plot_file_name, fig, axs, 'green')
    selected_segments_compression.append(transformed_segment)
    cntr= cntr+1

means, std_devs = aggregate_interpolations(selected_segments_compression)
axs[0].plot(means[:,1][145:416], means[:,0][145:416], linewidth=8, c='green')
axs[1].plot(means[:,2][145:416], means[:,0][145:416], linewidth=8, c='green')
axs[2].plot(means[:,2][145:416], means[:,1][145:416], linewidth=8, c='green')#[129:430]
plt.tight_layout()
plt.savefig(f"{output_base_name}compression_overlaid_PCA_subplots_centered.png")
plt.savefig(f"{output_base_name}compression_overlaid_PCA_subplots_centered.svg")
plt.close()

compression_stats = []
for i in range(0, len(selected_segments_compression)):
    compression_stats.append(analyze_segment(selected_segments_compression[i]))

compression_avg_stats = analyze_segment(means)

tension_stats

print('hi')