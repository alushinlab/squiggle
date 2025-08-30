from glob import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import os.path as op
import pickle
import pymannkendall as mk
import sh
import scipy.signal as ssi
import scipy.stats as sst
import scipy.optimize as so
import statistics as stat
from ellipse import LsqEllipse


def compute_props(fname_coords, fname_output, t_start, t_end, dim='3d', save_fig=False, check_length=False):
    '''
    Use window = 300 for long filament and contour length for short filament
    '''
    if save_fig and dim == '3d':
        folder_pc12 = op.join(op.dirname(fname_coords), '3d_window_pc12')
        sh.mkdir('-p', folder_pc12)
        folder_pc23 = op.join(op.dirname(fname_coords), '3d_window_pc23')
        sh.mkdir('-p', folder_pc23)

    if save_fig and dim == '2d_xz':
        folder_pc12 = op.join(op.dirname(fname_coords), '2d_xz_window_pc12')
        sh.mkdir('-p', folder_pc12)

    if save_fig and dim == '2d_yz':
        folder_pc12 = op.join(op.dirname(fname_coords), '2d_yz_window_pc12')
        sh.mkdir('-p', folder_pc12)

    with open(fname_coords, 'rb') as f:
        pkl = pickle.load(f)
    pos_tic = pkl['pos_tic']
    assert dim in ['3d', '2d_xz', '2d_yz']
    if dim == '3d':
        pos_tic == pos_tic
    elif dim == '2d_xz':
        pos_tic = pos_tic[:, :, ::2]
    elif dim == '2d_yz':
        pos_tic = pos_tic[:, :, 1:]

    # Trace center axis of the filament
    # Use _j to index the anchor points and center points
    # j = i - 2
    # Use _T to index selected frames
    print('Tracing centers')
    center_Tjc = []
    for t, pos_ic in enumerate(pos_tic[t_start:t_end]):
        anchor_jc = [(pos_ic[i - 1] + pos_ic[i + 1]) / 2 for i, pos_c in enumerate(pos_ic[1:-1, :], start=1)]
        center_jc = [(pos_c + anchor_c) / 2 for pos_c, anchor_c in zip(pos_ic[1:-1, :], anchor_jc)]
        center_Tjc.append(center_jc)
    center_Tjc = np.asarray(center_Tjc)

    amp_T = []
    pitch_T = []
    ecc_T = []
    center0_jc = center_Tjc[0]
    for T, center_jc in enumerate(center_Tjc):
        if check_length:
            length_current = center_jc[-1, -1] - center_jc[0, -1]
            length_initial = center0_jc[-1, -1] - center0_jc[0, -1]
            length_change = (length_initial - length_current) / length_initial
            if length_change > 0.5:
                break

        # Compute cumulative contour length
        # Use _h to index each segment. h = j - 1
        seg_h = np.linalg.norm(np.diff(center_jc, axis=0), axis=1)
        contour_j = np.concatenate(([0], np.cumsum(seg_h)))

        window = 300
        # window = contour_j[-1]
        step = 25
        start = 0
        # Use _w to index window
        diff_w = []
        half_pitch_w = []
        idx_pair_w = []
        proj1_wJ = []
        proj2_wJ = []
        proj3_wJ = []
        while start + window <= contour_j[-1]:
            end = start + window
            # Use _J to index centers within the window
            idx_J = np.where((contour_j >= start) & (contour_j <= end))
            center_Jc = center_jc[idx_J]
            # Perform PCA on the centers within the window
            center_Jc -= center_Jc.mean(axis=0)
            cov_window_cc = np.dot(center_Jc.T, center_Jc) / len(center_Jc)
            # Use _C to index principal component
            w_window_C, v_window_cC = np.linalg.eigh(cov_window_cc)
            proj_all_JC = center_Jc @ v_window_cC
            proj1_J = proj_all_JC[:, -1]
            proj2_J = proj_all_JC[:, -2]
            if dim == '3d':
                proj3_J = proj_all_JC[:, -3]
                proj3_wJ.append(proj3_J)
            proj1_wJ.append(proj1_J)
            proj2_wJ.append(proj2_J)
            # Find peaks and troughs of PC2 projections
            # Use _p to index peak and _v to index trough
            peak_p, _ = ssi.find_peaks(proj2_J)
            trough_v, _ = ssi.find_peaks(-proj2_J)
            if len(peak_p) == 0 or len(trough_v) == 0:
                diff, half_pitch = 0, 0
                diff_w.append(diff)
                half_pitch_w.append(half_pitch)
                idx_pair = ()
                idx_pair_w.append(idx_pair)
            else:
                short_s, long_l = (trough_v, peak_p) if len(peak_p) >= len(trough_v) else (peak_p, trough_v)
                diff_s = []
                half_pitch_s = []
                idx_pair_s = []
                for short in short_s:
                    # For each index in the short list, find its two closest neighbors from left and right in the long list
                    # Compute the difference between peak and trough. Record the maximum
                    long_left_x = [long for long in long_l if long < short]
                    if long_left_x:
                        closest_left = min(long_left_x, key=lambda long_left: abs(long_left - short))
                        diff_left = abs(proj2_J[short] - proj2_J[closest_left])
                    else:
                        diff_left = 0
                    long_right_y = [long for long in long_l if long > short]
                    if long_right_y:
                        closest_right = min(long_right_y, key=lambda long_right: abs(long_right - short))
                        diff_right = abs(proj2_J[short] - proj2_J[closest_right])
                    else:
                        diff_right = 0
                    diff_s.append(max(diff_left, diff_right))
                    neighbor = closest_left if diff_left >= diff_right else closest_right
                    half_pitch = abs(proj1_J[short] - proj1_J[neighbor])
                    half_pitch_s.append(half_pitch)
                    idx_pair = (min(short, neighbor), max(short, neighbor))
                    idx_pair_s.append(idx_pair)
                # Record the largest diff within the window and the correspoding half pitch
                diff_w.append(max(diff_s))
                half_pitch_w.append(half_pitch_s[np.argmax(diff_s)])
                idx_pair_w.append(idx_pair_s[np.argmax(diff_s)])

            start += step

        # Record the largest amplitude for each frame and the corresponding pitch
        assert len(diff_w) == len(half_pitch_w) == len(idx_pair_w) == len(proj1_wJ) == len(proj2_wJ)
        amp = max(diff_w) / 2
        amp_T.append(amp)
        window_idx = np.argmax(diff_w)
        pitch = half_pitch_w[window_idx] * 2
        pitch_T.append(pitch)

        # Compute the eccentricity of superhelical cross section
        if dim == '3d':
            proj1_J = proj1_wJ[window_idx]
            proj2_J = proj2_wJ[window_idx]
            proj3_J = proj3_wJ[window_idx]
            if idx_pair_w[window_idx]:
                idx_half_low, idx_half_high = idx_pair_w[window_idx]
                proj1_half_low, proj1_half_high = proj1_J[idx_half_low], proj1_J[idx_half_high]
                proj12_J = np.concatenate((proj1_J[:, None], proj2_J[:, None]), axis=1)
                proj1_mid = min(proj12_J[idx_half_low:idx_half_high], key=lambda proj: abs(proj[1]))[0]
                proj1_full_low = proj1_half_low - (proj1_mid - proj1_half_low)
                proj1_full_high = proj1_half_high + (proj1_half_high - proj1_mid)
                # Use _k to index projections within one full pitch
                idx_k = np.where((proj1_J >= proj1_full_low) & (proj1_J <= proj1_full_high))
                proj2_k = proj2_J[idx_k]
                proj3_k = proj3_J[idx_k]
                # Fit an ellipse to the PC2 & 3 plane
                # Skip fitting cross section with not enough points
                if len(idx_k[0]) >= 80:
                    proj23_k = np.concatenate((proj2_k[:, None], proj3_k[:, None]), axis=1)
                    reg = LsqEllipse().fit(proj23_k)
                    center, width, height, phi = reg.as_parameters()
                    long_axis = max(width, height)
                    short_axis = min(width, height)
                    ecc = math.sqrt(1 - (short_axis / long_axis) ** 2)
                    print('t: %d, T: %d, amp: %.2f, pitch: %.2f, ecc: %.2f' % (t_start + T, T, amp, pitch, ecc))
                else:
                    ecc = None
                    print('t: %d, T: %d, amp: %.2f, pitch: %.2f, ecc: %s' % (t_start + T, T, amp, pitch, ecc))
            else:
                ecc = None
            ecc_T.append(ecc)

            if save_fig:
                # Plot PC2 vs PC1 within the window
                # Mark peak and trough corresponding to the maximum amplitude
                plt.figure(figsize=(5, 4), num=1, clear=True)
                plt.scatter(proj1_J, proj2_J)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.tight_layout()
                if idx_pair_w[window_idx]:
                    proj2_half_low, proj2_half_high = proj2_J[idx_half_low], proj2_J[idx_half_high]
                    plt.plot(proj1_half_low, proj2_half_low, 'x', color='C1')
                    plt.plot(proj1_half_high, proj2_half_high, 'x', color='C1')
                    plt.vlines(proj1_half_low, min(0, proj2_half_low), max(0, proj2_half_low), color='C2')
                    plt.vlines(proj1_half_high, min(0, proj2_half_high), max(0, proj2_half_high), color='C2')
                    plt.hlines(0, proj1_half_low, proj1_half_high, color='C3')
                figname_pc12 = op.join(folder_pc12, 't%04d.png' % (t_start + T))
                plt.savefig(figname_pc12)

                # Plot PC3 vs PC2 within one full pitch overlaid with the fit ellipse
                if idx_pair_w[window_idx]:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5), num=1, clear=True)
                    ax.scatter(proj2_k, proj3_k, s=5)
                    if len(idx_k[0]) >= 80:
                        ellipse = Ellipse(
                            xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
                            edgecolor='#ff7f0e', fc='None', lw=2, label='Fit', zorder=2
                        )
                        ax.add_patch(ellipse)
                    ax.set_xlim([-100, 100])
                    ax.set_ylim([-100, 100])
                    ax.set_xlabel('PC2 (nm)', fontsize=16)
                    ax.set_ylabel('PC3 (nm)', fontsize=16)
                    plt.tight_layout()
                    figname_pc23 = op.join(folder_pc23, 't%04d.png' % (t_start + T))
                    plt.savefig(figname_pc23)

        elif dim == '2d_xz' or '2d_yz':
            proj1_J = proj1_wJ[window_idx]
            proj2_J = proj2_wJ[window_idx]
            if idx_pair_w[window_idx]:
                idx_half_low, idx_half_high = idx_pair_w[window_idx]
                proj1_half_low, proj1_half_high = proj1_J[idx_half_low], proj1_J[idx_half_high]
                print('t: %d, T: %d, amp: %.2f, pitch: %.2f' % (t_start + T, T, amp, pitch))

            if save_fig:
                # Plot PC2 vs PC1 within the window
                # Mark peak and trough corresponding to the maximum amplitude
                plt.figure(figsize=(5, 4), num=1, clear=True)
                plt.scatter(proj1_J, proj2_J)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.tight_layout()
                if idx_pair_w[window_idx]:
                    proj2_half_low, proj2_half_high = proj2_J[idx_half_low], proj2_J[idx_half_high]
                    plt.plot(proj1_half_low, proj2_half_low, 'x', color='C1')
                    plt.plot(proj1_half_high, proj2_half_high, 'x', color='C1')
                    plt.vlines(proj1_half_low, min(0, proj2_half_low), max(0, proj2_half_low), color='C2')
                    plt.vlines(proj1_half_high, min(0, proj2_half_high), max(0, proj2_half_high), color='C2')
                    plt.hlines(0, proj1_half_low, proj1_half_high, color='C3')
                figname_pc12 = op.join(folder_pc12, 't%04d.png' % (t_start + T))
                plt.savefig(figname_pc12)

    if dim == '3d':
        assert len(amp_T) == len(pitch_T) == len(ecc_T)
        output = dict(t_start=t_start, t_end=t_end, T=T, amp_T=amp_T, pitch_T=pitch_T, ecc_T=ecc_T)
    elif dim == '2d_xz' or '2d_yz':
        assert len(amp_T) == len(pitch_T)
        output = dict(t_start=t_start, t_end=t_end, T=T, amp_T=amp_T, pitch_T=pitch_T)

    print('Saving props to pkl')
    with open(fname_output, 'wb') as f:
        pickle.dump(output, f, protocol=2)


def compute_props_for_all(fname_coords_all, condition, dim='3d'):
    assert dim in ['3d', '2d_xz', '2d_yz']
    assert condition in ['constant', 'release', 'thermal', 'helical_params', 'flexural_rigidity']
    if condition == 'thermal':
        for fname_coords in sorted(glob(fname_coords_all)):
            print(fname_coords)
            fname_output = op.join(op.dirname(fname_coords), 'props_' + dim + '_' + condition + '.pkl')
            compute_props(fname_coords, fname_output, 0, 5000, dim)

    elif condition == 'constant':
        for fname_coords in sorted(glob(fname_coords_all)):
            print(fname_coords)
            with open(fname_coords, 'rb') as f:
                pkl = pickle.load(f)
            t_end = pkl['force_duration']
            fname_output = op.join(op.dirname(fname_coords), 'props_' + dim + '_' + condition + '.pkl')
            compute_props(fname_coords, fname_output, 0, t_end, dim)

    elif condition == 'release':
        for fname_coords in sorted(glob(fname_coords_all)):
            print(fname_coords)
            with open(fname_coords, 'rb') as f:
                pkl = pickle.load(f)
            t_start = pkl['force_duration']
            t_end = t_start + pkl['relax_duration']
            fname_output = op.join(op.dirname(fname_coords), 'props_' + dim + '_' + condition + '.pkl')
            compute_props(fname_coords, fname_output, t_start, t_end, dim)

    if condition == 'helical_params':
        for fname_coords in sorted(glob(fname_coords_all)):
            print(fname_coords)
            fname_output = op.join(op.dirname(fname_coords), 'props_' + dim + '.pkl')
            compute_props(fname_coords, fname_output, 0, 5000, dim, True, True)

    if condition == 'flexural_rigidity':
        for fname_coords in sorted(glob(fname_coords_all)):
            print(fname_coords)
            with open(fname_coords, 'rb') as f:
                pkl = pickle.load(f)
            pos_tic = pkl['pos_tic']
            t_end = len(pos_tic) - 1
            fname_output = op.join(op.dirname(fname_coords), 'props_' + dim + '.pkl')
            compute_props(fname_coords, fname_output, 0, t_end, dim, True, True)


def plot_box(amp_low=8, pitch_low=75, pitch_high=350):
    '''Use amp_low=7 for 2D data, amp_low=8 for 3D data'''
    def compute_fraction(fname):
        '''Compute the squiggle fraction with the cutoffs applied'''
        with open(fname, 'rb') as f:
            pkl = pickle.load(f)
        amp_t = pkl['amp_T']
        pitch_t = pkl['pitch_T']

        pitch_T = []
        for pitch, amp in zip(pitch_t, amp_t):
            if amp >= amp_low and pitch >= pitch_low and pitch <= pitch_high:
                pitch_T.append(pitch)

        fraction = len(pitch_T) / len(pitch_t)
        return fraction

    def compute_fraction_for_all(dname, fname_props):
        '''Compute squiggle fraction for all the seeds under one condition'''
        print(dname)
        fraction_s = []
        fname_all = op.join(dname, 'seed*', fname_props)
        for fname in sorted(glob(fname_all)):
            fraction = compute_fraction(fname)
            fraction_s.append(fraction)

        return fraction_s

    # Plot boxplot for thermal fluctuation
    fraction_thermal0_s = compute_fraction_for_all('output/thermal_fluct/fix0', 'props_3d_thermal.pkl')
    fraction_thermal1_s = compute_fraction_for_all('output/thermal_fluct/fix1', 'props_3d_thermal.pkl')
    fraction_thermal2_s = compute_fraction_for_all('output/thermal_fluct/fix2', 'props_3d_thermal.pkl')

    plt.figure(figsize=(4, 3))
    plt.boxplot([fraction_thermal0_s, fraction_thermal1_s, fraction_thermal2_s],
                labels=['Ends free', 'Fix one', 'Fix two'])
    plt.ylim(0, 1)
    plt.ylabel('Fraction')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/thermal_fluct/box_3d_thermal.png')
    # plt.savefig('output/thermal_fluct/box_2d_xz_thermal.png')
    # plt.savefig('output/thermal_fluct/box_2d_yz_thermal.png')
    plt.close()

    # Plot boxplot for constant force
    fraction_pullc_f0p5_s = compute_fraction_for_all('output/force_release/pull_f0.5', 'props_3d_constant.pkl')
    fraction_pullc_f1p0_s = compute_fraction_for_all('output/force_release/pull_f1.0', 'props_3d_constant.pkl')
    fraction_pullc_f1p5_s = compute_fraction_for_all('output/force_release/pull_f1.5', 'props_3d_constant.pkl')
    fraction_pullc_f2p0_s = compute_fraction_for_all('output/force_release/pull_f2.0', 'props_3d_constant.pkl')
    fraction_pushc_f0p5_s = compute_fraction_for_all('output/force_release/push_f0.5', 'props_3d_constant.pkl')
    fraction_pushc_f1p0_s = compute_fraction_for_all('output/force_release/push_f1.0', 'props_3d_constant.pkl')
    fraction_pushc_f1p5_s = compute_fraction_for_all('output/force_release/push_f1.5', 'props_3d_constant.pkl')
    fraction_pushc_f2p0_s = compute_fraction_for_all('output/force_release/push_f2.0', 'props_3d_constant.pkl')

    plt.figure(figsize=(4, 3))
    plt.boxplot([fraction_pullc_f0p5_s, fraction_pullc_f1p0_s, fraction_pullc_f1p5_s,
                 fraction_pullc_f2p0_s, fraction_pushc_f0p5_s, fraction_pushc_f1p0_s,
                 fraction_pushc_f1p5_s, fraction_pushc_f2p0_s],
                labels=['10', '20', '30', '40', '10', '20', '30', '40'])
    plt.ylim(0, 1)
    plt.xlabel('Force (pN)')
    plt.ylabel('Fraction')
    plt.tight_layout()
    plt.savefig('output/force_release/box_3d_constant.png')
    # plt.savefig('output/force_release/box_2d_xz_constant.png')
    # plt.savefig('output/force_release/box_2d_yz_constant.png')
    plt.close()

    # Plot boxplot for force release
    fraction_pullr_f0p5_s = compute_fraction_for_all('output/force_release/pull_f0.5', 'props_3d_release.pkl')
    fraction_pullr_f1p0_s = compute_fraction_for_all('output/force_release/pull_f1.0', 'props_3d_release.pkl')
    fraction_pullr_f1p5_s = compute_fraction_for_all('output/force_release/pull_f1.5', 'props_3d_release.pkl')
    fraction_pullr_f2p0_s = compute_fraction_for_all('output/force_release/pull_f2.0', 'props_3d_release.pkl')
    fraction_pushr_f0p5_s = compute_fraction_for_all('output/force_release/push_f0.5', 'props_3d_release.pkl')
    fraction_pushr_f1p0_s = compute_fraction_for_all('output/force_release/push_f1.0', 'props_3d_release.pkl')
    fraction_pushr_f1p5_s = compute_fraction_for_all('output/force_release/push_f1.5', 'props_3d_release.pkl')
    fraction_pushr_f2p0_s = compute_fraction_for_all('output/force_release/push_f2.0', 'props_3d_release.pkl')

    plt.figure(figsize=(4, 3))
    plt.boxplot([fraction_pullr_f0p5_s, fraction_pullr_f1p0_s, fraction_pullr_f1p5_s,
                 fraction_pullr_f2p0_s, fraction_pushr_f0p5_s, fraction_pushr_f1p0_s,
                 fraction_pushr_f1p5_s, fraction_pushr_f2p0_s],
                labels=['10', '20', '30', '40', '10', '20', '30', '40'])
    plt.ylim(0, 1)
    plt.xlabel('Force (pN)')
    plt.ylabel('Fraction')
    plt.tight_layout()
    plt.savefig('output/force_release/box_3d_release.png')
    # plt.savefig('output/force_release/box_2d_xz_release.png')
    # plt.savefig('output/force_release/box_2d_yz_release.png')
    plt.close()


def plot_violin(dname_all, fname_props, amp_low=8, pitch_low=75, pitch_high=350):
    amp_ft = []
    pitch_fp = []
    ecc_fe = []
    for dname in sorted(glob(dname_all)):
        fname_all = op.join(dname, 'seed*', fname_props)
        amp_t = []
        pitch_t = []
        ecc_t = []
        for fname in sorted(glob(fname_all)):
            print(fname)
            with open(fname, 'rb') as f:
                pkl = pickle.load(f)
            amp_t += pkl['amp_T']
            pitch_t += pkl['pitch_T']
            ecc_t += pkl['ecc_T']

        pitch_p = []
        for pitch, amp in zip (pitch_t, amp_t):
            if amp >= amp_low and pitch >= pitch_low and pitch <= pitch_high:
                pitch_p.append(pitch)

        ecc_e = []
        for pitch, amp, ecc in zip (pitch_t, amp_t, ecc_t):
            if amp >= amp_low and pitch >= pitch_low and pitch <= pitch_high and ecc != None:
                ecc_e.append(ecc)

        amp_ft.append(amp_t)
        pitch_fp.append(pitch_p)
        ecc_fe.append(ecc_e)

    print('amp', len(min(amp_ft, key=len)), len(max(amp_ft, key=len)))
    print('pitch', len(min(pitch_fp, key=len)), len(max(pitch_fp, key=len)))
    print('ecc', len(min(ecc_fe, key=len)), len(max(ecc_fe, key=len)))

    # Plot amplitude
    plt.figure(figsize=(4, 3))
    plt.violinplot(amp_ft, showextrema=False, showmedians=True)

    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=['Ends free', 'Fix one', 'Fix two'])

    plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[10, 20, 30, 40])
    plt.xlabel('Force (pN)')

    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[1.78, 2.28, 2.78, 3.28, 3.78])
    # plt.xlabel('Rise (nm)')

    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[136.67, 146.67, 156.67, 166.67, 176.67])
    # plt.xlabel('twist (\xb0)')

    # plt.xticks([i + 1 for i in range(len(amp_ft))])
    # plt.xlabel('Rigidity')

    plt.ylabel('Amplitude (nm)')
    plt.tight_layout()
    fig_name = op.join(op.dirname(dname_all), 'violin_3d_amp.png')
    plt.savefig(fig_name)
    plt.close()

    # Plot pitch
    plt.figure(figsize=(4, 3))
    plt.violinplot(pitch_fp, showextrema=False, showmedians=True)

    # plt.xticks([i + 1 for i in range(len(pitch_fp))], labels=['Ends free', 'Fix one', 'Fix two'])

    plt.xticks([i + 1 for i in range(len(pitch_fp))], labels=[10, 20, 30, 40])
    plt.xlabel('Force (pN)')

    # plt.xticks([i + 1 for i in range(len(pitch_fp))], labels=[1.78, 2.28, 2.78, 3.28, 3.78])
    # plt.xlabel('Rise (nm)')

    # plt.xticks([i + 1 for i in range(len(pitch_fp))], labels=[136.67, 146.67, 156.67, 166.67, 176.67])
    # plt.xlabel('twist (\xb0)')

    # plt.xticks([i + 1 for i in range(len(pitch_fp))])
    # plt.xlabel('Rigidity')

    plt.ylabel('Pitch (nm)')
    plt.tight_layout()
    fig_name = op.join(op.dirname(dname_all), 'violin_3d_pitch.png')
    plt.savefig(fig_name)
    plt.close()

    # Plot eccentricity
    plt.figure(figsize=(4, 3))
    plt.violinplot(ecc_fe, showextrema=False, showmedians=True)

    # plt.xticks([i + 1 for i in range(len(ecc_fe))], labels=['Ends free', 'Fix one', 'Fix two'])

    plt.xticks([i + 1 for i in range(len(ecc_fe))], labels=[10, 20, 30, 40])
    plt.xlabel('Force (pN)')

    # plt.xticks([i + 1 for i in range(len(ecc_fe))], labels=[1.78, 2.28, 2.78, 3.28, 3.78])
    # plt.xlabel('Rise (nm)')

    # plt.xticks([i + 1 for i in range(len(ecc_fe))], labels=[136.67, 146.67, 156.67, 166.67, 176.67])
    # plt.xlabel('twist (\xb0)')

    # plt.xticks([i + 1 for i in range(len(ecc_fe))])
    # plt.xlabel('Rigidity')

    plt.ylabel('Eccentricity')
    plt.tight_layout()
    fig_name = op.join(op.dirname(dname_all), 'violin_3d_ecc.png')
    plt.savefig(fig_name)
    plt.close()


def plot_proj(fname_coords, condition, num_frames=5):
    assert condition in ['thermal', 'constant', 'release']

    folder_output = op.join(op.dirname(fname_coords), 'pca', condition)
    sh.mkdir('-p', folder_output)

    with open(fname_coords, 'rb') as f:
        pkl = pickle.load(f)
    pos_tic = pkl['pos_tic']

    if condition == 'thermal':
        t_start, t_end = 0, 5000
    elif condition == 'constant':
        t_start = 0
        t_end = pkl['force_duration']
    elif condition == 'release':
        t_start = pkl['force_duration']
        t_end = pkl['relax_duration']

    pos_Tic = pos_tic[t_start:t_end]
    idx_n = np.linspace(0, len(pos_Tic) - 1, num_frames, dtype=int)
    print('Selected frames:', idx_n)

    # Perform PCA for frame 0
    pos_T0_ic = pos_Tic[0]
    pos_T0_ic -= np.mean(pos_T0_ic, axis=0)
    cov_T0_cc = np.dot(pos_T0_ic.T, pos_T0_ic) / len(pos_T0_ic)
    w_T0_cC, v_T0_cC = np.linalg.eigh(cov_T0_cc)
    proj321_T0_iC = pos_T0_ic @ v_T0_cC
    proj1_T0_i = proj321_T0_iC[:, 2]
    if proj1_T0_i[-1] < 0:
        v_T0_cC[:, -1] *= -1

    proj1_ni = []
    proj2_ni = []
    proj3_ni = []
    v_ncC = [v_T0_cC]
    for n, idx in enumerate(idx_n):
        pos_ic = pos_Tic[idx]
        # Perform PCA
        pos_ic -= np.mean(pos_ic, axis=0)
        cov_cc = np.dot(pos_ic.T, pos_ic) / len(pos_ic)
        # Use _C to index principal component
        w_cC, v_cC = np.linalg.eigh(cov_cc)
        v_Cc = []
        if n > 0:
            v_last_cC = v_ncC[n - 1]
            for v_c, v_last_c in zip(v_cC.T, v_last_cC.T):
                if np.dot(v_c, v_last_c) < 0:
                    v_c *= -1
                v_Cc.append(v_c)
            v_cC = np.asarray(v_Cc).T
            v_ncC.append(v_cC)

        proj321_iC = pos_ic @ v_cC
        proj1_i = proj321_iC[:, 2]
        proj2_i = proj321_iC[:, 1]
        proj3_i = proj321_iC[:, 0]
        proj1_ni.append(proj1_i)
        proj2_ni.append(proj2_i)
        proj3_ni.append(proj3_i)

    # Plot overlaid PCA projections
    # Extract colors evenly from viridis
    cmap = cm.get_cmap('viridis')
    color_n = [cmap(i) for i in np.linspace(0, 1, num_frames)]

    # Plot PC1 vs PC2
    plt.figure(figsize=(3, 12))
    for proj1_i, proj2_i, color in zip(proj1_ni, proj2_ni, color_n):
        plt.scatter(proj2_i, proj1_i, color=color, s=5)
    plt.xlim(-65, 65)
    plt.ylim(-600, 600)
    plt.xlabel('PC2 (nm)')
    plt.ylabel('PC1 (nm)')
    plt.tight_layout()
    fig12_name = op.join(folder_output, 'pc12.png')
    plt.savefig(fig12_name)
    plt.close()

    # Plot PC1 vs PC3
    plt.figure(figsize=(3, 12))
    for proj1_i, proj3_i, color in zip(proj1_ni, proj3_ni, color_n):
        plt.scatter(proj3_i, proj1_i, color=color, s=5)
    plt.xlim(-65, 65)
    plt.ylim(-600, 600)
    plt.xlabel('PC3 (nm)')
    plt.ylabel('PC1 (nm)')
    plt.tight_layout()
    fig13_name = op.join(folder_output, 'pc13.png')
    plt.savefig(fig13_name)
    plt.close()

    # Plot PC2 vs PC3
    plt.figure(figsize=(3, 3))
    for proj2_i, proj3_i, color in zip(proj2_ni, proj3_ni, color_n):
        plt.scatter(proj3_i, proj2_i, color=color, s=5)
    plt.xlim(-65, 65)
    plt.ylim(-65, 65)
    plt.xlabel('PC3 (nm)')
    plt.ylabel('PC2 (nm)')
    plt.tight_layout()
    fig23_name = op.join(folder_output, 'pc23.png')
    plt.savefig(fig23_name)
    plt.close()


def mk_test(dname_all, fname_props, amp_low=8, pitch_low=75, pitch_high=350):
    '''
    Perform Mann-Kendall test on medians to test for monotonic trend
    '''
    amp_f = []
    pitch_f = []
    ecc_f = []
    for dname in sorted(glob(dname_all)):
        print(dname)
        fname_all = op.join(dname, 'seed*', fname_props)
        amp_t = []
        pitch_t = []
        ecc_t = []
        for fname in sorted(glob(fname_all)):
            with open(fname, 'rb') as f:
                pkl = pickle.load(f)
            amp_t += pkl['amp_T']
            pitch_t += pkl['pitch_T']
            ecc_t += pkl['ecc_T']

        pitch_p = []
        for pitch, amp in zip (pitch_t, amp_t):
            if amp >= amp_low and pitch >= pitch_low and pitch <= pitch_high:
                pitch_p.append(pitch)

        ecc_e = []
        for pitch, amp, ecc in zip (pitch_t, amp_t, ecc_t):
            if amp >= amp_low and pitch >= pitch_low and pitch <= pitch_high and ecc != None:
                ecc_e.append(ecc)

        amp_f.append(stat.median(amp_t))
        pitch_f.append(stat.median(pitch_p))
        ecc_f.append(stat.median(ecc_e))

    print('amp:', mk.original_test(amp_f))
    print('pitch:', mk.original_test(pitch_f))
    print('ecc:', mk.original_test(ecc_f))


def ecc_vs_time(dname, fname_props, amp_low=8, pitch_low=75):
    plt.figure(figsize=(4, 3))
    fname_all = op.join(dname, 'seed*', fname_props)
    for seed, fname in enumerate(sorted(glob(fname_all))):
        print(fname)
        with open(fname, 'rb') as f:
            pkl = pickle.load(f)
        amp_t = pkl['amp_T']
        pitch_t = pkl['pitch_T']
        ecc_t = pkl['ecc_T']

        ecc_e = []
        for pitch, amp, ecc in zip (pitch_t, amp_t, ecc_t):
            if amp >= amp_low and pitch >= pitch_low and ecc != None:
                ecc_e.append(ecc)

        plt.plot(range(len(ecc_e)), ecc_e, label='seed %d' % seed)
        # plt.legend(loc='lower right')

    plt.xlabel('Frame')
    plt.ylabel('Eccentricity')
    plt.tight_layout()
    fig_name = op.join(dname, 'ecc_vs_time.png')
    plt.savefig(fig_name)
    plt.close()


def plot_xy(fname_coords):
    with open(fname_coords, 'rb') as f:
        pkl = pickle.load(f)
    pos_tic = pkl['pos_tic']
    folder_output = op.join(op.dirname(fname_coords), 'xy')
    sh.mkdir('-p', folder_output)

    for T, pos_ic in enumerate(pos_tic[::10]):
        x_i = pos_ic[:, 0]
        y_i = pos_ic[:, 1]
        plt.figure(figsize=(5, 5), num=1, clear=True)
        plt.scatter(x_i, y_i, s=3)
        plt.xlim(250, 500)
        plt.ylim(250, 500)
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.tight_layout()
        figname = op.join(folder_output, 't%03d.png' % T)
        plt.savefig(figname)


def plot_ellipse_xy(fname_coords, check_length=True):
    with open(fname_coords, 'rb') as f:
        pkl = pickle.load(f)
    pos_tic = pkl['pos_tic']

    folder_output = op.join(op.dirname(fname_coords), 'ellipse_xy')
    sh.mkdir('-p', folder_output)

    # Trace center axis of the filament
    # Use _j to index the anchor points and center points
    # j = i - 2
    # Use _T to index selected frames
    print('Tracing centers')
    center_Tjc = []
    for t, pos_ic in enumerate(pos_tic):
        anchor_jc = [(pos_ic[i - 1] + pos_ic[i + 1]) / 2 for i, pos_c in enumerate(pos_ic[1:-1, :], start=1)]
        center_jc = [(pos_c + anchor_c) / 2 for pos_c, anchor_c in zip(pos_ic[1:-1, :], anchor_jc)]
        center_Tjc.append(center_jc)
    center_Tjc = np.asarray(center_Tjc)

    # Use _e to index frames with identified peaks and troughs
    idx_ek = []
    center0_jc = center_Tjc[0]
    for T, center_jc in enumerate(center_Tjc):
        if check_length:
            length_current = center_jc[-1, -1] - center_jc[0, -1]
            length_initial = center0_jc[-1, -1] - center0_jc[0, -1]
            length_change = (length_initial - length_current) / length_initial
            if length_change > 0.5:
                break

        # Perform PCA on the centers
        center_jc -= center_jc.mean(axis=0)
        cov_cc = np.dot(center_jc.T, center_jc) / len(center_jc)
        # Use _C to index principal component
        w_C, v_cC = np.linalg.eigh(cov_cc)
        proj_all_jC = center_jc @ v_cC
        proj1_j = proj_all_jC[:, -1]
        proj2_j = proj_all_jC[:, -2]
        proj3_j = proj_all_jC[:, -3]
        # Find peaks and troughs of PC2 projections
        # Use _p to index peak and _v to index trough
        peak_p, _ = ssi.find_peaks(proj2_j)
        trough_v, _ = ssi.find_peaks(-proj2_j)
        if len(peak_p) == 0 or len(trough_v) == 0:
            diff, half_pitch = 0, 0
            idx_pair = ()
        else:
            short_s, long_l = (trough_v, peak_p) if len(peak_p) >= len(trough_v) else (peak_p, trough_v)
            diff_s = []
            half_pitch_s = []
            idx_pair_s = []
            for short in short_s:
                # For each index in the short list, find its two closest neighbors from left and right in the long list
                # Compute the difference between peak and trough. Record the maximum
                long_left_x = [long for long in long_l if long < short]
                if long_left_x:
                    closest_left = min(long_left_x, key=lambda long_left: abs(long_left - short))
                    diff_left = abs(proj2_j[short] - proj2_j[closest_left])
                else:
                    diff_left = 0
                long_right_y = [long for long in long_l if long > short]
                if long_right_y:
                    closest_right = min(long_right_y, key=lambda long_right: abs(long_right - short))
                    diff_right = abs(proj2_j[short] - proj2_j[closest_right])
                else:
                    diff_right = 0
                diff_s.append(max(diff_left, diff_right))
                neighbor = closest_left if diff_left >= diff_right else closest_right
                half_pitch = abs(proj1_j[short] - proj1_j[neighbor])
                half_pitch_s.append(half_pitch)
                idx_pair = (min(short, neighbor), max(short, neighbor))
                idx_pair_s.append(idx_pair)
            # Record the largest diff and the correspoding half pitch
            diff = max(diff_s)
            half_pitch = half_pitch_s[np.argmax(diff_s)]
            idx_pair = idx_pair_s[np.argmax(diff_s)]

        if idx_pair:
            idx_half_low, idx_half_high = idx_pair
            proj1_half_low, proj1_half_high = proj1_j[idx_half_low], proj1_j[idx_half_high]
            proj12_j = np.concatenate((proj1_j[:, None], proj2_j[:, None]), axis=1)
            proj1_mid = min(proj12_j[idx_half_low:idx_half_high], key=lambda proj: abs(proj[1]))[0]
            proj1_full_low = proj1_half_low - (proj1_mid - proj1_half_low)
            proj1_full_high = proj1_half_high + (proj1_half_high - proj1_mid)
            # Use _k to index projections within one full pitch
            idx_k = np.where((proj1_j >= proj1_full_low) & (proj1_j <= proj1_full_high))
            idx_k = idx_k[0]
            idx_ek.append(idx_k)

    print('Plotting x, y positions of particles corresponding to the max amplitude')
    idx_K = max(idx_ek, key=len)
    for t, pos_ic in enumerate(pos_tic[::10]):
        pos_Kc = pos_ic[idx_K]
        x_K = pos_Kc[:, 0]
        y_K = pos_Kc[:, 1]
        plt.figure(figsize=(5, 5), num=1, clear=True, dpi=300)
        plt.scatter(x_K, y_K, s=3)
        plt.xlim(250, 500)
        plt.ylim(250, 500)
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
        plt.tight_layout()
        figname = op.join(folder_output, 't%03d.png' % t)
        plt.savefig(figname)


if __name__ == '__main__':
    # compute_props_for_all('output/helical_params_short/rise/rise*/seed*/coords.pkl', 'helical_params', '3d')
    # plot_box()
    # plot_violin('output/thermal_fluct/fix*', 'props_3d_thermal.pkl')
    # plot_violin('output/force_release/push*', 'props_3d_release.pkl')
    # plot_proj('output/force_release/push_f1.5/seed4/coords.pkl', 'constant')
    # mk_test('output/helical_params_short/twist/twist*', 'props_3d.pkl')
    # mk_test('output/flexural_rigidity_short/compression/kb*', 'props_3d.pkl')
    # ecc_vs_time('output/helical_params_short/twist/twist_166.67', 'props_3d.pkl')
    plot_ellipse_xy('output/helical_params_short/twist/twist_166.67/seed0/coords.pkl')
