from glob import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import os.path as op
import pickle
import sh
import scipy.signal as ssi
import scipy.stats as sst
import scipy.optimize as so
from ellipse import LsqEllipse


def force_vs_extension(folder_param, part_num=39, box_l=[125, 125, 250]):
    '''Plot force extension curve'''
    ind_i = np.array(list(range(part_num)))
    z0_i = 2.78 * ind_i
    z_i = z0_i + box_l[2] / 2 - 2.78 * part_num / 2

    # Use _f to index force and extension
    extension_f = []
    force_f = []
    for fname_coords in sorted(glob(op.join(folder_param, '*/*/coords.pkl'))):
        with open(fname_coords, 'rb') as f:
            pkl = pickle.load(f)
        pos_tic = pkl['pos_tic']
        force_unit = 4
        force = pkl['force_total'] * force_unit
        force_f.append(force)
        extension = pos_tic[-1][-1][-1] - z_i[-1]
        extension_f.append(extension)
    extension_f = np.array(extension_f)
    force_f = np.array(force_f)

    # Perform linear regression
    extension_mean, force_mean = np.mean(extension_f), np.mean(force_f)
    slope = np.sum((extension_f - extension_mean) * (force_f - force_mean)) / np.sum((extension_f - extension_mean) ** 2)
    intercept = force_mean - slope * extension_mean
    force_fit_f = np.array([slope*extension+intercept for extension in extension_f])
    ss_mean = np.sum((force_f - force_mean) ** 2)   # Sum of squares around the mean
    ss_fit = np.sum((force_f - force_fit_f) ** 2)   # Sum of squares around the fit
    r_squared = 1 - ss_fit / ss_mean

    param = op.basename(folder_param)
    print('%s: stiffness = %.1f, r_squared = %.4f' % (param, slope, r_squared))
    folder_fig = 'output/fig/supp1/param_scan'
    sh.mkdir('-p', folder_fig)
    fname_fig = op.join(folder_fig, 'force_vs_extension.svg')
    plt.figure(figsize=(4.5, 3))
    plt.scatter(extension_f, force_f, color='grey')
    plt.plot(extension_f, force_fit_f, color='black', label='k = %.1f pN/nm\n$R^2$ = %.4f' % (slope, r_squared))
    plt.legend(loc='upper left')
    plt.xticks(fontsize=14)
    plt.yticks([50, 100, 150, 200], fontsize=14)
    plt.xlabel('Extension (nm)', fontsize=16)
    plt.ylabel('Force (pN)', fontsize=16)
    plt.tight_layout()
    plt.savefig(fname_fig)
    plt.close()


def trace_center(fname_coords, save_2d_fig=False, save_3d_fig=False):
    '''
    Trace the linear polymer through the center axis.
    Anchor points are defined by the average position between part i - 1 and part 1 + 1.
    Center points are defined by the average postion between the anchor point and part i.
    '''
    print('Tracing centers')
    with open(fname_coords, 'rb') as f:
        pkl = pickle.load(f)
    pos_tic = pkl['pos_tic']

    # Use j to index the anchor points and center points
    # j = i - 2
    center_tjc = []
    for t, pos_ic in enumerate(pos_tic):
        anchor_jc = [(pos_ic[i - 1] + pos_ic[i + 1]) / 2 for i, pos_c in enumerate(pos_ic[1:-1, :], start=1)]
        center_jc = [(pos_c + anchor_c) / 2 for pos_c, anchor_c in zip(pos_ic[1:-1, :], anchor_jc)]
        center_tjc.append(center_jc)
    center_tjc = np.array(center_tjc)

    if save_2d_fig:
        print('Plotting 2D projections')
        folder_figs_2d = op.join(op.dirname(fname_coords), 'center_2d')
        sh.mkdir('-p', folder_figs_2d)
        for t, center_jc in enumerate(center_tjc):
            plt.figure(figsize=(5, 15), dpi=100)
            plt.xlim(0, 200)
            plt.ylim(0, 400)
            plt.xlabel('x / nm', fontsize=16)
            plt.ylabel('z / nm', fontsize=16)
            plt.plot(center_jc[:, 0], center_jc[:, 2], linewidth=3)
            plt.tight_layout()
            fig_name = op.join(folder_figs_2d, '%03d.png' % t)
            plt.savefig(fig_name)
            plt.close()

    if save_3d_fig:
        print('Plotting 3D projections')
        folder_figs_3d = op.join(op.dirname(fname_coords), 'center_3d')
        sh.mkdir('-p', folder_figs_3d)
        for t, center_jc in enumerate(center_tjc):
            ax = plt.figure(dpi=200).add_subplot(projection='3d')
            ax.set_box_aspect([1, 1, 3])
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 200)
            ax.set_zlim(0, 400)
            ax.set_xlabel('x / nm')
            ax.set_ylabel('y / nm')
            ax.set_zlabel('z / nm')
            ax.plot(center_jc[:, 0], center_jc[:, 1], center_jc[:, 2])
            plt.tight_layout()
            fig_name = op.join(folder_figs_3d, '%03d.png' % t)
            plt.savefig(fig_name)
            plt.close()

    return center_tjc


def relaxation_end_to_end(fname_coords):
    '''
    Compute the relaxation time of the linear polymer by fitting the
    autocorrelation of the end-to-end vector to an exponential decay.
    '''
    center_tjc = trace_center(fname_coords)
    # Calculate end-to-end vector
    r_tc = center_tjc[:, -1, :] - center_tjc[:, 0, :]

    # Compute autocorrelation
    r_tc = r_tc - np.mean(r_tc, axis=0)
    corr_t = ssi.correlate(r_tc[:, 0], r_tc[:, 0]) + ssi.correlate(r_tc[:, 1], r_tc[:, 1]) + ssi.correlate(r_tc[:, 2], r_tc[:, 2])
    corr_t = corr_t[len(corr_t) // 2:]
    corr_t /= corr_t[0]

    def exp_decay(t, tau):
        return np.exp(-t / tau)

    lag_t = np.arange(len(corr_t))
    popt, _ = so.curve_fit(exp_decay, lag_t, corr_t)
    tau = popt[0]
    print('Relaxation time %.1f' % tau)

    # Plot autocorrelation and fitted curve
    plt.figure(figsize=(4, 3))
    plt.plot(lag_t, corr_t)
    plt.plot(lag_t, exp_decay(lag_t, *popt))
    plt.xlabel('Lag Time (Frame)')
    plt.ylabel('Autocorrelation')
    plt.text(300, 0.8, r'$\tau$ = %.1f' % tau)
    plt.tight_layout()
    fname_fig = op.join(op.dirname(fname_coords), 'autocorr_end.png')
    plt.savefig(fname_fig)
    plt.close()

    return tau, center_tjc


def persistence_length(fname_coords):
    tau, center_tjc = relaxation_end_to_end(fname_coords)

    # Sample independent snapshots
    # Use u to index sampled snapshots
    # u = t // math.ceil(tau) + 1
    step = math.ceil(tau)
    resampled_center_ujc = center_tjc[::step]

    # Find the segment vector connecting center point i to i + 1
    # Use k to index the vector
    # k = j - 1
    seg_ukc = resampled_center_ujc[:, 1:, :] - resampled_center_ujc[:, :-1, :]
    seg_norm_ukc = seg_ukc / np.linalg.norm(seg_ukc, axis=-1)[:, :, None]   # Normalize

    # Compute cosine correlation
    corr_k = [[] for contour in range(seg_norm_ukc.shape[1])]
    for m in range(seg_norm_ukc.shape[1]):
        for n in range(m, seg_norm_ukc.shape[1]):
            #corr_k[n - m].append(np.mean(np.sum(seg_norm_ukc[:, m, :] * seg_norm_ukc[:, n, :], axis=-1)))
            corr_k[n - m].append(np.mean(np.einsum('ij,ij->i', seg_norm_ukc[:, m, :], seg_norm_ukc[:, n, :])))
    corr_k = np.array([np.mean(corr) for corr in corr_k])

    # Fit and plot the initial subunits
    # Use l to index those subunits
    contour_l = np.arange(len(corr_k[:18]))
    corr_l = np.log(corr_k[:18])

    # Perform linear regression
    contour_mean, corr_mean = np.mean(contour_l), np.mean(corr_l)
    slope = np.sum((contour_l - contour_mean) * (corr_l - corr_mean)) / np.sum((contour_l - contour_mean) ** 2)
    intercept = corr_mean - slope * contour_mean
    fit_l = np.array([slope*contour+intercept for contour in contour_l])
    p = - 2.78 / slope
    ss_mean = np.sum((corr_l - corr_mean) ** 2)   # Sum of squares around the mean
    ss_fit = np.sum((corr_l - fit_l) ** 2)   # Sum of squares around the fit
    r_squared = 1 - ss_fit / ss_mean
    print('Persistence length %.1f, r_squared %.4f' % (p, r_squared))

    folder_fig = 'output/fig/supp1/param_scan'
    sh.mkdir('-p', folder_fig)
    fname_fig = op.join(folder_fig, 'pl.svg')
    plt.figure(figsize=(5, 3))
    plt.scatter(contour_l, corr_l, color='grey')
    plt.plot(contour_l, fit_l, color='black', label='Lp = %.1f $\mu$m\nR$^2$ = %.4f' % (p / 1000, r_squared))
    plt.xticks([0, 5, 10, 15], fontsize=14)
    plt.yticks([-0.006, -0.004, -0.002, 0.000], fontsize=14)
    plt.xlabel('Subunit index', fontsize=16)
    plt.ylabel(r'ln<cos$\theta$>', fontsize=16)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fname_fig)
    plt.close()


def compute_twist_rise(fname_coords):
    '''
    Compute twist and rise based on the particle projection onto the center segments.
    '''
    with open(fname_coords, 'rb') as f:
        pkl = pickle.load(f)
    pos_tic = pkl['pos_tic']
    # Use j to index center points
    # j = i - 2
    center_tjc = trace_center(fname_coords)
    # Use k to index the center segments
    # k = j - 1 = i - 3
    center_seg_tkc = center_tjc[:, 1:, :] - center_tjc[:, :-1, :]
    # Normalize each center segment to get the unit center vector
    center_seg_unit_tkc = center_seg_tkc / np.linalg.norm(center_seg_tkc, axis=-1, keepdims=True)
    # Find the vectors connecting each subunit to its corresponding center point
    r_tjc = pos_tic[:, 1:-1, :] - center_tjc
    # Project each r vector onto the center segment vector
    # Find normals
    normal1_tkc = []
    for r_kc, center_kc, center_seg_unit_kc in zip(r_tjc[:, :-1, :], center_tjc[:, :-1, :], center_seg_unit_tkc):
        normal1_kc = []
        for r_c, center_c, center_seg_unit_c in zip(r_kc, center_kc, center_seg_unit_kc):
            normal1_c = r_c - np.dot(r_c, center_seg_unit_c) * center_seg_unit_c
            normal1_kc.append(normal1_c)
        normal1_tkc.append(normal1_kc)
    normal1_tkc = np.array(normal1_tkc)

    normal2_tkc = []
    for r_kc, center_kc, center_seg_unit_kc in zip(r_tjc[:, 1:, :], center_tjc[:, :-1, :], center_seg_unit_tkc):
        normal2_kc = []
        for r_c, center_c, center_seg_unit_c in zip(r_kc, center_kc, center_seg_unit_kc):
            normal2_c = r_c - np.dot(r_c, center_seg_unit_c) * center_seg_unit_c
            normal2_kc.append(normal2_c)
        normal2_tkc.append(normal2_kc)
    normal2_tkc = np.array(normal2_tkc)

    # Compute twist based on neighboring normals
    twist_tk = []
    for normal1_kc, normal2_kc in zip(normal1_tkc, normal2_tkc):
        twist_k = []
        for normal1_c, normal2_c in zip(normal1_kc, normal2_kc):
            twist = - np.arccos(np.dot(normal1_c, normal2_c) / (np.linalg.norm(normal1_c) * np.linalg.norm(normal2_c)))
            twist = twist / np.pi * 180
            twist_k.append(twist)
        twist_tk.append(twist_k)
    twist_tk = np.array(twist_tk)

    # Project each subunit onto the center segment
    # Find the projection point
    proj1_tkc = []
    for pos_kc, normal1_kc in zip(pos_tic[:, 1:-2, :], normal1_tkc):
        proj1_kc = []
        for pos_c, normal1_c in zip(pos_kc, normal1_kc):
            proj1_c = pos_c - normal1_c
            proj1_kc.append(proj1_c)
        proj1_tkc.append(proj1_kc)
    proj1_tkc = np.array(proj1_tkc)

    proj2_tkc = []
    for pos_kc, normal2_kc in zip(pos_tic[:, 2:-1, :], normal2_tkc):
        proj2_kc = []
        for pos_c, normal2_c in zip(pos_kc, normal2_kc):
            proj2_c = pos_c - normal2_c
            proj2_kc.append(proj2_c)
        proj2_tkc.append(proj2_kc)
    proj2_tkc = np.array(proj2_tkc)

    # Compute rise based on neighboring projection points
    proj_seg_tkc = proj2_tkc - proj1_tkc
    rise_tk = np.linalg.norm(proj_seg_tkc, axis=-1)

    # Save results
    fname_twist = op.join(op.dirname(fname_coords), 'twist.pkl')
    with open(fname_twist, 'wb') as f:
        pickle.dump(twist_tk, f, protocol=2)

    fname_rise = op.join(op.dirname(fname_coords), 'rise.pkl')
    with open(fname_rise, 'wb') as f:
        pickle.dump(rise_tk, f, protocol=2)

    return twist_tk, rise_tk


def twist_var(fname_coords, theta=-166.67, part_start=20):
    '''
    Extract variance of twist change for each subunit.
    Plot variance against subunit index.
    Return slope of linear fit.
    theta: canonical twist.
    part_start: skip the first few particles to avoid edge effect.
    '''
    twist_tk, _ = compute_twist_rise(fname_coords)
    # Use l to index selected subunits
    twist_tl = twist_tk[:, part_start:(part_start + 60)]

    delta_twist_tl = []
    for twist_l in twist_tl:
        cum_twist_l = np.cumsum(np.array(twist_l))
        delta_twist_l = []
        for (l, cum_twist) in enumerate(cum_twist_l):
            delta_twist = cum_twist - (l + 1) * theta
            delta_twist_l.append(delta_twist)
        delta_twist_tl.append(delta_twist_l)
    delta_twist_tl = np.array(delta_twist_tl)

    # Treat part[part_start - 1] as a reference, i.e., the 0th particle.
    # Plot probability density distribution of twist change of a few selected particles.
    indices = [0, 15, 30, 45, 59]
    for index in indices:
        fname_hist = op.join(op.dirname(fname_coords), 'hist_n%02d.png' % index)
        plt.figure(figsize=(4, 3))
        _, bins, _ = plt.hist(delta_twist_tl[:, index], bins=20, density=True, color='tab:blue')
        # Fit Gaussian to the histogram with mean zero
        mu = 0
        sigma = math.sqrt(np.sum([delta_twist ** 2 for delta_twist in delta_twist_tl[:, index]]) / len(delta_twist_tl[:, index]))
        fit_line = sst.norm.pdf(bins, mu, sigma)
        plt.plot(bins, fit_line, color='tab:orange')
        plt.xlim([-50, 50])
        plt.xlabel('Twist change (Degree)')
        plt.ylabel('Probability density')
        plt.tight_layout()
        plt.savefig(fname_hist)
        plt.close()

    # Plot variance against l
    ind_l = range(delta_twist_tl.shape[-1])
    var_l = []
    for ind in ind_l:
        var = np.sum([delta_twist ** 2 for delta_twist in delta_twist_tl[:, ind]]) / len(delta_twist_tl[:, ind])
        var_l.append(var)
    var_l = np.array(var_l)
    ind_l = np.array(ind_l)

    folder_fig = 'output/fig/supp1/param_scan'
    sh.mkdir('-p', folder_fig)
    fname_fig = op.join(folder_fig, 'var_vs_n.svg')

    plt.figure(figsize=(4.5, 3))
    plt.scatter(ind_l, var_l, color='grey')

    # Perform linear regression
    ind_mean, var_mean = np.mean(ind_l), np.mean(var_l)
    slope = np.sum((ind_l - ind_mean) * (var_l - var_mean)) / np.sum((ind_l - ind_mean) ** 2)
    intercept = var_mean - slope * ind_mean
    var_fit_l = np.array([slope*ind+intercept for ind in ind_l])
    ss_mean = np.sum((var_l - var_mean) ** 2)   # Sum of squares around the mean
    ss_fit = np.sum((var_l - var_fit_l) ** 2)   # Sum of squares around the fit
    r_squared = 1 - ss_fit / ss_mean
    print('%s: slope = %.4f, r_squared = %.4f' % (op.dirname(fname_coords).split('/')[-2], slope, r_squared))
    plt.plot(ind_l, var_fit_l, color='black', label='Slope = %.1f\n$R^2$ = %.4f' % (slope, r_squared))
    plt.legend(loc='upper left')
    plt.xticks(fontsize=14)
    plt.yticks([15, 30, 45, 60], fontsize=14)
    plt.xlabel('Subunit index', fontsize=16)
    plt.ylabel('Variance', fontsize=16)
    plt.tight_layout()
    plt.savefig(fname_fig)
    plt.close()


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

    folder_fig = 'output/fig/supp1/boxplot'
    sh.mkdir('-p', folder_fig)

    # Plot boxplot for thermal fluctuation
    fraction_thermal0_s = compute_fraction_for_all('output/thermal_fluct/fix0', 'props_3d_thermal.pkl')
    fraction_thermal1_s = compute_fraction_for_all('output/thermal_fluct/fix1', 'props_3d_thermal.pkl')
    fraction_thermal2_s = compute_fraction_for_all('output/thermal_fluct/fix2', 'props_3d_thermal.pkl')

    plt.figure(figsize=(3, 3))
    plt.boxplot([fraction_thermal0_s, fraction_thermal1_s, fraction_thermal2_s],
                labels=['0', '0', '0'])
    plt.xticks(fontsize=14)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    plt.ylim(-0.1, 1.2)
    plt.xlabel('Force (pN)', fontsize=16)
    plt.ylabel('Fraction', fontsize=16)
    plt.tight_layout()
    figname_thermal = op.join(folder_fig, 'thermal.svg')
    plt.savefig(figname_thermal)
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

    plt.figure(figsize=(6, 3))
    plt.boxplot([fraction_pullc_f0p5_s, fraction_pullc_f1p0_s, fraction_pullc_f1p5_s,
                 fraction_pullc_f2p0_s, fraction_pushc_f0p5_s, fraction_pushc_f1p0_s,
                 fraction_pushc_f1p5_s, fraction_pushc_f2p0_s],
                labels=['10', '20', '30', '40', '10', '20', '30', '40'])
    plt.xticks(fontsize=14)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    plt.ylim(-0.1, 1.2)
    plt.xlabel('Force (pN)', fontsize=16)
    plt.ylabel('Fraction', fontsize=16)
    plt.tight_layout()
    figname_constant = op.join(folder_fig, 'constant.svg')
    plt.savefig(figname_constant)
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

    plt.figure(figsize=(6, 3))
    plt.boxplot([fraction_pullr_f0p5_s, fraction_pullr_f1p0_s, fraction_pullr_f1p5_s,
                 fraction_pullr_f2p0_s, fraction_pushr_f0p5_s, fraction_pushr_f1p0_s,
                 fraction_pushr_f1p5_s, fraction_pushr_f2p0_s],
                labels=['10', '20', '30', '40', '10', '20', '30', '40'])
    plt.xticks(fontsize=14)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    plt.ylim(-0.1, 1.2)
    plt.xlabel('Force (pN)', fontsize=16)
    plt.ylabel('Fraction', fontsize=16)
    plt.tight_layout()
    figname_release = op.join(folder_fig, 'release.svg')
    plt.savefig(figname_release)
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

    # Plot amplitude
    plt.figure(figsize=(5, 3))
    vp = plt.violinplot(amp_ft, showextrema=False, showmedians=True)
    for body in vp['bodies']:
        body.set_facecolor('C0')
    vp['cmedians'].set_colors('black')
    plt.ylim(0, 100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # folder_fig = 'output/fig/supp1/violinplot/thermal_fluct'
    # sh.mkdir('-p', folder_fig)
    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[0, 0, 0])
    # plt.xlabel('Force (pN)', fontsize=16)

    folder_fig = 'output/fig/supp1/violinplot/push_constant'
    sh.mkdir('-p', folder_fig)
    plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[10, 20, 30, 40])
    plt.xlabel('Force (pN)', fontsize=16)

    # folder_fig = 'output/fig/supp1/violinplot/helical_rise'
    # sh.mkdir('-p', folder_fig)
    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[1.78, 2.28, 2.78, 3.28, 3.78])
    # plt.xlabel('Rise (nm)', fontsize=16)

    # folder_fig = 'output/fig/supp1/violinplot/helical_twist'
    # sh.mkdir('-p', folder_fig)
    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[136.67, 146.67, 156.67, 166.67, 176.67])
    # plt.xlabel('twist (\xb0)', fontsize=16)

    # folder_fig = 'output/fig/supp1/violinplot/flexural_rigidity'
    # sh.mkdir('-p', folder_fig)
    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[5.3, 9.2, 18.3, 22.6, 26.8])
    # plt.xlabel('Persistence length ($\mu$m)', fontsize=16)

    plt.ylabel('Amplitude (nm)', fontsize=16)
    plt.tight_layout()
    fig_name = op.join(folder_fig, 'violin_3d_amp.svg')
    plt.savefig(fig_name)
    plt.close()

    # Plot pitch
    plt.figure(figsize=(5, 3))
    vp = plt.violinplot(pitch_fp, showextrema=False, showmedians=True)
    for body in vp['bodies']:
        body.set_facecolor('C1')
    vp['cmedians'].set_colors('black')
    plt.ylim(70, 500)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[0, 0, 0])
    # plt.xlabel('Force (pN)', fontsize=16)

    plt.xticks([i + 1 for i in range(len(pitch_fp))], labels=[10, 20, 30, 40])
    plt.xlabel('Force (pN)', fontsize=16)

    # plt.xticks([i + 1 for i in range(len(pitch_fp))], labels=[1.78, 2.28, 2.78, 3.28, 3.78])
    # plt.xlabel('Rise (nm)', fontsize=16)

    # plt.xticks([i + 1 for i in range(len(pitch_fp))], labels=[136.67, 146.67, 156.67, 166.67, 176.67])
    # plt.xlabel('twist (\xb0)', fontsize=16)

    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[5.3, 9.2, 18.3, 22.6, 26.8])
    # plt.xlabel('Persistence length ($\mu$m)', fontsize=16)

    plt.ylabel('Pitch (nm)', fontsize=16)
    plt.tight_layout()
    fig_name = op.join(folder_fig, 'violin_3d_pitch.svg')
    plt.savefig(fig_name)
    plt.close()

    # Plot eccentricity
    plt.figure(figsize=(5, 3))
    vp = plt.violinplot(ecc_fe, showextrema=False, showmedians=True)
    for body in vp['bodies']:
        body.set_facecolor('C2')
    vp['cmedians'].set_colors('black')
    plt.ylim(0.8, 1.0)
    plt.xticks(fontsize=14)
    plt.yticks([0.80, 0.85, 0.90, 0.95, 1.00], fontsize=14)

    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[0, 0, 0])
    # plt.xlabel('Force (pN)', fontsize=16)

    plt.xticks([i + 1 for i in range(len(ecc_fe))], labels=[10, 20, 30, 40])
    plt.xlabel('Force (pN)', fontsize=16)

    # plt.xticks([i + 1 for i in range(len(ecc_fe))], labels=[1.78, 2.28, 2.78, 3.28, 3.78])
    # plt.xlabel('Rise (nm)', fontsize=16)

    # plt.xticks([i + 1 for i in range(len(ecc_fe))], labels=[136.67, 146.67, 156.67, 166.67, 176.67])
    # plt.xlabel('twist (\xb0)', fontsize=16)

    # plt.xticks([i + 1 for i in range(len(amp_ft))], labels=[5.3, 9.2, 18.3, 22.6, 26.8])
    # plt.xlabel('Persistence length ($\mu$m)', fontsize=16)

    plt.ylabel('Eccentricity', fontsize=16)
    plt.tight_layout()
    fig_name = op.join(folder_fig, 'violin_3d_ecc.svg')
    plt.savefig(fig_name)
    plt.close()


def plot_proj(fname_coords, condition, num_frames=5):
    assert condition in ['thermal', 'push_constant', 'pull_constant', 'push_release', 'pull_release']

    folder_output = 'output/fig/fig2/pca/' + condition
    # folder_output = 'output/fig/supp2/pca/' + condition
    sh.mkdir('-p', folder_output)

    with open(fname_coords, 'rb') as f:
        pkl = pickle.load(f)
    pos_tic = pkl['pos_tic']

    if condition == 'thermal':
        t_start, t_end = 0, 4500
    elif condition == 'push_constant' or condition == 'pull_constant':
        t_start = 0
        t_end = pkl['force_duration']
    elif condition == 'push_release' or condition == 'pull_release':
        t_start = pkl['force_duration']
        t_end = t_start + 4500

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
    plt.xticks([-50, 0, 50], fontsize=14)
    plt.yticks([-600, -400, -200, 0, 200, 400, 600], fontsize=14)
    plt.xlabel('PC2 (nm)', fontsize=16)
    plt.ylabel('PC1 (nm)', fontsize=16)
    plt.tight_layout()
    fig12_name = op.join(folder_output, 'pc12.svg')
    plt.savefig(fig12_name)
    plt.close()

    # Plot PC1 vs PC3
    plt.figure(figsize=(3, 12))
    for proj1_i, proj3_i, color in zip(proj1_ni, proj3_ni, color_n):
        plt.scatter(proj3_i, proj1_i, color=color, s=5)
    plt.xlim(-65, 65)
    plt.ylim(-600, 600)
    plt.xticks([-50, 0, 50], fontsize=14)
    plt.yticks([-600, -400, -200, 0, 200, 400, 600], fontsize=14)
    plt.xlabel('PC3 (nm)', fontsize=16)
    plt.ylabel('PC1 (nm)', fontsize=16)
    plt.tight_layout()
    fig13_name = op.join(folder_output, 'pc13.svg')
    plt.savefig(fig13_name)
    plt.close()

    # Plot PC2 vs PC3
    plt.figure(figsize=(3, 3))
    for proj2_i, proj3_i, color in zip(proj2_ni, proj3_ni, color_n):
        plt.scatter(proj3_i, proj2_i, color=color, s=5)
    plt.xlim(-65, 65)
    plt.ylim(-65, 65)
    plt.xticks([-50, 0, 50], fontsize=14)
    plt.yticks([-50, 0, 50], fontsize=14)
    plt.xlabel('PC3 (nm)', fontsize=16)
    plt.ylabel('PC2 (nm)', fontsize=16)
    plt.tight_layout()
    fig23_name = op.join(folder_output, 'pc23.svg')
    plt.savefig(fig23_name)
    plt.close()


def ecc_vs_time(dname, fname_props, amp_low=8, pitch_low=75):
    folder_output = 'output/fig/supp2/ecc_vs_time'
    sh.mkdir('-p', folder_output)

    plt.figure(figsize=(3, 3))
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

        plt.plot(range(len(ecc_e)), ecc_e, linewidth=1)

    plt.xlabel('Frame')
    plt.ylabel('Eccentricity')
    plt.tight_layout()
    fig_name = op.join(folder_output, 'ecc_vs_time.svg')
    plt.savefig(fig_name)
    plt.close()


if __name__ == '__main__':
    ecc_vs_time('output/helical_params_short/twist/twist_166.67', 'props_3d.pkl')
