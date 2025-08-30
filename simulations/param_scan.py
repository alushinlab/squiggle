import argh
import numpy as np
import math
import scipy.optimize as so
import scipy.stats as sst
import scipy.signal as ssi
import matplotlib.pyplot as plt
import os.path as op
import sh
import pickle
from glob import glob
from threading import Thread
import subprocess
import sys

import espressomd
from espressomd import thermostat
from espressomd import integrate
from espressomd.interactions import HarmonicBond, AngleHarmonic, Dihedral
from espressomd import observables
from espressomd import visualization
import util


@argh.arg('param_type', type=str)
@argh.arg('k_long', type=int)
@argh.arg('tensile_factor', type=int)
@argh.arg('force', type=int)
@argh.arg('k_bending', type=int)
@argh.arg('bending_factor', type=int)
@argh.arg('k_twisting', type=int)
@argh.arg('twisting_factor', type=int)
@argh.arg('seed', type=int)
def run_params(param_type, k_long, tensile_factor, force, k_bending,
               bending_factor, k_twisting, twisting_factor, seed,
               save_img=False):
    '''
    Set up canonical actin filament, simulation system, and energy functions.
    '''
    assert param_type in ['tensile', 'bending', 'twisting']

    if param_type == 'tensile':
        required_features = ["EXTERNAL_FORCES"]
        espressomd.assert_features(required_features)

        box_l = [125, 125, 250]
        part_num = 39
        kT = 0.001
        gamma = 10
        time_step = 0.001
        skin = 0.01
        camera_position = [box_l[0] / 2, 350, box_l[2] / 2]

    elif param_type == 'bending' or 'twisting':
        box_l = [200, 200, 400]
        part_num = 100
        kT = 1.0
        gamma = 123
        time_step = 0.01
        skin = 0.01
        camera_position = [box_l[0] / 2, 600, box_l[2] / 2]

    r = 1.6  # Distance from particle center to filament axis, length unit nm
    ind_i = np.array(list(range(part_num)))
    z0_i = 2.78 * ind_i  # Helical rise
    theta_i = - 166.67 / 180 * np.pi * ind_i  # Helical twist
    x0_i = r * np.cos(theta_i)
    y0_i = r * np.sin(theta_i)

    # Move the actin filament to the center of the box
    z_i = z0_i + box_l[2] / 2 - 2.78 * part_num / 2
    x_i = x0_i + box_l[0] / 2
    y_i = y0_i + box_l[1] / 2
    pos_i = np.stack((x_i, y_i, z_i), axis=1)

    # Calculate the diagonal and longitudinal bond length
    r_diag = np.linalg.norm(pos_i[1] - pos_i[0], axis=-1)
    r_long = np.linalg.norm(pos_i[2] - pos_i[0], axis=-1)

    # Set up system
    system = espressomd.System(box_l=box_l)
    system.time_step = time_step
    system.cell_system.skin = skin
    system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=seed)

    # Set up particles
    # Use type to specify particles on each strand
    system.part.add(id=ind_i[::2], pos=pos_i[::2], type=np.zeros_like(ind_i[::2]))
    system.part.add(id=ind_i[1::2], pos=pos_i[1::2], type=np.ones_like(ind_i[1::2]))

    # Set harmonic bond potentials
    k_diag = k_long * tensile_factor
    bond_diag = HarmonicBond(k=k_diag, r_0=r_diag)
    bond_long = HarmonicBond(k=k_long, r_0=r_long)
    system.bonded_inter[0] = bond_diag
    system.bonded_inter[1] = bond_long

    # Add harmonic bond potential
    for i in range(part_num - 1):
        system.part.by_id(i).add_bond((bond_diag, system.part.by_id(i + 1)))
    for i in range(part_num - 2):
        system.part.by_id(i).add_bond((bond_long, system.part.by_id(i + 2)))

    # Set harmonic angle potentials
    # Treat particle2 as the center particle
    angle_021 = AngleHarmonic(bend=3*bending_factor*k_bending, phi0=util.bond_angle(pos_i[0], pos_i[2], pos_i[1]))
    angle_023 = AngleHarmonic(bend=2*bending_factor*k_bending, phi0=util.bond_angle(pos_i[0], pos_i[2], pos_i[3]))
    angle_024 = AngleHarmonic(bend=k_bending, phi0=util.bond_angle(pos_i[0], pos_i[2], pos_i[4]))
    angle_123 = AngleHarmonic(bend=4*bending_factor*k_bending, phi0=util.bond_angle(pos_i[1], pos_i[2], pos_i[3]))
    angle_124 = AngleHarmonic(bend=2*bending_factor*k_bending, phi0=util.bond_angle(pos_i[1], pos_i[2], pos_i[4]))
    angle_324 = AngleHarmonic(bend=3*bending_factor*k_bending, phi0=util.bond_angle(pos_i[3], pos_i[2], pos_i[4]))

    system.bonded_inter[2] = angle_021
    system.bonded_inter[3] = angle_023
    system.bonded_inter[4] = angle_024
    system.bonded_inter[5] = angle_123
    system.bonded_inter[6] = angle_124
    system.bonded_inter[7] = angle_324

    # Add harmonic angle potential
    # part[i] should always be the vertex of the angle
    # i.e. the center particle of the triplet
    for i in range(2, part_num - 2):
        system.part.by_id(i).add_bond((angle_021, system.part.by_id(i - 2), system.part.by_id(i - 1)))
        system.part.by_id(i).add_bond((angle_023, system.part.by_id(i - 2), system.part.by_id(i + 1)))
        system.part.by_id(i).add_bond((angle_024, system.part.by_id(i - 2), system.part.by_id(i + 2)))
        system.part.by_id(i).add_bond((angle_123, system.part.by_id(i - 1), system.part.by_id(i + 1)))
        system.part.by_id(i).add_bond((angle_124, system.part.by_id(i - 1), system.part.by_id(i + 2)))
        system.part.by_id(i).add_bond((angle_324, system.part.by_id(i + 1), system.part.by_id(i + 2)))

    # Set dihedral potentials
    # Dihedral angle with the connection between particles 1&2 as the edge
    dihedral_edge12 = Dihedral(
        bend=twisting_factor*k_twisting, mult=1, phase=util.dihedral_angle(pos_i[0], pos_i[1], pos_i[2], pos_i[3]))

    # Dihedral angle with the connection between particles 0&3 as the edge
    # This dihedral is the main regulator of the twisting/untwisting of long-pitched strands
    dihedral_edge03 = Dihedral(
        bend=k_twisting, mult=1, phase=util.dihedral_angle(pos_i[2], pos_i[0], pos_i[3], pos_i[1]))

    system.bonded_inter[8] = dihedral_edge12
    system.bonded_inter[9] = dihedral_edge03

    # Add dihedral potentials
    for i in range(part_num - 3):
        system.part.by_id(i + 1).add_bond((
            dihedral_edge12,
            system.part.by_id(i),
            system.part.by_id(i + 2),
            system.part.by_id(i + 3),
        ))

        system.part.by_id(i).add_bond((
            dihedral_edge03,
            system.part.by_id(i + 2),
            system.part.by_id(i + 3),
            system.part.by_id(i + 1),
        ))

    if save_img:
        visualizer = visualization.openGLLive(system,
                                              force_arrows=False,
                                              background_color=[1, 1, 1],   # white background
                                              draw_bonds=True,
                                              bond_type_radius=[0.3],
                                              ext_force_arrows=True,
                                              ext_force_arrows_type_scale=[1.0],
                                              ext_force_arrows_type_radii=[1.0],
                                              quality_particles=500,
                                              particle_sizes=[3.0],
                                              particle_coloring='type',
                                              particle_type_colors=[[0.39, 0.58, 0.93], [0.68, 0.85, 0.90], [0.77, 0.47, 0.24], [0.80, 0.33, 0.51], [0.44, 0.65, 0.35], [0.54, 0.46, 0.79]],
                                              particle_type_materials=['bright', 'bright'],
                                              camera_position=camera_position,   # side view
                                              camera_right=[1, 0, 0],
                                              window_size=[512, 1024],
                                              draw_box=False,
                                              draw_axis=False,
                                              )

    if param_type == 'tensile':
        # Fix the initial two subunits, one on each strand
        system.part.by_ids(range(2)).fix = [True, True, True]

        folder_param = 'output/param_scan/tensile_8/kl%03d_tf%02d' % (k_long, tensile_factor)
        sh.mkdir('-p', folder_param)
        force_unit = 4
        folder_force = op.join(folder_param, '%03dpN' % (2 * force * force_unit))
        folder_seed = op.join(folder_force, 'seed%d' % seed)
        sh.mkdir('-p', folder_seed)

        # Set up integration
        frame_step = 5000
        frame_num = 500

        pos_tic = []
        for frame in range(frame_num):
            if save_img:
                folder_img = op.join(folder_seed, 'img')
                sh.mkdir('-p', folder_img)
                fname_img = op.join(folder_img, '%03d.png' % frame)
                visualizer.screenshot(fname_img)

            print('run %d at time=%f' % (frame, system.time))
            # Exert force on terminal two subunits, one on each strand
            system.part.by_ids(list(range(part_num))[-2:]).ext_force = [0, 0, force]
            system.integrator.run(frame_step)
            pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
            pos_tic.append(pos_ic)
        pos_tic = np.array(pos_tic)
        force_total = force * 2

    elif param_type == 'bending' or 'twisting':
        print('Equilibration')
        system.integrator.run(500000)
        print('Equilibration finished')

        # Set up integration
        frame_step = 2000
        frame_num = 1000

        if param_type == 'bending':
            folder_param = 'output/param_scan/bending_8/kb%03d_bf%d' % (k_bending, bending_factor)
            sh.mkdir('-p', folder_param)
        elif param_type == 'twisting':
            folder_param = 'output/param_scan/twisting_8/kt%03d_tf%02d' % (k_twisting, twisting_factor)
            sh.mkdir('-p', folder_param)
        folder_seed = op.join(folder_param, 'seed%d' % seed)
        sh.mkdir('-p', folder_seed)

        pos_tic = []
        for frame in range(frame_num):
            if save_img:
                folder_img = op.join(folder_seed, 'img')
                sh.mkdir('-p', folder_img)
                fname_img = op.join(folder_img, '%03d.png' % frame)
                visualizer.screenshot(fname_img)

            print('run %d at time=%f' % (frame, system.time))
            system.integrator.run(frame_step)
            pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
            pos_tic.append(pos_ic)
        pos_tic = np.array(pos_tic)
        force_total = 0

    print('Saving coords to pkl')
    output = dict(param_type=param_type, k_long=k_long,
                  tensile_factor=tensile_factor, k_bending=k_bending,
                  bending_factor=bending_factor, k_twisting=k_twisting,
                  twisting_factor=twisting_factor, seed=seed, pos_tic=pos_tic,
                  force_total=force_total)
    fname_coords = op.join(folder_seed, 'coords.pkl')
    with open(fname_coords, 'wb') as f:
        pickle.dump(output, f, protocol=2)


def extension_vs_time(fname_coords, part_num=39, box_l=[125, 125, 250]):
    '''Plot extension in Z against time'''
    ind_i = np.array(list(range(part_num)))
    z0_i = 2.78 * ind_i
    z_i = z0_i + box_l[2] / 2 - 2.78 * part_num / 2

    print('Plotting extension against time for %s' % op.dirname(fname_coords))
    with open(fname_coords, 'rb') as f:
        pkl = pickle.load(f)
    k_long = pkl['k_long']
    tensile_factor = pkl['tensile_factor']
    force_total = pkl['force_total']
    pos_tic = pkl['pos_tic']

    extension_t = []
    for (t, pos_ic) in enumerate(pos_tic):
        extension = pos_ic[-1][-1] - z_i[-1]
        extension_t.append(extension)

    fname_fig = op.join(op.dirname(fname_coords), 'extension_vs_time.png')
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(range(len(extension_t)), extension_t, label='kd = %d, kl = %d, F = %d' % (k_long * tensile_factor, k_long, force_total))
    ax.legend(loc='lower right', fontsize='small')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Extension (nm)')
    plt.tight_layout()
    plt.savefig(fname_fig)
    plt.close()


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
    fname_fig = op.join(folder_param, 'force_vs_extension.png')
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.scatter(extension_f, force_f, c='tab:blue', label='%s' % param)
    ax.plot(extension_f, force_fit_f, c='tab:orange', label='k = %.1f' % slope)
    ax.legend(loc='upper left', fontsize='small')
    ax.set_xlabel('Extension (nm)')
    ax.set_ylabel('Force (pN)')
    plt.tight_layout()
    plt.savefig(fname_fig)
    plt.close()

    return slope


def force_vs_extension_for_all(path):
    for folder_param in sorted(glob(path)):
        force_vs_extension(folder_param)


def scan_tensile_params():
    # # Parameter sets for tensile scan round 1
    # param_type = 'tensile'
    # k_long_set = [25, 50, 75, 100]
    # tensile_factor_set = [5, 10, 15, 20]
    # force_set = [6, 12, 18, 24]
    # seed = 2
    # k_bending, bending_factor, k_twisting, twisting_factor = 0, 0, 0, 0

    # # Parameter sets for tensile scan round 2
    # param_type = 'tensile'
    # k_long_set = [10, 20, 30]
    # tensile_factor_set = [5, 10, 15, 20]
    # force_set = [6, 12, 18, 24]
    # k_bending = 400
    # bending_factor = 4
    # k_twisting = 50
    # twisting_factor = 10
    # seed = 2

    # # Parameter sets for tensile scan round 3
    # param_type = 'tensile'
    # k_long_set = [10, 15, 20, 25, 30]
    # tensile_factor_set = [5, 10, 15, 20]
    # force_set = [6, 12, 18, 24]
    # k_bending = 400
    # bending_factor = 1
    # k_twisting = 75
    # twisting_factor = 10
    # seed = 2

    # # Parameter sets for tensile scan round 4
    # param_type = 'tensile'
    # k_long_set = [10, 15, 20, 25]
    # tensile_factor_set = [5, 10, 15, 20]
    # force_set = [6, 12, 18, 24]
    # k_bending = 200
    # bending_factor = 1
    # k_twisting = 50
    # twisting_factor = 20
    # seed = 2

    # # Parameter sets for tensile scan round 5
    # param_type = 'tensile'
    # k_long_set = [10, 15, 20, 25]
    # tensile_factor_set = [5, 10, 15, 20]
    # force_set = [6, 12, 18, 24]
    # k_bending = 150
    # bending_factor = 1
    # k_twisting = 50
    # twisting_factor = 20
    # seed = 2

    # # Parameter sets for tensile scan round 6
    # param_type = 'tensile'
    # k_long_set = [10, 15, 20]
    # tensile_factor_set = [5, 10, 15, 20]
    # force_set = [6, 12, 18, 24]
    # k_bending = 175
    # bending_factor = 1
    # k_twisting = 50
    # twisting_factor = 20
    # seed = 2

    # # Parameter sets for tensile scan round 7
    # param_type = 'tensile'
    # k_long_set = [10, 15, 20]
    # tensile_factor_set = [5, 10, 15, 20]
    # force_set = [6, 12, 18, 24]
    # k_bending = 155
    # bending_factor = 1
    # k_twisting = 55
    # twisting_factor = 20
    # seed = 2

    # Parameter sets for tensile scan round 8
    param_type = 'tensile'
    k_long_set = [10, 15, 20]
    tensile_factor_set = [5, 10, 15, 20]
    force_set = [6, 12, 18, 24]
    k_bending = 145
    bending_factor = 1
    k_twisting = 55
    twisting_factor = 20
    seed = 2

    stop = False
    for k_long in k_long_set:
        for ind, tensile_factor in enumerate(tensile_factor_set):
            folder_param = 'output/param_scan/tensile_8/kl%03d_tf%02d' % (k_long, tensile_factor)

            for force in force_set:
                print('Scanning kl = %d, kd = %d, at F = %d using seed %d' % (k_long, k_long * tensile_factor, force, seed))
                subprocess.run([
                    sys.executable,
                    __file__,
                    'run-params', # _ is replaced with - by argh
                    str(param_type),
                    str(k_long),
                    str(tensile_factor),
                    str(force),
                    str(k_bending),
                    str(bending_factor),
                    str(k_twisting),
                    str(twisting_factor),
                    str(seed)
                ], check=True)

                force_unit = 4
                folder_force = op.join(folder_param, '%03dpN' % (2 * force * force_unit))
                folder_seed = op.join(folder_force, 'seed%d' % seed)
                fname_coords = op.join(folder_seed, 'coords.pkl')
                extension_vs_time(fname_coords)

            slope = force_vs_extension(folder_param)

            if slope > 31 and ind == 0:
                stop = True
                print('Stiffness %.1f > 31 pN/nm. Scan finished')
                break
            elif slope > 31 and ind > 0:
                print('Stiffness %.1f > 31 pN/nm, jump to the next k_long' % slope)
                break
        if stop:
            break


def trace_center(fname_coords, save_2d_fig=False, save_3d_fig=False):
    '''
    Trace the linear polymer through the center axis.
    Anchor points are defined by the average position between part i - 1 and part i + 1.
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

    plt.figure(figsize=(4, 3))
    plt.plot(contour_l, corr_l)
    plt.plot(contour_l, fit_l, label='R$^2$ = %.4f \nLp = %.1f nm' % (r_squared, p))
    plt.xlabel('Subunit index')
    plt.ylabel(r'ln<cos$\theta$>')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    fname_fig = op.join(op.dirname(fname_coords), 'pl.png')
    plt.savefig(fname_fig)
    plt.close()

    return p


def persistence_length_for_all(path):
    for fname_coords in sorted(glob(path)):
        print(fname_coords)
        persistence_length(fname_coords)


def scan_bending_params():
    # # Parameter sets for bending scan round 1
    # param_type = 'bending'
    # k_long = 75
    # tensile_factor = 5
    # force = 0
    # k_bending_set = [200, 400, 600, 800]
    # bending_factor_set = [1, 2, 4]
    # seed = 2
    # k_twisting, twisting_factor = 0, 0

    # # Parameter sets for bending scan round 2
    # param_type = 'bending'
    # k_long = 20
    # tensile_factor = 5
    # force = 0
    # k_bending_set = [200, 400, 600, 800]
    # bending_factor_set = [1, 2, 4]
    # k_twisting = 50
    # twisting_factor = 10
    # seed = 2

    # # Parameter sets for bending scan round 3
    # param_type = 'bending'
    # k_long = 15
    # tensile_factor = 10
    # force = 0
    # k_bending_set = [200, 300, 400, 500, 600]
    # bending_factor_set = [1, 2]
    # k_twisting = 75
    # twisting_factor = 10
    # seed = 2

    # # Parameter sets for bending scan round 4
    # param_type = 'bending'
    # k_long = 15
    # tensile_factor = 15
    # force = 0
    # k_bending_set = [100, 150, 200, 250, 300]
    # bending_factor_set = [1]
    # k_twisting = 50
    # twisting_factor = 20
    # seed = 2

    # # Parameter sets for bending scan round 5
    # param_type = 'bending'
    # k_long = 15
    # tensile_factor = 20
    # force = 0
    # k_bending_set = [100, 125, 150, 175, 200]
    # bending_factor_set = [1]
    # k_twisting = 50
    # twisting_factor = 20
    # seed = 2

    # # Parameter sets for bending scan round 6
    # param_type = 'bending'
    # k_long = 15
    # tensile_factor = 20
    # force = 0
    # k_bending_set = [155, 165, 175, 185, 195]
    # bending_factor_set = [1]
    # k_twisting = 50
    # twisting_factor = 20
    # seed = 2

    # # Parameter sets for bending scan round 7
    # param_type = 'bending'
    # k_long = 15
    # tensile_factor = 20
    # force = 0
    # k_bending_set = [145, 150, 155, 160, 165]
    # bending_factor_set = [1]
    # k_twisting = 55
    # twisting_factor = 20
    # seed = 2

    # Parameter sets for bending scan round 8
    param_type = 'bending'
    k_long = 20
    tensile_factor = 15
    force = 0
    k_bending_set = [135, 140, 145, 150, 155]
    bending_factor_set = [1]
    k_twisting = 55
    twisting_factor = 20
    seed = 2

    stop = False
    for k_bending in k_bending_set:
        for ind, bending_factor in enumerate(bending_factor_set):
            print('Scanning k_b = %d, bending_factor = %d, using seed %d' % (k_bending, bending_factor, seed))
            subprocess.run([
                sys.executable,
                __file__,
                'run-params', # _ is replaced with - by argh
                str(param_type),
                str(k_long),
                str(tensile_factor),
                str(force),
                str(k_bending),
                str(bending_factor),
                str(k_twisting),
                str(twisting_factor),
                str(seed)
            ], check=True)

            folder_param = 'output/param_scan/bending_8/kb%03d_bf%d' % (k_bending, bending_factor)
            folder_seed = op.join(folder_param, 'seed%d' % seed)
            fname_coords = op.join(folder_seed, 'coords.pkl')
            p = persistence_length(fname_coords)
            if p > 9500 and ind == 0:
                stop = True
                print('Persistence length %.1f > 9.5 um. Scan finished' % p)
                break
            elif p > 9500 and ind > 0:
                print('Persistence length %.1f > 9.5 um, jumping to the next k_bending' % p)
                break
        if stop:
            break


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

    fname_var = op.join(op.dirname(fname_coords), 'var_vs_n.png')
    plt.figure(figsize=(4, 3))
    plt.scatter(ind_l, var_l, color='tab:blue')

    # Perform linear regression
    ind_mean, var_mean = np.mean(ind_l), np.mean(var_l)
    slope = np.sum((ind_l - ind_mean) * (var_l - var_mean)) / np.sum((ind_l - ind_mean) ** 2)
    intercept = var_mean - slope * ind_mean
    var_fit_l = np.array([slope*ind+intercept for ind in ind_l])
    ss_mean = np.sum((var_l - var_mean) ** 2)   # Sum of squares around the mean
    ss_fit = np.sum((var_l - var_fit_l) ** 2)   # Sum of squares around the fit
    r_squared = 1 - ss_fit / ss_mean
    print('%s: slope = %.4f, r_squared = %.4f' % (op.dirname(fname_coords).split('/')[-2], slope, r_squared))
    plt.plot(ind_l, var_fit_l, c='tab:orange', label='Slope = %.4f, $R^2$ = %.4f' % (slope, r_squared))
    plt.legend(loc='lower right', fontsize='small')
    plt.xlabel('n')
    plt.ylabel('Variance')
    plt.tight_layout()
    plt.savefig(fname_var)
    plt.close()

    return slope


def twist_var_for_all(path):
    for fname_coords in sorted(glob(path)):
        twist_var(fname_coords)


def scan_twisting_params():
    # # Parameter sets for twisting scan round 1
    # param_type = 'twisting'
    # k_long = 75
    # tensile_factor = 5
    # force = 0
    # k_bending = 400
    # bending_factor = 4
    # k_twisting_set = [50, 100, 150, 200]
    # twisting_factor_set = [5, 10, 15, 20]
    # seed = 2

    # # Parameter sets for twisting scan round 2
    # param_type = 'twisting'
    # k_long = 20
    # tensile_factor = 5
    # force = 0
    # k_bending = 400
    # bending_factor = 1
    # k_twisting_set = [25, 50, 75, 100]
    # twisting_factor_set = [5, 10, 15, 20]
    # seed = 2

    # # Parameter sets for twisting scan round 3
    # param_type = 'twisting'
    # k_long = 15
    # tensile_factor = 10
    # force = 0
    # k_bending = 200
    # bending_factor = 1
    # k_twisting_set = [25, 50, 75, 100]
    # twisting_factor_set = [10, 15, 20]
    # seed = 2

    # # Parameter sets for twisting scan round 4
    # param_type = 'twisting'
    # k_long = 15
    # tensile_factor = 15
    # force = 0
    # k_bending = 150
    # bending_factor = 1
    # k_twisting_set = [25, 50, 75, 100]
    # twisting_factor_set = [10, 15, 20]
    # seed = 2

    # # Parameter sets for twisting scan round 5
    # param_type = 'twisting'
    # k_long = 15
    # tensile_factor = 20
    # force = 0
    # k_bending = 175
    # bending_factor = 1
    # k_twisting_set = [40, 50, 60]
    # twisting_factor_set = [10, 15, 20]
    # seed = 2

    # # Parameter sets for twisting scan round 6
    # param_type = 'twisting'
    # k_long = 15
    # tensile_factor = 20
    # force = 0
    # k_bending = 155
    # bending_factor = 1
    # k_twisting_set = [45, 50, 55]
    # twisting_factor_set = [10, 15, 20]
    # seed = 2

    # # Parameter sets for twisting scan round 7
    # param_type = 'twisting'
    # k_long = 15
    # tensile_factor = 20
    # force = 0
    # k_bending = 145
    # bending_factor = 1
    # k_twisting_set = [50, 55, 60]
    # twisting_factor_set = [10, 15, 20]
    # seed = 2

    # Parameter sets for twisting scan round 8
    param_type = 'twisting'
    k_long = 20
    tensile_factor = 15
    force = 0
    k_bending = 145
    bending_factor = 1
    k_twisting_set = [50, 55, 60]
    twisting_factor_set = [10, 15, 20]
    seed = 2

    stop = False
    for k_twisting in k_twisting_set:
        for ind, twisting_factor in enumerate(twisting_factor_set):
            print('Scanning k_t = %d, twisting_factor = %d, using seed %d' % (k_twisting, twisting_factor, seed))
            subprocess.run([
                sys.executable,
                __file__,
                'run-params', # _ is replaced with - by argh
                str(param_type),
                str(k_long),
                str(tensile_factor),
                str(force),
                str(k_bending),
                str(bending_factor),
                str(k_twisting),
                str(twisting_factor),
                str(seed)
            ], check=True)

            folder_param = 'output/param_scan/twisting_8/kt%03d_tf%02d' % (k_twisting, twisting_factor)
            folder_seed = op.join(folder_param, 'seed%d' % seed)
            fname_coords = op.join(folder_seed, 'coords.pkl')
            slope = twist_var(fname_coords)

            if slope < 1.1667 and ind == 0:
                stop = True
                print('Slope %.4f < 1.1667. Scan finished' % slope)
                break
            elif slope < 1.1667 and ind > 0:
                print('Slope %.4f < 1.1667, jump to the next k_twisting' % slope)
                break
        if stop:
            break


if __name__ == '__main__':
    # argh.dispatch_commands([scan_tensile_params, run_params])
    # force_vs_extension_for_all('output/param_scan/tensile_8/*')
    # argh.dispatch_commands([scan_bending_params, run_params])
    # persistence_length_for_all('output/param_scan/bending_8/*/*/coords.pkl')
    # argh.dispatch_commands([scan_twisting_params, run_params])
    twist_var_for_all('output/param_scan/twisting_8/*/*/coords.pkl')
