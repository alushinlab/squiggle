import argh
import numpy as np
import os.path as op
import sh
import pickle
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
@argh.arg('rise', type=float)
@argh.arg('twist', type=float)
@argh.arg('seed', type=int)
@argh.arg('save_img', type=int)
def run_helical_params(param_type, rise, twist, seed, save_img):
    '''
    Set up actin filament, simulation system, and energy functions.
    '''
    assert param_type in ['rise', 'twist']
    if param_type == 'rise':
        box_l = [1000, 1000, 1000]
        part_num = 200
        camera_position = [box_l[0] / 2, 1800, box_l[2] / 2]
    elif param_type == 'twist':
        box_l = [750, 750, 750]
        part_num = 200
        camera_position = [box_l[0] / 2, 1200, box_l[2] / 2]

    r = 1.6  # Distance from particle center to filament axis, length unit nm
    ind_i = np.array(list(range(part_num)))
    z0_i = rise * ind_i  # Helical rise
    theta_i = - twist / 180 * np.pi * ind_i  # Helical twist
    x0_i = r * np.cos(theta_i)
    y0_i = r * np.sin(theta_i)

    # Move the actin filament to the center of the box
    z_i = z0_i + box_l[2] / 2 - rise * part_num / 2
    x_i = x0_i + box_l[0] / 2
    y_i = y0_i + box_l[1] / 2
    pos_i = np.stack((x_i, y_i, z_i), axis=1)

    # Calculate the diagonal and longitudinal bond length
    r_diag = np.linalg.norm(pos_i[1] - pos_i[0], axis=-1)
    r_long = np.linalg.norm(pos_i[2] - pos_i[0], axis=-1)

    # Set up system
    system = espressomd.System(box_l=box_l)
    system.time_step = 0.01
    system.cell_system.skin = 0.01
    system.thermostat.set_langevin(kT=1.0, gamma=123, seed=seed)

    # Set up particles
    # Use type to specify particles on each strand
    system.part.add(id=ind_i[::2], pos=pos_i[::2], type=np.zeros_like(ind_i[::2]))
    system.part.add(id=ind_i[1::2], pos=pos_i[1::2], type=np.ones_like(ind_i[1::2]))

    # Set harmonic bond potentials
    k_long = 20
    tensile_factor = 15
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
    k_bending = 145
    bending_factor = 1

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
    k_twisting = 55
    twisting_factor = 20

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
                                              #background_color=[0, 0, 0],   # black background
                                              draw_bonds=False,
                                              bond_type_radius=[0.3],
                                              ext_force_arrows=True,
                                              ext_force_arrows_type_scale=[1.0],
                                              ext_force_arrows_type_radii=[1.0],
                                              quality_particles=500,
                                              particle_sizes=[3.0],
                                              particle_coloring='type',
                                              particle_type_colors=[[0.39, 0.58, 0.93], [0.68, 0.85, 0.90], [1, 1, 0], [1, 0.50, 0], [1, 0, 0], [0, 1, 0]],
                                              particle_type_materials=['bright', 'bright'],
                                              camera_position=camera_position,
                                              camera_right=[1, 0, 0],
                                              window_size=[1000, 1500],
                                              draw_box=False,
                                              draw_axis=False,
                                             )

    print('Equilibration')
    system.integrator.run(500000)
    print('Equilibration finished')

    if param_type == 'rise':
        folder_param = 'output/helical_params_short/rise/rise_%.2f' % rise
        sh.mkdir('-p', folder_param)
    elif param_type == 'twist':
        folder_param = 'output/helical_params_short/twist/twist_%.2f' % twist
        sh.mkdir('-p', folder_param)
    folder_seed = op.join(folder_param, 'seed%d' % seed)
    sh.mkdir('-p', folder_seed)

    required_features = ["EXTERNAL_FORCES"]
    espressomd.assert_features(required_features)

    # Fix the initial two subunits, one on each strand
    system.part.by_ids(range(2)).fix = [True, True, True]

    # Set up integration
    frame_step = 10000
    frame_num = 5000

    # Exert force on terminal five subunits
    print('Applying force')
    system.part.by_ids(list(range(part_num))[-5:]).fix = [True, True, False]
    system.part.by_ids(list(range(part_num))[-5:]).ext_force = [0, 0, -1.5]

    pos_tic = []
    for frame in range(frame_num):
        if save_img:
            folder_img = op.join(folder_seed, 'img')
            sh.mkdir('-p', folder_img)
            fname_img = op.join(folder_img, '%04d.png' % frame)
            visualizer.screenshot(fname_img)

        print('run %d at time=%f' % (frame, system.time))
        system.integrator.run(frame_step)
        pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
        pos_tic.append(pos_ic)
    pos_tic = np.array(pos_tic)

    print('Saving coords to pkl')
    output = dict(rise=rise, twist=twist, seed=seed, pos_tic=pos_tic)
    fname_coords = op.join(folder_seed, 'coords.pkl')
    with open(fname_coords, 'wb') as f:
        pickle.dump(output, f, protocol=2)


def scan_rise_params():
    param_type = 'rise'
    rise_set = [1.78, 2.28, 2.78, 3.28, 3.78]
    twist = 166.67
    seed_set = [0, 1, 2, 3, 4]
    save_img = 1

    for seed in seed_set:
        for rise in rise_set:
            print('Scanning rise = %.2f using seed %d' % (rise, seed))
            subprocess.run([
                sys.executable,
                __file__,
                'run-helical-params', # _ is replaced with - by argh
                str(param_type),
                str(rise),
                str(twist),
                str(seed),
                str(save_img)
            ], check=True)


def scan_twist_params():
    param_type = 'twist'
    rise = 2.78
    twist_set = [136.67, 146.67, 156.67, 166.67, 176.67]
    seed_set = [1, 2, 3, 4]
    save_img = 1

    for seed in seed_set:
        for twist in twist_set:
            print('Scanning twist = %.2f using seed %d' % (twist, seed))
            subprocess.run([
                sys.executable,
                __file__,
                'run-helical-params', # _ is replaced with - by argh
                str(param_type),
                str(rise),
                str(twist),
                str(seed),
                str(save_img)
            ], check=True)

if __name__ == '__main__':
    argh.dispatch_commands([scan_rise_params, run_helical_params])
    #argh.dispatch_commands([scan_twist_params, run_helical_params])
