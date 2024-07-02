import numpy as np
import math
import scipy.stats as ss
import matplotlib.pyplot as plt
import os.path as op
import sh
import pickle
from glob import glob
from threading import Thread
import espressomd
from espressomd import thermostat
from espressomd import integrate
from espressomd.interactions import HarmonicBond, AngleHarmonic, Dihedral
from espressomd import observables
from espressomd import visualization
import util


def tensile_param_scan(box_l=3*[200], n_part=39, kT=0, gamma=10, seed=42, img_interval=100):
    '''
    kT=2.44 kJ/mol at 20 celsius and 293.15 kelvin.
    gamma = 25151 pN*ps/nm is the estimated friction coefficient of G-actin in water at RT.
    Use kT=0, gamma=10 to scan spring constants for harmonic bond potentials in a stretched actin.
    Use kT=2.44, gamma=25151 to scan spring constants for harmonic angle potentials and dihedral potentials in an actin undergoing thermal fluctuation.
    '''

    required_features = ["EXTERNAL_FORCES", "MASS"]
    espressomd.assert_features(required_features)

    box_l = box_l  # nm
    n_part = n_part

    r = 1.5  # Radius of each particle/subunit in nm
    part_i = np.array(list(range(n_part)))
    z0_i = 2.78 * part_i  # Axial translocation of protomer is 2.78 nm
    theta_i = - 166.67 / 180 * np.pi * part_i  # Rotation angle around the helical axis is 166.67 degrees
    x0_i = r * np.cos(theta_i)
    y0_i = r * np.sin(theta_i)

    # Calculate the lateral and axial bond length
    r_lateral = np.sqrt((x0_i[1] - x0_i[0]) ** 2 + (y0_i[1] - y0_i[0]) ** 2 +
                        (z0_i[1] - z0_i[0]) ** 2)
    r_axial = np.sqrt((x0_i[2] - x0_i[0]) ** 2 + (y0_i[2] - y0_i[0]) ** 2 +
                      (z0_i[2] - z0_i[0]) ** 2)

    # Move the actin filament to the center of the box
    z_i = z0_i + box_l[2] / 2 - 2.78 * n_part / 2
    x_i = x0_i + box_l[0] / 2
    y_i = y0_i + box_l[1] / 2

    system = espressomd.System(box_l=box_l)
    system.time_step = 0.01  # time scale unit is ps
    system.cell_system.skin = 0.01  # nm
    system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=seed)

    # Set up particles
    pos_i = np.stack((x_i, y_i, z_i), axis=1)
    mass_i = 42051 * np.ones_like(part_i)
    system.part.add(id=part_i[::2], pos=pos_i[::2], mass=mass_i[::2], type=np.zeros_like(part_i[::2]))
    system.part.add(id=part_i[1::2], pos=pos_i[1::2], mass=mass_i[1::2], type=np.ones_like(part_i[1::2]))

    # Set up integration
    int_steps = 100
    int_iterations = 50000

    # Spring constants and force parameters to scan
    '''
    k_axial_set = [150, 200, 250, 275, 300]
    k_lateral_factor_set = [1/100, 1/50, 1/10, 1/5]
    force_set = range(15, 70, 15)
    '''
    k_axial_set = [265]
    k_lateral_factor_set = [1/5]
    force_set = range(15, 70, 15)

    for k_axial in k_axial_set:
        for k_lateral_factor in k_lateral_factor_set:
            k_lateral = k_axial * k_lateral_factor

            folder_output = 'output/tensile_param_scan_kb300_kt150/ka%.1f_kl%.1f' % (k_axial, k_lateral)
            sh.mkdir('-p', folder_output)

            # Set harmonic bond potentials
            bond_lateral = HarmonicBond(k=k_lateral, r_0=r_lateral)
            bond_axial = HarmonicBond(k=k_axial, r_0=r_axial)
            system.bonded_inter[0] = bond_lateral
            system.bonded_inter[1] = bond_axial

            # Add harmonic bond potential
            for i in range(n_part - 1):
                system.part.by_id(i).add_bond((bond_lateral, system.part.by_id(i + 1)))
            for i in range(n_part - 2):
                system.part.by_id(i).add_bond((bond_axial, system.part.by_id(i + 2)))

            k_bending = 300
            # Set harmonic angle potentials
            # Treat particle2 as the center particle
            angle_021 = AngleHarmonic(bend=4*k_bending, phi0=util.bond_angle(pos_i[0], pos_i[2], pos_i[1]))
            angle_023 = AngleHarmonic(bend=2*k_bending, phi0=util.bond_angle(pos_i[0], pos_i[2], pos_i[3]))
            angle_024 = AngleHarmonic(bend=k_bending, phi0=util.bond_angle(pos_i[0], pos_i[2], pos_i[4]))
            angle_123 = AngleHarmonic(bend=8*k_bending, phi0=util.bond_angle(pos_i[1], pos_i[2], pos_i[3]))
            angle_124 = AngleHarmonic(bend=2*k_bending, phi0=util.bond_angle(pos_i[1], pos_i[2], pos_i[4]))
            angle_324 = AngleHarmonic(bend=4*k_bending, phi0=util.bond_angle(pos_i[3], pos_i[2], pos_i[4]))

            system.bonded_inter[2] = angle_021
            system.bonded_inter[3] = angle_023
            system.bonded_inter[4] = angle_024
            system.bonded_inter[5] = angle_123
            system.bonded_inter[6] = angle_124
            system.bonded_inter[7] = angle_324

            # Add harmonic angle potential
            # part[i] should always be the vertex of the angle
            # i.e. the center particle of the triplet
            for i in range(2, n_part - 2):
                system.part.by_id(i).add_bond((angle_021, system.part.by_id(i - 2), system.part.by_id(i - 1)))
                system.part.by_id(i).add_bond((angle_023, system.part.by_id(i - 2), system.part.by_id(i + 1)))
                system.part.by_id(i).add_bond((angle_024, system.part.by_id(i - 2), system.part.by_id(i + 2)))
                system.part.by_id(i).add_bond((angle_123, system.part.by_id(i - 1), system.part.by_id(i + 1)))
                system.part.by_id(i).add_bond((angle_124, system.part.by_id(i - 1), system.part.by_id(i + 2)))
                system.part.by_id(i).add_bond((angle_324, system.part.by_id(i + 1), system.part.by_id(i + 2)))

            k_twisting = 150
            # Set dihedral potentials
            # Dihedral angle with the connection between particles 1&2 as the edge
            # This dihedral is the main regulator of the twisting/untwisting of F-actin
            dihedral_edge12 = Dihedral(
                bend=k_twisting, mult=1, phase=util.dihedral_angle(pos_i[0], pos_i[1], pos_i[2], pos_i[3]))

            # Dihedral angle with the connection between particles 0&3 as the edge
            dihedral_edge03 = Dihedral(
                bend=(5*k_twisting), mult=1, phase=util.dihedral_angle(pos_i[2], pos_i[0], pos_i[3], pos_i[1]))

            system.bonded_inter[8] = dihedral_edge12
            system.bonded_inter[9] = dihedral_edge03

            # Add dihedral potentials
            for i in range(n_part - 3):
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

            visualizer = visualization.openGLLive(system,
                                                  force_arrows=False,
                                                  background_color=[1, 1, 1],   # white background
                                                  bond_type_radius=[0.3],
                                                  ext_force_arrows=True,
                                                  ext_force_arrows_type_scale=[1.0],
                                                  ext_force_arrows_type_radii=[1.0],
                                                  particle_sizes=[3.0],
                                                  particle_type_colors=[[0.39, 0.58, 0.93], [0.68, 0.85, 0.90]],
                                                  particle_type_materials=['bright', 'bright'],
                                                  camera_position=[100, 400, 100],   # side view
                                                  camera_right=[1, 0, 0])

            # Fix the ends of both strands
            system.part.by_ids(range(2)).fix = [True, True, True]

            for force in force_set:
                folder_force = op.join(folder_output, '%03dpN' % (2 * force))
                sh.mkdir('-p', folder_force)
                z_force = force

                pos_tic = []
                for int_iteration in range(int_iterations):
                    print('run %d at time=%f' % (int_iteration, system.time))
                    system.part.by_ids(list(range(n_part))[-2:]).fix = [True, True, False]
                    system.part.by_ids(list(range(n_part))[-2:]).ext_force = [0, 0, z_force]
                    system.integrator.run(int_steps)

                    folder_img = op.join(folder_force, 'img')
                    sh.mkdir('-p', folder_img)
                    if (int_iteration % img_interval == 0):
                        fname_img = op.join(folder_img, '%05d.png' % int_iteration)
                        visualizer.screenshot(fname_img)

                    pos_ic = observables.ParticlePositions(ids=part_i)
                    pos_tic.append(pos_ic.calculate())

                pos_tic = np.array(pos_tic)
                output = dict(pos_tic=pos_tic)

                print('Saving coords to pkl')
                fname_coords = op.join(folder_force, 'coords.pkl')
                with open(fname_coords, 'wb') as f:
                    pickle.dump(output, f, protocol=2)

            # Delete all bonds to reset the system for next iteration
            for i in range(n_part):
                system.part.by_id(i).delete_all_bonds()


def extension_vs_time(n_part=39, box_l=3*[200]):
    '''Plot extension in Z against time'''
    part_i = np.array(list(range(n_part)))
    z0_i = 2.78 * part_i
    z_i = z0_i + box_l[2] / 2 - 2.78 * n_part / 2

    for fname_coords in sorted(glob('output/tensile_param_scan_kb300_kt150/*/*/coords.pkl')):
        print('Plotting extension against time for %s' % op.dirname(fname_coords))
        with open(fname_coords, 'rb') as f:
            pkl = pickle.load(f)
        pos_tic = pkl['pos_tic']

        extension_t = []
        for (t, pos_ic) in enumerate(pos_tic):
            extension = pos_ic[-1][-1] - z_i[-1]
            extension_t.append(extension)

        force = op.dirname(fname_coords).split('/')[-1]
        param = op.dirname(fname_coords).split('/')[-2]
        fname_fig = op.join(op.dirname(fname_coords), 'extension_vs_time.png')

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        assert force.endswith('pN')
        ax.plot(range(len(extension_t)), extension_t, label='%s, %d pN' % (param, int(force[:-2])))
        ax.legend(loc='upper right', fontsize='small')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Extension / nm')
        plt.tight_layout()
        plt.savefig(fname_fig)
        plt.close()


def force_vs_extension(n_part=39, box_l=3*[200]):
    '''Plot force extension curve'''
    part_i = np.array(list(range(n_part)))
    z0_i = 2.78 * part_i
    z_i = z0_i + box_l[2] / 2 - 2.78 * n_part / 2

    for dir_param in sorted(glob('output/tensile_param_scan_kb300_kt150/*')):
        force_set = []
        extension_set = []
        for dir_force in sorted(glob(op.join(dir_param, '*pN'))):
            force = int(op.basename(dir_force)[:-2])
            force_set.append(force)

            fname_coords= op.join(dir_force, 'coords.pkl')
            with open(fname_coords, 'rb') as f:
                pkl = pickle.load(f)
            pos_tic = pkl['pos_tic']
            extension = pos_tic[-1][-1][-1] - z_i[-1]
            extension_set.append(extension)

        slope, intercept = np.polyfit(extension_set, force_set, 1)
        print('%s: stiffness = %.1f' % (op.basename(dir_param), slope))
        param = op.basename(dir_param)
        fname_fig = op.join(dir_param, 'force_vs_extension.png')

        print('Plotting force extension curve for %s' % op.basename(dir_param))
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.scatter(extension_set, force_set, c='tab:blue', label='%s' % (param))
        ax.plot(extension_set, [slope*i+intercept for i in extension_set], c='tab:orange', label='k = %.1f' % slope)
        ax.legend(loc='upper left', fontsize='small')
        ax.set_xlabel('Extension / nm')
        ax.set_ylabel('Force / pN')
        plt.tight_layout()
        plt.savefig(fname_fig)
        plt.close()


def bending_twisting_param_scan(box_l=3*[1100], n_part=100, kT=2.44, gamma=25151, seed=41, img_interval=100):
    required_features = ["EXTERNAL_FORCES", "MASS"]
    espressomd.assert_features(required_features)

    box_l = box_l  # nm
    n_part = n_part

    r = 1.5  # Radius of each particle/subunit in nm
    part_i = np.array(list(range(n_part)))
    z0_i = 2.78 * part_i  # Axial translocation of protomer is 2.78 nm
    theta_i = - 166.67 / 180 * np.pi * part_i  # Rotation angle around the helical axis is 166.67 degrees
    x0_i = r * np.cos(theta_i)
    y0_i = r * np.sin(theta_i)

    # Calculate the lateral and axial bond length
    r_lateral = np.sqrt((x0_i[1] - x0_i[0]) ** 2 + (y0_i[1] - y0_i[0]) ** 2 +
                        (z0_i[1] - z0_i[0]) ** 2)
    r_axial = np.sqrt((x0_i[2] - x0_i[0]) ** 2 + (y0_i[2] - y0_i[0]) ** 2 +
                      (z0_i[2] - z0_i[0]) ** 2)

    # Move the actin filament to the center of the box
    z_i = z0_i + box_l[2] / 2 - 2.78 * n_part / 2
    x_i = x0_i + box_l[0] / 2
    y_i = y0_i + box_l[1] / 2

    system = espressomd.System(box_l=box_l)
    #system.time_step = 0.01  # time scale unit is ps
    system.time_step = 0.1  # time scale unit is ps
    system.cell_system.skin = 0.01  # nm
    system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=seed)

    # Set up particles
    pos_i = np.stack((x_i, y_i, z_i), axis=1)
    mass_i = 42051 * np.ones_like(part_i)
    system.part.add(id=part_i[::2], pos=pos_i[::2], mass=mass_i[::2], type=np.zeros_like(part_i[::2]))
    system.part.add(id=part_i[1::2], pos=pos_i[1::2], mass=mass_i[1::2], type=np.ones_like(part_i[1::2]))

    # Set up integration
    int_steps = 100
    int_iterations = 100000

    # Set harmonic bond potentials
    bond_lateral = HarmonicBond(k=53, r_0=r_lateral)
    bond_axial = HarmonicBond(k=265, r_0=r_axial)
    system.bonded_inter[0] = bond_lateral
    system.bonded_inter[1] = bond_axial

    # Add harmonic bond potential
    for i in range(n_part - 1):
        system.part.by_id(i).add_bond((bond_lateral, system.part.by_id(i + 1)))
    for i in range(n_part - 2):
        system.part.by_id(i).add_bond((bond_axial, system.part.by_id(i + 2)))

    # Bending and twisting spring constants to scan
    '''
    k_bending_set = [100, 500, 1000]
    k_twisting_set = [50, 100, 500]
    '''
    k_bending_set = [300]
    k_twisting_set = [150]

    for k_bending in k_bending_set:
            # Set harmonic angle potentials
            # Treat particle2 as the center particle
            angle_021 = AngleHarmonic(bend=4*k_bending, phi0=util.bond_angle(pos_i[0], pos_i[2], pos_i[1]))
            angle_023 = AngleHarmonic(bend=2*k_bending, phi0=util.bond_angle(pos_i[0], pos_i[2], pos_i[3]))
            angle_024 = AngleHarmonic(bend=k_bending, phi0=util.bond_angle(pos_i[0], pos_i[2], pos_i[4]))
            angle_123 = AngleHarmonic(bend=8*k_bending, phi0=util.bond_angle(pos_i[1], pos_i[2], pos_i[3]))
            angle_124 = AngleHarmonic(bend=2*k_bending, phi0=util.bond_angle(pos_i[1], pos_i[2], pos_i[4]))
            angle_324 = AngleHarmonic(bend=4*k_bending, phi0=util.bond_angle(pos_i[3], pos_i[2], pos_i[4]))

            system.bonded_inter[2] = angle_021
            system.bonded_inter[3] = angle_023
            system.bonded_inter[4] = angle_024
            system.bonded_inter[5] = angle_123
            system.bonded_inter[6] = angle_124
            system.bonded_inter[7] = angle_324

            # Add harmonic angle potential
            # part[i] should always be the vertex of the angle
            # i.e. the center particle of the triplet
            for i in range(2, n_part - 2):
                system.part.by_id(i).add_bond((angle_021, system.part.by_id(i - 2), system.part.by_id(i - 1)))
                system.part.by_id(i).add_bond((angle_023, system.part.by_id(i - 2), system.part.by_id(i + 1)))
                system.part.by_id(i).add_bond((angle_024, system.part.by_id(i - 2), system.part.by_id(i + 2)))
                system.part.by_id(i).add_bond((angle_123, system.part.by_id(i - 1), system.part.by_id(i + 1)))
                system.part.by_id(i).add_bond((angle_124, system.part.by_id(i - 1), system.part.by_id(i + 2)))
                system.part.by_id(i).add_bond((angle_324, system.part.by_id(i + 1), system.part.by_id(i + 2)))

            for k_twisting in k_twisting_set:
                folder_output = 'output/bending_twisting_param_scan/kb%.1f_kt%.1f' % (k_bending, k_twisting)
                sh.mkdir('-p', folder_output)

                # Set dihedral potentials
                # Dihedral angle with the connection between particles 1&2 as the edge
                # This dihedral is the main regulator of the twisting/untwisting of F-actin
                dihedral_edge12 = Dihedral(
                    bend=k_twisting, mult=1, phase=util.dihedral_angle(pos_i[0], pos_i[1], pos_i[2], pos_i[3]))

                # Dihedral angle with the connection between particles 0&3 as the edge
                dihedral_edge03 = Dihedral(
                    bend=(5*k_twisting), mult=1, phase=util.dihedral_angle(pos_i[2], pos_i[0], pos_i[3], pos_i[1]))

                system.bonded_inter[8] = dihedral_edge12
                system.bonded_inter[9] = dihedral_edge03

                # Add dihedral potentials
                for i in range(n_part - 3):
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

                visualizer = visualization.openGLLive(system,
                                                      force_arrows=False,
                                                      background_color=[1, 1, 1],   # white background
                                                      bond_type_radius=[0.3],
                                                      ext_force_arrows=True,
                                                      ext_force_arrows_type_scale=[1.0],
                                                      ext_force_arrows_type_radii=[1.0],
                                                      particle_sizes=[3.0],
                                                      particle_type_colors=[[0.39, 0.58, 0.93], [0.68, 0.85, 0.90]],
                                                      particle_type_materials=['bright', 'bright'],
                                                      camera_position=[550, 1000, 550],   # side view
                                                      camera_right=[1, 0, 0])


                pos_tic = []
                for int_iteration in range(int_iterations):
                    print('run %d at time=%f' % (int_iteration, system.time))
                    system.integrator.run(int_steps)

                    folder_img = op.join(folder_output, 'img')
                    sh.mkdir('-p', folder_img)
                    if (int_iteration % img_interval == 0):
                        fname_img = op.join(folder_img, '%05d.png' % int_iteration)
                        visualizer.screenshot(fname_img)

                    pos_ic = observables.ParticlePositions(ids=part_i)
                    pos_tic.append(pos_ic.calculate())

                pos_tic = np.array(pos_tic)
                output = dict(pos_tic=pos_tic)

                print('Saving coords to pkl')
                fname_coords = op.join(folder_output, 'coords.pkl')
                with open(fname_coords, 'wb') as f:
                    pickle.dump(output, f, protocol=2)

                for i in range(n_part):
                    # Use _k to index the indices in bond_j with a bond type of DIHEDRAL
                    index_k = []

                    # Use _j to index bonds belonging to a particular particle
                    for j, bond in enumerate(system.part.by_id(i).bonds):
                        if bond[0].type_name() == 'DIHEDRAL':
                            index_k.append(j)

                    for index in reversed(index_k):
                        system.part.by_id(i).delete_bond(system.part.by_id(i).bonds[index])

            # Delete angle bonds to reset the system for next bending param scan
            for i in range(n_part):
                # Use _h to index the indices in bond_l with a bond type of ANGLE_HARMONIC
                index_h = []

                # Use _l to index bonds belonging to a particular particle after deleting dihedral bonds
                for l, bond in enumerate(system.part.by_id(i).bonds):
                    if bond[0].type_name() == 'ANGLE_HARMONIC':
                        index_h.append(l)

                for index in reversed(index_h):
                    system.part.by_id(i).delete_bond(system.part.by_id(i).bonds[index])


def compute_rise_twist():
    for fname_coords in sorted(glob('output/bending_twisting_param_scan/*/coords.pkl')):
        print('Computing rise and twist for %s' % op.dirname(fname_coords).split('/')[-1])
        with open(fname_coords, 'rb') as f:
            pkl = pickle.load(f)
        pos_tic = pkl['pos_tic']

        mid1_tjc = []
        mid2_tkc = []
        for t, pos_ic in enumerate(pos_tic):
            # mid1 is the mid point between adjacent particles and is indexed by j.
            # n_mid1 = n_part - 1
            mid1_jc = [(pos_c + pos_ic[i - 1]) / 2 for i, pos_c in enumerate(pos_ic) if i != 0]
            mid1_tjc.append(mid1_jc)

            # mid2 is the mid point between adjacent mid1s and is indexd by h.
            # n_mid2 = n_mid1 - 1 = n_part - 2
            mid2_kc = [(mid1_c + mid1_jc[j - 1]) / 2 for j, mid1_c in enumerate(mid1_jc) if j != 0]
            mid2_tkc.append(mid2_kc)

        mid1_tjc = np.array(mid1_tjc)
        mid2_tkc = np.array(mid2_tkc)

        # mid2 vectors are generated by subtracting adjacent mid2 points and are indexed by h.
        # n_mid2_vector = n_mid2 - 1 = n_part - 3
        mid2_vector_thc = [mid2_kc[1:] - mid2_kc[:-1] for mid2_kc in mid2_tkc]
        mid2_vector_thc = np.array(mid2_vector_thc)

        # Particle vectors are generated by subtracting particle coords by mid2 coords.
        # The first and last particle are removed to match the number of mid points in mid2_tkc (k).
        part_vector_tkc = []
        for (pos_kc, mid2_kc) in zip(pos_tic[:, 1:-1, :], mid2_tkc):
            part_vector_kc = pos_kc - mid2_kc
            part_vector_tkc.append(part_vector_kc)
        part_vector_tkc = np.array(part_vector_tkc)

        # Project particle vectors onto mid2 vectors
        # Find coordinates of the projections
        proj1_thc = []
        for (part_vector_hc, mid2_vector_hc, mid2_hc) in zip(part_vector_tkc[:, :-1, :], mid2_vector_thc, mid2_tkc[:, :-1, :]):
            proj1_hc = []
            for (part_vector_c, mid2_vector_c, mid2_c) in zip(part_vector_hc, mid2_vector_hc, mid2_hc):
                proj1_vector_c = np.dot(part_vector_c, mid2_vector_c) / np.dot(mid2_vector_c, mid2_vector_c) * mid2_vector_c
                proj1_c = proj1_vector_c + mid2_c
                proj1_hc.append(proj1_c)
            proj1_thc.append(proj1_hc)
        proj1_thc = np.array(proj1_thc)

        proj2_thc = []
        for (part_vector_hc, mid2_vector_hc, mid2_hc) in zip(part_vector_tkc[:, 1:, :], mid2_vector_thc, mid2_tkc[:, 1:, :]):
            proj2_hc = []
            for (part_vector_c, mid2_vector_c, mid2_c) in zip(part_vector_hc, mid2_vector_hc, mid2_hc):
                proj2_vector_c = np.dot(part_vector_c, mid2_vector_c) / np.dot(mid2_vector_c, mid2_vector_c) * mid2_vector_c
                proj2_c = proj2_vector_c + mid2_c
                proj2_hc.append(proj2_c)
            proj2_thc.append(proj2_hc)
        proj2_thc = np.array(proj2_thc)

        # Compute rise
        rise_vector_thc = [proj2_hc - proj1_hc for (proj2_hc, proj1_hc) in zip(proj2_thc, proj1_thc)]
        rise_th = []
        for rise_vector_hc in rise_vector_thc:
            rise_h = [np.linalg.norm(rise_vector_c) for rise_vector_c in rise_vector_hc]
            rise_th.append(rise_h)
        rise_th = np.array(rise_th)

        output_rise = dict(rise_th=rise_th)
        print('Saving rise values to pkl')
        fname_rise = op.join(op.dirname(fname_coords), 'rise.pkl')
        with open(fname_rise, 'wb') as f:
            pickle.dump(output_rise, f, protocol=2)

        # Find normals to the projections
        normal1_thc = []
        for (proj1_hc, pos_hc) in zip(proj1_thc, pos_tic[:, 1:-2, :]):
            normal1_hc = []
            for (proj1_c, pos_c) in zip(proj1_hc, pos_hc):
                normal1_c = pos_c - proj1_c
                normal1_hc.append(normal1_c)
            normal1_thc.append(normal1_hc)
        normal1_thc = np.array(normal1_thc)

        normal2_thc = []
        for (proj2_hc, pos_hc) in zip(proj2_thc, pos_tic[:, 2:-1, :]):
            normal2_hc = []
            for (proj2_c, pos_c) in zip(proj2_hc, pos_hc):
                normal2_c = pos_c - proj2_c
                normal2_hc.append(normal2_c)
            normal2_thc.append(normal2_hc)
        normal2_thc = np.array(normal2_thc)

        # Compute twist based on the neighboring normals
        twist_th = []
        for (normal1_hc, normal2_hc) in zip(normal1_thc, normal2_thc):
            twist_h = []
            for (normal1_c, normal2_c) in zip(normal1_hc, normal2_hc):
                twist = - np.arccos(np.dot(normal1_c, normal2_c) / (np.linalg.norm(normal1_c) * np.linalg.norm(normal2_c)))
                twist = twist / np.pi * 180
                twist_h.append(twist)
            twist_th.append(twist_h)
        twist_th = np.array(twist_th)

        output_twist = dict(twist_th=twist_th)
        print('Saving twist values to pkl')
        fname_twist = op.join(op.dirname(fname_coords), 'twist.pkl')
        with open(fname_twist , 'wb') as f:
            pickle.dump(output_twist, f, protocol=2)


def delta_twist_distribution(theta=-166.67, time_start=1000, part_start=20):
    '''
    Plot probability density distribution of twist change.

    theta: canonical twist.
    time_start: frames before time_start are used to equilibrate the system.
    part_start: skip the first few particles to avoid edge effect.
    '''
    for fname_twist in glob('output/bending_twisting_param_scan/*/twist.pkl'):
        print(fname_twist)
        with open(fname_twist, 'rb') as f:
            pkl = pickle.load(f)
        twist_th = pkl['twist_th']
        twist_TH = twist_th[time_start:, part_start:(part_start + 60)]

        delta_twist_TH = []
        for twist_H in twist_TH:
            cum_twist_H = np.cumsum(np.array(twist_H))
            delta_twist_H = []
            for (H, cum_twist) in enumerate(cum_twist_H):
                delta_twist = cum_twist - (H + 1) * theta
                delta_twist_H.append(delta_twist)
            delta_twist_TH.append(delta_twist_H)
        delta_twist_TH = np.array(delta_twist_TH)

        # Treat part[part_start - 1] as a reference, i.e., the 0th particle.
        # Plot probability density distribution of twist change of the 1st, 10th, and 50th particle.
        print('Plotting probability density distribution')
        indices = [0, 9, 49]
        for index in indices:
            fname_hist = op.join(op.dirname(fname_twist), 'hist_n%02d.png' % index)
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            _, bins, _ = ax.hist(delta_twist_TH[:, index], bins=20, density=True, color='tab:blue')
            # Fit Gaussian to the histogram with mean zero
            # mu, sigma = ss.norm.fit(delta_twist_TH[:, index])
            mu = 0
            sigma = math.sqrt(np.sum([delta_twist ** 2 for delta_twist in delta_twist_TH[:, index]]) / len(delta_twist_TH[:, index]))
            fit_line = ss.norm.pdf(bins, mu, sigma)
            ax.plot(bins, fit_line, color='tab:orange')
            ax.set_xlim([-50, 50])
            ax.set_xlabel('Twist change / Degree')
            ax.set_ylabel('Probability density')
            plt.tight_layout()
            plt.savefig(fname_hist)
            plt.close()

        # Plot variance against H
        print('Plotting variance against n')
        var_H = []
        for index in range(delta_twist_TH.shape[-1]):
            var = np.sum([delta_twist ** 2 for delta_twist in delta_twist_TH[:, index]]) / len(delta_twist_TH[:, index])
            var_H.append(var)

        fname_var = op.join(op.dirname(fname_twist), 'var_vs_n.png')
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.scatter(range(len(var_H)), var_H, color='tab:blue')
        slope, intercept = np.polyfit(range(len(var_H)), var_H, 1)
        ax.plot(range(len(var_H)), [slope*i+intercept for i in range(len(var_H))], c='tab:orange')
        ax.set_xlabel('n')
        ax.set_ylabel('Variance')
        plt.tight_layout()
        plt.savefig(fname_var)
        plt.close()


def plot_force_vs_extension(dir_param, n_part=39, box_l=3*[200]):
    part_i = np.array(list(range(n_part)))
    z0_i = 2.78 * part_i
    z_i = z0_i + box_l[2] / 2 - 2.78 * n_part / 2

    force_set = []
    extension_set = []
    for dir_force in sorted(glob(op.join(dir_param, '*pN'))):
        force = int(op.basename(dir_force)[:-2])
        force_set.append(force)

        fname_coords= op.join(dir_force, 'coords.pkl')
        with open(fname_coords, 'rb') as f:
            pkl = pickle.load(f)
        pos_tic = pkl['pos_tic']
        extension = pos_tic[-1][-1][-1] - z_i[-1]
        extension_set.append(extension)

    slope, intercept = np.polyfit(extension_set, force_set, 1)
    print('%s: stiffness = %.1f' % (op.basename(dir_param), slope))

    output_fig = 'output/fig'
    sh.mkdir('-p', output_fig)
    fname_fig = op.join(output_fig, 'force_vs_extension.pdf')

    print('Plotting force extension curve for %s' % op.basename(dir_param))
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.scatter(extension_set, force_set, c='0.7')
    ax.plot(extension_set, [slope*i+intercept for i in extension_set],
            c='0', label='k = %.1f' % slope)
    ax.legend(loc='upper left', fontsize='small')
    ax.set_xlabel('Extension / nm')
    ax.set_ylabel('Force / pN')
    plt.tight_layout()
    plt.savefig(fname_fig)
    plt.close()


def plot_delta_twist_distribution(fname_twist, theta=-166.67, time_start=1000, part_start=20):
    with open(fname_twist, 'rb') as f:
        pkl = pickle.load(f)
    twist_th = pkl['twist_th']
    twist_TH = twist_th[time_start:, part_start:(part_start + 60)]

    delta_twist_TH = []
    for twist_H in twist_TH:
        cum_twist_H = np.cumsum(np.array(twist_H))
        delta_twist_H = []
        for (H, cum_twist) in enumerate(cum_twist_H):
            delta_twist = cum_twist - (H + 1) * theta
            delta_twist_H.append(delta_twist)
        delta_twist_TH.append(delta_twist_H)
    delta_twist_TH = np.array(delta_twist_TH)

    output_fig = 'output/fig'
    sh.mkdir('-p', output_fig)
    fname_fig = op.join(output_fig, 'var_vs_n.pdf')

    print('Plotting variance against n')
    var_H = []
    for index in range(delta_twist_TH.shape[-1]):
        var = np.sum([delta_twist ** 2 for delta_twist in delta_twist_TH[:, index]]) / len(delta_twist_TH[:, index])
        var_H.append(var)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.scatter(range(len(var_H)), var_H, c='0.7')
    slope, intercept = np.polyfit(range(len(var_H)), var_H, 1)
    ax.plot(range(len(var_H)), [slope*i+intercept for i in range(len(var_H))], c='0')
    ax.set_xlabel('Subunit index')
    ax.set_ylabel('Variance / Degree')
    plt.tight_layout()
    plt.savefig(fname_fig)
    plt.close()


if __name__ == '__main__':
    #tensile_param_scan()
    #extension_vs_time()
    #force_vs_extension()
    #bending_twisting_param_scan()
    #compute_rise_twist()
    #delta_twist_distribution()
    #plot_force_vs_extension('output/tensile_param_scan_kb300_kt150/ka265.0_kl53.0')
    plot_delta_twist_distribution('output/bending_twisting_param_scan_ka265/kb300.0_kt150.0/twist.pkl')
    pass
