import numpy as np
import os.path as op
import sh
from glob import glob
import pickle

import espressomd
from espressomd import integrate
from espressomd import observables
from espressomd import visualization
import actin_potential as ap

system = ap.system
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
                                      camera_position=[250, 2000, 750],
                                      camera_right=[1, 0, 0],
                                      window_size=[500, 1500],
                                      draw_box=False,
                                      draw_axis=False,
                                      )


def thermal_fluct(frame_step, frame_num, fix_start, fix_end, save_img=True):
    '''
    Simulate the thermal fluctuation of an actin filament in aqueous buffer at RT.
    frame_step: int, integrator run steps between sampling.
    frame_num: int, total sampling number.
    '''
    seed = ap.seed
    part_num = ap.part_num
    if not fix_start and not fix_end:
        folder_seed = 'output/thermal_fluct/fix0/seed%d' % seed
    elif fix_start and not fix_end:
        folder_seed = 'output/thermal_fluct/fix1/seed%d' % seed
    elif fix_start and fix_end:
        folder_seed = 'output/thermal_fluct/fix2/seed%d' % seed
    sh.mkdir('-p', folder_seed)
    folder_img = op.join(folder_seed, 'img')
    sh.mkdir('-p', folder_img)

    if fix_start:
        system.part.by_ids(range(2)).fix = [True, True, True]
    if fix_end:
        system.part.by_ids(range(part_num)[-2:]).fix = [True, True, True]

    print('Equilibration')
    system.integrator.run(500000)
    print('Equilibration finished')

    ind_i = ap.ind_i
    pos_tic = []
    for frame in range(frame_num):
        print('run %d at time=%f' % (frame, system.time))
        if save_img:
            fname_img = op.join(folder_img, '%04d.png' % frame)
            visualizer.screenshot(fname_img)

        system.integrator.run(frame_step)
        pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
        pos_tic.append(pos_ic)

    pos_tic = np.array(pos_tic)
    output = dict(pos_tic=pos_tic, seed=seed, frame_step=frame_step,
                  frame_num=frame_num)
    print('Saving coords to pkl')
    fname_coords = op.join(folder_seed, 'coords.pkl')
    with open(fname_coords, 'wb') as f:
        pickle.dump(output, f, protocol=2)


def force_constant(direction, magnitude, part_force_num, frame_step, frame_num, save_img=True):
    '''
    Fix one end of both strands. Pull or push constantly on the other end of both strands.
    direction: str, 'pull' or 'push'.
    magnitude: float, magnitude of force applied on each particle/subunit.
    part_force_num: int, number of particles under force.
    frame_step: int, integrator run steps between sampling.
    frame_num: int, total sampling number.
    '''
    seed = ap.seed
    system.part.by_ids(range(2)).fix = [True, True, True]
    z_force = {'pull': 1, 'push': -1}[direction] * magnitude

    folder_seed = 'output/force_constant/%s_f%.1f_n%d/seed%d' % (direction, magnitude, part_force_num, seed)
    sh.mkdir('-p', folder_seed)
    folder_img = op.join(folder_seed, 'img')
    sh.mkdir('-p', folder_img)

    print('Equilibration')
    system.integrator.run(500000)
    print('Equilibration finished')

    print('Applying force')
    system.part.by_ids(list(range(ap.part_num))[-part_force_num:]).fix = [True, True, False]
    system.part.by_ids(list(range(ap.part_num))[-part_force_num:]).ext_force = [0, 0, z_force]

    ind_i = ap.ind_i
    pos_tic = []
    for frame in range(frame_num):
        print('run %d at time=%f' % (frame, system.time))
        if save_img:
            fname_img = op.join(folder_img, '%05d.png' % frame)
            visualizer.screenshot(fname_img)

        system.integrator.run(frame_step)
        pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
        pos_tic.append(pos_ic)

        # Overwrite and save the pkl file every 500 iterations
        if frame % 500 == 0:
            output = dict(pos_tic=np.array(pos_tic), seed=seed,
                          direction=direction, magnitude=magnitude,
                          part_force_num=part_force_num, frame_step=frame_step,
                          frame_num=frame_num)
            print('Saving coords to pkl')
            fname_coords = op.join(folder_seed, 'coords.pkl')
            with open(fname_coords, 'wb') as f:
                pickle.dump(output, f, protocol=2)


def force_release(direction, magnitude, part_force_num, force_duration, relax_duration, frame_step, save_img=True):
    '''
    Fix one end of both strands. Pull or push on the other end of both strands, then release.
    direction: str, 'pull' or 'push'.
    magnitude: float, magnitude of force applied on each particle/subunit.
    part_force_num: int, number of particles under force.
    force_duration: int.
    relax_duration: int.
    frame_step: int, integrator run steps between sampling.
    '''
    seed = ap.seed
    system.part.by_ids(range(2)).fix = [True, True, True]
    z_force = {'pull': 1, 'push': -1}[direction] * magnitude

    folder_seed = 'output/force_release/%s_f%.1f/seed%d' % (direction, magnitude, seed)
    sh.mkdir('-p', folder_seed)
    folder_img = op.join(folder_seed, 'img')
    sh.mkdir('-p', folder_img)

    print('Equilibration')
    system.integrator.run(500000)
    print('Equilibration finished')

    print('Applying force')
    system.part.by_ids(list(range(ap.part_num))[-part_force_num:]).fix = [True, True, False]
    system.part.by_ids(list(range(ap.part_num))[-part_force_num:]).ext_force = [0, 0, z_force]

    ind_i = ap.ind_i
    pos_tic = []
    for frame in range(force_duration):
        print('run %d at time=%f' % (frame, system.time))
        system.integrator.run(frame_step)

        if save_img:
            fname_img = op.join(folder_img, '%05d.png' % frame)
            visualizer.screenshot(fname_img)

        pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
        pos_tic.append(pos_ic)

    print('Force released')
    system.part.by_ids(list(range(ap.part_num))[-part_force_num:]).fix = [False, False, False]
    system.part.by_ids(list(range(ap.part_num))[-part_force_num:]).ext_force = [0, 0, 0]

    for frame in range(force_duration, force_duration + relax_duration):
        print('run %d at time=%f' % (frame, system.time))
        if save_img:
            fname_img = op.join(folder_img, '%05d.png' % frame)
            visualizer.screenshot(fname_img)

        system.integrator.run(frame_step)
        pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
        pos_tic.append(pos_ic)

        # Overwrite and save the pkl file every 500 iterations
        if frame % 500 == 0:
            output = dict(pos_tic=np.array(pos_tic), seed=seed,
                          direction=direction, magnitude=magnitude,
                          part_force_num=part_force_num,
                          force_duration=force_duration,
                          relax_duration=relax_duration, frame_step=frame_step)
            print('Saving coords to pkl')
            fname_coords = op.join(folder_seed, 'coords.pkl')
            with open(fname_coords, 'wb') as f:
                pickle.dump(output, f, protocol=2)


def force_random_firing(direction, magnitude, part_force_num, frame_step,
                        frame_num, rng_seed, save_img=True):
    seed = ap.seed
    part_num = ap.part_num
    ind_i = ap.ind_i
    system.part.by_ids(range(2)).fix = [True, True, True]
    z_force = {'pull': 1, 'push': -1}[direction] * magnitude

    folder_seed = 'output/force_random_firing/%s_f%.1f_pn%d/seed%d_rng%d' % (direction, magnitude, part_force_num, seed, rng_seed)
    sh.mkdir('-p', folder_seed)
    folder_img = op.join(folder_seed, 'img')
    sh.mkdir('-p', folder_img)

    print('Equilibration')
    system.integrator.run(500000)
    print('Equilibration finished')

    rng = np.random.RandomState(rng_seed)
    part_n = rng.randint((part_num - 100), part_num, part_force_num)
    duration_n = rng.randint(200, 400, part_force_num)
    start_n = rng.randint(0, 200, part_force_num)
    end_n = [(start + duration) for (start, duration) in zip(start_n, duration_n)]
    print(part_n, start_n, end_n)

    print('Applying force randomly')
    pos_tic = []
    for frame in range(frame_num):
        if save_img:
            fname_img = op.join(folder_img, '%05d.png' % frame)
            visualizer.screenshot(fname_img)

        for (part, start, end) in zip(part_n, start_n, end_n):
            if frame == start:
                system.part.by_ids([part]).fix = [True, True, False]
                system.part.by_ids([part]).ext_force = [0, 0, z_force]
            elif frame > end:
                system.part.by_ids([part]).fix = [False, False, False]
                system.part.by_ids([part]).ext_force = [0, 0, 0]

        print('run %d at time=%f' % (frame, system.time))
        system.integrator.run(frame_step)

        pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
        pos_tic.append(pos_ic)

        # Overwrite and save the pkl file every 500 iterations
        if frame % 500 == 0:
            output = dict(pos_tic=np.array(pos_tic), seed=seed,
                          direction=direction, magnitude=magnitude,
                          part_force_num=part_force_num, frame_step=frame_step,
                          frame_num=frame_num, rng_seed=rng_seed)
            print('Saving coords to pkl')
            fname_coords = op.join(folder_seed, 'coords.pkl')
            with open(fname_coords, 'wb') as f:
                pickle.dump(output, f, protocol=2)


def bond_breaking(direction, magnitude, part_force_num, frame_step, frame_num,
                  bond_diag_thresh=1.072, bond_long_thresh=1.168, save_img=True):
    k_long = ap.k_long
    tensile_factor = ap.tensile_factor
    seed = ap.seed
    part_num = ap.part_num
    ind_i = ap.ind_i
    system.part.by_ids(range(2)).fix = [True, True, True]
    z_force = {'pull': 1, 'push': -1}[direction] * magnitude

    print('Equilibration')
    for frame in range(500000):
        if frame % 1000 == 0:
            print('run %d at time=%f' % (frame, system.time))
        system.integrator.run(1)

        r_diag_j = observables.ParticleDistances(ids=ind_i).calculate()   # j = i - 1
        ind_k = [j for j, r_diag in enumerate(r_diag_j) if r_diag > bond_diag_thresh * ap.r_diag]
        if ind_k:
            print('Diagonal bond length above threshold for particle', ind_k)

        r_long_l = observables.ParticleDistances(ids=ind_i[::2]).calculate()   # l = i / 2 - 1
        r_long_m = observables.ParticleDistances(ids=ind_i[1::2]).calculate()   # m = i / 2 - 1
        ind_n = []
        for ind, r_long in zip(ind_i[::2], r_long_l):
            if r_long > bond_long_thresh * ap.r_long:
                ind_n.append(ind)
        for ind, r_long in zip(ind_i[1::2], r_long_m):
            if r_long > bond_long_thresh * ap.r_long:
                ind_n.append(ind)
        ind_n.sort()
        if ind_n:
            print('Longitudinal bond length above threshold for particle', ind_n)
    print('Equilibration finished')

    folder_seed = 'output/bond_breaking/%s_f%.1f_n%d/seed%d' % (direction, magnitude, part_force_num, seed)
    sh.mkdir('-p', folder_seed)
    folder_img = op.join(folder_seed, 'img')
    sh.mkdir('-p', folder_img)

    print('Applying force')
    system.part.by_ids(list(range(ap.part_num))[-part_force_num:]).fix = [True, True, False]
    system.part.by_ids(list(range(ap.part_num))[-part_force_num:]).ext_force = [0, 0, z_force]

    pos_tic = []
    ind_p = []
    ind_u = []
    for frame in range(frame_num):
        if save_img and frame % frame_step == 0:
            print('run %d at time=%f' % (frame, system.time))
            fname_img = op.join(folder_img, '%07d.png' % frame)
            visualizer.screenshot(fname_img)

        system.integrator.run(1)

        r_diag_j = observables.ParticleDistances(ids=ind_i).calculate()   # j = i - 1
        ind_q = [j for j, r_diag in enumerate(r_diag_j) if r_diag > bond_diag_thresh * ap.r_diag]
        ind_k = [ind for ind in ind_q if ind not in ind_p]
        ind_p += ind_k
        if ind_k:
            print('Diagonal bond broken for particle', ind_k)
            for ind in ind_k:
                part_0 = system.part.by_id(ind)
                part_0.type = 2   # Assign type 2 to particles with broken diag bond
                for bond in part_0.bonds:
                    if ind + 1 in bond[1:]:
                        part_0.delete_bond(bond)
                if len(part_0.bonds) == 0:
                    part_0.type = 5   # Assign type 5 to particles with all bonds broken, i.e., a dissociated monomer
                else:
                    part_0_HarmonicBond = [bond for bond in part_0.bonds if isinstance(bond[0], espressomd.interactions.HarmonicBond)]
                    if not part_0_HarmonicBond:
                        part_0.type = 4   # Assign type 4 to particles with both diag and long bonds broken

                part_1 = system.part.by_id(ind + 1)
                for bond in part_1.bonds:
                    if ind in bond[1:]:
                        part_1.delete_bond(bond)
                if len(part_1.bonds) == 0:
                    part_1.type = 5   # Assign type 5 to particles with all bonds broken, i.e., a dissociated monomer
                else:
                    part_1_HarmonicBond = [bond for bond in part_1.bonds if isinstance(bond[0], espressomd.interactions.HarmonicBond)]
                    if not part_1_HarmonicBond:
                        part_1.type = 4   # Assign type 4 to particles with both diag and long bonds broken

        r_long_l = observables.ParticleDistances(ids=ind_i[::2]).calculate()   # l = i / 2 - 1
        r_long_m = observables.ParticleDistances(ids=ind_i[1::2]).calculate()   # m = i / 2 - 1
        ind_n = []
        for ind, r_long in zip(ind_i[::2], r_long_l):
            if r_long > bond_long_thresh * ap.r_long:
                ind_n.append(ind)
        for ind, r_long in zip(ind_i[1::2], r_long_m):
            if r_long > bond_long_thresh * ap.r_long:
                ind_n.append(ind)
        ind_n.sort()
        ind_v = [ind for ind in ind_n if ind not in ind_u]
        ind_u += ind_v
        if ind_v:
            print('Longitudinal bond broken for particle', ind_v)
            for ind in ind_v:
                part_0 = system.part.by_id(ind)
                part_0.type = 3   # Assign type 3 to particles with broken long bond
                for bond in part_0.bonds:
                    if ind + 2 in bond[1:]:
                        part_0.delete_bond(bond)
                if len(part_0.bonds) == 0:
                    part_0.type = 5   # Assign type 5 to particles with all bonds broken, i.e., a dissociated monomer
                else:
                    part_0_HarmonicBond = [bond for bond in part_0.bonds if isinstance(bond[0], espressomd.interactions.HarmonicBond)]
                    if not part_0_HarmonicBond:
                        part_0.type = 4   # Assign type 4 to particles with both diag and long bonds broken

                part_2 = system.part.by_id(ind + 2)
                for bond in part_2.bonds:
                    if ind in bond[1:]:
                        part_2.delete_bond(bond)
                if len(part_2.bonds) == 0:
                    part_2.type = 5   # Assign type 5 to particles with all bonds broken, i.e., a dissociated monomer
                else:
                    part_2_HarmonicBond = [bond for bond in part_2.bonds if isinstance(bond[0], espressomd.interactions.HarmonicBond)]
                    if not part_2_HarmonicBond:
                        part_2.type = 4   # Assign type 4 to particles with both diag and long bonds broken

        if frame % frame_step == 0:
            pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
            pos_tic.append(pos_ic)

        if frame % (frame_step * 500) == 0:
            output = dict(pos_tic=np.array(pos_tic), seed=seed, direction=direction,
                          magnitude=magnitude, part_force_num=part_force_num,
                          frame_step=frame_step, frame_num=frame_num,
                          bond_diag_thresh=bond_diag_thresh,
                          bond_long_thresh=bond_long_thresh)
            print('Saving coords to pkl')
            fname_coords = op.join(folder_seed, 'coords.pkl')
            with open(fname_coords, 'wb') as f:
                pickle.dump(output, f, protocol=2)


def torque(direction, magnitude, frame_step, frame_num, save_img=True):
    seed = ap.seed
    system.part.by_ids(range(2)).fix = [True, True, True]

    folder_seed = 'output/torque/%s_f%.1f/seed%d' % (direction, magnitude, seed)
    sh.mkdir('-p', folder_seed)
    folder_img = op.join(folder_seed, 'img')
    sh.mkdir('-p', folder_img)

    print('Equilibration')
    system.integrator.run(500000)
    print('Equilibration finished')

    print('Applying torque')
    ind_i = ap.ind_i
    pos_tic = []
    for frame in range(frame_num):
        pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
        if save_img and frame % frame_step == 0:
            print('run %d at time=%f' % (frame, system.time))
            fname_img = op.join(folder_img, '%05d.png' % frame)
            visualizer.screenshot(fname_img)
            pos_tic.append(pos_ic)

        mid1 = (pos_ic[-1] + pos_ic[-2]) / 2
        mid2 = (pos_ic[-2] + pos_ic[-3]) / 2
        axis = mid2 - mid1
        axis_unit = axis / np.linalg.norm(axis)
        proj1 = np.dot((pos_ic[-1] - mid1), axis_unit) * axis_unit
        proj_point_1 = mid1 + proj1
        lever1 = pos_ic[-1] - proj_point_1
        proj2 = np.dot((pos_ic[-2] - mid1), axis_unit) * axis_unit
        proj_point_2 = mid1 + proj2
        lever2 = pos_ic[-2] - proj_point_2
        assert direction in ['cw', 'ccw']
        if direction == 'cw':
            force_unit_1 = np.cross(axis, lever1) / np.linalg.norm(np.cross(axis, lever1))
            force_unit_2 = np.cross(axis, lever2) / np.linalg.norm(np.cross(axis, lever2))
        elif direction == 'ccw':
            force_unit_1 = np.cross(lever1, axis) / np.linalg.norm(np.cross(lever1, axis))
            force_unit_2 = np.cross(lever2, axis) / np.linalg.norm(np.cross(lever2, axis))
        system.part.by_id(list(range(ap.part_num))[-1]).ext_force = force_unit_1 * magnitude
        system.part.by_id(list(range(ap.part_num))[-2]).ext_force = force_unit_2 * magnitude
        system.integrator.run(1)

        # Overwrite and save the pkl file every 500 iterations
        if frame % (frame_step * 500) == 0:
            output = dict(pos_tic=np.array(pos_tic), seed=seed,
                          direction=direction, magnitude=magnitude,
                          frame_step=frame_step, frame_num=frame_num)
            print('Saving coords to pkl')
            fname_coords = op.join(folder_seed, 'coords.pkl')
            with open(fname_coords, 'wb') as f:
                pickle.dump(output, f, protocol=2)


def torque_bond_breaking(direction, magnitude, frame_step, frame_num,
                         bond_diag_thresh=1.072, bond_long_thresh=1.168,
                         save_img=True):
    seed = ap.seed
    ind_i = ap.ind_i
    system.part.by_ids(range(2)).fix = [True, True, True]

    folder_seed = 'output/torque_bond_breaking/%s_f%.1f/seed%d' % (direction, magnitude, seed)
    sh.mkdir('-p', folder_seed)
    folder_img = op.join(folder_seed, 'img')
    sh.mkdir('-p', folder_img)

    print('Equilibration')
    for frame in range(500000):
        if frame % 1000 == 0:
            print('run %d at time=%f' % (frame, system.time))
        system.integrator.run(1)

        r_diag_j = observables.ParticleDistances(ids=ind_i).calculate()   # j = i - 1
        ind_k = [j for j, r_diag in enumerate(r_diag_j) if r_diag > bond_diag_thresh * ap.r_diag]
        if ind_k:
            print('Diagonal bond length above threshold for particle', ind_k)

        r_long_l = observables.ParticleDistances(ids=ind_i[::2]).calculate()   # l = i / 2 - 1
        r_long_m = observables.ParticleDistances(ids=ind_i[1::2]).calculate()   # m = i / 2 - 1
        ind_n = []
        for ind, r_long in zip(ind_i[::2], r_long_l):
            if r_long > bond_long_thresh * ap.r_long:
                ind_n.append(ind)
        for ind, r_long in zip(ind_i[1::2], r_long_m):
            if r_long > bond_long_thresh * ap.r_long:
                ind_n.append(ind)
        ind_n.sort()
        if ind_n:
            print('Longitudinal bond length above threshold for particle', ind_n)
    print('Equilibration finished')

    print('Applying torque')
    pos_tic = []
    ind_p = []
    ind_u = []
    for frame in range(frame_num):
        pos_ic = observables.ParticlePositions(ids=ind_i).calculate()
        if save_img and frame % frame_step == 0:
            print('run %d at time=%f' % (frame, system.time))
            fname_img = op.join(folder_img, '%05d.png' % frame)
            visualizer.screenshot(fname_img)
            pos_tic.append(pos_ic)

        mid1 = (pos_ic[-1] + pos_ic[-2]) / 2
        mid2 = (pos_ic[-2] + pos_ic[-3]) / 2
        axis = mid2 - mid1
        axis_unit = axis / np.linalg.norm(axis)
        proj1 = np.dot((pos_ic[-1] - mid1), axis_unit) * axis_unit
        proj_point_1 = mid1 + proj1
        lever1 = pos_ic[-1] - proj_point_1
        proj2 = np.dot((pos_ic[-2] - mid1), axis_unit) * axis_unit
        proj_point_2 = mid1 + proj2
        lever2 = pos_ic[-2] - proj_point_2
        assert direction in ['cw', 'ccw']
        if direction == 'cw':
            force_unit_1 = np.cross(axis, lever1) / np.linalg.norm(np.cross(axis, lever1))
            force_unit_2 = np.cross(axis, lever2) / np.linalg.norm(np.cross(axis, lever2))
        elif direction == 'ccw':
            force_unit_1 = np.cross(lever1, axis) / np.linalg.norm(np.cross(lever1, axis))
            force_unit_2 = np.cross(lever2, axis) / np.linalg.norm(np.cross(lever2, axis))
        system.part.by_id(list(range(ap.part_num))[-1]).ext_force = force_unit_1 * magnitude
        system.part.by_id(list(range(ap.part_num))[-2]).ext_force = force_unit_2 * magnitude
        system.integrator.run(1)

        r_diag_j = observables.ParticleDistances(ids=ind_i).calculate()   # j = i - 1
        ind_q = [j for j, r_diag in enumerate(r_diag_j) if r_diag > bond_diag_thresh * ap.r_diag]
        ind_k = [ind for ind in ind_q if ind not in ind_p]
        ind_p += ind_k
        if ind_k:
            print('Diagonal bond broken for particle', ind_k)
            for ind in ind_k:
                part_0 = system.part.by_id(ind)
                part_0.type = 2   # Assign type 2 to particles with broken diag bond
                for bond in part_0.bonds:
                    if ind + 1 in bond[1:]:
                        part_0.delete_bond(bond)
                if len(part_0.bonds) == 0:
                    part_0.type = 5   # Assign type 5 to particles with all bonds broken, i.e., a dissociated monomer
                else:
                    part_0_HarmonicBond = [bond for bond in part_0.bonds if isinstance(bond[0], espressomd.interactions.HarmonicBond)]
                    if not part_0_HarmonicBond:
                        part_0.type = 4   # Assign type 4 to particles with both diag and long bonds broken

                part_1 = system.part.by_id(ind + 1)
                for bond in part_1.bonds:
                    if ind in bond[1:]:
                        part_1.delete_bond(bond)
                if len(part_1.bonds) == 0:
                    part_1.type = 5   # Assign type 5 to particles with all bonds broken, i.e., a dissociated monomer
                else:
                    part_1_HarmonicBond = [bond for bond in part_1.bonds if isinstance(bond[0], espressomd.interactions.HarmonicBond)]
                    if not part_1_HarmonicBond:
                        part_1.type = 4   # Assign type 4 to particles with both diag and long bonds broken

        r_long_l = observables.ParticleDistances(ids=ind_i[::2]).calculate()   # l = i / 2 - 1
        r_long_m = observables.ParticleDistances(ids=ind_i[1::2]).calculate()   # m = i / 2 - 1
        ind_n = []
        for ind, r_long in zip(ind_i[::2], r_long_l):
            if r_long > bond_long_thresh * ap.r_long:
                ind_n.append(ind)
        for ind, r_long in zip(ind_i[1::2], r_long_m):
            if r_long > bond_long_thresh * ap.r_long:
                ind_n.append(ind)
        ind_n.sort()
        ind_v = [ind for ind in ind_n if ind not in ind_u]
        ind_u += ind_v
        if ind_v:
            print('Longitudinal bond broken for particle', ind_v)
            for ind in ind_v:
                part_0 = system.part.by_id(ind)
                part_0.type = 3   # Assign type 3 to particles with broken long bond
                for bond in part_0.bonds:
                    if ind + 2 in bond[1:]:
                        part_0.delete_bond(bond)
                if len(part_0.bonds) == 0:
                    part_0.type = 5   # Assign type 5 to particles with all bonds broken, i.e., a dissociated monomer
                else:
                    part_0_HarmonicBond = [bond for bond in part_0.bonds if isinstance(bond[0], espressomd.interactions.HarmonicBond)]
                    if not part_0_HarmonicBond:
                        part_0.type = 4   # Assign type 4 to particles with both diag and long bonds broken

                part_2 = system.part.by_id(ind + 2)
                for bond in part_2.bonds:
                    if ind in bond[1:]:
                        part_2.delete_bond(bond)
                if len(part_2.bonds) == 0:
                    part_2.type = 5   # Assign type 5 to particles with all bonds broken, i.e., a dissociated monomer
                else:
                    part_2_HarmonicBond = [bond for bond in part_2.bonds if isinstance(bond[0], espressomd.interactions.HarmonicBond)]
                    if not part_2_HarmonicBond:
                        part_2.type = 4   # Assign type 4 to particles with both diag and long bonds broken

        if frame % frame_step == 0:
            pos_tic.append(pos_ic)

        # Overwrite and save the pkl file every 500 iterations
        if frame % (frame_step * 500) == 0:
            output = dict(pos_tic=np.array(pos_tic), seed=seed,
                          direction=direction, magnitude=magnitude,
                          frame_step=frame_step, frame_num=frame_num)
            print('Saving coords to pkl')
            fname_coords = op.join(folder_seed, 'coords.pkl')
            with open(fname_coords, 'wb') as f:
                pickle.dump(output, f, protocol=2)


if __name__ == '__main__':
    #thermal_fluct(10000, 5000, True, True)
    #force_constant('push', 2, 5, 10000, 3000)
    #bond_breaking('pull', 2.0, 5, 10000, 50000000)
    #force_release('push', 1.5, 5, 2250, 5000, 10000)
    #force_random_firing('pull', 1.5, 10, 10000, 1000, 0)
    #torque_bond_breaking('ccw', 200, 10000, 50000000)
    torque('ccw', 200, 10000, 50000000)
