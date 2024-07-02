import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import os.path as op
import sh
import pickle
import scipy.signal
from scipy import interpolate
from scipy import stats
import statistics
from threading import Thread
import espressomd
from espressomd import thermostat
from espressomd import integrate
from espressomd.interactions import HarmonicBond, AngleHarmonic, Dihedral
from espressomd import observables
from espressomd import visualization
from ellipse import LsqEllipse
import util
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
                                      particle_type_colors=[[0.39, 0.58, 0.93], [0.68, 0.85, 0.90]],
                                      particle_type_materials=['bright', 'bright'],
                                      #camera_position=[750, 3500, 750],
                                      camera_position=[750, 2500, 750],
                                      camera_right=[1, 0, 0],
                                      window_size=[9000, 9000],
                                      draw_box=False,
                                      draw_axis=False,
                                      )


def visualize():
    visualizer.screenshot('output/test.png')
    #visualizer.run(1)


def apply_force_on_one_strand(direction, magnitude, force_duration,
                              img_interval, same_strand=True, visual=False,
                              save_img=True, save_coords=True, fix_both=True):
    """
    Fix the end of one strand.
    Pull or push on the end of the same or the other strand.
    Visualize on the fly.
    """
    z_force = {'pull': 1, 'push': -1}[direction] * magnitude

    if fix_both:
        system.part.by_ids(range(5)).fix = [True, True, True]
    else:
        system.part.by_ids([p.id for p in system.part.select(type=0)][:5]).fix = [True, True, True]

    def main_thread():
        if fix_both:
            folder_img = 'output/squiggle/%s_duration%d_single/img' % (direction, force_duration)
            sh.mkdir('-p', folder_img)
        elif same_strand:
            folder_img = 'output/squiggle/%s_duration%d_same/img' % (direction, force_duration)
            sh.mkdir('-p', folder_img)
        else:
            folder_img = 'output/squiggle/%s_duration%d_diff/img' % (direction, force_duration)
            sh.mkdir('-p', folder_img)

        pos_tic = []
        for int_iteration in range(ap.int_iterations):
            print('run %d at time=%f' % (int_iteration, system.time))

            if int_iteration <= force_duration:
                if same_strand:
                    system.part.by_ids([p.id for p in system.part.select(type=0)][-5:]).fix = [True, True, False]
                    system.part.by_ids([p.id for p in system.part.select(type=0)][-5:]).ext_force = [0, 0, z_force]
                else:
                    system.part.by_ids([p.id for p in system.part.select(type=1)][-5:]).fix = [True, True, False]
                    system.part.by_ids([p.id for p in system.part.select(type=1)][-5:]).ext_force = [0, 0, z_force]

            else:
                if same_strand:
                    system.part.by_ids([p.id for p in system.part.select(type=0)][-5:]).fix = [False, False, False]
                    system.part.by_ids([p.id for p in system.part.select(type=0)][-5:]).ext_force = [0, 0, 0]
                else:
                    system.part.by_ids([p.id for p in system.part.select(type=1)][-5:]).fix = [False, False, False]
                    system.part.by_ids([p.id for p in system.part.select(type=1)][-5:]).ext_force = [0, 0, 0]

            system.integrator.run(ap.int_steps)

            if visual:
                visualizer.update()

            if save_img and (int_iteration % img_interval == 0):
                img_fn = op.join(folder_img, '%05d.png' % int_iteration)
                visualizer.screenshot(img_fn)

            if save_coords:
                pos_ic = observables.ParticlePositions(ids=ap.part_i)
                pos_tic.append(pos_ic.calculate())

        if save_coords:
            pos_tic = np.array(pos_tic)
            output = dict(pos_tic=pos_tic)

            print('Saving coords to pkl')
            fn_coords = op.join(op.dirname(folder_img), 'coords.pkl')
            with open(fn_coords, 'wb') as f:
                pickle.dump(output, f, protocol=2)

    assert not (visual and save_img), 'Cannot visualize and save snapshots at the same time'

    if visual:
        t = ap.Thread(target=main_thread)
        t.daemon = True
        t.start()
        visualizer.start()

    if save_img or save_coords:
        main_thread()


def apply_force_on_both_strands(direction, magnitude, force_duration,
                                img_interval, visual=False, save_img=True,
                                save_coords=True):
    """
    Fix the ends of both strands.
    Pull or push on the other ends of both strands.
    Visualize on the fly.
    """
    z_force = {'pull': 1, 'push': -1}[direction] * magnitude

    # Fix the ends of both strands
    system.part.by_ids(range(5)).fix = [True, True, True]

    def main_thread():
        folder_img = 'output/squiggle/%s_duration%d/img_zoom' % (direction, force_duration)
        sh.mkdir('-p', folder_img)

        pos_tic = []
        for int_iteration in range(ap.int_iterations):
            print('run %d at time=%f' % (int_iteration, system.time))

            if int_iteration <= force_duration:
                system.part.by_ids(list(range(ap.n_part))[-5:]).fix = [True, True, False]
                system.part.by_ids(list(range(ap.n_part))[-5:]).ext_force = [0, 0, z_force]
            else:
                system.part.by_ids(list(range(ap.n_part))[-5:]).fix = [False, False, False]
                system.part.by_ids(list(range(ap.n_part))[-5:]).ext_force = [0, 0, 0]

            system.integrator.run(ap.int_steps)

            if visual:
                visualizer.update()

            if save_img and (int_iteration % img_interval == 0):
                img_fn = op.join(folder_img, '%05d.png' % int_iteration)
                visualizer.screenshot(img_fn)

            if save_coords:
                pos_ic = observables.ParticlePositions(ids=ap.part_i)
                pos_tic.append(pos_ic.calculate())

        if save_coords:
            pos_tic = np.array(pos_tic)
            output = dict(pos_tic=pos_tic)

            print('Saving coords to pkl')
            fn_coords = op.join(op.dirname(folder_img), 'coords.pkl')
            with open(fn_coords, 'wb') as f:
                pickle.dump(output, f, protocol=2)

    assert not (visual and save_img), 'Cannot visualize and save snapshots at the same time'

    if visual:
        t = ap.Thread(target=main_thread)
        t.daemon = True
        t.start()
        visualizer.start()

    if save_img or save_coords:
        main_thread()


def plucking_both_strands(direction, magnitude, force_interval, relax_interval,
                          img_interval, visual=False, save_img=True,
                          save_coords=True):
    """
    Fix the ends of both strands.
    Pull or push on the other ends of both strands intermittently.
    Visualize on the fly.
    """
    z_force = {'pull': 1, 'push': -1}[direction] * magnitude

    # Fix the ends of both strands
    system.part.by_ids(range(5)).fix = [True, True, True]

    def main_thread():
        folder_img = 'output/squiggle/plucking_both_strands/%s%d_relax%d/img' % (direction, force_interval, relax_interval)
        sh.mkdir('-p', folder_img)

        pos_tic = []
        for int_iteration in range(ap.int_iterations):
            print('run %d at time=%f' % (int_iteration, system.time))

            if int_iteration % (force_interval + relax_interval) <= force_interval:
                system.part.by_ids(list(range(ap.n_part))[-5:]).fix = [True, True, False]
                system.part.by_ids(list(range(ap.n_part))[-5:]).ext_force = [0, 0, z_force]
            else:
                system.part.by_ids(list(range(ap.n_part))[-5:]).fix = [False, False, False]
                system.part.by_ids(list(range(ap.n_part))[-5:]).ext_force = [0, 0, 0]

            system.integrator.run(ap.int_steps)

            if visual:
                visualizer.update()

            if save_img and (int_iteration % img_interval == 0):
                img_fn = op.join(folder_img, '%05d.png' % int_iteration)
                visualizer.screenshot(img_fn)

            if save_coords:
                pos_ic = observables.ParticlePositions(ids=ap.part_i)
                pos_tic.append(pos_ic.calculate())

        if save_coords:
            pos_tic = np.array(pos_tic)
            output = dict(pos_tic=pos_tic)

            print('Saving coords to pkl')
            fn_coords = op.join(op.dirname(folder_img), 'coords.pkl')
            with open(fn_coords, 'wb') as f:
                pickle.dump(output, f, protocol=2)

    assert not (visual and save_img), 'Cannot visualize and save snapshots at the same time'

    if visual:
        t = ap.Thread(target=main_thread)
        t.daemon = True
        t.start()
        visualizer.start()

    if save_img or save_coords:
        main_thread()


def plucking_one_strand(direction, magnitude, force_interval, relax_interval,
                        img_interval, visual=False, save_img=True,
                        save_coords=True):
    """
    Fix the ends of both strands.
    Pull or push on one strand intermittently.
    Visualize on the fly.
    """
    z_force = {'pull': 1, 'push': -1}[direction] * magnitude

    # Fix the ends of both strands
    system.part.by_ids(range(5)).fix = [True, True, True]

    def main_thread():
        folder_img = 'output/squiggle/plucking_one_strand/%s%d_relax%d/img' % (direction, force_interval, relax_interval)
        sh.mkdir('-p', folder_img)

        pos_tic = []
        for int_iteration in range(ap.int_iterations):
            print('run %d at time=%f' % (int_iteration, system.time))

            if int_iteration % (force_interval + relax_interval) <= force_interval:
                system.part.by_ids([p.id for p in system.part.select(type=0)][-5:]).fix = [True, True, False]
                system.part.by_ids([p.id for p in system.part.select(type=0)][-5:]).ext_force = [0, 0, z_force]
            else:
                system.part.by_ids([p.id for p in system.part.select(type=0)][-5:]).fix = [False, False, False]
                system.part.by_ids([p.id for p in system.part.select(type=0)][-5:]).ext_force = [0, 0, 0]

            system.integrator.run(ap.int_steps)

            if visual:
                visualizer.update()

            if save_img and (int_iteration % img_interval == 0):
                img_fn = op.join(folder_img, '%05d.png' % int_iteration)
                visualizer.screenshot(img_fn)

            if save_coords:
                pos_ic = observables.ParticlePositions(ids=ap.part_i)
                pos_tic.append(pos_ic.calculate())

        if save_coords:
            pos_tic = np.array(pos_tic)
            output = dict(pos_tic=pos_tic)

            print('Saving coords to pkl')
            fn_coords = op.join(op.dirname(folder_img), 'coords.pkl')
            with open(fn_coords, 'wb') as f:
                pickle.dump(output, f, protocol=2)

    assert not (visual and save_img), 'Cannot visualize and save snapshots at the same time'

    if visual:
        t = ap.Thread(target=main_thread)
        t.daemon = True
        t.start()
        visualizer.start()

    if save_img or save_coords:
        main_thread()


def pair_cov(offset_X3, particle_size):
    """
    Compute the Gaussian pair covariance function.
    """
    prefactor = 1 / (2 * np.sqrt(np.pi) * particle_size) ** 3
    return prefactor * np.exp(
        -(offset_X3 ** 2).sum(axis=-1) / (4 * particle_size ** 2)
    )


def save_period_amplitude(fname_coords, particle_size=2, t_start=100, t_end=20000):
    with open(fname_coords, 'rb') as f:
        pkl = pickle.load(f)
    pos_tic = pkl['pos_tic']
    print(len(pos_tic))

    shift_k = np.linspace(0, 1200, 500)

    amp_max_t = []
    period_t = []
    for t, pos_ic in enumerate(pos_tic[t_start:t_end:10]):
        pos_ic -= pos_ic.mean(axis=0)
        cov_cc = np.dot(pos_ic.T, pos_ic) / len(pos_ic)
        w_C, v_cC = np.linalg.eigh(cov_cc)
        pc1_c = v_cC[:, -1]

        normed_pc1_c = pc1_c / np.linalg.norm(pc1_c)

        amp_i = []
        for pos_c in pos_ic:
            proj_c = pos_c - np.dot(normed_pc1_c, pos_c) * normed_pc1_c
            amp = np.linalg.norm(proj_c)
            amp_i.append(amp)
        amp_max = max(amp_i)
        amp_max_t.append(amp_max)

        # Scan shifts along the principal component direction.
        shift_k3 = shift_k[:, None] * normed_pc1_c

        # Compute the offsets, i.e., "q_i - q_j + shift" values.
        offset_kii3 = pos_ic[:, None, :] - pos_ic + shift_k3[:, None, None, :]

        # Compute the autocovariance contribution from each particle pair.
        pair_cov_kii = pair_cov(offset_kii3, particle_size)

        # Compute autocovariance by summing over all pair contributions.
        autocov_k = pair_cov_kii.sum(axis=(1, 2))

        # Convert to autocorrelation.
        assert shift_k[0] == 0, shift_k[0]
        autocorr_k = autocov_k / autocov_k[0]
        k_ixs, _ = scipy.signal.find_peaks(autocorr_k)
        period = shift_k[k_ixs[0]]
        period_t.append(period)

        print('t: %05d, period: %.2f, amplitude: %.2f' % (t, period, amp_max))

    output = dict(period_t=period_t, amp_max_t=amp_max_t)
    print('Saving periods to pkl')
    fname_output = op.join(op.dirname(fname_coords), 'period_amplitude.pkl')
    with open(fname_output, 'wb') as f:
        pickle.dump(output, f, protocol=2)


def plot_period_hist(fname_pull, fname_push, amp_cutoff=15):
    print(fname_pull)
    with open(fname_pull, 'rb') as f:
        pkl_pull = pickle.load(f)
    pull_period_t = pkl_pull['period_t']
    pull_amp_max_t = pkl_pull['amp_max_t']

    pull_period_T = []
    for period, amp_max in zip (pull_period_t, pull_amp_max_t):
        if amp_max >= amp_cutoff:
            if period >= 75 and period <=350:
                pull_period_T.append(period)

    mean = sum(pull_period_T) / len(pull_period_T)
    print('mean: %.2f' % mean)
    median = statistics.median(pull_period_T)
    print('median: %.2f' % median)

    print(fname_push)
    with open(fname_push, 'rb') as f:
        pkl_push = pickle.load(f)
    push_period_t = pkl_push['period_t']
    push_amp_max_t = pkl_push['amp_max_t']

    push_period_T = []
    for period, amp_max in zip (push_period_t, push_amp_max_t):
        if amp_max >= amp_cutoff:
            if period >= 75 and period <=350:
                push_period_T.append(period)

    mean = sum(push_period_T) / len(push_period_T)
    print('mean: %.2f' % mean)
    median = statistics.median(push_period_T)
    print('median: %.2f' % median)

    t_stat, p_value = stats.ttest_ind(pull_period_T, push_period_T)
    print('p = %g' % p_value)

    # if p_value < 0.0001:
    #     text = 'P < 0.0001'
    # elif p_value > 0.05:
    #     text = 'P = %.4f' % p_value

    output_fig = 'output/fig'
    sh.mkdir('-p', output_fig)
    pull_duration = op.dirname(fname_pull).split('/')[-1].split('_')[1]
    push_duration = op.dirname(fname_push).split('/')[-1].split('_')[1]
    assert pull_duration == push_duration
    if op.dirname(fname_pull).split('/')[-1].endswith('single') and op.dirname(fname_push).split('/')[-1].endswith('single'):
        fname_hist = op.join(output_fig, 'hist_%s_single.pdf' % pull_duration)
    else:
        fname_hist = op.join(output_fig, 'hist_%s.pdf' % pull_duration)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.hist((pull_period_T, push_period_T), bins=10, density=True,
            label=('Pull', 'Push'), color=('#7952a0', '#5bdc76'), edgecolor='black')
    ax.set_xlim([75, 350])
    ax.set_ylim([0, 0.0082])
    ax.legend(loc='upper right', fontsize='small')
    # ax.text(260, 0.006, text, transform=ax.transAxes)
    ax.set_xlabel('Period / nm')
    ax.set_ylabel('Probability density')
    plt.tight_layout()
    plt.savefig(fname_hist)
    plt.close()


def projection(fname_coords, t_start=0, t_end=-1):
    print('Loading coordinates: %s' % fname_coords)
    with open(fname_coords, 'rb') as f:
        pkl = pickle.load(f)
    pos_tic = pkl['pos_tic'][t_start:t_end]

    print('Finding central axis')
    spl_tjc = []
    for pos_ic in pos_tic:
        # mid is the mid point between adjacent particles and is indexed by j.
        # j = n_part - 1
        mid_jc = [(pos_c + pos_ic[i - 1]) / 2 for i, pos_c in enumerate(pos_ic) if i != 0]
        mid_jc = np.array(mid_jc)
        tck, u = interpolate.splprep(mid_jc.T, s=5)
        splx_j, sply_j, splz_j= interpolate.splev(u, tck)
        spl_jc = np.stack((splx_j, sply_j, splz_j), axis=1)
        spl_tjc.append(spl_jc)
    spl_tjc = np.array(spl_tjc)

    print('Projecting central axis spline to pricipal component vectors')
    proj_pc12_tjc = []
    proj_pc13_tjc = []
    proj_pc23_tjc = []
    for spl_jc in spl_tjc:
        spl_jc -= spl_jc.mean(axis=0)
        cov_cc = np.dot(spl_jc.T, spl_jc) / len(spl_jc)
        w_C, v_cC = np.linalg.eigh(cov_cc)
        pc1_c = v_cC[:, -1]
        pc2_c = v_cC[:, -2]
        pc3_c = v_cC[:, -3]

        normed_pc1_c = pc1_c / np.linalg.norm(pc1_c)
        normed_pc2_c = pc2_c / np.linalg.norm(pc2_c)
        normed_pc3_c = pc3_c / np.linalg.norm(pc3_c)

        proj_pc12_jc = []
        proj_pc13_jc = []
        proj_pc23_jc = []
        for spl_c in spl_jc:
            proj_pc12_c = spl_c - np.dot(normed_pc3_c, spl_c) * normed_pc3_c
            proj_pc13_c = spl_c - np.dot(normed_pc2_c, spl_c) * normed_pc2_c
            proj_pc23_c = spl_c - np.dot(normed_pc1_c, spl_c) * normed_pc1_c
            proj_pc12_jc.append(proj_pc12_c)
            proj_pc13_jc.append(proj_pc13_c)
            proj_pc23_jc.append(proj_pc23_c)

        proj_pc12_tjc.append(proj_pc12_jc)
        proj_pc13_tjc.append(proj_pc13_jc)
        proj_pc23_tjc.append(proj_pc23_jc)

    proj_pc12_tjc = np.array(proj_pc12_tjc)
    proj_pc13_tjc = np.array(proj_pc13_tjc)
    proj_pc23_tjc = np.array(proj_pc23_tjc)

    output_spl = dict(spl_tjc=spl_tjc)
    print('Saving central axis spline to pkl')
    fname_output_spl = op.join(op.dirname(fname_coords), 'spline.pkl')
    with open(fname_output_spl, 'wb') as f:
        pickle.dump(output_spl, f, protocol=2)

    output_proj = dict(proj_pc12_tjc=proj_pc12_tjc,
                       proj_pc13_tjc=proj_pc13_tjc,
                       proj_pc23_tjc=proj_pc23_tjc)
    print('Saving projections to pkl')
    fname_output_proj = op.join(op.dirname(fname_coords), 'projections.pkl')
    with open(fname_output_proj, 'wb') as f:
        pickle.dump(output_proj, f, protocol=2)


def plot_projection(fname_proj, t_indices, viridis=True):
    def color_fader(c1, c2, mix=0):
        c1=np.array(mpl.colors.to_rgb(c1))
        c2=np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    with open(fname_proj, 'rb') as f:
        pkl = pickle.load(f)
    proj_pc12_tjc = pkl['proj_pc12_tjc']
    proj_pc13_tjc = pkl['proj_pc13_tjc']
    proj_pc23_tjc = pkl['proj_pc23_tjc']

    if viridis:
        cmap = mpl.cm.get_cmap('viridis')

        fname_proj_pc12 = op.join(op.dirname(fname_proj), 'proj_pc12.pdf')
        fig, ax = plt.subplots(1, 1, figsize=(1.6, 4.8))
        for i, t_index in enumerate(t_indices):
            ax.plot(proj_pc12_tjc[t_index, :, 1], proj_pc12_tjc[t_index, :, 2],
                    c=cmap(i/(len(t_indices)-1)))
        ax.set_xlim([-80, 80])
        ax.set_ylim([-600, 600])
        ax.set_xlabel('PC2 / nm')
        ax.set_ylabel('PC1 / nm')
        plt.tight_layout()
        plt.savefig(fname_proj_pc12)
        plt.close()

        fname_proj_pc13 = op.join(op.dirname(fname_proj), 'proj_pc13.pdf')
        fig, ax = plt.subplots(1, 1, figsize=(1.6, 4.8))
        for i, t_index in enumerate(t_indices):
            ax.plot(proj_pc13_tjc[t_index, :, 0], proj_pc13_tjc[t_index, :, 2],
                    c=cmap(i/(len(t_indices)-1)))
        ax.set_xlim([-80, 80])
        ax.set_ylim([-600, 600])
        ax.set_xlabel('PC3 / nm')
        ax.set_ylabel('PC1 / nm')
        plt.tight_layout()
        plt.savefig(fname_proj_pc13)
        plt.close()

        fname_proj_pc23 = op.join(op.dirname(fname_proj), 'proj_pc23.pdf')
        fig, ax = plt.subplots(1, 1, figsize=(1.6, 1.6))
        for i, t_index in enumerate(t_indices):
            ax.plot(proj_pc23_tjc[t_index, :, 0], proj_pc23_tjc[t_index, :, 1],
                    c=cmap(i/(len(t_indices)-1)))
        ax.set_xlim([-65, 65])
        ax.set_ylim([-65, 65])
        ax.set_xlabel('PC3 / nm')
        ax.set_ylabel('PC2 / nm')
        plt.tight_layout()
        plt.savefig(fname_proj_pc23)
        plt.close()

    else:
        if op.dirname(fname_proj).split('/')[-1].startswith('push'):
            c1 = '#efe9f4'
            c2 = '#835bab'
        elif op.dirname(fname_proj).split('/')[-1].startswith('pull'):
            c1 = '#eef8f1'
            c2 = '#5dbc76'

        fname_proj_pc12 = op.join(op.dirname(fname_proj), 'proj_pc12.png')
        fig, ax = plt.subplots(1, 1, figsize=(3, 12))
        for i, t_index in enumerate(t_indices):
            ax.plot(proj_pc12_tjc[t_index, :, 1], proj_pc12_tjc[t_index, :, 2],
                    c=color_fader(c1, c2, mix=i/(len(t_indices)-1)))
        ax.set_xlim([-80, 80])
        ax.set_ylim([-600, 600])
        ax.set_xlabel('PC2')
        ax.set_ylabel('PC1')
        plt.tight_layout()
        plt.savefig(fname_proj_pc12)
        plt.close()

        fname_proj_pc13 = op.join(op.dirname(fname_proj), 'proj_pc13.png')
        fig, ax = plt.subplots(1, 1, figsize=(3, 12))
        for i, t_index in enumerate(t_indices):
            ax.plot(proj_pc13_tjc[t_index, :, 0], proj_pc13_tjc[t_index, :, 2],
                    c=color_fader(c1, c2, mix=i/(len(t_indices)-1)))
        ax.set_xlim([-80, 80])
        ax.set_ylim([-600, 600])
        ax.set_xlabel('PC3')
        ax.set_ylabel('PC1')
        plt.tight_layout()
        plt.savefig(fname_proj_pc13)
        plt.close()

        fname_proj_pc23 = op.join(op.dirname(fname_proj), 'proj_pc23.png')
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        for i, t_index in enumerate(t_indices):
            ax.plot(proj_pc23_tjc[t_index, :, 0], proj_pc23_tjc[t_index, :, 1],
                    c=color_fader(c1, c2, mix=i/(len(t_indices)-1)))
        ax.set_xlim([-65, 65])
        ax.set_ylim([-65, 65])
        ax.set_xlabel('PC3')
        ax.set_ylabel('PC2')
        plt.tight_layout()
        plt.savefig(fname_proj_pc23)
        plt.close()


def compute_rise_twist(fn_coords, folder_rise, folder_twist, folder_mid1, folder_mid2):
    with open(fn_coords, 'rb') as f:
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
    rise_norm_th = []
    for rise_vector_hc in rise_vector_thc:
        rise_norm_h = [np.linalg.norm(rise_vector_c) for rise_vector_c in rise_vector_hc]
        rise_norm_th.append(rise_norm_h)
    rise_norm_th = np.array(rise_norm_th)

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

    # Plot the rise of each strand
    sh.mkdir('-p', folder_rise)

    print('Plotting and saving rise')

    for (t, rise_norm_h) in enumerate(rise_norm_th):
        if t % 30 == 0:
            fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))
            ax.plot(range(len(rise_norm_th[t]))[0::2], rise_norm_th[t, 0::2], label='Strand 1')
            ax.plot(range(len(rise_norm_th[t]))[1::2], rise_norm_th[t, 1::2], label='Strand 2')
            ax.legend(loc='lower right', fontsize='small')
            ax.set_ylim(2.5, 3.0)
            ax.set_xlabel('Particle index')
            ax.set_ylabel('Rise / nm')
            plt.tight_layout()
            fn_rise = op.join(folder_rise, 'rise-%04d.png' % t)
            plt.savefig(fn_rise, dpi=300)
            plt.close()

    # Plot the twist of each strand
    sh.mkdir('-p', folder_twist)

    print('Plotting and saving twist')

    for (t, twist_h) in enumerate(twist_th):
        if t % 30 == 0:
            fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))
            ax.plot(range(len(twist_th[t]))[0::2], twist_th[t, 0::2], label='Strand 1')
            ax.plot(range(len(twist_th[t]))[1::2], twist_th[t, 1::2], label='Strand 2')
            ax.legend(loc='lower right', fontsize='small')
            ax.set_ylim(-190, -145)
            ax.set_xlabel('Particle index')
            ax.set_ylabel('Twist / Degree')
            plt.tight_layout()
            fn_twist = op.join(folder_twist, 'twist-%04d.png' % t)
            plt.savefig(fn_twist, dpi=300)
            plt.close()

    #from ptpython import embed
    #embed(globals(), locals())
    #return

    # Plot mid1 center axis as a function of z.
    sh.mkdir('-p', folder_mid1)

    print('Plotting and saving mid1')

    for (t, mid1_jc) in enumerate(mid1_tjc):
        if t % 30 == 0:
            fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
            ax.plot(mid1_tjc[t, :, 0], mid1_tjc[t, :, 2], label='mid1, t = %04d' % t)
            ax.legend(loc='lower left', fontsize='small')
            ax.set_xlabel('x / nm')
            ax.set_ylabel('z / nm')
            ax.set_xlim(475, 575)
            ax.set_ylim(500, 600)
            plt.tight_layout()
            fn_mid1 = op.join(folder_mid1, 'mid1-%04d.png' % t)
            plt.savefig(fn_mid1, dpi=300)
            plt.close()

    # Plot mid2 center axis as a function of z.
    sh.mkdir('-p', folder_mid2)

    print('Plotting and saving mid2')

    for (t, mid2_kc) in enumerate(mid2_tkc):
        if t % 30 == 0:
            fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
            ax.plot(mid2_tkc[t, :, 0], mid2_tkc[t, :, 2], label='mid2, t = %04d' % t)
            ax.legend(loc='lower left', fontsize='small')
            ax.set_xlabel('x / nm')
            ax.set_ylabel('z / nm')
            ax.set_xlim(475, 575)
            ax.set_ylim(500, 600)
            plt.tight_layout()
            fn_mid2 = op.join(folder_mid2, 'mid2-%04d.png' % t)
            plt.savefig(fn_mid2, dpi=300)
            plt.close()


if __name__ == '__main__':
    apply_force_on_one_strand('pull', 5, 1500, 50)
    apply_force_on_both_strands('push', 5, 5000, 50, save_coords=False)
    plucking_both_strands('push', 5, 250, 5000, 100)
    plucking_one_strand('pull', 5, 100, 3000, 100)
