import numpy as np
import espressomd
from espressomd import thermostat
from espressomd.interactions import HarmonicBond, AngleHarmonic, Dihedral
import util

box_l = [500, 500, 1500]
part_num = 400
kT = 1.0
gamma = 123
seed = 2

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
system.time_step = 0.01  # time unit 0.1 ns
system.cell_system.skin = 0.01
system.thermostat.set_langevin(kT=kT, gamma=gamma, seed=seed)

# Set up particles
# Use type to specify particles on each strand
system.part.add(id=ind_i[::2], pos=pos_i[::2], type=np.zeros_like(ind_i[::2]))
system.part.add(id=ind_i[1::2], pos=pos_i[1::2], type=np.ones_like(ind_i[1::2]))

# Set harmonic bond potentials
# Use optimum tensile params
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
# Use optimum bending params
k_bending = 145
bending_factor = 1

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
# Use optimum dihedral params
k_twisting = 55
twisting_factor = 20

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
