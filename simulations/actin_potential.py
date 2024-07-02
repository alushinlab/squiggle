import numpy as np
import os.path as op
import sh
import pickle
from threading import Thread
import espressomd
from espressomd import thermostat
from espressomd import integrate
from espressomd.interactions import HarmonicBond, AngleHarmonic, Dihedral
from espressomd import observables

import util

required_features = ["EXTERNAL_FORCES", "MASS"]
espressomd.assert_features(required_features)

box_l = 3 * [1500]  # nm
n_part = 400

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
system.time_step = 0.1  # time scale unit is ps
system.cell_system.skin = 0.01  # nm

# kT=2.44 at 20 celsius and 293.15 kelvin
# Use kT=0, gamma=10 to scan spring constants for harmonic bond potentials in a stretched actin
# Use kT=2.44, gamma=25151 pN*ps/nm to scan spring constants for harmonic angle potentials and dihedral potentials in an actin undergoing thermal fluctuation
system.thermostat.set_langevin(kT=0, gamma=0, seed=42)

# Set up particles
pos_i = np.stack((x_i, y_i, z_i), axis=1)
mass_i = 42051 * np.ones_like(part_i)
system.part.add(id=part_i[::2], pos=pos_i[::2], mass=mass_i[::2], type=np.zeros_like(part_i[::2]))
system.part.add(id=part_i[1::2], pos=pos_i[1::2], mass=mass_i[1::2], type=np.ones_like(part_i[1::2]))

# Set up integration
int_steps = 100
int_iterations = 20000

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


# Set harmonic angle potentials
# Treat particle2 as the center particle
k_bending = 300
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

# Set dihedral potentials
# Dihedral angle with the connection between particles 1&2 as the edge
# This dihedral is the main regulator of the twisting/untwisting of F-actin
k_twisting = 150
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
