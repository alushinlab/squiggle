This repository contains tools to simulate a single actin filament under force using the simulation package ESPResSo and analyze actin filament configuration.

`actin_potential.py` sets up the simulation system and defines harmonic bond, harmonic angle, and dihedral potentials.

`util.py` calculates bond angles and dihedral angles.

`param_scan.py` performs parameter scan to optimize the energy constants.

`sim.py` simulates a single actin filament under tension or compression, and analyzes the force-evoked configuration.

