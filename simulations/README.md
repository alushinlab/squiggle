This repository contains tools to simulate a single actin filament under force using the simulation package ESPResSo and analyze actin filament configuration.

`param_scan.py` performs parameter scan to optimize the force field.

`actin_potential.py` sets up the simulation system and defines harmonic bond, harmonic angle, and dihedral potentials using the optimized force field.

`util.py` calculates bond angles and dihedral angles.

`run_sim.py` simulates a single actin filament under thermal fluctuation, tension, or compression.

`analysis.py` analyzes force-evoked actin configurations.

`helical_params_short.py` simulates a single actin filament with a range of helical rise and twist under constant compression.

`flexural_rigidity_short.py` simulates a single actin filament with a range of flexural rigidity under constant compression.

`fig.py` contains functions to plot the figures in the paper.
