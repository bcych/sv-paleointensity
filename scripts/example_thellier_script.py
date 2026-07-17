from sdcc.treatment import thellier_experiment, HoldStep
from sdcc.particles import load_particle
from sdcc.simulation import parallelized_mono_dispersion, result_to_file
import pickle
import numpy as np
import os

##PARTICLE PARAMETERS##
TMx = "00"  # TM30
PRO = "1.44"  # Long/intermediate axis ratio
OBL = "1.00"  # Intermediate/short axis ratio
alignment = "hard"  # Align <1 1 1> direction to long axis
size = 100  # nm

##PARTICLE FILE##
particle_file = f"../smelt_files/Ellipsoid_TM{TMx}_PRO_{PRO}_OBL_{OBL}_{str(size).zfill(3)}nm_{alignment}.smelt"
particle = load_particle(particle_file)

##EXPERIMENT PARAMETERS##
if TMx == "00":
    temp_steps = [20, 100, 200, 300, 350, 400, 425, 450, 475, 500, 520, 540, 560, 579]
elif TMx == "30":
    temp_steps = [20, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 403]
else:
    raise ValueError("Wrong TM composition!")

B_anc_dir = [1, 0, 0]
B_lab_dir = [0, 1, 0]
B_anc = 40
B_lab = 40

##OUTPUT##
file_ext = ".thel"
output_dir = "../data/"

if __name__ == "__main__":
    # Run the Thellier Experiment
    routine = thellier_experiment(
        temp_steps,
        B_anc,
        B_lab,
        B_anc_dir,
        B_lab_dir,
        type="izzi",
        ptrm_checks=1,
    )

    # Hold for 1 Gyr - can remove this step for P1 and P2 experiments.
    hold = HoldStep(
        routine[0].ts[-1] + 1e-12, 20, B_anc, np.array([0, 0, 1]), hold_time=3.154e16
    )

    # Add and subtract time
    subtr_time = routine[1].ts[0]
    add_time = hold.ts[-1] + 1e-12
    for i in range(1, len(routine)):
        routine[i].ts -= subtr_time
        routine[i].ts += add_time

    # Insert VRM between initial cooling and heating

    routine = [routine[0]] + [hold] + routine[1:]
    # Set up initial state probability vector for particle
    num_states = len(particle.get_params(20)["min_e"])
    start_p = np.full(num_states, 1 / num_states)

    # Run experiment.
    moments, probs = parallelized_mono_dispersion(
        start_p, size, routine, particle, n_dirs=30, ctx="spawn"
    )

    # Save result to file.
    result_to_file(
        particle,
        size,
        routine,
        moments,
        probs,
        file_ext="thel",
        directory="../data/example.thel",
    )
