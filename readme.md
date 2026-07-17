# Overview

Data and code repository for the paper "Stable paleomagnetic records from single vortex particles". Reproduces Figures 2-4 from the paper. Examples of scripts used to generate data are found in the scripts folder.

# Figures

### Figure 2
Creates subplots showing the relaxation time and blocking temperature for spheres of titanomagnetite with different sizes. Data used are in data/Tb_tr.csv.

### Figure 3
Reproduces figure 3 showing the blocking temperature for elongate titanomagnetite particles with different sizes. Data used are in data/Tb_tr.csv, with more compositions possible to plot.

### Figure 4 
Reproduces Figure 4, plotting Arai and Zijderveld plots, as well as particle distributions. Uses data/TM30_particle_distribution.csv and data/TM30_P1.csv to reproduce the figure. These can easily be changed to look at other particle distributions. 

# Scripts

### calculate_relaxation_time.py 
Calculates the relaxation time and blocking temperature for a particle. Uses the "SMELT" files available in smelt_files/ with the sdcc to calculate these.

### example_thellier_script.py
Runs a simulation of a Thellier experiment for a single particle. Outputs the result to the data/ folder (warning, result is large).

# SMELT files
Available in smelt_files/ . Allows users to simulate arbitrary paleomagnetic routines using these particles with the SDCC. 

For any queries, contact me at bcych@umontpellier.fr


