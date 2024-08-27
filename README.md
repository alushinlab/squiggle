# Squiggle
This repository is a collection of scripts used in the project "Myosin forces elicit an F-actin structural landscape that modulates mechanosensitive protein recognition," which has been pre-printed on bioRXiv here: https://www.biorxiv.org/content/10.1101/2024.08.15.608188v1

The scripts used to process cryo-EM and cryo-ET data of F-actin subjected to myosin motor activity may be found in "in_vitro_tomo" and "single_particle". These scripts were developed by Matthew Reynolds with assistance from Ayala Carl.

The scripts used in coarse-grained molecular dynamics simulations of these systems may be found in "simulations". These scripts were developed by Xiaoyu Sun.

## A note on running scripts
To ensure all necessary python packages are available when developing these scripts, an anaconda environment called matt_EMAN2 was used. If you would like to use the scripts provided in this repository, it is recommended that you make an anaconda environment using the provided yml file. You will also need to update the top line of the script to your installation of the anaconda environment. You may also use your own anaconda environments provided it has the necessary packages installed. This software was all run on Linux operating systems, but it should in principle run on any system with minimal if any OS-specific adjustments.
