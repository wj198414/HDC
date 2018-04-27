#This program will run a grid of models using the simulation code and save the results.
#Plotting will be done using a separate script, so you don't have to rerun all the damn
#simulations just to look at the data...

#This version is for running a bunch of exozodi levels so I can interpolate the results
#and find the maximum allowable exozodi contamination for each aperture and line.

import numpy as np
from shutil import move
from os import remove
import hci_hrs_sim

#The parameter "grid".  I'm sure there's a more elegant\Pythonic way
#to do it, but this'll work.

texp_grid = [36e4, 144e4]
exozodi_grid = np.linspace(0, 60, num=31)
aperture_grid = [1., 2.4, 4., 6.5, 9., 12., 15.]

#The simulation loop.  Order of operations is as follows:
#1.  Rewrite the init file using the parameters for that spot on the grid.
#2.  Run the simulation code using those parameters.
#3.  Read in multi_sim_log.dat to get the CCF SNR and CCF SNR standard deviation.
#4.  Add the changed parameters and corresponding CCF SNRs as a row of an array.
#5.  Save the array to a file.

param_arr = []

for aper in aperture_grid:
    for texp in texp_grid:
        for Z in exozodi_grid:
            
            #Rewrite the init file with the parameters of the day

            with open("SunEarth_4m.init.new", "w") as newinitfile:
                with open("SunEarth_4m.init", "r") as initfile:
                    for line in initfile:
                        if "telescope_size" in line:
                            line = "telescope_size:\t" + str(aper) + "\t# in m\n"
                        if "t_exp" in line:
                            line = "t_exp:\t" + str(texp) + "\t# in second\n"
                        if "exozodi_level" in line:
                            line = "exozodi_level:\t" +str(Z) + "\t#level of exozodi relative to the Solar System\n"
                        newinitfile.write(line)
            remove("SunEarth_4m.init")
            move("SunEarth_4m.init.new", "SunEarth_4m.init")

            #Run the simulation with the parameters of the day

            remove("multi_sim_log.dat")
            hci_hrs_sim.__main__()

            #Read in multi_sim_log.dat, get the correct columns, and then save them along 
            #with the PotD to an array

            sim_results = np.genfromtxt("multi_sim_log.dat", dtype=str, delimiter=",")
            param_arr.append([aper, texp, Z, float(sim_results[6])])

#Save param_arr to a file for later use

np.savetxt("exozodi_grid_results.dat", param_arr)