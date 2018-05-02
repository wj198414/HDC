#This program will run a grid of models using the simulation code and save the results.
#Plotting will be done using a separate script, so you don't have to rerun all the damn
#simulations just to look at the data...

import numpy as np
from shutil import move
from os import remove
import hci_hrs_sim

#The parameter "grid".  I'm sure there's a more elegant\Pythonic way
#to do it, but this'll work.

spec_reso_grid = np.array([200, 400, 800, 1600, 3200, 6400, 12800, 25600])
texp_grid = np.array([3.6e5, 1.44e6])
contrast_grid = np.logspace(-11.0, -8.0, num=19)
zodi_grid = np.array([1., 5., 10., 20.])
aperture_grid = np.array([6.5])

#The simulation loop.  Order of operations is as follows:
#1.  Rewrite the init file using the parameters for that spot on the grid.
#2.  Run the simulation code using those parameters.
#3.  Read in multi_sim_log.dat to get the CCF SNR and CCF SNR standard deviation.
#4.  Add the changed parameters and corresponding CCF SNRs as a row of an array.
#5.  Save the array to a file.

param_arr = []
counter=0

#this change doesn't matter

for R in spec_reso_grid:
    for texp in texp_grid:
        for C in contrast_grid:
            for Z in zodi_grid:
                for aper in aperture_grid:
                
                #Rewrite the init file with the parameters of the day

                    with open("SunEarth_4m.init.new", "w") as newinitfile:
                        with open("SunEarth_4m.init", "r") as initfile:
                            for line in initfile:
                                if "spec_reso" in line:
                                    line = "spec_reso:\t" + str(R) + "\t# spectral resolution\n"
                                if "t_exp" in line:
                                    line = "t_exp:\t" + str(texp) + "\t# in second\n"
                                if "pl_st_contrast" in line:
                                    line = "pl_st_contrast:\t" + str(C) + "\t# star light suppression at fiber position\n"
                                if "exozodi_level" in line:
                                    line = "exozodi_level:\t" + str(Z) + "\t# level of exozodi relative to the Solar System\n"
                                if "telescope_size" in line:
                                    line = "telescope_size:\t" + str(aper) + "\t# in m\n"
                                newinitfile.write(line)
                    remove("SunEarth_4m.init")
                    move("SunEarth_4m.init.new", "SunEarth_4m.init")

                    #Run the simulation with the parameters of the day

                    remove("multi_sim_log.dat")
                    hci_hrs_sim.__main__()

                    #Read in multi_sim_log.dat, get the correct columns, and then save them along 
                    #with the PotD to an array

                    sim_results = np.genfromtxt("multi_sim_log.dat", dtype=str, delimiter=",")
                    param_arr.append([R, texp, C, Z, float(sim_results[6])])
                    counter+=1
                    print counter

#Save param_arr to a file for later use

np.savetxt("sim_grid_results.dat", param_arr)
