#This program will run a grid of models using the simulation code and save the results.
#Plotting will be done using a separate script, so you don't have to rerun all the damn
#simulations just to look at the data...

#This version is for running a bunch of temperatures so I can interpolate the results
#and find the maximum allowable instrument temperature for each aperture and line.

import numpy as np
from shutil import move
from os import remove
import hci_hrs_sim

#The parameter "grid".  I'm sure there's a more elegant\Pythonic way
#to do it, but this'll work.

texp_grid = [36e4, 144e4]
temperature_grid = np.linspace(100, 400, num=31)
aperture_grid = [1., 2.4, 4., 6.5, 9., 12., 15.]

#The simulation loop.  Order of operations is as follows:
#1.  Rewrite the init file using the parameters for that spot on the grid.
#2.  Run the simulation code using those parameters.
#3.  Read in multi_sim_log.dat to get the CCF SNR and CCF SNR standard deviation.
#4.  Add the changed parameters and corresponding CCF SNRs as a row of an array.
#5.  Save the array to a file.

param_arr = []

counter = 0

for aper in aperture_grid:
    for texp in texp_grid:
        for T in temperature_grid:
            
            #Rewrite the init file with the parameters of the day

            with open("SunEarth_4m.init.new", "w") as newinitfile:
                with open("SunEarth_4m.init", "r") as initfile:
                    for line in initfile:
                        if "telescope_size" in line:
                            line = "telescope_size:\t" + str(aper) + "\t# in m\n"
                        if "t_exp" in line:
                            line = "t_exp:\t" + str(texp) + "\t# in second\n"
                        if "temperature" in line:
                            line = "temperature:\t" +str(T) + "\t#temperature of the telescope in K\n"
                        newinitfile.write(line)
            remove("SunEarth_4m.init")
            move("SunEarth_4m.init.new", "SunEarth_4m.init")

            #Run the simulation with the parameters of the day

            remove("multi_sim_log.dat")
            hci_hrs_sim.__main__()

            #Read in multi_sim_log.dat, get the correct columns, and then save them along 
            #with the PotD to an array

            sim_results = np.genfromtxt("multi_sim_log.dat", dtype=str, delimiter=",")
            param_arr.append([aper, texp, T, float(sim_results[6])])
            counter+=1
            print counter

#Save param_arr to a file for later use

np.savetxt("temperature_grid_results.dat", param_arr)
