#This program is for quickly debugging the problem causing the
#partial flattening of the thermal background spectrum.

import numpy as np
import matplotlib.pyplot as mpl
import astropy.io.fits as pyfits
import scipy.constants

#Complex refractive index data (n+i*k) for gold.  
#Columns are:  0:  wavel in um; 1: n; 2: k.
gold_refrac_data = np.genfromtxt("goldrefrac.txt", dtype=float)

wavelidx = 0
nidx = 1
kidx = 2

#Assume normal incidence for all light.  Not necessarily true, but good enough for an estimate.  
#Any deviation from normal incidence only lowers the background, in any case.
gold_norm_reflec = np.abs((1.-(gold_refrac_data[:,nidx]+1j*gold_refrac_data[:,kidx]))/\
                  (1.+gold_refrac_data[:,nidx]+1j*gold_refrac_data[:,kidx]))**2.

mask_avg_reflec = 0.5 #Average reflectivity of the coronagraph mask
emiss = 1. - mask_avg_reflec*(gold_norm_reflec**10.) #Effective emissivity of the instrument is 1 - Re1*Re2*Re3*...

#Thermal flux in one diffraction-limited pixel in W/um.  Other samplings can be made by changing the pre-factor, presumably.

therm_flux = 2.*scipy.constants.c**2.*scipy.constants.h*emiss/((gold_refrac_data[:,wavelidx])**3.*(1.e-6)**2.*(np.exp(scipy.constants.h*scipy.constants.c/(gold_refrac_data[:,wavelidx]*1.e-6*scipy.constants.k*270.))-1))

wavelcol = pyfits.Column(name="Wavelength", array=gold_refrac_data[:,wavelidx], format="E", unit="um")
fluxcol = pyfits.Column(name="Flux", array=therm_flux, format="E", unit="W / um")
thermhdu = pyfits.BinTableHDU.from_columns([wavelcol, fluxcol])

mpl.figure()
mpl.plot(wavelcol.array, fluxcol.array)
mpl.show(block=False)

