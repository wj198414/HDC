#Test of the supersampling algorithm

import numpy as np
from math import sqrt
import astropy.io.fits as pyfits
import astropy.units as u
import os
import matplotlib.pyplot as plt

spec_path = "HaystacksCubes"

planetcen = [1666, 1633]
distance = 5.0
telescope_size = 10.0

wavel = []
zodi_per_pix = []

for zodicube in sorted(os.listdir(spec_path)):
    
    haystacks = pyfits.open(spec_path+"/"+zodicube)

    N_EXT = haystacks[0].header['N_EXT']
    wavel_temp = haystacks[N_EXT+1].data
    wavel.append(wavel_temp)
    pixel_scale = haystacks[0].header['PIXSCALE'] * u.au  #Pixel scale of the simulation, in AU/pix
    scale_distance = haystacks[0].header["DIST"] #This is needed later on outside the loop

    #Make a circular aperture corresponding to a lambda/D sized planet PSF
    #pix_rad = np.array((wavel_temp*u.um*distance*u.pc/(telescope_size*u.m*pixel_scale)).decompose()/2)
    pix_rad = np.ones(len(wavel_temp))
    #print pix_rad
    #Sum up the zodi inside the planet PSF at each wavelength
    for i in range(N_EXT):
        scale_factor = 2 #Supersampling scale factor
        #Spawn interpolator for each wavelength
        #Evaluate within upsacled PSF region
        #Sum
        #Divide by scale_factor**2 to correct for the extra "pixels" in the sum

        one_pix_flux = haystacks[i+1].data[planetcen[0], planetcen[1]]
        #print one_pix_flux
        PSF_flux = one_pix_flux*np.pi*pix_rad[i]**2.
        zodi_per_pix.append(PSF_flux)

zodi_per_pix = np.array(zodi_per_pix) #* (scale_distance/self.distance)**2 * self.exozodi_level
#Ensure that the data are ordered by wavelength
wavel = np.array(wavel).ravel()                                               
wavel_order = np.argsort(wavel)
wavel = wavel[wavel_order]
zodi_per_pix = zodi_per_pix[wavel_order]

wavelcol = pyfits.Column(name="Wavelength", array=wavel, format="E", unit="um")
fluxcol = pyfits.Column(name="Flux", array=zodi_per_pix, format="E", unit="Jy")
zodihdu = pyfits.BinTableHDU.from_columns([wavelcol, fluxcol])

plt.figure()
plt.plot(wavel, zodi_per_pix)
plt.show(block=False)

spec_reso = 10000.
pixel_sampling = 3.0
num_pixel = len(wavel)
num_pixel_new = (np.nanmax(wavel) - np.nanmin(wavel)) / (np.nanmedian(wavel) / spec_reso) * pixel_sampling
wav_new = np.linspace(np.nanmin(wavel), np.nanmax(wavel), num = int(num_pixel_new))
flx_new = np.interp(wav_new, wavel, zodi_per_pix)

plt.figure()
plt.plot(wav_new, flx_new, ".")
plt.show(block=False)