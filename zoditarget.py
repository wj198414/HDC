import numpy as np
import astropy.io.fits as pyfits
import astropy.units as u
from spectrum import Spectrum

#This class carries the exozodiacal background for the observation.  The relevant portions of the interface are the same as for Target().

class ZodiTarget():
    def __init__(self, instrument, distance=10.0, spec_path=None, exozodi_level=1.0, spec_reso=1e5):
        self.distance = distance
        self.instrument = instrument
        self.exozodi_level = exozodi_level  #Level of zodi relative to Solar System
        self.spec_path = spec_path
        self.spec_reso = spec_reso
        self.spec_header = pyfits.open(self.spec_path)[0].header
        self.spec_data = self.sumZodiFlux().data
        self.wavelength = self.spec_data["Wavelength"]
        self.flux = self.spec_data["Flux"]
        self.spectrum = Spectrum(self.wavelength, self.flux, spec_reso=self.spec_reso)

        #This function converts the simulation data into the format that the rest of the program expects.
        #Because the simulation is on a large physical scale with relatively fine resolution, each pixel on
        #the instrument corresponds to many pixels in the simulation data.  Therefore, it has to be summed
        #before being used.

    def sumZodiFlux(self):
        haystacks = pyfits.open(self.spec_path)
        #Location of the exoEarth in the exozodi simulation.  Using one side in particular
        #because the simulation includes the Solar System planets, and we don't want to
        #double count the Earth.  We will unfortunately however include light from Mars,
        #but c'est la vie.  Format is y, x.
        planetcen = [1666, 1633]
        N_EXT = self.spec_header['N_EXT']
        wavel = haystacks[N_EXT+1].data
        pixel_scale = self.spec_header['PIXSCALE'] * u.au  #Pixel scale of the simulation, in AU/pix
        
        #Half the number of zodi sim pixels on a side that fit into one diffraction-limited pixel
        #in the instrument

        N_pix = np.array(np.ceil((wavel*u.um*self.distance/(self.instrument.telescope_size*u.m*pixel_scale)).decompose()/2)).astype(int)
        
        #Sum up the zodi at each wavelength

        zodi_per_pix = []

        for i in range(N_EXT):
            zodi_per_pix.append(np.sum(haystacks[i+1].data[planetcen[0]-N_pix[i]:planetcen[0]+N_pix[i]+1, 
                                                           planetcen[1]-N_pix[i]:planetcen[1]+N_pix[i]+1]))

        #Scale the exozodi brightness to the proper distance and total exozodi level; 
        #the Haystacks model is the Solar System at 10 pc
        zodi_per_pix = np.array(zodi_per_pix) * (self.distance/self.spec_header["DIST"])**2 * self.exozodi_level

        wavelcol = pyfits.Column(name="Wavelength", array=wavel, format="E", unit="um")
        fluxcol = pyfits.Column(name="Flux", array=zodi_per_pix, format="E", unit="Jy")
        zodihdu = pyfits.BinTableHDU.from_columns([wavelcol, fluxcol])

        return zodihdu