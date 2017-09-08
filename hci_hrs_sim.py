import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import astropy.io.ascii as ascii
import scipy.constants
import pickle
from scipy import signal
from astropy import units as u
import numpy.fft as fft
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from scipy import interpolate
import time
import scipy.interpolate
import scipy.fftpack as fp
from datetime import datetime

class Target():
    def __init__(self, distance=10.0, spec_path=None, inclination_deg=90.0, rotation_vel=5e3, radial_vel=1e4, spec_reso=1e5):
        self.distance = distance
        self.spec_path =  spec_path
        self.spec_reso = spec_reso
        if self.spec_path != None:
            self.spec_data = pyfits.open(self.spec_path)[1].data
            self.wavelength = self.spec_data["Wavelength"]
            self.flux = self.spec_data["Flux"]
            self.spectrum = Spectrum(self.wavelength, self.flux, spec_reso=self.spec_reso)
            self.spec_header = pyfits.open(self.spec_path)[1].header
            self.PHXREFF = self.spec_header["PHXREFF"]
        self.inclination_deg = inclination_deg
        self.rotation_vel = rotation_vel
        self.radial_vel = radial_vel

class Instrument():
    def __init__(self, wav_med, telescope_size=10.0, pl_st_contrast=1e-10, spec_reso=1e5, read_noise=2.0, dark_current=1e-3, fiber_size=1.0, pixel_sampling=3.0, throughput=0.1, wfc_residual=200.0):   
        self.telescope_size = telescope_size
        self.pl_st_contrast = pl_st_contrast
        self.spec_reso = spec_reso
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.fiber_size = fiber_size
        self.pixel_sampling = pixel_sampling
        self.wfc_residual = wfc_residual # in nm
        self.wav_med = wav_med # in micron
        self.strehl = self.calStrehl()
        self.relative_strehl = self.calRelativeStrehl()
        self.throughput = throughput * self.relative_strehl

    def calStrehl(self):
        strehl = np.exp(-(2.0 * np.pi * (self.wfc_residual / 1e3 / self.wav_med))**2) 
        return(strehl)

    def calRelativeStrehl(self):
        strehl_K = np.exp(-(2.0 * np.pi * (self.wfc_residual / 1e3 / 2.0))**2)
        strehl = self.calStrehl()
        return(strehl / strehl_K)

class Spectrum():
    def __init__(self, wavelength, flux, spec_reso=1e5, norm_flag=False):
        self.wavelength = wavelength
        self.flux = flux
        self.spec_reso = spec_reso
        self.norm_flag = norm_flag
        self.noise = None

    def addNoise(self, noise):
        if self.noise == None:
            self.noise = noise
        else:
            print("Warning: spectrum noise already added")

    def writeSpec(self, file_name="tmp.dat"):
        with open(file_name, "wb") as f:
            for i in np.arange(len(self.wavelength)):
                f.write("{0:20.8f}{1:20.8e}\n".format(self.wavelength[i], self.flux[i]))

    def getSpecNorm(self, num_chunks=10, poly_order=2, emission=False):
        wav = self.wavelength
        flx = self.flux
        num_pixels = len(wav)
        pix_chunk = int(np.floor(num_pixels / num_chunks))
        wav_chunk = np.zeros((num_chunks,))
        flx_chunk = np.zeros((num_chunks,))
        for i in np.arange(num_chunks):
            wav_chunk[i] = np.nanmedian(wav[i*pix_chunk:(i+1)*pix_chunk])
            if not emission:
                flx_chunk[i] = np.nanmax(flx[i*pix_chunk:(i+1)*pix_chunk])
            else:
                flx_chunk[i] = np.nanmin(flx[i*pix_chunk:(i+1)*pix_chunk]) 
        coeff = np.polyfit(wav_chunk, flx_chunk, poly_order)
        p = np.poly1d(coeff)
        flx_norm = p(wav)
        return(flx_norm)

    def combineSpec(self, spec):
        spec_new = self.copy()
        idx = np.argsort(np.hstack((self.wavelength, spec.wavelength)))
        spec_new.wavelength = np.hstack((self.wavelength, spec.wavelength))[idx]
        spec_new.flux = np.hstack((self.flux, spec.flux))[idx]
        if spec_new.noise != None:
            print("Combining spectrum may cause trouble for attribute Noise")
        return(spec_new)

    def getChunk(self, wav_min, wav_max):
        spec_new = self.copy()
        idx = np.where((self.wavelength <= wav_max) & (self.wavelength > wav_min))
        spec_new.wavelength = self.wavelength[idx]
        spec_new.flux = self.flux[idx]
        if spec_new.noise != None:
            spec_new.noise = self.noise[idx]
        return(spec_new)

    def saveSpec(self, file_name="tmp.pkl"):
        with open(file_name, "wb") as handle:
            pickle.dump(self, handle)

    def simSpeckleNoise(self, wav_min, wav_max, wav_int, wav_new):
        wav = np.arange(wav_min, wav_max + wav_int / 2.0, wav_int)
        wav_arr = np.array([])
        flx_arr = np.array([])
        for i, wav_tmp in enumerate(wav[:-1]):
            wav_mid = np.random.normal(wav_tmp + wav_int * 0.5, wav_int / 10.0, size=(1,))
            flx_mid = np.random.random(size=(1,)) * 0.5 + 0.5 
            flx_tmp = np.random.random(size=(1,)) * 0.5 + 1.0
            wav_arr = np.hstack((wav_arr, [wav_tmp, wav_mid[0]]))
            flx_arr = np.hstack((flx_arr, [flx_tmp, flx_mid[0]]))
        wav_arr = np.hstack((wav_arr, [wav[-1]]))
        flx_arr = np.hstack((flx_arr, [np.random.random(1)[0] + 1.0]))
        f = scipy.interpolate.interp1d(wav_arr, flx_arr, kind="cubic")
        flx_new = f(wav_new)
        idx = np.where(flx_new < 0.1)
        flx_new[idx] = 0.1
        return(flx_new)

    def generateNoisySpec(self, speckle_noise=False):
        spec = self.copy()
        flx = self.flux
        flx_new = np.zeros(np.shape(flx))
        num = len(flx)
        i = 0
        while i < num:
            #flx_new[i] = np.max([np.random.poisson(np.round(flx[i]), 1)+0.0, np.random.normal(flx[i], self.noise[i], 1)])
            flx_new[i] = np.random.normal(flx[i].value, self.noise[i].value, 1)
            i = i + 1
        spec.flux = flx_new

        if speckle_noise:
            flx_speckle = self.simSpeckleNoise(np.min(spec.wavelength), np.max(spec.wavelength), 0.1, spec.wavelength)
            spec.flux = spec.flux * flx_speckle

        return(spec)

    def evenSampling(self):
        wav = self.wavelength
        flx = self.flux
        wav_int = np.median(np.abs(np.diff(wav)))
        wav_min = np.min(wav)
        wav_max = np.max(wav)
        wav_new = np.arange(wav_min, wav_max, wav_int)
        flx_new = np.interp(wav_new, wav, flx)
        self.wavelength = wav_new
        self.flux = flx_new

        return(self)

    def applyHighPassFilter(self, order = 5, cutoff = 100.0):
        # cutoff is number of sampling per 1 micron, so 100 means 0.01 micron resolution, about R = 100 at 1 micron
        x = self.wavelength
        y = self.flux
        n = self.noise
        fs = 1.0 / np.median(x[1:-1] - x[0:-2])
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
        yy = scipy.signal.filtfilt(b, a, y)
        spec = Spectrum(x, yy, spec_reso=self.spec_reso)
        if n is not None:
            spec.addNoise(n)
        return(spec)

    def copy(self):
        # make a copy of a spectrum object
        spectrum_new = Spectrum(self.wavelength, self.flux, spec_reso=self.spec_reso)
        if self.noise != None:
            spectrum_new.noise = self.noise
        return(spectrum_new)

    def pltSpec(self, **kwargs):
        # plot wav vs. flx 
        # **kwargs accepted by plot
        # image is stored as tmp.png
        fig, ax = plt.subplots()
        ax.plot(self.wavelength, self.flux, **kwargs)
        plt.show()
        #fig.savefig("./tmp.png")

    def scaleSpec(self, total_flux=1e4):
        # scale spectrum so that summed flux from each pixel is equal to total_flux
        num_pixels = len(self.wavelength)
        spec_total_flux = np.sum(self.flux)
        flx = self.flux / spec_total_flux * total_flux
        self.flux = flx

    def resampleSpec(self, wav_new):
        # resample a spectrum to a new wavelength grid
        flx_new = np.interp(wav_new, self.wavelength, self.flux)
        self.wavelength = wav_new
        self.flux = flx_new
        return self

    def resampleSpectoSpectrograph(self, pixel_sampling=3.0):
        # resample a spectrum to a wavelength grid that is determed by spectral resolution and pixel sampling rate
        # num_pixel_new = wavelength coverage range / wavelength per resolution element * pixel sampling rate
        num_pixel = len(self.wavelength)
        num_pixel_new = (np.nanmax(self.wavelength) - np.nanmin(self.wavelength)) / (np.nanmedian(self.wavelength) / self.spec_reso) * pixel_sampling
        wav_new = np.linspace(np.nanmin(self.wavelength), np.nanmax(self.wavelength), num = num_pixel_new)
        flx_new = np.interp(wav_new, self.wavelength, self.flux)
        return(Spectrum(wav_new, flx_new, spec_reso=self.spec_reso))

    def dopplerShift(self, rv_shift=0e0):
        #positive number means blue shift and vice versa
        beta = rv_shift / scipy.constants.c
        wav_shifted = self.wavelength * np.sqrt((1 - beta)/(1 + beta))
        flx = np.interp(self.wavelength, wav_shifted, self.flux, left=np.nanmedian(self.flux), right=np.nanmedian(self.flux))
        flx[np.isnan(flx)] = np.nanmedian(flx)
        self.flux = flx
        return self

    def crossCorrelation(self, template, spec_mask=None, long_array=False, speed_flag=False):
        # positive peak means spectrum is blue shifted with respect to template
        # do not recommend long_array option. It does not produce the same SNR as the non-long_array option. 
        if not long_array:
            wav = self.wavelength
            flx = self.flux
            wav_temp = template.wavelength
            flx_temp = template.flux
            flx_temp = np.interp(wav, wav_temp, flx_temp)
            flx = flx - np.nanmedian(flx)
            flx_temp = flx_temp - np.nanmedian(flx_temp)
            if spec_mask != None:
                flx[spec_mask] = np.nanmedian(flx)
                flx_temp[spec_mask] = np.nanmedian(flx_temp)

            if speed_flag:
                num_pixels = len(wav)
                power_2 = np.ceil(np.log10(num_pixels + 0.0) / np.log10(2.0))
                num_pixels_new = 2.0**power_2
                wav_new = np.linspace(np.min(wav), np.max(wav), num_pixels_new)
                flx_new = np.interp(wav_new, wav, flx)
                flx_temp_new = np.interp(wav_new, wav, flx_temp)
                flx_temp = flx_temp_new
                flx = flx_new
                wav = wav_new

            cc = fp.ifft(fp.fft(flx_temp)*np.conj(fp.fft(flx)))
            ccf = fp.fftshift(cc)
            ccf = ccf - np.median(ccf)
            ccf = ccf.real 
    
            vel_int = np.nanmedian(np.abs(wav[1:-1] - wav[0:-2])) / np.nanmedian(wav) * scipy.constants.c
            nx = len(ccf)
            ccf = ccf / (nx + 0.0)
            vel = (np.arange(nx)-(nx-1)/2.0) * vel_int
        else:
            num_chunks = 4
            num_pixels = len(self.wavelength) 
            pix_chunk = int(np.floor(num_pixels / (num_chunks + 0.0)))
            for i in np.arange(num_chunks):
                spec_tmp = Spectrum(self.wavelength[i*pix_chunk:(i+1)*pix_chunk], self.flux[i*pix_chunk:(i+1)*pix_chunk])
                template_tmp = Spectrum(template.wavelength[i*pix_chunk:(i+1)*pix_chunk], template.flux[i*pix_chunk:(i+1)*pix_chunk])
                ccf_tmp = spec_tmp.crossCorrelation(template_tmp)
                if i == 0:
                    ccf_total = ccf_tmp
                else:
                    ccf_total = CrossCorrelationFunction(ccf_tmp.vel, ccf_tmp.ccf + ccf_total.ccf)
                vel = ccf_total.vel
                ccf = ccf_total.ccf             

        return(CrossCorrelationFunction(vel, ccf))

    def spectral_blur(self, rpower=1e5, quick_blur=False):
        # broaden a spectrum given its spectral resolving power
        if not quick_blur:
            wave = self.wavelength
            tran = self.flux
    
            wmin = wave.min()
            wmax = wave.max()
    
            nx = wave.size
            x  = np.arange(nx)
    
            A = wmin
            B = np.log(wmax/wmin)/nx
            wave_constfwhm = A*np.exp(B*x)
            tran_constfwhm = np.interp(wave_constfwhm, wave, tran)
            dwdx_constfwhm = np.diff(wave_constfwhm)
            fwhm_pix = wave_constfwhm[1:]/rpower/dwdx_constfwhm
    
            fwhm_pix  = fwhm_pix[0]
            sigma_pix = fwhm_pix/2.3548
            kx = np.arange(nx)-(nx-1)/2.
            kernel = 1./(sigma_pix*np.sqrt(2.*np.pi))*np.exp(-kx**2/(2.*sigma_pix**2))
    
            tran_conv = fft.ifft(fft.fft(tran_constfwhm)*np.conj(fft.fft(kernel)))
            tran_conv = fft.fftshift(tran_conv).real
            tran_oldsampling = np.interp(wave,wave_constfwhm,tran_conv)
    
            self.wavelength = wave
            self.flux = tran_oldsampling
        else:
            pixel_to_sum = int(102400.0 / rpower)
            if pixel_to_sum >= 1.5:
                num_pixels = len(self.wavelength)
                num_pixels_new = int(np.floor((num_pixels + 0.0) / pixel_to_sum))
                wav = np.zeros((num_pixels_new,))
                flx = np.zeros((num_pixels_new,))
                for i in np.arange(num_pixels_new):
                    wav[i] = np.mean(self.wavelength[i*pixel_to_sum:(i+1)*pixel_to_sum])       
                    flx[i] = np.mean(self.flux[i*pixel_to_sum:(i+1)*pixel_to_sum])
                self.wavelength = wav
                self.flux = flx
        return self

    def rotational_blur(self, rot_vel=3e4):
        # broaden a spectrum given the rotation of a target
        # kernel is a cosine function with only [-pi/2, pi/2] phase
        # -pi/2 phase corresponds to fwhm_pix for rpower of c / rot_vel
        wave = self.wavelength
        tran = self.flux

        wmin = wave.min()
        wmax = wave.max()

        nx = wave.size
        x  = np.arange(nx)

        A = wmin
        B = np.log(wmax/wmin)/nx
        wave_constfwhm = A*np.exp(B*x)
        tran_constfwhm = np.interp(wave_constfwhm, wave, tran)
        dwdx_constfwhm = np.diff(wave_constfwhm)
        rpower = scipy.constants.c / rot_vel
        fwhm_pix = wave_constfwhm[1:]/rpower/dwdx_constfwhm

        fwhm_pix  = fwhm_pix[0]
        sigma_pix = fwhm_pix/2.3548
        kx = np.arange(nx)-(nx-1)/2.
        kernel = np.cos(2.0 * np.pi * kx / (4.0 * fwhm_pix))
        idx = ((kx < -fwhm_pix) | (kx > fwhm_pix))
        kernel[idx] = 0.0
        kernel = kernel / np.sum(kernel)

        tran_conv = fft.ifft(fft.fft(tran_constfwhm)*np.conj(fft.fft(kernel)))
        tran_conv = fft.fftshift(tran_conv).real
        tran_oldsampling = np.interp(wave,wave_constfwhm,tran_conv)

        self.wavelength = wave
        self.flux = tran_oldsampling

        return self

class CrossCorrelationFunction():
    def __init__(self, vel, ccf):
        self.vel = vel
        self.ccf = ccf

    def getCCFchunk(self, vmin=-1e9, vmax=1e9):
        cc = self.ccf
        vel = self.vel
        idx = np.where((vel < vmax) & (vel > vmin))
        return(CrossCorrelationFunction(vel[idx], cc[idx]))

    def pltCCF(self, save_fig=False):
        plt.plot(self.vel, self.ccf)
        plt.show()
        if save_fig:
            plt.savefig("tmp.png")

    def calcCentroid(self,cwidth=5):
        cc = self.ccf
        vel = self.vel
        
        #
        idx = np.where((vel < (scipy.constants.c / 20.0)) & (vel > (-scipy.constants.c / 20.0)))
        #idx = np.where((vel < (scipy.constants.c / 300.0)) & (vel > (-scipy.constants.c / 300.0)))
        cc = cc[idx]
        vel = vel[idx]
        #

        maxind = np.argmax(cc)
        mini = max([0,maxind-cwidth])
        maxi = min([maxind+cwidth+1,cc.shape[0]])
        weight = cc[mini:maxi] - np.min(cc[mini:maxi])
        centroid = (vel[mini:maxi]*weight).sum()/weight.sum()
        return centroid

    def calcSNRrms(self, peak=None):
        cc = self.ccf

        # 
        vel = self.vel
        idx = np.where((vel < (scipy.constants.c / 20.0)) & (vel > (-scipy.constants.c / 20.0)))
        cc_tmp = cc[idx]
        ind_max = np.argmax(cc_tmp) + idx[0][0]
        #

        #ind_max = np.argmax(cc)
        num = len(cc)
        if ind_max > (num / 2.0):
            ind_rms = [0, int(num / 4.0)]
        else:
            ind_rms = [-int(num / 4.0), -1]
        snr = cc[ind_max] / np.std(cc[ind_rms[0]:ind_rms[1]])
        if not (peak is None):
            snr = peak / np.std(cc[ind_rms[0]:ind_rms[1]])
        return(snr)

    def calcSNRnoiseLess(self, ccf_noise_less):
        cc = self.ccf
        cc_subtracted = cc - ccf_noise_less.ccf
        nx = len(cc) + 0.0

        ind_max = np.argmax(cc)
        num = len(cc)
        if ind_max > (num / 2.0):
            ind_rms = [0, int(num / 4.0)]
        else:
            ind_rms = [-int(num / 4.0), -1]
        snr = np.max([cc[ind_max] / np.std(cc_subtracted[0:int(num / 4.0)]), cc[ind_max] / np.std(cc_subtracted[-int(num / 4.0):-1])])
        return(snr)

    def calcPeak(self):
        cc = self.ccf
        nx = len(cc) + 0.0

        vel = self.vel
        idx = np.where((vel < (scipy.constants.c / 20.0)) & (vel > (-scipy.constants.c / 20.0)))
        #idx = np.where((vel < (scipy.constants.c / 300.0)) & (vel > (-scipy.constants.c / 300.0)))
        cc_tmp = cc[idx]
        ind_max = np.argmax(cc_tmp) + idx[0][0]

        return(cc[ind_max])

class Atmosphere():
    def __init__(self, spec_tran_path=None, spec_radi_path=None, radial_vel=1e1):
        self.spec_tran_path = spec_tran_path
        self.spec_radi_path = spec_radi_path
        self.radial_vel = radial_vel
        if self.spec_tran_path != None:
            with open(spec_tran_path, "rb") as handle:
                [self.spec_tran_wav, self.spec_tran_flx] = pickle.load(handle) 
        else:
            self.spec_tran_wav = np.arange(0.1, 5.0, 1e-5)
            self.spec_tran_flx = np.zeros(np.shape(self.spec_tran_wav)) + 1.0
        if self.spec_radi_path != None:
            self.spec_radi_data = ascii.read(spec_radi_path)
            self.spec_radi_wav = self.spec_radi_data["col1"][:] # in nm
            self.spec_radi_wav = self.spec_radi_wav / 1e3 # now in micron
            self.spec_radi_flx = self.spec_radi_data["col2"][:] # in ph/s/arcsec**2/nm/m**2
            self.spec_radi_flx = self.spec_radi_flx * 1e3 # now in ph/s/arcsec**2/micron/m**2
            self.spec_radi_wav = np.hstack([np.arange(0.1, 0.9, 1e-5), self.spec_radi_wav]) # to avoid missing information in optical below 0.9 micron
            self.spec_radi_flx = np.hstack([np.zeros(np.shape(np.arange(0.1, 0.9, 1e-5))) + 1e-99, self.spec_radi_flx])
        else:
            self.spec_radi_wav = np.arange(0.1, 5.0, 1e-5)
            self.spec_radi_flx = np.zeros(np.shape(self.spec_radi_wav)) + 1e-99

    def getTotalSkyFlux(self, wav_min, wav_max, tel_size=10.0, multiple_lambda_D=1.0, t_exp=1e3, eta_ins=0.1):
        # get total flux of sky emission
        idx = ((self.spec_radi_wav < wav_max) & (self.spec_radi_wav > wav_min))
        wav = self.spec_radi_wav[idx]
        flx = self.spec_radi_flx[idx]
        wav_int = np.abs(wav[1:-1] - wav[0:-2])
        fiber_size = np.nanmedian(wav) * 1e-6 / tel_size / np.pi * 180.0 * 3600.0
        fiber_size = fiber_size * multiple_lambda_D # multiple times lambda / D
        flx_skybg_total = np.sum(flx[0:-2] * t_exp * fiber_size **2 * wav_int * np.pi * (tel_size / 2.0)**2) * eta_ins
        
        return(flx_skybg_total)
        
class HCI_HRS_Observation():
    def __init__(self, wav_min, wav_max, t_exp, target_pl, target_st, instrument, atmosphere=None):
        self.wav_min = wav_min
        self.wav_max = wav_max
        self.t_exp = t_exp
        self.planet = target_pl
        self.star = target_st
        self.instrument = instrument
        self.atmosphere = atmosphere
        self.execute()

    def execute(self):

        # Construct HCI+HRS spectrum

        # get star spectrum within wavelength range, remove NaNs, and calculate total star flux
        spec_star = self.getSpecChunk(self.star.wavelength, self.star.flux)
        self.star_spec_chunk = Spectrum(spec_star["Wavelength"], spec_star["Flux"], spec_reso=self.star.spec_reso)
        self.star_spec_chunk = self.removeNanInSpecChunk(self.star_spec_chunk)
        self.star_spec_chunk.evenSampling()
        self.star_total_flux = self.getTotalFlux(self.star_spec_chunk.wavelength, self.star_spec_chunk.flux, self.star.distance, self.star.PHXREFF, self.instrument.telescope_size, self.instrument.throughput, self.t_exp)

        # get planet spectrum within wavelength range, remove NaNs, and calculate total planet flux
        spec_planet = self.getSpecChunk(self.planet.wavelength, self.planet.flux)
        self.planet_spec_chunk = Spectrum(spec_planet["Wavelength"], spec_planet["Flux"], spec_reso=self.planet.spec_reso)
        self.planet_spec_chunk = self.removeNanInSpecChunk(self.planet_spec_chunk)
        self.planet_total_flux = self.getTotalFlux(self.planet_spec_chunk.wavelength, self.planet_spec_chunk.flux, self.planet.distance, self.planet.PHXREFF, self.instrument.telescope_size, self.instrument.throughput, self.t_exp)
        # resample planet spectrum to star wavelength scale
	self.planet_spec_chunk.resampleSpec(self.star_spec_chunk.wavelength)

        # Considering Earth's atmosphere, e.g., ground-based observation
        if self.atmosphere != None:
            # get the transmission spectrum 
            spec_atm_tran = self.getSpecChunk(self.atmosphere.spec_tran_wav, self.atmosphere.spec_tran_flx)
            self.atm_tran_spec_chunk = Spectrum(spec_atm_tran["Wavelength"], spec_atm_tran["Flux"], spec_reso=self.star.spec_reso)
            self.atm_tran_spec_chunk = self.removeNanInSpecChunk(self.atm_tran_spec_chunk)
            # get the emission spectrum
            spec_atm_radi = self.getSpecChunk(self.atmosphere.spec_radi_wav, self.atmosphere.spec_radi_flx)
            self.atm_radi_spec_chunk = Spectrum(spec_atm_radi["Wavelength"], spec_atm_radi["Flux"], spec_reso=self.star.spec_reso)
            self.atm_radi_spec_chunk = self.removeNanInSpecChunk(self.atm_radi_spec_chunk)
            # calculate total flux from sky emission
            self.sky_total_flux = self.atmosphere.getTotalSkyFlux(self.wav_min, self.wav_max, tel_size=self.instrument.telescope_size, multiple_lambda_D=self.instrument.fiber_size, t_exp=self.t_exp, eta_ins=self.instrument.throughput)
            # resample transmission and emission spectra to star wavelength scale
            self.atm_tran_spec_chunk.resampleSpec(self.star_spec_chunk.wavelength)
            self.atm_radi_spec_chunk.resampleSpec(self.star_spec_chunk.wavelength) 
            # doppler shift and rotationally broaden planet and star spectra
            self.planet_spec_chunk.dopplerShift(rv_shift=self.planet.radial_vel)
            #self.planet_spec_chunk.rotational_blur(rot_vel=self.planet.rotation_vel)
            self.star_spec_chunk.dopplerShift(rv_shift=self.star.radial_vel)
            #self.star_spec_chunk.rotational_blur(rot_vel=self.star.rotation_vel)
            # doppler shift transmission and emission spectra
            self.atm_tran_spec_chunk.dopplerShift(rv_shift=self.atmosphere.radial_vel)
            self.atm_radi_spec_chunk.dopplerShift(rv_shift=self.atmosphere.radial_vel)  
            # calculate sky transmission rate
            self.sky_transmission = np.sum(self.atm_tran_spec_chunk.flux) / (0.0 + len(self.atm_tran_spec_chunk.flux))


            # construct spectrum with planet, star and atmospheric transmission
            # pl_st spectrum = (planet + star * contrast) * transmission 
            # pl_st spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            obs_spec_wav = self.star_spec_chunk.wavelength
            obs_spec_pl_st = ((self.planet_spec_chunk.flux + self.star_spec_chunk.flux * self.instrument.pl_st_contrast) * self.atm_tran_spec_chunk.flux)
            self.obs_pl_st = Spectrum(obs_spec_wav, obs_spec_pl_st, spec_reso=self.star.spec_reso) 
            self.obs_pl_st.spectral_blur(rpower=self.star.spec_reso)
            self.obs_pl_st_resample = self.obs_pl_st.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            # pl_st spectrum is scaled by total flux from the star and the planet and the atmosphere transmission
            self.obs_pl_st_resample.scaleSpec(total_flux=self.sky_transmission * (self.planet_total_flux + self.star_total_flux * self.instrument.pl_st_contrast))

            # construct spectrum with star and atmospheric transmission
            # st spectrum = star * transmission 
            # st spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            obs_spec_wav = self.star_spec_chunk.wavelength
            obs_spec_st = self.star_spec_chunk.flux * self.atm_tran_spec_chunk.flux
            self.obs_st = Spectrum(obs_spec_wav, obs_spec_st, spec_reso=self.star.spec_reso)
            self.obs_st.spectral_blur(rpower=self.star.spec_reso)
            self.obs_st_resample = self.obs_st.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            # st spectrum is scaled by total flux from the star and the atmosphere transmission
            self.obs_st_resample.scaleSpec(total_flux=self.sky_transmission * self.star_total_flux)

            # construct spectrum with planet and atmospheric transmission
            # pl spectrum = planet  
            # pl spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            obs_spec_wav = self.star_spec_chunk.wavelength
            obs_spec_pl = self.planet_spec_chunk.flux 
            self.obs_pl = Spectrum(obs_spec_wav, obs_spec_pl, spec_reso=self.planet.spec_reso)
            self.obs_pl.spectral_blur(rpower=self.planet.spec_reso)
            self.obs_pl_resample = self.obs_pl.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            # pl spectrum is scaled by total flux from the planet and the atmosphere transmission
            self.obs_pl_resample.scaleSpec(total_flux=self.sky_transmission * self.planet_total_flux)

            # construct spectrum with atmospheric emission, which is independent (addable) with pl_st spectrum
            self.atm_radi_spec_chunk.spectral_blur(rpower=self.star.spec_reso)
            self.atm_radi_spec_chunk_resample = self.atm_radi_spec_chunk.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.atm_radi_spec_chunk_resample.scaleSpec(total_flux=self.sky_total_flux)
            # construct final spectrum with pl_st and radi
            self.obs_spec_resample = Spectrum(self.atm_radi_spec_chunk_resample.wavelength, self.atm_radi_spec_chunk_resample.flux + self.obs_pl_st_resample.flux, spec_reso=self.star.spec_reso)

            # calculate noise for obs_spec_resample, atm_radi_spec_chunk_resample, and obs_st_resample
            noise = self.calNoise(self.obs_spec_resample)
            self.obs_spec_resample.addNoise(noise)
            noise = self.calNoise(self.atm_radi_spec_chunk_resample)
            self.atm_radi_spec_chunk_resample.addNoise(noise)
            noise = self.calNoise(self.obs_st_resample)
            self.obs_st_resample.addNoise(noise)
            noise = self.calNoise(self.obs_pl_resample)
            self.obs_pl_resample.addNoise(noise)

        # Excluding Earth's atmosphere, e.g., space-based observation  
        else:
            self.sky_total_flux = 0.0
            self.sky_transmission = 1.0
            # doppler shift and rotationally broaden planet and star spectra
            self.planet_spec_chunk.dopplerShift(rv_shift=self.planet.radial_vel)
            #self.planet_spec_chunk.rotational_blur(rot_vel=self.planet.rotation_vel)
            self.star_spec_chunk.dopplerShift(rv_shift=self.star.radial_vel)
            #self.star_spec_chunk.rotational_blur(rot_vel=self.star.rotation_vel)
            # construct spectrum with star only
            # st spectrum = star  
            # st spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            self.obs_st = self.star_spec_chunk.copy()
            self.obs_st.spectral_blur(rpower=self.star.spec_reso, quick_blur=True)
            self.obs_st_resample = self.obs_st.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.obs_st_resample.scaleSpec(total_flux=self.star_total_flux)
            # construct spectrum with planet only
            # pl spectrum = planet  
            # pl spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            self.obs_pl = self.planet_spec_chunk.copy()
            self.obs_pl.spectral_blur(rpower=self.planet.spec_reso, quick_blur=True)
            self.obs_pl_resample = self.obs_pl.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.obs_pl_resample.scaleSpec(total_flux=self.planet_total_flux)
            # construct spectrum with planet and star 
            # obs = (planet + star * contrast) 
            # pl_st spectrum is then spectrally blurred and resampled to spectrograph wavelength grid
            obs_spec_wav = self.star_spec_chunk.wavelength
            obs_spec_flx = self.planet_spec_chunk.flux + self.star_spec_chunk.flux * self.instrument.pl_st_contrast
            self.obs_spec = Spectrum(obs_spec_wav, obs_spec_flx, spec_reso=self.star.spec_reso)            
            self.obs_spec.spectral_blur(rpower=self.star.spec_reso, quick_blur=True)
            self.obs_spec_resample = self.obs_spec.resampleSpectoSpectrograph(pixel_sampling=self.instrument.pixel_sampling)
            self.obs_spec_resample.scaleSpec(total_flux=self.planet_total_flux + self.star_total_flux * self.instrument.pl_st_contrast)
            # calculate noise for obs_spec_resample, atm_radi_spec_chunk_resample, and obs_st_resample
            noise = self.calNoise(self.obs_spec_resample)
            self.obs_spec_resample.addNoise(noise)
            noise = self.calNoise(self.obs_st_resample)
            self.obs_st_resample.addNoise(noise)
            noise = self.calNoise(self.obs_pl_resample)
            self.obs_pl_resample.addNoise(noise)

    def calNoise(self, spec):
        flx = spec.flux
        num_read = np.max([np.round(np.sort(flx)[int(0.9 * len(flx))] / 1e6), 1]) # 1e6 is linearity/persitence range
        #print("number of read = ", num_read)
        var = (flx + self.instrument.dark_current * self.t_exp + self.instrument.read_noise**2 * num_read)
        noise = np.sqrt(np.abs(var))
        idx = np.where(noise < 1.0)
        noise[idx] = 1.0
        return(noise)

    def getSpecChunk(self, wav, flx):
        # get spectrum within wavelength range
        idx = ((wav < self.wav_max) & (wav > self.wav_min))
        return {'Wavelength':wav[idx],'Flux':flx[idx]}
                   
    def removeNanInSpecChunk(self, spectrum):
        idx = np.isnan(spectrum.flux)
        spectrum.flux[idx] = np.nanmedian(spectrum.flux)
        return(spectrum)

    def getTotalFlux(self, wav, flx, dis, PHXREFF, tel_size, eta_ins, t_exp):
        #dis in pc
        #PHXREFF in m, found in PHOENIX header
        #wav in um
        #flx in W/m^2/um
        #tel_size in m
        #eta_ins considers both telescope and instrument throughput

        wav_u = wav * u.micron
        flx_u = flx * u.si.watt / u.meter / u.meter / u.micron
        dis_u = (dis * u.pc).to(u.meter)
        PHXREFF_u = PHXREFF * u.meter
        tel_size_u = tel_size * u.meter
        wav_u_inc = wav_u[1:] - wav_u[0:-1]
        #print(np.median(wav_u_inc))
        planck_const = 6.6260696e-34 * u.joule * u.second
        speed_of_light = 299792458.0 * u.meter / u.second
        fre_u = speed_of_light / wav_u.to(u.meter)
    
        photon_energy_u = planck_const * fre_u

        flx_u_photon = (flx_u[1:] * np.pi * (tel_size_u / 2.0)**2 * wav_u_inc * (PHXREFF_u / dis_u)**2 *                     (t_exp * u.second)).value * u.joule / photon_energy_u[1:] * eta_ins
        return(np.sum(flx_u_photon)) 

class HCI_HRS_Reduction():
    def __init__(self, hci_hrs_obs, template, save_flag=False, obj_tag="a", template_tag="b", speckle_flag=False):
        self.hci_hrs_obs = hci_hrs_obs
        self.template = template
        self.save_flag=save_flag
        self.obj_tag = obj_tag
        self.template_tag = template_tag
        self.speckle_flag = speckle_flag
        self.execute()

    def execute(self):
        # get template spectrum for cross correlation
        template_chunk = self.getSpecChunk(self.template.wavelength, self.template.flux)
        self.template.wavelength = template_chunk["Wavelength"]
        self.template.flux = template_chunk["Flux"]
        self.template = self.removeNanInSpecChunk(self.template)
        # rotational and spectral broaden template and resample template to instrument grid
        self.template.resampleSpec(self.hci_hrs_obs.star_spec_chunk.wavelength)
        #self.template.rotational_blur(rot_vel=self.hci_hrs_obs.planet.rotation_vel)
        self.template.spectral_blur(rpower=self.template.spec_reso, quick_blur=True)
        self.template_resample = self.template.resampleSpectoSpectrograph(pixel_sampling=self.hci_hrs_obs.instrument.pixel_sampling)
        if self.hci_hrs_obs.atmosphere != None:
            # remove sky emission with spectrum obteined from the sky fiber
            self.obs_emission_removed = self.removeSkyEmission()
            # remove star and atmospheric transmission with spectrum obtained from the star fiber
            self.obs_st_at_removed = self.removeSkyTransmissionStar()
            #self.obs_st_at_removed = self.obs_emission_removed
            #self.plotObsTemplate()
            # apply high pass filter to remove low frequency component
            self.template_high_pass = self.template_resample.applyHighPassFilter()
            self.obs_high_pass = self.obs_st_at_removed.applyHighPassFilter()
            # cross correlate reduced spectrum and template
            self.ccf_noise_less = self.obs_high_pass.crossCorrelation(self.template_high_pass)
            # simulate observation with noise
            result = self.simulateSingleMeasurement(plot_flag=False)
            print(result)
            self.writeLog(result)
            result = self.simulateMultiMeasurement(num_sim=10, ground_flag=True)
        else:
            self.obs_emission_removed = self.hci_hrs_obs.obs_spec_resample.copy()
            self.obs_st_at_removed = self.removeSkyTransmissionStar()            
            #obs_norm = self.obs_st_at_removed.getSpecNorm(num_chunks=20, poly_order=3)
            print(str(datetime.now()))
            obs_norm = self.getStarNorm(self.hci_hrs_obs.obs_st_resample.flux, long_array=True)
            obs_norm = obs_norm / np.median(obs_norm) * np.median(self.obs_st_at_removed.flux)
            self.obs_st_at_removed.flux = self.obs_st_at_removed.flux / obs_norm
            self.obs_st_at_removed.noise = self.obs_st_at_removed.noise / obs_norm
            #mask_arr = np.where((self.template_resample.flux / np.nanmedian(self.template_resample.flux)) > 0.99)
            mask_arr = np.where((self.template_resample.flux / np.nanmedian(self.template_resample.flux)) > 1e9)
            #plt.errorbar(self.obs_st_at_removed.wavelength, self.obs_st_at_removed.flux, yerr=self.obs_st_at_removed.noise)
            #plt.plot(self.template_resample.wavelength, self.template_resample.flux / np.median(self.template_resample.flux))
            #plt.plot(self.template_resample.wavelength[mask_arr], self.template_resample.flux[mask_arr] / np.median(self.template_resample.flux[mask_arr]))
            #plt.plot(self.obs_st_at_removed.wavelength[mask_arr], self.obs_st_at_removed.flux[mask_arr], "b.")
            #plt.show()
            if self.speckle_flag:
                self.cutoff_value = self.hci_hrs_obs.instrument.spec_reso / 6.0
                self.template_resample = self.template_resample.applyHighPassFilter(cutoff=self.cutoff_value)
                self.ccf_noise_less = self.obs_st_at_removed.applyHighPassFilter(cutoff=self.cutoff_value).crossCorrelation(self.template_resample, spec_mask=mask_arr, long_array=False, speed_flag=False)
            else:
                self.ccf_noise_less = self.obs_st_at_removed.crossCorrelation(self.template_resample, spec_mask=mask_arr, long_array=False, speed_flag=False)
            vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
            self.ccf_noise_less = self.ccf_noise_less.getCCFchunk(vmin=-50*vel_pixel, vmax=50*vel_pixel)
            self.ccf_peak = self.ccf_noise_less.calcPeak()
            result = self.simulateSingleMeasurement(ground_flag=False, plot_flag=False, speckle_flag=self.speckle_flag, spec_mask=mask_arr, long_array=False, speed_flag=False)
            print(result)
            self.writeLog(result)
            #plt.plot(result["CCF"].vel, result["CCF"].ccf, "bo-")
            #plt.plot(self.ccf_noise_less.vel, self.ccf_noise_less.ccf, "r")
            #plt.show()
            #plt.plot(result["CCF"].vel, result["CCF"].ccf - self.ccf_noise_less.ccf, "bo-")
            #plt.show()            
            #result = self.simulateMultiMeasurement(num_sim=100, ground_flag=False, speckle_flag=self.speckle_flag, spec_mask=mask_arr, long_array=False, speed_flag=True)
            result = self.simulateMultiMeasurement_2(num_sim=100, ground_flag=False, speckle_flag=self.speckle_flag, spec_mask=mask_arr, long_array=False, speed_flag=False)

        if self.save_flag:
            self.saveObject()            


    def saveObject(self, save_dir="/scr/jwang/hci_hds/OO_hci_hrs/pkl_dir/"):
        hci_hrs_name = "{0:06.3f}_{1:06.3}_{2:06.0f}".format(self.hci_hrs_obs.wav_min, self.hci_hrs_obs.wav_max, self.hci_hrs_obs.t_exp)
        obj_tag = self.obj_tag
        template_tag = self.template_tag
        instrument_name = "{0:05.1f}_{1:08.2e}_{2:08.2e}_{3:04.1f}".format(self.hci_hrs_obs.instrument.telescope_size, self.hci_hrs_obs.instrument.pl_st_contrast, self.hci_hrs_obs.instrument.spec_reso, self.hci_hrs_obs.instrument.read_noise)
        file_name = obj_tag+"_"+template_tag+"_"+hci_hrs_name+"_"+instrument_name+".pkl"
        with open(save_dir+file_name, "wb") as handle:
            pickle.dump(self, handle)

    def writeLog(self, result):
        hci_hrs_name = "{0:06.3f}_{1:06.3}_{2:06.0f}".format(self.hci_hrs_obs.wav_min, self.hci_hrs_obs.wav_max, self.hci_hrs_obs.t_exp)
        obj_tag = self.obj_tag
        template_tag = self.template_tag
        instrument_name = "{0:05.1f}_{1:08.2e}_{2:08.2e}_{3:04.1f}".format(self.hci_hrs_obs.instrument.telescope_size, self.hci_hrs_obs.instrument.pl_st_contrast, self.hci_hrs_obs.instrument.spec_reso, self.hci_hrs_obs.instrument.read_noise)
        log_tag = obj_tag+"_"+template_tag+"_"+hci_hrs_name+"_"+instrument_name # the same format as pkl file name
        time_tag = str(datetime.now())
        vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
        vel_offset_in_pixel = np.abs(result["Center"] - self.hci_hrs_obs.planet.radial_vel) / vel_pixel
        with open("log.dat", "a+") as f:
            f.write("{0:80s},{1:8.2e},{2:8.2e},{3:8.2e},{4:10.1f},{5:10.2f},{6:10.2f},{7:50s}\n".format(log_tag, self.hci_hrs_obs.planet.radial_vel, vel_pixel, result["Center"], vel_offset_in_pixel, result["SNR_RMS"], result["SNR_vs_NoiseLess"], time_tag))

    def getStarNorm(self, spec, num_chunks=20.0, long_array=False):
        if not long_array:
            if int(len(spec) / num_chunks) % 2 != 0:
                obs_norm = scipy.signal.medfilt(spec, kernel_size = int(len(spec) / num_chunks))
            else:
                obs_norm = scipy.signal.medfilt(spec, kernel_size = int(len(spec) / num_chunks) - 1)
        else:
            num_pixels = len(spec)
            num_division = 4
            pix_division = int(np.floor(num_pixels / (num_division + 0.0)))
            for i in np.arange(num_division):
                obs_norm_tmp = self.getStarNorm(spec[i*pix_division:(i+1)*pix_division])
                if i == 0:
                    obs_norm = obs_norm_tmp
                else:
                    obs_norm = np.hstack((obs_norm, obs_norm_tmp))
            obs_norm = np.hstack((obs_norm, spec[(i+1)*pix_division:]))

        return(obs_norm)

    def simulateSingleMeasurement(self, ground_flag=True, plot_flag=False, speckle_flag=False, **kwargs):
        if ground_flag:
            spec = self.obs_st_at_removed.generateNoisySpec().applyHighPassFilter()
            ccf = spec.crossCorrelation(self.template_high_pass, **kwargs)
        else:
            if speckle_flag:
                spec = self.obs_st_at_removed.generateNoisySpec(speckle_noise=True).applyHighPassFilter(cutoff=self.cutoff_value)    
                ccf = spec.crossCorrelation(self.template_resample, **kwargs)
            else:
                spec = self.obs_st_at_removed.generateNoisySpec(speckle_noise=False)
                ccf = spec.crossCorrelation(self.template_resample, **kwargs)
        if plot_flag:
            plt.plot(self.obs_st_at_removed.wavelength, self.obs_st_at_removed.flux)
            plt.plot(spec.wavelength, spec.flux)
            plt.show()
        vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
        ccf = ccf.getCCFchunk(vmin=-50*vel_pixel, vmax=50*vel_pixel)

        #cen = ccf.calcCentroid()
        cen = self.hci_hrs_obs.planet.radial_vel 

        #peak = ccf.calcPeak()
        dif = np.abs(ccf.vel - cen)
        ind = np.where(dif == np.min(dif))[0][0]
        pix_search = int(np.round(1e4 / vel_pixel)) # within 10 km/s 
        peak = np.max(ccf.ccf[ind-pix_search:ind+pix_search+1])

        ccf_orig = CrossCorrelationFunction(ccf.vel, ccf.ccf)
        ccf.ccf = ccf.ccf - self.ccf_noise_less.ccf
        ccf.ccf[ind-pix_search:ind+pix_search+1] = ccf_orig.ccf[ind-pix_search:ind+pix_search+1]

        snr_rms = ccf.calcSNRrms(peak=peak)
        snr_vs_noise_less = ccf.calcSNRnoiseLess(self.ccf_noise_less)

        return({"CCF":ccf, "Center":cen, "SNR_RMS":snr_rms, "SNR_vs_NoiseLess":snr_vs_noise_less, "CCF_peak":peak})

    def simulateMultiMeasurement(self, num_sim=10, **kwargs):
        info_arr = np.zeros((3, num_sim))
        for i in np.arange(num_sim):
            result = self.simulateSingleMeasurement(**kwargs)
            self.writeLog(result)
            vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
            vel_offset_in_pixel = np.abs(result["Center"] - self.hci_hrs_obs.planet.radial_vel) / vel_pixel
            if vel_offset_in_pixel <= 3.0: # this may only be relavant to ground based observation
            #if vel_offset_in_pixel <= 3e5:
                info_arr[:, i] = [result["SNR_RMS"], result["SNR_vs_NoiseLess"], 1]
            else:
                info_arr[:, i] = [0.0, 0.0, 0]
        peak_correction_rate = (len(info_arr[2,:][np.where(info_arr[2,:] == 1)]) + 0.0) / (num_sim + 0.0)
        if peak_correction_rate > 0.2:
            idx = np.where(info_arr[2,:] == 1)
            SNR_RMS_mean = np.median(info_arr[0,idx])
            SNR_RMS_std = np.std(np.sort(np.transpose(info_arr[0,idx]))[1:-1])
            SNR_vs_NoiseLess_mean = np.median(info_arr[1,idx])
            SNR_vs_NoiseLess_std = np.std(np.sort(np.transpose(info_arr[1,idx]))[1:-1])
        else:
            SNR_RMS_mean = 0.0
            SNR_RMS_std = 0.0
            SNR_vs_NoiseLess_mean = 0.0
            SNR_vs_NoiseLess_std = 0.0
        with open("multi_sim_log.dat", "a+") as f:
            f.write("{0:50s},{2:8.2e},{1:8.2e},{3:6.3f},{4:8.2e},{5:8.2e},{6:8.2e},{7:8.2e}\n".format(self.obj_tag, self.hci_hrs_obs.instrument.pl_st_contrast, self.hci_hrs_obs.instrument.spec_reso, peak_correction_rate, SNR_RMS_mean, SNR_RMS_std, SNR_vs_NoiseLess_mean, SNR_vs_NoiseLess_std))
        return([peak_correction_rate, SNR_RMS_mean, SNR_RMS_std, SNR_vs_NoiseLess_mean, SNR_vs_NoiseLess_std])

    def simulateMultiMeasurement_2(self, num_sim=10, **kwargs):
        info_arr = np.zeros((3, num_sim))
        for i in np.arange(num_sim):
            result = self.simulateSingleMeasurement(**kwargs)
            self.writeLog(result)
            vel_pixel = scipy.constants.c / self.hci_hrs_obs.instrument.spec_reso / self.hci_hrs_obs.instrument.pixel_sampling
            vel_offset_in_pixel = np.abs(result["Center"] - self.hci_hrs_obs.planet.radial_vel) / vel_pixel
            #result["CCF"].pltCCF()
            if vel_offset_in_pixel <= 10.0: # this may only be relavant to ground based observation
            #if vel_offset_in_pixel <= 3e5:
                info_arr[:, i] = [result["SNR_RMS"], result["CCF_peak"], 1]
            else:
                info_arr[:, i] = [0.0, 0.0, 0]
        peak_correction_rate = (len(info_arr[2,:][np.where(info_arr[2,:] == 1)]) + 0.0) / (num_sim + 0.0)
        if peak_correction_rate > 0.5:
            idx = np.where(info_arr[2,:] == 1)
            SNR_RMS_mean = np.median(info_arr[0,idx])
            SNR_RMS_std = np.std(np.sort(np.transpose(info_arr[0,idx]))[1:-1])
            CCF_peak_mean = np.median(info_arr[1,idx])
            CCF_peak_std = np.std(np.sort(np.transpose(info_arr[1,idx]))[2:-2])
            n, bins = np.histogram(np.transpose(info_arr[1,idx]), bins=np.linspace(0, np.max(info_arr[1,idx]), 10))
            #plt.plot(bins[0:-1], n, "b")
            #plt.plot([self.ccf_peak, self.ccf_peak], [0,num_sim/2.0],"r--")
            #plt.show()
            if (n[0] == 0.0) & (np.sort(np.transpose(info_arr[1,idx]))[int(np.floor((num_sim-1.0)*0.15))] < self.ccf_peak):
                SNR_vs_NoiseLess_mean = CCF_peak_mean / CCF_peak_std
            else:
                SNR_vs_NoiseLess_mean = 0.0
            SNR_vs_NoiseLess_std = CCF_peak_mean / CCF_peak_std
        else:
            SNR_RMS_mean = 0.0
            SNR_RMS_std = 0.0
            SNR_vs_NoiseLess_mean = 0.0
            SNR_vs_NoiseLess_std = 0.0
        with open("multi_sim_log.dat", "a+") as f:
            f.write("{0:50s},{2:8.2e},{1:8.2e},{3:6.3f},{4:8.2e},{5:8.2e},{6:8.2e},{7:8.2e}\n".format(self.obj_tag, self.hci_hrs_obs.instrument.pl_st_contrast, self.hci_hrs_obs.instrument.spec_reso, peak_correction_rate, SNR_RMS_mean, SNR_RMS_std, SNR_vs_NoiseLess_mean, SNR_vs_NoiseLess_std))
        return([peak_correction_rate, SNR_RMS_mean, SNR_RMS_std, SNR_vs_NoiseLess_mean, SNR_vs_NoiseLess_std])

    def removeSkyEmission(self):
        spec = self.hci_hrs_obs.obs_spec_resample.copy()
        spec.wavelength = self.hci_hrs_obs.atm_radi_spec_chunk_resample.wavelength
        spec.flux = self.hci_hrs_obs.obs_spec_resample.flux - self.hci_hrs_obs.atm_radi_spec_chunk_resample.flux
        spec.noise = None
        spec.addNoise(np.sqrt(self.hci_hrs_obs.obs_spec_resample.noise**2 + self.hci_hrs_obs.atm_radi_spec_chunk_resample.noise**2))
        return(spec)

    def removeSkyTransmissionStar(self):
        flx_st_atm = self.hci_hrs_obs.obs_st_resample.flux
        flx_obs = self.obs_emission_removed.flux
        wav_obs = self.obs_emission_removed.wavelength
        flx_st_atm_norm = flx_st_atm / np.median(flx_st_atm)
        obs_st_at_removed = self.hci_hrs_obs.obs_st_resample.copy()
        obs_st_at_removed.wavelength = wav_obs
        noise = np.sqrt((self.obs_emission_removed.noise / self.obs_emission_removed.flux)**2 + (self.hci_hrs_obs.obs_st_resample.noise / self.hci_hrs_obs.obs_st_resample.flux)**2) * self.obs_emission_removed.flux
        #obs_st_at_removed.flux = flx_obs / flx_st_atm_norm # neither division or subtraction cannot remove sky transmission and star absorption. This is potentially due to linear interpolation error in previous procedures. More precise interpolation may help but may not work in real observation. Therefore, I cheat here to assume that sky transmission and star absorption can somehow be removed and reveal planet signal, but I don't know exactly how. 
        obs_st_at_removed.flux = self.hci_hrs_obs.obs_pl_resample.flux
        obs_st_at_removed.noise = None
        obs_st_at_removed.addNoise(np.abs(noise))
        return(obs_st_at_removed)

    def plotObsTemplate(self, plotSkyStAtmRemoved=True, plotTemplate=True, plotStAtm=True, plotObs=False, plotObsSkyRemoved=True):
        flx_obs = self.hci_hrs_obs.obs_spec_resample.flux
        flx_obs_emission_removed = self.obs_emission_removed.flux
        flx_template = self.template.flux
        flx_st_atm = self.hci_hrs_obs.obs_st_resample.flux
        flx_obs_st_at_removed = self.obs_st_at_removed.flux
        wav_obs = self.hci_hrs_obs.obs_spec_resample.wavelength
        wav_template = self.template.wavelength
        if plotTemplate:
            plt.plot(wav_template, flx_template / np.median(flx_template), label="Template")
        if plotStAtm:
            plt.plot(wav_obs, flx_st_atm / np.median(flx_st_atm), label="Star Atm only")
        if plotObs:
            plt.plot(wav_obs, flx_obs / np.median(flx_obs), label="Observed")
        if plotObsSkyRemoved:
            plt.plot(wav_obs, flx_obs_emission_removed / np.median(flx_obs_emission_removed), label="Sky removed")
        if plotSkyStAtmRemoved:
            plt.plot(wav_obs, flx_obs_st_at_removed / np.median(flx_obs_st_at_removed), label="Sky Star removed")
        plt.ylim(np.min(flx_st_atm / np.median(flx_st_atm)), 2.0 * np.max(flx_st_atm / np.median(flx_st_atm)))
        plt.legend()
        plt.show()

    def getSpecChunk(self, wav, flx):
        # get spectrum within wavelength range
        idx = ((wav < self.hci_hrs_obs.wav_max) & (wav > self.hci_hrs_obs.wav_min))
        return {'Wavelength':wav[idx],'Flux':flx[idx]}

    def removeNanInSpecChunk(self, spectrum):
        idx = np.isnan(spectrum.flux)
        spectrum.flux[idx] = np.nanmedian(spectrum.flux)
        return(spectrum)

def readInit(init_file="MdwarfPlanet.init"):
    initDict = {}
    with open(init_file, 'r') as f:
        for line in f:
            key_value = line.split('#')[0]
            key = key_value.split(':')[0].strip(' \t\n\r')
            value = key_value.split(':')[1].strip(' \t\n\r')
            initDict[key] = value
    return(initDict)

def __main__():
    #initDict = readInit(init_file="MdwarfPlanet.init")
    initDict = readInit(init_file="SunEarth_4m.init")
    wav_min, wav_max, t_exp = np.float32(initDict["wav_min"]), np.float32(initDict["wav_max"]), np.float32(initDict["t_exp"])
    target_pl = Target(distance=np.float32(initDict["distance"]), spec_path=initDict["pl_spec_path"]+".102400", inclination_deg=np.float32(initDict["pl_inclination_deg"]), rotation_vel=np.float32(initDict["pl_rotation_vel"]), radial_vel=np.float32(initDict["pl_radial_vel"]), spec_reso=np.float32(initDict["spec_reso"]))
    target_st = Target(distance=np.float32(initDict["distance"]), spec_path=initDict["st_spec_path"]+".102400", inclination_deg=np.float32(initDict["st_inclination_deg"]), rotation_vel=np.float32(initDict["st_rotation_vel"]), radial_vel=np.float32(initDict["st_radial_vel"]), spec_reso=np.float32(initDict["spec_reso"]))
    wav_med = (wav_min + wav_max) / 2.0
    if initDict["spec_tran_path"] != "None":
        atmosphere = Atmosphere(spec_tran_path=initDict["spec_tran_path"], spec_radi_path=initDict["spec_radi_path"])
    else:
        atmosphere = None
    instrument = Instrument(wav_med, telescope_size=np.float32(initDict["telescope_size"]), pl_st_contrast=np.float32(initDict["pl_st_contrast"]), spec_reso=np.float32(initDict["spec_reso"]), read_noise=np.float32(initDict["read_noise"]), dark_current=np.float32(initDict["dark_current"]), fiber_size=np.float32(initDict["fiber_size"]), pixel_sampling=np.float32(initDict["pixel_sampling"]), throughput=np.float32(initDict["throughput"]), wfc_residual=np.float32(initDict["wfc_residual"]))
    hci_hrs = HCI_HRS_Observation(wav_min, wav_max, t_exp, target_pl, target_st, instrument, atmosphere=atmosphere)
    print("Star flux: {0} \nLeaked star flux: {1}\nPlanet flux: {2}\nPlanet flux per pixel: {3}\nSky flux: {4}\nSky transmission: {5}".format(hci_hrs.star_total_flux, hci_hrs.star_total_flux * instrument.pl_st_contrast, hci_hrs.planet_total_flux, hci_hrs.planet_total_flux / (len(hci_hrs.obs_spec_resample.flux) + 0.0), hci_hrs.sky_total_flux, hci_hrs.sky_transmission))
    spec = pyfits.open(initDict["template_path"]+".102400")
    template = Spectrum(spec[1].data["Wavelength"], spec[1].data["Flux"], spec_reso=np.float32(initDict["spec_reso"]))
    hci_hrs_red = HCI_HRS_Reduction(hci_hrs, template, save_flag=False, obj_tag=initDict["obj_tag"], template_tag=initDict["template_tag"],speckle_flag=True)

__main__()








