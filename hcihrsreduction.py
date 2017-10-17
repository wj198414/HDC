import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from datetime import datetime
from crosscorrelationfunction import CrossCorrelationFunction

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
            #obs_norm = self.getStarNorm(self.hci_hrs_obs.obs_st_resample.flux, long_array=True)
            #obs_norm = obs_norm.value / np.median(obs_norm.value) * np.median(self.obs_st_at_removed.flux)
            obs_norm = np.median(self.obs_st_at_removed.flux)
            self.obs_st_at_removed.flux = self.obs_st_at_removed.flux / obs_norm
            self.obs_st_at_removed.noise = self.obs_st_at_removed.noise / obs_norm
            #mask_arr = np.where((self.template_resample.flux / np.nanmedian(self.template_resample.flux)) > 0.99)
            mask_arr = np.where((self.template_resample.flux / np.nanmedian(self.template_resample.flux)) > 1e9)
            #plt.errorbar(self.obs_st_at_removed.wavelength, self.obs_st_at_removed.flux, yerr=self.obs_st_at_removed.noise)
            plt.figure()
            plt.errorbar(self.obs_st_at_removed.wavelength, self.obs_st_at_removed.flux, self.obs_st_at_removed.noise)
            #plt.plot(self.template_resample.wavelength, self.template_resample.flux / np.median(self.template_resample.flux))
            plt.plot(self.template_resample.wavelength[mask_arr], self.template_resample.flux[mask_arr] / np.median(self.template_resample.flux[mask_arr]))
            plt.plot(self.obs_st_at_removed.wavelength[mask_arr], self.obs_st_at_removed.flux[mask_arr], "b.")
            plt.show()
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
            plt.figure()
            plt.plot(result["CCF"].vel, result["CCF"].ccf, "bo-")
            plt.plot(self.ccf_noise_less.vel, self.ccf_noise_less.ccf, "r")
            plt.show()
            plt.figure()
            plt.plot(result["CCF"].vel, result["CCF"].ccf - self.ccf_noise_less.ccf, "bo-")
            plt.show()            
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

    #Noise model and final observed spectrum made more realistic.  Can't easily remove thermal+zodi backgrounds!
        
    def removeSkyTransmissionStar(self):
        flx_st_atm = self.hci_hrs_obs.obs_st_resample.flux
        flx_obs = self.obs_emission_removed.flux
        wav_obs = self.obs_emission_removed.wavelength
        flx_st_atm_norm = flx_st_atm / np.median(flx_st_atm)
        obs_st_at_removed = self.hci_hrs_obs.obs_st_resample.copy()
        obs_st_at_removed.wavelength = wav_obs
        #noise = np.sqrt((self.obs_emission_removed.noise / self.obs_emission_removed.flux)**2 + (self.hci_hrs_obs.obs_st_resample.noise / self.hci_hrs_obs.obs_st_resample.flux)**2 + (self.hci_hrs_obs.obs_therm_resample.noise/self.hci_hrs_obs.obs_therm_resample.flux)**2) * self.obs_emission_removed.flux
        self.hci_hrs_obs.obs_st_resample.flux *= self.hci_hrs_obs.instrument.pl_st_contrast
        noise = np.sqrt(self.obs_emission_removed.noise**2 + self.hci_hrs_obs.calNoise(self.hci_hrs_obs.obs_st_resample)**2)
        #obs_st_at_removed.flux = flx_obs / flx_st_atm_norm # neither division or subtraction cannot remove sky transmission and star absorption. This is potentially due to linear interpolation error in previous procedures. More precise interpolation may help but may not work in real observation. Therefore, I cheat here to assume that sky transmission and star absorption can somehow be removed and reveal planet signal, but I don't know exactly how. 
        #obs_st_at_removed.flux = self.hci_hrs_obs.obs_pl_resample.flux
        obs_st_at_removed.flux = self.obs_emission_removed.flux - self.hci_hrs_obs.obs_st_resample.flux*self.hci_hrs_obs.instrument.pl_st_contrast
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