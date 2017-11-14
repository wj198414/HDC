import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import astropy.io.ascii as ascii
import scipy.constants
import pickle
from scipy import signal
from astropy import units as u
from astropy import constants as c
import numpy.fft as fft
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from scipy import interpolate
import time
import scipy.interpolate
import scipy.fftpack as fp
from datetime import datetime

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

    def calcSNRrmsNoiseless(self, ccf_noise_less, peak=None):
        cc = self.ccf
        cc_subtracted = cc - ccf_noise_less.ccf

        ind_max = np.argmax(cc)
        num = len(cc)
        snr = np.max([cc[ind_max] / np.std(cc_subtracted[0:int(num / 4.0)]), cc[ind_max] / np.std(cc_subtracted[-int(num / 4.0):-1])])
        if not (peak is None):
            snr = np.max([peak / np.std(cc_subtracted[0:int(num / 4.0)]), peak / np.std(cc_subtracted[-int(num / 4.0):-1])])
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
