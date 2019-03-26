from lsst.sims.photUtils import SignalToNoise
from lsst.sims.photUtils import PhotometricParameters
from lsst.sims.photUtils import Bandpass, Sed
from sn_utils.utils.sn_throughputs import Throughputs

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.constants import *

# decorator to access parameters of the class


def get_val_decor(func):
    def func_deco(theclass, what, xlist):
        for x in xlist:
            if x not in theclass.data[what].keys():
                func(theclass, what, x)
    return func_deco


class Telescope(Throughputs):
    """ Telescope class
    inherits from Throughputs
    estimate quantities defined in LSE-40

    Input
    ---------
    set of parameters to
    - locate and load instrument files:
    through_dir: throughput dir
    default: LSST_THROUGHPUTS_BASELINE
    atmos_dir: dir of atmos files
    default: THROUGHPUTS_DIR
    telescope_files: list of throughput files
    filterlist: list of filters to consider
    wave_min, wave_max: min and max wavelength of throughput
    - set the type of throughput to estimate:
    atmos: =True if atmosphere to consider
    aerosol: = True if aerosol to consider
    - airmass : value of airmass (default: 1.)

    Returns
    ---------
    Accessible throughputs (per band, from Throughput class):
    lsst_system: system throughput (lens+mirrors+filters)
    lsst_atmos: lsst_system+atmosphere
    lsst_atmos_aerosol: lsst_system+atmosphere+aerosol

    The following parameters are estimated "on demand":
    mag_sky: sky magnitude
    m5: 5-sigma depth
    Sigmab: see eq. (36) of LSE-40
    zp: see eq. (43) of LSE-40
    counts_zp:
    Skyb: see eq. (40) of LSE-40
    flux_sky:
    """

    def __init__(self, name='unknown', airmass=1, **kwargs):
        self.name = name
        Throughputs.__init__(self, **kwargs)

        params = ['mag_sky', 'm5', 'FWHMeff', 'Tb',
                  'Sigmab', 'zp', 'counts_zp', 'Skyb', 'flux_sky']

        self.data = {}
        for par in params:
            self.data[par] = {}

        self.data['FWHMeff'] = dict(
            zip('ugrizy', [0.92, 0.87, 0.83, 0.80, 0.78, 0.76]))

        #self.atmos = atmos

        # self.Load_Atmosphere(airmass)

    @get_val_decor
    def get(self, what, band):

        filter_trans = self.system[band]
        wavelen_min, wavelen_max, wavelen_step = filter_trans.getWavelenLimits(
            None, None, None)

        bandpass = Bandpass(wavelen=filter_trans.wavelen, sb=filter_trans.sb)

        flatSedb = Sed()
        flatSedb.setFlatSED(wavelen_min, wavelen_max, wavelen_step)
        flux0b = np.power(10., -0.4*self.mag_sky(band))
        flatSedb.multiplyFluxNorm(flux0b)
        photParams = PhotometricParameters(bandpass=band)
        norm = photParams.platescale**2/2.*photParams.exptime/photParams.gain
        trans = filter_trans

        if self.atmos:
            trans = self.atmosphere[band]
        self.data['m5'][band] = SignalToNoise.calcM5(
            flatSedb, trans, filter_trans,
            photParams=photParams,
            FWHMeff=self.FWHMeff(band))
        adu_int = flatSedb.calcADU(bandpass=trans, photParams=photParams)
        self.data['flux_sky'][band] = adu_int*norm

    @get_val_decor
    def get_inputs(self, what, band):

        myup = self.Calc_Integ_Sed(self.darksky, self.system[band])
        self.data['Tb'][band] = self.Calc_Integ(self.atmosphere[band])
        self.data['Sigmab'][band] = self.Calc_Integ(self.system[band])
        self.data['mag_sky'][band] = -2.5 * \
            np.log10(myup/(3631.*self.Sigmab(band)))

    @get_val_decor
    def get_zp(self, what, band):
        """ get zero points
        formula used here are extracted from LSE-40
        """
        photParams = PhotometricParameters(bandpass=band)
        Diameter = 2.*np.sqrt(photParams.effarea*1.e-4 /
                              np.pi)  # diameter in meter
        Cte = 3631.*np.pi*Diameter**2*2.*photParams.exptime/4/h/1.e36

        self.data['Skyb'][band] = Cte*np.power(Diameter/6.5, 2.)\
            * np.power(2.*photParams.exptime/30., 2.)\
            * np.power(photParams.platescale, 2.)\
            * 10.**0.4*(25.-self.mag_sky(band))\
            * self.Sigmab(band)

        Zb = 181.8*np.power(Diameter/6.5, 2.)*self.Tb(band)
        mbZ = 25.+2.5*np.log10(Zb)
        filtre_trans = self.system[band]
        wavelen_min, wavelen_max, wavelen_step = filtre_trans.getWavelenLimits(
            None, None, None)
        bandpass = Bandpass(wavelen=filtre_trans.wavelen, sb=filtre_trans.sb)
        flatSed = Sed()
        flatSed.setFlatSED(wavelen_min, wavelen_max, wavelen_step)
        flux0 = np.power(10., -0.4*mbZ)
        flatSed.multiplyFluxNorm(flux0)
        photParams = PhotometricParameters(bandpass=band)
        # number of counts for exptime
        counts = flatSed.calcADU(bandpass, photParams=photParams)
        self.data['zp'][band] = mbZ
        self.data['counts_zp'][band] = counts/2.*photParams.exptime

    def return_value(self, what, band):
        if len(band) > 1:
            return self.data[what]
        else:
            return self.data[what][band]

    def m5(self, filtre):
        self.get('m5', filtre)
        return self.return_value('m5', filtre)

    def Tb(self, filtre):
        self.get_inputs('Tb', filtre)
        return self.return_value('Tb', filtre)

    def mag_sky(self, filtre):
        self.get_inputs('mag_sky', filtre)
        return self.return_value('mag_sky', filtre)

    def Sigmab(self, filtre):
        self.get_inputs('Sigmab', filtre)
        return self.return_value('Sigmab', filtre)

    def zp(self, filtre):
        self.get_zp('zp', filtre)
        return self.return_value('zp', filtre)

    def FWHMeff(self, filtre):
        return self.return_value('FWHMeff', filtre)

    def Calc_Integ(self, bandpass):
        """ integration over bandpass
        """
        resu = 0.
        dlam = 0
        for i, wave in enumerate(bandpass.wavelen):
            if i < len(bandpass.wavelen)-1:
                dlam = bandpass.wavelen[i+1]-wave
                resu += dlam*bandpass.sb[i]/wave
            # resu+=dlam*bandpass.sb[i]

        return resu

    def Calc_Integ_Sed(self, sed, bandpass, wavelen=None, fnu=None):
        """ SED integration
        """
        use_self = sed._checkUseSelf(wavelen, fnu)
        # Use self values if desired, otherwise use values passed to function.
        if use_self:
            # Calculate fnu if required.
            if sed.fnu is None:
                # If fnu not present, calculate. (does not regrid).
                sed.flambdaTofnu()
            wavelen = sed.wavelen
            fnu = sed.fnu
        # Make sure wavelen/fnu are on the same wavelength grid as bandpass.
        wavelen, fnu = sed.resampleSED(
            wavelen, fnu, wavelen_match=bandpass.wavelen)

        # Calculate the number of photons.
        nphoton = (fnu / wavelen * bandpass.sb).sum()
        dlambda = wavelen[1] - wavelen[0]
        return nphoton * dlambda

    def flux_to_mag(self, flux, band, zp=None):
        """ conversion flux->mag
        """
        if zp is None:
            zp = self.zero_points(band)
        # print 'zp',zp,band
        m = -2.5 * np.log10(flux) + zp
        return m

    def mag_to_flux(self, mag, band, zp=None):
        """conversion mag->flux
        """
        if zp is None:
            zp = self.zero_points(band)
        return np.power(10., -0.4 * (mag-zp))

    def zero_points(self, band):
        return np.asarray([self.zp[b] for b in band])

    def mag_to_flux_e_sec(self, mag, band, exptime):
        """ mag to flux (in photoelec/sec)
        """
        if not hasattr(mag, '__iter__'):
            wavelen_min, wavelen_max, wavelen_step = self.atmosphere[band].getWavelenLimits(
                None, None, None)
            sed = Sed()
            sed.setFlatSED()
            flux0 = 3631.*10**(-0.4*mag)  # flux in Jy
            flux0 = sed.calcFluxNorm(mag, self.atmosphere[band])
            sed.multiplyFluxNorm(flux0)
            photParams = PhotometricParameters(nexp=exptime/15.)
            counts = sed.calcADU(
                bandpass=self.atmosphere[band], photParams=photParams)
            e_per_sec = counts
            e_per_sec /= exptime/photParams.gain
            # print('hello',photParams.gain,exptime)
            return counts, e_per_sec
        else:
            return np.asarray([self.mag_to_flux_e_sec(m, b, expt) for m, b, expt in zip(mag, band, exptime)])

    def gamma(self, mag, band, exptime):

        if not hasattr(mag, '__iter__'):
            photParams = PhotometricParameters(nexp=exptime/15.)
            counts, e_per_sec = self.mag_to_flux_e_sec(mag, band, exptime)
            return 0.04-1./(photParams.gain*counts)
        else:
            return np.asarray([self.gamma(m, b, e) for m, b, e in zip(mag, band, exptime)])
