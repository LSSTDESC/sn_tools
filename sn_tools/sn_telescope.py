from rubin_sim.phot_utils import signaltonoise
from rubin_sim.phot_utils import photometric_parameters
from rubin_sim.phot_utils import Bandpass, Sed
from sn_tools.sn_throughputs import Throughputs

import numpy as np
# import matplotlib.pyplot as plt
# import math
from scipy.constants import *
from functools import wraps

# decorator to access parameters of the class


def get_val_decor(func):
    @wraps(func)
    def func_deco(theclass, what, xlist):
        for x in xlist:
            if x not in theclass.data[what].keys():
                func(theclass, what, x)
    return func_deco


class Telescope(Throughputs):
    """ Telescope class
    inherits from Throughputs
    estimate quantities defined in LSE-40

    The following quantities are accessible:

    mag_sky: sky magnitude

    m5: 5-sigma depth

    Sigmab: see eq. (36) of LSE-40

    zp: see eq. (43) of LSE-40

    counts_zp:

    Skyb: see eq. (40) of LSE-40

    flux_sky:


    Parameters
    -------------
    through_dir : str, opt
       throughput directory
       Default : LSST_THROUGHPUTS_BASELINE
    atmos_dir : str, opt
       directory of atmos files
       Default : THROUGHPUTS_DIR
    telescope_files : list(str),opt
       list of of throughput files
       Default : ['detector.dat', 'lens1.dat','lens2.dat',
           'lens3.dat','m1.dat', 'm2.dat', 'm3.dat']
    filterlist: list(str), opt
       list of filters to consider
       Default : 'ugrizy'
    wave_min : float, opt
        min wavelength for throughput
        Default : 300
    wave_max : float, opt
        max wavelength for throughput
        Default : 1150
    atmos : bool, opt
         to include atmosphere affects
         Default : True
    aerosol : bool, opt
         to include aerosol effects
         Default : True
    airmass : float, opt
         airmass value
         Default : 1.

    Returns
    ---------
    Accessible throughputs (per band, from Throughput class):
    lsst_system: system throughput (lens+mirrors+filters)
    lsst_atmos: lsst_system+atmosphere
    lsst_atmos_aerosol: lsst_system+atmosphere+aerosol


    """

    def __init__(self, name='unknown', airmass=1., **kwargs):
        self.name = name
        Throughputs.__init__(self, **kwargs)

        params = ['mag_sky', 'm5', 'FWHMeff', 'Tb',
                  'Sigmab', 'zp', 'counts_zp', 'Skyb', 'flux_sky']

        self.data = {}
        for par in params:
            self.data[par] = {}

        self.data['FWHMeff'] = dict(
            zip('ugrizy', [0.92, 0.87, 0.83, 0.80, 0.78, 0.76]))

        # self.atmos = atmos

        self.Load_Atmosphere(airmass)

    @get_val_decor
    def get(self, what, band):
        """
        Decorator to access quantities

        Parameters
        ---------------
        what: str
          parameter to estimate
        band: str
          filter

        """
        filter_trans = self.system[band]
        wavelen_min, wavelen_max, wavelen_step = \
            filter_trans.get_wavelen_limits(None, None, None)

        # bpass = Bandpass(wavelen=filter_trans.wavelen, sb=filter_trans.sb)

        flatSedb = Sed()
        flatSedb.setFlatSED(wavelen_min, wavelen_max, wavelen_step)
        flux0b = np.power(10., -0.4*self.mag_sky(band))
        flatSedb.multiply_flux_norm(flux0b)
        photParams = photometric_parameters.PhotometricParameters(
            bandpass=band)
        norm = photParams.platescale**2/2.*photParams.exptime/photParams.gain
        trans = filter_trans

        if self.atmos:
            trans = self.atmosphere[band]
        self.data['m5'][band] = signaltonoise.calcM5(
            flatSedb, trans, filter_trans,
            photParams=photParams,
            FWHMeff=self.FWHMeff(band))
        adu_int = flatSedb.calc_adu(bandpass=trans, phot_params=photParams)
        self.data['flux_sky'][band] = adu_int*norm

    @get_val_decor
    def get_inputs(self, what, band):
        """
        decorator to access Tb, Sigmab, mag_sky

        Parameters
        ---------------
        what: str
          parameter to estimate
        band: str
          filter

        """
        myup = self.Calc_Integ_Sed(self.darksky, self.system[band])
        bpass = self.atmosphere[band]
        if self.aerosol_b:
            bpass = self.aerosol[band]
        self.data['Tb'][band] = self.Calc_Integ(bpass)
        self.data['Sigmab'][band] = self.Calc_Integ(self.system[band])
        self.data['mag_sky'][band] = -2.5 * \
            np.log10(myup/(3631.*self.Sigmab(band)))

    @get_val_decor
    def get_zp(self, what, band):
        """
        decorator get zero points
        formula used here are extracted from LSE-40

        Parameters
        ---------------
        band: str
          filter

        """
        photParams = photometric_parameters.PhotometricParameters(
            bandpass=band)
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
        """
        filtre_trans = self.atmosphere[band]
        if self.atmos:
            filtre_trans = self.atmosphere[band]
        if self.aerosol_b:
            filtre_trans = self.aerosol[band]
        wavelen_min, wavelen_max, wavelen_step = \
            filtre_trans.get_wavelen_limits(None, None, None)
        """
        filtre_trans = self.lsst_system[band]
        wavelen_min, wavelen_max, wavelen_step = \
            filtre_trans.get_wavelen_limits(None, None, None)
        bpass = Bandpass(wavelen=filtre_trans.wavelen, sb=filtre_trans.sb)
        flatSed = Sed()
        flatSed.set_flat_sed(wavelen_min, wavelen_max, wavelen_step)
        flux0 = np.power(10., -0.4*mbZ)
        flatSed.multiply_flux_norm(flux0)
        photParams = photometric_parameters.PhotometricParameters(
            bandpass=band)
        # number of counts for exptime
        counts = flatSed.calc_adu(bpass, phot_params=photParams)
        self.data['zp'][band] = mbZ
        self.data['counts_zp'][band] = counts/photParams.exptime

    def return_value(self, what, band):
        """
        accessor

        Parameters
        ---------------
        what: str
          parameter to estimate
        band: str
          filter

        """
        if len(band) > 1:
            return self.data[what]
        else:
            return self.data[what][band]

    def m5(self, filtre):
        """m5 accessor
        """
        self.get('m5', filtre)
        return self.return_value('m5', filtre)

    def Tb(self, filtre):
        """Tb accessor
        """
        self.get_inputs('Tb', filtre)
        return self.return_value('Tb', filtre)

    def mag_sky(self, filtre):
        """mag_sky accessor
        """
        self.get_inputs('mag_sky', filtre)
        return self.return_value('mag_sky', filtre)

    def Sigmab(self, filtre):
        """
        Sigmab accessor

        Parameters
        ----------------
        band: str
          filter

        """
        self.get_inputs('Sigmab', filtre)
        return self.return_value('Sigmab', filtre)

    def zp(self, filtre):
        """
        zp accessor

        Parameters
        ----------------
        band: str
          filter

        """
        self.get_zp('zp', filtre)
        return self.return_value('zp', filtre)

    def counts_zp(self, filtre):
        """
        counts_zp accessor

        Parameters
        ----------
        filtre : str
            filter to consider.

        Returns
        -------
        None.

        """
        self.get_zp('zp', filtre)
        return self.return_value('counts_zp', filtre)

    def FWHMeff(self, filtre):
        """
        FWHMeff accessor

        Parameters
        ----------------
        band: str
          filter
        """
        return self.return_value('FWHMeff', filtre)

    def Calc_Integ(self, bandpass):
        """
        integration over bandpass

        Parameters
        --------------
        bandpass : float

        Returns
        ---------
        integration

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
        """
        SED integration

        Parameters
        --------------
        sed : float
          sed to integrate
        bandpass : float
          bandpass
        wavelength : float, opt
          wavelength values
           Default : None
        fnu : float, opt
           fnu values
           Default : None

        Returns
        ----------
        integrated sed over the bandpass

        """
        use_self = sed._check_use_self(wavelen, fnu)
        # Use self values if desired, otherwise use values passed to function.
        if use_self:
            # Calculate fnu if required.
            if sed.fnu is None:
                # If fnu not present, calculate. (does not regrid).
                sed.flambda_tofnu()
            wavelen = sed.wavelen
            fnu = sed.fnu
        # Make sure wavelen/fnu are on the same wavelength grid as bandpass.
        wavelen, fnu = sed.resample_sed(
            wavelen, fnu, wavelen_match=bandpass.wavelen)

        # Calculate the number of photons.
        nphoton = (fnu / wavelen * bandpass.sb).sum()
        dlambda = wavelen[1] - wavelen[0]
        return nphoton * dlambda

    def flux_to_mag(self, flux, band, zp=None):
        """
        Flux to magnitude conversion

        Parameters
        --------------
        flux : float
          input fluxes
        band : str
           input band
        zp : float, opt
           zeropoints
           Default : None

        Returns
        ---------
        magnitudes

        """
        if zp is None:
            zp = self.zero_points(band)
        # print 'zp',zp,band
        m = -2.5 * np.log10(flux) + zp
        return m

    def mag_to_flux(self, mag, band, zp=None):
        """
        Magnitude to flux conversion

        Parameters
        --------------
        mag : float
          input mags
        band : str
           input band
        zp : float, opt
           zeropoints
           Default : None

        Returns
        ---------
        fluxes

        """
        if zp is None:
            zp = self.zero_points(band)
        return np.power(10., -0.4 * (mag-zp.item()))

    def zero_points(self, band):
        """
        Zero points estimation

        Parameters
        --------------
        band : list(str)
          list of bands

        Returns
        ---------
        array of zp

        """
        return np.asarray([self.zp(b) for b in band])

    def mag_to_flux_e_sec(self, mag, band, exptime, nexp):
        """
        Mag to flux (in photoelec/sec) conversion

        Parameters
        --------------
        mag : float
          input magnitudes
        band : str
          input bands
        exptime : float
          input exposure times
        nexp: int
          number of exposures

        Returns
        ----------
        counts : float
           number of ADU counts
        e_per_sec : float
           flux in photoelectron per sec.

        """
        if not hasattr(mag, '__iter__'):
            wavelen_min, wavelen_max, wavelen_step = \
                self.atmosphere[band].get_avelen_limits(
                    None, None, None)
            sed = Sed()
            sed.setFlatSED()

            flux0 = sed.calcFluxNorm(mag, self.atmosphere[band])
            sed.multiply_flux_norm(flux0)

            photParams = \
                photometric_parameters.PhotometricParameters(
                    exptime=exptime, nexp=nexp)

            counts = sed.calc_adu(
                bandpass=self.atmosphere[band], phot_params=photParams)
            e_per_sec = counts

            # counts per sec
            e_per_sec /= (exptime*nexp)
            # conversion to pe
            e_per_sec *= photParams.gain
            return counts, e_per_sec
        else:
            r = []
            for m, b, expt, nexpos in zip(mag, band, exptime, nexp):
                counts, flux_e = self.mag_to_flux_e_sec(m, b, expt, nexpos)
                r.append((counts, flux_e))
            return np.asarray(r)

    def gamma(self, mag, band, exptime, nexp):
        """
        gamma parameter estimation

        cf eq(5) of the paper LSST :
            from science drivers to reference design
            and anticipated data products

        with sigma_rand = 0.2 and m=m5

        Parameters
        --------------
        mag : float
          magnitudes
        band : str
          band
        exptime : float
          exposure time

        Returns
        ----------
        gamma, mag_to_flux (float)

        """

        if not hasattr(mag, '__iter__'):
            photParams = \
                photometric_parameters.PhotometricParameters(
                    nexp=nexp, exptime=exptime)
            counts, e_per_sec = self.mag_to_flux_e_sec(
                mag, band, exptime, nexp)
            gamma = 0.04-1./(photParams.gain*counts)
            return gamma, e_per_sec
        else:
            r = []
            for m, b, e, nexpo in zip(mag, band, exptime, nexp):
                gamma, flux_e = self.gamma(m, b, e, nexpo)
                r.append((gamma, flux_e))
            return np.asarray(r)
