from astropy import (units as u, constants as const)
from astropy.cosmology import FlatLambdaCDM
import numpy as np

STERADIAN2SQDEG = 180.**2 / np.pi**2
# Mpc^3 -> Mpc^3/sr
norm = 1. / (4. * np.pi)


class SN_Rate:
    """ 
    Estimate production rates of typeIa SN

    Available rates: Ripoche, Perrett, Dilday

    Parameters
    ---------------
    rate :  str,opt
      type of rate chosen (Ripoche, Perrett, Dilday) (default : Perrett)
    H0 : float, opt
       Hubble constant value :math:`H_{0}`(default : 70.)
    H0_err: float, opt
       H0 error (default: 2)
    Om0 : float, opt
        matter density value :math:`\Omega_{0}`  (default : 0.25)
    Om0_err: float, opt
       Om0 error (default: 0.025)
    min_rf_phase : float, opt
       min rest-frame phase (default : -15.)
    max_rf_phase : float, opt
       max rest-frame phase (default : 30.)
    error_params: bool, opt
      to propagate parameter errors (default: False)
    """

    def __init__(self, rate='Perrett', H0=70., H0_err=2.,
                 Om0=0.25, Om0_err=0.025,
                 min_rf_phase=-15., max_rf_phase=30.,
                 error_params=False):

        self.H0 = H0
        self.H0_err = H0_err
        self.Om0 = Om0
        self.Om0_err = Om0_err
        self.astropy_cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        self.rate = rate
        self.min_rf_phase = min_rf_phase
        self.max_rf_phase = max_rf_phase
        self.error_params = error_params

    def __call__(self, zmin=0.1, zmax=0.2,
                 dz=0.01, survey_area=9.6,
                 bins=None, account_for_edges=False,
                 duration=140., duration_z=None):
        """
        call method

        Parameters
        ----------------
        zmin : float, opt
          minimal redshift (default : 0.1)
        zmax : float,opt
           max redshift (default : 0.2)
        dz : float, opt
           redshift bin (default : 0.001)
        survey_area : float, opt
           area of the survey (:math:`deg^{2}`) (default : 9.6 :math:`deg^{2}`)
        bins : list(float), opt
          redshift bins (default : None)
        account_for_edges : bool
          to account for season edges. If true, duration of the survey will be reduced by (1+z)*(maf_rf_phase-min_rf_phase)/365.25 (default : False)
        duration : float, opt
           survey duration (in days) (default : 140 days)
        duration_z : list(float),opt
          survey duration (as a function of z) (default : None)

        Returns
        -----------
        Lists :
        zz : float
           redshift values
        rate : float
           production rate
        err_rate : float
           production rate error
        nsn : float
           number of SN
        err_nsn : float 
           error on the number of SN
        """

        if bins is None:
            thebins = np.arange(zmin, zmax+dz, dz)
            zz = 0.5 * (thebins[1:] + thebins[:-1])
        else:
            zz = bins
            thebins = bins

        rate, err_rate = self.SNRate(zz)
        error_rel = err_rate/rate

        area = survey_area / STERADIAN2SQDEG
        # or area= self.survey_area/41253.

        dvol = norm*self.astropy_cosmo.comoving_volume(thebins).value
        dvol = dvol[1:] - dvol[:-1]

        if account_for_edges:
            margin = (1.+zz) * (self.max_rf_phase-self.min_rf_phase) / 365.25
            effective_duration = duration / 365.25 - margin
            effective_duration[effective_duration <= 0.] = 0.
        else:
            # duration in days!
            effective_duration = duration/365.25
            if duration_z is not None:
                effective_duration = duration_z(zz)/365.25

        normz = (1.+zz)
        nsn = rate * area * dvol * effective_duration / normz
        err_rate = err_rate*area * dvol * effective_duration / normz

        err_dvol = 0.
        if self.error_params:
            dvol_err = self.comoving_volume_error(norm, thebins)
            err_dvol = rate * area * dvol_err * effective_duration / normz

        err_nsn = np.sqrt(err_rate**2+err_dvol**2)
        return zz, rate, err_rate, nsn, err_nsn

    def comoving_volume_error(self, norm, thebins):
        """
        Method to estimate the comoving volume error

        Parameters
        ---------------
        norm: float
          normalisation factor
        thebins: array
          redshift bins

        Returns
        ----------
        the comoving volume error

        """

        dx = 1.e-8

        dvol = {}

        for sign in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
            astropy_cosmo = FlatLambdaCDM(
                H0=self.H0+sign[0]*dx, Om0=self.Om0+sign[1]*dx)
            tag = 'H0_{}_Om0_{}'.format(sign[0], sign[1])
            dvol[tag] = norm*astropy_cosmo.comoving_volume(thebins).value

        for key in dvol.keys():
            dvol[key] = dvol[key][1:]-dvol[key][:-1]

        dvol_err_H0 = (dvol['H0_1_Om0_0']-dvol['H0_-1_Om0_0'])/(2.*dx)
        dvol_err_Om0 = (dvol['H0_0_Om0_1']-dvol['H0_0_Om0_-1'])/(2.*dx)

        dvol_err = (dvol_err_H0*self.H0_err)**2
        dvol_err += (dvol_err_Om0*self.Om0_err)**2

        return np.sqrt(dvol_err)

    def RipocheRate(self, z):
        """The SNLS SNIa rate according to the (unpublished) Ripoche et al study.

        Parameters
        --------------
        z : float
          redshift

        Returns
        ----------
        rate : float
        error_rate : float
        """
        rate = 1.53e-4*0.343
        expn = 2.14
        my_z = np.copy(z)
        my_z[my_z > 1.] = 1.
        rate_sn = rate * np.power((1+my_z)/1.5, expn)
        return rate_sn, 0.2*rate_sn

    def PerrettRate(self, z):
        """The SNLS SNIa rate according to (Perrett et al, 201?) 

        Parameters
        --------------
        z : float
          redshift

        Returns
        ----------
        rate : float
        error_rate : float
        """
        rate = 0.17E-4
        expn = 2.11
        err_rate = 0.03E-4
        err_expn = 0.28
        my_z = np.copy(z)
        rate_sn = rate * np.power(1+my_z, expn)
        err_rate_sn = np.power(1+my_z, 2.*expn)*np.power(err_rate, 2.)
        err_rate_sn += np.power(rate_sn*np.log(1+my_z)*err_expn, 2.)

        return rate_sn, np.power(err_rate_sn, 0.5)

    def DildayRate(self, z):
        """The Dilday rate according to

         Parameters
        --------------
        z : float
          redshift

        Returns
        ----------
        rate : float
        error_rate : float
        """

        rate = 2.6e-5
        expn = 1.5
        err_rate = 0.01
        err_expn = 0.6
        my_z = np.copy(z)
        my_z[my_z > 1.] = 1.
        rate_sn = rate * np.power(1+my_z, expn)
        err_rate_sn = rate_sn*np.log(1+my_z)*err_expn
        return rate_sn, err_rate_sn

    """
    def flat_rate(self, z):
        return 1., 0.1
    """

    def SNRate(self, z):
        """SN rate estimation

        Parameters
        --------------
        z : float
          redshift

        Returns
        ----------
        rate : float
        error_rate : float
        """
        if self.rate == 'Ripoche':
            return self.RipocheRate(z)
        if self.rate == 'Perrett':
            return self.PerrettRate(z)
        if self.rate == 'Dilday':
            return self.DildayRate(z)
        """
        if self.rate == 'Flat':
            return self.flat_rate(z)
       """

    """
    def N_SN(self, z, survey_area):

        rate, err_rate = self.SNRate(z)

        area = survey_area / STERADIAN2SQDEG
        vol = self.astropy_cosmo.comoving_volume(z).value
        duration = self.duration
        nsn = norm*rate*area*vol*duration/(1.+z)
        err_nsn = err_rate*norm*area*vol*duration/(1.+z)

        return nsn, err_nsn
    """

    def PlotNSN(self, zmin=0.1, zmax=0.2,
                dz=0.01, survey_area=9.6,
                bins=None, account_for_edges=False,
                duration=140., duration_z=None, norm=False):
        """ Plot integrated number of supernovae as a function of redshift
        uses the __call__ function

        Parameters
        --------------
        zmin : float, opt
          minimal redshift (default : 0.1)
        zmax : float,opt
           max redshift (default : 0.2)
        dz : float, opt
           redshift bin (default : 0.001)
        survey_area : float, opt
           area of the survey (:math:`deg^{2}`) (default : 9.6 :math:`deg^{2}`)
        bins : list(float), opt
          redshift bins (default : None)
        account_for_edges : bool
          to account for season edges. If true, duration of the survey will be reduced by (1+z)*(maf_rf_phase-min_rf_phase)/365.25 (default : False)
        duration : float, opt
           survey duration (in days) (default : 140 days)
        duration_z : list(float),opt
          survey duration (as a function of z) (default : None)
        norm: bool, opt
          to normalise the results (default: False)

        """
        import pylab as plt

        zz, rate, err_rate, nsn, err_nsn = self.__call__(
            zmin=zmin, zmax=zmax, dz=dz, bins=bins,
            account_for_edges=account_for_edges,
            duration=duration, survey_area=survey_area)

        nsn_sum = np.cumsum(nsn)

        if norm is False:
            plt.errorbar(zz, nsn_sum, yerr=np.sqrt(np.cumsum(err_nsn**2)))
        else:
            plt.errorbar(zz, nsn_sum/nsn_sum[-1])
        plt.xlabel('z')
        plt.ylabel('N$_{SN}$ <')
        plt.grid()


class NSN:
    """
    class to estimate the number of supernovae from rate

    Parameters
    ---------------
    H0: float, opt
      Hubble cte (default: 70)
    Om0: float, opt
      Omega_m value (default: 0.3)
    min_rf_phase: float, opt
      min rest-frame phase for nsn estimation (default: -15.)
    max_rf_phase: float, opt
      max rest-frame phase for nsn estimation (default: 30à

    """

    def __init__(self, H0=70., Om0=0.3,
                 min_rf_phase=-15., max_rf_phase=30.):

        self.H0 = H0
        self.Om0 = Om0
        self.min_rf_phase = min_rf_phase
        self.max_rf_phase = max_rf_phase

        self.rateSN = SN_Rate(H0=self.H0, Om0=self.Om0,
                              min_rf_phase=self.min_rf_phase, max_rf_phase=self.max_rf_phase)

    def __call__(self, zmin, zmax, dz, season_length, survey_area, account_for_edges=True, scale_factor=1):
        """
        Method to estimate the number of supernovae

        Parameters
        ----------------
        zmin: float
          min redshift
        zmax: float
          max redshift
        dz: float
          redshift binning value
        season_length: float
          season length (in days)
        survey_area: float
          area of the survey (in deg2)
        account_for_edges: bool, opt
          to account for edges when estimating nsn (default: True)
        scale_factor: float, opt
          scale factor for the number of sn (default: 1)

        Returns
        ----------
        The number of supernovae

        """
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=zmin,
                                                       zmax=zmax,
                                                       dz=dz,
                                                       duration=season_length,
                                                       survey_area=survey_area,
                                                       account_for_edges=account_for_edges)

        res = scale_factor*np.cumsum(nsn)[-1]

        return res
