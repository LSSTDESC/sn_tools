from astropy import (units as u, constants as const)
from astropy.cosmology import FlatLambdaCDM
import numpy as np

STERADIAN2SQDEG = 180.**2 / np.pi**2
# Mpc^3 -> Mpc^3/sr
norm = 1. / (4. * np.pi)


class SN_Rate:
    """ class SNRate
    Estimate production rates of typeIa SN
    Available rates: Ripoche, Perrett,Dilday

    Input
    ---------
    Rate type, cosmology (H0, Om0)
    min and max rf phases

    Returns (call)
    ---------
    Lists:
    zz: redshift
    rate: production rate
    err_rate: production rate error
    nsn: number of SN
    err_nsn: error on nsn
    """

    def __init__(self, rate='Ripoche', H0=70, Om0=0.25,
                 min_rf_phase=-15., max_rf_phase=30.):

        self.astropy_cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        self.rate = rate
        self.min_rf_phase = min_rf_phase
        self.max_rf_phase = max_rf_phase

    def __call__(self, zmin=0.1, zmax=0.2,
                 dz=0.01, survey_area = 9.6,
                 bins=None, account_for_edges=False,
                 duration=140., duration_z=None):
        """
        Input
        ---------
        Redshift limit and bin
        account for edges:
        duration: duration of obs (z indep.)
        duration_z: duration os obs (z dep.)

        Returns (call)
        ---------
        Lists:
        zz: redshift
        rate: production rate
        err_rate: production rate error
        nsn: number of SN
        err_nsn: error on nsn
        """

        if bins is None:
            thebins = np.arange(zmin, zmax+dz, dz)
            zz = 0.5 * (thebins[1:] + thebins[:-1])
        else:
            zz = bins
            thebins = bins

        rate, err_rate = self.sn_rate(zz)
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
        err_nsn = err_rate*area * dvol * effective_duration / normz

        return zz, rate, err_rate, nsn, err_nsn

    def ripoche_rate(self, z):
        """The SNLS SNIa rate according to the (unpublished) Ripoche et al study.
        """
        rate = 1.53e-4*0.343
        expn = 2.14
        my_z = np.copy(z)
        my_z[my_z > 1.] = 1.
        rate_sn = rate * np.power((1+my_z)/1.5, expn)
        return rate_sn, 0.2*rate_sn

    def perrett_rate(self, z):
        """The SNLS SNIa rate according to (Perrett et al, 201?)
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

    def dilday_rate(self, z):
        """The Dilday rate according to
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

    def flat_rate(self, z):
        return 1., 0.1

    def sn_rate(self, z):
        if self.rate == 'Ripoche':
            return self.ripoche_rate(z)
        if self.rate == 'Perrett':
            return self.perrett_rate(z)
        if self.rate == 'Dilday':
            return self.dilday_rate(z)
        """
        if self.rate == 'Flat':
            return self.flat_rate(z)
        """

    def N_SN(self, z, survey_area):

        rate, err_rate = self.sn_rate(z)

        area = survey_area / STERADIAN2SQDEG
        vol = self.astropy_cosmo.comoving_volume(z).value
        duration = self.duration
        nsn = norm*rate*area*vol*duration/(1.+z)
        err_nsn = err_rate*norm*area*vol*duration/(1.+z)

        return nsn, err_nsn

    def Plot_Rate(self, zmin=0.1, zmax=0.2, dz=0.01,
                  bins=None, account_for_edges=False,
                  duration_z=None):
        """ Plot production rate as a function of redshift
        uses the call function (see previoulsy)
        """
        import pylab as plt

        zz, rate, err_rate, nsn, err_nsn = self.__call__(
            zmin=zmin, zmax=zmax, dz=dz, bins=bins,
            account_for_edges=account_for_edges,
            duration_z=duration_z)

        plt.errorbar(zz, np.cumsum(nsn), yerr=np.sqrt(np.cumsum(err_nsn**2)))
        plt.xlabel('z')
        plt.ylabel('N$_{SN}$ <')
        plt.show()
