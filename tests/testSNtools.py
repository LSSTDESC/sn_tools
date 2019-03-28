from builtins import zip
import numpy as np
import unittest
import lsst.utils.tests
from sn_tools.sn_rate import SN_Rate
from sn_tools.sn_utils import GenerateSample

m5_ref = dict(zip('ugrizy', [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))


class TestSNRate(unittest.TestCase):

    def testSNRate(self):
        """Test sn_rate tool """

        rate_type = 'Perrett'
        rate = SN_Rate(rate=rate_type)
        zmin = 0.05
        zmax = 0.2
        dz = 0.01
        duration = 180.
        zz, rate, err_rate, nsn, err_nsn = rate(
            zmin=zmin, zmax=zmax, duration=duration)

        zz_ref = [0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.125, 0.135, 0.145, 0.155, 0.165,
                  0.175, 0.185, 0.195, 0.205]

        rate_ref = [1.90331912e-05, 1.94158583e-05, 1.98025346e-05, 2.01932243e-05,
                    2.05879314e-05, 2.09866601e-05, 2.13894142e-05, 2.17961980e-05,
                    2.22070152e-05, 2.26218698e-05, 2.30407657e-05, 2.34637068e-05,
                    2.38906970e-05, 2.43217399e-05, 2.47568395e-05, 2.51959995e-05]

        err_rate_ref = [3.37089644e-06, 3.44338970e-06, 3.51749665e-06, 3.59323907e-06,
                        3.67063815e-06, 3.74971453e-06, 3.83048831e-06, 3.91297906e-06,
                        3.99720582e-06, 4.08318713e-06, 4.17094104e-06, 4.26048514e-06,
                        4.35183652e-06, 4.44501188e-06, 4.54002746e-06, 4.63689909e-06]

        nsn_ref = [0.05936766, 0.08306682, 0.1108024, 0.14258917, 0.17843839, 0.21835781,
                   0.26235176, 0.31042126, 0.36256408, 0.41877485, 0.47904512, 0.54336348,
                   0.61171562, 0.68408448, 0.76045026, 0.84079061]

        err_nsn_ref = [0.01051438, 0.01473185, 0.01968168, 0.02537272, 0.03181392, 0.03901428,
                       0.04698284, 0.05572861, 0.06526061, 0.07558774, 0.08671886, 0.09866267,
                       0.11142774, 0.12502245, 0.139455,  0.15473334]

        assert(np.isclose(np.array(zz), np.array(zz_ref)).all())
        assert(np.isclose(np.array(rate), np.array(rate_ref)).all())
        assert(np.isclose(np.array(err_rate), np.array(err_rate_ref)).all())
        assert(np.isclose(np.array(nsn), np.array(nsn_ref)).all())
        assert(np.isclose(np.array(err_nsn), np.array(err_nsn_ref)).all())

    def testGenerateSample(self):
        sn_parameters = {}
        # redshift
        sn_parameters['z'] = {}
        sn_parameters['z']['type'] = 'random'
        sn_parameters['z']['min'] = 0.01
        sn_parameters['z']['max'] = 0.1
        sn_parameters['z']['step'] = 0.05
        sn_parameters['z']['rate'] = 'Perrett'
        # DayMax
        sn_parameters['daymax'] = {}
        sn_parameters['daymax']['type'] = 'uniform'
        sn_parameters['daymax']['step'] = 1.
        # Miscellaneous
        sn_parameters['min_rf_phase'] = -20.   # obs min phase (rest frame)
        sn_parameters['max_rf_phase'] = 60.  # obs max phase (rest frame)
        sn_parameters['absmag'] = -19.0906      # peak abs mag
        sn_parameters['band'] = 'bessellB'     # band for absmag
        sn_parameters['magsys'] = 'vega'      # magsys for absmag
        sn_parameters['differential_flux'] = False
        # X1_Color
        sn_parameters['x1_color'] = {}
        sn_parameters['x1_color']['min'] = [-2.0, 0.2]
        sn_parameters['x1_color']['max'] = [0.2, 0.2]
        sn_parameters['x1_color']['rate'] = 'JLA'
        # Cosmology
        cosmo_parameters = {}
        cosmo_parameters['model'] = 'w0waCDM'      # Cosmological model
        cosmo_parameters['Omega_m'] = 0.30             # Omega_m
        cosmo_parameters['Omega_l '] = 0.70             # Omega_l
        cosmo_parameters['H0'] = 72.0                  # H0
        cosmo_parameters['w0'] = -1.0                  # w0
        cosmo_parameters['wa'] = 0.0                   # wa

        genpar = GenerateSample(
            sn_parameters, cosmo_parameters, mjdCol='observationStartMJD')

        observations = self.Observations()
        params = genpar(observations)

        print(params)
        print(params.dtype)
        print(test)

    def Observations(self):
        band = 'r'
        # Define fake data
        names = ['observationStartMJD', 'fieldRA', 'fieldDec',
                 'fiveSigmaDepth', 'visitExposureTime', 'numExposures', 'visitTime', 'season']
        types = ['f8']*len(names)
        names += ['night']
        types += ['i2']
        names += ['filter']
        types += ['O']

        day0 = 59000
        daylast = day0+140.
        cadence = 3.
        dayobs = np.arange(day0, daylast, cadence)
        npts = len(dayobs)
        data = np.zeros(npts, dtype=list(zip(names, types)))
        data['observationStartMJD'] = dayobs
        # data['night'] = np.floor(data['observationStartMJD']-day0)
        data['night'] = 10
        data['fiveSigmaDepth'] = m5_ref[band]
        data['visitExposureTime'] = 15.
        data['numExposures'] = 2
        data['visitTime'] = 2.*15.
        data['filter'] = band
        data['season'] = 1.
        return data
