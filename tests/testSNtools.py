from builtins import zip
import numpy as np
import unittest
import lsst.utils.tests
from sn_tools.sn_rate import SN_Rate
from sn_tools.sn_utils import GenerateSample
from sn_tools.sn_cadence_tools import ReferenceData, GenerateFakeObservations, TemplateData
from sn_tools.sn_telescope import Telescope
import os
from lsst.sims.photUtils import PhotometricParameters
from lsst.sims.photUtils import Bandpass, Sed
from numpy.testing import assert_almost_equal, assert_equal

m5_ref = dict(zip('ugrizy', [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))


class TestSNRate(unittest.TestCase):

    def testSNRate(self):

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


class TestSNTelescope(unittest.TestCase):
    def testTelescope(self):

        airmass = 1.32
        bands = 'ugrizy'

        tel = Telescope(airmass=airmass)
        assert_equal(tel.airmass, airmass)
        sigmab = tel.Sigmab(bands)
        Tb = tel.Tb(bands)
        zp = tel.zp(bands)
        m5 = tel.m5(bands)
        gamma = {}
        for key, val in m5.items():
            gamma[key] = tel.gamma(val, key, 15.)

        mag_to_flux_e_sec = {}
        mag_to_flux = {}

        for key, val in m5.items():
            mag_to_flux_e_sec[key] = tel.mag_to_flux_e_sec(val, key, 15.)
            mag_to_flux[key] = tel.mag_to_flux(val, key, 15.)

        # This is to print the reference data
        """
        print('sigmab_ref = ', sigmab)
        print('Tb_ref = ', Tb)
        print('zp_ref = ', zp)
        print('m5_ref = ', m5)
        print('gamma_ref =', gamma)
        print('mag_to_flux_e_sec_ref = ', mag_to_flux_e_sec)
        print('mag_to_flux_ref = ', mag_to_flux)
        """
        sigmab_ref = {'u': 0.0508700300828208, 'g': 0.15101783445197073, 'r': 0.11436909314149583,
                      'i': 0.08338616181313455, 'z': 0.0556165878457169, 'y': 0.029824040498790286}
        Tb_ref = {'u': 0.030591304228780723, 'g': 0.12434745927780146, 'r': 0.10309893990743581,
                  'i': 0.07759477486679421, 'z': 0.05306487339997605, 'y': 0.025468375062201207}
        zp_ref = {'u': 26.837102506765557, 'g': 28.359699814071035, 'r': 28.156243023489445,
                  'i': 27.847688717586387, 'z': 27.435125355183057, 'y': 26.63811061635532}
        m5_ref = {'u': 23.703378601363863, 'g': 24.808696323169624, 'r': 24.364793723029457,
                  'i': 23.93699490521644, 'z': 23.357218514891976, 'y': 22.445603830018765}
        gamma_ref = {'u': 0.036281856881248506, 'g': 0.03746828818474719, 'r': 0.03797121537662722,
                     'i': 0.03818223331426463, 'z': 0.03844169333327794, 'y': 0.03859779035436962}
        mag_to_flux_e_sec_ref = {'u': (116.93541502018535, 17.930096969761752), 'g': (171.734636650276, 26.33264428637565), 'r': (214.30693218328977, 32.86039626810443), 'i': (
            239.18504619296752, 36.675040416255015), 'z': (279.009657072333, 42.78148075109106), 'y': (310.06961765705864, 47.544008040748984)}
        mag_to_flux_ref = {'u': 0.00033010230695535505, 'g': 0.00011926732280013303, 'r': 0.00017950746372769034,
                           'i': 0.0002661963142302241, 'z': 0.00045405931961482683, 'y': 0.0010513769899871202}

        for band in bands:
            for val in [(sigmab, sigmab_ref), (Tb, Tb_ref), (zp, zp_ref), (m5, m5_ref), (gamma, gamma_ref), (mag_to_flux, mag_to_flux_ref), (mag_to_flux_e_sec, mag_to_flux_e_sec_ref)]:
                assert_almost_equal(val[0][band], val[1][band])


class TestSNCadence(unittest.TestCase):

    def testGenerateSample(self):
        sn_parameters = {}
        # redshift
        sn_parameters['z'] = {}
        sn_parameters['z']['type'] = 'uniform'
        sn_parameters['z']['min'] = 0.1
        sn_parameters['z']['max'] = 0.2
        sn_parameters['z']['step'] = 0.1
        sn_parameters['z']['rate'] = 'Perrett'
        # X1_Color
        sn_parameters['x1_color'] = {}
        sn_parameters['x1_color']['type'] = 'unique'
        sn_parameters['x1_color']['min'] = [-2.0, 0.2]
        sn_parameters['x1_color']['max'] = [0.2, 0.2]
        sn_parameters['x1_color']['rate'] = 'JLA'
        # DayMax
        sn_parameters['daymax'] = {}
        sn_parameters['daymax']['type'] = 'unique'
        sn_parameters['daymax']['step'] = 1.
        # Miscellaneous
        sn_parameters['min_rf_phase'] = -20.   # obs min phase (rest frame)
        sn_parameters['max_rf_phase'] = 60.  # obs max phase (rest frame)
        sn_parameters['absmag'] = -19.0906      # peak abs mag
        sn_parameters['band'] = 'bessellB'     # band for absmag
        sn_parameters['magsys'] = 'vega'      # magsys for absmag
        sn_parameters['differential_flux'] = False
        # Cosmology
        cosmo_parameters = {}
        cosmo_parameters['model'] = 'w0waCDM'      # Cosmological model
        cosmo_parameters['Omega_m'] = 0.30             # Omega_m
        cosmo_parameters['Omega_l '] = 0.70             # Omega_l
        cosmo_parameters['H0'] = 72.0                  # H0
        cosmo_parameters['w0'] = -1.0                  # w0
        cosmo_parameters['wa'] = 0.0                   # wa

        # instantiate GenerateSample
        genpar = GenerateSample(
            sn_parameters, cosmo_parameters, mjdCol='observationStartMJD', dirFiles='../reference_files')

        # get some observations
        observations = self.Observations()

        # get simulation parameters from these observations
        params = genpar(observations)

        names = ['z', 'x1', 'color', 'daymax', 'epsilon_x0', 'epsilon_x1',
                 'epsilon_color', 'epsilon_daymax', 'min_rf_phase', 'max_rf_phase']

        types = ['f8']*len(names)

        # print(params.dtype.names)

        params_ref = np.array([(0.1, -2., 0.2, 59023.1, 0., 0., 0., 0., -15., 30.),
                               (0.2, -2., 0.2, 59025.2, 0., 0., 0., 0., -15., 30.)], dtype=list(zip(names, types)))

        for name in params_ref.dtype.names:
            assert(np.isclose(params[name], params_ref[name]).all())

    def Observations(self, daymin=59000, cadence=3., season_length=140.):
        band = 'r'
        # Define fake data
        names = ['observationStartMJD', 'fieldRA', 'fieldDec',
                 'fiveSigmaDepth', 'visitExposureTime', 'numExposures',
                 'visitTime', 'season', 'seeingFwhmEff', 'seeingFwhmGeom']
        types = ['f8']*len(names)
        names += ['night']
        types += ['i2']
        names += ['filter']
        types += ['O']

        day0 = daymin
        daylast = day0+season_length
        cadence = cadence
        dayobs = np.arange(day0, daylast, cadence)
        npts = len(dayobs)
        data = np.zeros(npts, dtype=list(zip(names, types)))
        data['observationStartMJD'] = dayobs
        # data['night'] = np.floor(data['observationStartMJD']-day0)
        data['night'] = 10
        data['fiveSigmaDepth'] = m5_ref[band]
        data['visitExposureTime'] = 30.
        data['numExposures'] = 2
        data['visitTime'] = 2.*15.
        data['filter'] = band
        data['season'] = 1.
        data['seeingFwhmEff'] = 0.8
        data['seeingFwhmGeom'] = 0.8
        return data

    def testReferenceData(self):
        # dirfiles = os.getenv('REF_FILES')
        dirfiles = '../reference_files'
        Li_files = [dirfiles+'/Li_SNCosmo_-2.0_0.2.npy']
        Mag_files = [dirfiles+'/Mag_to_Flux_SNCosmo.npy']
        band = 'r'
        z = 0.3

        refdata = ReferenceData(Li_files, Mag_files, band, z)

        arr = np.array(
            [('r', 25.6, 900.), ('r', 25.8, 1500.), ('r', 25.7, 1200.)])

        names = ['m5', 'deltaT']
        types = ['f8', 'f8']
        data = np.array(
            [(25.6, 10.), (25.8, -10.), (25.7, 28.)], dtype=list(zip(names, types)))

        mag_ref = np.array([10.55691041, 8.785232, 9.61469456])

        for val in refdata.mag_to_flux:
            res = val(data['m5'])
            assert(np.isclose(res, mag_ref).all())

        flux_ref = np.array([152.08206561, 101.68561218, 36.14289588])
        for val in refdata.fluxes:
            res = val(data['deltaT'])
            assert(np.isclose(res, flux_ref).all())

    def testGenerateFakeObservations(self):

        config = {}
        config['RA'] = 0.0  # RA of the field
        config['Dec'] = 0.0  # Dec of the field
        config['seasons'] = [1.]  # seasons
        config['season_length'] = 10.  # season_length (days)
        config['bands'] = ['g', 'r', 'i', 'z', 'y']  # bands to consider
        config['Cadence'] = [3., 3., 3., 3., 3.]  # Cadence[day] per band
        config['m5'] = [23.27, 24.58, 24.22, 23.65,
                        22.78, 22.00]  # 5-sigma depth values
        config['Nvisits'] = [1, 1, 1, 1, 1]
        config['Exposure_Time'] = [30., 30., 30., 30., 30.]
        config['shift_days'] = 0.0069444  # in days']= 10./(24.*60.)
        config['MJD_min'] = [-40.]
        config['seeingEff'] = [0.87, 0.83, 0.80, 0.78, 0.76]
        config['seeingGeom'] = [0.87, 0.83, 0.80, 0.78, 0.76]

        fake_obs = GenerateFakeObservations(config).Observations

        # this is to print the reference
        #print(fake_obs, fake_obs.dtype)

        dtype = [('observationStartMJD', '<f8'), ('fieldRA', '<f8'), ('fieldDec', '<f8'), ('filter', '<U1'),
                 ('fiveSigmaDepth', '<f8'), ('numExposures',
                                             '<i8'), ('visitExposureTime', '<f8'),
                 ('season', '<f8'), ('seeingFwhmEff', '<f8'), ('seeingFwhmGeom', '<f8'), ('observationId', '<i8')]
        ref_obs = np.array([(-40.0, 0.0, 0.0, 'g', 23.27, 1, 30.0, 1.0, 0.87, 0.87, 245),
                            (-39.9930556, 0.0, 0.0, 'r', 24.58,
                             1, 30.0, 1.0, 0.83, 0.83, 26),
                            (-39.9861112, 0.0, 0.0, 'i',
                             24.22, 1, 30.0, 1.0, 0.8, 0.8, 25),
                            (-39.9791668, 0.0, 0.0, 'z', 23.65,
                             1, 30.0, 1.0, 0.78, 0.78, 79),
                            (-39.9722224, 0.0, 0.0, 'y', 22.78,
                             1, 30.0, 1.0, 0.76, 0.76, 113),
                            (-37.0, 0.0, 0.0, 'g', 23.27,
                             1, 30.0, 1.0, 0.87, 0.87, 113),
                            (-36.9930556, 0.0, 0.0, 'r', 24.58,
                             1, 30.0, 1.0, 0.83, 0.83, 147),
                            (-36.9861112, 0.0, 0.0, 'i',
                             24.22, 1, 30.0, 1.0, 0.8, 0.8, 20),
                            (-36.9791668, 0.0, 0.0, 'z', 23.65,
                             1, 30.0, 1.0, 0.78, 0.78, 88),
                            (-36.9722224, 0.0, 0.0, 'y', 22.78,
                             1, 30.0, 1.0, 0.76, 0.76, 211),
                            (-34.0, 0.0, 0.0, 'g', 23.27,
                             1, 30.0, 1.0, 0.87, 0.87, 150),
                            (-33.9930556, 0.0, 0.0, 'r', 24.58,
                             1, 30.0, 1.0, 0.83, 0.83, 87),
                            (-33.9861112, 0.0, 0.0, 'i',
                             24.22, 1, 30.0, 1.0, 0.8, 0.8, 2),
                            (-33.9791668, 0.0, 0.0, 'z', 23.65,
                             1, 30.0, 1.0, 0.78, 0.78, 70),
                            (-33.9722224, 0.0, 0.0, 'y', 22.78,
                             1, 30.0, 1.0, 0.76, 0.76, 89),
                            (-31.0, 0.0, 0.0, 'g', 23.27,
                             1, 30.0, 1.0, 0.87, 0.87, 85),
                            (-30.9930556, 0.0, 0.0, 'r', 24.58,
                             1, 30.0, 1.0, 0.83, 0.83, 10),
                            (-30.9861112, 0.0, 0.0, 'i',
                             24.22, 1, 30.0, 1.0, 0.8, 0.8, 99),
                            (-30.9791668, 0.0, 0.0, 'z', 23.65,
                             1, 30.0, 1.0, 0.78, 0.78, 108),
                            (-30.9722224, 0.0, 0.0, 'y', 22.78,
                             1, 30.0, 1.0, 0.76, 0.76, 198),
                            (-28.0, 0.0, 0.0, 'g', 23.27,
                             1, 30.0, 1.0, 0.87, 0.87, 126),
                            (-27.9930556, 0.0, 0.0, 'r', 24.58,
                             1, 30.0, 1.0, 0.83, 0.83, 94),
                            (-27.9861112, 0.0, 0.0, 'i', 24.22,
                             1, 30.0, 1.0, 0.8, 0.8, 196),
                            (-27.9791668, 0.0, 0.0, 'z', 23.65,
                             1, 30.0, 1.0, 0.78, 0.78, 97),
                            (-27.9722224, 0.0, 0.0, 'y', 22.78, 1, 30.0, 1.0, 0.76, 0.76, 48)], dtype=dtype)

        for name in ref_obs.dtype.names:
            if name != 'observationId':  # observationId is random here
                if name != 'filter':
                    assert(np.isclose(fake_obs[name], ref_obs[name]).all())
                else:
                    assert((fake_obs[name] == ref_obs[name]).all())

    def testTemplateData(self):
        # refname = 'LC_Ref_-2.0_0.2.hdf5'
        refname = 'reference_files/LC_-2.0_0.2.hdf5'
        band = 'r'
        z = 0.3
        min_rf_phase = -20
        max_rf_phase = 60.

        templdata = TemplateData('../{}'.format(refname), band)

        daymin = 59000
        season_length = 180.
        obs = self.Observations(daymin=daymin, cadence=10,
                                season_length=season_length)
        T0 = daymin+season_length/2
        params = np.array([(z, T0, min_rf_phase, max_rf_phase)],
                          # (0.3, daymin+50, min_rf_phase, max_rf_phase)],
                          dtype=[('z', 'f8'), ('daymax', 'f8'), ('min_rf_phase', 'f8'), ('max_rf_phase', 'f8')])
        simulations = templdata.Simulation(obs['observationStartMJD'],
                                           obs['fiveSigmaDepth'], obs['visitExposureTime'], params)

        names = simulations.dtype.names

        # this is to print reference results in the correct format
        """
        for name in names:
            print('refsimu[\'{}\'] ='.format(name), [simulations[name][i]
                                                     for i in range(len(simulations[name]))])
        """
        refsimu = {}

        refsimu['flux'] = [2.0675389396331054e-08, 2.1815096896490717e-06, 4.541527969545595e-06, 3.260504094538401e-06, 1.55705043949865e-06,
                           6.851533091294574e-07, 3.441011637893282e-07, 2.9963633052407117e-07, 2.7463390255105426e-07, 2.461314897088084e-07]
        refsimu['fluxerr'] = [2.3839374730799952e-08, 2.5293512926073686e-08, 2.6791666386123333e-08, 2.5989336585073304e-08, 2.488204637725485e-08,
                              2.4295822705578456e-08, 2.4062667215186852e-08, 2.403210175073525e-08, 2.4014888121626345e-08, 2.3995262957623794e-08]
        refsimu['phase'] = [-15.384615384615383, -7.692307692307692, 0.0, 7.692307692307692, 15.384615384615383,
                            23.076923076923077, 30.769230769230766, 38.46153846153846, 46.15384615384615, 53.84615384615385]
        refsimu['snr_m5'] = [0.8672790133886734, 86.24779389185888, 169.51270981404375, 125.45545685113935, 62.5772661898894,
                             28.200457232186764, 14.300208730483254, 12.468170018250857, 11.435985092253489, 10.257503330698333]
        refsimu['mag'] = [28.111431377980104, 23.053172756431266, 22.257060639313785, 22.616853747456034,
                          23.41930891867002, 24.3105962232617, 25.05835026864019, 25.20857944759792, 25.30318025159253, 25.422147671610116]
        refsimu['magerr'] = [1.251888017577976, 0.012588567843479814, 0.006405043055173781, 0.008654356151653272, 0.017350329774130525,
                             0.03850065961054411, 0.07592450048954205, 0.08708063839110576, 0.09494033054429091, 0.10584799924059222]
        refsimu['time'] = [59070.0, 59080.0, 59090.0, 59100.0,
                           59110.0, 59120.0, 59130.0, 59140.0, 59150.0, 59160.0]
        refsimu['band'] = ['LSST::r', 'LSST::r', 'LSST::r', 'LSST::r',
                           'LSST::r', 'LSST::r', 'LSST::r', 'LSST::r', 'LSST::r', 'LSST::r']
        refsimu['zp'] = [8.90006562228223, 8.90006562228223, 8.90006562228223, 8.90006562228223, 8.90006562228223,
                         8.90006562228223, 8.90006562228223, 8.90006562228223, 8.90006562228223, 8.90006562228223]
        refsimu['zpsys'] = ['ab', 'ab', 'ab', 'ab',
                            'ab', 'ab', 'ab', 'ab', 'ab', 'ab']
        refsimu['z'] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        refsimu['daymax'] = [59090.0, 59090.0, 59090.0, 59090.0,
                             59090.0, 59090.0, 59090.0, 59090.0, 59090.0, 59090.0]
        for name in names:
            if name != 'band' and name != 'zpsys':
                assert(np.isclose(simulations[name], refsimu[name]).all())
            else:
                assert((simulations[name] == refsimu[name]).all())


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main(verbosity=5)
