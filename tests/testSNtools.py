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

        genpar = GenerateSample(
            sn_parameters, cosmo_parameters, mjdCol='observationStartMJD')

        observations = self.Observations()
        params = genpar(observations)

        r = []
        r.append([0.1, -2., 0.2, 59022., 0., 0., 0., -15., 30.])
        r.append([0.2, -2., 0.2, 59024., 0., 0., 0., -15., 30.])
        r.append([0.3, -2., 0.2, 59026., 0., 0., 0., -15., 30.])
        tt = None
        for val in r:
            if tt is None:
                tt = np.asarray([val])
            else:
                tt = np.concatenate((tt, np.asarray([val])))

        names = ['z', 'x1', 'color', 'daymax',
                 'epsilon_x0', 'epsilon_x1', 'epsilon_color',
                 'min_rf_phase', 'max_rf_phase']
        types = ['f8']*len(names)
        npts = len(r)
        params_ref = np.zeros(npts, dtype=list(zip(names, types)))
        params_ref = tt
        assert_almost_equal(params, params_ref, decimal=5)

    def Observations(self, daymin=59000, cadence=3., season_length=140.):
        band = 'r'
        # Define fake data
        names = ['observationStartMJD', 'fieldRA', 'fieldDec',
                 'fiveSigmaDepth', 'visitExposureTime', 'numExposures', 'visitTime', 'season']
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
        return data

    def testReferenceData(self):
        dirfiles = os.getenv('REF_FILES')

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
        config['Ra'] = 0.0  # Ra of the field
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

        fake_obs = GenerateFakeObservations(config).Observations

        dtype = [('observationStartMJD', '<f8'), ('fieldRA', '<f8'), ('fieldDec', '<f8'), ('filter', '<U1'),
                 ('fiveSigmaDepth', '<f8'), ('numExposures', '<i8'), ('visitExposureTime', '<f8'), ('season', '<f8')]
        ref_obs = np.array([(-40., 0., 0., 'g', 23.27, 1, 30., 1.),
                            (-39.9930556, 0., 0., 'r', 24.58, 1, 30., 1.),
                            (-39.9861112, 0., 0., 'i', 24.22, 1, 30., 1.),
                            (-39.9791668, 0., 0., 'z', 23.65, 1, 30., 1.),
                            (-39.9722224, 0., 0., 'y', 22.78, 1, 30., 1.),
                            (-37., 0., 0., 'g', 23.27, 1, 30., 1.),
                            (-36.9930556, 0., 0., 'r', 24.58, 1, 30., 1.),
                            (-36.9861112, 0., 0., 'i', 24.22, 1, 30., 1.),
                            (-36.9791668, 0., 0., 'z', 23.65, 1, 30., 1.),
                            (-36.9722224, 0., 0., 'y', 22.78, 1, 30., 1.),
                            (-34., 0., 0., 'g', 23.27, 1, 30., 1.),
                            (-33.9930556, 0., 0., 'r', 24.58, 1, 30., 1.),
                            (-33.9861112, 0., 0., 'i', 24.22, 1, 30., 1.),
                            (-33.9791668, 0., 0., 'z', 23.65, 1, 30., 1.),
                            (-33.9722224, 0., 0., 'y', 22.78, 1, 30., 1.),
                            (-31., 0., 0., 'g', 23.27, 1, 30., 1.),
                            (-30.9930556, 0., 0., 'r', 24.58, 1, 30., 1.),
                            (-30.9861112, 0., 0., 'i', 24.22, 1, 30., 1.),
                            (-30.9791668, 0., 0., 'z', 23.65, 1, 30., 1.),
                            (-30.9722224, 0., 0., 'y', 22.78, 1, 30., 1.),
                            (-28., 0., 0., 'g', 23.27, 1, 30., 1.),
                            (-27.9930556, 0., 0., 'r', 24.58, 1, 30., 1.),
                            (-27.9861112, 0., 0., 'i', 24.22, 1, 30., 1.),
                            (-27.9791668, 0., 0., 'z', 23.65, 1, 30., 1.),
                            (-27.9722224, 0., 0., 'y', 22.78, 1, 30., 1.)], dtype=dtype)

        for name in ref_obs.dtype.names:
            if name != 'filter':
                assert(np.isclose(fake_obs[name], ref_obs[name]).all())
            else:
                assert((fake_obs[name] == ref_obs[name]).all())

    def testTemplateData(self):
        # refname = 'LC_Ref_-2.0_0.2.hdf5'
        refname = 'LC_-2.0_0.2.hdf5'
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
        """
        for name in names:
            print(name, [simulations[name][i]
                         for i in range(len(simulations[name]))])
        """
        refsimu = {}

        refsimu['flux'] = [1.817312225268024e-08, 2.1814654556593405e-06, 4.541527969545595e-06, 3.260839109537984e-06, 1.5579757863909567e-06,
                           6.85197962424031e-07, 3.4423028877421304e-07, 2.995864230923943e-07, 2.746116879273338e-07, 2.4609834005024835e-07]
        refsimu['fluxerr'] = [2.383796072777225e-08, 2.529347401033142e-08, 2.6791666386123333e-08, 2.5989374273901374e-08, 2.4882630169041346e-08,
                              2.4295882829527847e-08, 2.4062753049354736e-08, 2.403206598940246e-08, 2.4014882971536566e-08, 2.3995240144740985e-08]
        refsimu['phase'] = [-15.384615384615383, -7.692307692307692, 0.0, 7.692307692307692, 15.384615384615383,
                            23.076923076923077, 30.769230769230766, 38.46153846153846, 46.15384615384615, 53.84615384615385]
        refsimu['snr_m5'] = [0.7623606087876372, 86.24617775985597, 169.51270981404375, 125.46816538066986,
                             62.61298648120288, 28.20222534129445, 14.305523896960905, 12.466111870053304, 11.435062509062192, 10.25613157300222]
        refsimu['mag'] = [28.25149175191791, 23.05319477188748, 22.257060639313785, 22.61674219439001, 23.418663863007797,
                          24.31052546520192, 25.057942919325697, 25.208760302898412, 25.303268078319512, 25.42229391136379]
        refsimu['magerr'] = [1.4241766852103601, 0.012588803735526176, 0.006405043055173781, 0.008653479561640282,
                             0.01734043152667825, 0.03849824585183941, 0.07589629101166896, 0.08709501535649919, 0.09494799034965419, 0.10586215641150437]
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

        sigmab_ref = {'u': 0.035785915980876104, 'g': 0.1432890267516797, 'r': 0.11262226010146574,
                      'i': 0.08043284004643284, 'z': 0.054764490348215365, 'y': 0.028509921564443697}
        Tb_ref = {'u': 0.021222760832847506, 'g': 0.11847125554400734, 'r': 0.10148007886034267,
                  'i': 0.0748474199497848, 'z': 0.05226328510071998, 'y': 0.024195249318660562}
        zp_ref = {'u': 26.4401122237477, 'g': 28.30714000177329, 'r': 28.13905951417312,
                  'i': 27.80854961022156, 'z': 27.418599285553352, 'y': 26.58243277815136}
        m5_ref = {'u': 23.35183820200688, 'g': 24.7706326858414, 'r': 24.35514965603368,
                  'i': 23.912873887926537, 'z': 23.34804737271212, 'y': 22.42200234510623}
        gamma_ref = {'u': 0.03612290845059692, 'g': 0.037434259447649, 'r': 0.03795707830010742,
                     'i': 0.03815691495227663, 'z': 0.03843110131265436, 'y': 0.038555746290894546}

        mag_to_flux_e_sec_ref = {'u': (112.14143466965382, 17.19501998268025), 'g': (169.45696566913534, 25.98340140260075), 'r': (212.82392209085376, 32.63300138726424), 'i': (
            235.8993738420849, 36.17123732245302), 'z': (277.1259943057538, 42.49265246021558), 'y': (301.04309648264643, 46.15994146067245)}
        mag_to_flux_ref = {'u': 0.00045631497273323314, 'g': 0.0001235227425539364, 'r': 0.00018110904377852902,
                           'i': 0.0002721763870527382, 'z': 0.0004579109731990962, 'y': 0.001074481795844473}

        """
        print(sigmab)
        print(Tb)
        print(zp)
        print(m5)
        print(gamma)
        print(mag_to_flux_e_sec, mag_to_flux)
        """
        for band in bands:
            for val in [(sigmab, sigmab_ref), (Tb, Tb_ref), (zp, zp_ref), (m5, m5_ref), (gamma, gamma_ref), (mag_to_flux, mag_to_flux_ref), (mag_to_flux_e_sec, mag_to_flux_e_sec_ref)]:
                assert_almost_equal(val[0][band], val[1][band])
