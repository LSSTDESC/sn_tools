from builtins import zip
import numpy as np
import unittest
import lsst.utils.tests
from sn_tools.sn_rate import SN_Rate
from sn_tools.sn_utils import GenerateSample, Make_Files_for_Cadence_Metric, X0_norm
from sn_tools.sn_utils import DiffFlux, MbCov, GetReference, Gamma
from sn_tools.sn_cadence_tools import ReferenceData, GenerateFakeObservations
from sn_tools.sn_cadence_tools import TemplateData, AnaOS, Match_DD
from sn_tools.sn_calcFast import LCfast, CalcSN, CalcSN_df, CovColor
from sn_tools.sn_telescope import Telescope
from sn_tools.sn_obs import DDFields
import os
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import h5py
from astropy.table import Table, vstack

m5_ref = dict(zip('ugrizy', [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))
repo_dir = 'https://me.lsst.eu/gris/Reference_Files'


def getFile(dbDir, dbName, dbExtens, repofile):
    repo_reffiles = '{}/{}'.format(repo_dir, repofile)
    # check whether the file is available; if not-> get it!
    if not os.path.isfile('{}/{}.{}'.format(dbDir, dbName, dbExtens)):
        path = '{}/{}.{}'.format(repo_reffiles,
                                 dbName, dbExtens)
        cmd = 'wget {}'.format(path)
        os.system(cmd)


def Observations(daymin=59000, cadence=3., season_length=140., band='r'):
    # Define fake data
    names = ['observationStartMJD', 'fieldRA', 'fieldDec',
             'fiveSigmaDepth', 'visitExposureTime', 'numExposures',
             'visitTime', 'season', 'seeingFwhmEff', 'seeingFwhmGeom',
             'pixRA', 'pixDec']
    types = ['f8']*len(names)
    names += ['night', 'healpixID']
    types += ['i2', 'i2']
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
    data['healpixID'] = 10
    data['pixRA'] = 0.0
    data['pixDec'] = 0.0
    return data


def getReference(x1, color):
    # first step: get reference files
    Instrument = {}
    Instrument['name'] = 'LSST'  # name of the telescope (internal)
    # dir of throughput
    Instrument['throughput_dir'] = 'LSST_THROUGHPUTS_BASELINE'
    Instrument['atmos_dir'] = 'THROUGHPUTS_DIR'  # dir of atmos
    Instrument['airmass'] = 1.2  # airmass value
    Instrument['atmos'] = True  # atmos
    Instrument['aerosol'] = False  # aerosol

    lc_reference = {}
    gamma_reference = '../../reference_files/gamma.hdf5'

    fDir = '.'
    fName = 'LC_{}_{}_vstack'.format(x1, color)
    fExtens = 'hdf5'

    getFile(fDir, fName, fExtens, 'Templates')
    fullname = '{}/{}.{}'.format(fDir, fName, fExtens)

    lc_ref = GetReference(
        fullname, gamma_reference, Instrument)

    return lc_ref


def snSimuParam(x1, color):
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
    sn_parameters['x1_color']['min'] = [x1, color]
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

    return sn_parameters, cosmo_parameters


def simuLCfast(x1, color, bands='r'):

    # get reference LC
    lc_ref = getReference(x1, color)

    # get telescope model
    telescope = Telescope()

    # instance of LCfast
    lcf = LCfast(lc_ref, x1, color, telescope)

    # need some observations to run...
    obs = None
    for band in bands:
        obs_b = Observations(season_length=180, band=band)
        if obs is None:
            obs = obs_b
        else:
            obs = np.concatenate((obs, obs_b))

    # and simulation parameters
    sn_parameters, cosmo_parameters = snSimuParam(x1, color)
    gen_par = GenerateSample(
        sn_parameters, cosmo_parameters, mjdCol='observationStartMJD', dirFiles='../reference_files')
    params = gen_par(obs)

    # perform simulation
    lc = lcf(obs, params)

    return lc


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
        # print(fake_obs, fake_obs.dtype)

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
        obs = Observations(daymin=daymin, cadence=10,
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

    def testAnaOS(self):

        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_twoyears'
        dbExtens = 'npy'
        nclusters = 5
        fields = DDFields()

        getFile(dbDir, dbName, dbExtens, 'unittests')

        stat = AnaOS(dbDir, dbName, dbExtens, nclusters,
                     fields).stat

        # thi is to get the reference data
        # print(stat.values.tolist(), stat.columns)

        valrefs = ['descddf_v1.4_10yrs_twoyears', 339316, 26443, 73015, 70055, 21483, 80777, 67543, 339316, 20513, 0.05700763418179191, 967, 3838, 1930, 1480, 1920, 10378, 20513, 3730.0, 272.0, 174.0, 348.0, 688.0, 1900.0, 348.0, 2.001879929708757, 1.883274130238675, 1.353426274694172, 2791.0, 248.0, 129.0, 254.0, 504.0, 1400.0, 256.0, 3.2066746992925483, 3.0166084454850193, 1.3534620445526713, 5313.0, 224.0, 256.0,
                   512.0, 1017.0, 2796.0, 508.0, 1.424606742986726, 1.3640734093042113, 1.3297419541315458, 5145.0, 464.0, 240.0, 480.0, 960.0, 2525.0, 476.0, 1.6342188350701072, 1.5566493003965718, 1.3366864617690055, 3534.0, 272.0, 168.0, 336.0, 669.0, 1757.0, 332.0, 1.4463375323299488, 1.3777211519657158, 1.3366522961266032, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        names = ['cadence', 'WFD', 'WFD_g', 'WFD_i', 'WFD_r', 'WFD_u', 'WFD_y', 'WFD_z',
                 'WFD_all', 'DD', 'frac_DD', 'DD_g', 'DD_i', 'DD_r', 'DD_u', 'DD_y',
                 'DD_z', 'DD_all', 'ELAIS', 'ELAIS_u', 'ELAIS_g', 'ELAIS_r', 'ELAIS_i',
                 'ELAIS_z', 'ELAIS_y', 'ELAIS_area', 'ELAIS_width_RA', 'ELAIS_width_Dec',
                 'SPT', 'SPT_u', 'SPT_g', 'SPT_r', 'SPT_i', 'SPT_z', 'SPT_y', 'SPT_area',
                 'SPT_width_RA', 'SPT_width_Dec', 'COSMOS', 'COSMOS_u', 'COSMOS_g',
                 'COSMOS_r', 'COSMOS_i', 'COSMOS_z', 'COSMOS_y', 'COSMOS_area',
                 'COSMOS_width_RA', 'COSMOS_width_Dec', 'CDFS', 'CDFS_u', 'CDFS_g',
                 'CDFS_r', 'CDFS_i', 'CDFS_z', 'CDFS_y', 'CDFS_area', 'CDFS_width_RA',
                 'CDFS_width_Dec', 'XMM-LSS', 'XMM-LSS_u', 'XMM-LSS_g', 'XMM-LSS_r',
                 'XMM-LSS_i', 'XMM-LSS_z', 'XMM-LSS_y', 'XMM-LSS_area',
                 'XMM-LSS_width_RA', 'XMM-LSS_width_Dec', 'ADFS1', 'ADFS1_u', 'ADFS1_g',
                 'ADFS1_r', 'ADFS1_i', 'ADFS1_z', 'ADFS1_y', 'ADFS1_area',
                 'ADFS1_width_RA', 'ADFS1_width_Dec', 'ADFS2', 'ADFS2_u', 'ADFS2_g',
                 'ADFS2_r', 'ADFS2_i', 'ADFS2_z', 'ADFS2_y', 'ADFS2_area',
                 'ADFS2_width_RA', 'ADFS2_width_Dec']

        dfref = pd.DataFrame([valrefs], columns=names)
        for name in names:
            if name != 'cadence':
                assert(np.isclose(stat[name], dfref[name]).all())
            else:
                assert((stat[name] == dfref[name]).all())

    def testMatch_DD(self):

        # get some data
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_NSNMetric_DD'
        dbExtens = 'hdf5'

        # grab the file if not already available
        getFile(dbDir, dbName, dbExtens, 'unittests')

        fName = '{}/{}.{}'.format(dbDir, dbName, dbExtens)
        fFile = h5py.File(fName, 'r')
        keys = list(fFile.keys())

        data = Table()
        for key in keys:
            data = vstack([data, Table.read(fFile, path=key)])
            break

        # get DD fields
        fields = DDFields()

        # now perform the matching on the first 10 rows of data
        datadf = data[:10].to_pandas()

        matched = Match_DD(fields, datadf)

        # this is to print reference values
        """
        print(type(matched))
        print(matched.values.tolist(), matched.columns)
        """

        valrefs = [[0, 351.0, -64.1987, 45407, 1, -2.0, -0.2, 0.0, -1, -1.0, -1.0, 'SPT', 290, 349.39, -63.32, 5],
                   [0, 351.0, -64.1987, 45407, 1, -2.0, 0.0, 0.0, -
                       1, -1.0, -1.0, 'SPT', 290, 349.39, -63.32, 5],
                   [0, 351.0, -64.1987, 45407, 1, -2.0, 0.2, 0.0, -
                       1, -1.0, -1.0, 'SPT', 290, 349.39, -63.32, 5],
                   [0, 351.0, -64.1987, 45407, 1, 0.0, -0.2, 0.0, -
                       1, -1.0, -1.0, 'SPT', 290, 349.39, -63.32, 5],
                   [0, 351.0, -64.1987, 45407, 1, 0.0, 0.0, 0.0, -
                       1, -1.0, -1.0, 'SPT', 290, 349.39, -63.32, 5],
                   [0, 351.0, -64.1987, 45407, 1, 0.0, 0.2, 0.0, -
                       1, -1.0, -1.0, 'SPT', 290, 349.39, -63.32, 5],
                   [0, 351.0, -64.1987, 45407, 1, 2.0, -0.2, 0.0, -
                       1, -1.0, -1.0, 'SPT', 290, 349.39, -63.32, 5],
                   [0, 351.0, -64.1987, 45407, 1, 2.0, 0.0, 0.0, -
                       1, -1.0, -1.0, 'SPT', 290, 349.39, -63.32, 5],
                   [0, 351.0, -64.1987, 45407, 1, 2.0, 0.2, 0.0, -
                       1, -1.0, -1.0, 'SPT', 290, 349.39, -63.32, 5],
                   [0, 351.0, -64.1987, 45407, 2, -2.0, -0.2, 0.7601352128498775, 1, 10.37962376408833, 9.0, 'SPT', 290, 349.39, -63.32, 5]]
        cols = ['index', 'pixRA', 'pixDec', 'healpixID', 'season', 'x1', 'color',
                'zlim', 'status', 'nsn_med', 'nsn', 'fieldname', 'fieldId', 'RA', 'Dec',
                'fieldnum']

        dataref = pd.DataFrame(valrefs, columns=cols)
        for name in cols:
            if name != 'fieldname':
                assert(np.isclose(dataref[name], matched[name]).all())
            else:
                assert((dataref[name] == matched[name]).all())


class TestSNUtils(unittest.TestCase):
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
        observations = Observations()

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

    def testMake_Files_for_Cadence_Metric(self):

        telescope = Telescope(airmass=1.2)
        x1 = -2.0
        color = 0.2
        dbDir = '.'
        fName = 'LC_{}_{}'.format(x1, color)
        fExtens = 'hdf5'
        simulator_name = 'SNCosmo'

        getFile(dbDir, fName, fExtens, 'Templates')

        proc = Make_Files_for_Cadence_Metric(
            '{}/{}.{}'.format(dbDir, fName, fExtens), telescope, simulator_name)

        assert(os.path.isfile('Mag_to_Flux_LSST_{}.npy'.format(simulator_name)))
        assert(os.path.isfile('Li_{}_{}_{}.npy'.format(simulator_name, x1, color)))

    def testX0_norm(self):

        salt2Dir = '../../SALT2_Files'
        outFile = 'X0_norm.npy'
        # X0_norm(salt2Dir=salt2Dir, outfile=outFile)

        assert(os.path.isfile(outFile))

    def testDiffFlux(self):

        print('Test to be implemented')
        test_implemented = False
        assert(test_implemented == True)

    def testMbCov(self):

        salt2Dir = 'SALT2_Files'
        # MbCov(salt2Dir)

    def testGetReference(self):

        x1, color = -2.0, 0.2
        lc_ref = getReference(x1, color)
        bands_ref = ['g', 'r', 'i', 'z', 'y']
        # check whether dict keys are ok

        assert(set(lc_ref.flux.keys()) == set(bands_ref))
        assert(set(lc_ref.fluxerr.keys()) == set(bands_ref))
        assert(set(lc_ref.param.keys()) == set(bands_ref))
        assert(set(lc_ref.gamma.keys()) == set(bands_ref))

        # now check interpolation
        band = 'i'
        phase = np.array([-20., 0., 30.])
        z = np.array([0.5]*len(phase))
        fluxes_ref = np.array([1.00000000e-10, 1.53458242e-06, 1.75235325e-07])
        fluxes_err_ref = np.array(
            [3.31944210e-08, 3.42222646e-08, 3.33133353e-08])

        fluxes = lc_ref.flux[band]((phase, z))
        fluxes_err = lc_ref.fluxerr[band]((phase, z))

        assert(np.isclose(fluxes, fluxes_ref).all())
        assert(np.isclose(fluxes_err, fluxes_err_ref).all())

        # print(lc_ref.flux, lc_ref.fluxerr)
        # print(lc_ref.param, lc_ref.gamma)

    def testGamma(self):

        bands = 'r'
        telescope = Telescope(airmass=1.2)
        outName = 'gamma_test.hdf5'
        mag_range = np.arange(20., 25., 1.)
        exptimes = np.array([15., 30.])
        Gamma(bands, telescope, outName,
              mag_range=mag_range,
              exptimes=exptimes)
        # check production
        fFile = h5py.File(outName, 'r')
        keys = list(fFile.keys())

        data = Table()
        for key in keys:
            data = vstack([data, Table.read(fFile, path=key)])

        gamma_ref = [0.039963862137247765, 0.03998193106862388, 0.03990922579288891, 0.039954612896444454, 0.0397719855008266,
                     0.039885992750413296, 0.03942725347333887, 0.039713626736669436, 0.038561325770985665, 0.03928066288549283]

        assert(np.isclose(data['gamma'], np.array(gamma_ref)).all())


class TestSNcalcFast(unittest.TestCase):

    def testLCfast(self):

        x1, color = -2.0, 0.2

        lc = simuLCfast(x1, color)

        # print(lc.columns)
        # for col in lc.columns:
        #    print(col, lc[col].values.tolist())

        # These are what the result should be
        dictRef = {}
        dictRef['flux'] = [4.511147986335053e-06, 1.2317188232550995e-05, 2.550111662322939e-05, 3.7578119573455526e-05, 4.55531457559699e-05, 4.891115014639708e-05, 4.7133300685124935e-05, 4.158604026288241e-05, 3.409213315546674e-05, 2.8570455907048627e-05, 2.5038831188298558e-05, 2.1858035689046604e-05, 1.879011450752236e-05, 1.577897479370136e-05, 1.2660074510369872e-05, 9.969941676665338e-06,
                           5.397161397972984e-07, 1.967780860222931e-06, 4.340271623278295e-06, 7.16463061377825e-06, 9.4576858760202e-06, 1.074632739127063e-05, 1.0806717280098711e-05, 1.0358122558068462e-05, 9.29955918584795e-06, 7.726015475790287e-06, 6.461230015881104e-06, 5.541974962087858e-06, 4.687433939533106e-06, 3.907532488461168e-06, 3.2920828790886484e-06, 2.726609113336508e-06, 2.199381111769534e-06, 1.7924998997463632e-06]
        dictRef['fluxerr'] = [1.3821321678798889e-07, 1.5590732755617273e-07, 1.8192455445555648e-07, 2.0285059931323458e-07, 2.1555825057784464e-07, 2.2069012111866136e-07, 2.1798814946172864e-07, 2.0933344110851897e-07, 1.9703905993883528e-07, 1.8746494699300317e-07, 1.8107622426999082e-07, 1.7512278927588011e-07, 1.6918224778846632e-07, 1.6314139119634574e-07, 1.5663895504416641e-07, 1.5080545075808596e-07,
                              1.2827790544421312e-07, 1.319367353776003e-07, 1.3780067391562013e-07, 1.444714740719694e-07, 1.4966887217260837e-07, 1.525119664284462e-07, 1.5264390402315115e-07, 1.5166109046153174e-07, 1.4931627272214047e-07, 1.4576103156036853e-07, 1.428392606613501e-07, 1.4067762168236507e-07, 1.3863793314338187e-07, 1.3674984997871739e-07, 1.3524128572955553e-07, 1.3384023244235643e-07, 1.3252059639045936e-07, 1.3149313102973464e-07]
        dictRef['phase'] = [-12.818181818180495, -10.090909090907767, -7.36363636363504, -4.636363636362313, -1.909090909089586, 0.8181818181831411, 3.545454545455868, 6.272727272728595, 9.000000000001322, 11.72727272727405, 14.454545454546777, 17.181818181819505, 19.90909090909223, 22.63636363636496, 25.363636363637685, 28.09090909091041, -13.499999999997575, -
                            10.999999999997575, -8.499999999997575, -5.999999999997575, -3.499999999997575, -0.9999999999975747, 1.5000000000024254, 4.000000000002426, 6.500000000002426, 9.000000000002427, 11.500000000002427, 14.000000000002427, 16.500000000002427, 19.000000000002427, 21.500000000002427, 24.000000000002427, 26.500000000002427, 29.000000000002427]
        dictRef['snr_m5'] = [32.6390492253349, 79.0032670408847, 140.17413262077946, 185.25022701771144, 211.32638455663877, 221.6281811730865, 216.2195550607211, 198.6593257277231, 173.02220770871315, 152.40425671747036, 138.27785116043069, 124.81548392089903, 111.0643389193895, 96.71962877103869, 80.82328247657294, 66.11128196326659, 4.207397508778439,
                             14.914578980533221, 31.496737279644844, 49.59200880174548, 63.190733909673284, 70.46219154424494, 70.79691356989717, 68.29782462032185, 62.28094913106561, 53.00467067969722, 45.23427232796791, 39.39485822841845, 33.81061613696503, 28.574309142344976, 24.342292084326132, 20.372118783571523, 16.5965228928586, 13.631890013638992]
        dictRef['time'] = [59009.0, 59012.0, 59015.0, 59018.0, 59021.0, 59024.0, 59027.0, 59030.0, 59033.0, 59036.0, 59039.0, 59042.0, 59045.0, 59048.0, 59051.0, 59054.0,
                           59009.0, 59012.0, 59015.0, 59018.0, 59021.0, 59024.0, 59027.0, 59030.0, 59033.0, 59036.0, 59039.0, 59042.0, 59045.0, 59048.0, 59051.0, 59054.0, 59057.0, 59060.0]
        dictRef['mag'] = [22.264347936828727, 21.17378667626179, 20.383667628776795, 19.962728012255248, 19.75376968891716, 19.676545933826414, 19.71674598749469, 19.85269669801393, 20.06843018222962, 20.26027269577972, 20.403530491878712, 20.551037795464264, 20.715242055501704, 20.904868666274634, 21.143974968002418, 21.40333407795156, 24.569652108525073,
                          23.165123792144367, 22.30627334873159, 21.76208111121057, 21.460603408714015, 21.321915453797978, 21.315831147725476, 21.361863008990856, 21.478909715330477, 21.68017668794674, 21.87427761744152, 22.04090422386863, 22.222727721721945, 22.420309128362387, 22.606388721881086, 22.8110084174069, 23.044314394722278, 23.266417772157325]

        for key in dictRef.keys():
            assert(np.isclose(dictRef[key], lc[key]).all())

    def testCalcSN(self):

        x1, color = -2.0, 0.2

        # Simulate LC
        lc = simuLCfast(x1, color, bands='griz')

        # instance of CalcSN

        sn = CalcSN(Table.from_pandas(lc), nPhamin=0, nPhamax=0).sn

        dictRef = {}

        """
        for col in sn.dtype.names:
            print('dictRef[\'', col, '\']=', sn[col].tolist())
        """
        dictRef['season'] = [1, 1]
        dictRef['healpixID'] = [10, 10]
        dictRef['pixRA'] = [0.0, 0.0]
        dictRef['pixDec'] = [0.0, 0.0]
        dictRef['z'] = [0.1, 0.2]
        dictRef['daymax'] = [59023.1, 59025.2]
        dictRef['n_bef'] = [20, 20]
        dictRef['n_aft'] = [44, 48]
        dictRef['n_phmin'] = [12, 16]
        dictRef['n_phmax'] = [0, 0]
        dictRef['Cov_x0x0'] = [
            1.2082989784919376e-13, 3.8313819364419196e-14]
        dictRef['Cov_x1x1'] = [0.00015952039943579143, 0.0019942190656627324]
        dictRef['Cov_colorcolor'] = [
            2.5922180044333713e-06, 2.0975146799526763e-05]

        for key in dictRef.keys():
            assert(np.isclose(dictRef[key], sn[key]).all())

    def testCalcSN_df(self):

        x1, color = -2.0, 0.2

        # Simulate LC
        lc = simuLCfast(x1, color, bands='griz')

        # instance of CalcSN

        # first case : estimated variance color only
        sn = CalcSN_df(lc, n_phase_min=0, n_phase_max=0,
                       invert_matrix=False).sn

        dictRef = {}
        """
        for col in sn.columns:
            print('dictRef[\'', col, '\']=', sn[col].to_list())
        """
        dictRef['color'] = [0.2, 0.2]
        dictRef['z'] = [0.1, 0.2]
        dictRef['daymax'] = [59023.1, 59025.2]
        dictRef['season'] = [1, 1]
        dictRef['healpixID'] = [10, 10]
        dictRef['pixRA'] = [0.0, 0.0]
        dictRef['pixDec'] = [0.0, 0.0]
        dictRef['level_8'] = [0, 0]
        dictRef['Cov_x0x0'] = [100.0, 100.0]
        dictRef['Cov_x1x1'] = [100.0, 100.0]
        dictRef['Cov_daymaxdaymax'] = [100.0, 100.0]
        dictRef['Cov_colorcolor'] = [
            2.630505097669433e-06, 2.109964241118505e-05]
        dictRef['n_aft'] = [44.0, 48.0]
        dictRef['n_bef'] = [20.0, 20.0]
        dictRef['n_phmin'] = [12.0, 16.0]
        dictRef['n_phmax'] = [12.0, 16.0]

        for key in dictRef.keys():
            assert(np.isclose(dictRef[key], sn[key]).all())

        # second case : estimated all variances from matrix inverting
        sn = CalcSN_df(lc, n_phase_min=0, n_phase_max=0,
                       invert_matrix=True).sn

        dictRef = {}
        """
        for col in sn.columns:
            print('dictRef[\'', col, '\']=', sn[col].to_list())
        """
        dictRef['color'] = [0.2, 0.2]
        dictRef['z'] = [0.1, 0.2]
        dictRef['daymax'] = [59023.1, 59025.2]
        dictRef['season'] = [1, 1]
        dictRef['healpixID'] = [10, 10]
        dictRef['pixRA'] = [0.0, 0.0]
        dictRef['pixDec'] = [0.0, 0.0]
        dictRef['level_8'] = [0, 0]
        dictRef['Cov_x0x0'] = [1.239736218613944e-13, 3.908852038544615e-14]
        dictRef['Cov_x1x1'] = [0.0001769038924859473, 0.0021595857993669952]
        dictRef['Cov_daymaxdaymax'] = [
            0.00014423656437498333, 0.0017679338165480843]
        dictRef['Cov_colorcolor'] = [
            2.6305050976694332e-06, 2.1099642411185038e-05]
        dictRef['n_aft'] = [44.0, 48.0]
        dictRef['n_bef'] = [20.0, 20.0]
        dictRef['n_phmin'] = [12.0, 16.0]
        dictRef['n_phmax'] = [12.0, 16.0]

        for key in dictRef.keys():
            assert(np.isclose(dictRef[key], sn[key]).all())

    def testCovColor(self):

        x1, color = -2.0, 0.2

        # Simulate LC
        lc = simuLCfast(x1, color, bands='griz')

        names = ['x1', 'color', 'z', 'daymax', 'season',
                 'healpixID', 'pixRA', 'pixDec']
        params = ['x0', 'x1', 'daymax', 'color']
        tosum = []
        for ia, vala in enumerate(params):
            for jb, valb in enumerate(params):
                if jb >= ia:
                    tosum.append('F_'+vala+valb)

        sums = lc.groupby(names)[tosum].sum()
        # sums = pd.DataFrame([lc.groupby(names)[tosum].sum()], columns=tosum)

        var_color = CovColor(sums).Cov_colorcolor

        var_ref = [2.630505097669433e-06, 2.109964241118505e-05]

        assert(np.isclose(var_ref, var_color.to_list()).all())


"""
if __name__ == "__main__":


"""
lsst.utils.tests.init()
snRate = TestSNRate
snTelescope = TestSNTelescope
snCadence = TestSNCadence
snUtil = TestSNUtils
calcFast = TestSNcalcFast
unittest.main(verbosity=5)
