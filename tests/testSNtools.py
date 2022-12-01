from builtins import zip
import numpy as np
import unittest
import lsst.utils.tests
from sn_tools.sn_rate import SN_Rate
from sn_tools.sn_utils import GenerateSample, Make_Files_for_Cadence_Metric, X0_norm
from sn_tools.sn_utils import SimuParameters
from sn_tools.sn_utils import DiffFlux, MbCov, GetReference, Gamma
from sn_tools.sn_cadence_tools import ReferenceData, GenerateFakeObservations
from sn_tools.sn_cadence_tools import TemplateData, AnaOS, Match_DD
from sn_tools.sn_calcFast import LCfast, CalcSN, CalcSN_df, CovColor
from sn_tools.sn_lcana import LCtoSN
from sn_tools.sn_obs import renameFields, patchObs, getPix, PavingSky, DDFields
from sn_tools.sn_obs import DataInside, proj_gnomonic_plane, proj_gnomonic_sphere
from sn_tools.sn_obs import pixelate, season, DataToPixels, ProcessPixels, ProcessArea, getFields
from sn_tools.sn_process import Process
from sn_tools.sn_clusters import ClusterObs
from sn_tools.sn_telescope import Telescope
from sn_tools.sn_visu import fieldType, SnapNight, CadenceMovie
from sn_tools.sn_io import Read_Sqlite
from sn_tools.sn_obs import getObservations
import os
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import h5py
from astropy.table import Table, vstack
import glob

m5_ref = dict(zip('ugrizy', [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))
main_repo = 'https://me.lsst.eu/gris'
ref_dir = 'Reference_Files'
db_dir = 'Scheduler_DB'


def getRefDir(dirname):
    fullname = '{}/{}/{}'.format(main_repo, ref_dir, dirname)

    if not os.path.exists(dirname):
        print('wget path:', fullname)
        cmd = 'wget --no-verbose --recursive {} --directory-prefix={} --no-clobber --no-parent -nH --cut-dirs=3 -R \'index.html*\''.format(
            fullname+'/', dirname)
        os.system(cmd)


def getRefFile(refdir, fname):
    fullname = '{}/{}/{}/{}'.format(main_repo, ref_dir, refdir, fname)

    # check whether the file is available; if not-> get it!
    if not os.path.isfile(fname):
        print('wget path:', fullname)
        cmd = 'wget --no-clobber --no-verbose {}'.format(fullname)
        os.system(cmd)


def getFile(dbDir, dbName, dbExtens, repmain, repofile=''):

    repo_reffiles = '{}/{}'.format(main_repo, repmain)
    if repofile != '':
        repo_reffiles = '{}/{}'.format(repo_reffiles, repofile)

    # check whether the file is available; if not-> get it!
    if not os.path.isfile('{}/{}.{}'.format(dbDir, dbName, dbExtens)):
        path = '{}/{}.{}'.format(repo_reffiles,
                                 dbName, dbExtens)
        print('wget path:', path)
        cmd = 'wget --no-clobber --no-verbose {}'.format(path)
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


def loadReference(x1, color):
    # first step: get reference files
    tel_par = {}
    tel_par['name'] = 'LSST'  # name of the telescope (internal)
    # dir of throughput
    tel_par['throughput_dir'] = 'LSST_THROUGHPUTS_BASELINE'
    tel_par['atmos_dir'] = 'THROUGHPUTS_DIR'  # dir of atmos
    tel_par['airmass'] = 1.2  # airmass value
    tel_par['atmos'] = True  # atmos
    tel_par['aerosol'] = False  # aerosol

    telescope = Telescope(name=tel_par['name'],
                          throughput_dir=tel_par['throughput_dir'],
                          atmos_dir=tel_par['atmos_dir'],
                          atmos=tel_par['atmos'],
                          aerosol=tel_par['aerosol'],
                          airmass=tel_par['airmass'])
    lc_reference = {}

    gamma_reference = 'gamma.hdf5'

    getRefFile('reference_files', gamma_reference)

    fDir = '.'
    fName = 'LC_{}_{}_vstack'.format(x1, color)
    fExtens = 'hdf5'

    getFile(fDir, fName, fExtens, ref_dir, 'Templates')
    fullname = '{}/{}.{}'.format(fDir, fName, fExtens)

    lc_ref = GetReference(
        fullname, gamma_reference, telescope)

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
    lc_ref = loadReference(x1, color)

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

    getRefDir('reference_files')

    gen_par = GenerateSample(
        sn_parameters, cosmo_parameters, mjdCol='observationStartMJD', dirFiles='reference_files')
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
        single_exposure_time = 30.
        nexp = 1.
        gamma = {}
        print(m5, zp)
        for key, val in m5.items():
            gamma[key] = tel.gamma(val, key, single_exposure_time, nexp)

        mag_to_flux_e_sec = {}
        mag_to_flux = {}

        for key, val in m5.items():
            mag_to_flux_e_sec[key] = tel.mag_to_flux_e_sec(
                val, key, single_exposure_time, nexp)
            mag_to_flux[key] = tel.mag_to_flux(
                val, key)

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
        gamma_ref = {'u': (0.03814092844062426, 17.930096969761752), 'g': (0.03873414409237359, 26.332644286375654), 'r': (0.038985607688313606, 32.86039626810443), 'i': (
            0.039091116657132316, 36.675040416255015), 'z': (0.03922084666663897, 42.78148075109106), 'y': (0.03929889517718481, 47.544008040748984)}
        mag_to_flux_e_sec_ref = {'u': (233.8708300403707, 17.930096969761752), 'g': (343.469273300552, 26.332644286375654), 'r': (428.61386436657955, 32.86039626810443), 'i': (
            478.37009238593504, 36.675040416255015), 'z': (558.019314144666, 42.78148075109106), 'y': (620.1392353141173, 47.544008040748984)}
        mag_to_flux_ref = {'u': 17.92625468, 'g': 26.32700139,
                           'r': 32.85335452, 'i': 36.66718122, 'z': 42.77231299, 'y': 47.5338197}

        for band in bands:
            for val in [(sigmab, sigmab_ref), (Tb, Tb_ref), (zp, zp_ref), (m5, m5_ref), (gamma, gamma_ref), (mag_to_flux, mag_to_flux_ref), (mag_to_flux_e_sec, mag_to_flux_e_sec_ref)]:
                assert_almost_equal(val[0][band], val[1][band])


class TestSNCadence(unittest.TestCase):

    def testReferenceData(self):
        # dirfiles = os.getenv('REF_FILES')
        dirfiles = 'reference_files'
        getRefDir(dirfiles)
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
        config['cadence'] = [3., 3., 3., 3., 3.]  # Cadence[day] per band
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

        band = 'r'
        z = 0.3
        min_rf_phase = -20
        max_rf_phase = 60.

        refname = 'LC_-2.0_0.2.hdf5'

        getRefDir('reference_files')
        templdata = TemplateData('reference_files/{}'.format(refname), band)

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

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        stat = AnaOS(dbDir, dbName, dbExtens, nclusters,
                     fields).stat

        # this is to get the reference data
        # print(stat.values.tolist(), stat.columns)

        valrefs = ['descddf_v1.4_10yrs_twoyears', 393929, 33174, 84973, 81143, 26010, 89868, 78761, 393929, 20513, 0.04949546619309819, 967, 3838, 1930, 1480, 1920, 10378, 20513, 3730.0, 272.0, 174.0, 348.0, 688.0, 1900.0, 348.0, 2.001879929708757, 1.883274130238675, 1.353426274694172, 2791.0, 248.0, 129.0, 254.0, 504.0, 1400.0, 256.0, 3.2066746992925483, 3.0166084454850193, 1.3534620445526713, 5313.0, 224.0, 256.0,
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
        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

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
        getRefDir('reference_files')
        genpar = GenerateSample(
            sn_parameters, cosmo_parameters, mjdCol='observationStartMJD', dirFiles='reference_files')

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

    def testSimuParameters(self):
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
        sn_parameters['x1_color']['rate'] = 'JLA'
        # x1
        sn_parameters['x1'] = {}
        sn_parameters['x1']['type'] = 'unique'
        sn_parameters['x1']['min'] = -2.0
        sn_parameters['x1']['max'] = 2.0
        sn_parameters['x1']['step'] = 0.2
        # color
        sn_parameters['color'] = {}
        sn_parameters['color']['type'] = 'unique'
        sn_parameters['color']['min'] = 0.2
        sn_parameters['color']['max'] = 0.3
        sn_parameters['color']['step'] = 0.1

        # DayMax
        sn_parameters['daymax'] = {}
        sn_parameters['daymax']['type'] = 'unique'
        sn_parameters['daymax']['step'] = 1.
        # Miscellaneous
        sn_parameters['min_rf_phase'] = -20.   # obs min phase (rest frame)
        sn_parameters['max_rf_phase'] = 60.  # obs max phase (rest frame)
        sn_parameters['min_rf_phase_qual'] = - \
            15.   # obs min phase (rest frame)
        sn_parameters['max_rf_phase_qual'] = 45.  # obs max phase (rest frame)
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
        getRefDir('reference_files')
        genpar = SimuParameters(
            sn_parameters, cosmo_parameters, mjdCol='observationStartMJD', dirFiles='reference_files')

        # get some observations
        observations = Observations()

        # get simulation parameters from these observations
        params = genpar.Params(observations)

        """
        for col in params.dtype.names:
            print('dictRef[\'{}\']='.format(col), params[col].tolist())
        """
        dictRef = {}
        dictRef['z'] = [0.1, 0.2, 0.30000000000000004]
        dictRef['daymax'] = [59023.1, 59025.2, 59027.3]
        dictRef['x1'] = [-2.0, -2.0, -2.0]
        dictRef['color'] = [0.2, 0.2, 0.2]
        dictRef['epsilon_x0'] = [0.0, 0.0, 0.0]
        dictRef['epsilon_x1'] = [0.0, 0.0, 0.0]
        dictRef['epsilon_color'] = [0.0, 0.0, 0.0]
        dictRef['epsilon_daymax'] = [0.0, 0.0, 0.0]
        dictRef['min_rf_phase'] = [-15.0, -15.0, -15.0]
        dictRef['max_rf_phase'] = [30.0, 30.0, 30.0]

        for key, vv in dictRef.items():
            assert(np.isclose(vv, params[key].tolist()).all())

    def testMake_Files_for_Cadence_Metric(self):

        telescope = Telescope(airmass=1.2)
        x1 = -2.0
        color = 0.2
        dbDir = '.'
        fName = 'LC_{}_{}'.format(x1, color)
        fExtens = 'hdf5'
        simulator_name = 'SNCosmo'

        getFile(dbDir, fName, fExtens, ref_dir, 'Templates')

        proc = Make_Files_for_Cadence_Metric(
            '{}/{}.{}'.format(dbDir, fName, fExtens), telescope, simulator_name)

        assert(os.path.isfile('Mag_to_Flux_LSST_{}.npy'.format(simulator_name)))
        assert(os.path.isfile('Li_{}_{}_{}.npy'.format(simulator_name, x1, color)))

    def testX0_norm(self):

        salt2Dir = 'SALT2_Files'

        getRefDir(salt2Dir)
        outFile = 'X0_norm.npy'
        X0_norm(salt2Dir=salt2Dir, outfile=outFile)

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
        lc_ref = loadReference(x1, color)
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
        if os.path.isfile(outName):
            os.system('rm {}'.format(outName))

        telescope = Telescope(airmass=1.2)

        mag_range = np.arange(20., 25., 1)
        nexps = range(1, 10, 1)
        single_exposure_time = [30.]

        Gamma(bands, telescope, outName,
              mag_range=mag_range,
              single_exposure_time=single_exposure_time, nexps=nexps)
        # check production
        fFile = h5py.File(outName, 'r')
        keys = list(fFile.keys())

        data = Table()
        for key in keys:
            data = vstack([data, Table.read(fFile, path=key)])
        """
        for col in data.columns:
            print('refDict[\'{}\']='.format(col), data[col].tolist())
        """
        refDict = {}

        refDict['band'] = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                           'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
        refDict['mag'] = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 22.0, 22.0, 22.0,
                          22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0]
        refDict['single_exptime'] = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0,
                                     30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        refDict['nexp'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2,
                           3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        refDict['gamma'] = [0.03998193106862388, 0.039990965534311945, 0.039993977022874626, 0.03999548276715597, 0.039996386213724776, 0.03999698851143731, 0.03999741872408913, 0.039997741383577985, 0.03999799234095821, 0.039954612896444454, 0.03997730644822223, 0.039984870965481485, 0.03998865322411111, 0.03999092257928889, 0.03999243548274074, 0.0399935161280635, 0.039994326612055556, 0.03999495698849383, 0.039885992750413296, 0.03994299637520665, 0.0399619975834711, 0.03997149818760332,
                            0.03997719855008266, 0.03998099879173555, 0.03998371325005904, 0.03998574909380166, 0.0399873325278237, 0.039713626736669436, 0.03985681336833472, 0.03990454224555648, 0.03992840668416736, 0.03994272534733389, 0.03995227112277824, 0.03995908953380992, 0.03996420334208368, 0.03996818074851883, 0.03928066288549283, 0.03964033144274642, 0.03976022096183095, 0.03982016572137321, 0.039856132577098566, 0.03988011048091547, 0.03989723755507041, 0.039910082860686605, 0.03992007365394365]
        refDict['flux_e_sec'] = [1844.7872007190508, 1844.7872007190508, 1844.7872007190508, 1844.7872007190508, 1844.787200719051, 1844.7872007190508, 1844.787200719051, 1844.7872007190508, 1844.787200719051, 734.423012751567, 734.423012751567, 734.4230127515671, 734.423012751567, 734.423012751567, 734.4230127515671, 734.4230127515671, 734.423012751567, 734.4230127515672, 292.3790675959012, 292.3790675959012, 292.37906759590123, 292.3790675959012,
                                 292.3790675959012, 292.37906759590123, 292.3790675959012, 292.3790675959012, 292.3790675959012, 116.39820332967382, 116.39820332967382, 116.39820332967382, 116.39820332967382, 116.39820332967382, 116.39820332967382, 116.39820332967383, 116.39820332967382, 116.39820332967383, 46.338959385087065, 46.338959385087065, 46.33895938508709, 46.338959385087065, 46.33895938508708, 46.33895938508709, 46.33895938508708, 46.338959385087065, 46.33895938508709]

        for key, val in refDict.items():
            if key not in ['band']:
                assert(np.isclose(data[key].tolist(), val).all())
            else:
                assert(data[key].tolist() == val)


class TestSNcalcFast(unittest.TestCase):

    def testLCfast(self):

        x1, color = -2.0, 0.2

        lc = simuLCfast(x1, color)

        # print(lc.columns)
        """
        for col in lc.columns:
            print(col, lc[col].values.tolist())
        """
        # These are what the result should be
        dictRef = {}

        dictRef['flux'] = [4.511147986335053e-06, 1.2317188232550995e-05, 2.550111662322939e-05, 3.7578119573455526e-05, 4.55531457559699e-05, 4.891115014639708e-05, 4.7133300685124935e-05, 4.158604026288241e-05, 3.409213315546674e-05, 2.8570455907048627e-05, 2.5038831188298558e-05, 2.1858035689046604e-05, 1.879011450752236e-05, 1.577897479370136e-05, 1.2660074510369872e-05, 9.969941676665338e-06,
                           5.397161397972984e-07, 1.967780860222931e-06, 4.340271623278295e-06, 7.16463061377825e-06, 9.4576858760202e-06, 1.074632739127063e-05, 1.0806717280098711e-05, 1.0358122558068462e-05, 9.29955918584795e-06, 7.726015475790287e-06, 6.461230015881104e-06, 5.541974962087858e-06, 4.687433939533106e-06, 3.907532488461168e-06, 3.2920828790886484e-06, 2.726609113336508e-06, 2.199381111769534e-06, 1.7924998997463632e-06]
        dictRef['fluxerr'] = [1.392078301131224e-07, 1.5855538429294586e-07, 1.867341224819313e-07, 2.0924362116801088e-07, 2.2286484029713726e-07, 2.2835723839164575e-07, 2.2546600672753466e-07, 2.1619643615366878e-07, 2.0300314387060767e-07, 1.9270479838749599e-07, 1.8581907285402417e-07, 1.7939130478141504e-07, 1.7296546725166e-07, 1.6641740835731415e-07, 1.593516146520308e-07, 1.5299534996930728e-07, 1.282492630747172e-07,
                              1.322944286542738e-07, 1.387543328679261e-07, 1.4607259278790437e-07, 1.5175482298157888e-07, 1.5485661548030596e-07, 1.5500045287161913e-07, 1.5392877531532006e-07, 1.5136982992469748e-07, 1.4748395311029048e-07, 1.4428471036593497e-07, 1.4191422660101663e-07, 1.3967454356583682e-07, 1.3759867053323253e-07, 1.3593814449537107e-07, 1.3439437253465088e-07, 1.3293886490742405e-07, 1.318046082162595e-07]
        dictRef['phase'] = [-12.818181818180495, -10.090909090907767, -7.36363636363504, -4.636363636362313, -1.909090909089586, 0.8181818181831411, 3.545454545455868, 6.272727272728595, 9.000000000001322, 11.72727272727405, 14.454545454546777, 17.181818181819505, 19.90909090909223, 22.63636363636496, 25.363636363637685, 28.09090909091041, -13.499999999997575, -
                            10.999999999997575, -8.499999999997575, -5.999999999997575, -3.499999999997575, -0.9999999999975747, 1.5000000000024254, 4.000000000002426, 6.500000000002426, 9.000000000002427, 11.500000000002427, 14.000000000002427, 16.500000000002427, 19.000000000002427, 21.500000000002427, 24.000000000002427, 26.500000000002427, 29.000000000002427]
        dictRef['snr_m5'] = [32.40584946025828, 77.68382188645099, 136.56377465611257, 179.59027550609252, 204.39808134488874, 214.18699267378446, 209.04836773058818, 192.35303320783584, 167.93894175942788, 148.2602205348223, 134.74844537604906, 121.84556946993732, 108.63506343831777, 94.81565029436335, 79.4474190802216, 65.16499801245874,
                             4.20833716200664, 14.87425343787795, 31.28026010841478, 49.04842501276887, 62.322143640655455, 69.39533941084555, 69.72055293960655, 67.29165834555658, 61.43601529098789, 52.38546508183618, 44.78111367097823, 39.05158133066383, 33.559686825277815, 28.398039554585882, 24.217506361510985, 20.288119672819686, 16.544304882557398, 13.599675489382772]
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
            print('dictRef[\'{}\']='.format(col), sn[col].tolist())
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
        dictRef['Cov_x0x0'] = [1.223597855577808e-13, 3.8627655216214425e-14]
        dictRef['Cov_x1x1'] = [0.00016174286076638242, 0.0020030002150465773]
        dictRef['Cov_colorcolor'] = [
            2.6208290665105743e-06, 2.1050920753550444e-05]

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
            print('dictRef[\'{}\']='.format(col), sn[col].to_list())
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
            2.6602780705309944e-06, 2.1176420857347407e-05]
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
            print('dictRef[\'{}\']='.format(col), sn[col].to_list())
        """

        dictRef['color'] = [0.2, 0.2]
        dictRef['z'] = [0.1, 0.2]
        dictRef['daymax'] = [59023.1, 59025.2]
        dictRef['season'] = [1, 1]
        dictRef['healpixID'] = [10, 10]
        dictRef['pixRA'] = [0.0, 0.0]
        dictRef['pixDec'] = [0.0, 0.0]
        dictRef['level_8'] = [0, 0]
        dictRef['Cov_x0x0'] = [1.2556727173847517e-13, 3.940921112025599e-14]
        dictRef['Cov_x1x1'] = [0.00017951638746726162, 0.002169144337622918]
        dictRef['Cov_daymaxdaymax'] = [
            0.0001469421723520269, 0.0017798915037562462]
        dictRef['Cov_colorcolor'] = [
            2.6602780705309914e-06, 2.1176420857347404e-05]
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

        # print(var_color.to_list())
        # var_ref = [2.630505097669433e-06, 2.109964241118505e-05]
        var_ref = [2.6602780705309944e-06, 2.1176420857347407e-05]
        assert(np.isclose(var_ref, var_color.to_list()).all())


class TestSNclusters(unittest.TestCase):
    def testClusters(self):
        # first grab some data
        fDir = '.'
        fName = 'descddf_v1.4_10yrs_DD_twoyears'
        fExtens = 'npy'

        getFile(fDir, fName, fExtens, ref_dir, 'unittests')
        data = np.load('{}/{}.{}'.format(fDir, fName, fExtens))
        nclusters = 5
        fields = DDFields()

        clus = ClusterObs(data, nclusters, fName, fields)

        # these are the clusters found and the number of visits associated
        clusters = clus.clusters
        dfclusters = clus.dfclusters
        dictRef = {}
        """
        for val in clusters.columns:
            print('dictRef[\'{}\']='.format(val), clusters[val].to_list())
        """

        dictRef['clusid'] = [0, 1, 2, 3, 4]
        dictRef['RA'] = [349.4901155249653, 35.74042365422678,
                         150.0614583198708, 9.500268366570257, 53.1644889383448]
        dictRef['Dec'] = [-63.26284809752175, -4.754220734785014,
                          2.1861679024847427, -43.954741632914974, -28.090789606070373]
        dictRef['width_RA'] = [3.0166084454850193, 1.3777211519657158,
                               1.3640734093042113, 1.883274130238675, 1.5566493003965718]
        dictRef['width_Dec'] = [1.3534620445526713, 1.3366522961266032,
                                1.3297419541315458, 1.353426274694172, 1.3366864617690055]
        dictRef['area'] = [3.2066746992925483, 1.4463375323299488,
                           1.424606742986726, 2.001879929708757, 1.6342188350701072]
        dictRef['dbName'] = ['descddf_v1.4_10yrs_DD_twoyears', 'descddf_v1.4_10yrs_DD_twoyears',
                             'descddf_v1.4_10yrs_DD_twoyears', 'descddf_v1.4_10yrs_DD_twoyears', 'descddf_v1.4_10yrs_DD_twoyears']
        dictRef['fieldName'] = ['SPT', 'XMM-LSS', 'COSMOS', 'ELAIS', 'CDFS']
        dictRef['Nvisits'] = [2791.0, 3534.0, 5313.0, 3701.0, 5145.0]
        dictRef['Nvisits_all'] = [2791.0, 3534.0, 5313.0, 3701.0, 5145.0]
        dictRef['Nvisits_u'] = [248.0, 272.0, 224.0, 272.0, 464.0]
        dictRef['Nvisits_g'] = [129.0, 168.0, 256.0, 174.0, 240.0]
        dictRef['Nvisits_r'] = [254.0, 336.0, 512.0, 348.0, 480.0]
        dictRef['Nvisits_i'] = [504.0, 669.0, 1017.0, 688.0, 960.0]
        dictRef['Nvisits_z'] = [1400.0, 1757.0, 2796.0, 1875.0, 2525.0]
        dictRef['Nvisits_y'] = [256.0, 332.0, 508.0, 344.0, 476.0]

        # transform in pandas df
        dfRef = pd.DataFrame.from_dict(dictRef)
        dfRef = dfRef.sort_values(by=['fieldName'])
        clusters = clusters.sort_values(by=['fieldName'])

        for key in dictRef.keys():
            if key != 'dbName' and key != 'fieldName' and key != 'clusid':
                assert(np.isclose(dfRef[key], clusters[key].to_list()).all())


class TestSNio(unittest.TestCase):
    def testRead_Sqlite(test):

        dbDir = '.'
        # dbDir = '../../../../DB_Files'
        dbName = 'descddf_v1.4_10yrs'
        dbExtens = 'db'

        getFile(dbDir, dbName, dbExtens, db_dir)

        mydb = Read_Sqlite('{}/{}.{}'.format(dbDir, dbName, dbExtens))

        data = mydb.get_data()

        # the result should look like...

        # fields
        list_names = ['observationId', 'fieldRA', 'fieldDec', 'observationStartMJD', 'flush_by_mjd', 'visitExposureTime', 'filter', 'rotSkyPos', 'numExposures', 'airmass', 'seeingFwhm500', 'seeingFwhmEff', 'seeingFwhmGeom', 'skyBrightness', 'night', 'slewTime', 'visitTime', 'slewDistance',
                      'fiveSigmaDepth', 'altitude', 'azimuth', 'paraAngle', 'cloud', 'moonAlt', 'sunAlt', 'note', 'fieldId', 'proposalId', 'block_id', 'observationStartLST', 'rotTelPos', 'moonAz', 'sunAz', 'sunRA', 'sunDec', 'moonRA', 'moonDec', 'moonDistance', 'solarElong', 'moonPhase']

        assert(set(list_names) & set(data.dtype.names))
        # and the first ten rows
        sel = data[:10]
        """
        for val in ['observationId', 'fieldRA', 'fieldDec', 'observationStartMJD', 'visitExposureTime', 'filter', 'numExposures', 'airmass', 'seeingFwhmEff', 'night', 'fiveSigmaDepth', 'moonPhase']:
            print('dictref[\'{}\']='.format(val), list(sel[val]))
        """
        dictRef = {}
        dictRef['observationId'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dictRef['fieldRA'] = [294.69610457523424, 297.73470761378616, 300.8015282209298, 298.33025759010536,
                              295.86447426843716, 295.2748042335168, 292.84821409582105, 293.4172383906381, 296.45390700382, 298.92295476890104]
        dictRef['fieldDec'] = [4.470981704726506, 3.424593021459912, 2.6710247648042724, 0.7349342312086529, -1.0548373274097236,
                               1.6887725976707535, 0.06880920114729176, -2.740942073461976, -3.7881327358747447, -1.962286622086737]
        dictRef['observationStartMJD'] = [59853.98564382085, 59853.98605776711, 59853.986472101155, 59853.98688407738,
                                          59853.987296601976, 59853.98771060965, 59853.98812335334, 59853.98853774942, 59853.988953736996, 59853.98936709884]
        dictRef['visitExposureTime'] = [30.0, 30.0, 30.0,
                                        30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        dictRef['filter'] = ['z', 'z', 'z', 'z', 'z', 'z', 'z', 'z', 'z', 'z']
        dictRef['numExposures'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dictRef['airmass'] = [1.216565554827875, 1.202931111818778, 1.1971128946542062, 1.1679834102356819, 1.1454852302352632,
                              1.1783448302877841, 1.1598998915757128, 1.1284788996408028, 1.1169976312341776, 1.136780340451723]
        dictRef['seeingFwhmEff'] = [1.4387115552139256, 1.4290152930845152, 1.3208391201575307, 1.301460113062717,
                                    1.2590134922059089, 1.280560579390107, 1.3422508848200254, 1.3203147043219936, 1.1866657592669003, 1.1992314031411724]
        dictRef['night'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dictRef['fiveSigmaDepth'] = [22.053188476046824, 22.067990308683584, 22.171052465090437, 22.220696408039043,
                                     22.286646108182094, 22.277470821951862, 22.25145893460083, 22.29951756613952, 22.440812015331417, 22.44686362191603]
        dictRef['moonPhase'] = [42.296594824736324, 42.2985582061688, 42.30052513176788, 42.30248255550283,
                                42.304444274435966, 42.30641474593086, 42.308380896727535, 42.31035662168153, 42.312341650472945, 42.314315852761425]

        for key in dictRef.keys():
            if key != 'filter':
                assert(np.isclose(
                    dictRef[key], list(sel[key])).all())


class TestSNlcana(unittest.TestCase):
    def testLCtoSN(self):

        print('Test to be implemented')
        test_implemented = False
        assert(test_implemented == True)


class TestSNobs(unittest.TestCase):
    def testDDFields(self):

        DDF = DDFields()

        cols = ['name', 'fieldId', 'RA', 'Dec', 'fieldnum']
        assert(set(cols) == set(DDF.columns))
        # print(DDF.to_dict(orient='list'))
        dictRef = {}
        dictRef['name'] = ['ELAIS', 'SPT', 'COSMOS',
                           'XMM-LSS', 'CDFS', 'ADFS1', 'ADFS2']
        dictRef['fieldId'] = [744, 290, 2786, 2412, 1427, 290, 290]
        dictRef['RA'] = [10.0, 349.39, 150.36, 34.39, 53.0, 63.59, 58.97]
        dictRef['Dec'] = [-45.52, -63.32, 2.84, -5.09, -27.44, -47.59, -49.28]
        dictRef['fieldnum'] = [4, 5, 1, 2, 3, 6, 7]

        for key in dictRef.keys():
            if key != 'name':
                assert(np.isclose(
                    dictRef[key], list(DDF[key])).all())
            else:
                assert(set(dictRef[key]) & set(DDF[key]))

    def testpatchObs(self):

        def check(dictRef, tab):

            for key in dictRef.keys():
                if key != 'dbName' and key != 'fieldName':
                    assert(np.isclose(
                        np.sort(dictRef[key]), np.sort(tab[key].to_list())).all())
                else:
                    assert(set(dictRef[key]) & set(tab[key]))

        # get some observations
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_twoyears'
        dbExtens = 'npy'

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        obs = np.load('{}/{}.{}'.format(dbDir, dbName, dbExtens))

        # rename some of the fields
        obs = renameFields(obs)

        RAmin = 0.
        RAmax = 360.
        Decmin = -50.
        Decmax = 10.
        RACol = 'fieldRA'
        DecCol = 'fieldDec'

        nside = 128
        nclusters = 5
        radius = 4.
        obs_DD, patches_DD = patchObs(obs, 'DD',
                                      dbName=dbName,
                                      nside=nside,
                                      RAmin=RAmin,
                                      RAmax=RAmax,
                                      Decmin=Decmin,
                                      Decmax=Decmax,
                                      RACol=RACol,
                                      DecCol=DecCol,
                                      display=False,
                                      nclusters=nclusters,
                                      radius=radius)

        """
        for vv in patches_DD.columns:
            print('dictRef[\'{}\']='.format(vv), patches_DD[vv].to_list())
        """
        dictRef = {}

        dictRef['clusid'] = [0, 1, 2, 3, 4]
        dictRef['RA'] = [349.4901155249653, 53.1644889383448,
                         150.0614583198708, 9.504023911471458, 35.74042365422678]
        dictRef['Dec'] = [-63.26284809752175, -28.090789606070373,
                          2.1861679024847427, -43.95381458076769, -4.754220734785014]
        dictRef['radius_RA'] = [4.0, 4.0, 4.0, 4.0, 4.0]
        dictRef['radius_Dec'] = [4.0, 4.0, 4.0, 4.0, 4.0]
        dictRef['area'] = [3.2066746992925483, 1.6342188350701072,
                           1.424606742986726, 2.001879929708757, 1.4463375323299488]
        dictRef['dbName'] = ['descddf_v1.4_10yrs_twoyears', 'descddf_v1.4_10yrs_twoyears',
                             'descddf_v1.4_10yrs_twoyears', 'descddf_v1.4_10yrs_twoyears', 'descddf_v1.4_10yrs_twoyears']
        dictRef['fieldName'] = ['SPT', 'CDFS', 'COSMOS', 'ELAIS', 'XMM-LSS']
        dictRef['Nvisits'] = [2791.0, 5145.0, 5313.0, 3730.0, 3534.0]
        dictRef['Nvisits_all'] = [2791.0, 5145.0, 5313.0, 3730.0, 3534.0]
        dictRef['Nvisits_u'] = [248.0, 464.0, 224.0, 272.0, 272.0]
        dictRef['Nvisits_g'] = [129.0, 240.0, 256.0, 174.0, 168.0]
        dictRef['Nvisits_r'] = [254.0, 480.0, 512.0, 348.0, 336.0]
        dictRef['Nvisits_i'] = [504.0, 960.0, 1017.0, 688.0, 669.0]
        dictRef['Nvisits_z'] = [1400.0, 2525.0, 2796.0, 1900.0, 1757.0]
        dictRef['Nvisits_y'] = [256.0, 476.0, 508.0, 348.0, 332.0]
        dictRef['radius'] = [4.0, 4.0, 4.0, 4.0, 4.0]

        check(dictRef, patches_DD)

        nside = 64
        nclusters = -1
        radius = 4.
        obs_WFD, patches_WFD = patchObs(obs, 'WFD',
                                        dbName=dbName,
                                        nside=nside,
                                        RAmin=RAmin,
                                        RAmax=RAmax,
                                        Decmin=Decmin,
                                        Decmax=Decmax,
                                        RACol=RACol,
                                        DecCol=DecCol,
                                        display=False,
                                        nclusters=nclusters,
                                        radius=radius)
        """
        for vv in patches_WFD.columns:
            print('dictRef[\'{}\']='.format(vv),
                  patches_WFD[:30][vv].to_list())
        """
        dictRef = {}
        dictRef['RA'] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                         2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
        dictRef['Dec'] = [-48.0, -44.0, -40.0, -36.0, -32.0, -28.0, -24.0, -20.0, -16.0, -12.0, -8.0, -4.0, 0.0,
                          4.0, 8.0, 12.0, -48.0, -44.0, -40.0, -36.0, -32.0, -28.0, -24.0, -20.0, -16.0, -12.0, -8.0, -4.0, 0.0, 4.0]
        dictRef['radius_RA'] = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        dictRef['radius_Dec'] = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        dictRef['minRA'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        dictRef['maxRA'] = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                            4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
        dictRef['minDec'] = [-50.0, -46.0, -42.0, -38.0, -34.0, -30.0, -26.0, -22.0, -18.0, -14.0, -10.0, -6.0, -2.0,
                             2.0, 6.0, 10.0, -50.0, -46.0, -42.0, -38.0, -34.0, -30.0, -26.0, -22.0, -18.0, -14.0, -10.0, -6.0, -2.0, 2.0]
        dictRef['maxDec'] = [-46.0, -42.0, -38.0, -34.0, -30.0, -26.0, -22.0, -18.0, -14.0, -10.0, -6.0, -2.0, 2.0,
                             6.0, 10.0, 14.0, -46.0, -42.0, -38.0, -34.0, -30.0, -26.0, -22.0, -18.0, -14.0, -10.0, -6.0, -2.0, 2.0, 6.0]

        check(dictRef, patches_WFD[:30])

    def testgetPix(self):

        nside = 128
        fieldRA = 50.
        fieldDec = -40.

        healpixId, pixRA, pixDec = getPix(nside, fieldRA, fieldDec)

        # print(healpixId, pixRA, pixDec)
        healpixId_ref = 137931
        pixRA_ref = 49.921875
        pixDec_ref = -39.838439

        assert(healpixId == healpixId_ref)
        assert(np.isclose(pixRA, pixRA_ref))
        assert(np.isclose(pixDec, pixDec_ref))

    def testPavingSky(self):

        minRA = 20.
        maxRA = 30.
        minDec = -15.
        maxDec = 10.
        radius_RA = 5.
        radius_Dec = 5.

        sky = PavingSky(minRA, maxRA, minDec, maxDec, radius_RA, radius_Dec)

        patches = sky.patches

        dictRef = {}
        """
        for col in patches.dtype.names:
            print('dictRef[\'{}\']='.format(col), patches[col].tolist())
        """

        dictRef['RA'] = [22.5, 22.5, 22.5, 22.5,
                         22.5, 27.5, 27.5, 27.5, 27.5, 27.5]

        dictRef['Dec'] = [-12.5, -7.5, -2.5, 2.5,
                          7.5, -12.5, -7.5, -2.5, 2.5, 7.5]
        dictRef['radius_RA'] = [5.0, 5.0, 5.0,
                                5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        dictRef['radius_Dec'] = [5.0, 5.0, 5.0,
                                 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        dictRef['minRA'] = [20.0, 20.0, 20.0, 20.0,
                            20.0, 25.0, 25.0, 25.0, 25.0, 25.0]
        dictRef['maxRA'] = [25.0, 25.0, 25.0, 25.0,
                            25.0, 30.0, 30.0, 30.0, 30.0, 30.0]
        dictRef['minDec'] = [-15.0, -10.0, -5.0,
                             0.0, 5.0, -15.0, -10.0, -5.0, 0.0, 5.0]
        dictRef['maxDec'] = [-10.0, -5.0, 0.0,
                             5.0, 10.0, -10.0, -5.0, 0.0, 5.0, 10.0]

        for col in patches.dtype.names:
            assert(np.isclose(dictRef[col], patches[col].tolist()).all())

        # visual check
        # import matplotlib.pyplot as plt
        # sky.plot()

    def testDataInside(self):

        # get some observations
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_twoyears'
        dbExtens = 'npy'

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        obs = np.load('{}/{}.{}'.format(dbDir, dbName, dbExtens))

        # rename some of the fields
        obs = renameFields(obs)

        RA = 20.
        widthRA = 10
        Dec = -50
        widthDec = 20

        dataIns = DataInside(obs, RA, Dec, widthRA, widthDec)
        dictArea = {'minRA': 10.0, 'maxRA': 30.0, 'minDec': -70, 'maxDec': -30}
        assert(dictArea.keys() & dataIns.areas[0].keys())
        assert(dictArea.items() & dataIns.areas[0].items())

        """
        # visual check
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        dataIns.plot(ax)
        plt.show()
        """

    def testProj_gnomo(self):

        lamb0 = np.deg2rad(-100.)
        phi1 = np.deg2rad(40.)
        lamb = np.deg2rad(-110.)
        phi = np.deg2rad(30.)

        x, y = proj_gnomonic_plane(lamb0, phi1, lamb, phi)

        xref = -0.1542826082868
        yref = -0.1694738770630
        assert(np.isclose(x, xref))
        assert(np.isclose(y, yref))

        lamb_n, phi_n = proj_gnomonic_sphere(lamb0, phi1, x, y)
        assert(np.isclose(lamb, lamb_n))
        assert(np.isclose(phi, phi_n))

    def testPixelate(self):

        # get some observations
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_twoyears'
        dbExtens = 'npy'

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        obs = np.load('{}/{}.{}'.format(dbDir, dbName, dbExtens))

        # rename some of the fields
        obs = renameFields(obs)

        # select a subset of the data

        obs = obs[:10]

        nside = 128
        pixels = pixelate(obs, nside, RACol='fieldRA', DecCol='fieldDec')

        dictRef = {}
        """
        for name in pixels.dtype.names:
            print('dictRef[\'{}\']='.format(name), pixels[name].tolist())
        """
        dictRef['fieldRA'] = [294.69610457523424, 297.73470761378616, 300.8015282209298, 298.33025759010536,
                              295.86447426843716, 295.2748042335168, 292.84821409582105, 293.4172383906381, 296.45390700382, 298.92295476890104]
        dictRef['fieldDec'] = [4.470981704726506, 3.424593021459912, 2.6710247648042724, 0.7349342312086529, -1.0548373274097236,
                               1.6887725976707535, 0.06880920114729176, -2.740942073461976, -3.7881327358747447, -1.962286622086737]
        dictRef['healpixID'] = [121956, 120569, 120704, 120513,
                                120455, 120508, 120490, 119768, 119669, 120347]
        dictRef['pixRA'] = [294.609375, 297.7734375, 300.9375, 298.4765625,
                            296.015625, 295.3125, 292.8515625, 293.203125, 296.3671875, 298.828125]
        dictRef['pixDec'] = [4.480798785982216, 3.5833216984719627, 2.686724185691588, 0.5968418305070173, -
                             0.8952829865701233, 1.49224628962034, 0.0, -2.6867241856916024, -3.583321698471977, -2.0893716714986112]
        dictRef['ebv'] = [0.419977605342865, 0.18318484723567963, 0.10688654333353043, 0.1799226850271225, 0.1963985562324524,
                          0.42131543159484863, 0.3438258171081543, 0.39373722672462463, 0.2425287663936615, 0.3707665205001831]

        for key in dictRef.keys():
            assert(np.isclose(dictRef[key], np.copy(
                pixels)[key].tolist()).all())

        diff_RA = list(
            map(float.__sub__, dictRef['fieldRA'], dictRef['pixRA']))

        diff_Dec = list(
            map(float.__sub__, dictRef['fieldDec'], dictRef['pixDec']))

        # visu check
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(dictRef['pixRA'], dictRef['pixDec'], 'ro', mfc=None)
        # ax.plot(dictRef['fieldRA'], dictRef['fieldDec'], 'k*')
        # plt.show()

    def testSeason(self):
        # get some observations
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_DD'
        dbExtens = 'npy'

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        obs = np.load('{}/{}.{}'.format(dbDir, dbName, dbExtens))

        # rename some of the fields
        obs = renameFields(obs)

        # select a field (here COSMOS)
        idx = np.abs(obs['fieldRA']-150.36) < 4.
        idx &= np.abs(obs['fieldDec']-2.84) < 4.

        # get seasons
        obs = season(obs[idx])

        diff = np.diff(obs['season'])
        flag = np.where(diff > 0)[0]

        fortest = obs[flag+1]
        dictRef = {}
        varnames = ['fieldRA', 'fieldDec', 'observationStartMJD', 'season']
        """
        for what in varnames:
            print('dictRef[\'{}\']='.format(what), fortest[what].tolist())
        """
        dictRef['fieldRA'] = [150.29170876551157, 150.36537175642601, 149.41838100971165, 150.21494643771322,
                              150.42962747936306, 150.26403688022518, 149.86520912950522, 150.31111909947305, 149.89980741941818]
        dictRef['fieldDec'] = [2.5454172590465762, 2.1041838251395437, 2.309646398454121, 1.8293401068476973,
                               2.258112925296989, 2.4145625605821475, 2.350960616183958, 2.4306675623352088, 1.5295098852165476]
        dictRef['observationStartMJD'] = [60281.310608226704, 60649.31193012857, 61013.3125870484, 61377.311062612964,
                                          61744.312224690744, 62112.30832476759, 62474.30787504029, 62844.29449134687, 63204.30846311501]
        dictRef['season'] = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        for vv in varnames:
            assert(np.isclose(dictRef[vv], fortest[vv].tolist()).all())

    def testDataToPixelsandProcessPixels(self):

        # define a metric class (requested to test ProcessPixels)

        class metric:
            def __init__(self, name):

                self.name = name

            def run(self, datapixels):
                tab = pd.DataFrame(np.copy(datapixels))
                summ = tab.groupby(['season']).sum().reset_index()
                med = tab.groupby(['season']).median().reset_index()
                return pd.DataFrame({'healpixID': med['healpixID'],
                                     'season': med['season'],
                                     'Nvisits': summ['numExposures']})

        # get some observations
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_DD'
        dbExtens = 'npy'

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        obs = np.load('{}/{}.{}'.format(dbDir, dbName, dbExtens))

        # rename some of the fields
        obs = renameFields(obs)

        # select a field (here COSMOS)
        idx = np.abs(obs['fieldRA']-150.36) < 4.
        idx &= np.abs(obs['fieldDec']-2.84) < 4.

        # get seasons
        obs = season(obs[idx])

        ###################################################################
        # First step : test DataToPixels
        ####################################################################

        # instantiating DataToPixels class
        nside = 128
        RACol = 'fieldRA'
        DecCol = 'fieldDec'
        outDir = '.'
        data_pixels = DataToPixels(nside, RACol, DecCol, outDir, dbName)

        # get the pixels here
        RA = np.mean(obs['fieldRA'])
        Dec = np.mean(obs['fieldDec'])
        widthRA = 4.
        widthDec = 4.

        pixels = data_pixels(obs, RA, Dec, widthRA, widthDec)

        assert(len(pixels) == 68398)

        healpixIDs = list(np.unique(pixels['healpixID']))

        healpixIDs_ref = [108949, 108950, 108951, 108953, 108954, 108955, 108956, 108957, 108958, 108959, 108965, 108973, 108976, 108977, 108978, 108979, 108980, 108981, 108982, 108983, 108984, 108985, 108986, 108987, 108988, 108989, 108990, 108991, 108993, 108994, 108995, 108998, 109000, 109001, 109002, 109003, 109004, 109005, 109006, 109007, 109018, 109024, 109025, 109026, 109027, 109028, 109029,
                          109030, 109031, 109032, 109033, 109034, 109035, 109036, 109037, 109038, 109039, 109040, 109041, 109042, 109043, 109046, 109048, 109049, 109050, 109051, 109052, 109054, 109329, 109332, 109333, 109334, 109335, 109340, 109341, 109376, 109377, 109378, 109379, 109380, 109381, 109382, 109383, 109384, 109385, 109386, 109387, 109388, 109389, 109390, 109391, 109392, 109393, 109394, 109395, 109396, 109400]

        assert(set(healpixIDs) == set(healpixIDs_ref))

        idx = pixels['healpixID'] == 109026

        sel = pixels[idx]
        assert(len(sel) == 1397)

        indx = range(1, len(sel), 100)
        selb = sel.iloc[indx, :]

        """
        for col in selb.columns:
            print('dictRef[\'{}\']='.format(col), selb[col].tolist())
        """
        dictRef = {}
        dictRef['fieldRA'] = [149.4102, 149.5775, 149.6916, 149.7733, 149.8627, 149.9418,
                              150.016, 150.0894, 150.1559, 150.2423, 150.3238, 150.4181, 150.5116, 150.628]
        dictRef['fieldDec'] = [2.2899, 1.9401, 2.4913, 2.3169, 1.6148, 1.8242,
                               2.5419, 2.6691, 2.7623, 2.5537, 2.1937, 2.7137, 2.4346, 2.3416]
        dictRef['level_2'] = [16, 25, 15, 19, 31,
                              26, 16, 11, 12, 15, 22, 12, 18, 21]
        dictRef['healpixID'] = [109026, 109026, 109026, 109026, 109026, 109026,
                                109026, 109026, 109026, 109026, 109026, 109026, 109026, 109026]
        dictRef['pixRA'] = [150.1171875, 150.1171875, 150.1171875, 150.1171875, 150.1171875, 150.1171875,
                            150.1171875, 150.1171875, 150.1171875, 150.1171875, 150.1171875, 150.1171875, 150.1171875, 150.1171875]
        dictRef['pixDec'] = [1.7907846593289491, 1.7907846593289491, 1.7907846593289491, 1.7907846593289491, 1.7907846593289491, 1.7907846593289491, 1.7907846593289491,
                             1.7907846593289491, 1.7907846593289491, 1.7907846593289491, 1.7907846593289491, 1.7907846593289491, 1.7907846593289491, 1.7907846593289491]

        for key, vals in dictRef.items():
            assert(np.isclose(selb[key].tolist(), vals).all())

        # visu inspection
        # import matplotlib.pyplot as plt
        # data_pixels.plot(pixels, plt)

        #########################################################################
        # Second step : test ProcessPixels
        ########################################################################

        # instance of class

        metricList = [metric('mymetric')]
        ipoint = 0
        process = ProcessPixels(
            metricList, ipoint, outDir='.', dbName=dbName, saveData=True)

        process(pixels, np.copy(obs), 1)

        # check results
        # a file must have been generated
        finame = 'descddf_v1.4_10yrs_DD_mymetric_0.hdf5'
        assert(os.path.isfile(finame))

        # if yes, check what is inside (at least part of it)

        fFile = h5py.File(finame, 'r')
        keys = list(fFile.keys())

        data = Table()
        for key in keys:
            data = vstack([data, Table.read(fFile, path=key)])
            break

        healpixIDs = list(np.unique(data['healpixID']))

        healpixIDs_ref = [108958.0, 108977.0, 108978.0, 108979.0, 108980.0, 108981.0, 108982.0, 108983.0, 108984.0, 108985.0,
                          108986.0, 108987.0, 108988.0, 108989.0, 108990.0, 108991.0, 109002.0, 109024.0, 109025.0, 109026.0, 109027.0]

        assert(set(healpixIDs) & set(healpixIDs_ref))

        dictRef = {}
        """
        for col in data.colnames:
            print('dictRef[\'{}\']='.format(col), list(data[col]))
        """
        dictRef['season'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2,
                             3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dictRef['Nvisits'] = [2019, 1407, 2002, 2302, 1915, 1682, 1990, 1645, 1474, 2213, 1679, 1206, 1580, 1800, 1390, 1331, 1632, 1310, 1168, 1835, 468, 526, 451, 830, 584, 412, 568, 475, 532, 513, 1902, 1343, 1762, 2064, 1705, 1241, 1712, 1621, 1635, 2123, 2693, 2011, 2634, 2907, 2611, 2321, 2678, 2435, 2304, 3114, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 2829, 2036, 2805, 2931, 2725, 2291, 2756, 2761, 2465, 3154, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 567, 495, 327, 623, 536, 248, 436, 295, 411, 433, 1734, 1431, 1638, 1761, 1383, 1237, 1773, 1298, 1285, 1837, 296, 380, 287, 449, 328,
                              240, 332, 249, 321, 320, 1345, 1097, 1164, 1474, 1048, 872, 1217, 1214, 1214, 1518, 2915, 1963, 2692, 2846, 2728, 2292, 2718, 2624, 2319, 3109, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 2544, 1864, 2368, 2502, 2128, 1983, 2441, 2469, 2237, 2819, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 2765, 1899, 2685, 2962, 2731, 2400, 2726, 2545, 2438, 3002, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309]

        for key, vals in dictRef.items():
            assert(set(vals) & set(list(data[key])))

    def testProcessArea(self):

        # define a metric class

        class metric:
            def __init__(self, name):

                self.name = name

            def run(self, datapixels):
                tab = pd.DataFrame(np.copy(datapixels))
                summ = tab.groupby(['season']).sum().reset_index()
                med = tab.groupby(['season']).median().reset_index()
                df = pd.DataFrame(med['healpixID'], columns=['healpixID'])
                df['season'] = med['season']
                df['Nvisits'] = summ['numExposures']

                return df.to_records(index=False)

        # get some observations
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_DD'
        dbExtens = 'npy'

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        obs = np.load('{}/{}.{}'.format(dbDir, dbName, dbExtens))

        # rename some of the fields
        obs = renameFields(obs)

        # select a field (here COSMOS)
        idx = np.abs(obs['fieldRA']-150.36) < 4.
        idx &= np.abs(obs['fieldDec']-2.84) < 4.

        # get seasons
        obs = season(obs[idx])

        nside = 128
        RACol = 'fieldRA'
        DecCol = 'fieldDec'
        RA = np.mean(obs[RACol])
        Dec = np.mean(obs[DecCol])
        widthRA = 0.5
        widthDec = 0.5
        num = 1

        # instance of ProcessArea
        procArea = ProcessArea(nside, RACol, DecCol, num,
                               outDir='.', dbName=dbName, saveData=True)

        # instance of metric
        metricList = [metric('mymetric')]

        procArea(obs, metricList, RA, Dec, widthRA, widthDec, 1)

        # analyze the results
        # a file should have been produced
        finame = 'descddf_v1.4_10yrs_DD_mymetric_1.hdf5'
        assert(os.path.isfile(finame))

        # if yes, check what is inside (at least part of it)

        fFile = h5py.File(finame, 'r')
        keys = list(fFile.keys())

        data = Table()
        for key in keys:
            data = vstack([data, Table.read(fFile, path=key)])
            break

        dictRef = {}
        """
        for col in data.colnames:
            print('dictRef[\'{}\']='.format(col), list(data[col]))
        """
        dictRef['healpixID'] = [108981.0, 108981.0, 108981.0, 108981.0, 108981.0, 108981.0, 108981.0, 108981.0, 108981.0, 108981.0, 108982.0, 108982.0, 108982.0, 108982.0, 108982.0, 108982.0, 108982.0, 108982.0, 108982.0, 108982.0, 108983.0, 108983.0, 108983.0, 108983.0, 108983.0, 108983.0, 108983.0, 108983.0, 108983.0, 108983.0, 108988.0, 108988.0, 108988.0, 108988.0, 108988.0, 108988.0, 108988.0, 108988.0, 108988.0, 108988.0, 108989.0, 108989.0, 108989.0, 108989.0, 108989.0, 108989.0, 108989.0, 108989.0, 108989.0, 108989.0, 108991.0, 108991.0, 108991.0, 108991.0,
                                108991.0, 108991.0, 108991.0, 108991.0, 108991.0, 108991.0, 109002.0, 109002.0, 109002.0, 109002.0, 109002.0, 109002.0, 109002.0, 109002.0, 109002.0, 109002.0, 109003.0, 109003.0, 109003.0, 109003.0, 109003.0, 109003.0, 109003.0, 109003.0, 109003.0, 109003.0, 109024.0, 109024.0, 109024.0, 109024.0, 109024.0, 109024.0, 109024.0, 109024.0, 109024.0, 109024.0, 109025.0, 109025.0, 109025.0, 109025.0, 109025.0, 109025.0, 109025.0, 109025.0, 109025.0, 109025.0, 109026.0, 109026.0, 109026.0, 109026.0, 109026.0, 109026.0, 109026.0, 109026.0, 109026.0, 109026.0]
        dictRef['season'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2,
                             3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dictRef['Nvisits'] = [3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948,
                              3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309, 3089, 2224, 2948, 3147, 2982, 2512, 3025, 2897, 2656, 3309]

        for key, vals in dictRef.items():
            assert(set(vals) & set(list(data[key])))

    def testgetFields(self):

        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_twoyears'
        dbExtens = 'npy'
        nclusters = 5
        fieldType = 'DD'
        nside = 64

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        # loading observations

        observations = getObservations(dbDir, dbName, dbExtens)

        observations = renameFields(observations)

        fieldIds = [290, 744, 1427, 2412, 2786]
        observations = getFields(observations, fieldType, fieldIds, nside)

        sel = observations[:10]

        """
        names = ['observationId', 'numExposures', 'airmass',
                 'fieldRA', 'fieldDec', 'pixRA', 'pixDec']

        for name in names:
            print('dictRef[\'{}\']='.format(name), sel[name].tolist())
        """

        dictRef = {}
        dictRef['observationId'] = [324, 325, 326,
                                    327, 328, 329, 330, 331, 332, 333]
        dictRef['numExposures'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dictRef['airmass'] = [1.0966219814866531, 1.0958608026924606, 1.0924408700941448, 1.0917071597704562,
                              1.0909783310331298, 1.0902543705331267, 1.0870034369549904, 1.0863063803843198, 1.0856141067826373, 1.0849266035895995]
        dictRef['fieldRA'] = [10.323732389844585, 10.323732389844585, 10.323732389844585, 10.323732389844585,
                              10.323732389844585, 10.323732389844585, 10.323732389844585, 10.323732389844585, 10.323732389844585, 10.323732389844585]
        dictRef['fieldDec'] = [-44.11670006140491, -44.11670006140491, -44.11670006140491, -44.11670006140491, -
                               44.11670006140491, -44.11670006140491, -44.11670006140491, -44.11670006140491, -44.11670006140491, -44.11670006140491]
        dictRef['pixRA'] = [10.887096774193546, 10.887096774193546, 10.887096774193546, 10.887096774193546, 10.887096774193546,
                            10.887096774193546, 10.887096774193546, 10.887096774193546, 10.887096774193546, 10.887096774193546]
        dictRef['pixDec'] = [-43.40685848593699, -43.40685848593699, -43.40685848593699, -43.40685848593699, -
                             43.40685848593699, -43.40685848593699, -43.40685848593699, -43.40685848593699, -43.40685848593699, -43.40685848593699]

        for key in dictRef.keys():
            assert(np.isclose(dictRef[key], sel[key].tolist()).all())


class TestSNVisu(unittest.TestCase):

    def testfieldType(self):

        # get some observations
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_twoyears'
        dbExtens = 'npy'

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        obs = np.load('{}/{}.{}'.format(dbDir, dbName, dbExtens))

        obs = renameFields(obs)
        obs_type = fieldType(obs, 'fieldRA', 'fieldDec')

        idx = obs_type['fieldType'] == 'DD'
        nDD = len(obs[idx])
        nWFD = len(obs[~idx])
        assert(nDD == 13467)
        assert(nWFD == 400975)

    def testSnapNight(self):

        # get some observations
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_twoyears'
        dbExtens = 'npy'

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        snapnight = SnapNight(dbDir, dbName, saveFig=True,
                              areaTime=False, realTime=False)

        # with these options three figs must have been created
        for i in range(3):
            assert(os.path.isfile('{}_night_{}.png'.format(dbName, i+1)))

    def testCadenceMovie(self):
        # get some observations
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_twoyears'
        dbExtens = 'npy'

        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        cadencemovie = CadenceMovie(dbDir, dbName, saveFig=True,
                                    areaTime=False, realTime=False, saveMovie=False)
        # saveMovie = True crashes (environment problem)

        assert(os.path.isfile('{}_night_{}.png'.format(dbName, 1)))


class TestSNProcess(unittest.TestCase):

    def testProcess(self):

        class metric:
            def __init__(self):
                self.name = 'testMetric'

            def run(self, observations):
                var_med = ['numExposures', 'airmass', 'seeingFwhm500', 'seeingFwhmEff', 'seeingFwhmGeom',
                           'sky', 'night', 'fiveSigmaDepth', 'moonPhase', 'observationStartMJD', 'pixRA', 'pixDec']

                obsdf = pd.DataFrame(np.copy(observations))

                res = obsdf.groupby(['healpixID'])[
                    var_med].median().reset_index()

                return res

        # grab some data
        dbDir = '.'
        dbName = 'descddf_v1.4_10yrs_twoyears'
        dbExtens = 'npy'
        nclusters = 5
        nprocs = 4
        nside = 64
        RAmin = 30.
        RAmax = 32.
        Decmin = -47.
        Decmax = -45.
        remove_dithering = 0
        saveData = 1
        outDir = '.'
        fieldType = 'WFD'
        getFile(dbDir, dbName, dbExtens, ref_dir, 'unittests')

        mymetric = metric()
        metricList = [mymetric]

        Process(dbDir, dbName, dbExtens,
                fieldType, nside,
                RAmin, RAmax,
                Decmin, Decmax,
                saveData, remove_dithering,
                outDir, nprocs, metricList,
                pixelmap_dir='', npixels=0,
                nclusters=nclusters, radius=1.)

        # after this files dbName_metricname_*.hdf5 must have been generated in outDir

        fi = '{}/{}_{}_*.hdf5'.format(outDir, dbName, mymetric.name)

        fNames = glob.glob(fi)
        assert((len(fNames) > 0))

        data = Table()
        for fName in fNames:

            # check the content of this file
            fFile = h5py.File(fName, 'r')
            keys = list(fFile.keys())

            for key in keys:
                data = vstack([data, Table.read(fFile, path=key)])

        """
        for col in data.columns:
            print('dictRef[\'{}\']='.format(col), data[col].tolist())
        """
        dictRef = {}
        dictRef['healpixID'] = [35122, 35117, 35128]
        dictRef['numExposures'] = [1, 1, 1]
        dictRef['airmass'] = [1.0747176432719823,
                              1.0714165087126517, 1.0680236668903138]
        dictRef['seeingFwhm500'] = [0.6497019216104826,
                                    0.6620616111824902, 0.6280208668699482]
        dictRef['seeingFwhmEff'] = [0.868967529123832,
                                    0.8700945690021802, 0.8475501955312515]
        dictRef['seeingFwhmGeom'] = [0.7662913089397899,
                                     0.7672177357197921, 0.7486862607266888]
        dictRef['sky'] = [19.40994419707803,
                          19.787288276424675, 19.97947894641904]
        dictRef['night'] = [360, 360, 360]
        dictRef['fiveSigmaDepth'] = [23.26533308769796,
                                     23.394831087726786, 23.475629113872266]
        dictRef['moonPhase'] = [63.120351071229464,
                                57.1956151355722, 51.837966744094985]
        dictRef['observationStartMJD'] = [
            60209.26687181253, 60209.22450262563, 60209.21580961598]
        dictRef['pixRA'] = [31.810344827586203,
                            30.25862068965517, 31.271186440677965]
        dictRef['pixDec'] = [-46.5718474134496, -
                             46.5718474134496, -45.783967161775024]

        for key in dictRef.keys():
            assert(np.isclose(dictRef[key], data[key]).all())


class TestSNclean(unittest.TestCase):
    def testClean(self):

        def cmdrm(fi):
            return 'rm {}'.format(fi)

        def looprm(llist):
            for fi in llist:
                os.system(cmdrm(fi))

        fDir = '.'

        files_to_rm = []
        for extens in ['npy', 'hdf5', 'db', 'png']:
            searchf = glob.glob('{}/*.{}'.format(fDir, extens))
            if len(searchf) > 0:
                files_to_rm += searchf
        print('Cleaning - removing the following files:', files_to_rm)
        looprm(files_to_rm)

        for ddir in ['SALT2_Files', 'reference_files']:
            if os.path.exists(ddir):
                cmd = 'rm -rf {}'.format(ddir)
                os.system(cmd)


"""
if __name__ == "__main__":


"""
lsst.utils.tests.init()
snRate = TestSNRate
snTelescope = TestSNTelescope
snCadence = TestSNCadence
snUtil = TestSNUtils
calcFast = TestSNcalcFast
clusters = TestSNclusters
snio = TestSNio
lcana = TestSNlcana
snobs = TestSNobs
visu = TestSNVisu
process = TestSNProcess
clean = TestSNclean
unittest.main(verbosity=5)
