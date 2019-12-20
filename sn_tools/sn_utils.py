import numpy as np
from sn_tools.sn_rate import SN_Rate
import os
import numpy.lib.recfunctions as rf
from astropy.table import Table, vstack, Column
from sn_tools.sn_throughputs import Throughputs
from scipy import interpolate, integrate
from lsst.sims.photUtils import Sed, PhotometricParameters, Bandpass, Sed
import sncosmo
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline as Spline1d
from scipy.interpolate import griddata,interp2d
from sn_tools.sn_telescope import Telescope
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import multiprocessing
import pprint

class GenerateSample:
    """ Generates a sample of parameters for simulation

    Parameters
    -----------
    sn_parameters : dict
      supernovae parameters: x1, color, z, daymax, ...
    cosmo_parameters: dict
      cosmology parameters: H0, Om0
    mjdCol : str,opt
       name of the column corresponding to MJD
       Default : 'mjd'
    seasonCol : str, opt
      name of the column corresponding to season
      Default : 'season'
    filterCol : str, opt
      name of the column corresponding to filter
      Default : 'filter'
    min_rf_phase : float, opt
       min rest-frame phase for supernovae
       Default : -15.
    max_rf_phase : float, opt
       max rest-frame phase for supernovae
       Default : 30.
    area : float, opt
       area of the survey (in deg\^2)
       Default : 9.6 deg\^2

    """

    def __init__(self, sn_parameters, cosmo_parameters, mjdCol='mjd', seasonCol='season', filterCol='filter', min_rf_phase=-15., max_rf_phase=30., area=9.6, dirFiles='reference_files'):
        self.dirFiles = dirFiles
        self.params = sn_parameters
        self.sn_rate = SN_Rate(rate=self.params['z']['rate'],
                               H0=cosmo_parameters['H0'],
                               Om0=cosmo_parameters['Omega_m'])

        self.x1_color = self.getDist(self.params['x1_color']['rate'])
        self.mjdCol = mjdCol
        self.seasonCol = seasonCol
        self.filterCol = filterCol
        self.area = area
        self.min_rf_phase = min_rf_phase
        self.max_rf_phase = max_rf_phase

    def __call__(self, obs):
        """
        Compute set of parameters for simulation

        Parameters
        ---------
        array of observations

        Returns
        ---------
        array of sn parameters for simulation:
        z, float
          redshift
        x1, float
           x1 parameter
        color, float,
          color parameter
        daymax, float
          T0 parameter
        epsilon_x0,float
          epsilon for x0 parameter
        epsilon_x1, float
          epsilon for x1 parameter
        epsilon_color, float
          epsilon for color parameter
        min_rf_phase, float
          min rest-frame phase for LC points
        max_rf_phase, float
          max rest-frame phase for LC points
        """
        epsilon = 1.e-08

        r = []
        for season in np.unique(obs[self.seasonCol]):
            idx = (obs[self.seasonCol] == season) & (
                obs[self.filterCol] != 'u')
            sel_obs = obs[idx]
            # get duration of obs
            daymin = np.min(sel_obs[self.mjdCol])
            daymax = np.max(sel_obs[self.mjdCol])
            duration = daymax-daymin

            rp = self.getParameters(daymin, daymax, duration)
            if len(rp) > 0:
                r += rp
        print('Number of SN to simulate:', len(r))
        # print(r)
        if len(r) > 0:
            names = ['z', 'x1', 'color', 'daymax',
                     'epsilon_x0', 'epsilon_x1', 'epsilon_color',
                     'epsilon_daymax','min_rf_phase', 'max_rf_phase']
            types = ['f8']*len(names)
            #params = np.zeros(len(r), dtype=list(zip(names, types)))
            params = np.asarray(r, dtype=list(zip(names, types)))
            return params
        else:
            return None

    def getParameters(self, daymin, daymax, duration):
        """ Get parameters

        Parameters
        --------------
        daymin : float
           min MJD
        daymax, float
           max MJD
        duration : float
           duration of observations

        Returns
        ----------
        list of parameters
        z, float
          redshift
        x1, float
           x1 parameter
        color, float,
          color parameter
        daymax, float
          T0 parameter
        epsilon_x0,float
          epsilon for x0 parameter
        epsilon_x1, float
          epsilon for x1 parameter
        epsilon_color, float
          epsilon for color parameter
        epsilon_daymax, float
          epsilon for daymax(T0) parameter
        min_rf_phase, float
          min rest-frame phase for LC points
        max_rf_phase, float
          max rest-frame phase for LC points

        """

        # get z range
        zmin = self.params['z']['min']
        zmax = self.params['z']['max']
        print('zmin max',zmin,zmax)
        r = []
        epsilon = 1.e-8
        if self.params['z']['type'] == 'random':
            # get sn rate for this z range
            
            if zmin<1.e-6:
                zmin = 0.01
            print(zmin, zmax, duration, self.area)
            zz, rate, err_rate, nsn, err_nsn = self.sn_rate(
                zmin=zmin, zmax=zmax,
                duration=duration,
                survey_area=self.area,
                account_for_edges=True, dz=0.001)
            # get number of supernovae
            N_SN = int(np.cumsum(nsn)[-1])
            if np.cumsum(nsn)[-1] < 0.5:
                return r
            weight_z = np.cumsum(nsn)/np.sum(np.cumsum(nsn))

            if N_SN < 1:
                N_SN = 1
                #weight_z = 1
            print('nsn', N_SN)
            for j in range(N_SN):
                z = self.getVal(self.params['z']['type'], zmin, zz, weight_z)
                zrange = 'low_z'
                if z >= 0.1:
                    zrange = 'high_z'
                x1_color = self.getVal(self.params['x1_color']['type'],
                                       self.params['x1_color']['min'],
                                       self.x1_color[zrange][['x1', 'color']],
                                       self.x1_color[zrange]['weight'])
                T0_values = []
                if self.params['daymax']['type'] == 'unique':
                    T0_values = [daymin+21.*(1.+z)]
                if self.params['daymax']['type'] == 'random':
                    T0_values = np.arange(
                        daymin-(1.+z)*self.min_rf_phase, daymax-(1.+z)*self.max_rf_phase, 0.1)
                dist_daymax = T0_values
                # print('daymax',dist_daymax,type(dist_daymax))
                if dist_daymax.size == 0:
                    continue
                T0 = self.getVal(self.params['daymax']['type'],
                                 -1., dist_daymax,
                                 [1./len(dist_daymax)]*len(dist_daymax))
                r.append((z, x1_color[0], x1_color[1], T0, 0.,
                          0., 0.,0., self.min_rf_phase, self.max_rf_phase))

        if self.params['z']['type'] == 'uniform':
            zstep = self.params['z']['step']
            daystep = self.params['daymax']['step']
            x1_color = self.params['x1_color']['min']

            nz = int((zmax-zmin)/zstep)
            
            for z in np.linspace(zmin,zmax,nz+1):
                if z < 1.e-6:
                    z = 0.01
                if self.params['daymax']['type'] == 'uniform':
                    T0_min = daymin-(1.+z)*self.min_rf_phase
                    T0_max = daymax-(1.+z)*self.max_rf_phase
                    T0_min = daymin
                    T0_max = daymax
                    nT0 = int((T0_max-T0_min)/daystep)
                    widthWindow = T0_max-T0_min
                    if widthWindow < 1.:
                        break
                    #T0_values = np.linspace(T0_min,T0_max,nT0+1)
                    T0_values = np.arange(T0_min,T0_max,daystep)
                if self.params['daymax']['type'] == 'unique':
                    T0_values = [daymin+21.*(1.+z)]

                for T0 in T0_values:
                    r.append((z, x1_color[0], x1_color[1], T0, 0.,
                              0., 0., 0., self.min_rf_phase, self.max_rf_phase))

        if self.params['z']['type'] == 'unique':
            daystep = self.params['daymax']['step']
            x1_color = self.params['x1_color']['min']
            z = self.params['z']['min']
            if self.params['daymax']['type'] == 'uniform':
                T0_min = daymin-(1.+z)*self.min_rf_phase
                T0_max = daymax-(1.+z)*self.max_rf_phase
                nT0 = int((T0_max-T0_min)/daystep)
                T0_values = np.linspace(T0_min,T0_max,nT0+1)
            if self.params['daymax']['type'] == 'unique':
                T0_values = [daymin+21.*(1.+z)]
            for T0 in T0_values:
                r.append((z, x1_color[0], x1_color[1], T0, 0.,
                          0., 0., 0.,self.min_rf_phase, self.max_rf_phase))
        rdiff = []
        if self.params['differential_flux']:
            for rstart in r:
                for kdiff in [4, -4, 5, -5, 6, -6, 7,-7]:
                    rstartc = list(rstart)
                    rstartc[np.abs(kdiff)] = epsilon*np.sign(kdiff)
                    rdiff.append(tuple(rstartc))
        if rdiff:
            r += rdiff

        #print(r[:20])
        #print(test)
        return r

    def getVal(self, type, val, distrib, weight):
        """ Get values of a given parameter

        Parameters
        ------------
        type: str
           random or not
        val : float
            value to retuen (if not random)
        distrib : float
           distribution where to get the parameters
        weight : float
           weight corresponding to the distribution

        Returns
        ---------
        parameter value : float
        """

        if type == 'random':
            return np.random.choice(distrib, 1, p=weight)[0]
        else:
            return val

    def getDist(self, rate):
        """ get (x1,color) distributions

        Parameters
        ---------
        rate: str
            name of the x1_color distrib (JLA, ...)

        Returns
        ---------
        dict of (x1,color) rates
        keys : 'low_z' and 'high_z'
        values (float) : recarray with X1,Color,weight_X1,weight_Color,weight 
        """

        #prefix = os.getenv('SN_UTILS_DIR')+'/input/Dist_X1_Color_'+rate+'_'
        prefix = '{}/Dist_X1_Color_{}'.format(self.dirFiles, rate)
        suffix = '.txt'
        # names=['x1','c','weight_x1','weight_c','weight_tot']
        dtype = np.dtype([('x1', np.float), ('color', np.float),
                          ('weight_x1', np.float), ('weight_color', np.float),
                          ('weight', np.float)])
        x1_color = {}
        for val in ['low_z', 'high_z']:
            x1_color[val] = np.loadtxt('{}_{}{}'.format(
                prefix, val, suffix), dtype=dtype)

        return x1_color

    def plotParameters(self, gen_params):
        """ Plot the generated parameters
        (z,x1,color,daymax)

        Parameters
        --------------
        gen_params : array
          array of parameters
          should at least contain ['z', 'x1', 'color', 'daymax'] fields

        """
        import pylab as plt

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 9))

        todraw = ['z', 'x1', 'color', 'daymax']
        idx = dict(zip(todraw, [(0, 0), (0, 1), (1, 1), (1, 0)]))

        for name in todraw:
            i = idx[name][0]
            j = idx[name][1]
            ax[i][j].hist(gen_params[name], histtype='step')
            ax[i][j].set_xlabel(name)
            ax[i][j].set_ylabel('Number of entries')

        plt.show()


class Make_Files_for_Cadence_Metric:

    def __init__(self, file_name, telescope, simulator_name):
        """ Class to generate two files that will be used as input for the Cadence metric

        Parameters
        ---------
        LC file (filename)

        Returns
        ---------
        recordarray of LC:
        MJD, Ra, Dec, band,m5,Nexp, ExpTime, Season
        recordarray of mag_to_flux values
        """

        self.telescope = telescope
        self.simulator_name = simulator_name

        self.prod_mag_to_flux()
        if file_name != '':
            self.prod_(file_name)

    def prod_mag_to_flux(self):
        """
        Mag to flux estimation
        Output file created: Mag_to_Flux_simulator.npy
        """
        mag_range = (12., 28.0)
        m5 = np.linspace(mag_range[0], mag_range[1], 50)
        mag_to_flux_tot = None
        for band in 'grizy':
            mag_to_flux = np.array(m5, dtype=[('m5', 'f8')])
            exptime = [30.] * len(m5)
            b = [band] * len(m5)
            f5 = self.telescope.mag_to_flux_e_sec(m5, b, exptime)
            mag_to_flux = rf.append_fields(mag_to_flux, ['band', 'flux_e'], [
                b, f5[:, [1]]], dtypes=['U256', 'f8'])
            if mag_to_flux_tot is None:
                mag_to_flux_tot = mag_to_flux
            else:
                mag_to_flux_tot = np.concatenate(
                    (mag_to_flux_tot, mag_to_flux))
        # print(mag_to_flux_tot)
        # print('done')
        # np.save('Mag_to_Flux_'+self.simulator_name +
        # '.npy', np.copy(mag_to_flux_tot))
        np.save('Mag_to_Flux_LSST_sim' +
                '.npy', np.copy(mag_to_flux_tot))

    def prod_(self, filename):
        """
        Reformat LC input files to numpy array

        Parameters
        --------------
        filename : str
          name of the file with LC points (format: hdf5 ! )

        Returns
        ----------
        npy file produced 
        name of the file: Li_simulator_x1_color.npy
        numpy array with the following fields:
        time : float
           time (MJD)
         band : str
           band
         flux_e : float
           flux (in e/sec)
        flux : float
          flux (Jky?)
        z : float
          redshift
        daymax : float
          T0
        """
        import h5py
        f = h5py.File(filename, 'r')
        # print(f.keys())
        simu = {}
        for i, key in enumerate(f.keys()):
            # print(i)
            simu[i] = Table.read(filename, path=key)

        restot = None
        for key, val in simu.items():
            # print(val.meta)
            z = val.meta['z']
            x1 = val.meta['x1']
            color = val.meta['color']
            daymax = val.meta['daymax']
            idx = val['flux_e'] > 0.
            sel = val[idx]

            res = np.array(np.copy(sel[['time', 'band', 'flux_e', 'flux']]), dtype=[
                ('time', '<f8'), ('band', 'U8'), ('flux_e', '<f8'), ('flux', '<f8')])

            res = rf.append_fields(res, 'z', [z]*len(res))
            res = rf.append_fields(res, 'DayMax', [DayMax]*len(res))
            if restot is None:
                restot = res
            else:
                restot = np.concatenate((restot, res))

        # print(restot)
        np.save('Li_'+self.simulator_name+'_'+str(x1) +
                '_'+str(color)+'.npy', np.copy(restot))


class X0_norm:
    """ X0 estimations
         to be used as input for SN input params
         for simulation

        Parameters
        ----------
        salt2Dir: str, opt
          directory where to find SALT2 ref. files (default:'SALT2_Files')
        model: str, opt
          model use for SN (default:'salt2-extended')
        version: str, opt
          version used (default:'1.0')
        absmag: float, opt 
          absolute mag for the supernova (default:-19.0906)
        outfile: str,opt
         name/location of the output file (default:'reference_files/X0_norm.npy')

        Returns
        --------
        None
        The output file contains a numpy array with the following fields:
         x1: float, stretch
         color: float, color
         flux_10pc: float, flux at 10pc
         x0_norm: float, x0 value
    """

    def __init__(self, salt2Dir='SALT2_Files', model='salt2-extended', version='1.0', absmag=-19.0906, outfile='reference_files/X0_norm.npy'):

        self.salt2Dir = salt2Dir
        self.model = model
        self.version = version
        self.absmag = absmag

        if model == 'salt2-extended':
            model_min = 300.
            model_max = 180000.
            wave_min = 3000.
            wave_max = 11501.

        if model == 'salt2':
            model_min = 3400.
            model_max = 11501.
            wave_min = model_min
            wave_max = model_max

        self.wave = np.arange(wave_min, wave_max, 1.)

        # estimate flux at 10pc
        self.flux_10pc()
        source = sncosmo.get_source(self.model, version=self.version)
        self.SN = sncosmo.Model(source=source)
        r = []
        for x1 in np.arange(-3., 3., 0.01):
            for color in np.arange(-0.3, 0.3, 0.01):
                r.append((x1, color, self.flux_at_10pc, self.X0_norm(x1, color)))
        tab = np.rec.fromrecords(
            r, names=['x1', 'color', 'flux_10pc', 'x0_norm'])

        np.save(outfile, tab)

    def flux_10pc(self):
        """ Extimate flux at 10pc
        using Vega spectrum

        Parameters
        ----------
        None

        Returns
        --------
        None
        """
        name = 'STANDARD'
        band = 'B'
        #thedir = os.getenv('SALT2_DIR')
        thedir = self.salt2Dir

        os.environ[name] = thedir+'/Instruments/Landolt'

        self.trans_standard = Throughputs(through_dir='STANDARD',
                                          telescope_files=[],
                                          filter_files=['sb_-41A.dat'],
                                          atmos=False,
                                          aerosol=False,
                                          filterlist=('A'),
                                          wave_min=3559,
                                          wave_max=5559)

        mag, spectrum_file = self.getMag(
            thedir+'/MagSys/VegaBD17-2008-11-28.dat',
            np.string_(name),
            np.string_(band))

        sourcewavelen, sourcefnu = self.readSED_fnu(
            filename=thedir+'/'+spectrum_file)
        CLIGHT_A_s = 2.99792458e18         # [A/s]
        HPLANCK = 6.62606896e-27

        sedb = Sed(wavelen=sourcewavelen, flambda=sourcewavelen *
                   sourcefnu/(CLIGHT_A_s * HPLANCK))

        flux = self.calcInteg(
            bandpass=self.trans_standard.system['A'],
            signal=sedb.flambda,
            wavelen=sedb.wavelen)

        zp = 2.5*np.log10(flux)+mag
        self.flux_at_10pc = np.power(10., -0.4 * (self.absmag-zp))

    def X0_norm(self, x1, color):
        """ Extimate X0 from flux at 10pc
        using Vega spectrum

        Parameters
        ----------
        x1: float
         stretch of the supernova
        color: float
         color of the supernova


        Returns
        --------
        x0: float
          x0 from flux at 10pc
        """

        self.SN.set(z=0.)
        self.SN.set(t0=0)
        self.SN.set(c=color)
        self.SN.set(x1=x1)
        self.SN.set(x0=1)

        fluxes = 10.*self.SN.flux(0., self.wave)

        wavelength = self.wave/10.
        SED_time = Sed(wavelen=wavelength, flambda=fluxes)

        expTime = 30.
        photParams = PhotometricParameters(nexp=expTime/15.)
        trans = Bandpass(
            wavelen=self.trans_standard.system['A'].wavelen/10.,
            sb=self.trans_standard.system['A'].sb)
        # number of ADU counts for expTime
        e_per_sec = SED_time.calcADU(bandpass=trans, photParams=photParams)
        # e_per_sec = sed.calcADU(bandpass=self.transmission.lsst_atmos[filtre], photParams=photParams)
        e_per_sec /= expTime/photParams.gain*photParams.effarea

        return self.flux_at_10pc * 1.E-4 / e_per_sec

    def getMag(self, filename, name, band):
        """ Get magnitude in filename

        Parameters
        --------------
        filename: str
          name of the file to scan
        name: str
           throughtput used
        band: str
          band to consider

        Returns
        ----------
        mag: float
         mag
        spectrum_file: str
         spectrum file

        """
        sfile = open(filename, 'rb')
        spectrum_file = 'unknown'
        for line in sfile.readlines():
            if np.string_('SPECTRUM') in line:
                spectrum_file = line.decode().split(' ')[1].strip()
            if name in line and band in line:
                return float(line.decode().split(' ')[2]), spectrum_file

        sfile.close()

    def calcInteg(self, bandpass, signal, wavelen):
        """ Estimate integral of signal
        over wavelength using bandpass

        Parameters
        --------------
        bandpass: list(float)
          bandpass
        signal:  list(float)
          signal to integrate (flux)
        wavelength: list(float)
          wavelength used for integration

        Returns
        -----------
        integrated signal (float)
        """

        fa = interpolate.interp1d(bandpass.wavelen, bandpass.sb)
        fb = interpolate.interp1d(wavelen, signal)

        min_wave = np.max([np.min(bandpass.wavelen), np.min(wavelen)])
        max_wave = np.min([np.max(bandpass.wavelen), np.max(wavelen)])

        wavelength_integration_step = 5
        waves = np.arange(min_wave, max_wave, wavelength_integration_step)

        integrand = fa(waves) * fb(waves)

        range_inf = min_wave
        range_sup = max_wave
        n_steps = int((range_sup-range_inf) / wavelength_integration_step)

        x = np.core.function_base.linspace(range_inf, range_sup, n_steps)

        return integrate.simps(integrand, x=waves)

    def readSED_fnu(self, filename, name=None):
        """
        Read a file containing [lambda Fnu] (lambda in nm) (Fnu in Jansky).
        Extracted from sims/photUtils/Sed.py which does not seem to work

        Parameters
        --------------
        filename: str
          name of the file to process
        name: str,opt
          default: None

        Returns
        ----------
        sourcewavelen: list(float)
         wavelength with lambda in nm
        sourcefnu: list(float)
         signal with Fnu in Jansky
        """
        # Try to open the data file.
        try:
            if filename.endswith('.gz'):
                f = gzip.open(filename, 'rt')
            else:
                f = open(filename, 'r')
        # if the above fails, look for the file with and without the gz
        except IOError:
            try:
                if filename.endswith(".gz"):
                    f = open(filename[:-3], 'r')
                else:
                    f = gzip.open(filename+".gz", 'rt')
            except IOError:
                raise IOError(
                    "The throughput file %s does not exist" % (filename))
        # Read source SED from file
        # lambda, fnu should be first two columns in the file.
        # lambda should be in nm and fnu should be in Jansky.
        sourcewavelen = []
        sourcefnu = []
        for line in f:
            if line.startswith("#"):
                continue
            values = line.split()
            sourcewavelen.append(float(values[0]))
            sourcefnu.append(float(values[1]))
        f.close()
        # Convert to numpy arrays.
        sourcewavelen = np.array(sourcewavelen)
        sourcefnu = np.array(sourcefnu)
        return sourcewavelen, sourcefnu


class DiffFlux:
    def __init__(self, id_prod, dirFiles, outDir):
        """ Class to estimate flux derivatives wrt SN parameters (x0,x1,color)

        Parameters
        ----------
        id_prod: str
         production id
        dirFile: str
         directory where the files are located
        outDir: std
         output directory for the results

        """

        # check whether outputdir is ready
        if not os.path.isdir(outDir):
            os.makedirs(outDir)

        # SN parameters
        self.snParams = ['x0', 'x1', 'color']

        """
        # id of the production
        id_prod = '_'.join(metaFile.split(
            '/')[-1].split('.')[0].split('_')[1:])
        """
        # get metadata
        metaFile='{}/Simu_{}.hdf5'.format(dirFiles,id_prod)
        metaTable = self.metaData(metaFile)
        print(metaTable.dtype, len(metaTable))

        # get corresponding LC file
        self.lcFile = h5py.File('{}/LC_{}.hdf5'.format(dirFiles, id_prod), 'r')

        # Two output files, one for metadata and the other for LC with deriv. fluxes
        self.summaryOut = '{}/Simu_{}.hdf5'.format(outDir, id_prod)
        self.lcOut = '{}/LC_{}.hdf5'.format(outDir, id_prod)

        # Remove outputs if already exist
        if os.path.exists(self.summaryOut):
            os.remove(self.summaryOut)
        if os.path.exists(self.lcOut):
            os.remove(self.lcOut)

        # Make groups of lcs
        groups = metaTable.group_by(['z', 'x1', 'color', 'daymax'])

        # self.procMulti(groups[:81],8)
        self.procSimple(groups)

    def procSimple(self, groups):
        """ Process (astropy) groups

        Parameters
        ----------
        groups: astropy table groups
         each group is composed by a list of SN light curves

        """

        # This is for output metadata
        metaTot = []
        names = []

        # Loop on the groups
        for ii, gr in enumerate(groups.groups):

            # get differential fluxes
            lsdiff = self.diffFlux(gr)

            # Store the metadata
            metaTot.append(tuple([lsdiff.meta[key]
                                  for key in lsdiff.meta.keys()]))
            names = [key for key in lsdiff.meta.keys()]

            # Save the lightcurves
            lsdiff.write(self.lcOut, 'lc_{}'.format(
                lsdiff.meta['index_hdf5']), compression=True, append=True)

        # Save the metadata as an astropy table (hdf5)
        res = np.rec.fromrecords(metaTot, names=names)
        Table(res).write(self.summaryOut, 'summary', compression=True)

    """
    def procMulti(self,groups, nproc):

        result_queue = multiprocessing.Queue()
        ngr = len(groups)
        delta = int(ngr/nproc)
        batch = range(0, ngr, delta)
        if ngr not in batch:
            batch = np.append(batch, ngr)

        print(batch)
        for j in range(len(batch)-1):
            ida = batch[j]
            idb = batch[j+1]
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.procGroups, args=(
                groups[ida:idb],j, result_queue))
            p.start()

        resultdict = {}
        for i in range(len(batch)-1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        r = []
        names = []
        for key,vals in resultdict.items():
            r += vals[0]
            names = vals[1]
            for lc in vals[3]:
                lc.write(self.lcOut,'lc_{}'.format(lc.meta['index_hdf5']),compression=True,append=True)
            
        res = np.rec.fromrecords(r,names=names)
        Table(res).write(self.summaryOut,'summary',compression=True)


    def procGroups(self,group, j=0, output_q=None):
        
        metaTot = []
        names = []
        lclist = []
        for gr in group.groups:
            #print(len(gr),gr[['z','x1','color','daymax','epsilon_x0','epsilon_x1','epsilon_color']])
            #print(gr.dtype)
            lsdiff = self.diffFlux(gr)
            metaTot.append(tuple([lsdiff.meta[key] for key in lsdiff.meta.keys()]))
            names = lsdiff.keys()
            lclist.append(lsdiff)

        if output_q is not None:
            return ({j: (metaTot,names,lclist)})
        else:
            return (metaTot,names,lclist)
    """

    def metaData(self, metaFile):
        """ Get metadata

        Parameters
        ----------

        metaFile: str
         filename of the metadata

        Returns
        -------
        tabres : astropy Table
         astropy table with metadata

        """

        # open metadata file
        fMeta = h5py.File(metaFile, 'r')

        # loop on the keys, stack output
        tabres = Table()
        for i, key in enumerate(fMeta.keys()):
            tabres = vstack([tabres, Table.read(fMeta, path=key)])

        return tabres

    def diffFlux(self, tab):
        """ Evaluate flux derivatives wrt SN parameters (x0, x1, color)
        using df/dp = (f(p+h)-f(p-h))/2h

        Parameters
        ----------
        tab: astropy Table
         table of metadata

        Returns
        -------
        lcnom : astropy Table
         light curve with three additional columns:
         dx0 = dflux/dx0
         dx1 = dflux/dx1
         dcolor = dflux/dcolor

        """

        idx = True
        for par in self.snParams:
            idx &= (tab['epsilon_{}'.format(par)] < 1.e-10)&(tab['epsilon_{}'.format(par)] >=0.)
        
        # Get the "nominal" LC with epsilon_snpar == 0.
        lcnom = Table.read(self.lcFile, path='lc_{}'.format(tab['id_hdf5'][idx].data[0]))
       

        for i, par in enumerate(self.snParams):
            ida = tab['epsilon_{}'.format(par)] > 1.e-9
            idb = tab['epsilon_{}'.format(par)] < 0.
            lca = Table.read(
                self.lcFile, path='lc_{}'.format(tab['id_hdf5'][ida].data[0]))
            lcb = Table.read(
                self.lcFile, path='lc_{}'.format(tab['id_hdf5'][idb].data[0]))
            epsilon = lca.meta['epsilon_{}'.format(par)]
            diff = (lca['flux']-lcb['flux'])/(2.*epsilon)
            lcnom.add_column(Column(diff, name='d{}'.format(par)))

        return lcnom


class MbCov:
    def __init__(self, salt2Dir, paramNames=dict(zip(['x0', 'x1', 'color'], ['x0', 'x1', 'color'])),interp=True):
        """ Class to estimate covariance matrix with mb

        Parameters
        ----------
        salt2Dir : str
         director where SALT2 reference files are to be found

        """

        self.load(salt2Dir)
        self.paramNames = paramNames
        # self.transNames = dict(
        #    zip(['t0', 'x0', 'x1', 'c'], ['t0', 'x0', 'x1', 'color']))
        self.transNames = dict(map(reversed, paramNames.items()))
        self.interp = interp
        if self.interp:
            # check whether outputdir is ready
            self.ratName = '{}/RatInt_for_mb.npy'.format(salt2Dir)
            if not os.path.exists(self.ratName):
                print('You would like to use the fast calculation for mB')
                print('But you need a file that does not exist')
                print('Will create it for you (it will take ~25 min)')
                self.genRat_int()
                print('The file has been generated.')
            self.ratio_Int = np.load(self.ratName)

    def load(self, salt2Dir):

        # from F. Mondon 2017/10/20
        # wavelength limits for salt2 model
        wl_min_sal = 3000
        wl_max_sal = 7000

        # interpolation of TB and Trest
        #filt2 = np.genfromtxt('{}/snfit_data/Instruments/SNLS3-Landolt-model/sb-shifted.dat'.format(salt2Dir))
        filt2 = np.genfromtxt(
            '{}/Instruments/SNLS3-Landolt-model/sb-shifted.dat'.format(salt2Dir))
        filt2 = np.genfromtxt(
            '{}/Instruments/Landolt/sb_-41A.dat'.format(salt2Dir))
        wlen = filt2[:, 0]
        tran = filt2[:, 1]
        self.splB = Spline1d(wlen, tran, k=1, ext=1)

        # interpolation of ref spectrum
        #data = np.genfromtxt(thedir+'/snfit_data/MagSys/bd_17d4708_stisnic_002.ascii')
        data = np.genfromtxt(
            '{}/MagSys/bd_17d4708_stisnic_002.ascii'.format(salt2Dir))
        dispersion = data[:, 0]
        flux_density = data[:, 1]
        self.splref = Spline1d(dispersion, flux_density, k=1, ext=1)

        # interpolation of the spectrum model
        template_0 = np.genfromtxt(
            '{}/snfit_data/salt2-4/salt2_template_0.dat'.format(salt2Dir))
        template_1 = np.genfromtxt(
            '{}/snfit_data/salt2-4/salt2_template_1.dat'.format(salt2Dir))

        wlM0 = []
        M0 = []
        for i in range(len(template_0[:, 0])):
            if template_0[:, 0][i] == 0.0:
                wlM0.append(template_0[:, 1][i])
                M0.append(template_0[:, 2][i])
        self.splM0 = Spline1d(wlM0, M0, k=1, ext=1)

        wlM1 = []
        M1 = []
        for i in range(len(template_1[:, 0])):
            if template_1[:, 0][i] == 0.0:
                wlM1.append(template_1[:, 1][i])
                M1.append(template_1[:, 2][i])
        self.splM1 = Spline1d(wlM1, M1, k=1, ext=1)

        # computation of the integral
        dt = 100000
        self.xs = np.linspace(float(wl_min_sal), float(wl_max_sal), dt)
        self.dxs = (float(wl_max_sal-wl_min_sal)/(dt-1))

        self.I2 = np.sum(self.splref(self.xs)*self.xs*self.splB(self.xs)*self.dxs)

    def genRat_int(self):
        
        x1 = np.arange(-3.0,3.0,0.1)
        color = np.arange(-0.3,0.3,0.01)
        x1_all = np.repeat(x1,len(color))
        color_all = np.tile(color,len(x1))

        r = []
        mref = 9.907
        for (x1v,colorv) in zip(x1_all, color_all):
            r.append((x1v,colorv,self.ratInt(x1v,colorv),mref))

        tab= np.rec.fromrecords(r, names=['x1','color','ratioInt','mref'])

        np.save(self.ratName,tab)
        

    def ratInt(self,x1,color):
        """ Estimate a ratio of two sums requested to estimated mb

        Parameters
        ----------
        x1: float
         x1 of the supernova
        color: float
         color of the supernova

        Returns
        -------
        float
         ratio value

        """
        I1 = np.sum((self.splM0(self.xs)*10**-12+x1*self.splM1(self.xs)*10**-12)*(
            10**(-0.4*self.color_law_salt2(self.xs)*color))*self.xs*self.splB(self.xs)*self.dxs)
        
        return I1/self.I2
   
    def mB_interp(self, x0, x1, color):
        """ Estimate mB for supernovae

        Parameters
        ----------
        params: dict
         dict of parameters: x0, x1, color 

        Returns
        -------
        mb : float
         mb value

        """

        rat = griddata((self.ratio_Int['x1'],self.ratio_Int['color']),self.ratio_Int['ratioInt'],(x1,color),method='cubic')
        mb = -2.5*np.log10(x0*rat)+np.mean(self.ratio_Int['mref'])

        return mb

    def mB(self, params):
        """ Estimate mB for supernovae

        Parameters
        ----------
        params: dict
         dict of parameters: x0, x1, color 

        Returns
        -------
        mb : float
         mb value

        """


        rat = self.ratInt(params[self.paramNames['x1']],params[self.paramNames['color']])
        # computation of mb
        mref = 9.907
        mb = -2.5*np.log10(params[self.paramNames['x0']]*rat)+mref

        return mb

    def mB_old(self, params):
        """ Estimate mB for supernovae

        Parameters
        ----------
        params: dict
         dict of parameters: x0, x1, color 

        Returns
        -------
        mb : float
         mb value

        """

        #    I1=np.sum((splM0(xs)*10**-12+res.parameters[3]*splM1(xs)*10**-12)*(10**(-0.4*salt2source.colorlaw(xs)*res.parameters[4]))*xs*splB(xs)*dxs)
        I1 = np.sum((self.splM0(self.xs)*10**-12+params[self.paramNames['x1']]*self.splM1(self.xs)*10**-12)*(
            10**(-0.4*self.color_law_salt2(self.xs)*params[self.paramNames['color']]))*self.xs*self.splB(self.xs)*self.dxs)
        I2 = np.sum(self.splref(self.xs)*self.xs*self.splB(self.xs)*self.dxs)
        #print(I1, I2,params['x1'],params['c'])

        # computation of mb
        mref = 9.907
        mb = -2.5*np.log10(params[self.paramNames['x0']]*(I1/I2))+mref

        return mb
    """
    def calcInteg(self, bandpass, signal,wavelen):
        
        
        
        fa = interpolate.interp1d(bandpass.wavelen,bandpass.sb)
        fb = interpolate.interp1d(wavelen,signal)
        
        min_wave=np.max([np.min(bandpass.wavelen),np.min(wavelen)])
        max_wave=np.min([np.max(bandpass.wavelen),np.max(wavelen)])
        
        #print 'before integrand',min_wave,max_wave
        
        wavelength_integration_step=5
        waves=np.arange(min_wave,max_wave,wavelength_integration_step)
        
        integrand=fa(waves) *fb(waves) 
        #print 'rr',len(f2(wavelen)),len(wavelen),len(integrand)
        
        range_inf=min_wave
        range_sup=max_wave
        n_steps = int((range_sup-range_inf) / wavelength_integration_step)
        

        x = np.core.function_base.linspace(range_inf, range_sup, n_steps)
        #print len(waves),len(x)
        return integrate.simps(integrand,x=waves)

    def Get_Mag(self,filename,name,band):
        
        sfile=open(filename,'rb')
        spectrum_file='unknown'
        for line in sfile.readlines():
            if 'SPECTRUM' in line:
                spectrum_file=line.split(' ')[1].strip()
            if name in line and band in line:
                return float(line.split(' ')[2]),spectrum_file

        sfile.close()
    """

    def color_law_salt2(self, wl):
        """ Color law for SALT2

        """
        B_wl = 4302.57
        V_wl = 5428.55
        l = (wl-B_wl)/(V_wl-B_wl)
        l_lo = (2800.-B_wl)/(V_wl-B_wl)
        l_hi = (7000.-B_wl)/(V_wl-B_wl)
        a = -0.504294
        b = 0.787691
        c = -0.461715
        d = 0.0815619
        cst = 1-(a+b+c+d)
        cl = []
        for i in range(len(l)):
            if l[i] > l_hi:
                cl.append(-(cst*l_hi+l_hi**2*a+l_hi**3*b+l_hi**4*c+l_hi**5*d +
                            (cst+2*l_hi*a+3*l_hi**2*b+4*l_hi**3*c+5*l_hi**4*d)*(l[i]-l_hi)))
            if l[i] < l_lo:
                cl.append(-(cst*l_lo+l_lo**2*a+l_lo**3*b+l_lo**4*c+l_lo**5*d +
                            (cst+2*l_lo*a+3*l_lo**2*b+4*l_lo**3*c+5*l_lo**4*d)*(l[i]-l_lo)))
            if l[i] >= l_lo and l[i] <= l_hi:
                cl.append(-(cst*l[i]+l[i]**2*a+l[i]**3*b+l[i]**4*c+l[i]**5*d))
        return np.array(cl)

    def mbCovar(self, params, covar, vparam_names):
        """ mb covariance matrix wrt fit parameters

        Parameters
        ----------
        params: dict
         parameter values
        covar: matrix
         covariance matrix of the parameters
        vparam_names: list
         names of the parameters

        Returns
        -------
        res: dict
         final covariance dict

        """

        res = {}
        h_ref = 1.e-8
        Der = np.zeros(shape=(len(vparam_names), 1))

       
        par_var = params.copy()
        ider = -1
        for i, key in enumerate(vparam_names):
            h = h_ref
            if np.abs(par_var[key]) < 1.e-5:
                h = 1.e-10

            par_var[key] += h
            ider += 1
            Der[ider] = (self.mB(par_var)-self.mB(params))/h
            
            par_var[key] -= h

        Prod = np.dot(covar, Der)

        for i, key in enumerate(vparam_names):
            res['Cov_{}mb'.format(self.transNames[key])] = Prod[i, 0]
            """
            if key != 'c':
                res['Cov_'+key.upper()+'mb']=Prod[i,0]
            else:
               res['Cov_Colormb']=Prod[i,0] 
            """
        res['Cov_mbmb'] = np.asscalar(np.dot(Der.T, Prod))
        res['mb_recalc'] = self.mB(par_var)

        print(res)
        return res

    def mbCovar_int(self, params, covar, vparam_names):
        """ mb covariance matrix wrt fit parameters
            uses mb_int (griddata from template of mb)

        Parameters
        ----------
        params: dict
         parameter values
        covar: matrix
         covariance matrix of the parameters
        vparam_names: list
         names of the parameters

        Returns
        -------
        res: dict
         final covariance dict

        """

        res = {}
        h_ref = 1.e-8
        Der = np.zeros(shape=(len(vparam_names), 1))
        
        rt = []
        r = []
        for i, key in enumerate(vparam_names):
            r.append(params[key])
        r.append(0.0)
        rt.append(tuple(r))

        for i, key in enumerate(vparam_names):
            rot = list(rt[0])
            h = h_ref
            if np.abs(rot[i])<1.e-5:
                h = 1.e-10
            rot[i]+=h
            rot[-1]=h
            rt.append(tuple(rot))

        tabDiff = np.rec.fromrecords(rt,names=vparam_names+['h'])
        mbVals = self.mB_interp(tabDiff[self.paramNames['x0']],tabDiff[self.paramNames['x1']],tabDiff[self.paramNames['color']])
        tabDiff = rf.append_fields(tabDiff,'mB',mbVals)

        ider = -1
        for i,key in enumerate(vparam_names):
            ider += 1
            Der[ider] = (tabDiff['mB'][i+1]-tabDiff['mB'][0])/tabDiff['h'][i+1]
            

        Prod = np.dot(covar, Der)

        for i, key in enumerate(vparam_names):
            res['Cov_{}mb'.format(self.transNames[key])] = Prod[i, 0]
          
        res['Cov_mbmb'] = np.asscalar(np.dot(Der.T, Prod))
        res['mb_recalc'] = self.mB_interp(params[self.paramNames['x0']],params[self.paramNames['x1']],params[self.paramNames['color']]) 
        
        return res
    """
    def mbDeriv(self,params,vparam_names):
        
        res={}
        h=1.e-6
        #Der=np.zeros(shape=(len(vparam_names),1))
        Der={}

        #print params
        par_var=params.copy()
        ider=-1
        for i,key in enumerate(vparam_names):
            par_var[key]+=h
            ider+=1
            Der[key]=(self.mB(par_var)-self.mB(params))/h
            par_var[key]-=h

        return Der
    """

    def test(self):
        """ Test function

        To test whether this class is usable or not


        """

        """
        Salt2Model
        BEGIN_OF_FITPARAMS Salt2Model
        DayMax 53690.0336018 0.105513809169 
        Redshift 0.1178 0 F 
        Color -0.0664131339433 0.0234330339301 
        X0 0.00030732251016 8.89813428854e-06 
        X1 -0.0208012409076 0.160846457522 
        CovColorColor 0.00054910707917 -1 
        CovColorDayMax 0.00040528682468 -1 
        CovColorX0 -1.68238293879e-07 -1 
        CovColorX1 0.00114702847231 -1 
        CovDayMaxDayMax 0.0111331639253 -1 
        CovDayMaxX0 -2.94345317778e-07 -1 
        CovDayMaxX1 0.0131008809199 -1 
        CovX0X0 7.91767938168e-11 -1 
        CovX0X1 -7.23852420336e-07 -1 
        CovX1X1 0.0258715828973 
        """

        salt2_res = {}
        salt2_res['DayMax'] = 53690.0336018
        salt2_res['Color'] = -0.0664131339433
        salt2_res['X0'] = 0.00030732251016
        salt2_res['X1'] = -0.0208012409076
        salt2_res['Color'] = 0.0
        salt2_res['X1'] = 0.0
        salt2_res['CovColorColor'] = 0.00054910707917
        salt2_res['CovColorDayMax'] = 0.00040528682468
        salt2_res['CovColorX0'] = -1.68238293879e-07
        salt2_res['CovColorX1'] = 0.00114702847231
        salt2_res['CovDayMaxDayMax'] = 0.0111331639253
        salt2_res['CovDayMaxX0'] = -2.94345317778e-07
        salt2_res['CovDayMaxX1'] = 0.0131008809199
        salt2_res['CovX0X0'] = 7.91767938168e-11
        salt2_res['CovX0X1'] = -7.23852420336e-07
        salt2_res['CovX1X1'] = 0.0258715828973
        # salt2_res['']=
        vparam_names = [self.paramNames['t0'], self.paramNames['color'],
                        self.paramNames['x0'], self.paramNames['x1']]
        covar = np.zeros(shape=(len(vparam_names), len(vparam_names)))

        covar[0, 1] = salt2_res['CovColorDayMax']
        covar[0, 2] = salt2_res['CovDayMaxX0']
        covar[0, 3] = salt2_res['CovDayMaxX1']

        covar[1, 2] = salt2_res['CovColorX0']
        covar[1, 3] = salt2_res['CovColorX1']

        covar[2, 3] = salt2_res['CovX0X1']

        covar = covar+covar.T

        covar[0, 0] = salt2_res['CovDayMaxDayMax']
        covar[1, 1] = salt2_res['CovColorColor']
        covar[2, 2] = salt2_res['CovX0X0']
        covar[3, 3] = salt2_res['CovX1X1']

        # print covar

        params = {}
        params[self.paramNames['t0']] = salt2_res['DayMax']
        params[self.paramNames['color']] = salt2_res['Color']
        params[self.paramNames['x0']] = salt2_res['X0']
        params[self.paramNames['x1']] = salt2_res['X1']

        cov = self.mbCovar(params, covar, vparam_names)
        print(cov)
        if self.interp:
            cov_int = self.mbCovar_int(params, covar, vparam_names)
            print(cov_int)

class GetReference:

    def __init__(self, lcName,gammaName,tel_par,param_Fisher=['x0', 'x1', 'color','daymax']):
        """ Class to load reference data
            used for the fast SN simulator

        Parameters
        -----------
        lcName: str
         name of the reference file to load (lc)
        gammaName: str
         name of the reference file to load (gamma)
        tel_par: dict
         telescope parameters

        """""
        # Load the file - lc reference
        
        f = h5py.File(lcName, 'r')
        keys = list(f.keys())
        #lc_ref_tot = Table.read(filename, path=keys[0])
        lc_ref_tot = Table.from_pandas(pd.read_hdf(lcName))

        idx = lc_ref_tot['z']>0.005
        lc_ref_tot = np.copy(lc_ref_tot[idx])


        # telescope requested
        telescope = Telescope(name=tel_par['name'],
                              throughput_dir=tel_par['throughput_dir'],
                              atmos_dir=tel_par['atmos_dir'],
                              atmos=tel_par['atmos'],
                              aerosol=tel_par['aerosol'],
                              airmass=tel_par['airmass'])

        # Load the file - gamma values
        if not os.path.exists(gammaName):
            print('gamma file {} does not exist')
            print('will generate it - few minutes')
            Gamma('ugrizy',telescope,gammaName)
            print('end of gamma estimation')

        fgamma = h5py.File(gammaName, 'r')

        # Load references needed for the following
        self.lc_ref = {}
        self.gamma_ref = {}
        self.gamma = {}
        self.m5_ref = {}
        self.mag_to_flux_e_sec = {}

        self.flux = {}
        self.fluxerr = {}
        self.param = {}

        bands = np.unique(lc_ref_tot['band'])
        mag_range = np.arange(10., 38., 0.01)
        #exptimes = np.linspace(15.,30.,2)
        #exptimes = [15.,30.,60.,100.]

        #gammArray = self.loopGamma(bands, mag_range, exptimes,telescope)

        method = 'linear'

        # for each band: load data to be used for interpolation 
        for band in bands:
            idx = lc_ref_tot['band'] == band
            lc_sel = Table(lc_ref_tot[idx])
            
            lc_sel['z'] = lc_sel['z'].data.round(decimals=4)
            lc_sel['phase'] = lc_sel['phase'].data.round(decimals=4)

            
            fluxes_e_sec = telescope.mag_to_flux_e_sec(
                mag_range, [band]*len(mag_range), [30]*len(mag_range))
            self.mag_to_flux_e_sec[band] = interpolate.interp1d(
                mag_range, fluxes_e_sec[:, 1], fill_value=0., bounds_error=False)


            # these reference data will be used for griddata interp.
            self.lc_ref[band] = lc_sel
            self.gamma_ref[band] = lc_sel['gamma'][0]
            self.m5_ref[band] = np.unique(lc_sel['m5'])[0]

            # Another interpolator, faster than griddata: regulargridinterpolator

            # Fluxes and errors
            zmin, zmax, zstep, nz = self.limVals(lc_sel,'z')
            phamin, phamax, phastep, npha= self.limVals(lc_sel,'phase')

            zstep = np.round(zstep,3)
            phastep = np.round(phastep,3)
            
            zv = np.linspace(zmin,zmax,nz)
            #zv = np.round(zv,2)
            #print(band,zv)
            phav = np.linspace(phamin,phamax,npha)
            
            
            index = np.lexsort((lc_sel['z'],lc_sel['phase']))
            flux = np.reshape(lc_sel[index]['flux'],(npha,nz))
            fluxerr = np.reshape(lc_sel[index]['fluxerr'],(npha,nz))
            
            
            self.flux[band] = RegularGridInterpolator((phav,zv),flux,method=method,bounds_error=False,fill_value=0.)
            self.fluxerr[band] = RegularGridInterpolator((phav,zv),fluxerr,method=method,bounds_error=False,fill_value=0.)


            # Flux derivatives
            self.param[band] = {}
            for par in param_Fisher:
                valpar = np.reshape(lc_sel[index]['d{}'.format(par)],(npha,nz))
                self.param[band][par] = RegularGridInterpolator((phav,zv),valpar,method=method,bounds_error=False,fill_value=0.)

            # gamma estimator
           
            rec = Table.read(gammaName,path='gamma_{}'.format(band))

            rec['mag'] = rec['mag'].data.round(decimals=4)
            rec['exptime'] = rec['exptime'].data.round(decimals=4)


            magmin, magmax, magstep, nmag = self.limVals(rec,'mag')
            expmin, expmax, expstep, nexp = self.limVals(rec,'exptime')
            mag = np.linspace(magmin,magmax,nmag)
            exp = np.linspace(expmin,expmax,nexp)

            

            index = np.lexsort((np.round(rec['exptime'],4),rec['mag']))
            gammab = np.reshape(rec[index]['gamma'],(nmag,nexp))
            self.gamma[band] = RegularGridInterpolator((mag,exp),gammab,method=method,bounds_error=False,fill_value=0.)


    def limVals(self,lc, field):
        """ Get unique values of a field in  a table

        Parameters
        ----------
        lc: Table
         astropy Table (here probably a LC)
        field: str
         name of the field of interest

        Returns
        -------
        vmin: float
         min value of the field
        vmax: float
         max value of the field
        vstep: float
         step value for this field (median)
        nvals: int
         number of unique values
         



        """
        

        lc.sort(field)
        vals = np.unique(lc[field].data.round(decimals=4))
        #print(vals)
        vmin = np.min(vals)
        vmax = np.max(vals)
        vstep = np.median(vals[1:]-vals[:-1]) 
    
        return vmin,vmax,vstep,len(vals)



    def Read_Ref(self, fi, j=-1, output_q=None):
        """" Load the reference file and
        make a single astopy Table from a set of.

        Parameters
        ----------
        fi: str,
         name of the file to be loaded

        Returns
        -------
        tab_tot: astropy table
         single table = vstack of all the tables in fi.

        """
        


        tab_tot = Table()
        """
        keys=np.unique([int(z*100) for z in zvals])
        print(keys)
        """
        f = h5py.File(fi, 'r')
        keys = f.keys()
        zvals = np.arange(0.01, 0.9, 0.01)
        zvals_arr = np.array(zvals)

        for kk in keys:

            tab_b = Table.read(fi, path=kk)
            
            if tab_b is not None:
                tab_tot = vstack([tab_tot, tab_b],metadata_conflicts='silent')
                """
                diff = tab_b['z']-zvals_arr[:, np.newaxis]
                #flag = np.abs(diff)<1.e-3
                flag_idx = np.where(np.abs(diff) < 1.e-3)
                if len(flag_idx[1]) > 0:
                    tab_tot = vstack([tab_tot, tab_b[flag_idx[1]]])
                """

            """
            print(flag,flag_idx[1])
            print('there man',tab_b[flag_idx[1]])
            mtile = np.tile(tab_b['z'],(len(zvals),1))
            #print('mtile',mtile*flag)
                
            masked_array = np.ma.array(mtile,mask=~flag)
            
            print('resu masked',masked_array,masked_array.shape)
            print('hhh',masked_array[~masked_array.mask])
            
            
        for val in zvals:
            print('hello',tab_b[['band','z','time']],'and',val)
            if np.abs(np.unique(tab_b['z'])-val)<0.01:
            #print('loading ref',np.unique(tab_b['z']))
            tab_tot=vstack([tab_tot,tab_b])
            break
            """
        if output_q is not None:
            output_q.put({j: tab_tot})
        else:
            return tab_tot

    def Read_Multiproc(self, tab):

        # distrib=np.unique(tab['z'])
        nlc = len(tab)
        print('ici pal', nlc)
        # n_multi=8
        if nlc >= 8:
            n_multi = min(nlc, 8)
            nvals = nlc/n_multi
            batch = range(0, nlc, nvals)
            batch = np.append(batch, nlc)
        else:
            batch = range(0, nlc)

        # lc_ref_tot={}
        #print('there pal',batch)
        result_queue = multiprocessing.Queue()
        for i in range(len(batch)-1):

            ida = int(batch[i])
            idb = int(batch[i+1])

            p = multiprocessing.Process(
                name='Subprocess_main-'+str(i), target=self.Read_Ref, args=(tab[ida:idb], i, result_queue))
            p.start()

        resultdict = {}
        for j in range(len(batch)-1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        tab_res = Table()
        for j in range(len(batch)-1):
            if resultdict[j] is not None:
                tab_res = vstack([tab_res, resultdict[j]])

        return tab_res

class Gamma:
    def __init__(self,bands,telescope, fileout):
        """ class to estimate gamma parameters
            depending on mag and exposure time

        Parameters
        -----------
        bands: str
         bands to process
        telescope: Telescope
         telescope used to estimate gamma
        fileout: str
         output file name (extension: hdf5)

        Returns
        --------
        None

        Output
        ------
        output fileout is generated.
        hdf5 format with the keys:
        gamma_band1
        gamma_band2
        ........
        for each key: astropy table for band b
        with the following fields:
        band (str): band name
        mag (float): mag
        exptime (float): exposure time
        gamma (float): gamma estimation
        

        """
        # mag range to consider
        mag_range = np.arange(10., 35., 0.05)

        # exptimes to consider
        exptimes = np.arange(1.,900.,1.)

        # gamma estimation
        tab = self.loopGamma(bands,mag_range,exptimes, telescope)

        #dump the result in a hdf5 file

        for band in bands:
            idx = tab['band']==band
            sel = Table(tab[idx])
            sel.write(fileout,path='gamma_{}'.format(band),
                      append=True,compression=True)



    def loopGamma(self,bands,mag_range,exptimes, telescope):
        """ Gamma parameter estimation - loop on bands

        Parameters
        -----------
        bands: str
         bands to process
        mag_range: list(float)
         mag values to estimate gamma
        exptimes: list(float)
         exposure times to estimate gamma
        telescope: Telescope
         telescope used to estimate gamma
        j: int
         tagger for multiprocessing (default: -1)
        output_q: multiprocessing.Queue()
         queue for multiprocessing (default: None)

        Returns
        --------
        astropy table with the following fields:
        band (str): band name
        mag (float): mag
        exptime (float): exposure time
        gamma (float): gamma estimation

        """
        result_queue = multiprocessing.Queue()
        for i,band in enumerate(bands):
            p = multiprocessing.Process(
                name='Subprocess-'+str(i), target=self.calcGamma, args=(band, mag_range,exptimes,telescope,i,result_queue))
            p.start()
            

        resultdict = {}
        for j in range(len(bands)):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        restot = Table()
        for j in range(len(bands)):
            restot = vstack([restot,resultdict[j]])
                
        return restot


    def calcGamma(self,band, mag_range,exptimes,telescope, j=-1, output_q=None):
        """ Gamma parameter estimation

        Parameters
        -----------
        band: str
         band to process
        mag_range: list(float)
         mag values to estimate gamma
        exptimes: list(float)
         exposure times to estimate gamma
        telescope: Telescope
         telescope used to estimate gamma
        j: int
         tagger for multiprocessing (default: -1)
        output_q: multiprocessing.Queue()
         queue for multiprocessing (default: None)

        Returns
        --------
        astropy table with the following fields:
        band (str): band name
        mag (float): mag
        exptime (float): exposure time
        gamma (float): gamma estimation
        

        """

        gamm = []
        for mag in mag_range:
            for exptime in exptimes:
                gamm.append((band,mag,exptime,telescope.gamma(mag,band,exptime)))

        rec = Table(rows=gamm, names=['band','mag','exptime','gamma'])
    
        if output_q is not None:
            output_q.put({j: rec})
        else:
            return rec

