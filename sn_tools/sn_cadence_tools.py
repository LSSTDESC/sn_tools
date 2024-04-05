import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy.lib.recfunctions as rf
import h5py
from astropy.table import Table, Column, vstack
from scipy.interpolate import CloughTocher2DInterpolator
from sn_tools.sn_obs import renameFields, getFields
from sn_tools.sn_obs import getObservations
import pandas as pd
from sn_tools.sn_obs import DataInside, season
from sn_tools.sn_clusters import ClusterObs
from sn_tools.sn_utils import multiproc
from sn_tools.sn_io import get_beg_table, get_end_table, dumpIt
import os
import operator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class ReferenceData:
    """
    class to handle light curve of SN

    Parameters
    --------------

    Li_files : list[str]
      names light curve reference file
    mag_to_flux_files : list[str]
      names of files of magnitude to flux
    band : str
      band considered
    z : float
      redshift considered

    """

    def __init__(self, Li_files, mag_to_flux_files, band, z):
        self.band = band
        self.z = z
        self.fluxes = []
        self.mag_to_flux = []

        # load the files
        for val in Li_files:
            self.fluxes.append(self.GetInterpFluxes(
                self.band, np.load(val), self.z))
        for val in mag_to_flux_files:
            self.mag_to_flux.append(
                self.GetInterpmag(self.band, np.load(val)))

    def GetInterpFluxes(self, band, tab, z):
        """
        Flux interpolator

        Parameters
        ---------------
        band : str
           band considered
        tab : array
           reference data with (at least) fields z,band,time,DayMax
        z : float
         redshift considered

        Returns
        ---------
        list (float) of interpolated fluxes (in e/sec)
        """
        lims = {}
        idx = (np.abs(tab['z'] - z) < 1.e-5) & (tab['band'] == 'LSST::'+band)
        sel = tab[idx]
        selc = np.copy(sel)
        difftime = (sel['time']-sel['DayMax'])
        selc = rf.append_fields(selc, 'deltaT', difftime)

        return interpolate.interp1d(selc['deltaT'], selc['flux_e'], bounds_error=False, fill_value=0.)

    def GetInterpmag(self, band, tab):
        """
        Magnitude (m5) to flux (e/sec) interpolator

        Parameters
        ---------------
        band : str
           band considered
        tab : array
           reference data with (at least) fields band,m5,flux_e

        Returns
        ----------
        list (float) of interpolated fluxes (in e/sec)

        """

        idx = tab['band'] == band
        sel = tab[idx]

        return interpolate.interp1d(sel['m5'], sel['flux_e'], bounds_error=False, fill_value=0.)


class TemplateData(object):
    """
    class to load template LC

    Parameters
    --------------
    filename, str
       name of the file to load
       format : hdf5
    band, str
       band to consider

    """

    def __init__(self, filename, band):
        self.fi = filename
        self.refdata = self.Stack()
        from sn_telmodel.sn_telescope import Telescope
        self.telescope = Telescope(airmass=1.1)
        self.blue_cutoff = 300.
        self.red_cutoff = 800.
        self.param_Fisher = ['X0', 'X1', 'Color']
        self.method = 'cubic'

        self.band = band
        idx = self.refdata['band'] == band
        lc_ref = self.refdata[idx]

        # load reference values (only once)
        phase_ref = lc_ref['phase']
        z_ref = lc_ref['z']
        flux_ref = lc_ref['flux']
        fluxerr_ref = lc_ref['fluxerr']
        self.gamma_ref = lc_ref['gamma'][0]
        self.m5_ref = np.unique(lc_ref['m5'])[0]
        self.dflux_interp = {}
        self.flux_interp = CloughTocher2DInterpolator(
            (phase_ref, z_ref), flux_ref, fill_value=1.e-5)
        self.fluxerr_interp = CloughTocher2DInterpolator(
            (phase_ref, z_ref), fluxerr_ref, fill_value=1.e-8)
        for val in self.param_Fisher:
            dflux = lc_ref['d'+val]
            self.dflux_interp[val] = CloughTocher2DInterpolator(
                (phase_ref, z_ref), dflux, fill_value=1.e-8)
            """
            dFlux[val] = griddata((phase, z), dflux, (phase_obs, yi_arr),
                                  method=self.method, fill_value=0.)
            """
        # this is to convert mag to flux in e per sec
        self.mag_to_flux_e_sec = {}
        mag_range = np.arange(14., 32., 0.1)
        for band in 'grizy':
            fluxes_e_sec = self.telescope.mag_to_flux_e_sec(
                mag_range, [band]*len(mag_range), [30]*len(mag_range))
            self.mag_to_flux_e_sec[band] = interpolate.interp1d(
                mag_range, fluxes_e_sec[:, 1], fill_value=0., bounds_error=False)

    def Stack(self):
        """
        Stack of all the files in the original hdf5 file.

        """
        tab_tot = None
        f = h5py.File(self.fi, 'r')
        keys = f.keys()

        for kk in keys:

            tab_b = Table.read(self.fi, path=kk)
            if tab_tot is None:
                tab_tot = tab_b
            else:
                tab_tot = vstack([tab_tot, tab_b], metadata_conflicts='silent')

        return tab_tot

    def Fluxes(self, mjd_obs, param):
        """
        Flux interpolator

        Parameters
        --------------
        mjd_obs : list(float)
           list of MJDs
        param : array
           array of parameters:
           z (float) : redshift
           daymax (float) : T0

        Returns
        ----------
        flux : list(float)
          interpolated fluxes

        """

        z = param['z']
        daymax = param['daymax']

        # observations (mjd, daymax, z) where to get the fluxes
        phase_obs = mjd_obs-daymax[:, np.newaxis]
        phase_obs = phase_obs/(1.+z[:, np.newaxis])  # phases of LC points
        z_arr = np.ones_like(phase_obs)*z[:, np.newaxis]

        flux = self.flux_interp((phase_obs, z_arr))

        return flux

    def Simulation(self, mjd_obs, m5_obs, exptime_obs, param):
        """
        LC simulation

        Parameters
        --------------
        mjd_obs : list(float)
           list of MJDs
        m5_obs : list(float)
          list of corresponding five-sigma depth values
        exptime_obs : list(float)
          list of corresponding exposure times
        param : array
          array of parameters:
          z (float) : redshift
           daymax (float) : T0

        Returns
        ----------
        astropy table with the following infos:
        flux (float) : flux
        fluxerr (float) : flux error
        phase (float) : phase
        snr_m5 (float) : Signal-to-Noise Ratio
        mag (float) : magnitude
        magerr (float) : magnitude error
        time (float) : time (MJD) (in days)
        band (str) : band
        zp (float) : zeropoint
        zpsys (str) : zeropint system
        z (float) : redshift
        daymax (float) : T0

        """

        z = param['z']
        daymax = param['daymax']

        # observations (mjd, daymax, z) where to get the fluxes
        phase_obs = mjd_obs-daymax[:, np.newaxis]
        phase_obs = phase_obs/(1.+z[:, np.newaxis])  # phases of LC points
        z_arr = np.ones_like(phase_obs)*z[:, np.newaxis]

        flux = self.flux_interp((phase_obs, z_arr))
        fluxerr = self.fluxerr_interp((phase_obs, z_arr))

        flux[flux <= 0] = 1.e-5
        """
        fluxerr_corr = self.FluxErrCorr(
            flux, m5_obs, exptime_obs, self.gamma_ref, self.m5_ref)

        fluxerr /= fluxerr_corr
        """
        tab = self.SelectSave(param, flux, fluxerr,
                              phase_obs, mjd_obs)
        return tab

    def FluxErrCorr(self, fluxes_obs, m5_obs, exptime_obs, gamma_ref, m5_ref):
        """
        Flux error correction
        (because m5 values different between template file and observations)

        Parameters
        --------------
        fluxes_obs : float
          observed fluxes
        m5_obs : float
          five-sigma depths obs values
        exptime_obs : float
          exposure times of observations
        gamma_ref : float
          gamma reference values
        m5_ref : float
          five-sigma depth reference values

        Returns
        ----------
        correct_m5 : float
         correction factor for error flux

        """

        gamma_obs = self.telescope.gamma(
            m5_obs, [self.band]*len(m5_obs), exptime_obs)
        mag_obs = -2.5*np.log10(fluxes_obs/3631.)

        gamma_tile = np.tile(gamma_obs, (len(mag_obs), 1))
        m5_tile = np.tile(m5_obs, (len(mag_obs), 1))
        srand_obs = self.Srand(gamma_tile, mag_obs, m5_tile)

        # srand_obs = self.srand(gamma_obs, mag_obs, m5_obs)
        # print('yes', band, m5_ref, gamma_ref, gamma_obs, mag_obs, srand_obs)

        m5 = np.asarray([m5_ref]*len(m5_obs))
        gamma = np.asarray([gamma_ref]*len(m5_obs))
        srand_ref = self.Srand(
            np.tile(gamma, (len(mag_obs), 1)), mag_obs, np.tile(m5, (len(mag_obs), 1)))

        correct_m5 = srand_ref/srand_obs

        return correct_m5

    def Srand(self, gamma, mag, m5):
        """
        \sigma_{rand} estimation (eq. 5 in
        LSST: from science drivers to reference design and anticipated data products)

        .. math::

        \sigma_{rand} = (0.04-\gamma)x+\gamma x^{2} (mag\^2)


        Parameters
        --------------
        gamma : float
          gamma values
        mag : float
          magnitude values
        m5: float
           five-sigma depth values

        Returns
        ----------
        sigma_rand : float
           equal to \sqrt((0.04-gamma)*x+gamma*x**2)
        """

        x = 10**(0.4*(mag-m5))
        return np.sqrt((0.04-gamma)*x+gamma*x**2)

    def FisherValues(self, param, phase, fluxerr):
        """
        Estimate Fisher matrix elements

        Parameters
        --------------
        param : array
          array of parameters:
            z : float
             redshift
        phase : float
           phases of observations
        fluxerr_obs : float
           error flux errors

        Returns
        ----------
        FisherEl : array
          array with Fisher matrix elements


        """

        """
        idx = self.refdata['band'] == band
        lc_ref = self.refdata[idx]
        phase = lc_ref['phase']
        z = lc_ref['z']
        """
        yi_arr = np.ones_like(phase)*param['z'][:, np.newaxis]
        dFlux = {}
        for val in self.param_Fisher:
            """
            dflux = lc_ref['d'+val]
            dFlux[val] = griddata((phase, z), dflux, (phase_obs, yi_arr),
                                  method=self.method, fill_value=0.)
            """
            dFlux[val] = self.dflux_interp[val]((phase_obs, yi_arr))

        FisherEl = {}
        for ia, vala in enumerate(self.param_Fisher):
            for jb, valb in enumerate(self.param_Fisher):
                if jb >= ia:
                    FisherEl[vala+valb] = dFlux[vala] * \
                        dFlux[valb]/fluxerr**2
        return FisherEl

    def SelectSave(self, param, flux, fluxerr, phase, mjd):
        """"
        Generate astropy table of light curve points

        Parameters
        --------------
        param : array
          array of parameters :
            min_rf_phase (float) : min rest-frame phase
            max_rf_phase (float) : max rest-frame phase
            daymax (float) : T0
        flux : float
          fluxes
        fluxerr : float
          flux errors
        phase : float
           phase of observations
        mjd : float
           MJD of observations


        Returns
        ----------
        astropy table with the following infos:
        flux (float) : flux
        fluxerr (float) : flux error
        phase (float) : phase
        snr_m5 (float) : Signal-to-Noise Ratio
        mag (float) : magnitude
        magerr (float) : magnitude error
        time (float) : time (MJD) (in days)
        band (str) : band
        zp (float) : zeropoint
        zpsys (str) : zeropint system
        z (float) : redshift
        daymax (float) : T0

        """

        min_rf_phase = param['min_rf_phase']
        max_rf_phase = param['max_rf_phase']
        z = param['z']
        daymax = param['daymax']

        # estimate element for Fisher matrix
        # FisherEl = self.FisherValues(param, phase, fluxerr)

        # flag for LC points outside the restframe phase range
        min_rf_phase = min_rf_phase[:, np.newaxis]
        max_rf_phase = max_rf_phase[:, np.newaxis]
        flag = (phase >= min_rf_phase) & (phase <= max_rf_phase)

        # flag for LC points outside the (blue-red) range
        mean_restframe_wavelength = np.array(
            [self.telescope.mean_wavelength[self.band]]*len(mjd))
        mean_restframe_wavelength = np.tile(
            mean_restframe_wavelength, (len(z), 1))/(1.+z[:, np.newaxis])
        flag &= (mean_restframe_wavelength > self.blue_cutoff) & (
            mean_restframe_wavelength < self.red_cutoff)
        flag_idx = np.argwhere(flag)

        # Now apply the flags to grab only interested values
        fluxes = np.ma.array(flux, mask=~flag)
        fluxes_err = np.ma.array(fluxerr, mask=~flag)
        mag = -2.5*np.log10(fluxes/3631.)
        phases = np.ma.array(phase, mask=~flag)
        snr_m5 = np.ma.array(flux/fluxerr, mask=~flag)
        obs_time = np.ma.array(
            np.tile(mjd, (len(mag), 1)), mask=~flag)
        """
        seasons = np.ma.array(
            np.tile(season, (len(mag_obs), 1)), mask=~flag)
        """
        z_vals = z[flag_idx[:, 0]]
        DayMax_vals = daymax[flag_idx[:, 0]]

        # Results are stored in an astropy Table
        tab = Table()
        tab.add_column(Column(fluxes[~fluxes.mask], name='flux'))
        tab.add_column(Column(fluxes_err[~fluxes_err.mask], name='fluxerr'))
        tab.add_column(Column(phases[~phases.mask], name='phase'))
        tab.add_column(Column(snr_m5[~snr_m5.mask], name='snr_m5'))
        tab.add_column(Column(mag[~mag.mask], name='mag'))
        tab.add_column(
            Column((2.5/np.log(10.))/snr_m5[~snr_m5.mask], name='magerr'))
        tab.add_column(Column(obs_time[~obs_time.mask], name='time'))

        tab.add_column(
            Column(['LSST::'+self.band]*len(tab), name='band',
                   dtype=h5py.special_dtype(vlen=str)))

        tab.add_column(Column([2.5*np.log10(3631)]*len(tab),
                              name='zp'))

        tab.add_column(
            Column(['ab']*len(tab), name='zpsys',
                   dtype=h5py.special_dtype(vlen=str)))

        # tab.add_column(Column(seasons[~seasons.mask], name='season'))
        tab.add_column(Column(z_vals, name='z'))
        tab.add_column(Column(DayMax_vals, name='daymax'))
        """
        for key, vals in FisherEl.items():
            matel = np.ma.array(vals, mask=~flag)
            tab.add_column(Column(matel[~matel.mask], name='F_'+key))
        """
        return tab


"""
class TemplateData_x1color(object):

    # class to load template LC


    def __init__(self, filenames, band):

        self.fi = filenames
        self.refdata = self.Stack()
        self.telescope = Telescope(airmass=1.1)
        self.blue_cutoff = 300.
        self.red_cutoff = 800.
        self.param_Fisher = ['X0', 'X1', 'Color']
        self.method = 'cubic'

        self.band = band
        idx = self.refdata['band'] == band
        lc_ref = self.refdata[idx]
        # load reference values (only once)
        phase_ref = lc_ref['phase']
        z_ref = lc_ref['z']
        flux_ref = lc_ref['flux']
        fluxerr_ref = lc_ref['fluxerr']
        x1_ref = lc_ref['x1']
        color_ref = lc_ref['color']
        self.gamma_ref = lc_ref['gamma'][0]
        self.m5_ref = np.unique(lc_ref['m5'])[0]
        print(x1_ref, color_ref)
        self.dflux_interp = {}
        self.flux_interp = LinearNDInterpolator(
            (phase_ref, z_ref, x1_ref, color_ref), flux_ref, fill_value=1.e-5)
        print('interpolation ready')
"""
"""
    self.fluxerr_interp = CloughTocher2DInterpolator(
        (phase_ref, z_ref, x1_ref, color_ref), fluxerr_ref, fill_value=1.e-8)
    for val in self.param_Fisher:
        dflux = lc_ref['d'+val]
        self.dflux_interp[val] = CloughTocher2DInterpolator(
            (phase_ref, z_ref, x1_ref, color_ref), dflux, fill_value=1.e-8)



    # this is to convert mag to flux in e per sec
    self.mag_to_flux_e_sec = {}
    mag_range = np.arange(14., 32., 0.1)
    for band in 'grizy':
        fluxes_e_sec = self.telescope.mag_to_flux_e_sec(
            mag_range, [band]*len(mag_range), [30]*len(mag_range))
        self.mag_to_flux_e_sec[band] = interpolate.interp1d(
            mag_range, fluxes_e_sec[:, 1], fill_value=0., bounds_error=False)
    """
"""
    def Stack(self):

        tab_tot = None

        for fname in self.fi:

            print('loading', fname)
            f = h5py.File(fname, 'r')
            keys = f.keys()

            for kk in keys:
                tab_b = Table.read(fname, path=kk)
                if tab_tot is None:
                    tab_tot = tab_b
                else:
                    tab_tot = vstack([tab_tot, tab_b],
                                     metadata_conflicts='silent')

        return tab_tot

    def Fluxes(self, mjd_obs, param):

        z = param['z']
        daymax = param['DayMax']
        x1 = param['x1']
        color = param['color']

        # observations (mjd, daymax, z) where to get the fluxes
        phase_obs = mjd_obs-daymax[:, np.newaxis]
        phase_obs = phase_obs/(1.+z[:, np.newaxis])  # phases of LC points
        z_arr = np.ones_like(phase_obs)*z[:, np.newaxis]
        x1_arr = np.ones_like(phase_obs)*x1[:, np.newaxis]
        color_arr = np.ones_like(phase_obs)*color[:, np.newaxis]
        flux = self.flux_interp((phase_obs, z_arr, x1_arr, color_arr))

        return flux
"""


class AnaOS:
    """
    class to analyze an observing strategy
    The idea here is to disentangle WFD and DD obs
    so as to estimate statistics such as the total
    number of visits or the DDF fraction.

    Parameters
    ---------------
    dbDir: str
     path to the location dir of the database
    dbName: str
     name of the dbfile to load
    dbExtens: str
     extension of the dbfile
          two possibilities:
          - dbExtens = db for scheduler files
          - dbExtens = npy for npy files (generated from scheduler files)
     nclusters: int
      number of clusters to search in DD data
    fields: pandas df
      fields to consider for matching
    RACol: str,opt
      RA colname
    DecCol: str,opt
      Dec colname

    """

    def __init__(self, dbDir, dbName, dbExtens, nclusters, fields,
                 RACol='fieldRA', DecCol='fieldDec'):

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        self.nclusters = nclusters
        self.fields = fields
        self.RACol = RACol
        self.DecCol = DecCol

        self.stat = self.process()

    def process(self):
        """
        Method for processing

        """
        df = pd.DataFrame(columns=['cadence'])
        df.loc[0] = self.dbName

        # load observations
        observations = self.load_obs()

        print('total number of obs', len(observations))
        # WDF obs
        obs_WFD = np.copy(getFields(observations, 'WFD',
                                    RACol=self.RACol, DecCol=self.DecCol))
        df['WFD'] = len(obs_WFD)
        df_bands = pd.DataFrame(obs_WFD).groupby(
            ['filter']).size().to_frame('count').reset_index()
        for index, row in df_bands.iterrows():
            df['WFD_{}'.format(row['filter'])] = row['count']
        df['WFD_all'] = df_bands['count'].sum()

        # DDF obs
        nside = 128
        fieldIds = [290, 744, 1427, 2412, 2786]
        obs_DD = getFields(observations, 'DD', fieldIds, nside,
                           RACol=self.RACol, DecCol=self.DecCol)
        df['DD'] = len(obs_DD)

        df['frac_DD'] = df['DD']/(df['DD']+df['WFD'])

        if len(obs_DD) == 0:
            return None

        # get the number of visits per band
        df_bands = pd.DataFrame(np.copy(obs_DD)).groupby(
            ['filter']).size().to_frame('count').reset_index()
        for index, row in df_bands.iterrows():
            df['DD_{}'.format(row['filter'])] = row['count']
        df['DD_all'] = df_bands['count'].sum()

        # make clusters
        self.clus = ClusterObs(obs_DD, self.nclusters,
                               self.dbName, self.fields)
        clusters = self.clus.clusters

        for index, row in clusters.iterrows():
            self._fill_field(df, row['fieldName'], row)

        # check whether all fields are there if not set 0 to missing fields

        for index, row in self.fields.iterrows():
            if row['name'] not in df.columns:
                self._fill_field(df, row['name'])

        return df

    def _fill_field(self, df, fieldName, ddc=None):
        """
        Method to fill infos from clusters

        Parameters
        ----------------
        df: pandas df
         dataframe of data (to be appended)
        ddc: numpy array, opt
          cluster array (default: None)


        Returns
        -----------
        pandas df with appended infos


        """
        df.loc[:, fieldName] = ddc['Nvisits'] if ddc is not None else 0
        for band in 'ugrizy':
            df.loc[:, '{}_{}'.format(fieldName, band)] = ddc['Nvisits_{}'.format(
                band)] if ddc is not None else 0
        for val in ['area', 'width_RA', 'width_Dec']:
            df.loc[:, '{}_{}'.format(fieldName, val)
                   ] = ddc[val] if ddc is not None else 0

    def load_obs(self):
        """
        Method to load observations (from simulation)

        Parameters
        ----------

        Returns
        -------

        numpy record array with scheduler information like
         observationId, fieldRA, fieldDec, observationStartMJD,
         flush_by_mjd, visitExposureTime, filter, rotSkyPos,
         numExposures, airmass, seeingFwhm500, seeingFwhmEff,
         seeingFwhmGeom, sky, night, slewTime, visitTime,
         slewDistance, fiveSigmaDepth, altitude, azimuth,
         paraAngle, cloud, moonAlt, sunAlt, note, fieldId,
         proposalId, block_id, observationStartLST, rotTelPos,
         moonAz, sunAz, sunRA, sunDec, moonRA, moonDec,
         moonDistance, solarElong, moonPhase
         This list may change from file to file.

        """

        observations = getObservations(self.dbDir, self.dbName, self.dbExtens)
        observations = renameFields(observations)

        return observations

    def fillVisits(self, thedict, name):
        """
        Method to fill the number of visits

        Parameters
        ----------
        thedict: dict
         dict of visits; keys: bands; values: number of visits
        name: str
         name of the field (DD or WFD)
        """

        r = []
        names = []

        for key, vals in thedict.items():
            r.append(vals)
            names.append('{}_{}'.format(name, key))

        return r, names

    def plot_dithering(self):
        """
        Method to plot clusters of data

        """

        fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(10, 9))
        fig.suptitle('{}'.format(self.dbName))
        fig.subplots_adjust(right=0.75)

        color = ['red', 'black', 'blue', 'cyan', 'green', 'purple']
        pos = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

        # fields = DDFields()

        lista = []
        listb = []
        for io in range(self.nclusters):
            RA = self.clus.points[self.clus.clus == io, 0]
            Dec = self.clus.points[self.clus.clus == io, 1]
            xp = pos[io][0]
            yp = pos[io][1]
            idx = self.clus.clusters['clusid'] == io
            val = self.clus.clusters[idx]

            # label = '{} - {} deg2'.format(val['fieldName'][0],np.round(val['area'][0],2))
            # print('hello',xp,yp,label,io)
            # ab  = ax[xp][yp].plot(RA,Dec,marker='o',color=color[io],label=label)
            print(RA, Dec)
            ax[xp][yp].plot(RA, Dec, marker='.',
                            color=color[io], lineStyle='None')

            # lista.append(ab)
            # listb.append(label)

            ell = Ellipse((val['RA'], val['Dec']), val['width_RA'],
                          val['width_Dec'], facecolor='none', edgecolor='black')
            print(val['RA'], val['Dec'], val['width_RA'], val['width_Dec'])
            ax[xp][yp].add_patch(ell)

            # ell = Ellipse((0.,0.),val['width_RA'],val['width_Dec'],facecolor='none',edgecolor=color[io])
            # ax[2][1].add_patch(ell)
            if yp == 0:
                ax[xp][yp].set_ylabel('Dec [rad]')
            if xp == 2:
                ax[xp][yp].set_xlabel('RA [rad]')

        """
        RAmax = np.max(self.clus.clusters['width_RA'])+0.3
        Decmax = np.max(self.clus.clusters['width_Dec'])+0.3

        ax[2][1].set_xlim(-RAmax/2.,RAmax/2.)
        ax[2][1].set_ylim(-Decmax/2.,Decmax/2.)
        ax[2][1].set_xlabel('RA [rad]')
        """

        # plt.legend(lista,listb,loc='center left', bbox_to_anchor=(1, 2.0),ncol=1,fontsize=12)


def Match_DD(fields_DD, df, radius=5):
    """
    Method to match df data to DD fields

    Parameters
    ---------------
    fields_DD: pandas df
      DD fields to match to data
    df: pandas df
     data (results from a metric) to match to DD fields

    Returns
    ----------
    pandas df with matched DD information added.

    """""

    dfb = pd.DataFrame()
    # for field in fields_DD:
    for index, field in fields_DD.iterrows():
        dataSel = DataInside(
            df.to_records(index=False), field['RA'], field['Dec'], radius, radius, 'pixRA', 'pixDec')
        if dataSel is not None:
            dfSel = pd.DataFrame(np.copy(dataSel.data))
            dfSel['fieldname'] = field['name']
            dfSel['fieldId'] = field['fieldId']
            dfSel['RA'] = field['RA']
            dfSel['Dec'] = field['Dec']
            dfSel['fieldnum'] = field['fieldnum']
            dfb = pd.concat([dfb, dfSel], sort=False)

    return dfb


class Stat_DD_night:
    """
    class to estimate statistical estimator related to DD fields per night

    Parameters
    --------------
    dbDir: str
      db directory
    dbName: str
      db name
    dbExtens: str
      db extension
    prefix: str, opt
      prefix to tag DD fields in data (default: DD)
    """

    def __init__(self, dbDir, dbName, dbExtens, prefix='DD'):

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        self.prefix = prefix

        self.obs = self.load()
        self.obs_DD = self.get_DD()
        budget = time_budget(self.obs, self.obs_DD)
        nDD_night = nvisits_DD_night(self.obs_DD)

        print('DD budget', budget, len(self.obs), nDD_night)

        # print(test)
        params = {}
        params['obs_DD'] = self.obs_DD
        params['Nvisits'] = len(self.obs)
        params['dbName'] = self.dbName
        params['mjdCol'] = 'mjd'
        params['nightCol'] = 'night'
        params['fieldCol'] = 'field'
        params['fieldColdb'] = 'note'
        params['filterCol'] = 'band'
        params['list_moon'] = ['moonAz', 'moonRA',
                               'moonDec', 'moonDistance', 'season', 'moonPhase']

        res = multiproc(
            np.unique(self.obs_DD['note']), params, ana_DDF, 6)

        tab = Table.from_pandas(res)
        tab.meta = dict(zip(['dbName'], [dbName]))
        tab.meta = {**tab.meta, **budget}
        tab.meta['nDD_night'] = nDD_night
        self.summary = tab

    def load(self):
        """
        Method to load data

        Returns
        ----------
        numpy array of data
        """

        from sn_tools.sn_obs import getObservations
        """
        fName = '{}/{}.{}'.format(self.dbDir, self.dbName, self.dbExtens)
        data = np.load(fName, allow_pickle=True)
        """
        data = getObservations(self.dbDir, self.dbName, self.dbExtens)
        return data

    def get_DD(self):
        """
        Method to select observations corresponding to DD fields

        Returns
        ----------
        array of obs corresponding to DD fields

        """
        field_list = np.unique(self.obs['note'])
        self.field_DD = list(
            filter(lambda x: x.startswith(self.prefix), field_list))

        # select DD only
        id_ddf = np.in1d(self.obs['note'], self.field_DD)

        return np.copy(self.obs[id_ddf])


def stat_DD_night_pixel_deprecated(obsPixelDir, dbName, nproc=8):
    """
    Function to perform cadence stat per pixel

    Parameters
    ---------------
    obsPixelDir: str
      directory where the data are to be found
    dbName: str
      OS name
    nproc: int, opt
      number of procs for multiprocessing (default: 8)

    Returns
    -----------
    pandas df with stat values per pixel/season

    """
    import glob
    obsPixelFiles = glob.glob('{}/{}/*.npy'.format(obsPixelDir, dbName))

    rtot = pd.DataFrame()
    for fi in obsPixelFiles:
        pixels = pd.DataFrame(np.load(fi, allow_pickle=True))
        print(fi, len(pixels), pixels.columns)
        if len(pixels) > 0:
            rr = process_night_pixel(pixels.to_records(), dbName, nproc=nproc)
            print(rr, rr.columns)
            rtot = pd.concat((rtot, rr))

    return rtot


def stat_DD_night_pixel(pixels, dbName, nproc=8):
    """
    Function to perform cadence stat per pixel

    Parameters
    ---------------
    obsPixelDir: str
      directory where the data are to be found
    dbName: str
      OS name
    nproc: int, opt
      number of procs for multiprocessing (default: 8)

    Returns
    -----------
    pandas df with stat values per pixel/season

    """

    rtot = pd.DataFrame()

    if len(pixels) > 0:
        rr = process_night_pixel(pixels.to_records(), dbName, nproc=nproc)
        print(rr, rr.columns)
        rtot = pd.concat((rtot, rr))

    return rtot


def process_night_pixel(pixels, dbName, nproc=8):

    params = {}
    params['obs_DD'] = pixels
    params['Nvisits'] = len(pixels)
    params['dbName'] = dbName
    params['mjdCol'] = 'observationStartMJD'
    params['nightCol'] = 'night'
    params['fieldCol'] = 'healpixID'
    params['fieldColdb'] = 'healpixID'
    params['filterCol'] = 'filter'
    params['list_moon'] = ['pixRA', 'pixDec', 'season']

    res = multiproc(
        np.unique(pixels['healpixID']), params, ana_DDF, nproc)

    # print('allo', np.unique(pixels['fieldName']))
    res['field'] = np.unique(pixels['fieldName'])[0]

    return res


def time_budget(obs, obs_DD):
    """"
    Method to estimate the time budget from DD

    Returns
    -----------
    time budget (float)
    """

    fields = np.unique(obs_DD['note'])

    dictout = {}

    DD_time = np.sum(obs_DD['numExposures']*obs_DD['exptime'])
    obs_time = np.sum(obs['numExposures']*obs['exptime'])
    dictout['time_budget'] = DD_time/obs_time

    for fi in fields:
        idx = obs_DD['note'] == fi
        sel = obs_DD[idx]
        fi_time = np.sum(sel['numExposures']*sel['exptime'])
        dictout['time_budget_{}'.format(fi)] = fi_time/obs_time

    return dictout


def nvisits_DD_night(obs):
    """
    Method to estimate the number of DD visits/obs. night

    Parameters
    ---------------
    obs: numpy array
     data to process (DD obs)

    Returns
    ----------
    mean number of visits per obs. night

    """

    tt = pd.DataFrame(obs)
    rr = tt.groupby(['night']).size()

    return np.mean(rr)


def ana_DDF(list_DD, params, j, output_q):
    """
    Method to analyze DDFs

    """

    obs_DD = params['obs_DD']
    Nvisits = params['Nvisits']
    dbName = params['dbName']
    mjdCol = params['mjdCol']
    nightCol = params['nightCol']
    fieldCol = params['fieldCol']
    fieldColdb = params['fieldColdb']
    filterCol = params['filterCol']
    list_moon = params['list_moon']

    res_DD = pd.DataFrame()

    for field in list_DD:
        # print('analyzing', field)
        idx = obs_DD[fieldColdb] == field
        res = ana_field(np.copy(obs_DD[idx]), dbName, Nvisits,
                        mjdCol=mjdCol,  nightCol=nightCol,
                        fieldColdb=fieldColdb,
                        fieldCol=fieldCol,
                        filterCol=filterCol,
                        list_moon=list_moon)
        res_DD = pd.concat((res_DD, res))

    if output_q is not None:
        return output_q.put({j: res_DD})
    else:
        return res_DD


def ana_field(obs, dbName, Nvisits, mjdCol='mjd',
              nightCol='night', fieldColdb='note',
              fieldCol='field', filterCol='band',
              list_moon=['moonAz', 'moonRA', 'moonDec',
                         'moonDistance', 'season']):
    """
    Method to analyze a single DD field

    Parameters
    --------------
    obs: array
        observation corresponding to a single DD field

    """
    # estimate seasons
    obs = season(obs, mjdCol=mjdCol)
    # print('aoooo', np.unique(obs[['note', 'season']]), obs.dtype.names)
    # print(test)
    field = np.unique(np.copy(obs[fieldColdb]))[0]

    res = ana_visits(obs, field, Nvisits,
                     mjdCol=mjdCol,
                     nightCol=nightCol,
                     fieldCol=fieldCol,
                     filterCol=filterCol,
                     list_moon=list_moon)
    res['dbName'] = dbName

    return res


def ana_visits(obs, field, Nvisits,
               nightCol='night', fieldCol='field',
               filterCol='band', mjdCol='mjd',
               list_moon=['moonAz', 'moonRA', 'moonDec', 'moonDistance', 'season']):
    """
    Method to analyze the number of visits per obs night

    Parameters
    --------------
    obs: array
        array of observations
    field: str
        field name

    Returns
    ----------
    pandas df with the number of visits per night and per band

    """

    rtot = pd.DataFrame()
    for night in np.unique(obs[nightCol]):
        idx = obs[nightCol] == night
        obs_night = obs[idx]
        # estimate the number of filter changes per night
        obs_night.sort(order=mjdCol)
        diff = np.diff(obs_night[mjdCol])
        idx = diff > 0.0005
        # print('night', night, np.diff(
        # obs_night['mjd']), obs_night['band'], len(diff[idx]))

        dd = {}
        dd[fieldCol] = field
        dd[nightCol] = night
        dd['Nfc'] = len(diff[idx])
        dd['time_budget_night'] = len(obs_night)/Nvisits
        dd['nvisits_DD'] = len(obs_night)
        if list_moon:
            for ll in list_moon:
                dd[ll] = np.median(obs_night[ll])

        # for b in np.unique(obs_night[filterCol]):
        for b in 'ugrizy':
            idb = obs_night[filterCol] == b
            obs_filter = obs_night[idb]
            resa, resb = -1.0, -1.0
            if len(obs_filter) > 0:
                dd[b] = len(obs_filter)
                if 'mjd' in obs_filter.dtype.names:
                    diff = np.diff(obs_filter['mjd'])
                    resa = np.mean(diff)
                    resb = np.std(diff)

            dd['deltaT_{}_mean'.format(b)] = resa
            dd['deltaT_{}_rms'.format(b)] = resb

        str_combi = ''
        for b in 'ugrizy':
            if not b in dd.keys():
                dd[b] = 0
            str_combi += '{}{}'.format(dd[b], b)
            if b != 'y':
                str_combi += '-'
        dd['config'] = str_combi
        # print(dd)
        ddmod = {}
        for key, val in dd.items():
            ddmod[key] = [val]
        rtt = pd.DataFrame.from_dict(ddmod)
        rtot = pd.concat((rtot, rtt))

    return rtot


def summary(res):

    for config in res['config'].unique():
        idx = res['config'] == config
        sel = res[idx]
        print(config, len(sel)/len(res))

    seas_cad(res)


def seas_cad(obs, meta={}):

    dictout = {}

    if meta:
        # print('aoooou', obs.name, meta['time_budget_{}'.format(obs.name[0])])
        dictout['time_budget_field'] = [
            meta['time_budget_{}'.format(obs.name[0])]]

    # get cadence and season length
    seas_min, seas_max = np.min(obs['night']), np.max(obs['night'])
    seas_length = seas_max-seas_min
    diff = np.diff(obs['night'])
    cad_med, cad, cad_std = np.median(diff), np.mean(diff), np.std(diff)

    dictout['cadence_median'] = [cad_med]
    dictout['cadence_mean'] = [cad]
    dictout['cadence_std'] = [cad_std]
    dictout['season_length'] = [seas_length]
    dictout['time_budget_field_season'] = [np.sum(obs['time_budget_night'])]

    # get gaps_stat
    df_diff = pd.DataFrame(diff, columns=['cad'])
    gapvals = [5, 10, 15, 20, 25, 30, 100]
    group = df_diff.groupby(pd.cut(df_diff.cad, np.array(gapvals)))

    for group_name, df_group in group:
        gmin = group_name.left
        gmax = group_name.right
        dictout['gap_{}_{}'.format(gmin, gmax)] = [len(df_group)]

    # stat on filter allocation
    combis = obs.groupby(['config'])['config'].count()/len(obs)
    dictout['filter_alloc'] = [combis.index.to_list()]
    dictout['filter_frac'] = [
        list(np.around(np.array(combis.values.tolist()), 2))]

    dictout['Nfc'] = [np.sum(obs['Nfc'])]
    for b in 'ugrizy':
        dictout[b] = [np.sum(obs[b])]

    return pd.DataFrame.from_dict(dictout)


def Stat_DD_season(data_tab, cols=['field', 'season']):
    """
    Method to analyze a set of observing data per obs night

    Parameters
    --------------
    data_tab: astopy table
      data to process

    """

    print('hhh', data_tab.meta)

    res = data_tab.to_pandas().groupby(cols).apply(
        lambda x: seas_cad(x, data_tab.meta)).reset_index()

    res['dbName'] = data_tab.meta['dbName']
    res['time_budget'] = np.round(data_tab.meta['time_budget'], 3)
    res['nDD_night'] = data_tab.meta['nDD_night']
    return res


def stat_DD_season_pixel(data_tab, cols=['healpixID', 'field',
                                         'pixRA', 'pixDec',
                                         'season', 'dbName']):
    """
    Function to analyze a set of observing data per obs night and per pixel

    Parameters
    --------------
    data_tab: astopy table
      data to process
    cols: list(str)
      list of variable to use foe groupby estimation
      default: ['healpixID', 'field', 'pixRA', 'pixDec', 'season', 'dbName']

    Returns
    ----------
    pandas df with stat cadence data

    """

    res = data_tab.groupby(cols).apply(
        lambda x: seas_cad(x)).reset_index()

    return res


class Survey_depth:
    def __init__(self, dbDir, configFile, outName='depth.tex'):
        """
        Class to print (latex table format) coadded depth and total number of visits

        Parameters
        ----------
        dbDir : str
            Location data dir.
        configFile : str
            config csv file.
        outName : str, optional
            Output name file. The default is 'depth.tex'.

        Returns
        -------
        None.

        """

        self.dbDir = dbDir
        self.configFile = configFile
        self.outName = outName

        # check if the file exist; yes? -> remove it
        if os.path.isfile(outName):
            os.system('rm {}'.format(outName))

        # process the data
        df = self.process_OS_depths()

        idx = df['note'].isin(['DD:COSMOS', 'DD:ECDFS'])
        df = df[idx]
        # dump in file
        self.print_latex_depth_one_table(df)
        # self.print_latex_depth(df)
        # self.print_latex_depth_two(df)

    def process_OS_depths(self):
        """
        Method to process the data

        Returns
        -------
        df : pandas df
            Processed data.

        """

        conf = pd.read_csv(self.configFile)

        df = pd.DataFrame()

        for i, row in conf.iterrows():
            dfa = self.process_OS_depth(row['dbName'])
            dfa['dbName'] = row['dbName']
            dfa['dbNamePlot'] = row['dbNamePlot']
            df = pd.concat((df, dfa))

        return df

    def process_OS_depth(self, dbName):
        """
        Method to process a strategy

        Parameters
        ----------
        dbName : str
            Db name (OS) to process.

        Returns
        -------
        res : pandas df
            Processed data.

        """

        full_path = '{}/{}.npy'.format(self.dbDir, dbName)

        data = np.load(full_path, allow_pickle=True)

        data = pd.DataFrame.from_records(season(data))

        idx = data['season'] == 1

        res = pd.DataFrame()

        for seas in data['season'].unique():
            idx = data['season'] == seas

            sela = data[idx]

            res_y = sela.groupby(['note', 'filter']).apply(
                lambda x: self.gime_m5_visits(x)).reset_index()
            res_y['season'] = seas

            res = pd.concat((res, res_y))

        # cumulate seasons 2-10
        idx = data['season'] >= 2
        selb = data[idx]
        res_c = selb.groupby(['note', 'filter']).apply(
            lambda x: self.gime_m5_visits(x)).reset_index()
        res_c['season'] = 11

        res = pd.concat((res, res_c))

        return res

    def gime_m5_visits(self, grp):
        """
        Method to estimate coadded m5 and total number of visits

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        res : pandas df
            output data.

        """

        m5_coadd = 1.25*np.log10(np.sum(10**(0.8*grp['fiveSigmaDepth'])))
        nvisits = len(grp)

        res = pd.DataFrame({'m5': [m5_coadd], 'nvisits': [nvisits]})

        return res

    def print_latex_depth(self, df):
        """
        Method to print results

        Parameters
        ----------
        df : pandas df
            Data to print.

        Returns
        -------
        None.

        """

        dbNames = df['dbName'].unique()

        for io, dbName in enumerate(dbNames):
            idx = df['dbName'] == dbName
            sel = df[idx]
            dbNameb = sel['dbNamePlot'].unique()[0]
            self.print_latex_depth_os(sel, dbName, io, dbNameb)

    def print_latex_depth_one_table(self, df):
        """
        Method to print results

        Parameters
        ----------
        df : pandas df
            Data to print.

        Returns
        -------
        None.

        """
        tta = ['DD:COSMOS', 'DD:ELAISS1', 'DD:XMM_LSS',
               'DD:ECDFS', 'DD:EDFS_a', 'DD:EDFS_b']
        ttb = ['\cosmos', '\elais', '\\xmm', '\cdfs', '\\adfa', '\\adfb']
        fty = ['UDF', 'DF', 'UDF', 'DF', 'DF', 'DF']
        trans_ddf = dict(zip(tta, ttb))
        trans_ddfb = dict(zip(tta, fty))
        dbNames = df['dbName'].unique()
        bands = list('ugrizy')

        caption = 'Coadded \\fivesig~depth and total number of visits $N_v$ per band.'

        caption = '{'+caption+'}'
        label = 'tab:depth'
        label = '{'+label+'}'
        r = get_beg_table(tab='{table*}', tabcols='{l|l|c|c|c}',
                          fontsize='\\tiny',
                          caption=caption, label=label, center=True)
        rr = ' & & '
        pp = 'Strategy & Field & '
        pp += 'season & $m_5$ & $N_v$'
        bb = '/'.join(bands)
        rr += ' & {} & {}'.format(bb, bb)
        rr += ' \\\\'
        pp += ' \\\\'

        r += [pp]
        r += [rr]
        r += [' & & & & \\\\']
        r += ['\hline']
        for io, dbName in enumerate(dbNames):
            idx = df['dbName'] == dbName
            sel = df[idx]
            dbNameb = sel['dbNamePlot'].unique()[0]
            dbNameb = dbNameb.replace('_', '\_')
            fields = sel['note'].unique()
            idb = 0
            for ifi, field in enumerate(fields):
                idxb = sel['note'] == field
                selb = sel[idxb]
                rb = self.print_latex_depth_field(selb)
                vva = ' & {} & {}'.format(trans_ddfb[field], rb[0])
                r += [vva]
                vvb = ' & & {}'.format(rb[10])
                if idb == 0:
                    vvb = '{} {}'.format(dbNameb, vvb)
                    idb = 1
                r += [vvb]
                if ifi == 1:
                    r += ['\\hline']
                else:
                    r += ['\\cline{2-5}']
            # r += ['\\hline']

        print(r)
        r += get_end_table(tab='{table*}', center=True)

        # dump in file

        dumpIt(self.outName, r)

    def print_latex_depth_two(self, df):
        """
        Method to print results

        Parameters
        ----------
        df : pandas df
            Data to print.

        Returns
        -------
        None.

        """

        dbNames = df['dbName'].unique()

        for io, dbName in enumerate(dbNames):
            idx = df['dbName'] == dbName
            sel = df[idx]
            dbNameb = sel['dbNamePlot'].unique()[0]
            self.print_latex_depth_os_single(sel, dbName, io, dbNameb)
            self.print_latex_depth_os_single(
                sel, dbName, io, dbNameb, 'nv',
                'total number of visits $N_v$ per band', varn='$N_v$')

    def print_latex_depth_os(self, df, dbName, io, dbNameb):
        """
        Method to print os results

        Parameters
        ----------
        df : pandas df
            Data to print.
        dbName : str
            OS to process.
        io : int
            tag for label.
        dbNameb : str
            OS name to use for printing.

        Returns
        -------
        None.

        """

        tta = ['DD:COSMOS', 'DD:ELAISS1', 'DD:XMM_LSS',
               'DD:ECDFS', 'DD:EDFS_a', 'DD:EDFS_b']
        ttb = ['\cosmos', '\elais', '\\xmm', '\cdfs', '\\adfa', '\\adfb']
        fty = ['UDF', 'DF', 'UDF', 'DF', 'DF', 'DF']
        trans_ddf = dict(zip(tta, ttb))
        trans_ddfb = dict(zip(tta, fty))

        fields = df['note'].unique()

        bands = list('ugrizy')
        dbNameb = dbNameb.split('_')
        dbNameb = '\_'.join(dbNameb)
        caption = '{} strategy: coadded \\fivesig~depth and total number of visits $N_v$ per band.'.format(
            dbNameb)

        caption = '{'+caption+'}'
        label = 'tab:total_depth_{}'.format(io)
        label = '{'+label+'}'
        r = get_beg_table(tab='{table*}', tabcols='{l|c|c|c}',
                          fontsize='\\tiny',
                          caption=caption, label=label, center=True)
        rr = ' & '
        pp = 'Field & '
        seas_max = 10
        # for seas in df['season'].unique():
        for seas in [1]:
            pp += 'season & $m_5$ & $N_v$'
            bb = '/'.join(bands)
            rr += ' & {}'.format(bb)
            rr += ' \\\\'
            pp += ' \\\\'
            """
           if seas < seas_max:
               rr += ' & '
               pp += ' & '
           else:
               rr += ' \\\\'
               pp += ' \\\\'
           """
            r += [pp]
            r += [rr]
            r += [' & & \\\\']
            r += ['\hline']
        for fi in fields:
            idx = df['note'] == fi
            selb = df[idx]
            ll = self.print_latex_depth_field(selb)
            """
            for io, vv in enumerate(ll):
                if io != 5:
                    tt = ' & {}'.format(vv)
                else:
                    tt = '{} & {}'.format(trans_ddf[fi], vv)
                r += [tt]
            """
            for io, vv in enumerate(ll):
                tt = ''
                if io == 10:
                    tt = ' & {}'.format(vv)
                else:
                    if io == 0:
                        tt = '{} & {}'.format(trans_ddfb[fi], vv)
                if tt != '':
                    r += [tt]
            r += ['\hline']
        r += get_end_table(tab='{table*}', center=True)

        """
       r.append('\newpage')
       r.append('\vspace*{20cm}')
       r.append('\newpage')
       """
        # dump in file
        """
       for vv in r:
           print(vv)
       """

        dumpIt(self.outName, r)

    def print_latex_depth_os_single(self, df, dbName, io, dbNameb,
                                    var='m5',
                                    forcap='coadded \\fivesig~depth per band',
                                    varn='$m_5$'):
        """
        Method to print os results

        Parameters
        ----------
        df : pandas df
            Data to print.
        dbName : str
            OS to process.
        io : int
            tag for label.
        dbNameb : str
            OS name to use for printing.

        Returns
        -------
        None.

        """

        tta = ['DD:COSMOS', 'DD:ELAISS1', 'DD:XMM_LSS',
               'DD:ECDFS', 'DD:EDFS_a', 'DD:EDFS_b']
        ttb = ['\cosmos', '\elais', '\\xmm', '\cdfs', '\\adfa', '\\adfb']
        fty = ['UDF', 'DF', 'UDF', 'DF', 'DF', 'DF']
        trans_ddf = dict(zip(tta, ttb))
        trans_ddfb = dict(zip(tta, fty))

        fields = df['note'].unique()

        bands = list('ugrizy')
        dbNameb = dbNameb.split('_')
        dbNameb = '\_'.join(dbNameb)
        # caption = '{} strategy: coadded \\fivesig~depth and total number of visits $N_v$ per band.'.format(
        #    dbNameb)

        caption = '{} strategy: {}.'.format(dbNameb, forcap)
        caption = '{'+caption+'}'
        label = 'tab:total_depth_{}_{}'.format(var, io)
        label = '{'+label+'}'
        r = get_beg_table(tab='{table}', tabcols='{l|c|c}',
                          fontsize='\\tiny',
                          caption=caption, label=label, center=True)
        rr = ' & '
        pp = 'Field & '
        seas_max = 10
        # for seas in df['season'].unique():
        for seas in [1]:
            pp += 'season & '+varn
            bb = '/'.join(bands)
            rr += ' & {}'.format(bb)
            rr += ' \\\\'
            pp += ' \\\\'
            """
            if seas < seas_max:
                rr += ' & '
                pp += ' & '
            else:
                rr += ' \\\\'
                pp += ' \\\\'
            """
            r += [pp]
            r += [rr]
            r += [' & & \\\\']
            r += ['\hline']
        for fi in fields:
            idx = df['note'] == fi
            selb = df[idx]
            ll = self.print_latex_depth_field_single(selb, var=var)
            """
            for io, vv in enumerate(ll):
                if io != 5:
                    tt = ' & {}'.format(vv)
                else:
                    tt = '{} & {}'.format(trans_ddf[fi], vv)
                r += [tt]
            """
            for io, vv in enumerate(ll):
                if io == 11:
                    tt = ' & {}'.format(vv)
                else:
                    if io == 0:
                        tt = '{} & {}'.format(trans_ddf[fi], vv)
                r += [tt]
            r += ['\hline']
        r += get_end_table(tab='{table}', center=True)
        # r.append('\\newpage')
        """
        r.append('\newpage')
        r.append('\vspace*{20cm}')
        r.append('\newpage')
        """
        # dump in file
        """
        for vv in r:
            print(vv)
        """

        dumpIt(self.outName, r)

    def print_latex_depth_field(self, data, bands='ugrizy'):
        """
        Method to print field results in latex mode

        Parameters
        ----------
        data : pandas df
            Data to print.
        bands : str, optional
            List of bands to consider. The default is 'ugrizy'.

        Returns
        -------
        r : list(str)
            Result to be printed.

        """

        r = []
        seasons = range(1, 12, 1)
        seas_tt = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '2-10']
        # seasons = range(1, 11, 1)
        # seas_tt = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        trans_seas = dict(zip(seasons, seas_tt))

        for seas in data['season'].unique():
            idx = data['season'] == seas
            sel = data[idx]
            m5 = []
            nv = []
            for b in list(bands):
                idxb = sel['filter'] == b
                selb = sel[idxb]
                mm5 = selb['m5'].values[0]
                nvv = selb['nvisits'].values[0]
                m5.append('{}'.format(np.round(mm5, 1)))
                nv.append('{}'.format(int(nvv)))
            m5_tot = '/'.join(m5)
            nv_tot = '/'.join(nv)

            rr = '{} & {} & {}'.format(trans_seas[seas], m5_tot, nv_tot)

            rr += '\\\\'
            r.append(rr)
            """
            if seas != 10:
                rr += ' & '
            else:
                rr += ' \\\\'
            """
        return r

    def print_latex_depth_field_single(self, data, bands='ugrizy', var='m5'):
        """
        Method to print field results in latex mode

        Parameters
        ----------
        data : pandas df
            Data to print.
        bands : str, optional
            List of bands to consider. The default is 'ugrizy'.

        Returns
        -------
        r : list(str)
            Result to be printed.

        """

        r = []
        seasons = range(1, 12, 1)
        seas_tt = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '2-10']
        # seasons = range(1, 11, 1)
        # seas_tt = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        trans_seas = dict(zip(seasons, seas_tt))

        for seas in data['season'].unique():
            idx = data['season'] == seas
            sel = data[idx]
            m5 = []
            nv = []
            for b in list(bands):
                idxb = sel['filter'] == b
                selb = sel[idxb]
                mm5 = selb['m5'].values[0]
                nvv = selb['nvisits'].values[0]
                m5.append('{}'.format(np.round(mm5, 1)))
                nv.append('{}'.format(int(nvv)))
            m5_tot = '/'.join(m5)
            nv_tot = '/'.join(nv)

            # rr = '{} & {} & {}'.format(trans_seas[seas], m5_tot, nv_tot)
            gg = m5_tot
            if var == 'nv':
                gg = nv_tot
            rr = '{} & {}'.format(trans_seas[seas], gg)
            rr += '\\\\'
            r.append(rr)
            """
            if seas != 10:
                rr += ' & '
            else:
                rr += ' \\\\'
            """
        return r


class Survey_time:
    def __init__(self, dbDir, configFile, outName='survey_exptime.tex'):
        """
        class to estimate exposure times

        Parameters
        ----------
        dbDir : str
            Data location dir.
        configFile : str
            configuration file (csv).
        outName : str, optional
            Output file name. The default is 'survey_exptime.tex'.

        Returns
        -------
        None.

        """

        self.dbDir = dbDir
        self.configFile = configFile
        self.outName = outName

        # check if the file exist; yes? -> remove it
        if os.path.isfile(outName):
            os.system('rm {}'.format(outName))

        # processing data
        df = self.process_survey_time()

        # print file
        self.print_survey_time(df)

    def process_survey_time(self):
        """
        Method to process survey exptimes

        Returns
        -------
        df : pandas df
            Processed data.

        """

        conf = pd.read_csv(self.configFile)

        df = pd.DataFrame()

        for i, row in conf.iterrows():
            dfa = self.process_OS(row['dbName'])
            dfa['dbName'] = row['dbName']
            dfa['dbNamePlot'] = row['dbNamePlot']
            df = pd.concat((df, dfa))
            # break

        return df

    def process_OS(self, dbName):
        """
        Method to process an observing strategy

        Parameters
        ----------
        dbName : str
            OS name.

        Returns
        -------
        df_tot : pandas df
            Processed data.

        """

        full_path = '{}/{}.npy'.format(self.dbDir, dbName)

        data = np.load(full_path, allow_pickle=True)

        data = pd.DataFrame.from_records(season(data))

        # season 1 stat
        df_y1 = self.get_infos(data, [1])

        # season 2 fields
        df_y2 = pd.DataFrame()
        for vv in ['COSMOS', 'ELAISS1', 'EDFS_a']:
            df_y = self.get_infos(data, [2], vv)
            df_y2 = pd.concat((df_y, df_y2))

        # print(df_y2)
        # all fields, all seasons

        df_tot = pd.concat((df_y1, df_y2))
        # df_all = get_infos(data, range(2, 11))

        idx = data['season'] > 1
        # print(df_all['expTime_sum'].sum(), data[idx]['visitExposureTime'].sum())

        expTime = data[idx]['visitExposureTime'].sum()

        df_all = pd.DataFrame([expTime], columns=['expTime_sum'])
        df_all['season'] = 10
        df_all['field'] = 'all'
        df_all['moon'] = -1
        df_all['expTime_nightly'] = -1

        df_tot = pd.concat((df_tot, df_all))

        return df_tot

    def print_survey_time(self, df):
        """
        Method to print(dump) results

        Parameters
        ----------
        df : pandas df
            Data to dump.

        Returns
        -------
        None.

        """

        # udf
        idx = df['season'] == 2
        sel_y2 = df[idx]

        idd = df['season'] == 10
        sel_all = df[idd]

        rtot = []
        for dbName in sel_y2['dbName'].unique():
            idxa = sel_y2['dbName'] == dbName
            sela = sel_y2[idxa]
            dbNameb = sela['dbNamePlot'].values[0]
            r = [dbNameb]
            for vv in ['COSMOS', 'ELAISS1']:
                rr = self.get_vals(sela, vv)
                r += rr
            # rtot.append(r)
            idf = sel_all['dbName'] == dbName
            r_all = sel_all[idf]['expTime_sum'].mean()
            r_all = np.round(r_all/3600., 1)
            r.append(r_all)
            rtot.append(r)
            # break
        # print(rtot)

        rr = []
        caption = '{Exposure times (in hours) for seasons 2-10.}'
        # caption += ' $\Phi_{Moon}$ is the Moon phase.}'
        rr = get_beg_table(tab='{table}',
                           caption=caption,
                           label='{tab:exptime_1}',
                           tabcols='{l|c|c|c}')
        rr.append(' &  \multicolumn{2}{|c|}{nightly} & \\\\')
        rr.append('Strategy & UDF & DF & survey \\\\')
        # rr.append(' & $\Phi_{Moon}\leq 20\\%$ & $\Phi_{Moon} > 20\\%$ \
        #     & $\Phi_{Moon}\leq 20\\%$ & $\Phi_{Moon} > 20\\% $& \\\\')
        # rr.append(' & $\Phi_{Moon}$ & $\Phi_{Moon}$ \
        #      & $\Phi_{Moon}$ & $\Phi_{Moon}$& \\\\')
        # rr.append(' & $\leq 20\\%$ & $> 20\\%$ \
        #      & $\leq 20\\%$ & $> 20\\% $& \\\\')
        rr.append('\hline')

        for vv in rtot:
            # dbNameb = '_'.join(vv[0].split('_')
            dbNameb = vv[0].replace('_', '\_')
            rr.append('{} & {} & {} & {} \\\\'.format(
                dbNameb, vv[2], vv[4], vv[5]))
        rr.append('\\hline')
        rr += get_end_table(tab='{table}')

        # dump data
        dumpIt(self.outName, rr)

    def print_survey_time_orig(self, df):
        """
        Method to print(dump) results

        Parameters
        ----------
        df : pandas df
            Data to dump.

        Returns
        -------
        None.

        """

        # udf
        idx = df['season'] == 2
        sel_y2 = df[idx]

        idd = df['season'] == 10
        sel_all = df[idd]

        rtot = []
        for dbName in sel_y2['dbName'].unique():
            idxa = sel_y2['dbName'] == dbName
            sela = sel_y2[idxa]
            dbNameb = sela['dbNamePlot'].values[0]
            r = [dbNameb]
            for vv in ['COSMOS', 'ELAISS1']:
                rr = self.get_vals(sela, vv)
                r += rr
            # rtot.append(r)
            idf = sel_all['dbName'] == dbName
            r_all = sel_all[idf]['expTime_sum'].mean()
            r_all = np.round(r_all/3600., 1)
            r.append(r_all)
            rtot.append(r)
            # break
        # print(rtot)

        rr = []
        caption = '{Exposure times (in hours) for seasons 2-10.'
        caption += ' $\Phi_{Moon}$ is the Moon phase.}'
        rr = get_beg_table(tab='{table}',
                           caption=caption,
                           label='{tab:exptime_1}',
                           tabcols='{l|c|c|c|c|c}')
        rr.append(' & \multicolumn{4}{|c|}{nightly} & \\\\')
        rr.append(
            'Strategy & \multicolumn{2}{|c|}{UDF} & \
                \multicolumn{2}{|c|}{DF} & survey \\\\')
        # rr.append(' & $\Phi_{Moon}\leq 20\\%$ & $\Phi_{Moon} > 20\\%$ \
        #     & $\Phi_{Moon}\leq 20\\%$ & $\Phi_{Moon} > 20\\% $& \\\\')
        rr.append(' & $\Phi_{Moon}$ & $\Phi_{Moon}$ \
              & $\Phi_{Moon}$ & $\Phi_{Moon}$& \\\\')
        rr.append(' & $\leq 20\\%$ & $> 20\\%$ \
              & $\leq 20\\%$ & $> 20\\% $& \\\\')
        rr.append('\hline')

        for vv in rtot:
            # dbNameb = '_'.join(vv[0].split('_')
            dbNameb = vv[0].replace('_', '\_')
            rr.append('{} & {} & {} & {} & {} & {} \\\\'.format(
                dbNameb, vv[1], vv[2], vv[3], vv[4], vv[5]))

        rr += get_end_table(tab='{table}')

        # dump data
        dumpIt(self.outName, rr)

    def get_vals(self, sela, field):
        """
        Method to get rounded exptimes

        Parameters
        ----------
        sela : pandas df
            Data to process.
        field : str
            Field name.

        Returns
        -------
        list(float)
            exptimes in hours.

        """

        idxb = sela['field'] == field
        selb = sela[idxb]

        idxc = selb['moon'] == 0
        ra = selb[idxc]['expTime_nightly'].values[0]
        rb = selb[~idxc]['expTime_nightly'].values[0]

        ra /= 3600
        rb /= 3600.
        return [np.round(ra, 1), np.round(rb, 1)]

    def nvisits(self, grp, varsel, valsel, op):
        """
        Method to get nvisits

        Parameters
        ----------
        grp : pandas df
            Data to process.
        varsel : str
            Selection var name.
        valsel : float
            selection value.
        op : operator
            Operator to apply.

        Returns
        -------
        pandas df
            Processed data.

        """

        idx = op(grp[varsel], valsel)

        sel = grp[idx]

        nvisits = len(sel)
        expTime = sel['visitExposureTime'].sum()

        return pd.DataFrame([[expTime, len(sel)]],
                            columns=['expTime', 'nvisits'])

    def nightly_visits(self, data):
        """
        Method to estimate nightly visits

        Parameters
        ----------
        data : pandas df
            Data to process.

        Returns
        -------
        df : pandas df
            processed data.

        """

        # get the nightly number of visits
        nv_moon = data.groupby(['note', 'night']).apply(
            lambda x: self.nvisits(x, 'moonPhase', 20, operator.gt))
        nv_nomoon = data.groupby(['note', 'night']).apply(
            lambda x: self.nvisits(x, 'moonPhase', 20, operator.le))

        dfa = self.calc_exptime(nv_moon)
        dfb = self.calc_exptime(nv_nomoon)

        dfa['moon'] = 1
        dfb['moon'] = 0

        df = pd.concat((dfa, dfb))

        return df

    def calc_exptime(self, data):
        """
        Method to estimate exposure time

        Parameters
        ----------
        data : pandas df
            Data to process.

        Returns
        -------
        pandas df
            Processed data.

        """

        idx = data['expTime'] > 0.
        sel_data = data[idx]

        med_exptime = sel_data['expTime'].median()
        sum_exptime = sel_data['expTime'].sum()

        return pd.DataFrame([[med_exptime, sum_exptime]],
                            columns=['expTime_nightly', 'expTime_sum'])

    def get_infos(self, data, season, field='all'):
        """
        Method to get field infos

        Parameters
        ----------
        data : pandas df
            Data to process.
        season : int
            Season to process.
        field : str, optional
            Field to process. The default is 'all'.

        Returns
        -------
        dfb : pandas df
            Processed data.

        """

        dfb = pd.DataFrame()
        for seas in season:
            idx = data['season'] == seas
            if field != 'all':
                idx &= data['note'] == 'DD:{}'.format(field)
            data_y = data[idx]

            df_y = self.nightly_visits(data_y)

            df_y['season'] = seas
            df_y['field'] = field

            dfb = pd.concat((dfb, df_y))

        return dfb
