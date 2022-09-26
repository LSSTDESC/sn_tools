import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy.lib.recfunctions as rf
import h5py
from astropy.table import Table, Column, vstack
from scipy.interpolate import griddata, interpn, CloughTocher2DInterpolator, LinearNDInterpolator
from sn_tools.sn_telescope import Telescope
from sn_tools.sn_io import Read_Sqlite
from sn_tools.sn_obs import renameFields, getFields
from sn_tools.sn_io import getObservations
import pandas as pd
from sn_tools.sn_obs import DataInside, season
from sn_tools.sn_clusters import ClusterObs
from sn_tools.sn_utils import multiproc


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


class GenerateFakeObservations:
    """
    class to generate Fake observations

    Parameters
    ---------------

    config: dict
      dict of parameters
    list : str,opt
        Name of the columns for data generation.
        Default : 'observationStartMJD', 'fieldRA', 'fieldDec','filter','fiveSigmaDepth','visitExposureTime','numExposures','visitTime','season'

    """

    def __init__(self, config,
                 mjdCol='observationStartMJD', RACol='fieldRA',
                 DecCol='fieldDec', filterCol='filter', m5Col='fiveSigmaDepth',
                 exptimeCol='visitExposureTime', nexpCol='numExposures',
                 seasonCol='season', seeingEffCol='seeingFwhmEff', seeingGeomCol='seeingFwhmGeom',
                 visitTime='visitTime',
                 sequences=False):

        # config = yaml.load(open(config_filename))
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RACol = RACol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = seasonCol
        self.nexpCol = nexpCol
        self.seeingEffCol = seeingEffCol
        self.seeingGeomCol = seeingGeomCol
        self.visitTime = visitTime

        # now make fake obs
        if not sequences:
            if config['m5File'] == 'NoData':
                self.makeFake(config)
            else:
                self.makeFake_from_simu(config)
        else:
            self.makeFake_sqs(config)

    def makeFake_sqs(self, config):
        """ Generate Fake observations

        Parameters
        -------------
        config : dict
           dict of parameters (config file)

        Returns
        ---------
        recordarray of observations with the fields:
        MJD, RA, Dec, band,m5,Nexp, ExpTime, Season
        accessible through self.Observations

        """

        bands = config['bands']
        cadence = config['cadence']
        shift_days = dict(
            zip(bands, [config['shift_days']*io for io in range(len(bands))]))
        # m5 = dict(zip(bands, config['m5']))
        # Nvisits = dict(zip(bands, config['Nvisits']))
        # Single_Exposure_Time = dict(zip(bands, config['Single_Exposure_Time']))
        inter_season_gap = 100.
        seeingEff = dict(zip(bands, config['seeingEff']))
        seeingGeom = dict(zip(bands, config['seeingGeom']))
        airmass = dict(zip(bands, config['airmass']))
        sky = dict(zip(bands, config['sky']))
        moonphase = dict(zip(bands, config['moonphase']))
        RA = config['RA']
        Dec = config['Dec']
        rtot = []
        # for season in range(1, config['nseasons']+1):
        Nvisits = {}
        Single_Exposure_Time = {}
        for il, season in enumerate(config['seasons']):
            m5 = dict(zip(bands, config['m5'][season]))
            mjd_min = config['MJD_min']+il * \
                (config['season_length'][season]+inter_season_gap)
            mjd_max = mjd_min+config['season_length'][season]
            n_visits = config['Nvisits'][season]
            seqs = config['sequences'][season]
            sing_exp_time = config['Single_Exposure_Time'][season]
            cadence = config['cadence'][season]

            for i in range(len(seqs)):
                # for i,val in enumerate(config['sequences']):
                mjd_min_seq = mjd_min+i*config['deltaT_seq'][i]
                mjd_max_seq = mjd_max+i*config['deltaT_seq'][i]
                mjd = np.arange(mjd_min_seq, mjd_max_seq+cadence, cadence)
                night = (mjd-config['MJD_min']).astype(int)+1

                for j, band in enumerate(seqs[i]):
                    mjd += shift_days[band]
                    Nvisits[band] = n_visits[i][j]
                    Single_Exposure_Time[band] = sing_exp_time[i][j]
                    m5_coadded = self.m5coadd(m5[band],
                                              Nvisits[band],
                                              Single_Exposure_Time[band])

                    myarr = np.array(mjd, dtype=[(self.mjdCol, 'f8')])
                    myarr = rf.append_fields(myarr, 'night', night)
                    myarr = rf.append_fields(myarr, [self.RACol, self.DecCol, self.filterCol], [
                        [RA]*len(myarr), [Dec]*len(myarr), [band]*len(myarr)])
                    myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                        [m5_coadded]*len(myarr), [Nvisits[band]]*len(myarr), [Nvisits[band]*Single_Exposure_Time[band]]*len(myarr), [season]*len(myarr)])
                    myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                        [seeingEff[band]]*len(myarr), [seeingGeom[band]]*len(myarr)])
                    myarr = rf.append_fields(myarr, self.visitTime, [
                                             Nvisits[band]*Single_Exposure_Time[band]]*len(myarr))
                    myarr = rf.append_fields(myarr, ['airmass', 'sky', 'moonPhase'], [
                        [airmass[band]]*len(myarr), [sky[band]]*len(myarr), [moonphase[band]]*len(myarr)])
                    rtot.append(myarr)

            """
            for i, band in enumerate(bands):
                mjd = np.arange(mjd_min, mjd_max+cadence[band],cadence[band])
                # if mjd_max not in mjd:
                #    mjd = np.append(mjd, mjd_max)
                mjd += shift_days[band]
                m5_coadded = self.m5coadd(m5[band],
                                          Nvisits[band],
                                          Exposure_Time[band])

                myarr = np.array(mjd, dtype=[(self.mjdCol, 'f8')])
                myarr = rf.append_fields(myarr, [self.RACol, self.DecCol, self.filterCol], [
                                         [RA]*len(myarr), [Dec]*len(myarr), [band]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                                         [m5_coadded]*len(myarr), [Nvisits[band]]*len(myarr), [Nvisits[band]*Exposure_Time[band]]*len(myarr), [season]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                                         [seeingEff[band]]*len(myarr), [seeingGeom[band]]*len(myarr)])
                rtot.append(myarr)
            """
        res = np.copy(np.concatenate(rtot))
        res.sort(order=self.mjdCol)

        res = rf.append_fields(res, 'observationId',
                               np.random.randint(10*len(res), size=len(res)))

        self.Observations = res

    def makeFake(self, config):
        """ Generate Fake observations

        Parameters
        ---------------
        config : dict
           dict of parameters (config file)

        Returns
        -----------
        recordarray of observations with the fields:
        MJD, RA, Dec, band,m5,Nexp, ExpTime, Season
        accessible through self.Observations

        """

        bands = config['bands']
        #cadence = dict(zip(bands, config['cadence']))
        cadence = config['cadence']
        shift_days = dict(
            zip(bands, [config['shiftDays']*io for io in range(len(bands))]))
        #m5 = dict(zip(bands, config['m5']))
        m5 = config['m5']
        Nvisits = config['Nvisits']
        Exposure_Time = config['ExposureTime']
        seeingEff = config['seeingEff']
        seeingGeom = config['seeingGeom']
        airmass = config['airmass']
        sky = config['sky']
        moonphase = config['moonphase']
        """
        Nvisits = dict(zip(bands, config['Nvisits']))
        Exposure_Time = dict(zip(bands, config['Exposure_Time']))
       
        seeingEff = dict(zip(bands, config['seeingEff']))
        seeingGeom = dict(zip(bands, config['seeingGeom']))
        airmass = dict(zip(bands, config['airmass']))
        sky = dict(zip(bands, config['sky']))
        moonphase = dict(zip(bands, config['moonphase']))
        """
        inter_season_gap = 300.
        RA = config['RA']
        Dec = config['Dec']
        rtot = []
        # for season in range(1, config['nseasons']+1):
        for il, season in enumerate(config['seasons']):
            # mjd_min = config['MJD_min'] + float(season-1)*inter_season_gap
            mjd_min = config['MJDmin'][il]
            mjd_max = mjd_min+config['seasonLength'][il]

            for i, band in enumerate(bands):
                mjd = np.arange(mjd_min, mjd_max+cadence[band], cadence[band])
                # if mjd_max not in mjd:
                #    mjd = np.append(mjd, mjd_max)
                mjd += shift_days[band]
                m5_coadded = self.m5coadd(m5[band],
                                          Nvisits[band],
                                          Exposure_Time[band])

                myarr = np.array(mjd, dtype=[(self.mjdCol, 'f8')])
                myarr = rf.append_fields(myarr, [self.RACol, self.DecCol, self.filterCol], [
                                         [RA]*len(myarr), [Dec]*len(myarr), [band]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                                         [m5_coadded]*len(myarr), [Nvisits[band]]*len(myarr), [Nvisits[band]*Exposure_Time[band]]*len(myarr), [season]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                                         [seeingEff[band]]*len(myarr), [seeingGeom[band]]*len(myarr)])
                myarr = rf.append_fields(myarr, ['airmass', 'sky', 'moonPhase'], [
                                         [airmass[band]]*len(myarr), [sky[band]]*len(myarr), [moonphase[band]]*len(myarr)])
                rtot.append(myarr)

        res = np.copy(np.concatenate(rtot))
        res.sort(order=self.mjdCol)

        res = rf.append_fields(res, 'observationId',
                               np.random.randint(10*len(res), size=len(res)))

        self.Observations = res

    def makeFake_from_simu(self, config):
        """ Generate Fake observations
        fiveSigmaDepth are taken from an input file

        Parameters
        ---------------
        config : dict
           dict of parameters (config file)

        Returns
        -----------
        recordarray of observations with the fields:
        MJD, RA, Dec, band,m5,Nexp, ExpTime, Season
        accessible through self.Observations

        """

        bands = config['bands']
        #cadence = dict(zip(bands, config['cadence']))
        cadence = config['cadence']
        shift_days = dict(
            zip(bands, [config['shiftDays']*io for io in range(len(bands))]))
        #m5 = dict(zip(bands, config['m5']))
        m5 = config['m5']
        Nvisits = config['Nvisits']
        Exposure_Time = config['ExposureTime']
        seeingEff = config['seeingEff']
        seeingGeom = config['seeingGeom']
        airmass = config['airmass']
        sky = config['sky']
        moonphase = config['moonphase']

        RA = config['RA']
        Dec = config['Dec']
        rtot = []
        # prepare m5 for interpolation

        # for season in range(1, config['nseasons']+1):
        for il, season in enumerate(config['seasons']):
            m5_interp, mjds = self.m5Interp(
                season, config['m5File'], config['healpixID'])
            # search for mjdmin and mjdmax for this season
            mjd_min = mjds[0]
            mjd_max = mjds[1]
            for i, band in enumerate(bands):
                mjd = np.arange(mjd_min, mjd_max+cadence[band], cadence[band])
                # if mjd_max not in mjd:
                #    mjd = np.append(mjd, mjd_max)
                #mjd += shift_days[band]
                m5_coadded = self.m5coadd(m5_interp[band](mjd),
                                          Nvisits[band],
                                          Exposure_Time[band])
                myarr = np.array(mjd, dtype=[(self.mjdCol, 'f8')])

                myarr = rf.append_fields(myarr, [self.RACol, self.DecCol, self.filterCol], [
                                         [RA]*len(myarr), [Dec]*len(myarr), [band]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                                         m5_coadded, [Nvisits[band]]*len(myarr), [Nvisits[band]*Exposure_Time[band]]*len(myarr), [season]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                                         [seeingEff[band]]*len(myarr), [seeingGeom[band]]*len(myarr)])
                myarr = rf.append_fields(myarr, ['airmass', 'sky', 'moonPhase'], [
                                         [airmass[band]]*len(myarr), [sky[band]]*len(myarr), [moonphase[band]]*len(myarr)])
                rtot.append(myarr)

        res = np.copy(np.concatenate(rtot))
        res.sort(order=self.mjdCol)

        res = rf.append_fields(res, 'observationId',
                               np.random.randint(10*len(res), size=len(res)))

        self.Observations = res

    def m5Interp(self, season, fName, healpixID):
        """
        Method to prepare interpolation of m5 vs time per band

        Parameters
        --------------
        season: int
          season number
        fName: str
           name of the m5 file with ref values
        healpixID: int
          healpixID to tag for ref values

        Returns
        ----------
        dictout: dict
          dict of interpolators (time,m5) (key: band)
        mjds: pair
          mjds min and max of the season

        """

        # load the file
        tab = np.load(fName, allow_pickle=True)
        dictout = {}

        # now make interps for band and seasons
        # and get mjdmin and mjdmax

        idx = tab['season'] == season
        if healpixID != -1:
            idx &= tab['healpixID'] == healpixID

        sela = tab[idx]

        rmin = []
        rmax = []
        for b in np.unique(sela['filter']):
            idxb = sela['filter'] == b
            selb = sela[idxb]
            dictout[b] = interpolate.interp1d(
                selb['observationStartMJD'], selb['fiveSigmaDepth'], fill_value=0., bounds_error=False)
            rmin.append(np.min(selb['observationStartMJD']))
            rmax.append(np.max(selb['observationStartMJD']))
        mjds = (np.max(rmin), np.min(rmax))

        return dictout, mjds

    def m5coadd(self, m5, Nvisits, Tvisit):
        """ Coadded :math:`m_{5}` estimation
        use approx. :math:`\Delta m_{5}=1.25*log_{10}(N_{visits}*T_{visits}/30.)
        with : :math:`N_{visits}` : number of visits
                    :math:`T_{visits}` : single visit exposure time

        Parameters
        --------------
        m5 : list(float)
           list of five-sigma depth values
         Nvisits : list(float)
           list of the number of visits
          Tvisit : list(float)
           list of the visit times

       Returns
        ---------
       m5_coadd : list(float)
          list of m5 coadded values

        """
        m5_coadd = m5+1.25*np.log10(float(Nvisits)*Tvisit/30.)
        return m5_coadd


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

        print('DD budget', budget)

        params = {}
        params['obs_DD'] = self.obs_DD
        params['dbName'] = self.dbName
        res = multiproc(
            np.unique(self.obs_DD['note']), params, ana_DDF, 6)

        tab = Table.from_pandas(res)
        tab.meta = dict(zip(['dbName', 'time_budget'], [dbName, budget]))

        self.summary = tab

    def load(self):
        """
        Method to load data

        Returns
        ----------
        numpy array of data
        """

        fName = '{}/{}.{}'.format(self.dbDir, self.dbName, self.dbExtens)
        data = np.load(fName)

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


def time_budget(obs, obs_DD):
    """"
    Method to estimate the time budget from DD

    Returns
    -----------
    time budget (float)
    """
    return len(obs_DD)/len(obs)


def ana_DDF(list_DD, params, j, output_q):
    """
    Method to analyze DDFs

    """
    obs_DD = params['obs_DD']
    dbName = params['dbName']
    res_DD = pd.DataFrame()

    for field in list_DD:
        print('analyzing', field)
        idx = obs_DD['note'] == field
        res = ana_field(np.copy(obs_DD[idx]), dbName)
        res_DD = pd.concat((res_DD, res))

    if output_q is not None:
        return output_q.put({j: res_DD})
    else:
        return res_DD


def ana_field(obs, dbName):
    """
    Method to analyze a single DD field

    Parameters
    --------------
    obs: array
        observation corresponding to a single DD field

    """
    # estimate seasons
    obs = season(obs, mjdCol='mjd')
    #print('aoooo', np.unique(obs[['note', 'season']]), obs.dtype.names)
    # print(test)
    field = np.unique(np.copy(obs['note']))[0]

    res = ana_visits(obs, field)
    res['dbName'] = dbName

    return res


def ana_visits(obs, field):
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

    list_moon = ['moonAz', 'moonRA', 'moonDec', 'moonDistance', 'season']
    rtot = pd.DataFrame()
    for night in np.unique(obs['night']):
        idx = obs['night'] == night
        obs_night = obs[idx]
        #print('night', night)

        dd = {}
        dd['field'] = field
        dd['night'] = night
        for ll in list_moon:
            dd[ll] = np.median(obs_night[ll])

        for b in np.unique(obs_night['band']):
            idb = obs_night['band'] == b
            obs_filter = obs_night[idb]
            dd[b] = len(obs_filter)

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


def seas_cad(obs):

    dictout = {}

    # get cadence and season length
    seas_min, seas_max = np.min(obs['night']), np.max(obs['night'])
    seas_length = seas_max-seas_min
    diff = np.diff(obs['night'])
    cad_med, cad, cad_std = np.median(diff), np.mean(diff), np.std(diff)

    dictout['cadence_median'] = [cad_med]
    dictout['cadence_mean'] = [cad]
    dictout['cadence_std'] = [cad_std]
    dictout['season_length'] = [seas_length]

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

    return pd.DataFrame.from_dict(dictout)


def Stat_DD_season(data_tab, cols=['field', 'season']):
    """
    Method to analyze a set of observing data per obs night

    Parameters
    --------------
    data_tab: astopy table
      data to process

    """

    print(data_tab.meta)

    res = data_tab.to_pandas().groupby(cols).apply(
        lambda x: seas_cad(x)).reset_index()

    res['dbName'] = data_tab.meta['dbName']
    res['time_budget'] = np.round(data_tab.meta['time_budget'], 3)
    return res
