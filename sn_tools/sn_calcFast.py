import multiprocessing
import pandas as pd
import numpy as np
from astropy.table import Table, Column, vstack
import time
#from scipy.linalg import lapack
# lapack_routine = lapack_lite.dgesv
import scipy.linalg as la
import operator
#import numpy.lib.recfunctions as rf
from sn_tools.sn_io import check_get_file
import h5py
from scipy.interpolate import RegularGridInterpolator

import warnings
# this is to remove runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class LCfast:
    """
    class to simulate supernovae light curves in a fast way
    The method relies on templates and broadcasting to increase speed

    Parameters
    ---------------
    reference_lc: RegularGridData
        lc reference files
    dustcorr: dict of dict of RegularGridData
       dust correction map
    x1: float
      SN stretch
    color: float
      SN color
    mjdCol: str, opt
      name of the MJD col in data to simulate (default: observationStartMJD)
    RACol: str, opt
      name of the RA col in data to simulate (default: fieldRA)
    DecCol: str, opt
       name of the Dec col in data to simulate (default: fieldDec)
    filterCol: str, opt
       name of the filter col in data to simulate (default: filter)
    exptimeCol: str, opt
      name of the exposure time  col in data to simulate (default: visitExposureTime)
    m5Col: str, opt
       name of the fiveSigmaDepth col in data to simulate (default: fiveSigmaDepth)
    seasonCol: str, opt
       name of the season col in data to simulate (default: season)
    snr_min: float, opt
       minimal Signal-to-Noise Ratio to apply on LC points (default: 5)
    lightOutput: bool, opt
        to get a lighter output (ie lower number of cols) (default: True)
    bluecutoff: float,opt
       blue cutoff for SN (default: 380.0 nm)
    redcutoff: float, opt
       red cutoff for SN (default: 800.0 nm)
    """

    def __init__(self, reference_lc, dustcorr, x1, color, mjdCol='observationStartMJD',
                 RACol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', exptimeCol='visitExposureTime',
                 m5Col='fiveSigmaDepth', seasonCol='season', nexpCol='numExposures', seeingCol='seeingFwhmEff',
                 snr_min=5.,
                 lightOutput=True,
                 ebvofMW=-1.0,
                 bluecutoff=380.0,
                 redcutoff=800.0):

        # grab all vals
        self.RACol = RACol
        self.DecCol = DecCol
        self.filterCol = filterCol
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.exptimeCol = exptimeCol
        self.seasonCol = seasonCol
        self.nexpCol = nexpCol
        self.seeingCol = seeingCol

        self.x1 = x1
        self.color = color
        self.lightOutput = lightOutput
        self.ebvofMW = ebvofMW
        self.dustcorr = dustcorr
        # Loading reference file
        self.reference_lc = reference_lc

        # This cutoffs are used to select observations:
        # phase = (mjd - DayMax)/(1.+z)
        # selection: min_rf_phase < phase < max_rf_phase
        # and        blue_cutoff < mean_rest_frame < red_cutoff
        # where mean_rest_frame = telescope.mean_wavelength/(1.+z)
        self.blue_cutoff = bluecutoff
        self.red_cutoff = redcutoff

        # SN parameters for Fisher matrix estimation
        self.param_Fisher = ['x0', 'x1', 'daymax', 'color']

        self.snr_min = snr_min

    def __call__(self, obs, ebvofMW, gen_par=None, bands='grizy'):
        """ Simulation of the light curve


        Parameters
        ----------------
        obs: array
         array of observations
        ebvofMW: float
           E(B-V) for MW
        gen_par: array, opt
         simulation parameters (default: None)
        bands: str, opt
          filters to consider for simulation (default: grizy)


        Returns
        ------------
        astropy table with:
        columns: band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
        metadata : SNID,RA,Dec,DayMax,X1,Color,z
        """

        if len(obs) == 0:
            return None

        # result in this df
        tab_tot = pd.DataFrame()

        # loop on the bands
        for band in bands:
            idx = obs[self.filterCol] == band
            if len(obs[idx]) > 0:
                resband = self.processBand(obs[idx], ebvofMW, band, gen_par)
                #tab_tot = tab_tot.append(resband, ignore_index=True)
                tab_tot = pd.concat((tab_tot, resband))

        # return produced LC
        return tab_tot

    def call_multiproc(self, obs, gen_par=None, bands='grizy'):
        """ Simulation of the light curve
        This method uses multiprocessing (one band per process) to increase speed

        Parameters
        ----------------
        obs: array
         array of observations
        gen_par: array, opt
         simulation parameters (default: None)
        bands: str, opt
          filters to consider for simulation (default: grizy)

        Returns
        ------------
        astropy table with:
        columns: band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
        metadata : SNID,RA,Dec,DayMax,X1,Color,z
        """

        ra = np.mean(obs[self.RACol])
        dec = np.mean(obs[self.DecCol])

        if len(obs) == 0:
            return None

        result_queue = multiprocessing.Queue()

        tab_tot = pd.DataFrame()

        # multiprocessing here: one process (processBand) per band
        jproc = -1
        for j, band in enumerate(bands):
            idx = obs[self.filterCol] == band
            # print('multiproc',band,j,len(obs[idx]))
            if len(obs[idx]) > 0:
                jproc += 1
                p = multiprocessing.Process(name='Subprocess-'+str(
                    j), target=self.processBand, args=(obs[idx], band, gen_par, jproc, result_queue))
                p.start()

        resultdict = {}
        for j in range(jproc+1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        for j in range(jproc+1):
            if not resultdict[j].empty:
                tab_tot = tab_tot.append(resultdict[j], ignore_index=True)

        # return produced LC
        return tab_tot

    def processBand(self, sel_obs, ebvofMW, band, gen_par, j=-1, output_q=None):
        """ LC simulation of a set of obs corresponding to a band
        The idea is to use python broadcasting so as to estimate
        all the requested values (flux, flux error, Fisher components, ...)
        in a single path (i.e no loop!)

        Parameters
        ---------------
        sel_obs: array
         array of observations
        band: str
         band of observations
        gen_par: array
         simulation parameters
        j: int, opt
         index for multiprocessing (default: -1)
        output_q: multiprocessing.Queue(),opt
         queue for multiprocessing (default: None)


        Returns
        -------
        astropy table with fields corresponding to LC components

        """

        # if there are no observations in this filter: return None
        if len(sel_obs) == 0:
            if output_q is not None:
                output_q.put({j: None})
            else:
                return None

        # Get the fluxes (from griddata reference)

        # xi = MJD-T0
        xi = sel_obs[self.mjdCol]-gen_par['daymax'][:, np.newaxis]

        # yi = redshift simulated values
        # requested to avoid interpolation problems near boundaries
        yi = np.round(gen_par['z'], 4)
        # yi = gen_par['z']

        # p = phases of LC points = xi/(1.+z)
        p = xi/(1.+yi[:, np.newaxis])
        yi_arr = np.ones_like(p)*yi[:, np.newaxis]

        # get fluxes
        pts = (p, yi_arr)
        fluxes_obs = self.reference_lc.flux[band](pts)
        fluxes_model_err = self.reference_lc.fluxerr_model[band](pts)

        # Fisher components estimation

        dFlux = {}

        # loop on Fisher parameters
        for val in self.param_Fisher:
            dFlux[val] = self.reference_lc.param[band][val](pts)

        # Fisher matrix components estimation
        # loop on SN parameters (x0,x1,color)
        # estimate: dF/dxi*dF/dxj
        Derivative_for_Fisher = {}
        for ia, vala in enumerate(self.param_Fisher):
            for jb, valb in enumerate(self.param_Fisher):
                if jb >= ia:
                    Derivative_for_Fisher[vala +
                                          valb] = dFlux[vala] * dFlux[valb]

        flag = self.getFlag(sel_obs, gen_par, fluxes_obs, band, p)
        flag_idx = np.argwhere(flag)

        # now apply the flag to select LC points - masked arrays
        fluxes = self.marray(fluxes_obs, flag)
        fluxes_err_model = self.marray(fluxes_model_err, flag)
        phases = self.marray(p, flag)
        z_vals = gen_par['z'][flag_idx[:, 0]]
        daymax_vals = gen_par['daymax'][flag_idx[:, 0]]
        Fisher_Mat = {}
        for key, vals in Derivative_for_Fisher.items():
            Fisher_Mat[key] = self.marray(vals, flag)

        ndata = len(fluxes[~fluxes.mask])

        if ndata <= 0:
            return pd.DataFrame()

        nvals = len(phases)
        # masked - tile arrays

        sel_obs['healpixID'] = sel_obs['healpixID'].astype(int)

        cols = [self.mjdCol, self.seasonCol, self.m5Col, self.exptimeCol,
                self.nexpCol, self.seeingCol, 'night', 'healpixID', 'pixRA', 'pixDec']

        # take columns common with obs cols
        colcoms = set(cols).intersection(sel_obs.dtype.names)

        masked_tile = {}
        for vv in list(colcoms):
            masked_tile[vv] = self.tarray(sel_obs[vv], nvals, flag)

        # mag_obs = np.ma.array(mag_obs, mask=~flag)

        # put data in a pandas df
        lc = pd.DataFrame()

        lc['flux'] = fluxes[~fluxes.mask]
        lc['mag'] = -2.5*np.log10(lc['flux']/3631.)
        lc['phase'] = phases[~phases.mask]
        lc['band'] = ['LSST::'+band]*len(lc)
        lc['zp'] = self.reference_lc.zp[band]
        lc['zp'] = 2.5*np.log10(3631)
        lc['zpsys'] = 'ab'
        lc['z'] = z_vals
        lc['daymax'] = daymax_vals

        for key, vals in masked_tile.items():
            lc[key] = vals[~vals.mask]

        lc['time'] = lc[self.mjdCol]
        lc.loc[:, 'x1'] = self.x1
        lc.loc[:, 'color'] = self.color

        # estimate errors
        lc['fluxerr_model'] = fluxes_err_model[~fluxes_err_model.mask]
        lc['flux_e_sec'] = self.reference_lc.mag_to_flux[band]((
            lc['mag'], lc[self.exptimeCol]/lc[self.nexpCol], lc[self.nexpCol]))
        lc['flux_5_old'] = 10**(-0.4*(lc[self.m5Col] -
                                      self.reference_lc.zp[band]))
        lc['flux_5'] = self.reference_lc.mag_to_flux[band]((
            lc[self.m5Col], lc[self.exptimeCol]/lc[self.nexpCol], lc[self.nexpCol]))
        lc['snr_m5'] = lc['flux_e_sec'] / \
            np.sqrt((lc['flux_5']/5.)**2 +
                    lc['flux_e_sec']/lc[self.exptimeCol])
        lc['fluxerr_photo'] = lc['flux']/lc['snr_m5']
        lc['magerr'] = (2.5/np.log(10.))/lc['snr_m5']
        lc['fluxerr'] = np.sqrt(
            lc['fluxerr_photo']**2+lc['fluxerr_model']**2)
        lc['snr'] = lc['snr_m5']
        # Fisher matrix components
        for key, vals in Fisher_Mat.items():
            lc.loc[:, 'F_{}'.format(
                key)] = vals[~vals.mask]/(lc['fluxerr_photo'].values**2)
        
        lc.loc[:, 'n_aft'] = (np.sign(lc['phase']) == 1) & (
            lc['snr_m5'] >= self.snr_min)
        lc.loc[:, 'n_bef'] = (np.sign(lc['phase'])
                              == -1) & (lc['snr_m5'] >= self.snr_min)
        

        lc.loc[:, 'n_phmin'] = (lc['phase'] <= -5.)
        lc.loc[:, 'n_phmax'] = (lc['phase'] >= 20)
        
        
        # remove lc points with no flux
        idx = lc['flux_e_sec'] > 0.
        lc_flux = lc[idx]

        if len(lc_flux) > 0.:
            lc_flux = self.dust_corrections(lc_flux, ebvofMW)

        if output_q is not None:
            output_q.put({j: lc_flux})
        else:
            return lc_flux

    def getFlag(self, sel_obs, gen_par, fluxes_obs, band, p):
        """
        Method to flag events corresponding to selection criteria

        Parameters
        ---------------
        sel_obs: array
          observations
        gen_par: array
           simulation parameters
        fluxes_obs: array
           LC fluxes
        band: str
          filter
        p: array
          LC phases
        """
        # remove LC points outside the restframe phase range
        min_rf_phase = gen_par['minRFphase'][:, np.newaxis]
        max_rf_phase = gen_par['maxRFphase'][:, np.newaxis]
        flag = (p >= min_rf_phase) & (p <= max_rf_phase)

        # remove LC points outside the (blue-red) range
        mean_restframe_wavelength = np.array(
            [self.reference_lc.mean_wavelength[band]]*len(sel_obs))
        mean_restframe_wavelength = np.tile(
            mean_restframe_wavelength, (len(gen_par), 1))/(1.+gen_par['z'][:, np.newaxis])
        # flag &= (mean_restframe_wavelength > 0.) & (
        #    mean_restframe_wavelength < 1000000.)
        flag &= (mean_restframe_wavelength > self.blue_cutoff) & (
            mean_restframe_wavelength < self.red_cutoff)

        # remove points with neg flux
        flag &= fluxes_obs > 1.e-10

        return flag

    def marray(self, val, flag):

        return np.ma.array(val, mask=~flag)

    def tarray(self, arr, nvals, flag):

        vv = np.ma.array(np.tile(arr, (nvals, 1)), mask=~flag)
        return vv

    def processBand_gamma(self, sel_obs, ebvofMW, band, gen_par, j=-1, output_q=None):
        """ LC simulation of a set of obs corresponding to a band
        The idea is to use python broadcasting so as to estimate
        all the requested values (flux, flux error, Fisher components, ...)
        in a single path (i.e no loop!)

        Parameters
        ---------------
        sel_obs: array
         array of observations
        band: str
         band of observations
        gen_par: array
         simulation parameters
        j: int, opt
         index for multiprocessing (default: -1)
        output_q: multiprocessing.Queue(),opt
         queue for multiprocessing (default: None)


        Returns
        -------
        astropy table with fields corresponding to LC components

        """

        # method used for interpolation
        method = 'linear'
        interpType = 'griddata'
        interpType = 'regular'

        # if there are no observations in this filter: return None
        if len(sel_obs) == 0:
            if output_q is not None:
                output_q.put({j: None})
            else:
                return None

        # Get the fluxes (from griddata reference)

        # xi = MJD-T0
        xi = sel_obs[self.mjdCol]-gen_par['daymax'][:, np.newaxis]

        # yi = redshift simulated values
        # requested to avoid interpolation problems near boundaries
        yi = np.round(gen_par['z'], 4)
        # yi = gen_par['z']

        # p = phases of LC points = xi/(1.+z)
        p = xi/(1.+yi[:, np.newaxis])
        yi_arr = np.ones_like(p)*yi[:, np.newaxis]

        if interpType == 'griddata':
            # Get reference values: phase, z, flux, fluxerr
            x = self.reference_lc.lc_ref[band]['phase']
            y = self.reference_lc.lc_ref[band]['z']
            z = self.reference_lc.lc_ref[band]['flux']
            zb = self.reference_lc.lc_ref[band]['fluxerr']

            # flux interpolation
            fluxes_obs = griddata((x, y), z, (p, yi_arr),
                                  method=method, fill_value=0.)

            # flux error interpolation
            fluxes_obs_err = griddata(
                (x, y), zb, (p, yi_arr), method=method, fill_value=0.)

            # Fisher components estimation

            dFlux = {}

            # loop on Fisher parameters
            for val in self.param_Fisher:
                # get the reference components
                z_c = self.reference_lc.lc_ref[band]['d'+val]
                # get Fisher components from interpolation
                dFlux[val] = griddata((x, y), z_c, (p, yi_arr),
                                      method=method, fill_value=0.)

        if interpType == 'regular':

            """
            # remove LC points outside the restframe phase range
            min_rf_phase = gen_par['min_rf_phase'][:, np.newaxis]
            max_rf_phase = gen_par['max_rf_phase'][:, np.newaxis]
            flag = (p >= min_rf_phase) & (p <= max_rf_phase)

            time_ref = time.time()
            p_mask = np.ma.array(p, mask=~flag)
            yi_mask = np.ma.array(yi_arr, mask=~flag)

            pts = (p_mask[~p.mask],yi_mask[~p.mask])
            """
            pts = (p, yi_arr)
            fluxes_obs = self.reference_lc.flux[band](pts)
            fluxes_obs_err = self.reference_lc.fluxerr_photo[band](pts)
            fluxes_model_err = self.reference_lc.fluxerr_model[band](pts)

            """
            print('***************************', band)
            print('phases', p)
            print('redshift', yi_arr)
            print('fluxes', fluxes_obs)
            print('flux errors', fluxes_obs_err)
            """

            # Fisher components estimation

            dFlux = {}

            # loop on Fisher parameters
            for val in self.param_Fisher:
                dFlux[val] = self.reference_lc.param[band][val](pts)
            # get the reference components
            # z_c = self.reference_lc.lc_ref[band]['d'+val]
            # get Fisher components from interpolation
            # dFlux[val] = griddata((x, y), z_c, (p, yi_arr),
            #                      method=method, fill_value=0.)

        # replace crazy fluxes by dummy values
        # fluxes_obs_err[fluxes_obs <= 0.] = 5.e-10
        # fluxes_obs[fluxes_obs <= 0.] = 1.e-10
        """
        print('***')
        print('fluxes', fluxes_obs)
        print('flux errors', fluxes_obs_err)
        """
        # Fisher matrix components estimation
        # loop on SN parameters (x0,x1,color)
        # estimate: dF/dxi*dF/dxj
        Derivative_for_Fisher = {}
        for ia, vala in enumerate(self.param_Fisher):
            for jb, valb in enumerate(self.param_Fisher):
                if jb >= ia:
                    Derivative_for_Fisher[vala +
                                          valb] = dFlux[vala] * dFlux[valb]

        # remove LC points outside the restframe phase range
        min_rf_phase = gen_par['minRFphase'][:, np.newaxis]
        max_rf_phase = gen_par['maxRFphase'][:, np.newaxis]
        flag = (p >= min_rf_phase) & (p <= max_rf_phase)
        # remove LC points outside the (blue-red) range

        mean_restframe_wavelength = np.array(
            [self.mean_wavelength[band]]*len(sel_obs))
        mean_restframe_wavelength = np.tile(
            mean_restframe_wavelength, (len(gen_par), 1))/(1.+gen_par['z'][:, np.newaxis])
        # flag &= (mean_restframe_wavelength > 0.) & (
        #    mean_restframe_wavelength < 1000000.)
        flag &= (mean_restframe_wavelength > self.blue_cutoff) & (
            mean_restframe_wavelength < self.red_cutoff)

        flag &= fluxes_obs > 1.e-10
        flag_idx = np.argwhere(flag)

        # Correct fluxes_err (m5 in generation probably different from m5 obs)

        # gamma_obs = self.telescope.gamma(
        #    sel_obs[self.m5Col], [band]*len(sel_obs), sel_obs[self.exptimeCol])

        gamma_obs = self.reference_lc.gamma[band](
            (sel_obs[self.m5Col], sel_obs[self.exptimeCol]/sel_obs[self.nexpCol], sel_obs[self.nexpCol]))

        mag_obs = -2.5*np.log10(fluxes_obs/3631.)

        m5 = np.asarray([self.reference_lc.m5_ref[band]]*len(sel_obs))

        gammaref = np.asarray([self.reference_lc.gamma_ref[band]]*len(sel_obs))

        m5_tile = np.tile(m5, (len(p), 1))

        srand_ref = srand(
            np.tile(gammaref, (len(p), 1)), mag_obs, m5_tile)

        srand_obs = srand(np.tile(gamma_obs, (len(p), 1)), mag_obs, np.tile(
            sel_obs[self.m5Col], (len(p), 1)))

        correct_m5 = srand_ref/srand_obs

        """
        print(band, gammaref, gamma_obs, m5,
              sel_obs[self.m5Col], sel_obs[self.exptimeCol])
        """

        fluxes_obs_err = fluxes_obs_err/correct_m5

        # now apply the flag to select LC points
        fluxes = np.ma.array(fluxes_obs, mask=~flag)
        fluxes_err_photo = np.ma.array(fluxes_obs_err, mask=~flag)
        fluxes_err_model = np.ma.array(fluxes_model_err, mask=~flag)
        phases = np.ma.array(p, mask=~flag)
        snr_m5 = np.ma.array(fluxes_obs/fluxes_obs_err, mask=~flag)

        nvals = len(phases)

        obs_time = np.ma.array(
            np.tile(sel_obs[self.mjdCol], (nvals, 1)), mask=~flag)
        seasons = np.ma.array(
            np.tile(sel_obs[self.seasonCol], (nvals, 1)), mask=~flag)
        if not self.lightOutput:
            gammas = np.ma.array(
                np.tile(gamma_obs, (nvals, 1)), mask=~flag)
            exp_time = np.ma.array(
                np.tile(sel_obs[self.exptimeCol], (nvals, 1)), mask=~flag)
            nexposures = np.ma.array(
                np.tile(sel_obs[self.nexpCol], (nvals, 1)), mask=~flag)
            m5_obs = np.ma.array(
                np.tile(sel_obs[self.m5Col], (nvals, 1)), mask=~flag)
            if self.seeingCol in sel_obs.dtype.names:
                seeings = np.ma.array(
                    np.tile(sel_obs[self.seeingCol], (nvals, 1)), mask=~flag)

        # add night value here
        if 'night' in sel_obs.dtype.names:
            nights = np.ma.array(
                np.tile(sel_obs['night'], (nvals, 1)), mask=~flag)

        healpixIds = np.ma.array(
            np.tile(sel_obs['healpixID'].astype(int), (nvals, 1)), mask=~flag)

        pixRAs = np.ma.array(
            np.tile(sel_obs['pixRA'], (nvals, 1)), mask=~flag)

        pixDecs = np.ma.array(
            np.tile(sel_obs['pixDec'], (nvals, 1)), mask=~flag)

        z_vals = gen_par['z'][flag_idx[:, 0]]
        daymax_vals = gen_par['daymax'][flag_idx[:, 0]]
        mag_obs = np.ma.array(mag_obs, mask=~flag)
        Fisher_Mat = {}
        for key, vals in Derivative_for_Fisher.items():
            Fisher_Mat[key] = np.ma.array(vals, mask=~flag)

        # Store in a panda dataframe
        lc = pd.DataFrame()

        ndata = len(fluxes_err_photo[~fluxes_err_photo.mask])

        if ndata > 0:

            lc['flux'] = fluxes[~fluxes.mask]
            lc['fluxerr_photo'] = fluxes_err_photo[~fluxes_err_photo.mask]
            lc['fluxerr_model'] = fluxes_err_model[~fluxes_err_model.mask]
            lc['fluxerr'] = np.sqrt(
                lc['fluxerr_photo']**2+lc['fluxerr_model']**2)
            lc['snr'] = lc['flux']/lc['fluxerr_photo']
            lc['phase'] = phases[~phases.mask]
            lc['snr_m5'] = snr_m5[~snr_m5.mask]
            lc['time'] = obs_time[~obs_time.mask]
            lc['mag'] = mag_obs[~mag_obs.mask]
            if not self.lightOutput:
                lc['gamma'] = gammas[~gammas.mask]
                lc['m5'] = m5_obs[~m5_obs.mask]
                lc['mag'] = mag_obs[~mag_obs.mask]
                lc['magerr'] = (2.5/np.log(10.))/snr_m5[~snr_m5.mask]
                lc['time'] = obs_time[~obs_time.mask]
                lc[self.exptimeCol] = exp_time[~exp_time.mask]
                lc[self.nexpCol] = nexposures[~nexposures.mask]
                if self.seeingCol in sel_obs.dtype.names:
                    lc[self.seeingCol] = seeings[~seeings.mask]

            if 'night' in sel_obs.dtype.names:
                lc['night'] = nights[~nights.mask]
            lc['band'] = ['LSST::'+band]*len(lc)
            lc['zp'] = self.zp[band]
            lc['zp'] = 2.5*np.log10(3631)
            lc['zpsys'] = 'ab'
            lc['season'] = seasons[~seasons.mask]
            lc['season'] = lc['season'].astype(int)
            lc['healpixID'] = healpixIds[~healpixIds.mask]
            lc['pixRA'] = pixRAs[~pixRAs.mask]
            lc['pixDec'] = pixDecs[~pixDecs.mask]
            lc['z'] = z_vals
            lc['daymax'] = daymax_vals

            if not self.lightOutput:
                lc['flux_e_sec'] = self.reference_lc.mag_to_flux[band]((
                    lc['mag'], lc[self.exptimeCol]/lc[self.nexpCol], lc[self.nexpCol]))
                lc['flux_5'] = self.reference_lc.mag_to_flux[band]((
                    lc['m5'], lc[self.exptimeCol]/lc[self.nexpCol], lc[self.nexpCol]))
                lc['flux_5_back'] = 10**(-0.4*(lc['m5']-self.zp[band]))
                lc['snr_back'] = lc['flux_e_sec'] / \
                    np.sqrt((lc['flux_5_back']/5.)**2 +
                            lc['flux_e_sec']/lc[self.exptimeCol])
                lc.loc[:, 'ratio'] = lc['snr_back']/lc['snr_m5']
                lc['fluxerr_photob'] = lc['flux']/lc['snr_back']
                # lc['fluxerr_photo'] = lc['fluxerr_photob']
                # lc['snr_m5'] = lc['snr_back']
            for key, vals in Fisher_Mat.items():
                lc.loc[:, 'F_{}'.format(
                    key)] = vals[~vals.mask]/(lc['fluxerr_photo'].values**2)
                # lc.loc[:, 'F_{}'.format(key)] = 999.
            lc.loc[:, 'x1'] = self.x1
            lc.loc[:, 'color'] = self.color

            lc.loc[:, 'n_aft'] = (np.sign(lc['phase']) == 1) & (
                lc['snr_m5'] >= self.snr_min)
            lc.loc[:, 'n_bef'] = (np.sign(lc['phase'])
                                  == -1) & (lc['snr_m5'] >= self.snr_min)

            lc.loc[:, 'n_phmin'] = (lc['phase'] <= -5.)
            lc.loc[:, 'n_phmax'] = (lc['phase'] >= 20)

            # transform boolean to int because of some problems in the sum()

            for colname in ['n_aft', 'n_bef', 'n_phmin', 'n_phmax']:
                lc.loc[:, colname] = lc[colname].astype(int)

        if len(lc) > 0.:
            lc = self.dust_corrections(lc, ebvofMW)

        if output_q is not None:
            output_q.put({j: lc})
        else:
            return lc

    def dust_corrections(self, tab, ebvofMW):
        """
        Method to apply dust corrections on flux and related data

        Parameters
        ---------------
        tab: pandas df
          LC points to apply dust corrections on
        ebvofMW: float
          E(B-V) for MW

        Returns
        -----------
        tab: pandas df
          LC points with dust corrections applied
        """

        # no dust correction here
        if np.abs(ebvofMW) < 1.e-5:
            return tab

        tab['ebvofMW'] = ebvofMW

        for vv in ['F_x0x0', 'F_x0x1', 'F_x0daymax', 'F_x0color', 'F_x1x1',
                   'F_x1daymax', 'F_x1color', 'F_daymaxdaymax', 'F_daymaxcolor',
                   'F_colorcolor']:
            tab[vv] *= tab['fluxerr']**2

        # test = pd.DataFrame(tab)

        tab = tab.groupby(['band']).apply(
            lambda x: self.corrFlux(x)).reset_index()

        # mag correction - after flux correction
        # print('there man',tab['flux'])
        tab = tab.replace({'flux': 0.0}, 1.e-10)
        tab['mag'] = -2.5 * np.log10(tab['flux'] / 3631.0)
        # snr_m5 correction
        tab['snr_m5'] = 1./srand(tab['gamma'], tab['mag'], tab['m5'])
        tab['magerr'] = (2.5/np.log(10.))/tab['snr_m5']
        tab['fluxerr_photo'] = tab['flux']/tab['snr_m5']
        tab['fluxerr'] = np.sqrt(tab['fluxerr_photo']**2 +
                                 tab['fluxerr_model']**2)

        # tab['old_flux'] = test['flux']
        # tab['old_fluxerr'] = test['fluxerr']

        # print(toat)

        for vv in ['F_x0x0', 'F_x0x1', 'F_x0daymax', 'F_x0color', 'F_x1x1',
                   'F_x1daymax', 'F_x1color', 'F_daymaxdaymax', 'F_daymaxcolor',
                   'F_colorcolor']:
            tab[vv] /= tab['fluxerr']**2

        return tab

    def corrFlux(self, grp):
        """
        Method to correct flux and Fisher matrix elements for dust

        Parameters
        ---------------
        grp: pandas group
           data to process

        Returns
        ----------
        pandas grp with corrected values
        """

        band = grp.name.split(':')[-1]

        corrdust = self.dustcorr[band]['ratio_flux'](
            (grp['phase'], grp['z'], grp['ebvofMW']))
        for vv in ['flux', 'flux_e_sec']:
            grp[vv] *= corrdust

        corrdust = self.dustcorr[band]['ratio_fluxerr_model'](
            (grp['phase'], grp['z'], grp['ebvofMW']))
        grp['fluxerr_model'] *= corrdust

        for va in ['x0', 'x1', 'color', 'daymax']:
            for vb in ['x0', 'x1', 'color', 'daymax']:
                corrdusta = self.dustcorr[band]['ratio_d{}'.format(va)](
                    (grp['phase'], grp['z'], grp['ebvofMW']))
                corrdustb = self.dustcorr[band]['ratio_d{}'.format(vb)](
                    (grp['phase'], grp['z'], grp['ebvofMW']))
                varcorr = 'F_{}{}'.format(va, vb)
                if varcorr in grp.columns:
                    grp[varcorr] *= corrdusta*corrdustb

        return grp


def srand(gamma, mag, m5):
    """
    Function to estimate :math:`srand=\sqrt((0.04-\gamma)*x+\gamma*x^2)`

    with :math:`x = 10^{0.4*(m-m_5)}`


    Parameters
    ---------------
    gamma: float
     gamma value
    mag: float
     magnitude
    m5: float
      fiveSigmaDepth value

    Returns
    ----------
    srand = np.sqrt((0.04-gamma)*x+gamma*x**2)
    with x = 10**(0.4*(mag-m5))

    """

    x = 10**(0.4*(mag-m5))
    return np.sqrt((0.04-gamma)*x+gamma*x**2)


class CalcSN:
    """
    class to estimate SN parameters from light curve

    Parameters
    ---------------
    lc_all: astropy Table
      light curve points
    nBef: int, opt
      quality selection: number of LC points before max (default: 2)
    nAft: int, opt
       quality selection: number of LC points after max (default: 5)
    nPhamin: int, opt
       quality selection: number of point with phase <= -5(default: 1)
    nPhamax: int, opt
    quality selection: number of point with phase >= 30 (default: 1)

    params: list(str)
      list of Fisher parameters to estimate (default: ['x0', 'x1', 'color'])

    """

    def __init__(self, lc_all,
                 nBef=2, nAft=5,
                 nPhamin=1, nPhamax=1,
                 params=['x0', 'x1', 'color']):

        self.nBef = nBef
        self.nAft = nAft
        self.nPhamin = nPhamin
        self.nPhamax = nPhamax
        self.params = params
        # select only fields involved in the calculation
        # this would save some memory
        fields = []
        for fi in ['season', 'healpixID', 'pixRA', 'pixDec', 'z',
                   'daymax', 'band', 'snr_m5', 'time', 'fluxerr', 'fluxerr_photo', 'phase']:
            fields.append(fi)

        # Fisher parameters
        for ia, vala in enumerate(self.params):
            for jb, valb in enumerate(self.params):
                if jb >= ia:
                    fields.append('F_'+vala+valb)

        # lc to process
        lc = Table(lc_all[fields])

        # lc['fluxerr'] = lc['fluxerr_photo']
        # LC selection

        goodlc, badlc = self.selectLC(lc)

        res = badlc
        self.fitstatus = 0
        if len(goodlc) > 0:
            self.calcSigma(lc, goodlc)
            res = vstack([res, goodlc])

        self.sn = res

    def calcSigma(self, lc, restab_good):
        """
        Method to estimate sigma of SN paremeters

        Parameters
        ---------------
        lc: astropy Table
          light curve points
        restab_good:

        """

        valu = np.unique(restab_good['z', 'daymax'])
        diff = lc['daymax']-valu['daymax'][:, np.newaxis]
        flag = np.abs(diff) < 1.e-5
        diffb = lc['z']-valu['z'][:, np.newaxis]
        flag &= np.abs(diffb) < 1.e-5
        mystr = []
        for ia, vala in enumerate(self.params):
            for jb, valb in enumerate(self.params):
                if jb >= ia:
                    mystr.append('F_'+vala+valb)
        mystr += ['fluxerr']

        resu = np.ma.array(
            np.tile(np.copy(lc[mystr]), (len(valu), 1)), mask=~flag)

        # grab errors on "good" LC only ie the ones that passed the cuts
        self.sigmaSNparams(resu, restab_good)

    def selectLC(self, lc):
        """
        Method to select LC

        Parameters
        ---------------
        lc: astropy Table
          light curve points

        Returns
        ----------
        Two astropy Table: one with LC that passed the selection cuts
        and one that did not pass the selection cuts.
        For this last sample, covariances of SN parameters are set to 100.
        """
        restab = Table(
            np.unique(lc[['season', 'healpixID', 'pixRA', 'pixDec', 'z', 'daymax']]))

        valu = np.unique(lc['z', 'daymax'])
        diff = lc['daymax']-valu['daymax'][:, np.newaxis]
        flag = np.abs(diff) < 1.e-5
        diffb = lc['z']-valu['z'][:, np.newaxis]
        flag &= np.abs(diffb) < 1.e-5

        tile_band = np.tile(lc['band'], (len(valu), 1))
        tile_snr = np.tile(lc['snr_m5'], (len(valu), 1))

        flag_snr = tile_snr > 0.

        # difftime = lc['time']-lc['daymax']
        phase = np.tile(lc['phase'], (len(valu), 1))

        count_bef = self.select_phase(phase, flag, flag_snr, operator.lt, 0.)
        restab.add_column(Column(count_bef, name='n_bef'))

        count_aft = self.select_phase(phase, flag, flag_snr, operator.ge, 0.)
        restab.add_column(Column(count_aft, name='n_aft'))

        count_pmin = self.select_phase(phase, flag, True, operator.le, -5.)
        restab.add_column(Column(count_pmin, name='n_phmin'))

        count_pmax = self.select_phase(phase, flag, True, operator.ge, 30.)
        restab.add_column(Column(count_pmax, name='n_phmax'))

        # LC selection here
        idx = (restab['n_bef'] >= self.nBef) & (restab['n_aft'] >= self.nAft)
        idx &= (restab['n_phmin'] >= self.nPhamin) & (
            restab['n_phmax'] >= self.nPhamax)
        restab_good = Table(restab[idx])
        restab_bad = Table(restab[~idx])
        for par in self.params:
            restab_bad.add_column(
                Column([100.]*len(restab_bad), name='Cov_{}{}'.format(par, par)))

        return restab_good, restab_bad

    def select_phase(self, phase, flag, flag_snr, op, phase_cut):
        """
        Method to estimate the number of points depending on a phase cut

        Parameters
        ----------------
        phase:  array of phases
        flag: array of flag selection
        flag_snr: array of flags for snr selection
        op: operator to apply
        phase_cut: float
          phase selection

        Returns
        ----------
        number of points corresponding to the selection

        """
        fflag = op(phase, phase_cut)
        fflag &= flag
        fflag &= flag_snr

        ma_diff = np.ma.array(phase, mask=~fflag)
        count = ma_diff.count(axis=1)
        return count

    def sigmaSNparams(self, resu, restab):
        """
        Method to estimate SN parameter errors from Fisher elements

        Parameters
        ---------------
        resu: masked array
          light curve points used to estimate sigmas
        restab: astropy Table

        Returns
        -----------
        None. astropy Table of results (restab) is updated
        """

        time_ref = time.time()
        parts = {}
        for ia, vala in enumerate(self.params):
            for jb, valb in enumerate(self.params):
                if jb >= ia:
                    parts[ia, jb] = np.sum(resu['F_'+vala+valb], axis=1)

        # print('one',time.time()-time_ref,parts)
        time_ref = time.time()
        size = len(resu)
        npars = len(self.params)
        Fisher_Big = np.zeros((npars*size, npars*size))
        Big_Diag = np.zeros((npars*size, npars*size))
        Big_Diag = []

        for iv in range(size):
            Fisher_Matrix = np.zeros((npars, npars))
            for ia, vala in enumerate(self.params):
                for jb, valb in enumerate(self.params):
                    if jb >= ia:
                        Fisher_Big[ia+npars*iv][jb+npars *
                                                iv] = parts[ia, jb][iv]
        # print('two',time.time()-time_ref)
        time_ref = time.time()
        # pprint.pprint(Fisher_Big)

        Fisher_Big = Fisher_Big + np.triu(Fisher_Big, 1).T
        # Big_Diag = np.diag(np.linalg.inv(Fisher_Big))
        detmat = np.linalg.det(Fisher_Big)
        if detmat > 1.e-5:
            # matrix not singular - should be invertible
            Big_Diag = np.diag(faster_inverse(Fisher_Big))
            # print('three',time.time()-time_ref)
            time_ref = time.time()
            for ia, vala in enumerate(self.params):
                indices = range(ia, len(Big_Diag), npars)
                restab.add_column(
                    Column(np.take(Big_Diag, indices), name='Cov_{}{}'.format(vala, vala)))
            self.fitstatus = 1
        else:
            for ia, vala in enumerate(self.params):
                restab.add_column(
                    Column(-99., name='Cov_{}{}'.format(vala, vala)))
            self.fitstatus = -1
    # print('four',time.time()-time_ref)
    # time_ref = time.time()


class CalcSN_df:
    """
    class to estimate SN parameters from light curve

    Parameters
    ---------------
    lc_all: astropy Table
      light curve points
    n_bef: int, opt
      quality selection: number of LC points before max (default: 4)
    n_aft: int, opt
       quality selection: number of LC points after max (default: 10)
    snr_min: float, opt
       quality selection: min SNR for LC points (default: 5.)
    phase_min: float, opt
        phase corresponding to n_phase_min cut (default: -5)
    phase_max: float, opt
        phase corresponding to n_phase_max cut (default: 20)
    n_phase_min: int, opt
       quality selection: number of point with phase <= -5(default: 1)
    n_phase_max: int, opt
    quality selection: number of point with phase >= 30 (default: 1)
    params: list(str)
      list of Fisher parameters to estimate (default: ['x0', 'x1', 'daymax','color'])
    invert_matrix: bool, opt
      if True, SN parameter variances are estimated by inverting the Fisher matrix
      if False, only color variance is estimated from analytic estimation.
    """

    def __init__(self, lc,
                 n_bef=4, n_aft=10, snr_min=5.,
                 phase_min=-5, phase_max=20,
                 n_phase_min=1, n_phase_max=1,
                 params=['x0', 'x1', 'daymax', 'color'],
                 invert_matrix=False):

        self.n_bef = n_bef
        self.n_aft = n_aft
        self.snr_min = snr_min
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.n_phase_min = n_phase_min
        self.n_phase_max = n_phase_max
        self.params = params
        self.invert_matrix = invert_matrix
        # list of variables in the output df
        self.names_out = ['x1', 'color', 'z', 'daymax', 'season',
                          'healpixID', 'pixRA', 'pixDec']

        # select only fields involved in the calculation
        # this would save some memory
        fields = []
        for fi in ['season', 'healpixID', 'pixRA', 'pixDec', 'z',
                   'daymax', 'snr_m5', 'phase', 'x1', 'color']:
            fields.append(fi)

        tosum = []
        for ia, vala in enumerate(params):
            for jb, valb in enumerate(params):
                if jb >= ia:
                    fields.append('F_'+vala+valb)
                    tosum.append('F_'+vala+valb)
        time_ref = time.time()
        lc.loc[:] = lc[fields]

        # prepare LC for selection
        lc = self.statLC(lc)

        for colname in ['n_aft', 'n_bef', 'n_phmin', 'n_phmax']:
            lc.loc[:, colname] = lc[colname].astype(int)

        time_ref = time.time()

        # This is the final LC result
        sndf = lc.groupby(self.names_out).apply(
            lambda x: self.sigmaColor(x, tosum)).reset_index()

        self.sn = sndf

    def sigmaColor(self, grp, tosam):
        """
        Method to estimate sigmaColor for a group

        Parameters
        ---------------
        grp : df group
          data to process
        tosam: list(str)
           list of var to sum-up

        Returns
        -----------
        pandas df with the following cols:
        n_bef: number of LC point before max
        n_aft: number of LC points after max
        n_phmin: number of LC points with phase<=phase_min
        n_phmax: number of LC points with phase>= phase_max
        Cov_colorcolor: color variance

        """

        tosuma = list(tosam)
        tosuma += ['n_aft', 'n_bef', 'n_phmin', 'n_phmax']
        # sums = grp[tosuma].sum().to_frame()
        sums = pd.DataFrame([grp[tosuma].sum()], columns=tosuma)

        # selection cuts
        idx = sums['n_aft'] >= self.n_aft
        idx &= sums['n_bef'] >= self.n_bef
        idx &= sums['n_phmin'] >= self.n_phase_min
        idx &= sums['n_phmax'] >= self.n_phase_max

        # estimate sigma_color**2 for grp passing the cuts only.
        dict_out = {}
        for var in self.params:
            dict_out['Cov_{}{}'.format(var, var)] = [100.]

        if len(sums[idx]) > 0:
            if not self.invert_matrix:
                dict_out['Cov_colorcolor'] = CovColor(
                    sums).Cov_colorcolor.to_list()
            else:
                covCalc = self.sigmaSNparams(sums)
                for key in dict_out.keys():
                    dict_out[key] = covCalc[key].to_list()

        for vv in ['n_aft', 'n_bef', 'n_phmin', 'n_phmax']:
            dict_out[vv] = sums[vv].to_list()

        # return results as a df
        return pd.DataFrame.from_dict(dict_out)

    def statLC(self, lc):
        """
        Method to add var to lc so as to ease future selection

        Parameters
        ---------------
        lc: pandas df
          data to process

        Returns
        -----------
        initial df with added cols:
        n_bef: number of LC point before max
        n_aft: number of LC points after max
        n_phmin: number of LC points with phase<=phase_min
        n_phmax: number of LC points with phase>= phase_max

        """
        lc.loc[:, 'n_aft'] = (np.sign(lc['phase']) == 1) & (
            lc['snr_m5'] >= self.snr_min)
        lc.loc[:, 'n_bef'] = (np.sign(lc['phase']) == -
                              1) & (lc['snr_m5'] >= self.snr_min)
        # lc.loc[:, 'N_aft'] = (np.sign(lc['phase']) == 1)
        # lc.loc[:, 'N_bef'] = (np.sign(lc['phase']) == -1)
        # lc.loc[:,'N_phmin'] = (lc['phase']<=-5.)&(lc['snr_m5']>=5.)
        # lc.loc[:,'N_phmax'] = (lc['phase']>=20)&(lc['snr_m5']>=5.)
        lc.loc[:, 'n_phmin'] = (lc['phase'] <= self.phase_min)
        lc.loc[:, 'n_phmax'] = (lc['phase'] >= self.phase_max)

        return lc

    def sigmaSNparams(self, grp):
        """
        Method to estimate variances of SN parameters
        from inversion of the Fisher matrix

        Parameters
        ---------------
        grp: pandas df of flux derivatives wrt SN parameters

        Returns
        ----------
        Diagonal elements of the inverted matrix (as pandas df)

        """
        parts = {}
        for ia, vala in enumerate(self.params):
            for jb, valb in enumerate(self.params):
                if jb >= ia:
                    parts[ia, jb] = grp['F_'+vala+valb]

            # print(parts)
        size = len(grp)
        npar = len(self.params)
        Fisher_Big = np.zeros((npar*size, npar*size))
        Big_Diag = np.zeros((npar*size, npar*size))
        Big_Diag = []

        for iv in range(size):
            Fisher_Matrix = np.zeros((npar, npar))
            for ia, vala in enumerate(self.params):
                for jb, valb in enumerate(self.params):
                    if jb >= ia:
                        Fisher_Big[ia+npar*iv][jb+npar*iv] = parts[ia, jb]

        # pprint.pprint(Fisher_Big)

        Fisher_Big = Fisher_Big + np.triu(Fisher_Big, 1).T
        Big_Diag = np.diag(np.linalg.inv(Fisher_Big))

        res = pd.DataFrame()
        for ia, vala in enumerate(self.params):
            indices = range(ia, len(Big_Diag), npar)
            # restab.add_column(
            #    Column(np.take(Big_Diag, indices), name='Cov_{}{}'.format(vala,vala)))
            res['Cov_{}{}'.format(vala, vala)] = np.take(Big_Diag, indices)

        return res


class CovColor:
    """
    class to estimate CovColor from lc using Fisher matrix element

    Parameters
    ---------------
    lc: pandas df
    lc to process. Should contain the Fisher matrix components
    ie the sum of the derivative of the fluxes wrt SN parameters

    """

    def __init__(self, lc):

        self.Cov_colorcolor = self.varColor(lc)

    def varColor(self, lc):
        """
        Method to estimate the variance color from matrix element

        Parameters
        --------------
        lc: pandas df
          data to process containing the derivative of the flux with respect to SN parameters

        Returns
        ----------
        float: Cov_colorcolor

        """
        a1 = lc['F_x0x0']
        a2 = lc['F_x0x1']
        a3 = lc['F_x0daymax']
        a4 = lc['F_x0color']

        b1 = a2
        b2 = lc['F_x1x1']
        b3 = lc['F_x1daymax']
        b4 = lc['F_x1color']

        c1 = a3
        c2 = b3
        c3 = lc['F_daymaxdaymax']
        c4 = lc['F_daymaxcolor']

        d1 = a4
        d2 = b4
        d3 = c4
        d4 = lc['F_colorcolor']

        detM = a1*self.det(b2, b3, b4, c2, c3, c4, d2, d3, d4)
        detM -= b1*self.det(a2, a3, a4, c2, c3, c4, d2, d3, d4)
        detM += c1*self.det(a2, a3, a4, b2, b3, b4, d2, d3, d4)
        detM -= d1*self.det(a2, a3, a4, b2, b3, b4, c2, c3, c4)

        res = -a3*b2*c1+a2*b3*c1+a3*b1*c2-a1*b3*c2-a2*b1*c3+a1*b2*c3

        return res/detM

    def det(self, a1, a2, a3, b1, b2, b3, c1, c2, c3):
        """
        Method to estimate the det of a matrix from its values

        Parameters
        ---------------
        Values of the matrix
        ( a1 a2 a3)
        (b1 b2 b3)
        (c1 c2 c3)

        Returns
        -----------
        det value
        """
        resp = a1*b2*c3+b1*c2*a3+c1*a2*b3
        resm = a3*b2*c1+b3*c2*a1+c3*a2*b1

        return resp-resm


"""
def det(a1, a2, a3, b1, b2, b3, c1, c2, c3):

    resp = a1*b2*c3+b1*c2*a3+c1*a2*b3
    resm = a3*b2*c1+b3*c2*a1+c3*a2*b1

    return resp-resm


def covColor(lc):

    a1 = lc['F_x0x0']
    a2 = lc['F_x0x1']
    a3 = lc['F_x0daymax']
    a4 = lc['F_x0color']

    b1 = a2
    b2 = lc['F_x1x1']
    b3 = lc['F_x1daymax']
    b4 = lc['F_x1color']

    c1 = a3
    c2 = b3
    c3 = lc['F_daymaxdaymax']
    c4 = lc['F_daymaxcolor']

    d1 = a4
    d2 = b4
    d3 = c4
    d4 = lc['F_colorcolor']

    detM = a1*det(b2, b3, b4, c2, c3, c4, d2, d3, d4)
    detM -= b1*det(a2, a3, a4, c2, c3, c4, d2, d3, d4)
    detM += c1*det(a2, a3, a4, b2, b3, b4, d2, d3, d4)
    detM -= d1*det(a2, a3, a4, b2, b3, b4, c2, c3, c4)

    res = -a3*b2*c1+a2*b3*c1+a3*b1*c2-a1*b3*c2-a2*b1*c3+a1*b2*c3

    return res/detM

"""

"""
def calcSN_last(lc_all, params=['x0', 'x1', 'color'], j=-1, output_q=None):

    time_ref = time.time()

    # print('go man',j,len(lc_all))
    fields = []
    for fi in ['season', 'healpixID', 'pixRA', 'pixDec', 'z',
               'daymax', 'band', 'snr_m5', 'time', 'fluxerr', 'phase']:
        fields.append(fi)

    for ia, vala in enumerate(params):
        for jb, valb in enumerate(params):
            if jb >= ia:
                fields.append('F_'+vala+valb)

    lc = Table(lc_all[fields])

    # print('here',lc)

    restab = Table(
        np.unique(lc[['season', 'healpixID', 'pixRA', 'pixDec', 'z', 'daymax']]))

    valu = np.unique(lc['z', 'daymax'])
    diff = lc['daymax']-valu['daymax'][:, np.newaxis]
    flag = np.abs(diff) < 1.e-5
    diffb = lc['z']-valu['z'][:, np.newaxis]
    flag &= np.abs(diffb) < 1.e-5

    tile_band = np.tile(lc['band'], (len(valu), 1))
    tile_snr = np.tile(lc['snr_m5'], (len(valu), 1))

    flag_snr = tile_snr >= 5.

    # difftime = lc['time']-lc['daymax']
    phase = np.tile(lc['phase'], (len(valu), 1))

    fflag = phase < 0.
    fflag &= flag
    fflag &= flag_snr

    ma_diff = np.ma.array(phase, mask=~fflag)
    count = ma_diff.count(axis=1)
    restab.add_column(Column(count, name='N_bef'))

    fflag = phase >= 0.
    fflag &= flag
    fflag &= flag_snr
    ma_diff = np.ma.array(phase, mask=~fflag)
    count = ma_diff.count(axis=1)
    restab.add_column(Column(count, name='N_aft'))

    # print('calc',restab)

    # print('after selection',time.time()-time_ref)
    # Select LCs with a sufficient number of LC points before and after max

    idx = (restab['N_bef'] >= 2) & (restab['N_aft'] >= 5)
    restab_good = Table(restab[idx])
    restab_bad = Table(restab[~idx])
    for par in params:
        restab_bad.add_column(
            Column([100.]*len(restab_bad), name='Cov_{}{}'.format(par, par)))

    # print('here again',valu[['z','daymax']],len(restab_good),len(restab_bad))

    if len(restab_good) == 0:
        if output_q is not None:
            output_q.put({j: restab_bad})
        else:
            return restab_bad

    valu = np.unique(restab_good['z', 'daymax'])
    diff = lc['daymax']-valu['daymax'][:, np.newaxis]
    flag = np.abs(diff) < 1.e-5
    diffb = lc['z']-valu['z'][:, np.newaxis]
    flag &= np.abs(diffb) < 1.e-5
    mystr = []
    for ia, vala in enumerate(params):
        for jb, valb in enumerate(params):
            if jb >= ia:
                mystr.append('F_'+vala+valb)
    mystr += ['fluxerr']

    resu = np.ma.array(np.tile(np.copy(lc[mystr]), (len(valu), 1)), mask=~flag)

    # print('hello',valu[['z','daymax']],np.copy(resu))

    restab = restab_bad
    time_ref = time.time()

    if len(restab_good) > 0:

        sigmaSNparams(resu, restab_good, params=['x0', 'x1', 'color'])

        restab = vstack([restab, restab_good])

    # print('after sigma',time.time()-time_ref)

    if output_q is not None:
        output_q.put({j: restab})
    else:
        return restab
"""

"""
def npoints(gr):

    idx = gr['phase'] < 0.
    gr['N_bef'] = len(gr.loc[idx])
    gr['N_aft'] = len(gr)-len(gr.loc[idx])

    return gr
"""

"""
def npointBand(lc, band):

    valu = np.unique(lc['z', 'daymax'])
    diff = lc['daymax']-valu['daymax'][:, np.newaxis]
    flag = np.abs(diff) < 1.e-5
    diffb = lc['z']-valu['z'][:, np.newaxis]
    flag &= np.abs(diffb) < 1.e-5

    tile_snr = np.tile(lc['snr_m5'], (len(valu), 1))
    difftime = lc['time']-lc['daymax']
    tile_diff = np.tile(difftime, (len(valu), 1))

    flagp = tile_diff >= 0.
    flagn = tile_diff < 0.

    restab = Table()
    for key, fl in dict(zip(['aft', 'bef'], [flagp, flagn])).items():
        fflag = np.copy(flag)
        fflag &= fl
        ma_diff = np.ma.array(tile_diff, mask=~fflag)
        count = ma_diff.count(axis=1)
        if band != '':
            restab.add_column(Column(count, name='N_{}'.format(key, band)))
        else:
            restab.add_column(Column(count, name='N_{}'.format(key, band)))

    return restab
"""


def sigmaSNparams(resu, restab, params=['x0', 'x1', 'color']):
    """
    Method to estimate SN parameter errors from Fisher elements

    Parameters
    ---------------
    resu:
    restab:
    params: list(str)
      list of parameters to consider (default: ['x0', 'x1', 'color'])

    Returns
    -----------

    """

    print(type(resu), type(restab))

    time_ref = time.time()
    parts = {}
    for ia, vala in enumerate(params):
        for jb, valb in enumerate(params):
            if jb >= ia:
                # parts[ia, jb] = np.sum(
                #    resu['F_'+vala+valb]/(resu['fluxerr']**2.), axis=1)
                parts[ia, jb] = np.sum(resu['F_'+vala+valb], axis=1)

    # print('one',time.time()-time_ref,parts)
    time_ref = time.time()
    size = len(resu)
    Fisher_Big = np.zeros((3*size, 3*size))
    Big_Diag = np.zeros((3*size, 3*size))
    Big_Diag = []

    for iv in range(size):
        Fisher_Matrix = np.zeros((3, 3))
        for ia, vala in enumerate(params):
            for jb, valb in enumerate(params):
                if jb >= ia:
                    Fisher_Big[ia+3*iv][jb+3 *
                                        iv] = parts[ia, jb][iv]
    # print('two',time.time()-time_ref)
    time_ref = time.time()
    # pprint.pprint(Fisher_Big)

    Fisher_Big = Fisher_Big + np.triu(Fisher_Big, 1).T
    # Big_Diag = np.diag(np.linalg.inv(Fisher_Big))
    # print('hhhh',Fisher_Big.shape)
    Big_Diag = np.diag(faster_inverse(Fisher_Big))
    # print('three',time.time()-time_ref)
    time_ref = time.time()
    for ia, vala in enumerate(params):
        indices = range(ia, len(Big_Diag), 3)
        restab.add_column(
            Column(np.take(Big_Diag, indices), name='Cov_{}{}'.format(vala, vala)))

    # print('four',time.time()-time_ref)
    # time_ref = time.time()


def faster_inverse(A):
    """
    Method to invert a matrix in a fast way

    Parameters
    --------------
    A: matrix to invert

    Returns
    ----------
    A-1: inverse matrix

    """

    b = np.identity(A.shape[1], dtype=A.dtype)
    # u, piv, x, info = lapack.dgesv(A, b)
    return la.solve(A, b)


class GetReference:
    """
    Class to load reference data
    used for the fast SN simulator

    Parameters
    ----------------
    templateDir: str
      location dir of the reference LC files
    lcName: str
      name of the reference file to load (lc)
    gammaDir: str
       location dir where gamma files are located
    gammaName: str
      name of the reference file to load (gamma)
    web_path: str
      web adress where files (LC reference and gamma) can be found if not already on disk
    param_Fisher : list(str),opt
      list of SN parameter for Fisher estimation to consider
      (default: ['x0', 'x1', 'color', 'daymax'])

    Returns
    -----------
    The following dict can be accessed:

    mag_to_flux_e_sec : Interp1D of mag to flux(e.sec-1)  conversion
    flux : dict of RegularGridInterpolator of fluxes (key: filters, (x,y)=(phase, z), result=flux)
    fluxerr : dict of RegularGridInterpolator of flux errors (key: filters, (x,y)=(phase, z), result=fluxerr)
    param : dict of dict of RegularGridInterpolator of flux derivatives wrt SN parameters
                  (key: filters plus param_Fisher parameters; (x,y)=(phase, z), result=flux derivatives)
    gamma : dict of RegularGridInterpolator of gamma values (key: filters)


    """""

    def __init__(self, templateDir, lcName,
                 gammaDir, gammaName,
                 web_path, param_Fisher=['x0', 'x1', 'color', 'daymax']):

        check_get_file(web_path, templateDir, lcName)

        # Load the file - lc reference
        lcFullName = '{}/{}'.format(templateDir, lcName)
        f = h5py.File(lcFullName, 'r')
        keys = list(f.keys())
        # lc_ref_tot = Table.read(filename, path=keys[0])
        lc_ref_tot = Table.from_pandas(pd.read_hdf(lcFullName))
        lc_ref_tot.convert_bytestring_to_unicode()
        idx = lc_ref_tot['z'] > 0.005
        lc_ref_tot = np.copy(lc_ref_tot[idx])

        # telescope requested
        # Load the file - gamma values

        # fgamma = h5py.File(gammaName, 'r')
        gammas = LoadGamma('grizy', gammaDir, gammaName, web_path)
        self.gamma = gammas.gamma
        self.mag_to_flux = gammas.mag_to_flux
        self.zp = gammas.zp
        self.mean_wavelength = gammas.mean_wavelength

        # Load references needed for the following
        self.lc_ref = {}
        self.gamma_ref = {}
        # self.gamma = {}
        self.m5_ref = {}
        # self.mag_to_flux_e_sec = {}

        self.flux = {}
        self.fluxerr_photo = {}
        self.fluxerr_model = {}
        self.param = {}

        bands = np.unique(lc_ref_tot['band'])
        mag_range = np.arange(10., 38., 0.01)
        # exptimes = np.linspace(15.,30.,2)
        # exptimes = [15.,30.,60.,100.]

        # gammArray = self.loopGamma(bands, mag_range, exptimes,telescope)

        method = 'linear'

        # for each band: load data to be used for interpolation
        for band in bands:
            idx = lc_ref_tot['band'] == band
            lc_sel = Table(lc_ref_tot[idx])
            lc_sel['z'] = lc_sel['z'].data.round(decimals=2)
            lc_sel['phase'] = lc_sel['phase'].data.round(decimals=1)

            """
            select phases between -20 and -60 only

            """

            idx = lc_sel['phase'] < 60.
            idx &= lc_sel['phase'] > -20.
            lc_sel = lc_sel[idx]

            """
            for z in np.unique(lc_sel['z']):
                ig = lc_sel['z'] == z
                print(band,z,len(lc_sel[ig]))
            """
            """
            fluxes_e_sec = telescope.mag_to_flux_e_sec(
                mag_range, [band]*len(mag_range), [30]*len(mag_range))
            self.mag_to_flux_e_sec[band] = interpolate.interp1d(
                mag_range, fluxes_e_sec[:, 1], fill_value=0., bounds_error=False)
            """

            # these reference data will be used for griddata interp.
            self.lc_ref[band] = lc_sel
            #self.gamma_ref[band] = lc_sel['gamma'][0]
            #self.m5_ref[band] = np.unique(lc_sel['m5'])[0]

            # Fluxes and errors
            zmin, zmax, zstep, nz = self.limVals(lc_sel, 'z')
            phamin, phamax, phastep, npha = self.limVals(lc_sel, 'phase')

            zstep = np.round(zstep, 2)
            phastep = np.round(phastep, 1)

            zv = np.linspace(zmin, zmax, nz)
            phav = np.linspace(phamin, phamax, npha)

            index = np.lexsort((lc_sel['z'], lc_sel['phase']))
            flux = np.reshape(lc_sel[index]['flux'], (npha, nz))
            fluxerr_photo = np.reshape(
                lc_sel[index]['fluxerr_photo'], (npha, nz))
            fluxerr_model = np.reshape(
                lc_sel[index]['fluxerr_model'], (npha, nz))

            self.flux[band] = RegularGridInterpolator(
                (phav, zv), flux, method=method, bounds_error=False, fill_value=-1.0)
            self.fluxerr_photo[band] = RegularGridInterpolator(
                (phav, zv), fluxerr_photo, method=method, bounds_error=False, fill_value=-1.0)
            self.fluxerr_model[band] = RegularGridInterpolator(
                (phav, zv), fluxerr_model, method=method, bounds_error=False, fill_value=-1.0)

            """
            zref = 0.8
            if band == 'g':
                phases = [-5,12.]
                interp = self.flux[band]((phases,[zref,zref]))
                print('interp',interp)
                import matplotlib.pyplot as plt
                iu = np.abs(lc_sel['z']-zref)<1.e-8
                ll = lc_sel[iu]
                plt.plot(ll['phase'],ll['flux'],'ko',mfc='None')
                plt.plot(phases,interp,'r*')
                plt.show()
              """

            # Flux derivatives
            self.param[band] = {}
            for par in param_Fisher:
                valpar = np.reshape(
                    lc_sel[index]['d{}'.format(par)], (npha, nz))
                self.param[band][par] = RegularGridInterpolator(
                    (phav, zv), valpar, method=method, bounds_error=False, fill_value=0.)

            # gamma estimator

            """
            rec = Table.read(gammaName, path='gamma_{}'.format(band))

            rec['mag'] = rec['mag'].data.round(decimals=4)
            rec['exptime'] = rec['exptime'].data.round(decimals=4)

            magmin, magmax, magstep, nmag = self.limVals(rec, 'mag')
            expmin, expmax, expstep, nexp = self.limVals(rec, 'exptime')
            mag = np.linspace(magmin, magmax, nmag)
            exp = np.linspace(expmin, expmax, nexp)

            index = np.lexsort((np.round(rec['exptime'], 4), rec['mag']))
            gammab = np.reshape(rec[index]['gamma'], (nmag, nexp))
            self.gamma[band] = RegularGridInterpolator(
                (mag, exp), gammab, method=method, bounds_error=False, fill_value=0.)
            # print(band, gammab, mag, exp)
            """

    def limVals(self, lc, field):
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
        # vals = np.unique(lc[field].data.round(decimals=4))
        vals = np.unique(lc[field].data)
        vmin = np.min(vals)
        vmax = np.max(vals)
        vstep = np.median(vals[1:]-vals[:-1])

        # make a check here
        test = list(np.round(np.arange(vmin, vmax+vstep, vstep), 2))
        if len(test) != len(vals):
            print('problem here with ', field)
            print('missing value', set(test).difference(set(vals)))
            print('Interpolation results may not be accurate!!!!!')
        return vmin, vmax, vstep, len(vals)

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
                tab_tot = vstack([tab_tot, tab_b], metadata_conflicts='silent')
                """
                diff = tab_b['z']-zvals_arr[:, np.newaxis]
                # flag = np.abs(diff)<1.e-3
                flag_idx = np.where(np.abs(diff) < 1.e-3)
                if len(flag_idx[1]) > 0:
                    tab_tot = vstack([tab_tot, tab_b[flag_idx[1]]])
                """

            """
            print(flag,flag_idx[1])
            print('there man',tab_b[flag_idx[1]])
            mtile = np.tile(tab_b['z'],(len(zvals),1))
            # print('mtile',mtile*flag)
                
            masked_array = np.ma.array(mtile,mask=~flag)
            
            print('resu masked',masked_array,masked_array.shape)
            print('hhh',masked_array[~masked_array.mask])
            
            
        for val in zvals:
            print('hello',tab_b[['band','z','time']],'and',val)
            if np.abs(np.unique(tab_b['z'])-val)<0.01:
            # print('loading ref',np.unique(tab_b['z']))
            tab_tot=vstack([tab_tot,tab_b])
            break
            """
        if output_q is not None:
            output_q.put({j: tab_tot})
        else:
            return tab_tot

    def Read_Multiproc(self, tab):
        """
        Multiprocessing method to read references

        Parameters
        ---------------
        tab: astropy Table of data

        Returns
        -----------
        stacked astropy Table of data

        """
        # distrib=np.unique(tab['z'])
        nlc = len(tab)
        # n_multi=8
        if nlc >= 8:
            n_multi = min(nlc, 8)
            nvals = nlc/n_multi
            batch = range(0, nlc, nvals)
            batch = np.append(batch, nlc)
        else:
            batch = range(0, nlc)

        # lc_ref_tot={}
        # print('there pal',batch)
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


class LoadGamma:
    """
    class to load gamma and mag_to_flux values 
    and make regulargrid out of it

    Parameters
    ---------------
    bands: str
      bands to consider
    fDir: str
      location dir of the gamma file
    gammaName: str
      name of the file containing gamma values
    web_path: str
        web server where the file could be loaded from.
    telescope: Telescope
      instrument throughput
    """

    def __init__(self, bands, fDir, gammaName, web_path):

        self.gamma = {}
        self.mag_to_flux = {}
        self.zp = {}
        self.mean_wavelength = {}

        check_get_file(web_path, fDir, gammaName)

        gammaFullName = '{}/{}'.format(fDir, gammaName)
        """
        if not os.path.exists(gammaFullName):
            self.generateGamma(bands, telescope, gammaFullName)
        """

        for band in bands:
            rec = Table.read(gammaFullName,
                             path='gamma_{}'.format(band))
            self.zp = rec.meta['zp']
            self.mean_wavelength = rec.meta['mean_wavelength']

            rec['mag'] = rec['mag'].data.round(decimals=4)
            rec['single_exptime'] = rec['single_exptime'].data.round(
                decimals=4)

            magmin, magmax, magstep, nmag = limVals(rec, 'mag')
            expmin, expmax, expstep, nexpo = limVals(rec, 'single_exptime')
            nexpmin, nexpmax, nexpstep, nnexp = limVals(rec, 'nexp')
            mag = np.linspace(magmin, magmax, nmag)
            exp = np.linspace(expmin, expmax, nexpo)
            nexp = np.linspace(nexpmin, nexpmax, nnexp)

            index = np.lexsort(
                (rec['nexp'], np.round(rec['single_exptime'], 4), rec['mag']))
            gammab = np.reshape(rec[index]['gamma'], (nmag, nexpo, nnexp))
            fluxb = np.reshape(rec[index]['flux_e_sec'], (nmag, nexpo, nnexp))
            self.gamma[band] = RegularGridInterpolator(
                (mag, exp, nexp), gammab, method='linear', bounds_error=False, fill_value=0.)
            self.mag_to_flux[band] = RegularGridInterpolator(
                (mag, exp, nexp), fluxb, method='linear', bounds_error=False, fill_value=0.)

    def generateGammas(self, bands, telescope, outName):
        """
        Method to generate gamma file (if does not exist)

        Parameters
        ---------------
        bands: str
          bands to consider
        telescope: Telescope
           instrument
        outName: str
           output file name

        """
        print('gamma file {} does not exist')
        print('will generate it - few minutes')
        mag_range = np.arange(13., 38., 0.05)
        nexps = range(1, 500, 1)
        single_exposure_time = [15., 30.]
        Gamma(bands, telescope, outName,
              mag_range=mag_range,
              single_exposure_time=single_exposure_time, nexps=nexps)

        print('end of gamma estimation')


class LoadDust:

    """
    class to load dust correction file
    and make regulargrid out of it

    Parameters
    ---------------
    fDir: str
      location directory of the file
    fName: str
      name of the file containing dust correction values
    web_path: str
      web server adress where the file could be retrieved
    bands: str, opt
       bands to consider (default: 'grizy')
    """

    def __init__(self, fDir, fName, web_path, bands='grizy'):

        # check whether the file is available
        # if not grab it from the web server
        check_get_file(web_path, fDir, fName)

        self.dustcorr = {}

        tab = Table.read('{}/{}'.format(fDir, fName), path='dust')
        tab['z'] = tab['z'].round(2)
        tab['ebvofMW'] = tab['ebvofMW'].round(2)
        tab['phase'] = tab['phase'].round(1)
        tab.convert_bytestring_to_unicode()

        for b in bands:
            idx = tab['band'] == b
            rec = tab[idx]

            phasemin, phasemax, phasestep, nphase = limVals(rec, 'phase')
            zmin, zmax, zstep, nz = limVals(rec, 'z')
            ebvofMWmin, ebvofMWmax, ebvofMWstep, nebvofMW = limVals(
                rec, 'ebvofMW')

            phase = np.linspace(phasemin, phasemax, nphase)
            z = np.linspace(zmin, zmax, nz)
            ebvofMW = np.linspace(ebvofMWmin, ebvofMWmax, nebvofMW)

            index = np.lexsort(
                (rec['ebvofMW'], rec['z'], rec['phase']))

            self.dustcorr[b] = {}
            for vv in ['flux', 'dx0', 'dx1', 'dcolor', 'ddaymax', 'fluxerr_model']:
                ratio = np.reshape(
                    rec[index]['ratio_{}'.format(vv)], (nphase, nz, nebvofMW))
                self.dustcorr[b]['ratio_{}'.format(vv)] = RegularGridInterpolator(
                    (phase, z, ebvofMW), ratio, method='linear', bounds_error=False, fill_value=0.)

    def complete_missing(self, grp, zvals):
        """
        Method to complete a grp if missing values

        Parameters
        ----------------
        grp: pandas df group
        zvals: reference zvals

        Returns
        -----------
        pandas df with completed (0) values

        """

        if len(grp) != len(zvals):
            print(len(grp), grp.columns)

        return pd.DataFrame({'test': [0]})


def limVals(lc, field):
    """ Get unique values of a field in  a table

    Parameters
    --------------
    lc: Table
     astropy Table (here probably a LC)
    field: str
     name of the field of interest

    Returns
    -----------
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
    vals = np.unique(lc[field].data)
    # vals = np.unique(lc[field].data.round(decimals=4))
    # print(vals)
    vmin = np.min(vals)
    vmax = np.max(vals)
    vstep = np.median(vals[1:]-vals[:-1])

    return vmin, vmax, vstep, len(vals)
