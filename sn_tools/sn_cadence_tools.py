import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
import h5py
from astropy.table import Table, Column, vstack
from scipy.interpolate import griddata, interpn, CloughTocher2DInterpolator, LinearNDInterpolator
from sn_tools.sn_telescope import Telescope

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
        magnitude (m5) to flux (e/sec) interpolator

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
    """ Class to generate Fake observations

    Parameters
    -------------
    config: dict
      dict of parameters

    list : str,opt
        Name of the columns for data generation.
        Default : 'observationStartMJD', 'fieldRA', 'fieldDec','filter','fiveSigmaDepth','visitExposureTime','numExposures','visitTime','season'

    """

    def __init__(self, config,
                 mjdCol='observationStartMJD', RaCol='fieldRA',
                 DecCol='fieldDec', filterCol='filter', m5Col='fiveSigmaDepth',
                 exptimeCol='visitExposureTime', nexpCol='numExposures',
                 seasonCol='season', seeingEffCol='seeingFwhmEff', seeingGeomCol='seeingFwhmGeom',
                 visitTime='visitTime',
                 sequences=False):

        # config = yaml.load(open(config_filename))
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = seasonCol
        self.nexpCol = nexpCol
        self.seeingEffCol = seeingEffCol
        self.seeingGeomCol = seeingGeomCol
        self.visitTime = visitTime
        # now make fake obs
        if not sequences:
            self.makeFake(config)
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
        MJD, Ra, Dec, band,m5,Nexp, ExpTime, Season
        accessible through self.Observations

        """

        bands = config['bands']
        cadence = config['cadence']
        shift_days = dict(
            zip(bands, [config['shift_days']*io for io in range(len(bands))]))
        m5 = dict(zip(bands, config['m5']))
        #Nvisits = dict(zip(bands, config['Nvisits']))
        #Single_Exposure_Time = dict(zip(bands, config['Single_Exposure_Time']))
        inter_season_gap = 100.
        seeingEff = dict(zip(bands, config['seeingEff']))
        seeingGeom = dict(zip(bands, config['seeingGeom']))
        Ra = config['Ra']
        Dec = config['Dec']
        rtot = []
        # for season in range(1, config['nseasons']+1):
        Nvisits = {}
        Single_Exposure_Time = {}
        for il, season in enumerate(config['seasons']):
            #mjd_min = config['MJD_min'] + float(season-1)*inter_season_gap
            mjd_min = config['MJD_min']+il*(config['season_length'][season]+inter_season_gap)
            mjd_max = mjd_min+config['season_length'][season]
            n_visits = config['Nvisits'][season]
            seqs = config['sequences'][season]
            sing_exp_time = config['Single_Exposure_Time'][season]
            cadence = config['cadence'][season]
            
            for i in range(len(seqs)):
            #for i,val in enumerate(config['sequences']):
                mjd_min_seq = mjd_min+i*config['deltaT_seq'][i]
                mjd_max_seq = mjd_max+i*config['deltaT_seq'][i]
                mjd = np.arange(mjd_min_seq, mjd_max_seq+cadence,cadence)
                night = (mjd-config['MJD_min']).astype(int)+1
    
                for j,band in enumerate(seqs[i]):
                    mjd += shift_days[band]
                    Nvisits[band] = n_visits[i][j]
                    Single_Exposure_Time[band] = sing_exp_time[i][j]
                    m5_coadded = self.m5coadd(m5[band],
                                                  Nvisits[band],
                                                  Single_Exposure_Time[band])

                    myarr = np.array(mjd, dtype=[(self.mjdCol, 'f8')])
                    myarr = rf.append_fields(myarr,'night',night)
                    myarr = rf.append_fields(myarr, [self.RaCol, self.DecCol, self.filterCol], [
                        [Ra]*len(myarr), [Dec]*len(myarr), [band]*len(myarr)])
                    myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                                         [m5_coadded]*len(myarr), [Nvisits[band]]*len(myarr), [Nvisits[band]*Single_Exposure_Time[band]]*len(myarr), [season]*len(myarr)])
                    myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                                         [seeingEff[band]]*len(myarr), [seeingGeom[band]]*len(myarr)])
                    myarr = rf.append_fields(myarr,self.visitTime,[Nvisits[band]*Single_Exposure_Time[band]]*len(myarr))
                    rtot.append(myarr)
                    

            """
            for i, band in enumerate(bands):
                mjd = np.arange(mjd_min, mjd_max+cadence[band],cadence[band])
                #if mjd_max not in mjd:
                #    mjd = np.append(mjd, mjd_max)
                mjd += shift_days[band]
                m5_coadded = self.m5coadd(m5[band],
                                          Nvisits[band],
                                          Exposure_Time[band])

                myarr = np.array(mjd, dtype=[(self.mjdCol, 'f8')])
                myarr = rf.append_fields(myarr, [self.RaCol, self.DecCol, self.filterCol], [
                                         [Ra]*len(myarr), [Dec]*len(myarr), [band]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                                         [m5_coadded]*len(myarr), [Nvisits[band]]*len(myarr), [Nvisits[band]*Exposure_Time[band]]*len(myarr), [season]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                                         [seeingEff[band]]*len(myarr), [seeingGeom[band]]*len(myarr)])
                rtot.append(myarr)
            """
        res = np.copy(np.concatenate(rtot))
        res.sort(order=self.mjdCol)

        res = rf.append_fields(res,'observationId',np.random.randint(10*len(res),size=len(res)))

        self.Observations = res

    def makeFake(self, config):
        """ Generate Fake observations

        Parameters
        -------------
        config : dict
           dict of parameters (config file)

        Returns
        ---------
        recordarray of observations with the fields:
        MJD, Ra, Dec, band,m5,Nexp, ExpTime, Season
        accessible through self.Observations

        """

        bands = config['bands']
        cadence = dict(zip(bands, config['Cadence']))
        shift_days = dict(
            zip(bands, [config['shift_days']*io for io in range(len(bands))]))
        m5 = dict(zip(bands, config['m5']))
        Nvisits = dict(zip(bands, config['Nvisits']))
        Exposure_Time = dict(zip(bands, config['Exposure_Time']))
        inter_season_gap = 300.
        seeingEff = dict(zip(bands, config['seeingEff']))
        seeingGeom = dict(zip(bands, config['seeingGeom']))
        Ra = config['Ra']
        Dec = config['Dec']
        rtot = []
        # for season in range(1, config['nseasons']+1):
        for il, season in enumerate(config['seasons']):
            #mjd_min = config['MJD_min'] + float(season-1)*inter_season_gap
            mjd_min = config['MJD_min'][il]
            mjd_max = mjd_min+config['season_length']

            for i, band in enumerate(bands):
                mjd = np.arange(mjd_min, mjd_max+cadence[band],cadence[band])
                #if mjd_max not in mjd:
                #    mjd = np.append(mjd, mjd_max)
                mjd += shift_days[band]
                m5_coadded = self.m5coadd(m5[band],
                                          Nvisits[band],
                                          Exposure_Time[band])

                myarr = np.array(mjd, dtype=[(self.mjdCol, 'f8')])
                myarr = rf.append_fields(myarr, [self.RaCol, self.DecCol, self.filterCol], [
                                         [Ra]*len(myarr), [Dec]*len(myarr), [band]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                                         [m5_coadded]*len(myarr), [Nvisits[band]]*len(myarr), [Nvisits[band]*Exposure_Time[band]]*len(myarr), [season]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                                         [seeingEff[band]]*len(myarr), [seeingGeom[band]]*len(myarr)])
                rtot.append(myarr)

        res = np.copy(np.concatenate(rtot))
        res.sort(order=self.mjdCol)

        res = rf.append_fields(res,'observationId',np.random.randint(10*len(res),size=len(res)))

        self.Observations = res

    def m5coadd(self, m5, Nvisits, Tvisit):
        """ Coadded m5 estimation
        use approx. m5+=1.25*log10(Nvisits*Tvisits/30.)

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
        Flux error correction (m5 values different between template file and observations)

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
        Estimate Fisher elements

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
        #FisherEl = self.FisherValues(param, phase, fluxerr)

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
    
    #class to load template LC
    

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

    


