import multiprocessing
import pandas as pd
import numpy as np
from astropy.table import Table, Column, vstack
import time
from scipy.linalg import lapack
#lapack_routine = lapack_lite.dgesv
import scipy.linalg as la
import operator
import numpy.lib.recfunctions as rf


class LCfast:
    def __init__(self, reference_lc, x1, color, telescope, mjdCol, RaCol, DecCol,
                 filterCol, exptimeCol,
                 m5Col, seasonCol):

        self.RaCol = RaCol
        self.DecCol = DecCol
        self.filterCol = filterCol
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.exptimeCol = exptimeCol
        self.seasonCol = seasonCol
        self.x1 = x1
        self.color = color

        # Loading reference file
        self.reference_lc = reference_lc

        self.telescope = telescope

        # This cutoffs are used to select observations:
        # phase = (mjd - DayMax)/(1.+z)
        # selection: min_rf_phase < phase < max_rf_phase
        # and        blue_cutoff < mean_rest_frame < red_cutoff
        # where mean_rest_frame = telescope.mean_wavelength/(1.+z)
        self.blue_cutoff = 300.
        self.red_cutoff = 800.

        # SN parameters for Fisher matrix estimation
        self.param_Fisher = ['x0', 'x1', 'daymax', 'color']

    def __call__(self, obs, index_hdf5, gen_par=None, bands='grizy'):
        """ Simulation of the light curve
        We use multiprocessing (one band per process) to increase speed

        Parameters
        ---------
        obs: array
         array of observations
        index_hdf5: int
         index of the LC in the hdf5 file (to allow fast access)
        gen_par: array
         simulation parameters
        display: bool,opt
         to display LC as they are generated (default: False)
        time_display: float, opt
         time persistency of the displayed window (defalut: 0 sec)



        Returns
        ---------
        astropy table with:
        columns: band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
        metadata : SNID,Ra,Dec,DayMax,X1,Color,z
        """

        ra = np.mean(obs[self.RaCol])
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
            tab_tot = tab_tot.append(resultdict[j], ignore_index=True)

        # return produced LC
        return tab_tot

    def processBand(self, sel_obs, band, gen_par, j=-1, output_q=None):
        """ LC simulation of a set of obs corresponding to a band
        The idea is to use python broadcasting so as to estimate 
        all the requested values (flux, flux error, Fisher components, ...)
        in a single path (i.e no loop!)

        Parameters
        -----------
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
        #yi = gen_par['z']

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
            fluxes_obs_err = self.reference_lc.fluxerr[band](pts)

            """
            fluxes_obs = np.nan_to_num(fluxes_obs)
            fluxes_obs_err = np.nan_to_num(fluxes_obs_err)
            """

            # Fisher components estimation

            dFlux = {}

            # loop on Fisher parameters
            for val in self.param_Fisher:
                dFlux[val] = self.reference_lc.param[band][val](pts)
            # get the reference components
            #z_c = self.reference_lc.lc_ref[band]['d'+val]
            # get Fisher components from interpolation
            # dFlux[val] = griddata((x, y), z_c, (p, yi_arr),
            #                      method=method, fill_value=0.)

        # replace crazy fluxes by dummy values
        fluxes_obs[fluxes_obs <= 0.] = 1.e-10
        fluxes_obs_err[fluxes_obs_err <= 0.] = 1.e-10

        # Fisher matrix components estimation
        # loop on SN parameters (x0,x1,color)
        # estimate: dF/dxi*dF/dxj/sigma_flux**2
        Derivative_for_Fisher = {}
        for ia, vala in enumerate(self.param_Fisher):
            for jb, valb in enumerate(self.param_Fisher):
                if jb >= ia:
                    Derivative_for_Fisher[vala +
                                          valb] = dFlux[vala] * dFlux[valb]

        # remove LC points outside the restframe phase range
        min_rf_phase = gen_par['min_rf_phase'][:, np.newaxis]
        max_rf_phase = gen_par['max_rf_phase'][:, np.newaxis]
        flag = (p >= min_rf_phase) & (p <= max_rf_phase)

        # remove LC points outside the (blue-red) range
        mean_restframe_wavelength = np.array(
            [self.telescope.mean_wavelength[band]]*len(sel_obs))
        mean_restframe_wavelength = np.tile(
            mean_restframe_wavelength, (len(gen_par), 1))/(1.+gen_par['z'][:, np.newaxis])
        flag &= (mean_restframe_wavelength > self.blue_cutoff) & (
            mean_restframe_wavelength < self.red_cutoff)

        flag_idx = np.argwhere(flag)

        # Correct fluxes_err (m5 in generation probably different from m5 obs)

        # gamma_obs = self.telescope.gamma(
        #    sel_obs[self.m5Col], [band]*len(sel_obs), sel_obs[self.exptimeCol])

        gamma_obs = self.reference_lc.gamma[band](
            (sel_obs[self.m5Col], sel_obs[self.exptimeCol]))

        mag_obs = -2.5*np.log10(fluxes_obs/3631.)

        m5 = np.asarray([self.reference_lc.m5_ref[band]]*len(sel_obs))

        gammaref = np.asarray([self.reference_lc.gamma_ref[band]]*len(sel_obs))

        m5_tile = np.tile(m5, (len(p), 1))

        srand_ref = self.srand(
            np.tile(gammaref, (len(p), 1)), mag_obs, m5_tile)

        srand_obs = self.srand(np.tile(gamma_obs, (len(p), 1)), mag_obs, np.tile(
            sel_obs[self.m5Col], (len(p), 1)))

        correct_m5 = srand_ref/srand_obs

        fluxes_obs_err = fluxes_obs_err/correct_m5

        # now apply the flag to select LC points
        fluxes = np.ma.array(fluxes_obs, mask=~flag)
        fluxes_err = np.ma.array(fluxes_obs_err, mask=~flag)
        phases = np.ma.array(p, mask=~flag)
        snr_m5 = np.ma.array(fluxes_obs/fluxes_obs_err, mask=~flag)

        nvals = len(phases)

        obs_time = np.ma.array(
            np.tile(sel_obs[self.mjdCol], (nvals, 1)), mask=~flag)
        seasons = np.ma.array(
            np.tile(sel_obs[self.seasonCol], (nvals, 1)), mask=~flag)
        exp_time = np.ma.array(
            np.tile(sel_obs[self.exptimeCol], (nvals, 1)), mask=~flag)
        m5_obs = np.ma.array(
            np.tile(sel_obs[self.m5Col], (nvals, 1)), mask=~flag)
        healpixIds = np.ma.array(
            np.tile(sel_obs['healpixID'].astype(int), (nvals, 1)), mask=~flag)

        pixRas = np.ma.array(
            np.tile(sel_obs['pixRa'], (nvals, 1)), mask=~flag)

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
        lc['flux'] = fluxes[~fluxes.mask]
        lc['fluxerr'] = fluxes_err[~fluxes_err.mask]
        lc['phase'] = phases[~phases.mask]
        lc['snr_m5'] = snr_m5[~snr_m5.mask]
        lc['m5'] = m5_obs[~m5_obs.mask]
        lc['mag'] = mag_obs[~mag_obs.mask]
        lc['magerr'] = (2.5/np.log(10.))/snr_m5[~snr_m5.mask]
        lc['time'] = obs_time[~obs_time.mask]
        lc['exposuretime'] = exp_time[~exp_time.mask]
        lc['band'] = ['LSST::'+band]*len(lc)
        lc['zp'] = [2.5*np.log10(3631)]*len(lc)
        lc['zpsys'] = ['ab']*len(lc)
        lc['season'] = seasons[~seasons.mask]
        lc['healpixID'] = healpixIds[~healpixIds.mask]
        lc['pixRa'] = pixRas[~pixRas.mask]
        lc['pixDec'] = pixDecs[~pixDecs.mask]
        lc['z'] = z_vals
        lc['daymax'] = daymax_vals
        lc['flux_e_sec'] = self.reference_lc.mag_to_flux_e_sec[band](
            lc['mag'])
        lc['flux_5'] = self.reference_lc.mag_to_flux_e_sec[band](
            lc['m5'])
        for key, vals in Fisher_Mat.items():
            lc['F_{}'.format(key)] = vals[~vals.mask]/(lc['fluxerr']**2)
        lc.loc[:, 'x1'] = self.x1
        lc.loc[:, 'color'] = self.color

        if output_q is not None:
            output_q.put({j: lc})
        else:
            return lc

    def srand(self, gamma, mag, m5):
        x = 10**(0.4*(mag-m5))
        return np.sqrt((0.04-gamma)*x+gamma*x**2)


class CalcSN:
    def __init__(self, lc_all, params=['x0', 'x1', 'color']):

        # select only fields involved in the calculation
        # this would save some memory
        fields = []
        for fi in ['season', 'healpixID', 'pixRa', 'pixDec', 'z',
                   'daymax', 'band', 'snr_m5', 'time', 'fluxerr', 'phase']:
            fields.append(fi)

        for ia, vala in enumerate(params):
            for jb, valb in enumerate(params):
                if jb >= ia:
                    fields.append('F_'+vala+valb)

        # lc to process
        lc = Table(lc_all[fields])

        # LC selection

        goodlc, badlc = self.selectLC(lc, params)

        res = badlc
        if len(goodlc) > 0:
            self.calcSigma(lc, goodlc, params)
            res = vstack([res, goodlc])

        self.sn = res

    def calcSigma(self, lc, restab_good, params):

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

        resu = np.ma.array(
            np.tile(np.copy(lc[mystr]), (len(valu), 1)), mask=~flag)

        sigma_x0_x1_color(resu, restab_good, params=['x0', 'x1', 'color'])

    def selectLC(self, lc, params):

        restab = Table(
            np.unique(lc[['season', 'healpixID', 'pixRa', 'pixDec', 'z', 'daymax']]))

        valu = np.unique(lc['z', 'daymax'])
        diff = lc['daymax']-valu['daymax'][:, np.newaxis]
        flag = np.abs(diff) < 1.e-5
        diffb = lc['z']-valu['z'][:, np.newaxis]
        flag &= np.abs(diffb) < 1.e-5

        tile_band = np.tile(lc['band'], (len(valu), 1))
        tile_snr = np.tile(lc['snr_m5'], (len(valu), 1))

        flag_snr = tile_snr > 5.

        #difftime = lc['time']-lc['daymax']
        phase = np.tile(lc['phase'], (len(valu), 1))

        count_bef = self.select_phase(phase, flag, flag_snr, operator.lt, 0.)
        restab.add_column(Column(count_bef, name='N_bef'))

        count_aft = self.select_phase(phase, flag, flag_snr, operator.ge, 0.)
        restab.add_column(Column(count_aft, name='N_aft'))

        count_pmin = self.select_phase(phase, flag, True, operator.le, -5.)
        restab.add_column(Column(count_pmin, name='N_phmin'))

        count_pmax = self.select_phase(phase, flag, True, operator.ge, 30.)
        restab.add_column(Column(count_pmax, name='N_phmax'))

        # LC selection here
        idx = (restab['N_bef'] >= 2) & (restab['N_aft'] >= 5)
        idx &= (restab['N_phmin'] >= 1) & (restab['N_phmax'] >= 1)
        restab_good = Table(restab[idx])
        restab_bad = Table(restab[~idx])
        for par in params:
            restab_bad.add_column(
                Column([100.]*len(restab_bad), name='Cov_{}{}'.format(par, par)))

        return restab_good, restab_bad

    def select_phase(self, phase, flag, flag_snr, op, phase_cut):

        fflag = op(phase, phase_cut)
        fflag &= flag
        fflag &= flag_snr

        ma_diff = np.ma.array(phase, mask=~fflag)
        count = ma_diff.count(axis=1)
        return count


class CalcSN_df:
    def __init__(self, lc_all, params=['x0', 'x1', 'daymax', 'color']):

        # select only fields involved in the calculation
        # this would save some memory
        fields = []
        for fi in ['season', 'healpixID', 'pixRa', 'pixDec', 'z',
                   'daymax', 'snr_m5', 'phase', 'x1', 'color']:
            fields.append(fi)

        tosum = []
        for ia, vala in enumerate(params):
            for jb, valb in enumerate(params):
                if jb >= ia:
                    fields.append('F_'+vala+valb)
                    tosum.append('F_'+vala+valb)
        time_ref = time.time()
        # lc to process - move to pandas DataFrame format
        lc = pd.DataFrame(np.copy(lc_all[fields]))

        # LC selection

        lc.loc[:, 'N_aft'] = (np.sign(lc['phase']) == 1) & (lc['snr_m5'] >= 5.)
        lc.loc[:, 'N_bef'] = (np.sign(lc['phase']) == -
                              1) & (lc['snr_m5'] >= 5.)
        #lc.loc[:,'N_phmin'] = (lc['phase']<=-5.)&(lc['snr_m5']>=5.)
        #lc.loc[:,'N_phmax'] = (lc['phase']>=20)&(lc['snr_m5']>=5.)
        lc.loc[:, 'N_phmin'] = (lc['phase'] <= -5.)
        lc.loc[:, 'N_phmax'] = (lc['phase'] >= 20)

        # transform boolean to int because of some problems in the sum()

        for colname in ['N_aft', 'N_bef', 'N_phmin', 'N_phmax']:
            lc[colname] = lc[colname].astype(int)

        tosum += ['N_aft', 'N_bef', 'N_phmin', 'N_phmax']
        sums = lc.groupby(['x1', 'color', 'season', 'healpixID',
                           'pixRa', 'pixDec', 'z', 'daymax'])[tosum].sum()

        idx = sums['N_aft'] >= 10
        idx &= sums['N_bef'] >= 5
        idx &= sums['N_phmin'] >= 1
        idx &= sums['N_phmax'] >= 1
        #idx &= sums['F_colorcolor']>=1.e-8

        goodsn = sums.loc[idx]
        badsn = sums.loc[~idx]

        time_ref = time.time()

        res = self.calcDummy(badsn, params)

        #time_ref = time.time()

        if len(goodsn) > 0:
            restab = self.calcSigma(goodsn, params)
            res = np.concatenate((res, restab))

        #print('after sigma',time.time()-time_ref)
        #time_ref = time.time()

        #print('final result',res)

        self.sn = res

    def calcDummy(self, lc, params):

        df = lc.copy()
        # for ia, vala in enumerate(params):
        df.loc[:, 'Cov_colorcolor'] = [100.]*len(lc)

        """
        groups = lc.groupby(['z','daymax'])

        df = groups.apply(lambda x : fillBad(x))
        """
        #names = ['z','daymax','season', 'pixRa', 'pixDec', 'Cov_x0x0', 'Cov_x1x1','Cov_colorcolor']
        names = ['x1', 'color', 'z', 'daymax', 'season',
                 'healpixID', 'pixRa', 'pixDec', 'Cov_colorcolor']
        names += ['N_aft', 'N_bef', 'N_phmin', 'N_phmax']
        gr = df.groupby(names)
        # print(gr.keys,list(gr.groups.keys()))
        restab = np.rec.fromrecords(list(gr.groups.keys()), names=gr.keys)

        return restab

    """
    def npoints(self,gr):

        # only points with snr>5.
        ida =  gr['snr_m5']>=5.
        grsel = gr.loc[ida]

        idx = grsel['phase']<0.
        gr['N_bef'] = len(grsel.loc[idx])
        gr['N_aft'] = len(grsel)-len(grsel.loc[idx])

        idxa = grsel['phase']<=-5.
        idxb = grsel['phase']>=30.
        
        gr['N_phmin'] = len(grsel.loc[idxa])
        gr['N_phmax'] = len(grsel.loc[idxb])
        return gr
    """

    def covColor(self, lc):

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

    def covColor_old(self, lc):

        a = lc['F_x0x0']
        e = lc['F_x1x1']
        i = lc['F_colorcolor']
        b = lc['F_x0x1']
        c = lc['F_x0color']
        f = lc['F_x1color']
        d = b
        g = c
        h = f

        det = a*e*i+b*f*g+c*d*h-c*e*g-f*h*a-i*b*d

        res = (a*e-b*d)/det

        return res

    def det(self, a1, a2, a3, b1, b2, b3, c1, c2, c3):

        resp = a1*b2*c3+b1*c2*a3+c1*a2*b3
        resm = a3*b2*c1+b3*c2*a1+c3*a2*b1

        return resp-resm

    def calcSigma(self, lc, params):

        # first: group by (z,daymax)

        df = lc.copy()

        df.loc[:, 'Cov_colorcolor'] = self.covColor(lc)

        """
        gg = df.groupby(['x1','color','z','daymax','season', 'pixRa', 'pixDec']).apply(lambda x: sigma_x0_x1_color_grp(x,params=['x0','x1','daymax','color']))

        print(gg)
        """
        names = ['x1', 'color', 'z', 'daymax', 'season',
                 'healpixID', 'pixRa', 'pixDec', 'Cov_colorcolor']
        names += ['N_aft', 'N_bef', 'N_phmin', 'N_phmax']
        grfi = df.groupby(names)

        """
        # for each group: estimate the sum

        #sums = groups.sum().groupby(['z','daymax','season', 'pixRa', 'pixDec'])

        gr = groups.apply(lambda x: sigma_x0_x1_color_grp(x,params=['x0','x1','color']))

        names = ['z','daymax','season', 'pixRa', 'pixDec', 'Cov_x0x0', 'Cov_x1x1','Cov_colorcolor']
        grfi = gr.groupby(names)
        """

        restab = np.rec.fromrecords(list(grfi.groups.keys()), names=grfi.keys)
        # print('test',restab)

        return restab
    """
    def selectLC(self,lc, params):
        
        restab = Table(np.unique(lc[['season','pixRa','pixDec','z','daymax']]))
    
        valu = np.unique(lc['z','daymax'])
        diff = lc['daymax']-valu['daymax'][:, np.newaxis]
        flag = np.abs(diff) < 1.e-5
        diffb = lc['z']-valu['z'][:, np.newaxis]
        flag &= np.abs(diffb) < 1.e-5

        tile_band = np.tile(lc['band'], (len(valu), 1))
        tile_snr = np.tile(lc['snr_m5'], (len(valu), 1))

        flag_snr = tile_snr > 5.

        #difftime = lc['time']-lc['daymax']
        phase = np.tile(lc['phase'], (len(valu), 1))

        count_bef = self.select_phase(phase, flag, flag_snr,operator.lt,0.)
        restab.add_column(Column(count_bef,name='N_bef'))

        count_aft = self.select_phase(phase, flag, flag_snr,operator.ge,0.)
        restab.add_column(Column(count_aft,name='N_aft'))

        count_pmin = self.select_phase(phase, flag, True,operator.le,-5.)
        restab.add_column(Column(count_pmin,name='N_phmin'))
        
        count_pmax = self.select_phase(phase, flag, True,operator.ge,30.)
        restab.add_column(Column(count_pmax,name='N_phmax'))

        # LC selection here
        idx = (restab['N_bef']>=2)&(restab['N_aft']>=5)
        idx &= (restab['N_phmin']>=1)&(restab['N_phmax']>=1)
        restab_good = Table(restab[idx])
        restab_bad = Table(restab[~idx])
        for par in params:
            restab_bad.add_column(Column([100.]*len(restab_bad),name='Cov_{}{}'.format(par,par)))

        return restab_good, restab_bad
    """


def calcSN_last(lc_all, params=['x0', 'x1', 'color'], j=-1, output_q=None):

    time_ref = time.time()

    #print('go man',j,len(lc_all))
    fields = []
    for fi in ['season', 'healpixID', 'pixRa', 'pixDec', 'z',
               'daymax', 'band', 'snr_m5', 'time', 'fluxerr', 'phase']:
        fields.append(fi)

    for ia, vala in enumerate(params):
        for jb, valb in enumerate(params):
            if jb >= ia:
                fields.append('F_'+vala+valb)

    lc = Table(lc_all[fields])

    # print('here',lc)

    restab = Table(
        np.unique(lc[['season', 'healpixID', 'pixRa', 'pixDec', 'z', 'daymax']]))

    valu = np.unique(lc['z', 'daymax'])
    diff = lc['daymax']-valu['daymax'][:, np.newaxis]
    flag = np.abs(diff) < 1.e-5
    diffb = lc['z']-valu['z'][:, np.newaxis]
    flag &= np.abs(diffb) < 1.e-5

    tile_band = np.tile(lc['band'], (len(valu), 1))
    tile_snr = np.tile(lc['snr_m5'], (len(valu), 1))

    flag_snr = tile_snr >= 5.

    #difftime = lc['time']-lc['daymax']
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
    """
    flagp = tile_diff>=0.
    flagn = tile_diff<0.
   
    for key,fl in dict(zip(['aft','bef'],[flagp, flagn])).items():
        fflag = np.copy(flag)
        fflag &= fl
        ma_diff = np.ma.array(tile_diff, mask=~fflag)
        count = ma_diff.count(axis=1)
        restab.add_column(Column(count,name='N_{}'.format(key)))
    """

    #print('after selection',time.time()-time_ref)
    # Select LCs with a sufficient number of LC points before and after max

    idx = (restab['N_bef'] >= 2) & (restab['N_aft'] >= 5)
    restab_good = Table(restab[idx])
    restab_bad = Table(restab[~idx])
    for par in params:
        restab_bad.add_column(
            Column([100.]*len(restab_bad), name='Cov_{}{}'.format(par, par)))

    #print('here again',valu[['z','daymax']],len(restab_good),len(restab_bad))

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

        sigma_x0_x1_color(resu, restab_good, params=['x0', 'x1', 'color'])

        restab = vstack([restab, restab_good])

    #print('after sigma',time.time()-time_ref)

    if output_q is not None:
        output_q.put({j: restab})
    else:
        return restab


def calcSN_old(lc_all, params=['x0', 'x1', 'color'], j=-1, output_q=None):

    time_ref = time.time()

    #print('go man',j,len(lc_all))
    fields = []
    for fi in ['season', 'pixRa', 'pixDec', 'z',
               'daymax', 'snr_m5', 'time', 'fluxerr', 'phase']:
        fields.append(fi)

    for ia, vala in enumerate(params):
        for jb, valb in enumerate(params):
            if jb >= ia:
                fields.append('F_'+vala+valb)
    #lc = Table(lc_all[fields])

    # lc.write('lctest.hdf5',path='lc')

    # print(type(lc_all[fields]))
    df = lc_all[fields].to_pandas()

    """
    store = pd.HDFStore('lctest.h5')
    store['df'] = df
    """

    groups = df.groupby(['season', 'pixRa', 'pixDec', 'z', 'daymax'])

    # print(groups.head(5))
    groups = groups.apply(npoints)

    # print(groups.head(5))
    #groups['N_bef']=groups['phase'].filter(lambda x: x<0).size()
    #groups['N_aft']=groups['phase'].filter(lambda x: x>=0).size()

    idx = (groups['N_bef'] >= 2) & (groups['N_aft'] >= 5)

    group_good = groups.loc[idx]
    group_bad = groups.loc[~idx]

    group_bad = group_bad.groupby(
        ['season', 'pixRa', 'pixDec', 'z', 'daymax', 'N_bef', 'N_aft'])
    restab_bad = Table(rows=list(group_bad.groups.keys()),
                       names=group_bad.keys)
    for par in params:
        restab_bad.add_column(
            Column([100.]*len(restab_bad), name='Cov_{}{}'.format(par, par)))

    group_good = group_good.groupby(
        ['season', 'pixRa', 'pixDec', 'z', 'daymax', 'N_bef', 'N_aft'])

    group_good = group_good.apply(
        lambda x: sigma_x0_x1_color_grp(x, params=['x0', 'x1', 'color']))

    groupfields = ['season', 'pixRa', 'pixDec',
                   'z', 'daymax', 'N_bef', 'N_aft']
    for par in params:
        groupfields.append('Cov_{}{}'.format(par, par))
    group_good = group_good.groupby(groupfields)
    restab_good = Table(rows=list(group_good.groups.keys()),
                        names=group_good.keys)
    restab = restab_bad

    restab = vstack([restab, restab_good])

    print('done', j)
    if output_q is not None:
        output_q.put({j: restab})
    else:
        return restab
    print('yes', group_good)

    print('oooo', group_good)

    restab = Table(np.unique(lc[['season', 'pixRa', 'pixDec', 'z', 'daymax']]))

    """
    for band in 'ugrizy':
        idx = lc['band'] == 'LSST::'+band
        resp = npointBand(lc[idx],band)
    """

    valu = np.unique(lc['z', 'daymax'])
    diff = lc['daymax']-valu['daymax'][:, np.newaxis]
    flag = np.abs(diff) < 1.e-5
    diffb = lc['z']-valu['z'][:, np.newaxis]
    flag &= np.abs(diffb) < 1.e-5

    tile_band = np.tile(lc['band'], (len(valu), 1))
    tile_snr = np.tile(lc['snr_m5'], (len(valu), 1))
    #tile_flux = np.tile(lc['flux_e_sec'], (len(valu), 1))
    #tile_flux_5 = np.tile(lc['flux_5'], (len(valu), 1))
    difftime = lc['time']-lc['daymax']
    tile_diff = np.tile(difftime, (len(valu), 1))

    flagp = tile_diff >= 0.
    flagn = tile_diff < 0.

    for key, fl in dict(zip(['aft', 'bef'], [flagp, flagn])).items():
        fflag = np.copy(flag)
        fflag &= fl
        ma_diff = np.ma.array(tile_diff, mask=~fflag)
        count = ma_diff.count(axis=1)
        restab.add_column(Column(count, name='N_{}'.format(key)))

    """
    for band in 'ugrizy':
        maskb = np.copy(flag)
        maskb &= tile_band=='LSST::'+band
        ma_band = np.ma.array(tile_snr, mask=~maskb,fill_value=0.).filled()
        snr_band = np.sqrt(np.sum(ma_band**2, axis=1))
        #ratflux = tile_flux/tile_flux_5
        #ma_flux = np.ma.array(ratflux, mask=~maskb,fill_value=0.).filled()
        #snr_band_5 = 5.*np.sqrt(np.sum(ma_flux**2, axis=1))
        restab.add_column(Column(snr_band,name='snr_{}'.format(band)))
        #restab.add_column(Column(snr_band_5,name='snr_5_{}'.format(band)))
        for key,fl in dict(zip(['aft','bef'],[flagp, flagn])).items():
            fflag = np.copy(maskb)
            fflag &= fl
            ma_diff = np.ma.array(tile_diff, mask=~fflag)
            count = ma_diff.count(axis=1)
            restab.add_column(Column(count,name='N_{}_{}'.format(key,band)))
    """

    print('after selection', time.time()-time_ref)
    # Select LCs with a sufficient number of LC points before and after max

    idx = (restab['N_bef'] >= 2) & (restab['N_aft'] >= 5)
    restab_good = Table(restab[idx])
    restab_bad = Table(restab[~idx])
    for par in params:
        restab_bad.add_column(
            Column([100.]*len(restab_bad), name='Cov_{}{}'.format(par, par)))

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

    resu = np.ma.array(
        np.tile(np.copy(lc[mystr]), (len(valu), 1)), mask=~(flag & flag_snr))

    # print(resu['snr_m5'])
    restab = restab_bad
    time_ref = time.time()
    """
    if len(restab_good) > 0:

        sigma_x0_x1_color(resu,restab_good,params=['x0','x1','color'])
        
        restab = vstack([restab,restab_good])
    """
    print('after sigma', time.time()-time_ref)

    if output_q is not None:
        output_q.put({j: restab})
    else:
        return restab


def npoints(gr):

    idx = gr['phase'] < 0.
    gr['N_bef'] = len(gr.loc[idx])
    gr['N_aft'] = len(gr)-len(gr.loc[idx])

    return gr


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


def sigma_x0_x1_color(resu, restab, params=['x0', 'x1', 'color']):

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
    #Big_Diag = np.diag(np.linalg.inv(Fisher_Big))
    # print('hhhh',Fisher_Big.shape)
    Big_Diag = np.diag(faster_inverse(Fisher_Big))
    # print('three',time.time()-time_ref)
    time_ref = time.time()
    for ia, vala in enumerate(params):
        indices = range(ia, len(Big_Diag), 3)
        restab.add_column(
            Column(np.take(Big_Diag, indices), name='Cov_{}{}'.format(vala, vala)))

    # print('four',time.time()-time_ref)
    #time_ref = time.time()


def sigma_x0_x1_color_grp(grp, params=['x0', 'x1', 'daymax', 'color']):

    # print('grp',grp)
    parts = {}
    for ia, vala in enumerate(params):
        for jb, valb in enumerate(params):
            if jb >= ia:
                parts[ia, jb] = grp['F_'+vala+valb]

    # print(parts)
    size = len(grp)
    npar = len(params)
    Fisher_Big = np.zeros((npar*size, npar*size))
    Big_Diag = np.zeros((npar*size, npar*size))
    Big_Diag = []

    for iv in range(size):
        Fisher_Matrix = np.zeros((npar, npar))
        for ia, vala in enumerate(params):
            for jb, valb in enumerate(params):
                if jb >= ia:
                    Fisher_Big[ia+npar*iv][jb+npar*iv] = parts[ia, jb]

    # pprint.pprint(Fisher_Big)

    Fisher_Big = Fisher_Big + np.triu(Fisher_Big, 1).T
    Big_Diag = np.diag(np.linalg.inv(Fisher_Big))

    for ia, vala in enumerate(params):
        indices = range(ia, len(Big_Diag), npar)
        # restab.add_column(
        #    Column(np.take(Big_Diag, indices), name='Cov_{}{}'.format(vala,vala)))
        grp['Co_{}{}'.format(vala, vala)] = np.take(Big_Diag, indices)

    return grp


def faster_inverse(A):
    b = np.identity(A.shape[1], dtype=A.dtype)
    #u, piv, x, info = lapack.dgesv(A, b)
    return la.solve(A, b)
