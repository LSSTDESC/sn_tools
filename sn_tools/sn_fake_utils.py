# from sn_tools.sn_cadence_tools import GenerateFakeObservations
from sn_tools.sn_io import make_dict_from_optparse
import numpy.lib.recfunctions as rf
import numpy as np
import pandas as pd


def add_option(parser, confDict):
    """
    Function to add options to a parser from dict

    Parameters
    --------------
    parser: parser
      parser of interest
    confDict: dict
      dict of values to add to parser

    """
    for key, vals in confDict.items():
        vv = vals[1]
        if vals[0] != 'str':
            vv = eval('{}({})'.format(vals[0], vals[1]))
        parser.add_option('--{}'.format(key), help='{} [%default]'.format(
            vals[2]), default=vv, type=vals[0], metavar='')


def config(confDict, opts):
    """
    Method to update a dict from opts parser values

    Parameters
    ---------------
    confDict: dict
       initial dict
    opts: opts.parser
      parser values

    Returns
    ----------
    updated dict

    """
    # make the fake config file here
    newDict = {}
    for key, vals in confDict.items():
        newval = eval('opts.{}'.format(key))
        newDict[key] = (vals[0], newval)

    dd = make_dict_from_optparse(newDict)

    return dd


class FakeObservations:
    """
    class to generate fake observations

    Parameters
    ----------------
    dict_config: dict
      configuration parameters

    """

    def __init__(self, dict_config):

        self.dd = dict_config

        # transform input conf dict
        self.transform_fakes()

        # generate fake observations

        self.obs = self.genData()

    def transform_fakes(self):
        """
        Method to transform the input dict
        to make it compatible with the fake observation generator

        """
        # few changes to be made here: transform some of the input to list
        for vv in ['seasons', 'seasonLength']:
            what = self.dd[vv]
            if '-' not in what or what[0] == '-':
                nn = list(map(int, what.split(',')))
            else:
                nn = list(map(int, what.split('-')))
                nn = range(np.min(nn), np.max(nn))
            self.dd[vv] = nn

        """
        for vv in ['MJDmin']:
            what = self.dd[vv]
            if '-' not in what or what[0] == '-':
                nn = list(map(float, what.split(',')))
            else:
                nn = list(map(float, what.split('-')))
                nn = range(np.min(nn), np.max(nn))
            self.dd[vv] = nn
        """

    def genData(self):
        """
        Method to generate fake observations

        Returns
        -----------
        numpy array with fake observations

        """

        mygen = GenerateFakeObservations(self.dd).Observations
        # add a night column

        # add pixRA, pixDex, healpixID columns
        for vv in ['pixRA', 'pixDec', 'healpixID']:
            mygen = rf.append_fields(mygen, vv, [0.]*len(mygen))

        # add Ra, Dec,
        # mygen = rf.append_fields(mygen, 'Ra', mygen['fieldRA'])
        # mygen = rf.append_fields(mygen, 'RA', mygen['fieldRA'])
        # mygen = rf.append_fields(mygen, 'Dec', mygen['fieldRA'])

        # plot test

        """
        plot_test(mygen, 'observationStartMJD', 'mjd',
                  'moonPhase', 'Moon phase', 'uz')
        plot_test(mygen, 'night', 'night', 'moonPhase', 'Moon phase', 'ugrizy')
        """

        # print(mygen)
        return mygen


class GenerateFakeObservations:
    """
    class to generate Fake observations

    Parameters
    ---------------

    config: dict
      dict of parameters
    list : str,opt
        Name of the columns for data generation.
        Default : 'observationStartMJD', 'fieldRA', 'fieldDec','filter',
        'fiveSigmaDepth','visitExposureTime','numExposures','visitTime',
        'season'

    """

    def __init__(self, config,
                 mjdCol='observationStartMJD', RACol='fieldRA',
                 DecCol='fieldDec', filterCol='filter', m5Col='fiveSigmaDepth',
                 exptimeCol='visitExposureTime', nexpCol='numExposures',
                 seasonCol='season', seeingEffCol='seeingFwhmEff',
                 seeingGeomCol='seeingFwhmGeom',
                 visitTime='visitTime', rotTelPosCol='rotTelPos',
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
        self.rotTelPosCol = rotTelPosCol

        self.obs_from_simu = config['obsFromSimu']
        self.obs_conditions = {}

        if self.obs_from_simu:
            # seasons = config['seasons']
            self.obs_conditions, self.mjd_min = \
                self.load_observing_conditions(
                    config['obsCondFile'])
            self.obs_conditions = self.obs_conditions[config['field']]
            """
            # [config['field']]
            print('obs conditions', self.obs_conditions)
            print(self.mjd_min)
            print(test)
            """

        # now make fake obs
        res = None
        if not sequences:
            if not self.obs_from_simu:
                res = self.makeFake(config)
            else:
                res = self.makeFake_from_simu(config)
            """
            if config['m5File'] == 'NoData':
                res = self.makeFake(config)

            else:
                res = self.makeFake_from_simu(config)
            """
        else:
            res = self.makeFake_sqs(config)

        # add note column
        res = rf.append_fields(
            res, 'note', [config['field']]*len(res), dtypes='<U11')

        self.Observations = res

    def load_observing_conditions(self, csvFile,
                                  cols=['fiveSigmaDepth', 'airmass', 'night']):
        """
        Method to load obs condition file

        Parameters
        ----------
        csvFile : str
            name of the files to load.
        seasons: list(int)
            List of seasons to consider. The default is [1].
        cols : list(str), optional
            list of cols to load. The default is ['fiveSigmaDepth'].

        Returns
        -------
        dictfi : dict
            obs conditions (interpolator).

        """

        # read csv file as pandas df
        obs_cond = pd.read_csv(csvFile, comment='#')

        #
        notes = obs_cond['note'].unique()
        dictfi = {}
        mjd_min = {}
        for note in notes:
            idx = obs_cond['note'] == note
            sel = obs_cond[idx]
            self.print_median_values(sel)
            dd = self.make_obs_interp(sel, cols)
            dictfi.update(dd)
            mjd_min[note] = self.get_mjd_min(sel)

        return dictfi, mjd_min

    def get_mjd_min(self, data):

        ddict = {}
        seasons = np.unique(data['season'])
        for seas in seasons:
            idx = data['season'] == seas
            selb = data[idx]
            ddict[seas] = selb['mjd'].min()

        return ddict

    def print_median_values(self, data, cols=['fiveSigmaDepth', 'airmass']):
        """
        Method to print median obs conditions

        Parameters
        ----------
        data : array
            Data to process.
        cols : list(str), optional
            List of columns to print. The default is ['fiveSigmaDepth','airmass'].

        Returns
        -------
        None.

        """
        vv = ['note', 'season', 'filter']

        df = data.groupby(vv)[
            cols].median().reset_index()

        vv += cols
        print(df[vv])

    def make_obs_interp(self, tt, cols):
        """
        Method to load obs conditions, interpolate and make a dict

        Parameters
        ----------
        grp : pandas df
            data to process.
        cols : list(str)
            columns to load.

        Returns
        -------
        dict
            output dict: keys=(note,col), val=interp1d(night,col).

        """

        from scipy.interpolate import interp1d
        grp = pd.DataFrame(tt)
        field = grp['note'].unique()

        dictOut = {}
        for vv in cols:
            ddict = {}
            seasons = np.unique(grp['season'])
            for seas in seasons:
                ida = grp['season'] == seas
                sela = grp[ida]
                ddict[seas] = {}
                if vv != 'night':
                    for b in 'ugrizy':
                        idx = grp['filter'] == b
                        sel = grp[idx]
                        """
                        # first rescale nights to start with the first night
                        min_night = sel['night'].min()
                        sel.loc[:, 'night'] -= min_night-1
                        """
                        ddict[seas][b] = interp1d(sel['mjd'], sel[vv],
                                                  bounds_error=False,
                                                  fill_value=sel[vv].median())
                else:
                    ddict[seas] = interp1d(sel['mjd'], sel[vv],
                                           bounds_error=False,
                                           fill_value=0.)

            dictOut[vv] = ddict

        return dict(zip(field, [dictOut]))

    def make_obs_interp_deprecated(self, tt, cols, seasons):
        """
        Method to load obs conditions, interpolate and make a dict

        Parameters
        ----------
        grp : pandas df
            data to process.
        cols : list(str)
            columns to load.
        seasons: list(int)
            list of seasons.

        Returns
        -------
        dict
            output dict: keys=(note,col), val=interp1d(night,col).

        """

        from scipy.interpolate import interp1d
        grp = pd.DataFrame(tt)
        field = grp['note'].unique()

        dictOut = {}
        for vv in cols:
            ddict = {}
            for seas in seasons:
                ida = grp['season'] == seas
                sela = grp[ida]
                ddict[seas] = {}
                for b in 'ugrizy':
                    idx = grp['filter'] == b
                    sel = grp[idx]
                    # first rescale nights to start with the first night
                    min_night = sel['night'].min()
                    sel.loc[:, 'night'] -= min_night-1
                    ddict[seas][b] = interp1d(sel['night'], sel[vv],
                                              bounds_error=False, fill_value=0.)
            dictOut[vv] = ddict

        return dict(zip(field, [dictOut]))

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
        # Single_Exposure_Time =
        # dict(zip(bands, config['Single_Exposure_Time']))
        inter_season_gap = 100.
        seeingEff = dict(zip(bands, config['seeingEff']))
        seeingGeom = dict(zip(bands, config['seeingGeom']))
        airmass = dict(zip(bands, config['airmass']))
        sky = dict(zip(bands, config['sky']))
        moonphase = dict(zip(bands, config['moonphase']))
        RA = config['RA']
        Dec = config['Dec']
        rotTelPos = config['rotTelPos']
        rtot = []
        # for season in range(1, config['nseasons']+1):
        Nvisits = {}
        Single_Exposure_Time = {}
        for il, season in enumerate(config['seasons']):
            m5 = dict(zip(bands, config['m5'][season]))
            mjd_min = config['MJDmin']+il * \
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
                    nvals = len(myarr)
                    myarr = rf.append_fields(myarr, [self.RACol, self.DecCol, self.filterCol, self.rotTelPosCol], [
                        [RA]*nvals, [Dec]*nvals, [band]*nvals], [rotTelPos]*nvals)
                    myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                        [m5_coadded]*nvals, [Nvisits[band]]*nvals, [Nvisits[band]*Single_Exposure_Time[band]]*nvals, [season]*nvals])
                    myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                        [seeingEff[band]]*nvals, [seeingGeom[band]]*nvals])
                    myarr = rf.append_fields(myarr, self.visitTime, [
                        Nvisits[band]*Single_Exposure_Time[band]]*nvals)
                    myarr = rf.append_fields(myarr, ['airmass', 'sky', 'moonPhase'], [
                        [airmass[band]]*nvals, [sky[band]]*nvals, [moonphase[band]]*nvals])
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
                                         [RA]*nvals, [Dec]*nvals, [band]*nvals])
                myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                                         [m5_coadded]*nvals, [Nvisits[band]]*nvals, [Nvisits[band]*Exposure_Time[band]]*nvals, [season]*nvals])
                myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                                         [seeingEff[band]]*nvals, [seeingGeom[band]]*nvals])
                rtot.append(myarr)
            """
        res = np.copy(np.concatenate(rtot))
        res.sort(order=self.mjdCol)

        res = rf.append_fields(res, 'observationId',
                               np.random.randint(10*len(res), size=len(res)))

        MJD_min = np.min(res['observationStartMJD'])
        MJD_min = config['MJDmin']
        nights = list(map(int, res['observationStartMJD']-MJD_min+1))

        res = rf.append_fields(res, 'night', nights)

        return res

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
        # cadence = dict(zip(bands, config['cadence']))
        cadence = config['cadence']
        shift_days = dict(
            zip(bands, [config['shiftDays']*io for io in range(len(bands))]))
        # m5 = dict(zip(bands, config['m5']))
        seasons = config['seasons']
        m5_dict = {}
        for b in bands:
            dd = config['m5'][b].split(',')
            m5_dict[b] = list(map(float, dd))
            if len(dd) == 1:
                m5_dict[b] = m5_dict[b]*len(seasons)
        print('alors', m5_dict)

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
        inter_season_gap = 100.
        RA = config['RA']
        Dec = config['Dec']
        rotTelPos = config['rotTelPos']
        rtot = []
        print(config)

        # for season in range(1, config['nseasons']+1):
        for il, season in enumerate(config['seasons']):
            # mjd_min = config['MJD_min'] + float(season-1)*inter_season_gap
            if len(config['seasonLength']) == 1:
                seasonLength = config['seasonLength'][0]
            else:
                seasonLength = config['seasonLength'][il]
            mjd_min = config['MJDmin']+(season-1) * \
                (seasonLength+inter_season_gap)
            mjd_max = mjd_min+seasonLength

            for i, band in enumerate(bands):
                mjd = np.arange(mjd_min, mjd_max+cadence[band], cadence[band])
                # if mjd_max not in mjd:
                #    mjd = np.append(mjd, mjd_max)
                mjd += shift_days[band]
                m5_coadded = self.m5coadd(m5_dict[band][il],
                                          Nvisits[band],
                                          Exposure_Time[band])

                myarr = np.array(mjd, dtype=[(self.mjdCol, 'f8')])
                nvals = len(myarr)
                myarr = rf.append_fields(myarr, [self.RACol,
                                                 self.DecCol,
                                                 self.filterCol,
                                                 self.rotTelPosCol],
                                         [[RA]*nvals,
                                         [Dec]*nvals,
                                         [band]*nvals,
                                         [rotTelPos]*nvals])
                myarr = rf.append_fields(myarr,
                                         [self.m5Col,
                                          self.nexpCol,
                                          self.exptimeCol,
                                          self.seasonCol],
                                         [[m5_coadded]*nvals,
                                          [Nvisits[band]]*nvals,
                                          [Nvisits[band]*Exposure_Time[band]]
                                          * nvals,
                                          [season]*nvals])
                myarr = rf.append_fields(myarr,
                                         [self.seeingEffCol,
                                          self.seeingGeomCol],
                                         [[seeingEff[band]]*nvals,
                                          [seeingGeom[band]]*nvals])

                myarr = rf.append_fields(myarr,
                                         ['airmass', 'sky', 'moonPhase'], [
                                             [airmass[band]]*nvals,
                                             [sky[band]]*nvals,
                                             [moonphase[band]]*nvals])
                rtot.append(myarr)

        res = np.copy(np.concatenate(rtot))
        res.sort(order=self.mjdCol)

        res = rf.append_fields(res, 'observationId',
                               np.random.randint(10*len(res), size=len(res)))

        MJD_min = np.min(res['observationStartMJD'])
        MJD_min = config['MJDmin']
        nights = list(map(int, res['observationStartMJD']-MJD_min+1))

        res = rf.append_fields(res, 'night', nights)

        # before moon impact: get the total number of visits per band

        Nvisits = {}
        for b in np.unique(res['filter']):
            idx = res['filter'] == b
            sel = res[idx]
            Nvisits[b] = np.sum(sel['numExposures'])

        res = self.moonImpact(config, res)

        res = self.moonCompensate(config, res, Nvisits)

        ll = ['fiveSigmaDepth', 'airmass', 'filter',
              'season', 'night', 'visitExposureTime']

        res = self.estimate_obs_conditions(res)

        return res

    def moonCompensate(self, config, res, Nvisits):

        comp = config['mooncompensate']

        moonPhase_u = config['moonPhaseu']

        if comp == 0 or moonPhase_u < 0:
            return res

        # comp = 1: need to rescale swapfilter obs

        swapfilter = config['moonswapFilter']

        idx = res['filter'] == swapfilter
        sel = res[idx]
        selb = res[~idx]

        Nvisits_swap = np.sum(sel['numExposures'])
        Nnights_swap = len(sel)
        Nvisits_req = Nvisits[swapfilter]
        delta_Nvisits = Nvisits_req-Nvisits_swap
        Nvisits_to_add = int(delta_Nvisits/Nnights_swap)
        sel['numExposures'] += Nvisits_to_add
        sel['visitExposureTime'] += Nvisits_to_add*30.
        # correct for m5 value
        sel['fiveSigmaDepth'] = config['m5'][swapfilter] + \
            1.25*np.log10(sel['numExposures'])

        res = np.concatenate((selb, sel))
        return res

    def moonImpact(self, config, res):
        """
        Method to add u-obs and remove u <-> filter obs.

        Parameters
        ----------
        config : dict
            config parameters.
        res : numpy array
            array of obs.

        Returns
        -------
        numpy array
            array with u-obs if moonPhase_u >= 0.

        """

        moonPhase_u = config['moonPhaseu']

        if moonPhase_u < 0:
            return res

        import ephem
        # estimate moonPhase
        r = []
        for mjd in res['observationStartMJD']:
            moon = ephem.Moon(mjd2djd(mjd))
            r.append((moon.phase))

        # print(r)

        # remove moonPhase field from res

        res = rf.drop_fields(res, 'moonPhase')

        # and replace this monnPhase col by estimation
        res = rf.append_fields(res, 'moonPhase', r)

        # remove x-band observations to be replaced by u-band obs
        idx = res['moonPhase'] <= moonPhase_u
        idx &= res['filter'] == config['moonswapFilter']
        sel_drop = np.copy(res[idx])

        sel_main = np.copy(res[~idx])

        # drop filter field and replace by u-obs

        vvals = ['Nvisits', 'ExposureTime',
                 'seeingEff', 'seeingGeom', 'airmass', 'm5']
        dict_var = {}

        seasons = np.unique(sel_drop['season'])
        for vv in vvals:
            if vv != 'm5':
                dict_var[vv] = config[vv]['u']
            else:
                tt = config[vv]['u'].split(',')
                dict_var[vv] = list(map(float, tt))
                if len(tt) == 1:
                    dict_var[vv] = dict_var[vv]*len(seasons)

        selu = None
        for io, seas in enumerate(seasons):
            dictb = {}
            idx = sel_drop['season'] == seas
            selb = sel_drop[idx]

            dictb['numExposures'] = dict_var['Nvisits']
            dictb['visitExposureTime'] = dict_var['ExposureTime']
            dictb['seeingFwhmEff'] = dict_var['seeingEff']
            dictb['seeingFwhmGeom'] = dict_var['seeingGeom']
            dictb['filter'] = 'u'
            dictb['fiveSigmaDepth'] = dict_var['m5'][io] + \
                1.25*np.log10(dictb['numExposures'])

            for key, vals in dictb.items():
                # sel_drop = drop_add(sel_drop, key, [vals]*len(sel_drop))
                selb[key] = [vals]*len(selb)

            if selu is None:
                selu = selb
            else:
                selu = np.concatenate((selu, selb))

        sel_main = np.concatenate((sel_main, selu))

        # plot_test(sel_main)

        return sel_main

    def estimate_obs_conditions(self, res):
        """
        Method to estimate obs conditions (m5, airmass...) from input csv file

        Parameters
        ----------
        res : array
            data to process.

        Returns
        -------
        resfi : array
            processed data.

        """

        if not self.obs_from_simu:
            return res

        bands = np.unique(res['filter'])
        resfi = None
        seasons = np.unique(res['season'])
        for seas in seasons:
            idx = res['season'] == seas
            sela = res[idx]
            for b in bands:
                idx = sela['filter'] == b
                sel = np.copy(sela[idx])
                for key, vals in self.obs_conditions.items():
                    # vv = vals[b](sel['night'])
                    # sel = rf.append_fields(sel, key, vv)
                    if key != 'night':
                        sel[key] = vals[seas][b](sel['observationStartMJD'])
                    else:
                        sel[key] = vals[seas](sel['observationStartMJD'])
                if resfi is None:
                    resfi = sel
                else:
                    resfi = np.concatenate((resfi, sel))

        # coadd m5 if necessary

        if 'fiveSigmaDepth' in resfi.dtype.names:
            resfi['fiveSigmaDepth'] += 1.25 * \
                np.log10(resfi['visitExposureTime']/30.)

        return resfi

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
        # cadence = dict(zip(bands, config['cadence']))
        cadence = config['cadence']
        season_length = config['seasonLength'][0]

        print('tt', bands, cadence, season_length)
        shift_days = dict(
            zip(bands, [config['shiftDays']*io for io in range(len(bands))]))
        # m5 = dict(zip(bands, config['m5']))
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
        rotTelPos = config['rotTelPos']
        rtot = []
        # prepare m5 for interpolation
        field = config['field']
        ddict = self.obs_conditions
        for il, season in enumerate(config['seasons']):
            # search for mjdmin and mjdmax for this season
            mjd_min = self.mjd_min[field][season]
            mjd_max = mjd_min+season_length
            for i, band in enumerate(bands):
                mjd = np.arange(mjd_min, mjd_max+cadence[band], cadence[band])
                # if mjd_max not in mjd:
                #    mjd = np.append(mjd, mjd_max)
                # mjd += shift_days[band]
                m5_interp = ddict['fiveSigmaDepth'][season][band]
                airmass_interp = ddict['airmass'][season][band]
                night_interp = ddict['night'][season]
                m5_coadded = self.m5coadd(m5_interp(mjd),
                                          Nvisits[band],
                                          Exposure_Time[band])
                myarr = np.array(mjd, dtype=[(self.mjdCol, 'f8')])
                nvals = len(myarr)
                ccols = [self.RACol, self.DecCol,
                         self.filterCol, self.rotTelPosCol]
                vals = [[RA]*nvals, [Dec]*nvals,
                        [band]*nvals, [rotTelPos]*nvals]
                myarr = rf.append_fields(myarr, ccols, vals)

                ccols = [self.m5Col, self.nexpCol,
                         self.exptimeCol, self.seasonCol]
                vals = [m5_coadded,
                        [Nvisits[band]]*nvals,
                        [Nvisits[band]*Exposure_Time[band]] * nvals,
                        [season]*nvals]

                myarr = rf.append_fields(myarr, ccols, vals)

                ccols = [self.seeingEffCol, self.seeingGeomCol]
                vals = [[seeingEff[band]]*nvals, [seeingGeom[band]]*nvals]

                myarr = rf.append_fields(myarr, ccols, vals)

                ccols = ['airmass', 'sky', 'moonPhase', 'night']
                vals = [airmass_interp(mjd),
                        [sky[band]]*nvals,
                        [moonphase[band]]*nvals,
                        night_interp(mjd)]

                myarr = rf.append_fields(myarr, ccols, vals)
                rtot.append(myarr)

        res = np.copy(np.concatenate(rtot))
        res.sort(order=self.mjdCol)

        res = rf.append_fields(res, 'observationId',
                               np.random.randint(10*len(res), size=len(res)))

        # before moon impact: get the total number of visits per band

        Nvisits = {}
        for b in np.unique(res['filter']):
            idx = res['filter'] == b
            sel = res[idx]
            Nvisits[b] = np.sum(sel['numExposures'])

        res = self.moonImpact(config, res)

        res = self.moonCompensate(config, res, Nvisits)

        ll = ['fiveSigmaDepth', 'airmass', 'filter',
              'season', 'night', 'visitExposureTime']

        res = self.estimate_obs_conditions(res)

        return res

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
            from scipy.interpolate import interp1d
            dictout[b] = interp1d(
                selb['observationStartMJD'], selb['fiveSigmaDepth'],
                fill_value=0., bounds_error=False)
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


def mjd2djd(mjd):
    """Convert from modified Julian date to Dublin Julian date
    pyephem uses Dublin Julian dates
    Parameters
    ----------
    mjd : float
        The modified Julian date to be converted.
    Returns
    -------
    The Dublin Julian date corresponding to `mjd`.
    """
    # (this function adapted from Peter Yoachim's code)
    doff = 15019.5  # this equals ephem.Date(0)-ephem.Date('1858/11/17')
    return mjd - doff


def drop_add(res, col, vals):
    """
    Function to drop a column and add it again (with different values)

    Parameters
    ----------
    res : record array
        data to process.
    col : str
        field to consider.
    vals : list
        values for replacement.

    Returns
    -------
    res : record array
        updated array.

    """

    # drop

    res = rf.drop_fields(res, col)

    # replace
    res = rf.append_fields(res, col, vals)

    return res


def plot_test(sel_main, xvar, xlabel, yvar, ylabel, bands):
    """
    Method to plot moonPhase vs mjd for cross check

    Parameters
    ----------
    sel_main : numpy array
        data to plot.
    xvar : str
        x-axis var.
    xlabel : str
        x-axis label.
    yvar : str
        y-axis var.
    ylabel : str
        y-axis label.
    bands: str
        list of bands to consider.

    Returns
    -------
    None.

    """

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(sel_main[xvar],
            sel_main[yvar], 'ko', mfc='None')

    filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))
    for b in bands:
        idx = sel_main['filter'] == b
        selp = sel_main[idx]
        color = filtercolors[b]
        ax.plot(selp[xvar], selp[yvar],
                '{}*'.format(color), mfc='None')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()
