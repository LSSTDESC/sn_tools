#from sn_tools.sn_cadence_tools import GenerateFakeObservations
from sn_tools.sn_io import make_dict_from_optparse
import numpy.lib.recfunctions as rf
import numpy as np


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

        for vv in ['MJDmin']:
            what = self.dd[vv]
            if '-' not in what or what[0] == '-':
                nn = list(map(float, what.split(',')))
            else:
                nn = list(map(float, what.split('-')))
                nn = range(np.min(nn), np.max(nn))
            self.dd[vv] = nn

    def genData(self):
        """
        Method to generate fake observations

        Returns
        -----------
        numpy array with fake observations

        """

        mygen = GenerateFakeObservations(self.dd).Observations
        # add a night column

        MJD_min = np.min(mygen['observationStartMJD'])
        nights = list(map(int, mygen['observationStartMJD']-MJD_min+1))

        mygen = rf.append_fields(mygen, 'night', nights)
        # add pixRA, pixDex, healpixID columns
        for vv in ['pixRA', 'pixDec', 'healpixID']:
            mygen = rf.append_fields(mygen, vv, [0.]*len(mygen))

        # add Ra, Dec,
        #mygen = rf.append_fields(mygen, 'Ra', mygen['fieldRA'])
        #mygen = rf.append_fields(mygen, 'RA', mygen['fieldRA'])
        #mygen = rf.append_fields(mygen, 'Dec', mygen['fieldRA'])

        print('array', mygen.dtype)
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
        Default : 'observationStartMJD', 'fieldRA', 'fieldDec','filter','fiveSigmaDepth','visitExposureTime','numExposures','visitTime','season'

    """

    def __init__(self, config,
                 mjdCol='observationStartMJD', RACol='fieldRA',
                 DecCol='fieldDec', filterCol='filter', m5Col='fiveSigmaDepth',
                 exptimeCol='visitExposureTime', nexpCol='numExposures',
                 seasonCol='season', seeingEffCol='seeingFwhmEff', seeingGeomCol='seeingFwhmGeom',
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
        rotTelPos = config['rotTelPos']
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
                    nvals = len(myarr)
                    myarr = rf.append_fields(myarr, [self.RACol, self.DecCol, self.filterCol,self.rotTelPosCol], [
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
        # cadence = dict(zip(bands, config['cadence']))
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
        rotTelPos = config['rotTelPos']
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
                nvals = len(myarr)
                myarr = rf.append_fields(myarr, [self.RACol, self.DecCol, self.filterCol,self.rotTelPosCol], [
                                         [RA]*nvals, [Dec]*nvals, [band]*nvals,[rotTelPos]*nvals])
                myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                                         [m5_coadded]*nvals, [Nvisits[band]]*nvals, [Nvisits[band]*Exposure_Time[band]]*nvals, [season]*nvals])
                myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                                         [seeingEff[band]]*nvals, [seeingGeom[band]]*nvals])
                myarr = rf.append_fields(myarr, ['airmass', 'sky', 'moonPhase'], [
                                         [airmass[band]]*nvals, [sky[band]]*nvals, [moonphase[band]]*nvals])
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
        rotTelPos = config['rotTelPos']
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
                nvals = len(myarr)
                myarr = rf.append_fields(myarr, [self.RACol, self.DecCol, self.filterCol,self.rotTelPosCol], [
                                         [RA]*nvals, [Dec]*nvals, [band]*nvals],[rotTelPos]*nvals)
                myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                                         m5_coadded, [Nvisits[band]]*nvals, [Nvisits[band]*Exposure_Time[band]]*nvals, [season]*nvals])
                myarr = rf.append_fields(myarr, [self.seeingEffCol, self.seeingGeomCol], [
                                         [seeingEff[band]]*nvals, [seeingGeom[band]]*nvals])
                myarr = rf.append_fields(myarr, ['airmass', 'sky', 'moonPhase'], [
                                         [airmass[band]]*nvals, [sky[band]]*nvals, [moonphase[band]]*nvals])
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
            from scipy.interpolate import interp1d
            dictout[b] = interp1d(
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
