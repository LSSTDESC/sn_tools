import numpy as np


class Observations:
    """ class Observations
    Input
    ---------
    File (txt) with a list of observations
    (dumped from OpSim output for instance)
    Returns
    ---------
    seasons: the list of seasons with observations


    by default : poor seasons (< 10 observations all bands) are removed
    """

    def __init__(self, data,
                 nseasons=10, season_length=95,
                 names=dict(zip(['band', 'mjd', 'rawSeeing', 'sky', 'exptime', 'moonPhase', 'pixRA', 'pixDec', 'Nexp', 'fiveSigmaDepth', 'seeing', 'airmass', 'night', 'season'], ['band', 'mjd', 'seeingFwhm500', 'sky', 'exptime', 'moonPhase', 'pixRA', 'pixDec', 'numExposures', 'fiveSigmaDepth', 'seeingFwhmEff', 'airmass', 'night', 'season'])), coadd_night=True):

        self.names = names

        data.sort(order=self.names['mjd'])

        self.all_seasons = data
        self.seasons = self.Get_Seasons(data, season_length)

        if coadd_night:
            self.Coadd()
        self.Remove_Poor_Seasons()
        # print 'Nseasons',len(self.seasons)
        # print data

    def Coadd(self):

        # self.seasons_coadd={}
        for i in range(len(self.seasons)):
            season = self.seasons[i]
            coadd_tot = None
            for band in np.unique(season['band']):
                idx = season['band'] == band
                coadd = self.Coadd_Season_night(season[idx])

                if coadd_tot is None:
                    coadd_tot = coadd
                else:
                    coadd_tot = np.concatenate((coadd_tot, coadd))
            self.seasons[i] = coadd_tot

    def Coadd_Season_night(self, filt):

        if len(filt) == 0:
            return None

        r = []

        nights = np.unique(filt[self.names['night']])

        var_tot = ['band', 'pixRA', 'pixDec', 'pixarea']
        vars_mean = ['mjd', 'rawSeeing', 'sky', 'airmass',
                     'fiveSigmaDepth', 'moonPhase', 'night']

        for night in nights:
            idx = filt[self.names['night']] == night
            theslice = filt[idx]
            restot = dict(zip([var for var in var_tot], [
                          theslice[self.names[var]][0] for var in var_tot]))

            N5 = 10**(-0.4*theslice[self.names['fiveSigmaDepth']])/5.
            # print('N5',N5)
            Ntot = np.sum((1./N5)**2)
            Ftot = 5./np.sqrt(Ntot)
            m5_tot = -2.5*np.log10(Ftot)

            res = dict(zip([var for var in vars_mean], [np.mean(
                theslice[self.names[var]]) for var in vars_mean]))
            res['exptime'] = np.sum(theslice[self.names['exptime']])
            res['seeing'] = np.mean(theslice[self.names['seeing']])

            res['fiveSigmaDepth'] = m5_tot
            res['Nexp'] = np.sum(theslice[self.names['Nexp']])
            restot.update(res)
            r.append(tuple([restot[key] for key in restot.keys()]))

        # print(restot)
        # print('hello',restot.keys(),r)

        return np.rec.fromrecords(r, names=[key for key in restot.keys()])

    def Load(self, filename):
        """ Load txt file of observations
         Input
        ---------
        File (txt) with a list of observations
        (dumped from OpSim output for instance)
        Returns
        ---------
        recordarray of observations
        """
        sfile = open(filename, 'r')
        varname = []
        r = []
        for line in sfile.readlines():
            if line[0] == '#':
                varname.append(line.split(' ')[1])
            else:
                tofill = []
                thesplit = line.strip().split(' ')
                tofill.append(thesplit[0])
                for i in range(1, len(thesplit)):
                    tofill.append(float(thesplit[i]))
                r.append(tuple(tofill))
        return np.rec.fromrecords(r, names=varname)

    def Get_Seasons(self, data, season_length):
        """ Gather list of observations
        to make seasons
        Input
        ---------
        data : recordarray of observations

        Returns
        ---------
        seasons: the list of seasons with observations
        each element of the list is a recordarray
        """

        thediff = data[self.names['mjd']][1:]-data[self.names['mjd']][:-1]
        idx, = np.where(thediff > season_length)
        lidx = [val+1 for val in list(idx)]

        lidx.insert(0, 0)
        lidx.append(len(data[self.names['mjd']]))

        seasons = {}

        for i in range(len(lidx)-1):
            seasons[i] = data[lidx[i]:lidx[i+1]]

        return seasons

    def Remove_Poor_Seasons(self):
        """ Remove seasons with too few obs (<10)
        Returns
        ---------
        List of seasons with at least
        10 observations.
        """

        iseason = -1
        res = {}

        for i in range(len(self.seasons)):
            # print 'season',i,len(self.seasons[i])
            if len(self.seasons[i]) > 10:
                iseason += 1
                res[iseason] = self.seasons[i]

        self.seasons = res
