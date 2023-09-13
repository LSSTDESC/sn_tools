import numpy as np
import numpy.lib.recfunctions as rf
import pandas as pd

__all__ = ['CoaddStacker']


# class CoaddStacker(BaseStacker):
class CoaddStacker:
    """
    Class to coadd observations per night
    """

    def __init__(self, col_sum=['numExposures', 'visitTime',
                                'visitExposureTime'],
                 col_mean=['observationStartMJD', 'fieldRA', 'fieldDec',
                           'fiveSigmaDepth', 'pixRA', 'pixDec',
                           'healpixID', 'season'],
                 col_median=['airmass', 'sky', 'moonPhase',
                             'seeingFwhmEff', 'seeingFwhmGeom'],
                 col_group=['filter', 'night'],
                 col_coadd=['fiveSigmaDepth', 'visitExposureTime']):

        self.col_sum = col_sum
        self.col_mean = col_mean
        self.col_median = col_median
        self.col_coadd = col_coadd
        self.col_group = col_group
        self.visitTimeCol = col_sum[1]
        self.exptimeCol = col_coadd[1]

    def _run(self, simData, cols_present=False):
        """Main run method

        Parameters
        ---------------

        simulation data

        Returns
        -----------

        numpy array with the following fields:
        fieldRA : RA of the field (median per night)
        fieldDec: Dec of the field (median per night)
        fiveSigmaDepth: coadded m5
        night: night number
        filter: filter
        numExposures: number of exposures (sum per night)
        visitTime: visit time (sum per night)
        visitExposureTime: visit exposure time (sum per night)

        """

        # clean the list if necessary

        for vv in self.col_median:
            if vv not in simData.dtype.names:
                self.col_median.remove(vv)

        if 'note' in simData.dtype.names:
            simData = rf.drop_fields(simData, 'note')
        if cols_present:
            # Column already present in data;
            # assume it is correct and does not need recalculating.
            return simData
        self.dtype = simData.dtype

        if self.visitTimeCol not in simData.dtype.names:
            simData = rf.append_fields(
                simData, self.visitTimeCol, [999.]*len(simData))

        r = []

        # print(type(simData))
        df = pd.DataFrame(np.copy(simData))

        # print(df)
        # time_ref = time.time()

        #keygroup = [self.filterCol, self.nightCol]
        keygroup = self.col_group
        """
        keysums =  [self.numExposuresCol,self.visitExposureTimeCol]
        if self.visitTimeCol in simData.dtype.names:
            keysums += [self.visitTimeCol]
        keymeans = [self.mjdCol, self.RACol, self.DecCol, self.m5Col]
        """

        """
        exptime_single = \
            df.groupby(['filter'])[self.exptimeCol].mean().to_frame(
                name='exptime_single').reset_index()
        """
        bands = 'ugrizy'
        exptime_single = pd.DataFrame(list(bands), columns=['filter'])
        exptime_single['exptime_single'] = 30.
        groups = df.groupby(keygroup)
        listref = df.columns
        tt = self.get_vals(listref, self.col_sum, groups, np.sum)
        vv = self.get_vals(listref, self.col_mean, groups, np.mean)
        if not vv.empty:
            tt = tt.merge(vv, left_on=['night', 'filter'],
                          right_on=['night', 'filter'])

        # vv = self.get_vals_b(listref, self.col_median,
        #                     df, keygroup, 'median')
        vv = df.groupby(keygroup)[self.col_median].apply(
            lambda x: x.median()).reset_index()

        if not vv.empty:
            tt = tt.merge(vv, left_on=['night', 'filter'],
                          right_on=['night', 'filter'])

        tt = tt.merge(exptime_single, left_on=['filter'], right_on=['filter'])
        tt = tt.sort_values(by=['night'])
        tt.loc[:, self.col_coadd[0]] += 1.25 * \
            np.log10(tt[self.col_coadd[1]]/tt['exptime_single'])

        return tt.to_records(index=False)

    def get_vals(self, listref, cols, group, op):
        """
        Method to estimate a set of value using op

        Parameters
        --------------
        listref: list(str)
          list of columns of the group
        cols: list(str)
          list of columns to estimate values (op)
        group: pandas group
          data to process
        op: operator
         operator to apply

        Returns
        ----------
          pandas df with oped values

        """
        col_df = list(set(listref).intersection(cols))
        res = pd.DataFrame()
        if col_df:
            res = group[col_df].apply(lambda x: op(x)).reset_index()

        return res

    def get_vals_b(self, listref, cols, df, keygroup, op):
        """
        Method to estimate a set of value using op

        Parameters
        --------------
        listref: list(str)
          list of columns of the group
        cols: list(str)
          list of columns to estimate values (op)
        group: pandas group
          data to process
        op: operator
         operator to apply

        Returns
        ----------
          pandas df with oped values

        """
        col_df = list(set(listref).intersection(cols))
        print('col df', col_df)
        res = pd.DataFrame()
        if col_df:
            print('go man', df[col_df])
            res = df.groupby(keygroup)[col_df].apply(
                lambda x: eval('{}.{}()'.format(x, op))).reset_index()
            print(res)

        return res

    def fill(self, tab):
        """
        Field values estimation per night

        Parameters
        ---------------

        tab input table of field values (list given above)


        Returns
        -----------

        Field values per night : list
        all fields but m5, numexposures, visittime, visitexposuretime, 
        filter: median value
        m5 : coadded (cf m5_coadd)
        numexposures, visittime, visitexposuretime: added per night
        band: unique band value

        """

        r = []

        for colname in self.dtype.names:
            # print('there lan',colname)
            if colname in ['note']:
                r.append(np.unique(tab[colname])[0])
                continue
            if colname not in [self.m5Col, self.numExposuresCol,
                               self.visitTimeCol, self.visitExposureTimeCol,
                               self.filterCol]:
                if colname == 'sn_coadd':
                    r.append(1)
                else:
                    r.append(np.median(tab[colname]))
            if colname == self.m5Col:
                r.append(self.m5_coadd(np.copy(tab[self.m5Col])))
            if colname in [self.numExposuresCol, self.visitTimeCol,
                           self.visitExposureTimeCol]:
                r.append(np.sum(tab[colname]))
            if colname == self.filterCol:
                r.append(np.unique(tab[self.filterCol])[0])

        # print('done here',r)
        return r

    def m5_coadd(self, m5):
        """ Method to coadd m5 values

        .. math::
           \phi_{5\sigma} = 10^{-0.4*m_{5}}

           \sigma = \phi_{5\sigma}/5.

           \sigma_{tot} = 1./\sqrt(\sum(1./\sigma^2))

           \phi_{tot} = 5.*\sigma_{tot}

           m_{5}^{coadd} = -2.5*np.log10(\phi_{tot})

        Parameters
        --------------

        m5 : 5 sigma-depth values

        Returns
        ----------

        coadded m5 value


        """
        fluxes = 10**(-0.4*m5)
        sigmas = fluxes/5.
        sigma_tot = 1./np.sqrt(np.sum(1./sigmas**2))
        flux_tot = 5.*sigma_tot

        return -2.5*np.log10(flux_tot)

    def m5_coadd_grp(self, grp):
        """ Method to coadd m5 values

        .. math::
           \phi_{5\sigma} = 10^{-0.4*m_{5}}

           \sigma = \phi_{5\sigma}/5.

           \sigma_{tot} = 1./\sqrt(\sum(1./\sigma^2))

           \phi_{tot} = 5.*\sigma_{tot}

           m_{5}^{coadd} = -2.5*np.log10(\phi_{tot})

        Parameters
        --------------

        m5 : 5 sigma-depth values

        Returns
        ----------

        coadded m5 value


        """
        fluxes = 10**(-0.4*grp[self.m5Col])
        sigmas = fluxes/5.
        sigma_tot = 1./np.sqrt(np.sum(1./sigmas**2))
        flux_tot = 5.*sigma_tot

        grp[self.m5Col] = -2.5*np.log10(flux_tot)

        return grp
