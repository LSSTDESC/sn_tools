import numpy as np
import numpy.lib.recfunctions as rf
import pandas as pd

__all__ = ['CoaddStacker']


# class CoaddStacker(BaseStacker):
class CoaddStacker:
    """
    Class to coadd observations per night
    """

    def __init__(self, col_sum=['numExposures', 'visitTime'],
                 col_mean=['observationStartMJD', 'fieldRA', 'fieldDec',
                           'pixRA', 'pixDec',
                           'healpixID', 'season'],
                 col_median=['airmass', 'sky', 'moonPhase',
                             'seeingFwhmEff', 'seeingFwhmGeom'],
                 col_group=['note', 'filter', 'night', 'visitExposureTime'],
                 col_coadd='fiveSigmaDepth',
                 col_visit='visitExposureTime'):

        self.col_sum = col_sum
        self.col_mean = col_mean
        self.col_median = col_median
        self.col_coadd = col_coadd
        self.col_group = col_group
        self.col_visit = col_visit

        # self.exptimeCol = col_coadd[1]

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
        names = simData.dtype.names
        col_sum = list(set(self.col_sum) & set(names))
        col_median = list(set(self.col_median) & set(names))
        col_mean = list(set(self.col_mean) & set(names))

        if cols_present:
            # Column already present in data;
            # assume it is correct and does not need recalculating.
            return simData

        self.dtype = simData.dtype

        if self.col_visit not in simData.dtype.names:
            simData = rf.append_fields(
                simData, self.visitTimeCol, [999.]*len(simData))

        df = pd.DataFrame(np.copy(simData))

        df[self.col_visit] = df[self.col_visit].astype(int)

        groups = df.groupby(self.col_group)
        listref = df.columns
        # get sum values
        tt = groups[col_sum].sum().reset_index()
        tta = groups[col_mean].mean().reset_index()
        ttb = groups[col_median].median().reset_index()
        ttc = groups.apply(lambda x: self.coadd_m5(x)).reset_index()
        ttd = groups.apply(lambda x: self.sum_colvisit(x)).reset_index()

        tt = self.merge_it(tt, tta)
        tt = self.merge_it(tt, ttb)
        tt = self.merge_it(tt, ttc)
        tt = self.merge_it(tt, ttd)

        tt = tt[tt.columns.drop(list(tt.filter(regex='level')))]

        tt = tt.drop(columns=[self.col_visit])
        tt = tt.rename(columns={'{}_sum'.format(
            self.col_visit): self.col_visit})

        return tt.to_records(index=False)

    def sum_colvisit(self, grp):
        """
        Method to get the sum of col_visit

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        pandas df
            Processed data.

        """

        res = grp[self.col_visit].sum()

        return pd.DataFrame({'{}_sum'.format(self.col_visit): [res]})

    def merge_it(self, data, datb):
        """
        Method to merge pandas df

        Parameters
        ----------
        data : pandas df
            first df to merge.
        datb : pandas df
            second df to merge.

        Returns
        -------
        data : pandas df
            Merged df.

        """

        if not datb.empty:
            data = data.merge(datb, left_on=self.col_group,
                              right_on=self.col_group)

        return data

    def coadd_m5(self, grp):
        """
        Method to estimate coadded m5

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        pandas df
            Data with coadded m5.

        """

        res = 1.25*np.log10(np.sum(10**(0.8*grp[self.col_coadd])))

        return pd.DataFrame({self.col_coadd: [res]})

    def get_vals_deprecated(self, listref, cols, group, op):
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

    def get_vals_b_deprecated(self, listref, cols, df, keygroup, op):
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

    def m5_coadd_deprecated(self, m5):
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

    def m5_coadd_grp_deprecated(self, grp):
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
