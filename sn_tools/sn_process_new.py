
# from sn_tools.sn_obs import renameFields, patchObs
from sn_tools.sn_obs import ProcessPixels_metric
from sn_tools.sn_io import colName
# from sn_tools.sn_obs import getObservations, get_obs
from sn_tools.sn_obs import get_obs
from sn_tools.sn_obs import ebv_pixels
from sn_tools.sn_utils import multiproc
from sn_tools.sn_fp_pixel import get_window, get_pixels_in_window
from sn_tools.sn_fp_pixel import FocalPlane
from sn_tools.sn_fp_pixel import get_xy_pixels, get_data_window
import time
import numpy as np
import pandas as pd
# import multiprocessing
import glob
import random
import numpy.lib.recfunctions as rf


class FP2pixels:

    def __init__(self, dbDir='', dbName='', dbExtens='',
                 fieldType='', fieldName='', lookup_ddf='', nside=128,
                 RAmin=0., RAmax=360.,
                 Decmin=-80., Decmax=80,
                 pixelmap_dir='', npixels=0, nproc_pixels=1,
                 VRO_FP='circular', project_FP='gnomonic', telrot=0.,
                 ebvofMW_pixel=-1.0,
                 radius=4., pixelList='None', display=False,
                 seasons='-1', **kwargs):

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        self.fieldType = fieldType
        self.fieldNames = fieldName.split(',')
        self.nside = nside
        self.RAmin = RAmin
        self.RAmax = RAmax
        self.Decmin = Decmin
        self.Decmax = Decmax
        self.nproc_pixels = nproc_pixels
        self.pixelmap_dir = pixelmap_dir
        self.npixels = npixels
        self.radius = radius
        self.VRO_FP = VRO_FP
        self.project_FP = project_FP
        self.telrot = telrot
        self.ebvofMW_pixel = ebvofMW_pixel
        self.pixelList = pixelList
        self.display = display

        assert (self.RAmin <= self.RAmax)

        # loading data (complete file)
        observations = get_obs(fieldType, dbDir, dbName, dbExtens, lookup_ddf)

        names = observations.dtype.names
        self.RACol = colName(names, ['fieldRA', 'RA', 'Ra'])
        self.DecCol = colName(names, ['fieldDec', 'Dec'])

        # select observations
        if self.fieldType != 'DD':
            self.fieldNames = [self.fieldType]
        obs = None

        # select observations to needs here
        pixels = pd.DataFrame()
        for fieldName in self.fieldNames:
            obs_sel = self.select_obs(observations, [fieldName])

            # print('there man', len(obs), these_seasons)
            if len(obs_sel) > 0:
                # get the pixels
                pixels_obs = self.get_pixels_obs(obs_sel, fieldName)
                pixels = pd.concat((pixels, pixels_obs))
                if obs is None:
                    obs = obs_sel
                else:
                    obs = np.concatenate((obs, obs_sel))

        # not sure this is necessary: gnomonic projection taks care of this.
        # but necessary to select obs aroud a (pixRA,pixDec) window
        obs = self.check_obs_pixels(obs, pixels['pixRA'].mean())
        # self.plot_obs_pix(obs, pixels)

        pixels = self.select_pixels(pixels)
        # self.plot_obs_pix(obs, pixels)

        # observations are in self.obs
        # pixels are in self.pixels
        self.obs = obs
        self.pixels = pixels

    def check_obs_pixels(self, obsa, pixRA_mean):
        """
        Method to "translate" obs in RA for pixels close to 0/360

        Parameters
        ----------
        obsa : array
            Obs.
        pixRA_mean : float
            Mean RA for pixels

        Returns
        -------
        obs : array
            Modified data.

        """

        # some modifs to data have to be done near RA=0/360.
        obs = np.array(obsa)
        ra_mean = pixRA_mean

        r = obs[self.RACol]-ra_mean
        obs = rf.append_fields(obs, 'diff_RA', r.tolist())
        idx = obs['diff_RA'] < -100.
        obs[self.RACol][idx] += 360.

        idx = obs['diff_RA'] > 100.
        obs[self.RACol][idx] -= 360.

        return obs

    def plot_obs_pix(self, obs, pixels):
        """
        Method to plot obs and pixels in (RA,Dec)

        Parameters
        ----------
        obs : array
            Observations.
        pixels : pandas df
            Pixels.

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.plot(obs[self.RACol], obs[self.DecCol], 'k.')
        ax.plot(pixels['pixRA'], pixels['pixDec'], 'r*')

        plt.show()

    def select_pixels(self, pixels_in):
        """
        Method to select pixels according to processing parameters

        Parameters
        ----------
        pixels : pandas df
            Data to process.

        Returns
        -------
        pixels : pandas df
            selected pixels.

        """
        pixels = pd.DataFrame(pixels_in)
        # do not process pixels with high E(B-V)
        if self.ebvofMW_pixel > 0.:
            pixels = self.select_ebv(pixels)

        # grab pixel from pixellist
        if self.pixelList != 'None' and not pixels.empty:
            hh = pd.read_csv(self.pixelList, comment='#')
            healpixIDs = hh['healpixID'].to_list()
            idx = pixels['healpixID'].isin(healpixIDs)
            pixels = pixels[idx]

        # grab random pixels
        if self.npixels > 0 and not pixels.empty:
            hlist = pixels['healpixID'].tolist()
            random_pixels = randomPixels(hlist, self.npixels)
            idx = pixels['healpixID'].isin(random_pixels)
            pixels = pixels[idx]

        return pixels

    def get_pixels_obs(self, obs, fieldName):
        """
        Get pixels related to observations

        Parameters
        ----------
        obs : numpy array
            Observations.
        fieldName : str
            field name.

        Returns
        -------
        pixels : pandas df
            list of pixels.

        """

        if fieldName == 'WFD':
            RA_min = self.RAmin
            RA_max = self.RAmax
            Dec_min = np.min(obs[self.DecCol])-self.radius
            Dec_max = np.max(obs[self.DecCol])+self.radius
            Dec_min = np.max([-89., Dec_min])
        else:
            RA_min, RA_max, Dec_min, Dec_max = get_window(
                obs, radius=np.sqrt(20./3.14))

        pixels = get_pixels_in_window(
            self.nside, RA_min, RA_max, Dec_min, Dec_max)

        return pixels

    def select_ebv(self, pixels, ebvofMW_pixel):

        hpixes = np.unique(pixels[['healpixID', 'pixRA', 'pixDec']], axis=0)
        hpixes = ebv_pixels(hpixes)
        pixels = pixels.merge(hpixes,
                              left_on=['healpixID'], right_on=['healpixID'])

        idx = pixels['ebvofMW'] <= ebvofMW_pixel
        pixels = pixels[idx]

        print('allo', pixels)

        return pixels

    def load_season(self, seasons):
        """
        Method to get the list of seasons

        Parameters
        ----------
        seasons : str
               list of seasons.

        Returns
        -------
        season : list(int)
            list of seasons to process

        """

        if '-' not in seasons or seasons[0] == '-':
            season = list(map(int, seasons.split(',')))
        else:
            seasl = seasons.split('-')
            seasmin = int(seasl[0])
            seasmax = int(seasl[1])
            season = list(range(seasmin, seasmax+1))

        return season

    def __call__(self, obs=None):

        # print('getting pixels')
        if obs is None:
            obs = self.obs

        pixels = self.get_pixels_field(obs)

        if self.healpixIDs:
            idx = pixels['healpixID'].isin(self.healpixIDs)
            pixels = pixels[idx]

        # re-select data to avoid having data too far from pixels
        pixRA_min = pixels['pixRA'].min()
        pixRA_max = pixels['pixRA'].max()
        pixDec_min = pixels['pixDec'].min()
        pixDec_max = pixels['pixDec'].max()

        obs = self.select_zone(obs, pixRA_min, pixRA_max,
                               self.RACol, pixDec_min, pixDec_max,
                               self.DecCol, 8.)

        if self.display:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.suptitle('Pixels and observations')
            ax.plot(pixels['pixRA'], pixels['pixDec'], 'r*')
            ax.plot(obs[self.RACol],
                    obs[self.DecCol], 'ko', mfc='None')
            ax.set_xlabel('RA [deg]')
            ax.set_ylabel('Dec [deg]')
            plt.show()

        # print('getting pixels', self.nside, self.project_FP, self.VRO_FP,
        #      self.RACol, self.DecCol, self.telrot, self.nproc)
        time_ref = time.time()
        from sn_tools.sn_obs import DataToPixels
        nproc_p = self.nproc_pixels
        if self.healpixIDs:
            nproc_p = np.min([nproc_p, len(self.healpixIDs)])
        datapixels = DataToPixels(
            self.nside, self.project_FP, self.VRO_FP,
            RACol=self.RACol, DecCol=self.DecCol,
            telrot=self.telrot, nproc=nproc_p)

        pixels = datapixels(obs, pixels, display=self.display)

        pixels['healpixID'] = pixels['healpixID'].astype(int)
        # print('FP2pixels done', time.time() -
        #       time_ref, pixels['healpixID'].unique())

        if self.display:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(pixels['pixRA'], pixels['pixDec'], 'k.')
            plt.show()

        # get the list of pixels Id to process
        pixel_Ids = self.pixelList(self.npixels, pixels, self.healpixIDs)

        idx = pixels['healpixID'].isin(pixel_Ids)

        return pixels[idx]

    def select_obs(self, observations, fieldName):
        """
        Method to select observations

        Parameters
        ---------------
        observations: array
          data to process
        fieldName: str
          fieldName to select

        """
        noteCol = 'note'
        if 'scheduler_note' in observations.dtype.names:
            noteCol = 'scheduler_note'

        if self.fieldType == 'DD':
            idx = np.in1d(observations[noteCol], fieldName)
            observations = observations[idx]
            return observations

            # return self.select_zone(observations, RAmin, RAmax,
            #                        self.RACol, DecCol=self.DecCol)

        if self.fieldType == 'WFD':
            return self.select_zone(observations, self.RAmin, self.RAmax,
                                    self.RACol, Decmin=self.Decmin,
                                    Decmax=self.Decmax, DecCol=self.DecCol)

        if self.fieldType == 'Fake':
            return observations

    def select_zone(self, data, RAmin, RAmax, RACol, Decmin=None,
                    Decmax=None, DecCol='Dec', delta_coord=5.):
        """
        Method to select data in a given area defined by (RA_min, RA_max)

        Parameters
        --------------
        data: array
          data to process
        RA_min: float
          min RA value
        RA_max: float
          max RA value
        RACol: str
           RA colname
        Dec_min: float
          min Dec value
        Dec_max: float
          max Dec value
        DecCol: str
           Dec colname
        delta_coord: float, opt
          deltaRA to extend the (RA_min, RA_max) window (default: 5)

        """

        idx = data[RACol] >= RAmin-delta_coord
        idx &= data[RACol] < RAmax+delta_coord
        if Decmin is not None:
            idx &= data[DecCol] >= Decmin-delta_coord
            idx &= data[DecCol] < Decmax+delta_coord

        obs = np.copy(data[idx])

        if self.display:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            self.plotPixels(obs, RACol, DecCol, ax=ax, show=False)
        if RAmin < delta_coord:
            idk = data[RACol] > 360.-delta_coord
            obs = np.concatenate((obs, data[idk]))

        if RAmax >= 360.-delta_coord:
            idk = data[RACol] < delta_coord
            obs = np.concatenate((obs, data[idk]))

        if self.display:
            self.plotPixels(obs, RACol, DecCol, ax=ax, show=False)
            plt.show()
        return obs

    def get_pixels_field(self, observations):
        """
        Method to get pixels corresponding to a given area

        Parameters
        --------------
        observations: array
          data to process

        Returns
        -----------
        list of pixels

        """
        if self.fieldType == 'DD':
            mean_RA = np.mean(observations[self.RACol])
            mean_Dec = np.mean(observations[self.DecCol])
            width_RA = np.max(observations[self.RACol]) - \
                np.min(observations[self.RACol])
            width_Dec = np.max(observations[self.DecCol]) - \
                np.min(observations[self.DecCol])

        if self.fieldType == 'Fake':
            mean_RA = np.mean(observations[self.RACol])
            mean_Dec = np.mean(observations[self.DecCol])
            width_RA = 0.1
            width_Dec = 0.1

        if self.fieldType == 'WFD':
            width_RA = self.RAmax-self.RAmin
            width_Dec = self.Decmax-self.Decmin
            mean_RA = np.mean([self.RAmin, self.RAmax])
            mean_Dec = np.mean([self.Decmin, self.Decmax])

        # get pixels

        pixels = self.gime_pixels(
            mean_RA, mean_Dec, np.max([width_RA, width_Dec]))

        if self.fieldType == 'WFD':
            pixelsb = self.select_zone(
                pixels.to_records(index=False), self.RAmin, self.RAmax,
                'pixRA', self.Decmin, self.Decmax, 'pixDec', 0.)
            pixels = pd.DataFrame.from_records(pixelsb)

        if self.fieldType == 'Fake':

            pixels['diff_RA'] = pixels['pixRA']-mean_RA
            pixels['diff_Dec'] = pixels['pixDec']-mean_Dec
            idx = np.abs(pixels['diff_RA']) <= 0.5
            idx &= np.abs(pixels['diff_Dec']) <= 0.5
            pixels = pixels[idx][:1]

        # print('nb pixels', len(pixels))
        # print('there we go man', mean_RA, mean_Dec,
        #      width_RA, width_Dec, len(pixels))

        return pixels

    def plotPixels(self, pixels, RACol, DecCol, ax=None, show=True):
        """
        Method to plot (RA,Dec) of data

        Parameters
        ----------
        pixels : array
            data to plot
        RACol : str
            x-axis column
        DecCol : str
            y-axis column
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        show : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(pixels[RACol], pixels[DecCol], 'ko')
        if show:
            plt.show()

    def pixelList(self, npixels, pixels, healpixIDs=[]):
        """
        Method to get the list of pixels to process

        Parameters
        --------------
        npixels: int
          number of pixels to grab
        pixels: array
          list of "available" pixels
        healpixIDs: list, opt
          list of healpixIDs to process (default: [])

        Returns
        ----------
        list of healpixIDs to process

        """
        pixList = list(np.unique(pixels['healpixID']))

        if healpixIDs:
            # check wether required healpixID are in pixList
            ll = list(set(pixList).intersection(healpixIDs))
            if not ll:
                print('Big problem here: healpixIDs are not in pixel list!!')
                return None
            else:
                return healpixIDs

        if npixels == -1:
            return pixList

        random_pixels = randomPixels(pixList, npixels)

        return random_pixels

    def path_to_pixels(self):
        """
        method to grab pixel files

        Returns
        -----------
        full search path

        """

        search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_\
        {}'.format(self.pixelmap_dir,
                   self.dbName, self.dbName,
                   self.fieldType, self.nside,
                   self.RAmin, self.RAmax,
                   self.Decmin, self.Decmax)
        if self.fieldType == 'DD':
            search_path += '_{}'.format(self.fieldName)
        search_path += '.npy'

        return search_path

    def gime_pixels(self, RA, Dec, width, inclusive=True):
        """
        method to get pixels corresponding to obs area

        Parameters
        --------------
        RA: float
          mean RA position
        Dec: float
           mean Dec position
        width: float
           width of the window around (RA,Dec)
        inclusive: bool, opt
          inclusive bool for healpix (default: True)

        Returns
        -----------
        list of pixels corresponding to (RA, Dec) central position

        """
        import healpy as hp
        # get central pixel ID
        healpixID = hp.ang2pix(self.nside, RA,
                               Dec, nest=True, lonlat=True)

        # get nearby pixels
        vec = hp.pix2vec(self.nside, healpixID, nest=True)
        healpixIDs = hp.query_disc(
            self.nside, vec, np.deg2rad(width)+np.deg2rad(3.5),
            inclusive=inclusive, nest=True)

        # get pixel coordinates
        coords = hp.pix2ang(self.nside, healpixIDs,
                            nest=True, lonlat=True)
        pixRA, pixDec = coords[0], coords[1]

        pixels = pd.DataFrame(healpixIDs, columns=['healpixID'])
        pixels['pixRA'] = pixRA
        pixels['pixDec'] = pixDec

        return pixels

    def file_pixels(self):
        """
        Method to search for and load an ObsTopixel file

        Returns
        ----------
        obsTopixel file (pandas df)

        """
        search_path = self.path_to_pixels()
        pixelmap_file = glob.glob(search_path)[0]
        print('searching in', search_path, pixelmap_file)

        res = np.load(pixelmap_file, allow_pickle=True)

        return pd.DataFrame(res)

    def build_pixels(self, pixelIds, params, j=0, output_q=None):
        """
        Method to grab pixels matching VRO FP

        Parameters
        ---------------
        obs: array
           data to process (VRO FP centers)

        Returns
        -----------
        obstopixel file with a list of pixels and matched obs (pandas df)

        """

        time_ref = time.time()
        obs = params['obs']
        pixels = params['pixels']
        display = params['display']

        idx = pixels['healpixID'].isin(pixelIds)
        pixels = pixels[idx]

        idx = pixels['healpixID'].isin(pixelIds)
        pixels = pixels[idx]

        from sn_tools.sn_obs import DataToPixels
        datapixels = DataToPixels(
            self.nside, self.project_FP, self.VRO_FP,
            RACol=self.RACol, DecCol=self.DecCol,
            telrot=self.telrot, nproc=self.nproc)
        pixels = datapixels(obs, pixels, display=display)

        print('after pixel/obs matching', time.time()-time_ref)

        if output_q is not None:
            return output_q.put({j: pixels})
        else:
            return pixels


class Process(FP2pixels):
    """
    Class to process data ie run metrics on a set of pixels

    Parameters
    --------------
    dbDir: str
      dir location of observing strategy file
    dbName: str
      observing strategy name
    dbExtens: str
      database extension (npy, db, ...)
    fieldType: str
      type of field: DD, WFD, Fake
    nside: int
      healpix nside parameter
    RAmin: float
      min RA of the area to process
    RAmax: float
      max RA of the area to process
    Decmin: float
      min Dec of the area to process
    Decmax: float
      max Dec of the area to process
    saveData: bool
      to save ouput data or not
    remove_dithering: bool
      to remove dithering (to use for DD studies)
    outDir: str
      output directory location
    nprocs: int,
      number of cores to run
    metricList: list(metrics)
      list of metrics to process
    pixelmap_dir: str, opt
      location directory where maps pixels<->observations are (default: '')
    npixels: int, opt
      number of pixels to run on (default: 0)
    nclusters: int, opt
       number of clusters to make (DD only)(default: 5)
    radius: float, opt
      radius to get pixels arounf clusters (DD only) (default: 4 deg)
    pixelList: str, opt
      cvs file of pixels to process

    """

    def __init__(self, dbDir='', dbName='', dbExtens='',
                 fieldType='', fieldName='', lookup_ddf='',
                 nside=128,
                 RAmin=0., RAmax=360.,
                 Decmin=-80., Decmax=80,
                 saveData=False, remove_dithering=False,
                 outDir='', nproc=1, nproc_pixels=1, seasons=-1, metricList=[],
                 pixelmap_dir='', npixels=0, ebvofMW_pixel=-1.,
                 VRO_FP='circular', project_FP='gnomonic', telrot=0.,
                 fp_level='ccd',
                 radius=4., pixelList='None', display=False, **kwargs):
        super().__init__(dbDir, dbName, dbExtens,
                         fieldType, fieldName, lookup_ddf, nside,
                         RAmin, RAmax,
                         Decmin, Decmax,
                         pixelmap_dir, npixels, nproc_pixels,
                         VRO_FP, project_FP, telrot,
                         ebvofMW_pixel,
                         radius, pixelList, display, seasons)

        self.saveData = saveData
        self.remove_dithering = remove_dithering
        self.outDir = outDir
        self.nproc_pixels = nproc_pixels
        self.nproc = nproc
        self.metricList = metricList
        self.ebvofMW_pixel = ebvofMW_pixel
        self.fp_level = fp_level

        print('Npixels to process:', len(self.pixels), len(self.obs))
        if len(self.pixels) > 0:
            self.processIt()

    def processIt(self):
        """
        Method to process the pixels

        Returns
        -------
        None.

        """

        # FP instance
        df_fp = FocalPlane(level=self.fp_level)
        # quick check
        # df_fp.check_fp(top_level='raft', low_level='ccd')

        params = {}
        params['FP'] = df_fp
        if self.nproc > 1:
            multiproc(self.pixels, params, self.process_multipix, self.nproc)
        else:
            self.process_multipix(self.pixels, params)

    def process_multipix(self, pixels, params, j=0, output_q=None):
        """
        Method to process a set of pixels 

        Returns
        -------
        None.

        """

        procpix = ProcessPixels_metric(self.metricList, j,
                                       outDir=self.outDir, dbName=self.dbName,
                                       saveData=self.saveData)

        # loop on pixels
        obsCol = 'observationId'
        df_fp = params['FP']
        print('pixels to process', len(pixels))
        for i, pix in pixels.iterrows():
            print('processing pixel', pix['healpixID'])
            # gnomonic proj
            obs = np.copy(self.obs)
            time_ref = time.time()
            """
            print('before', len(obs))
            self.plot_ra_dec(obs, pix['pixRA'],
                             pix['pixDec'], self.RACol, self.DecCol)
            """
            obs = get_data_window(pix['pixRA'], pix['pixDec'],
                                  obs,
                                  RACol=self.RACol, DecCol=self.DecCol,
                                  radius=10.)
            """
            self.plot_ra_dec(obs, pix['pixRA'],
                             pix['pixDec'], self.RACol, self.DecCol)
            print('after', len(obs))
            """
            ppars = {}
            ppars['pixel'] = pix
            ppars['FP'] = df_fp
            pix_obs = multiproc(obs, ppars, self.proj_pixel, 1)
            # get matching obsId
            obsIDs = pix_obs[obsCol].to_list()
            idx = np.in1d(obs[obsCol], obsIDs)
            obs = obs[idx]
            obs.sort(order=obsCol)
            pix_arr = pix_obs.to_records(index=False)
            pix_arr.sort(order=obsCol)
            pix_arr = rf.drop_fields(pix_arr, obsCol)
            for vv in pix_arr.dtype.names:
                obs = rf.append_fields(obs, vv, pix_arr[vv].tolist())

            """
            obs_df = pd.DataFrame.from_records(obs[idx])
            cols = [obsCol, 'healpixID', 'pixRA', 'pixDec']
            pp = pix_obs[cols]
            obs_df = obs_df.merge(
                pp, left_on=['observationId'], right_on=['observationId'])
            obs = obs_df.to_records(index=False)
            """

            procpix(pix_obs, obs, j)
            print('processed', time.time()-time_ref)
        procpix.finish()

        print('process done', j)
        if output_q is not None:
            return output_q.put({j: 1})
        else:
            return 1

    def proj_pixel(self, obs, params, j=0, output_q=None):

        pix = params['pixel']
        df_fp = params['FP']
        dd = get_xy_pixels(obs,
                           pix['healpixID'],
                           pix['pixRA'],
                           pix['pixDec'],
                           RACol=self.RACol, DecCol=self.DecCol)
        # self.plot_pixels(dd)
        # df_fp.plot_fp_pixels(dd)
        # pixels in FP
        pix_obs = df_fp.pix_to_obs(dd)

        if output_q is not None:
            return output_q.put({j: pix_obs})
        else:
            return pix_obs

    def plot_ra_dec(self, data, pixRA, pixDec, RACol, DecCol):
        """
        Method to plot (RA,Dec)

        Parameters
        ----------
        data : pandas df
            Data to process.
        pixRA : float
            pixel RA.
        pixDec : float
            pixel Dec.
        RACol : str
            RA colname.
        DecCol : str
            Dec colname.

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.plot(data[RACol], data[DecCol], 'k.')
        ax.plot([pixRA], [pixDec], 'r*')

        plt.show()

    def plot_pixels(self, pp):
        """
        Method to plot pixels

        Parameters
        ----------
        pp : array
            Data to plot.

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(pp['xpixel'], pp['fieldRA']-pp['pixRA'], 'k.')
        plt.show()

    def processIt_deprecated(self, observations):
        """
        Method to process a field

        Parameters
        --------------
        observations: array
            data to process

        """

        # pixels = self.get_pixels_field(observations)
        # getting the pixels
        # print('getting pixels call')

        print('in processit, getting pixels')
        noteCol = 'note'
        if 'scheduler_note' in observations.dtype.names:
            noteCol = 'scheduler_note'
        pixels = pd.DataFrame()
        for field in self.fieldNames:
            if self.fieldType == 'DD':
                idx = observations[noteCol] == field
                selobs = observations[idx]
                ppix = super(Process, self).__call__(selobs)
                ppix['fieldName'] = field
                pixels = pd.concat((pixels, ppix))
            else:
                pixels = super(Process, self).__call__(observations)
                pixels['fieldName'] = field

        print('pixels', len(pixels))
        # self.display = True
        if self.display:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.suptitle('pixel coord.')
            ax.plot(pixels['pixRA'], pixels['pixDec'], 'r*')
            ax.plot(observations[self.RACol],
                    observations[self.DecCol], 'ko', mfc='None')
            ax.set_xlabel('pixRA [deg]')
            ax.set_ylabel('pixDec [deg]')
            plt.show()

        """
        pixRAmin = pixels['pixRA'].min()
        pixRAmax = pixels['pixRA'].max()
        deltaRA = (pixRAmax-pixRAmin)/self.nproc

        params_multi = np.arange(pixRAmin, pixRAmax, deltaRA).tolist()
        """
        print('number of pixels', len(pixels['healpixID'].unique()))
        # get E(B-V) for these pixels
        hpixes = np.unique(pixels[['healpixID', 'pixRA', 'pixDec']], axis=0)
        hpixes = ebv_pixels(hpixes)
        pixels = pixels.merge(hpixes,
                              left_on=['healpixID'], right_on=['healpixID'])

        # do not process pixels with high E(B-V)
        if self.ebvofMW_pixel > 0.:
            idx = pixels['ebvofMW'] <= self.ebvofMW_pixel
            pixels = pixels[idx]

        print('number of pixels - ebvofMW', len(pixels['healpixID'].unique()))
        params = {}
        params['observations'] = observations
        params['pixelmap'] = pixels
        params_multi = np.unique(pixels['healpixID'])
        nprocb = min(self.nproc, len(params_multi))
        if not self.display:
            multiproc(params_multi, params, self.process_metric, nprocb)
        else:
            self.process_metric(params_multi, params)

        """
        eval('multiproc(params_multi, params, self.process_metric_{}, nprocb)'.format(
            self.fieldType))
        """
        """
        self.nproc = 1
        params_multi = [0.]
        eval('self.process_metric_{}(params_multi, params)'.format(self.fieldType))
        """

    def process_metric(self, pixels, params, j=0, output_q=None):
        """
        Method to process metric on a set of pixels

        Parameters
        --------------
      Rarange: numpy array
          array with a set of area on the sky
        observations: numpy array
           array of observations
        j: int, opt
          index number of multiprocessing (default: 0)
        output_q: multiprocessing.Queue(), opt
          queue of the multiprocessing (default: None)
        """
        time_ref = time.time()

        observations = params['observations']
        pixelmap = params['pixelmap']

        procpix = ProcessPixels(
            self.metricList, j, outDir=self.outDir,
            dbName=self.dbName, saveData=self.saveData)

        valsdf = pd.DataFrame(pixelmap)
        ido = valsdf['healpixID'].isin(pixels)
        ppix = valsdf[ido]

        # print('processing', j, len(pixels), pixels)
        if self.display:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.suptitle('Observation and pixels')
            ax.plot(observations[self.RACol],
                    observations[self.DecCol], 'ko', mfc='None')
            ax.plot(pixelmap['pixRA'], pixelmap['pixDec'], 'r*')
            RAmin = pixelmap['pixRA'].min()
            RAmax = pixelmap['pixRA'].max()
            Decmin = pixelmap['pixDec'].min()
            Decmax = pixelmap['pixDec'].max()
            ax.plot([RAmin, RAmin], [Decmin, Decmax], color='r')
            ax.plot([RAmax, RAmax], [Decmin, Decmax], color='r')
            plt.show()

        print('hello', ppix)
        if len(ppix) > 0:
            # print('processing pixels bb', len(ppix))
            procpix(ppix, observations, self.npixels)
        else:
            print('No matching obs found!')

        print('end of processing for', j, time.time()-time_ref)

        if output_q is not None:
            return output_q.put({j: 1})
        else:
            return 1


def randomPixels(hIDs, npixels):
    """
    Function to choose a random set of pixels


    Returns
    ---------
    healpixIDs: list of randomly chosen healpix IDs

    """

    healpixIDs = random.sample(hIDs, npixels)

    return healpixIDs
