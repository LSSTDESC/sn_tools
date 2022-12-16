
from sn_tools.sn_obs import renameFields, patchObs
from sn_tools.sn_obs import ProcessPixels
from sn_tools.sn_io import colName
from sn_tools.sn_obs import getObservations, get_obs
from sn_tools.sn_utils import multiproc
import time
import numpy as np
import pandas as pd
import multiprocessing
import glob
import random


class Process_deprecated:
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
    healpixIDs: list, opt
      list of healpixels to process (default: [])

    """

    def __init__(self, dbDir, dbName, dbExtens,
                 fieldType, fieldName, nside,
                 RAmin, RAmax,
                 Decmin, Decmax,
                 saveData, remove_dithering,
                 outDir, nprocs, metricList,
                 pixelmap_dir='', npixels=0,
                 nclusters=5, radius=4., healpixIDs=[]):

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        self.fieldType = fieldType
        self.fieldName = fieldName
        self.nside = nside
        self.RAmin = RAmin
        self.RAmax = RAmax
        self.Decmin = Decmin
        self.Decmax = Decmax
        self.saveData = saveData
        self.remove_dithering = remove_dithering
        self.outDir = outDir
        self.nprocs = nprocs
        self.metricList = metricList
        self.pixelmap_dir = pixelmap_dir
        self.npixels = npixels
        self.nclusters = nclusters
        self.radius = radius
        self.healpixIDs = healpixIDs

        assert(self.RAmin <= self.RAmax)

        # loading observations

        obs, patches = self.load()

        if 'DD' in self.fieldType:
            print('DD clusters', patches[[
                  'fieldName', 'RA', 'Dec', 'radius_RA', 'radius_Dec']])

        # print('in Process - observations loaded')
        """
        import matplotlib.pyplot as plt
        plt.plot(observations[self.RACol],observations[self.DecCol],'ko')
        plt.show()
        """

        if self.pixelmap_dir == '':
            nnproc = self.nprocs
            if 'DD' in self.fieldType:
                nnproc = len(patches)

            self.pixelmap = self.multiprocess_getpixels(
                patches, obs, func=self.processPatch, nnproc=nnproc).to_records(index=False)
        else:
            # load the pixel maps
            print('pixel map loading', self.pixelmap_dir, self.fieldType,
                  self.nside, self.dbName, self.npixels)
            search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_{}'.format(self.pixelmap_dir,
                                                                    self.dbName, self.dbName,
                                                                    self.fieldType, self.nside,
                                                                    self.RAmin, self.RAmax,
                                                                    self.Decmin, self.Decmax)
            if self.fieldType == 'DD':
                search_path += '_{}'.format(self.fieldName)
            # else:
            #    search_path += '_WFD'
            search_path += '.npy'
            pixelmap_files = glob.glob(search_path)
            if not pixelmap_files:
                print('Severe problem: pixel map does not exist!!!!', search_path)
            else:
                self.pixelmap = np.load(pixelmap_files[0], allow_pickle=True)

        if self.npixels == -1:
            self.npixels = len(
                np.unique(self.pixelmap['healpixID']))
        random_pixels = self.healpixIDs
        if not self.healpixIDs:
            random_pixels = self.randomPixels(self.pixelmap, self.npixels)
        print('number of pixels to process', len(random_pixels))

        self.multiprocess(random_pixels, obs,
                          func=self.procix)

    def load(self):
        """
        Method to load observations and patches dims on the sky

        Returns
        ------------
       observations: numpy array
         numpy array with observations
       patches: pandas df
        patches coordinates on the sky
        """
        observations = getObservations(self.dbDir, self.dbName, self.dbExtens)

        names = observations.dtype.names
        self.RACol = colName(names, ['fieldRA', 'RA', 'Ra'])
        self.DecCol = colName(names, ['fieldDec', 'Dec'])

        # print('dtype',observations.dtype.names)
        # select observation in this area
        delta_coord = 5.
        idx = observations[self.RACol] >= self.RAmin-delta_coord
        idx &= observations[self.RACol] < self.RAmax+delta_coord
        # idx &= observations[self.DecCol] >= Decmin-delta_coord
        # idx &= observations[self.DecCol] < Decmax+delta_coord

        obs = observations[idx]

        if self.RAmin < delta_coord:
            idk = observations[self.RACol] >= 360.-delta_coord
            obs = np.concatenate((obs, observations[idk]))

        if self.RAmax >= 360.-delta_coord:
            idk = observations[self.RACol] <= delta_coord
            obs = np.concatenate((obs, observations[idk]))

        # rename fields
        observations = renameFields(obs)
        names = observations.dtype.names
        self.RACol = colName(names, ['fieldRA', 'RA', 'Ra'])
        self.DecCol = colName(names, ['fieldDec', 'Dec'])

        """
        self.RACol = 'fieldRA'
        self.DecCol = 'fieldDec'

        if 'RA' in observations.dtype.names:
            self.RACol = 'RA'
            self.DecCol = 'Dec'
        """
        observations, patches = patchObs(observations, self.fieldType, self.fieldName,
                                         self.dbName,
                                         self.nside,
                                         self.RAmin, self.RAmax,
                                         self.Decmin, self.Decmax,
                                         self.RACol, self.DecCol,
                                         display=False,
                                         nclusters=self.nclusters, radius=self.radius)

        return observations, patches

    def multiprocess_getpixels(self, patches, observations, func, nnproc):
        """
        Method to grab (healpix) pixels matching observations

        Parameters
        ---------------
        patches: pandas df
          patches coordinates on the sky
        observations: numpy array
         numpy array with observations
        func: the function to apply for multiprocessing
        nnproc: int
          number of procs to use

        Returns
        ----------
        numpy array with a list of pixels and a link to observations

        """

        timeref = time.time()

        healpixels = patches
        npixels = int(len(healpixels))

        tabpix = np.linspace(0, npixels, nnproc+1, dtype='int')
        result_queue = multiprocessing.Queue()
        nmulti = len(tabpix)-1

        # multiprocessing
        for j in range(nmulti):
            ida = tabpix[j]
            idb = tabpix[j+1]

            print('go for multiprocessing in getpixels',
                  j, func, len(healpixels[ida:idb]))
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=func, args=(
                healpixels[ida:idb], observations, j, result_queue))

            p.start()

        # get the results
        resultdict = {}
        for i in range(nmulti):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        restot = pd.DataFrame()
        # gather the results
        for key, vals in resultdict.items():
            restot = pd.concat((restot, vals), sort=False)
        return restot

    def multiprocess(self, patches, observations, func):
        """
        Method to perform multiprocessing of metrics

        Parameters
        ---------------
        patches: pandas df
          patches coordinates on the sky
        observations: numpy array
         numpy array with observations

        """

        timeref = time.time()

        healpixels = patches
        npixels = int(len(healpixels))

        tabpix = np.linspace(0, npixels, self.nprocs+1, dtype='int')
        # print(tabpix, len(tabpix))
        result_queue = multiprocessing.Queue()

        # print('in multi process', npixels, self.nprocs)
        # multiprocessing
        for j in range(len(tabpix)-1):

            ida = tabpix[j]
            idb = tabpix[j+1]

            # print('Field', j, len(healpixels[ida:idb]),len(observations))

            # field = healpixels[ida:idb]

            """
            idx = field['fieldName'] == 'SPT'
            if len(field[idx]) > 0:
            """
            print('go for multiprocessing', j, func, len(healpixels[ida:idb]))
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=func, args=(
                healpixels[ida:idb], observations, j, result_queue))
            print('starting')
            # p.daemon = True
            p.start()

    def processPatch(self, pointings, observations, j=0, output_q=None):
        """
        Method to process a patch

        Parameters
        --------------
        pointings: numpy array
          array with a set of area on the sky
        observations: numpy array
           array of observations
        j: int, opt
          index number of multiprocessing (default: 0)
        output_q: multiprocessing.Queue(), opt
          queue of the multiprocessing (default: None)
        """

        # print('processing area', j, pointings)

        time_ref = time.time()
        ipoint = 1

        datapixels = DataToPixels(
            self.nside, self.RACol, self.DecCol, self.outDir, self.dbName)

        print('there man', self.outDir, self.dbName)
        procpix = ProcessPixels(
            self.metricList, j, outDir=self.outDir, dbName=self.dbName, saveData=self.saveData)

        # print('pointings', len(pointings))

        for index, pointing in pointings.iterrows():
            ipoint += 1
            # print('pointing', ipoint)

            # print('there man', np.unique(observations[[self.RACol, self.DecCol]]), pointing[[
            #      'RA', 'Dec', 'radius_RA', 'radius_Dec']])
            # get the pixels
            pixels = datapixels(observations, pointing['RA'], pointing['Dec'],
                                pointing['radius_RA'], pointing['radius_Dec'], self.remove_dithering, display=False)

            if pixels is None:
                if output_q is not None:
                    output_q.put({j: pixels})
                    return pixels
                else:
                    return pixels

            # select pixels that are inside the original area
            pixels_run = pixels
            """
            print('number of pixels', len(pixels_run), pixels)
            import matplotlib.pyplot as plt
            plt.plot(pixels['pixRA'], pixels['pixDec'], 'ko')
            zonex = [pointing['minRA'], pointing['maxRA'],
                     pointing['maxRA'], pointing['minRA'], pointing['minRA']]
            zoney = [pointing['minDec'], pointing['minDec'],
                     pointing['maxDec'], pointing['maxDec'], pointing['minDec']]
            plt.plot(zonex, zoney)
            plt.show()
            """

            if self.fieldType != 'Fake' and self.fieldType != 'DD':
                """
                idx = (pixels['pixRA']-pointing['RA']
                       ) >= pointing['radius_RA']/2.
                idx &= (pixels['pixRA']-pointing['RA']
                        ) < pointing['radius_RA']/2.
                idx &= (pixels['pixDec']-pointing['Dec']
                        ) >= pointing['radius_Dec']/2.
                idx &= (pixels['pixDec']-pointing['Dec']
                        ) < pointing['radius_Dec']/2.
                """
                idx = pixels['pixRA'] > pointing['minRA']
                idx &= pixels['pixRA'] < pointing['maxRA']
                idx &= pixels['pixDec'] > pointing['minDec']
                idx &= pixels['pixDec'] < pointing['maxDec']

                pixels_run = pixels[idx]

        print('Grabing pixels - end of processing for', j, time.time()-time_ref)

        if output_q is not None:
            output_q.put({j: pixels_run})
        else:
            return pixels_run

    def procix(self, pixels, observations, j=0, output_q=None):
        """
        Method to process a pixel

        Parameters
        --------------
        pointings: numpy array
          array with a set of area on the sky
        observations: numpy array
           array of observations
        j: int, opt
          index number of multiprocessing (default: 0)
        output_q: multiprocessing.Queue(), opt
          queue of the multiprocessing (default: None)
        """
        time_ref = time.time()
        # print('there we go instance procpix')

        print('processing pixel', pixels, len(
            observations), self.outDir, self.dbName)
        procpix = ProcessPixels(
            self.metricList, j, outDir=self.outDir, dbName=self.dbName, saveData=self.saveData)

        # print('continuing')
        valsdf = pd.DataFrame(self.pixelmap)
        ido = valsdf['healpixID'].isin(pixels)
        ppix = valsdf[ido]
        if len(ppix) > 0:
            minDec = np.min(ppix['pixDec'])
            maxDec = np.max(ppix['pixDec'])
            minDecobs = np.min(observations[self.DecCol])
            maxDecobs = np.max(observations[self.DecCol])
            print('processing pixels', len(ppix),
                  minDec, maxDec, minDecobs, maxDecobs)
            procpix(ppix, np.copy(observations), self.npixels)
        else:
            print('pb here no data in ', pixels)

        print('end of processing for', j, time.time()-time_ref)

    def randomPixels(self, pixelmap, npixels):
        """
        Method to choose a random set of pixels


        Returns
        ---------
       healpixIDs: list of randomly chosen healpix IDs

        """

        hIDs = np.unique(pixelmap['healpixID'])
        healpixIDs = random.sample(hIDs.tolist(), npixels)

        return healpixIDs


class Process:
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
                 fieldType='', fieldName='', nside=128,
                 RAmin=0., RAmax=360.,
                 Decmin=-80., Decmax=80,
                 saveData=False, remove_dithering=False,
                 outDir='', nproc=1, metricList=[],
                 pixelmap_dir='', npixels=0,
                 VRO_FP='circular', project_FP='gnomonic', telrot=0.,
                 radius=4., pixelList='None', display=False, **kwargs):

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        self.fieldType = fieldType
        self.fieldName = fieldName
        self.nside = nside
        self.RAmin = RAmin
        self.RAmax = RAmax
        self.Decmin = Decmin
        self.Decmax = Decmax
        self.saveData = saveData
        self.remove_dithering = remove_dithering
        self.outDir = outDir
        self.nproc = nproc
        self.metricList = metricList
        self.pixelmap_dir = pixelmap_dir
        self.npixels = npixels
        self.radius = radius
        self.VRO_FP = VRO_FP
        self.project_FP = project_FP
        self.telrot = telrot
        self.display = display

        assert(self.RAmin <= self.RAmax)

        observations = get_obs(fieldType, dbDir,
                               dbName, dbExtens)

        names = observations.dtype.names
        self.RACol = colName(names, ['fieldRA', 'RA', 'Ra'])
        self.DecCol = colName(names, ['fieldDec', 'Dec'])

        # select observations
        obs = self.select_obs(observations, [fieldName], RAmin, RAmax)

        # load healpixIDs if not None
        healpixIDs = []
        if pixelList != 'None':
            hh = pd.read_csv(pixelList)
            healpixIDs = hh['healpixID'].to_list()

        print('hello', nside)
        # process
        self.processIt(obs, npixels, healpixIDs)

    def select_obs(self, observations, fieldName, RAmin, RAmax):
        """
        Method to select observations

        Parameters
        ---------------
        observations: array
          data to process
        fieldName: str
          fieldName to select

        """
        if self.fieldType == 'DD':
            idx = np.in1d(observations['note'], fieldName)
            observations = observations[idx]
            return self.select_zone(observations, RAmin, RAmax, self.RACol)

        if self.fieldType == 'WFD':
            return self.select_zone(observations, RAmin, RAmax, self.RACol)

        if self.fieldType == 'Fake':
            return observations

    def select_zone(self, data, RAmin, RAmax, RACol, Decmin=None, Decmax=None, DecCol='Dec', delta_coord=5.):
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

        obs = data[idx]

        if RAmin < delta_coord:
            idk = data[RACol] >= 360.-delta_coord
            obs = np.concatenate((obs, data[idk]))

        if RAmax >= 360.-delta_coord:
            idk = data[RACol] <= delta_coord
            obs = np.concatenate((obs, data[idk]))

        return obs

    def processIt(self, observations, npixels=-1, healpixIDs=[]):
        """
        Method to process a field

        Parameters
        --------------
        fieldName: list(str)
          list of fields to process
        npixels: int, opt
          number of pixels to process (default: -1=all)
        healpixIDs: list(int), opt
          list of hpixId to process (default: [])

        """

        pixels = self.get_pixels_field(observations)

        if self.display:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(pixels['pixRA'], pixels['pixDec'], 'r*')
            ax.plot(observations[self.RACol],
                    observations[self.DecCol], 'ko', mfc='None')
            ax.set_xlabel('RA [deg]')
            ax.set_ylabel('Dec [deg]')
            plt.show()

        pixRAmin = pixels['pixRA'].min()
        pixRAmax = pixels['pixRA'].max()
        deltaRA = (pixRAmax-pixRAmin)/self.nproc

        params_multi = np.arange(pixRAmin, pixRAmax, deltaRA).tolist()

        params = {}
        params['observations'] = observations
        params['pixelmap'] = pixels
        params['deltaRA'] = deltaRA

        if self.fieldType == 'DD':
            paramsb = dict(zip(['obs', 'pixels'], [observations, pixels]))
            print('getting pixels')
            time_ref = time.time()
            from sn_tools.sn_obs import DataToPixels
            datapixels = DataToPixels(
                self.nside, self.project_FP, self.VRO_FP, RACol=self.RACol, DecCol=self.DecCol, telrot=self.telrot, nproc=self.nproc)
            pixels = datapixels(observations, pixels)
            params_multi = np.unique(pixels['healpixID'])
            print('after pixels', time.time()-time_ref, len(pixels))
            params['pixelmap'] = pixels
            #pixels.to_hdf('allpixels.hdf5', key='pixels')

        # get the list of pixels Id to process
        pixel_Ids = self.pixelList(npixels, pixels, healpixIDs)

        nprocb = self.nproc
        if self.fieldType == 'DD':
            params_multi = pixel_Ids
            nprocb = min(self.nproc, len(pixel_Ids))

        eval('multiproc(params_multi, params, self.process_metric_{}, nprocb)'.format(
            self.fieldType))
        """
        self.nproc = 1
        params_multi = [0.]
        eval('self.process_metric_{}(params_multi, params)'.format(self.fieldType))
        """

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
            # get pixels corresponding to observations

            mean_RA = np.mean(observations[RACol])
            mean_Dec = np.mean(observations[DecCol])
            width_RA = np.max(observations[RACol]) - \
                np.min(observations[RACol])
            width_Dec = np.max(observations[DecCol]) - \
                np.min(observations[DecCol])

        if self.fieldType == 'WFD':
            width_RA = self.RAmax-self.RAmin
            width_Dec = self.Decmax-self.Decmin
            mean_RA = np.mean([self.RAmin, self.RAmax])
            mean_Dec = np.mean([self.Decmin, self.Decmax])

        pixels = self.gime_pixels(
            mean_RA, mean_Dec, np.max([width_RA, width_Dec]))

        if self.fieldType == 'WFD':
            pixels = self.select_zone(
                pixels, self.RAmin, self.RAmax, 'pixRA', self.Decmin, self.Decmax, 'pixDec', 0.)
            print('pixels', len(pixels))

        return pixels

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

        random_pixels = self.randomPixels(pixList, npixels)

        return random_pixels

    def path_to_pixels(self):
        """
        method to grab pixel files

        Returns
        -----------
        full search path

        """

        search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_{}'.format(self.pixelmap_dir,
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
        import numpy.lib.recfunctions as rf
        # get central pixel ID
        healpixID = hp.ang2pix(self.nside, RA,
                               Dec, nest=True, lonlat=True)

        # get nearby pixels
        vec = hp.pix2vec(self.nside, healpixID, nest=True)
        healpixIDs = hp.query_disc(
            self.nside, vec, np.deg2rad(width)+np.deg2rad(3.5), inclusive=inclusive, nest=True)

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
            self.nside, self.project_FP, self.VRO_FP, RACol=self.RACol, DecCol=self.DecCol, telrot=self.telrot, nproc=self.nproc)
        pixels = datapixels(obs, pixels, display=display)

        print('after pixel/obs matching', time.time()-time_ref)

        if output_q is not None:
            return output_q.put({j: pixels})
        else:
            return pixels

    def process_metric_WFD(self, RArange, params, j=0, output_q=None):
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
        deltaRA = params['deltaRA']

        procpix = ProcessPixels(
            self.metricList, j, outDir=self.outDir, dbName=self.dbName, saveData=self.saveData)

        RAmin = RArange[0]
        RAmax = RAmin+deltaRA

        # requested at the border
        if RAmin <= 10.:
            idx = observations[self.RACol] >= 340.
            sel = observations[idx]
            sel[self.RACol] -= 360.
            observations = np.concatenate(
                (sel, observations[~idx]))

        if self.display:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.suptitle('Observations and pixels')
            ax.plot(observations[self.RACol],
                    observations[self.DecCol], 'ko', mfc='None')
            ax.plot(pixelmap['pixRA'], pixelmap['pixDec'], 'r*')
            Decmin = pixelmap['pixDec'].min()
            Decmax = pixelmap['pixDec'].max()
            ax.plot([RAmin, RAmin], [Decmin, Decmax], color='r')
            ax.plot([RAmax, RAmax], [Decmin, Decmax], color='r')
            plt.show()

        pixels = self.select_zone(
            pixelmap, RAmin, RAmax, 'pixRA', delta_coord=0.)

        RAmin_pix = pixels['pixRA'].min()
        RAmax_pix = pixels['pixRA'].max()

        obs = self.select_zone(
            observations, RAmin_pix, RAmax_pix, self.RACol, delta_coord=3.3)

        print('processing pixel', RArange, len(obs), len(pixels))

        params = {}
        params['obs'] = obs
        params['pixels'] = pixels
        params['display'] = self.display

        time_ref = time.time()
        ppix = self.build_pixels(pixels['healpixID'], params)
        print('matching obs/pixel done', time.time()-time_ref)
        # ppix.to_hdf('allpixels_test.hdf5', key='pixels')

        if len(ppix) > 0:
            print('processing pixels', len(ppix))
            procpix(ppix, obs, self.npixels)
        else:
            print('No matching obs found!')

        print('end of processing for', j, time.time()-time_ref)

        if output_q is not None:
            return output_q.put({j: 1})
        else:
            return 1

    def process_metric_DD(self, pixels, params, j=0, output_q=None):
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
            self.metricList, j, outDir=self.outDir, dbName=self.dbName, saveData=self.saveData)

        valsdf = pd.DataFrame(pixelmap)
        ido = valsdf['healpixID'].isin(pixels)
        ppix = valsdf[ido]

        if self.display:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(observations[self.RACol],
                    observations[self.DecCol], 'ko', mfc='None')
            ax.plot(pixelmap['pixRA'], pixelmap['pixDec'], 'r*')
            Decmin = pixelmap['pixDec'].min()
            Decmax = pixelmap['pixDec'].max()
            ax.plot([RAmin, RAmin], [Decmin, Decmax], color='r')
            ax.plot([RAmax, RAmax], [Decmin, Decmax], color='r')
            plt.show()

        if len(ppix) > 0:
            print('processing pixels', len(ppix))
            procpix(ppix, observations, self.npixels)
        else:
            print('No matching obs found!')

        print('end of processing for', j, time.time()-time_ref)

        if output_q is not None:
            return output_q.put({j: 1})
        else:
            return 1

    def procix_deprecated(self, pixels, observations, j=0, output_q=None):
        """
        Method to process a pixel

        Parameters
        --------------
        pointings: numpy array
          array with a set of area on the sky
        observations: numpy array
           array of observations
        j: int, opt
          index number of multiprocessing (default: 0)
        output_q: multiprocessing.Queue(), opt
          queue of the multiprocessing (default: None)
        """
        time_ref = time.time()
        # print('there we go instance procpix')

        print('processing pixel', pixels, len(observations))
        procpix = ProcessPixels(
            self.metricList, j, outDir=self.outDir, dbName=self.dbName, saveData=self.saveData)

        # print('continuing')
        valsdf = pd.DataFrame(self.pixelmap)
        ido = valsdf['healpixID'].isin(pixels)
        ppix = valsdf[ido]
        if len(ppix) > 0:
            minDec = np.min(ppix['pixDec'])
            maxDec = np.max(ppix['pixDec'])
            minDecobs = np.min(observations[self.DecCol])
            maxDecobs = np.max(observations[self.DecCol])
            print('processing pixels', len(ppix),
                  minDec, maxDec, minDecobs, maxDecobs)
            procpix(ppix, np.copy(observations), self.npixels)
        else:
            print('pb here no data in ', pixels)

        print('end of processing for', j, time.time()-time_ref)

    def randomPixels(self, hIDs, npixels):
        """
        Method to choose a random set of pixels


        Returns
        ---------
       healpixIDs: list of randomly chosen healpix IDs

        """

        healpixIDs = random.sample(hIDs, npixels)

        return healpixIDs
