from sn_tools.sn_obs import DataToPixels, ProcessPixels, renameFields, patchObs
from sn_tools.sn_io import getObservations
import time
import numpy as np
import pandas as pd
import multiprocessing
import glob
import random


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

    """

    def __init__(self, dbDir, dbName, dbExtens,
                 fieldType, nside,
                 RAmin, RAmax,
                 Decmin, Decmax,
                 saveData, remove_dithering,
                 outDir, nprocs, metricList,
                 pixelmap_dir='', npixels=0,
                 nclusters=5, radius=4.):

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        self.fieldType = fieldType
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

        # this is to take multiproc into account

        observations, patches = self.load()

        # print('coordinates for patch',RAmin,RAmax,Decmin,Decmax)
        # select observation in this area
        delta_coord = 5.
        idx = observations[self.RACol] >= RAmin-delta_coord
        idx &= observations[self.RACol] < RAmax+delta_coord
        # idx &= observations[self.DecCol] >= Decmin-delta_coord
        # idx &= observations[self.DecCol] < Decmax+delta_coord

        # print('before', len(observations), RAmin, RAmax, Decmin, Decmax)
        obs = observations[idx]

        if RAmin < delta_coord:
            idx = observations[self.RACol] >= 360.-delta_coord
            obs = np.concatenate((obs, observations[idx]))

        if RAmax >= 360.-delta_coord:
            idx = observations[self.RACol] <= delta_coord
            obs = np.concatenate((obs, observations[idx]))

        """
        import matplotlib.pyplot as plt
        plt.plot(observations[self.RACol],observations[self.DecCol],'ko')
        plt.show()
        """

        if self.pixelmap_dir == '':
            self.multiprocess(patches, obs, func=self.processPatch)
        else:
            # load the pixel maps
            print('pixel map loading', self.pixelmap_dir, self.fieldType,
                  self.nside, self.dbName, self.npixels)
            search_path = '{}/{}/{}_{}_nside_{}_{}_{}_{}_{}.npy'.format(self.pixelmap_dir,
                                                                        self.dbName, self.dbName,
                                                                        self.fieldType, self.nside,
                                                                        self.RAmin, self.RAmax,
                                                                        self.Decmin, self.Decmax)
            pixelmap_files = glob.glob(search_path)
            if not pixelmap_files:
                print('Severe problem: pixel map does not exist!!!!')
            else:
                self.pixelmap = np.load(pixelmap_files[0])
                if self.npixels == -1:
                    self.npixels = len(
                        np.unique(self.pixelmap['healpixID']))
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

        # loading observations

        observations = getObservations(self.dbDir, self.dbName, self.dbExtens)

        # rename fields

        observations = renameFields(observations)

        self.RACol = 'fieldRA'
        self.DecCol = 'fieldDec'

        if 'RA' in observations.dtype.names:
            self.RACol = 'RA'
            self.DecCol = 'Dec'

        observations, patches = patchObs(observations, self.fieldType,
                                         self.dbName,
                                         self.nside,
                                         self.RAmin, self.RAmax,
                                         self.Decmin, self.Decmax,
                                         self.RACol, self.DecCol,
                                         display=False,
                                         nclusters=self.nclusters, radius=self.radius)

        return observations, patches

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
            print('go for multiprocessing', j, func, healpixels[ida:idb])
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

        procpix = ProcessPixels(
            self.metricList, j, outDir=self.outDir, dbName=self.dbName, saveData=self.saveData)

        print('pointings', len(pointings))

        for index, pointing in pointings.iterrows():
            ipoint += 1
            # print('pointing', ipoint)

            # print('there man', np.unique(observations[[self.RACol, self.DecCol]]), pointing[[
            #      'RA', 'Dec', 'radius_RA', 'radius_Dec']])
            # get the pixels
            pixels = datapixels(observations, pointing['RA'], pointing['Dec'],
                                pointing['radius_RA'], pointing['radius_Dec'], self.remove_dithering, display=False)

            if pixels is None:
                return

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

            """
            print('cut', pointing['RA'], pointing['radius_RA'],
                  pointing['Dec'], pointing['radius_Dec'])
            """

            # datapixels.plot(pixels)
            # print('after selection', len(pixels_run), datapixels.observations)

            npixels = len(np.unique(pixels_run['healpixID']))

            if self.npixels > -1:
                if npixels >= self.npixels:
                    # too many pixel found: should choose self.npixels among this
                    random_pixels = self.randomPixels(pixels_run, self.npixels)
                    idx = np.in1d(pixels_run['healpixID'], random_pixels)
                    pixels_run = pixels_run[idx]

            procpix(pixels_run, datapixels.observations, ipoint)

        # print('end of processing for', j, time.time()-time_ref)

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
        procpix = ProcessPixels(
            self.metricList, j, outDir=self.outDir, dbName=self.dbName, saveData=self.saveData)

        # print('continuing')
        valsdf = pd.DataFrame(self.pixelmap)
        ido = valsdf['healpixID'].isin(pixels)
        if len(valsdf[ido]) > 0:
            minDec = np.min(valsdf[ido]['pixDec'])
            maxDec = np.max(valsdf[ido]['pixDec'])
            minDecobs = np.min(observations[self.DecCol])
            maxDecobs = np.max(observations[self.DecCol])
            print('processing pixels', len(
                valsdf[ido]), minDec, maxDec, minDecobs, maxDecobs)
            procpix(valsdf[ido], np.copy(observations), self.npixels)
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
