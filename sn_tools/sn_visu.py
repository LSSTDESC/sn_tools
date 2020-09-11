import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as manimation
import matplotlib.animation as anim
from descartes.patch import PolygonPatch
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
import time
import itertools
import numpy.lib.recfunctions as rf
import pandas as pd
from shapely import geometry
from shapely import affinity
import os
import glob
import healpy as hp

def fieldType(obs, RACol, DecCol):
    """
    Function estimateing the type of field - DD or WFD -
    according to the number of exposures

    Parameters
    ---------------
    obs: numpy array
      array of observations
    RACol: str
       RA col name
    DecCol: str
       Dec col name

    Returns
    -----------
    original numpy array with type appended

    """

    df = pd.DataFrame(np.copy(obs))

    df = df.round({RACol: 3, DecCol: 3})

    df['sumExposures'] = df.groupby([RACol, DecCol])[
        'numExposures'].transform('sum')

    idx = df['sumExposures'] >= 20

    df.loc[idx, 'fieldType'] = 'DD'
    df.loc[~idx, 'fieldType'] = 'WFD'

    return df.to_records(index=False)

def LSSTPointing(xc, yc, angle_rot=0., area=None, maxbound=None):
    """
    Function to build a focal plane for LSST

    Parameters
    ---------------

    xc: float
       x-position of the center FP (RA)
    yc: float
       y-position of the center FP (Dec)
    angle_rot: float, opt
      angle of rotation of the FP (default: 0.)
    area: float
      area for the FP (default: None)
    maxbound: float
      to reduce area  (default: None)
    Returns
    ----------
    LSST FP (geometry.Polygon)

    """

    """
    arr = [[3, 0], [12, 0], [12, 1], [13, 1], [13, 2], [14, 2], [14, 3], [15, 3],
           [15, 12], [14, 12], [14, 13], [13, 13], [
               13, 14], [12, 14], [12, 15],
           [3, 15], [3, 14], [2, 14], [2, 13], [1, 13], [1, 12], [0, 12],
           [0, 3], [1, 3], [1, 2], [2, 2], [2, 1], [3, 1]]
    """
    # this is a quarter of LSST FP (with corner rafts)
    arr = [[0.0, 7.5], [4.5, 7.5], [4.5, 6.5], [5.5, 6.5], [
        5.5, 5.5], [6.5, 5.5], [6.5, 4.5], [7.5, 4.5], [7.5, 0.0]]

    # this is a quarter of LSST FP (without corner rafts)
    arr = [[0.0, 7.5], [4.5, 7.5], [4.5, 4.5], [7.5, 4.5], [7.5, 0.0]]
    if maxbound is not None:
        arr = [[0.0, maxbound], [maxbound*4.5/7.5, maxbound], [maxbound*4.5 /
                                                               7.5, maxbound*4.5/7.5], [maxbound, maxbound*4.5/7.5], [maxbound, 0.0]]
    # symmetry I: y -> -y
    arrcp = list(arr)
    for val in arr[::-1]:
        if val[1] > 0.:
            arrcp.append([val[0], -val[1]])

    # symmetry II: x -> -x
    arr = list(arrcp)
    for val in arrcp[::-1]:
        if val[0] > 0.:
            arr.append([-val[0], val[1]])

    # build polygon
    poly_orig = geometry.Polygon(arr)

    # set area
    if area is not None:
        poly_orig = affinity.scale(poly_orig, xfact=np.sqrt(
            area/poly_orig.area), yfact=np.sqrt(area/poly_orig.area))

    # set rotation angle
    rotated_poly = affinity.rotate(poly_orig, angle_rot)

    return affinity.translate(rotated_poly,
                              xoff=xc-rotated_poly.centroid.x,
                              yoff=yc-rotated_poly.centroid.y)


def area(polylist):
    """Estimate area of a set of polygons (without overlaps)

    Parameters
    ----------------
    polylist: list
     list of polygons

    Returns
    -----------
    area: float
     area corresponding to this set of polygons.
    """

    bigpoly = unary_union(MultiPolygon(polylist))
    return bigpoly.area


class SnapNight:
    """
    class to get a snapshot of the (RA, Dec) pointings (LSST FP) observed map per night

    Parameters
    ---------------
    dbDir: str
       location dir of the db file
    dbName: str
        name of the db of interest. Extension: npy!
    nights: list(int)
      list of nights to display
    saveFig: bool, opt
      to save the figure result or not (default: False)
    areaTime: bool, opt
      to estimate area covered during the night (time consuming) (default: False)
    realTime: bool, opt
      if True results are displayed in 'real time' (default: False)

    """

    def __init__(self, dbDir, dbName, nights=[1,2,3], saveFig=False, areaTime=False, realTime=False):

        self.dbName = dbName
        self.saveFig = saveFig
        self.areaTime = areaTime
        self.realTime = realTime

        # Loading observations
        obs = np.load('{}/{}.npy'.format(dbDir, dbName))
        obs = renameFields(obs)
        obs.sort(order='observationStartMJD')

        # Select observations
        """
        idx = obs['night'] <= nightmax
        idx &= obs['night'] >= nightmin
        obs = obs[idx]
        """
        idx = np.isin(obs['night'],nights)
        obs = obs[idx]
        
        #for night in range(np.min(obs['night']), np.max(obs['night'])+1):
        for night in nights:
            # if night > np.min(obs['night']):
            #    self.ax.clear()
            idx = obs['night'] == night
            obs_disp = obs[idx]
            # get fieldtype(WFD or DDF)
            obs_disp = fieldType(obs_disp, 'fieldRA', 'fieldDec')
            self.frame()
            mjd0 = np.min(obs_disp['observationStartMJD'])
            polylist = []
            polyarea = []
            mjds = obs_disp['observationStartMJD']-mjd0
            nchanges = len(list(itertools.groupby(obs_disp['filter'])))-1
            iwfd = obs_disp['fieldType'] == 'WFD'
            selWFD = obs_disp[iwfd]
            nchanges_noddf = len(
                list(itertools.groupby(selWFD['filter'])))-1

            countfilter = {}
            for b in 'ugrizy':
                idx = selWFD['filter'] == b
                countfilter[b] = len(selWFD[idx])
            iddf = obs_disp['fieldType'] != 'WFD'
            nddf = len(obs_disp[iddf])

            nvisits = ''
            for key, vals in countfilter.items():
                nvisits += '{} : {} - '.format(key,int(vals))

            nvisits += 'ddf : {}'.format(nddf)
            
            self.fig.suptitle(
                'night {} \n filter changes: {}/{} \n {}'.format(night, nchanges_noddf, nchanges,nvisits))

            # area observed versus time
            for val in obs_disp:
                pointing = LSSTPointing(val['fieldRA'], val['fieldDec'],area=1.)
                if val['fieldType'] == 'WFD':
                    p = PolygonPatch(
                        pointing, facecolor=self.colors[val['filter']], edgecolor=self.colors[val['filter']])
                else:
                    p = PolygonPatch(
                        pointing, facecolor='k', edgecolor='k')
                self.ax.add_patch(p)
                # area observed versus time
                if self.areaTime:
                    polylist.append(pointing)
                    polyarea.append(area(polylist))

            if self.areaTime:
                self.ax2.plot(24.*mjds, polyarea, 'b.')
            if self.realTime:
                plt.draw()
                plt.pause(3.)
                if saveFig:
                    plt.savefig('{}_night_{}.png'.format(self.dbName, night))
                plt.close()

            # self.ax.clear()
            if self.areaTime:
                self.ax2.clear()

    def frame(self):
        """
        Frame to display the results

        """
        # Prepare the frame
        self.plotParams()
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        adjr = 0.85
        if self.areaTime:
            self.ax2 = self.fig.add_axes([0.71, 0.675, 0.20, 0.2])
            adjr = 0.7
        P1 = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(right=adjr)

        self.fig.canvas.draw()

        self.customize()

    def plotParams(self):
        """Define general parameters for display
        """

        # display parameters
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['legend.facecolor'] = 'w'
        plt.rcParams['figure.figsize'] = (11, 6)
        plt.rcParams['figure.titlesize'] = 12
        # switch on if you want dark background
        # plt.style.use('dark_background')
        self.colors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))

    def customize(self):
        """Define axis parameters for display
        """
        if self.areaTime:
            self.ax2.set_xlabel('Time [hour]', size=8)
            self.ax2.set_ylabel('area [deg2]', size=8)
            self.ax2.xaxis.set_tick_params(labelsize=8)
            self.ax2.yaxis.set_tick_params(labelsize=8)
            self.ax2.yaxis.tick_right()
            self.ax2.yaxis.set_label_position("right")

        self.ax.set_xlabel('RA [deg]')
        self.ax.set_ylabel('Dec [deg]')
        self.ax.set_xlim(0, 360.)
        self.ax.set_ylim(-90., 10.)

        colorfilters = []
        for band in self.colors.keys():
            colorfilters.append(mpatches.Patch(
                color=self.colors[band], label='{}'.format(band)))
        colorfilters.append(mpatches.Patch(
            color='k', label='ddf'))
        plt.legend(handles=colorfilters, loc='upper left',
                   bbox_to_anchor=(1., 0.5))


class CadenceMovie:
    """ Display obs footprint vs time

    Parameters
    ---------------
    dbDir : str
      database directory
    dbName : str
      database name
    nights : int,opt
      nights to process (default: 1)
    title : str,opt
         title of the movie (default : '')
    total: int,opt
       default: 600
    sub: int, opt
      default: 100
    fps: int, opt
      default: 24
    saveMovie: bool, opt
     save the movie as mp4 file (default: False)
    realTime: bool, opt
      display results in real-time (default: False)
    saveFig: bool, opt
      save fig at the end of each night (default: False)
    areaTime: bool, opt
      draw observed area vs time in an embedded histo (default: False)
    """

    def __init__(self, dbDir, dbName, nights=[1,2,3], title='', total=600, sub=100, fps=24, saveMovie=False, realTime=False, saveFig=False, areaTime=False):

        self.realTime = realTime
        self.plotParams()
        self.dbName = dbName
        self.saveFig = saveFig
        self.areaTime = areaTime
        # Loading observations
        obs = np.load('{}/{}.npy'.format(dbDir, dbName))
        obs = renameFields(obs)
        obs.sort(order='observationStartMJD')

        # prepare frame
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        adjr = 0.9
        self.ax2 = None
        if self.areaTime:
            self.ax2 = self.fig.add_axes([0.75, 0.675, 0.20, 0.2])
            adjr = 0.74
        P1 = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(right=adjr)
        self.fig.canvas.draw()
        self.customize()

        # Select observations
        """
        idx = obs['night'] <= nights
        obs = obs[idx]
        """
        
        idx = np.isin(obs['night'],nights)
        obs = obs[idx]

        #print('hello',np.unique(obs['night']))
        
        self.polylist = []
        if saveMovie:
            # Warning : to save the movie ffmpeg needs to be installed!
            print(manimation.writers.list())
            writer_type = 'pillow'
            extension = 'gif'
            """
            writer_type = 'ffmpeg'
            extension = 'mp4'
            """
            #FFMpegWriter = manimation.writers['ffmpeg']
            Writer = manimation.writers[writer_type]
            metadata = dict(title=title, artist='Matplotlib',
                            comment=title)
            #writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=6000)
            writer = Writer(fps=fps, metadata=metadata, bitrate=6000)
            #writer = anim.FFMpegWriter(fps=30, codec='hevc')
            Name_mp4 = '{}.{}'.format(title,extension)
            print('name for saving',Name_mp4)
            with writer.saving(self.fig, Name_mp4, 250):
                self.loopObs(obs, writer=writer)
        else:
            self.loopObs(obs)

    def showObs(self, obs, nchanges, nchanges_noddf, area, mjd0,countfilter,nddf):
        """ Display observation (RA, Dec) as time (MJD) evolves.

        Parameters
        ---------------
        obs: array
         array of observation
        nchanges: int
         number of filter changes
        area: float
         area observed (deg2)
        mjd0: float
         mjd of the first obs that night

        """

        # grab MJD and night
        mjd = obs['observationStartMJD']
        night = obs['night']
        moonPhase = obs['moonPhase']

        nvisits = ''
        for key, vals in countfilter.items():
            nvisits += '{} : {} - '.format(key,int(vals))

        nvisits += 'ddf : {}'.format(nddf)
        
        self.fig.suptitle(
            'night {} - MJD {} \n filter changes: {}/{} \n {}'.format(night, np.round(mjd, 3), nchanges_noddf, nchanges,nvisits))

        # LSST focal plane corresponding to this pointing
        pointing = LSSTPointing(obs['fieldRA'], obs['fieldDec'],area=1.)
        if obs['fieldType'] == 'WFD':
            p = PolygonPatch(
                pointing, facecolor=self.colors[obs['filter']], edgecolor=self.colors[obs['filter']])
        else:
            p = PolygonPatch(
                pointing, facecolor='k', edgecolor='k')
        self.ax.add_patch(p)

        # area observed versus time
        if self.areaTime:
            self.ax2.plot([24.*(mjd-mjd0)], [area], 'b.')

    def loopObs(self, obs, writer=None):
        """Loop on observations

        Parameters
        -----------
        obs: array
         array of observations
        writer: bool, opt
         write movie on a file
         default: None
        """

        #for night in range(np.min(obs['night']), np.max(obs['night'])+1):
        for night in np.unique(obs['night']):
            idx = obs['night'] == night
            obs_disp = obs[idx]
            # get fieldtype(WFD or DDF)
            obs_disp = fieldType(obs_disp, 'fieldRA', 'fieldDec')
            mjd0 = np.min(obs_disp['observationStartMJD'])
            if len(obs_disp) > 0:
                for k in range(len(obs_disp)):
                    # number of filter changes up to now
                    sel = obs_disp[:k]
                    nchanges = len(list(itertools.groupby(sel['filter'])))-1
                    ifw = sel['fieldType'] == 'WFD'
                    selWFD = sel[ifw]
                    nchanges_noddf = len(
                        list(itertools.groupby(selWFD['filter'])))-1
                    countfilter = {}
                    for b in 'ugrizy':
                        idx = selWFD['filter'] == b
                        countfilter[b] = len(selWFD[idx])
                    iddf = sel['fieldType'] != 'WFD'
                    nddf = len(sel[iddf])
                    
                    # show observations
                    if self.areaTime:
                        self.polylist.append(LSSTPointing(
                            obs_disp[k]['fieldRA'], obs_disp[k]['fieldDec']),area=1.)
                        self.showObs(obs_disp[k], nchanges, nchanges_noddf,
                                     area(self.polylist), mjd0,countfilter,nddf)
                    else:
                        #print(sel[['observationStartMJD','fieldRA','fieldDec','filter']])
                        self.showObs(obs_disp[k], nchanges, nchanges_noddf,
                                     0., mjd0,countfilter,nddf)
                    self.fig.canvas.flush_events()
                    if writer:
                        writer.grab_frame()
            if self.saveFig:
                plt.savefig('{}_night_{}.png'.format(self.dbName, night))
            # clear before next night
            self.ax.clear()
            if self.areaTime:
                self.ax2.clear()
            self.customize()
            self.polylist = []

    def plotParams(self):
        """Define general parameters for display
        """

        # display parameters
        if self.realTime:
            plt.ion()  # real-time mode
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['legend.facecolor'] = 'w'
        plt.rcParams['figure.figsize'] = (11, 6)
        plt.rcParams['figure.titlesize'] = 12
        # switch on if you want dark background
        # plt.style.use('dark_background')
        self.colors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))

    def customize(self):
        """Define axis parameters for display
        """
        if self.areaTime:
            self.ax2.set_xlabel('Time [hour]', size=8)
            self.ax2.set_ylabel('area [deg2]', size=8)
            self.ax2.xaxis.set_tick_params(labelsize=8)
            self.ax2.yaxis.set_tick_params(labelsize=8)
            self.ax2.yaxis.tick_right()
            self.ax2.yaxis.set_label_position("right")

        self.ax.set_xlabel('RA [deg]')
        self.ax.set_ylabel('Dec [deg]')
        self.ax.set_xlim(0, 360.)
        self.ax.set_ylim(-90., 20.)

        colorfilters = []
        for band in self.colors.keys():
            colorfilters.append(mpatches.Patch(
                color=self.colors[band], label='{}'.format(band)))
        colorfilters.append(mpatches.Patch(
            color='k', label='ddf'))
        plt.legend(handles=colorfilters, loc='upper left',
                   bbox_to_anchor=(1., 0.5))

def renameFields(tab):
    """
    Function to rename fields

    Parameters
    --------------
    tab: array
      original array of data

    Returns
    ---------
    array of data with modified field names

    """
    corresp = {}

    fillCorresp(tab, corresp, 'mjd', 'observationStartMJD')
    fillCorresp(tab, corresp, 'RA', 'fieldRA')
    fillCorresp(tab, corresp, 'Ra', 'fieldRA')
    fillCorresp(tab, corresp, 'Dec', 'fieldDec')
    fillCorresp(tab, corresp, 'band', 'filter')
    fillCorresp(tab, corresp, 'exptime', 'visitExposureTime')
    fillCorresp(tab, corresp, 'nexp', 'numExposures')

    # print(tab.dtype)

    rb = np.copy(tab)
    for vv, vals in corresp.items():
        rb = rf.drop_fields(rb, vv)
        if vv != 'band':
            rb = rf.append_fields(rb, vals, tab[vv])
        else:
            rb = rf.append_fields(rb, vals, tab[vv], dtypes='<U9')
        # rb = rf.rename_fields(rb, {vv: vals})

    # return rf.rename_fields(tab, corresp)
    return rb

def fillCorresp(tab, corres, vara, varb):
    """
    Function to fill a dict used to change colnams of a nupy array

    Parameters
    --------------
    tab: array
      original array of data
    corresp: dict
      keys: string, items: string
    vara: str
     colname in tab to change
    varb: str
     new colname to replace vara

    Returns
    ---------
    dict with str as keys and items
      correspondence vara<-> varb

    """

    if vara in tab.dtype.names and varb not in tab.dtype.names:
        corres[vara] = varb


class MoviePixels:
    
    def __init__(self, dbDir, dbName,saveMovie=False, realTime=False, saveFig=False, nightmin=0,nightmax=365,time_display=5):

        self.realTime = realTime
        self.dbName = dbName
        self.dbDir = dbDir
        self.saveFig = saveFig
        self.time_display = time_display
        self.nightmin = nightmin
        self.nightmax = nightmax
        self.saveMovie = saveMovie
        if saveMovie:
            self.saveFig = True
        
        plotDir_gen = 'Plots_Stat'
        self.plotDir = '{}/{}'.format(plotDir_gen,self.dbName)
        
        if not os.path.isdir(plotDir_gen):
            os.makedirs(plotDir_gen)

        if os.path.isdir(self.plotDir):
            os.rmdir(self.plotDir)
            
        os.makedirs(self.plotDir)
            
        
            
            
        # load observations
        data = self.loadPixels()

        # looping on data
        self.loopObs(data)
        
    def loadPixels(self):
        """
        Method to load the data

        Returns
        ----------
        numpy array with pixels infos

        """

        
        files = glob.glob('{}/{}/*.npy'.format(self.dbDir,self.dbName))

        pixel = None

        for fi in files:
            tab = np.load(fi,allow_pickle=True)
            if pixel is None:
                pixel = tab
            else:
                pixel = np.concatenate((pixel,tab))

        return pixel

    def loopObs(self,data):
        
        pixel_night = pd.DataFrame(np.unique(data['healpixID']),columns=['healpixID'])
        pixel_night['night_last'] = -1
        pixel_night = pixel_night.sort_values(by=['healpixID'])

        r = []
        filters = ['g','r','i']
        nVisits_min = dict(zip(filters,[1,1,1]))

        print('hello',self.nightmax)
        for night in range(self.nightmin,self.nightmax):
            
            idx = data['night']==night
            pix = pd.DataFrame(np.copy(data[idx]))
            for b in filters:
                pix[b] = pix['filter']==b
                pix[b] = pix[b].astype(int)

            groups = pix.groupby(['healpixID','night'])[filters].sum().reset_index()
            idx = True
            for f in filters:
                idx &= groups[f]<nVisits_min[f]
        
            pixsel_night = groups[~idx][['healpixID','night']]
            pixsel_night = pixsel_night.rename(columns={'night':'night_last'})
    
            idx = pixel_night['healpixID'].isin(pixsel_night['healpixID'])
            
            pixel_night = pd.concat((pixel_night[~idx],pixsel_night))

            iop = pixel_night['night_last']>-1
            iop &= (night-pixel_night['night_last'])<=10

            r.append((night,np.median(night-pixel_night[iop]['night_last']),len(pixel_night[iop])))
            res = np.rec.fromrecords(r, names=['night','deltaT_median','Npixels_observed'])
            self.plotNight(night,pixel_night,res)

            if self.realTime:
                plt.draw()
                plt.pause(self.time_display)
                plt.close()

        if self.saveMovie:
            self.makeMovie()
            
                
    def plotNight(self,night,pixels, stat,nside=64):

        pixels['deltaT'] = -1
        io = pixels['night_last']>-1
        pixels.loc[io,'deltaT'] = night-pixels.loc[io,'night_last']

    
        #fig, ax =plt.subplots(nrows=2,ncols=2,figsize=(15,12))
        fig = plt.figure(figsize=(15,12))
        fig.suptitle('{} - night {}'.format(self.dbName,int(night)))

        """
        ax_a = fig.add_axes([0.4,0.25,0.5,0.5])
        ax_b = fig.add_axes([0.1,0.1,0.25,0.25])
        ax_c = fig.add_axes([0.1,0.4,0.25,0.25])
        ax_d = fig.add_axes([0.1,0.7,0.25,0.25])
        """

        ax_a = fig.add_axes([0.25,0.4,0.5,0.5])
        ax_b = fig.add_axes([0.1,0.1,0.25,0.25])
        ax_c = fig.add_axes([0.4,0.1,0.25,0.25])
        ax_d = fig.add_axes([0.7,0.1,0.25,0.25])
        
        #axa = ax[0,1]
        axa = ax_d
        plt.sca(axa)
        idx = pixels['night_last']>-1
        idx &= pixels['deltaT']<=10
        
        axa.hist(pixels[idx]['deltaT'],histtype='step')
        axa.set_xlabel('$\Delta$T')
        axa.set_ylabel('Number of entries')
        #axa.set_ylabel('Number of entries',rotation=270)
        #axa.yaxis.set_label_position("right")
        #axa.yaxis.tick_right()

        #axa = ax[1,0]
        axa = ax_c
        plt.sca(axa)
        axa.plot(stat['night'],stat['deltaT_median'])
        axa.set_ylabel('Median $\Delta$T')
        axa.set_xlabel('night')

        #axa = ax[1,1]
        axa = ax_b
        plt.sca(axa)
        axa.plot(stat['night'],stat['Npixels_observed'])
        axa.set_ylabel('Number of pixels observed')
        axa.set_xlabel('night')
        
        #print(pixels)
        #idtest = pixels['healpixID'].isin([39143,39149,39154])
        #print('test',night,pixels[idtest])

        #axa = ax[0,0]
        axa = ax_a
        plt.sca(axa)
        npixels = hp.nside2npix(nside)
        xmin = -1.e-8
        xmax = np.max([np.max(pixels['deltaT']),1])
    
        norm = plt.cm.colors.Normalize(xmin, xmax)
        #cmap = plt.get_cmap('jet', int(xmax))
        n = int(xmax)+1
        n = 9
        from_list = matplotlib.colors.LinearSegmentedColormap.from_list
        cmap = from_list(None, plt.cm.Set1(range(0,n)), n)
        cmap.set_under('w')

        hpxmap = np.zeros(npixels, dtype=np.int)
        hpxmap = np.full(hpxmap.shape, -2)
        hpxmap[pixels['healpixID']] = pixels['deltaT'].astype(int)

        #print('hello ',xmin,xmax)

        dd = '$\Delta$T = current night-last obs night (gri)'
        hp.mollview(hpxmap,nest=True,cmap=cmap,
                    min=xmin, max=n,norm=norm,cbar=False,
                    title=dd,hold=True,badcolor='white',xsize=1600)

            
        hp.graticule()
        ax = plt.gca()
        image = ax.get_images()[0]
        cbar= fig.colorbar(image, ax=ax, ticks=range(0,n+1), orientation='horizontal')
        #cbar = fig.colorbar(ax[0,0], ticks=range(0,n), orientation='horizontal')  # set some values to ticks

        labels = list(range(0,n))
    
        tick_label = list(map(str, labels))
    
        tick_label[-1] = '>{}'.format(tick_label[-2])
        #print(tick_label)
        cbar.ax.set_xticklabels([])
        cbar.ax.tick_params(size=0)
        for j, lab in enumerate(tick_label):
            cbar.ax.text(labels[j]+0.5, -10.,lab)

        if self.saveFig:
            plt.savefig('{}/{}_{}.png'.format(self.plotDir,self.dbName,str(night).zfill(3)))

    def makeMovie(self):

        dirFigs = self.plotDir
        cmd = 'ffmpeg -r 3 -f image2 -s 1920x1080 -i {}/{}_%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}.mp4 -y'.format(self.plotDir,self.dbName,self.dbName)
        os.system(cmd)
