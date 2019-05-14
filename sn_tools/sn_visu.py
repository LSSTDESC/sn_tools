import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as manimation
import matplotlib.animation as anim
from descartes.patch import PolygonPatch
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from sn_tools.sn_obs import LSSTPointing, renameFields
import time
import itertools


def area(polylist):
    """Estimate area of a set of polygons (without overlaps)

    Parameters
    -----------
    polylist: list
     list of polygons

    Returns
    -------
    area: float
     area corresponding to this set of polygons.
    """

    bigpoly = unary_union(MultiPolygon(polylist))
    return bigpoly.area


class SnapNight:
    def __init__(self, dbDir, dbName, nights=1, saveFig=False, areaTime=False, realTime=False):

        self.dbName = dbName
        self.saveFig = saveFig
        self.areaTime = areaTime
        self.realTime = realTime

        # Loading observations
        obs = np.load('{}/{}.npy'.format(dbDir, dbName))
        obs = renameFields(obs)
        obs.sort(order='observationStartMJD')

        # Select observations
        idx = obs['night'] <= nights
        obs = obs[idx]

        for night in range(np.min(obs['night']), np.max(obs['night'])):
            idx = obs['night'] == night
            obs_disp = obs[idx]
            self.frame()
            mjd0 = np.min(obs_disp['observationStartMJD'])
            polylist = []
            polyarea = []
            mjds = obs_disp['observationStartMJD']-mjd0
            nchanges = len(list(itertools.groupby(obs_disp['filter'])))-1
            self.fig.suptitle(
                'night {} - filter changes: {}'.format(night, nchanges))

            # area observed versus time
            for val in obs_disp:
                pointing = LSSTPointing(val['fieldRA'], val['fieldDec'])
                p = PolygonPatch(
                    pointing, facecolor=self.colors[val['filter']], edgecolor=self.colors[val['filter']])
                self.ax.add_patch(p)
                # area observed versus time
                if self.areaTime:
                    polylist.append(pointing)
                    polyarea.append(area(polylist))

            if self.areaTime:
                self.ax2.plot(24.*mjds, polyarea, 'b.')
            if self.realTime:
                plt.draw()
                plt.pause(1.)
                plt.close()
            if saveFig:
                plt.savefig('{}_night_{}.png'.format(self.dbName, night))
            self.ax.clear()
            if self.areaTime:
                self.ax2.clear()

    def frame(self):
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

        self.ax.set_xlabel('Ra [deg]')
        self.ax.set_ylabel('Dec [deg]')
        self.ax.set_xlim(0, 360.)
        self.ax.set_ylim(-90., 10.)

        colorfilters = []
        for band in self.colors.keys():
            colorfilters.append(mpatches.Patch(
                color=self.colors[band], label='{}'.format(band)))
        plt.legend(handles=colorfilters, loc='upper left',
                   bbox_to_anchor=(1., 0.5))


class CadenceMovie:
    def __init__(self, dbDir, dbName, nights=1, title='', total=600, sub=100, fps=24, saveMovie=False, realTime=False, saveFig=False, areaTime=False):
        """ Display obs footprint vs time

        Parameters
        ----------
        dbDir : str
         database directory
        dbName : str
         database name
        nights : int,opt
         nights to process
         default: 1
        title : str,opt
         title of the movie
         default : ''
        total: int,opt

         default: 600
        sub: int, opt

         default: 100
        fps: int, opt

         default: 24
        saveMovie: bool, opt
         save the movie as mp4 file
         default: False
        realTime: bool, opt
         display results in real-time
         default: False
        saveFig: bool, opt
         save fig at the end of each night
         default: false
        areaTime: bool, opt
         draw observed area vs time in an embedded histo
         default: False
        """

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
        fig = plt.figure()
        ax = fig.gca()
        adjr = 0.9
        ax2 = None
        if self.areaTime:
            ax2 = self.fig.add_axes([0.75, 0.675, 0.20, 0.2])
            adjr = 0.74
        P1 = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(right=adjr)
        fig.canvas.draw()
        self.customize(ax, ax2)

        # Select observations
        idx = obs['night'] <= nights
        obs = obs[idx]

        self.polylist = []
        if saveMovie:
            # Warning : to save the movie ffmpeg needs to be installed!
            print(manimation.writers.list())
            FFMpegWriter = manimation.writers['ffmpeg']
            metadata = dict(title=title, artist='Matplotlib',
                            comment=title)
            writer = FFMpegWriter(
                fps=fps, metadata=metadata, bitrate=6000)
            writer = anim.FFMpegWriter(fps=30, codec='hevc')
            Name_mp4 = title
            with writer.saving(fig, Name_mp4, 250):
                self.loopObs(obs, fig, ax, ax2, writer=writer)
        else:
            self.loopObs(obs, fig, ax, ax2)

    def showObs(self, obs, nchanges, area, mjd0, fig, ax, ax2):
        """ Display observation (RA, Dec) as time (MJD) evolves.

        Parameters
        ----------
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

        fig.suptitle(
            'night {} - MJD {} \n filter changes: {}'.format(night, np.round(mjd, 3), nchanges))

        # LSST focal plane corresponding to this pointing
        pointing = LSSTPointing(obs['fieldRA'], obs['fieldDec'])
        p = PolygonPatch(
            pointing, facecolor=self.colors[obs['filter']], edgecolor=self.colors[obs['filter']])
        ax.add_patch(p)

        # area observed versus time
        if self.areaTime:
            ax2.plot([24.*(mjd-mjd0)], [area], 'b.')

    def loopObs(self, obs, fig, ax, ax2, writer=None):
        """Loop on observations

        Parameters
        -----------
        obs: array
         array of observations
        writer: bool, opt
         write movie on a file
         default: None
        """

        for night in range(np.min(obs['night']), np.max(obs['night'])):
            idx = obs['night'] == night
            obs_disp = obs[idx]
            mjd0 = np.min(obs_disp['observationStartMJD'])
            if len(obs_disp) > 0:
                for k in range(len(obs_disp)):
                    # number of filter changes up to now
                    sel = obs_disp[:k]
                    nchanges = len(list(itertools.groupby(sel['filter'])))-1
                    # show observations
                    if self.areaTime:
                        self.polylist.append(LSSTPointing(
                            obs_disp[k]['fieldRA'], obs_disp[k]['fieldDec']))
                        self.showObs(obs_disp[k], nchanges,
                                     area(self.polylist), mjd0, fig, ax, ax2)
                    else:
                        self.showObs(obs_disp[k], nchanges,
                                     0., mjd0, fig, ax, ax2)
                    fig.canvas.flush_events()
                    if writer:
                        print('grabing frame', k)
                        writer.grab_frame()
            if self.saveFig:
                plt.savefig('{}_night_{}.png'.format(self.dbName, night))
            # clear before next night
            ax.clear()
            if self.areaTime:
                ax2.clear()
            self.customize(ax, ax2)
            self.polylist = []
            """
            self.ax.clear()
            if self.areaTime:
                self.ax2.clear()
            self.customize()
            self.polylist = []
            """

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

    def customize(self, ax, ax2):
        """Define axis parameters for display
        """
        """
        if self.areaTime:
            self.ax2.set_xlabel('Time [hour]', size=8)
            self.ax2.set_ylabel('area [deg2]', size=8)
            self.ax2.xaxis.set_tick_params(labelsize=8)
            self.ax2.yaxis.set_tick_params(labelsize=8)
            self.ax2.yaxis.tick_right()
            self.ax2.yaxis.set_label_position("right")

        self.ax.set_xlabel('Ra [deg]')
        self.ax.set_ylabel('Dec [deg]')
        self.ax.set_xlim(0, 360.)
        self.ax.set_ylim(-90., 10.)
        """
        if self.areaTime:
            ax2.set_xlabel('Time [hour]', size=8)
            ax2.set_ylabel('area [deg2]', size=8)
            ax2.xaxis.set_tick_params(labelsize=8)
            ax2.yaxis.set_tick_params(labelsize=8)
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")

        ax.set_xlabel('Ra [deg]')
        ax.set_ylabel('Dec [deg]')
        ax.set_xlim(0, 360.)
        ax.set_ylim(-90., 10.)

        colorfilters = []
        for band in self.colors.keys():
            colorfilters.append(mpatches.Patch(
                color=self.colors[band], label='{}'.format(band)))
        plt.legend(handles=colorfilters, loc='upper left',
                   bbox_to_anchor=(1., 0.5))
