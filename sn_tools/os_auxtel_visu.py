#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:52:47 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import numpy as np
#import matplotlib.pyplot as plt
from sn_plotter_analysis import plt
import pandas as pd
import healpy as hp
import numpy.lib.recfunctions as rf
import os
from sn_tools.sn_io import checkDir

class AuxTelSurvey:
    def __init__(self,obs,dbName,nside=64,figDir='None',
                 movieDir='None',ffmpeg_cmd='ffmpeg'):
        """
        class to visualize LSST and AuxTel surveys

        Parameters
        ----------
        obs : array
            observations.

        Returns
        -------
        None.

        """
        
        self.obs = obs
        ra_min ,ra_max = np.min(obs['RA']),np.max(obs['RA'])
        dec_min ,dec_max = np.min(obs['Dec']),np.max(obs['Dec'])
        
        self.coord_lim = {}
        
        self.coord_lim['RA'] = [ra_min,ra_max]
        self.coord_lim['Dec'] = [dec_min,dec_max]
        
        self.npixels = hp.nside2npix(nside)
        # prepare output directory
        self.plotDir = 'None'
        if figDir != 'None':
            self.plotDir = '{}/{}'.format(figDir,dbName)
            checkDir(self.plotDir)
        
        self.dbName = dbName
        
        self.movieDir = movieDir
        if self.movieDir != 'None':
            self.ffmpeg_cmd = ffmpeg_cmd
            checkDir(self.movieDir)
        
    def __call__(self,night_min=1,night_max=365):
        
        idx = self.obs['night'] >= night_min
        idx &= self.obs['night'] <=  night_max
        
        obs = self.obs[idx]
        
        self.loop_obs(obs)
        
        if self.movieDir != 'None':
            self.makeMovie()
        
    def loop_obs(self,obs):
        """
        Method to loop on observations

        Returns
        -------
        None.

        """
        
        r = []
        colNames = ['MJD','dist_obs','dist_calib','dist_obs_calib',
                    'deltaT [s]','slew_time [s]','slewing_speed [deg/s]']
        
        for i in range(1,len(obs)):
            night = obs[i]['night']
            mjd = obs[i]['mjd']
            if i > 0:
                dd_cur = self.get_vals(obs[i])
                dd_bef = self.get_vals(obs[i-1])
                dista = dist(dd_cur['RA'],dd_cur['Dec'],
                             dd_bef['RA'],dd_bef['Dec'])
                distb = dist(dd_cur['RA_deg'],dd_cur['Dec_deg'],
                             dd_bef['RA_deg'],dd_bef['Dec_deg'])
                distc = dist(dd_cur['RA_deg'],dd_cur['Dec_deg'],
                             dd_cur['RA'],dd_bef['Dec'])
                deltaT = 24*3600*(dd_cur['mjd']-dd_bef['mjd'])-dd_bef['visitTime']
                slew_time = dd_bef['slewTime']
                r.append((mjd,dista,distb,distc,deltaT,slew_time,dista/slew_time))
        
            else:
                r.append((mjd,0,0,0,0,0,0))
                
            df_pl = pd.DataFrame(r,columns=colNames)
        
            obs_pl = obs[:i+1]
            
            rb = [1]*len(obs_pl)
            rb[-1] = 2
            if len(obs_pl) == 1:
                rb = [2]
            obs_pl = rf.append_fields(obs_pl,'count',rb)
            
            self.plot_night(obs_pl,df_pl,night,i)
    
    def get_vals(self,obs):
        
        cols = ['RA','Dec','mjd','visitTime','slewTime','RA_deg','Dec_deg']
       
        dd = dict(zip(cols,[obs[col] for col in cols]))
       
        return dd
        
    
    def plot_night(self,obs,df_pl,night,inum):
        
        fig = plt.figure(figsize=(15, 12))
        fig.subplots_adjust(top=0.99)
        fig.suptitle('night {}'.format(night))
        ax_a = fig.add_axes([0.1, 0.50, 0.34, 0.40])
        ax_b = fig.add_axes([0.1, 0.01, 0.3, 0.35])
        ax_c = fig.add_axes([0.5, 0.65, 0.35, 0.25])
        ax_d = fig.add_axes([0.5, 0.4, 0.35, 0.25])
        ax_e = fig.add_axes([0.5, 0.15, 0.35, 0.25])
        
        self.plot_ra_dec(ax_a,obs)
        self.plot_Mollview(ax_b,obs[['healpixID','count']],fig,night)
        self.plot_info(ax_c,df_pl,
                       varx='MJD',legx='MJD',
                       vary='dist_obs',legy='dist [deg]',label='obs')
        self.plot_info(ax_c,df_pl,
                       varx='MJD',legx='MJD',
                       vary='dist_calib',
                       legy='dist [deg]',color='b',label='calib star',
                       lstyle='dashed')
        self.plot_info(ax_c,df_pl,
                      varx='MJD',legx='MJD',
                      vary='dist_obs_calib',legy='dist [deg]',
                      color='r',label='obs/calib star',lstyle='dotted')
        
        self.plot_info(ax_d,df_pl,
                       varx='MJD',legx='MJD',
                       vary='deltaT [s]',legy='slew time [s]')
        self.plot_info(ax_d,df_pl,
                       varx='MJD',legx='MJD',
                       vary='slew_time [s]',legy='slew time [s]',color='b')
        self.plot_info(ax_e,df_pl,
                       varx='MJD',legx='MJD',
                       vary='slewing_speed [deg/s]',
                       legy='slewing \n speed [deg/s]')
        
        ax_c.set_xticklabels([])
        ax_d.set_xticklabels([])
        
        ax_c.legend(bbox_to_anchor=(0.85, 1.15), ncol=3,
             fontsize=10, frameon=False)
        
        if self.plotDir != 'None':
            fName = '{}/fig_{}.jpg'.format(self.plotDir,str(inum).zfill(3))
            plt.savefig(fName)
        
        #plt.show()
        
        
    def plot_ra_dec(self,ax,obs,varx='RA',legx='RA [deg]',
                    vary='Dec',legy='Dec [deg]'):
        
        
        plt.sca(ax)
        
        ax.set_xlim([self.coord_lim[varx][0],self.coord_lim[varx][1]])
        ax.set_ylim([self.coord_lim[vary][0],self.coord_lim[vary][1]])
        
        n = len(obs)-1
        ax.plot(obs[:n][varx],obs[:n][vary],
                color='k',marker='o',markersize=6,linestyle='None')
  
        ax.plot(obs[n][varx],obs[n][vary],
                color='r',marker='o',markersize=6,linestyle='None')
        
        ax.plot(obs[n]['RA_deg'],obs[n]['Dec_deg'],
                color='b',marker='*',markersize=6,linestyle='None')
        
        ax.set_xlabel(r'{}'.format(legx))
        ax.set_ylabel(r'{}'.format(legy))
        ax.grid(visible=True)
        
    def plot_Mollview(self, axa,obs, fig, night,var='count'):
        """
        Method to display a Mollweid view

        Parameters
        --------------
        pixels: pandas df
          data to plots
        axa: matplotlib axis
          axis to use for the plot
        fig: matplotlib figure
          figure to use for plot
        night: int
          night number
        """
        
        
        plt.sca(axa)

        xmin = 0.99999
        xmax = np.max([np.max(obs[var]), 1])

        norm = plt.cm.colors.Normalize(xmin, xmax)
        cmap = plt.get_cmap('jet', xmax)
        cmap.set_under('w')

        hpxmap = np.zeros(self.npixels, dtype=int)
        hpxmap = np.full(hpxmap.shape, 0)
        hpxmap[obs['healpixID']] = obs[var].astype(int)

        # print('hello ',xmin,xmax)

        hp.mollview(hpxmap, nest=True, cmap=cmap,
                    min=xmin, max=xmax, norm=norm, cbar=False,
                    title='', hold=True, badcolor='white', xsize=1600)

        hp.graticule(verbose=False)

    def plot_info(self,ax,df_pl,
                      varx='MJD',legx='MJD',
                      vary='dist',legy='dist [deg]',color='k',
                      label=None,lstyle='solid'):
        
        ax.plot(df_pl[varx],df_pl[vary],color=color,
                label=label,linestyle=lstyle)
        
        ax.set_ylabel(r'{}'.format(legy),rotation=270,labelpad=50)
        
        ax.yaxis.tick_right()
        ax.grid(visible=True)
        ax.yaxis.set_label_position("right")
        ax.set_xlabel(r'{}'.format(legx))
        
        
    def makeMovie(self):
         """
         Method to make a movie from jpg files
         """
         dirFigs = self.plotDir
         # ffmpeg_cmd = '../../ffmpeg/ffmpeg-4.3.1-i686-static/ffmpeg'

         cmd = '{} -r 2 -f image2 -s 1920x1080 -i {}/fig_%03d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p {}/{}.mp4 -y'.format(
             self.ffmpeg_cmd, self.plotDir, self.movieDir, self.dbName)
         os.system(cmd)
   
def dist(ra_a,dec_a,ra_b,dec_b):
    dist = ((ra_a-ra_b)*np.cos(np.deg2rad(dec_a)))**2
    dist =+ (dec_a-dec_b)**2
    dist = np.sqrt(dist)
    
    return dist