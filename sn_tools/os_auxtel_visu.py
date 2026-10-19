#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:52:47 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
import numpy.lib.recfunctions as rf

class AuxTelSurvey:
    def __init__(self,obs,nside=64):
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
        print('coords',self.coord_lim)
        self.loop_obs()
        
    def loop_obs(self):
        """
        Method to loop on observations

        Returns
        -------
        None.

        """
        
        r = []
        colNames = ['MJD','dist','deltat [s]','slewing_speed [deg/s]']
        
        for i in range(1,len(self.obs)):
            night = self.obs[i]['night']
            mjd = self.obs[i]['mjd']
            if i > 0:
                ra_a = self.obs[i]['RA']
                time_a = self.obs[i]['mjd']
                ra_b = self.obs[i-1]['RA']
                dec_a = self.obs[i]['Dec']
                dec_b = self.obs[i-1]['Dec']
                time_b = self.obs[i-1]['mjd']
                visit_b = self.obs[i-1]['visitTime']
                slew_b = self.obs[i-1]['slewTime']
                slew_a = self.obs[i]['slewTime']
                dist = ((ra_a-ra_b)*np.cos(np.deg2rad(dec_a)))**2
                dist =+ (dec_a-dec_b)**2
                dist = np.sqrt(dist)
                deltaT = 24*3600*(time_a-time_b)-visit_b
                r.append((mjd,dist,deltaT,dist/deltaT))
        
            else:
                r.append((mjd,0,0,0))
                
            df_pl = pd.DataFrame(r,columns=colNames)
        
            obs_pl = self.obs[:i+1]
            print(obs_pl)
            rb = [1]*len(obs_pl)
            rb[-1] = 2
            if len(obs_pl) == 1:
                rb = [2]
            obs_pl = rf.append_fields(obs_pl,'count',rb)
            
            self.plot_night(obs_pl,df_pl,night)
        
    def plot_night(self,obs,df_pl,night):
        
        fig = plt.figure(figsize=(15, 12))
        fig.subplots_adjust(top=0.99)
        fig.suptitle('night {}'.format(night))
        ax_a = fig.add_axes([0.1, 0.50, 0.34, 0.40])
        ax_b = fig.add_axes([0.1, 0.01, 0.3, 0.3])
        ax_c = fig.add_axes([0.5, 0.60, 0.35, 0.25])
        ax_d = fig.add_axes([0.5, 0.35, 0.35, 0.25])
        ax_e = fig.add_axes([0.5, 0.1, 0.35, 0.25])
        
        self.plot_ra_dec(ax_a,obs)
        self.plot_Mollview(ax_b,obs[['healpixID','count']],fig,night)
        self.plot_info(ax_c,df_pl,
                       varx='MJD',legx='MJD',
                       vary='dist',legy='dist [deg]')
        self.plot_info(ax_d,df_pl,
                       varx='MJD',legx='MJD',
                       vary='deltat [s]',legy='slew time [s]')
        self.plot_info(ax_e,df_pl,
                       varx='MJD',legx='MJD',
                       vary='slewing_speed [deg/s]',
                       legy='slewing \n speed [deg/s]')
        
        ax_c.set_xticklabels([])
        ax_d.set_xticklabels([])
        
        plt.show()
        
        
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
        
        ax.set_xlabel(r'{}'.format(legx),font='bold')
        ax.set_ylabel(r'{}'.format(legy),font='bold')
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
                      vary='dist',legy='dist [deg]',color='k'):
        
        ax.plot(df_pl[varx],df_pl[vary],color=color)
        
        ax.set_ylabel(r'{}'.format(legy),rotation=270,labelpad=25)
        
        ax.yaxis.tick_right()
        ax.grid(visible=True)
        ax.yaxis.set_label_position("right")
        