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
import matplotlib
from sn_analysis.sn_os_atmos_calib_tools import load_star_calib
from sn_tools.sn_utils import multiproc

np.set_printoptions(legacy='1.25')

class AuxTelSurvey:
    def __init__(self,obs,dbName,star_list,
                 year=1,width_ra=10.,dist_min=30,
                 slew_speed=1.,survey_type='asynchronous',
                 num_exposures=2,exposure_time=30):

        #obs corresponding to this year
        idx = obs['year'] == year
        self.obs = pd.DataFrame.from_records(obs[idx])
        del obs
        

        #grab reference stars
        self.ref_stars = load_star_calib(star_list)
        
        #parameters
        self.year=year
        self.width_ra = width_ra
        self.dist_min = dist_min
        self.slew_speed = slew_speed
        self.survey_type = survey_type
        self.num_exposures = num_exposures
        self.exposure_time = exposure_time
    
    
    def __call__(self):
        
        params = {}
        
        nights = self.obs['night'].unique()
        df = multiproc(nights,params,self.auxtel_nights,nproc=16)
        
        return df
    
    def auxtel_nights(self,nights,params,j=0,output_q=None):
        
        
        idx = self.obs['night'].isin(nights)
        
        obsb = self.obs[idx]
        
        
        df = obsb.groupby(['night']).apply(lambda x:
                                          self.auxtel_night(x),
                                          include_groups=False).reset_index()
        if output_q is not None:
            return output_q.put({j: df})
        else:
            return df       
    
    def auxtel_night(self,grp):
        
        
        #get ra_min and ra_max
        
        ra_min = grp['RA'].min()
        ra_max = grp['RA'].max()
        
        stars_night = self.select_stars(ra_min,ra_max,grp)
        
        df = pd.DataFrame()
        if self.survey_type == 'asynchronous':
            df = self.survey_async(stars_night,grp)
        
        
        return df
        
    def select_stars(self,ra_min,ra_max,obs):
        
        #select ref stars in [ra_min,ra_max] window
        
        idx = self.ref_stars['RA_deg'] >= ra_min+self.width_ra
        idx &= self.ref_stars['RA_deg'] <= ra_max+self.width_ra
        
        stars_night = self.ref_stars[idx]
        
        dd = stars_night.groupby('Star_name').apply(lambda x:\
                                                    self.dist_min_star(x,obs),
                                               include_groups=False)\
                                                .reset_index()
        idx = dd['dist_min'] <= self.dist_min
        
        """
        fig, ax = plt.subplots()
        self.plot_night(obs,stars_night,fig=fig,ax=ax)
        self.plot_night(obs,dd[idx],fig=fig,ax=ax,mfc='r')        
        
        plt.show()
        """
        res = pd.DataFrame(dd[idx])
        
        return res
    
    def plot_night(self,obs,stars,fig=None,ax=None,mfc='None'):
    
        if fig is None:
            fig, ax = plt.subplots()
         
        ax.plot(obs['RA'],obs['Dec'],'ko')
        print('allo mfc',mfc)
        ax.plot(stars['RA_deg'],stars['Dec_deg'],
                marker='*',color='r',linestyle='None',mfc=mfc)

        

    def dist_min_star(self,grp_star,obs):
        
        ra_ref = grp_star['RA_deg'].to_list()*len(obs)
        dec_ref = grp_star['Dec_deg'].to_list()*len(obs)
        ra_obs = obs['RA']
        dec_obs = obs['Dec']
        dist_obs_star = dist(ra_obs,dec_obs,ra_ref,dec_ref).to_list()
        
        """
        index = dist_obs_star.index(np.min(dist_obs_star))
        print('allo',np.min(dist_obs_star),np.max(dist_obs_star),index)
        print(ra_obs.to_list()[index],dec_obs.to_list()[index],
              ra_ref[0],dec_ref[0])
        """
        grp_star['dist_min'] = np.min(dist_obs_star)
        
        return grp_star
        
    def survey_async(self, stars, obs):
        
        #get mjd_min and mjd_max
        
        mjd_min = obs['mjd'].min()
        mjd_max = obs['mjd'].max()
        
        #define the order of star observation 
        #according to the travel salesman problem
        
        from sn_analysis.sn_algo_metrop import Metrop_Anneal
        
        mm = Metrop_Anneal(stars)
        
        dd = mm()
        
        dd = dd.sort_values(by='iorder')
        #add Star_name
        colsb = ['RA_deg','Dec_deg']
        cols = ['Star_name']+colsb
        dd = dd.merge(self.ref_stars[cols],
                      left_on=colsb,right_on=colsb)
        #print(dd)
        dist_stars = self.calc_dist(dd)
        
        #print(dist_stars)
        mjd_cur = mjd_min
        
        ncity = len(dd)
        r = []
        corfact = 1./(24*3600)
        while mjd_cur <= mjd_max:
            for i,row in dd.iterrows():
                iorder = row['iorder']
                #grab the next star in the list
                iorder_next = (iorder+1)%ncity
                idx = dd['iorder'] == iorder_next
                sel = pd.DataFrame(dd[idx])
                #grab the distance
                """
                combi='{}/{}'.format(row['Star_name'],
                                     sel['Star_name'].to_list()[0])
                """
                dist_star = self.get_dist_from_combi(dist_stars,
                                                     row['Star_name'],
                                     sel['Star_name'].to_list()[0])
                #print('dist',dist_star)
                slew_time = self.slew_speed*dist_star
                
                r.append((row['Star_name'],int(row['iorder']),
                          row['RA_deg'],row['Dec_deg'],
                          self.num_exposures,self.exposure_time,
                          slew_time,mjd_cur))
                mjd_cur += self.num_exposures*self.exposure_time*corfact
                mjd_cur += slew_time*corfact
               
                if mjd_cur >= mjd_max:
                    break
                
        cols = ['Star_name','iorder','RA_deg',
                'Dec_deg','numExposure','exposure_time',
                'slew_time','mjd']     
        df = pd.DataFrame(r,columns=cols)
        
        return df
    def get_dist_from_combi(self,dists,star_a,star_b):
        
        combi = '{}/{}'.format(star_a,star_b)
        
        if combi not in dists.keys():
            combi = '{}/{}'.format(star_b,star_a)
        
        return dists[combi]
        
    def calc_dist(self,df):
        
        
        r = {}
        for i, rowa in df.iterrows():
            ra_a = rowa['RA_deg']
            dec_a = rowa['Dec_deg']
            name_a = rowa['Star_name']
            for j,rowb in df.iterrows():
                if j > i:
                    ra_b = rowb['RA_deg']
                    dec_b = rowb['Dec_deg']
                    name_b = rowb['Star_name']
                    dist_star = dist(ra_a,dec_a,ra_b,dec_b)
                    combi = '{}/{}'.format(name_a,name_b)
                    r[combi]=dist_star
        
        return r
        
class AuxTelSurvey_visu:
    def __init__(self,obs,dbName,nside=64,figDir='None',
                 movieDir='None',ffmpeg_cmd='ffmpeg',year=1):
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
        
        idx = obs['year'] == year
        self.obs = obs[idx]
        self.year=year
        
        ra_min ,ra_max = np.min(obs['RA']),np.max(obs['RA'])
        dec_min ,dec_max = np.min(obs['Dec']),np.max(obs['Dec'])
        
        self.coord_lim = {}
        
        self.coord_lim['RA'] = [ra_min,ra_max]
        self.coord_lim['Dec'] = [dec_min,dec_max]
        
        self.npixels = hp.nside2npix(nside)
        # prepare output directory
        self.plotDir = 'None'
        if figDir != 'None':
            self.plotDir = '{}/{}/year_{}'.format(figDir,dbName,year)
            checkDir(self.plotDir)
        
        self.dbName = dbName
        
        self.movieDir = movieDir
        if self.movieDir != 'None':
            self.ffmpeg_cmd = ffmpeg_cmd
            checkDir(self.movieDir)
        
    def __call__(self,nights=[2],nproc=8):
        
        params = {}
        
        if nproc > 1:
            from sn_tools.sn_utils import multiproc
            multiproc(nights,params,self.process,nproc)
        else:
            self.process(nights,params)
        
    def process(self,nights, params, j=0, output_q=None):
        
        for nnight in nights:
            print('processing night',nnight)
            night = nnight+365*(self.year-1)
            idx = self.obs['night']==night
            obs_night = self.obs[idx]
            if len(obs_night) == 0:
                continue
            idxb = self.obs['night'] < night
            obs_bef = self.obs[idxb]
            
            obs_bef = rf.append_fields(obs_bef,'count',
                                       [1]*len(obs_bef),
                                       dtypes='<i8')
            self.loop_obs_night(obs_night,obs_bef,night)
        
            del obs_bef
            del obs_night
            
            if self.movieDir != 'None':
                self.makeMovie(night)
        
        if output_q is not None:
            return output_q.put({j: 1})
        else:
            return 1
        
        
        
    def loop_obs_night(self,obs,obs_bef,night):
        """
        Method to loop on observations

        Returns
        -------
        None.

        """
        
        colNames = ['MJD','dist_obs','dist_calib','dist_obs_calib',
                    'deltaT [s]','slew_time [s]','slewing_speed [deg/s]']
        df_pl = pd.DataFrame()
        for i in range(1,len(obs)):
            r = []
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
                
            df_plb = pd.DataFrame(r,columns=colNames)
            df_pl = pd.concat((df_pl,df_plb))
        
            obs_pl = obs[:i+1]
            
            rb = [2]*len(obs_pl)
            rb[-1] = 3
            if len(obs_pl) == 1:
                rb = [3]
        
            obs_pl = rf.append_fields(obs_pl,'count',rb)
            
            self.plot_night(obs_bef,obs_pl,df_pl,night,i)
    
        del df_pl
        
    def get_vals(self,obs):
        
        cols = ['RA','Dec','mjd','visitTime','slewTime','RA_deg','Dec_deg']
       
        dd = dict(zip(cols,[obs[col] for col in cols]))
       
        return dd
        
    
    def plot_night(self,obs_bef,obs,df_pl,night,inum):
        
        fig = plt.figure(figsize=(15, 12))
        fig.subplots_adjust(top=0.99)
        nnight = night-365*(self.year-1)
        fig.suptitle('year {} - night {}'.format(self.year,nnight))
        ax_a = fig.add_axes([0.1, 0.50, 0.34, 0.40])
        ax_b = fig.add_axes([0.1, 0.01, 0.3, 0.35])
        ax_c = fig.add_axes([0.5, 0.65, 0.35, 0.25])
        ax_d = fig.add_axes([0.5, 0.4, 0.35, 0.25])
        ax_e = fig.add_axes([0.5, 0.15, 0.35, 0.25])
        
        self.plot_ra_dec(ax_a,obs,obs_bef)
        self.plot_Mollview(ax_b,obs[['healpixID','count']],
                           obs_bef[['healpixID','count']],
                           fig,night)
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
                       vary='deltaT [s]',legy='$\Delta t$ [s]',
                       label='$\Delta t$')
        self.plot_info(ax_d,df_pl,
                       varx='MJD',legx='MJD',
                       vary='slew_time [s]',legy='$\Delta t$ [s]',
                       color='m',lstyle='dashed',
                       label='slew time')
        self.plot_info(ax_e,df_pl,
                       varx='MJD',legx='MJD',
                       vary='slewing_speed [deg/s]',
                       legy='slewing \n speed [deg/s]')
        
        ax_c.set_xticklabels([])
        ax_d.set_xticklabels([])
        
        ax_a.legend(bbox_to_anchor=(0.85, 1.15), ncol=2,
             fontsize=10, frameon=False)
        ax_c.legend(bbox_to_anchor=(0.85, 1.15), ncol=3,
             fontsize=10, frameon=False)
        ax_d.legend(bbox_to_anchor=(0.95, 0.97), ncol=2,
             fontsize=10, frameon=False)
        
        
        if self.plotDir != 'None':
            outDir = '{}/night_{}'.format(self.plotDir,nnight)
            checkDir(outDir)
            fName = '{}/fig_{}.jpg'.format(outDir,str(inum).zfill(3))
            plt.savefig(fName)
        
        #plt.show()
        
        plt.close(fig)
        
    def plot_ra_dec(self,ax,obs,obs_bef,varx='RA',legx='RA [deg]',
                    vary='Dec',legy='Dec [deg]'):
        
        
        plt.sca(ax)
        msize=5
        if len(obs_bef) > 0:
            ax.plot(obs_bef[varx],obs_bef[vary],
                color='grey',marker='o',
                markersize=msize,linestyle='None')
        
       
        
        n = len(obs)-1
        ax.plot(obs[:n][varx],obs[:n][vary],
                color='k',marker='o',markersize=msize,
                linestyle='None',label='LSST pointings')
  
        ax.plot(obs[n][varx],obs[n][vary],
                color='r',marker='o',
                markersize=msize,linestyle='None')
        
        ax.plot(obs[n]['RA_deg'],obs[n]['Dec_deg'],
                color='b',marker='*',markersize=msize,
                linestyle='None',label='calib star')
        
        ax.set_xlim([self.coord_lim[varx][0],self.coord_lim[varx][1]])
        ax.set_ylim([self.coord_lim[vary][0],self.coord_lim[vary][1]])
        
        ax.set_xlabel(r'{}'.format(legx))
        ax.set_ylabel(r'{}'.format(legy))
        ax.grid(visible=True)
        
    def plot_Mollview(self, axa,obs,obs_bef,fig, night,var='count'):
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
        
        cmap = matplotlib.colors.ListedColormap(["grey", "black", "red"])
        cmap.set_under('w')
        """
        cmap = plt.get_cmap('jet', xmax)
        cmap.set_under('w')
        """
    
        hpxmap = np.zeros(self.npixels, dtype=int)
        hpxmap = np.full(hpxmap.shape, 0)
        hpxmap[obs_bef['healpixID']] = obs_bef[var]
        hpxmap[obs['healpixID']] = obs[var].astype(int)

    
        # print('hello ',xmin,xmax)

        hp.mollview(hpxmap, nest=True, cmap=cmap,
                    min=xmin, max=xmax, norm=norm, cbar=False,
                    title='', hold=True, badcolor='white', xsize=1600)

        hp.graticule()

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
        
        
    def makeMovie(self,night):
         """
         Method to make a movie from jpg files
         """
         nnight = night-365*(self.year-1)
         dirFigs = '{}/night_{}'.format(self.plotDir,nnight)
         movieDir = '{}/year_{}'.format(self.movieDir,self.year)
         checkDir(movieDir)
         # ffmpeg_cmd = '../../ffmpeg/ffmpeg-4.3.1-i686-static/ffmpeg'
         
         cmd = '{} -r 2 -f image2 -s 1920x1080 -i {}/fig_%03d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p {}/{}_{}.mp4 -y'.format(
             self.ffmpeg_cmd, dirFigs, movieDir, self.dbName,nnight)
         os.system(cmd)
   
def dist(ra_a,dec_a,ra_b,dec_b):
    dist = ((ra_a-ra_b)*np.cos(np.deg2rad(dec_a)))**2
    dist += (dec_a-dec_b)**2
    dist = np.sqrt(dist)
    
    return dist