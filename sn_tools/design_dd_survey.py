from sn_plotters.sn_cadencePlotters import Lims
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sn_tools.sn_telescope import Telescope
from scipy import interpolate
from sn_tools.sn_cadence_tools import AnaOS

filtercolors = dict(zip('ugrizy',['b','c','g','y','r','m']))
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 12

class Lims_z_m5_band:

    def __init__(self,x1,color,band,refDir, 
                 SNR,
                 namesRef = ['SNCosmo'], 
                 mag_range = [23., 27.5], 
                 dt_range=[0.5, 25.],
                 m5_str='m5_mean',cadence_str='cadence_mean'):

        
        namesRef = ['SNCosmo']
        Li_files = []
        mag_to_flux_files = []
        self.mag_range = mag_range
        self.dt_range = dt_range
        self.namesRef = namesRef

        for name in namesRef:
            Li_files = ['{}/Li_{}_{}_{}.npy'.format(refDir,name,x1,color)]
            mag_to_flux_files = ['{}/Mag_to_Flux_{}.npy'.format(refDir,name)]

        self.lim_z = Lims(Li_files, mag_to_flux_files,
                      band, SNR, mag_range=mag_range, 
                      dt_range=dt_range)


    def plot(self,res,dbName,saveFig=False):

        self.lim_z.plotCadenceMetric(res,dbName=dbName,saveFig=saveFig)

    def getLims(self, data,m5_str,cadence_str,blue_zcut=-1):

        idx = (data[m5_str] >= self.mag_range[0]) & (
             data[m5_str] <= self.mag_range[1])
        idx &= (data[cadence_str] >= self.dt_range[0]) & (
            data[cadence_str] <= self.dt_range[1])
        data = data[idx]
        
        restot = None

        #self.plot(data,'test')

        if len(data) > 0:
            resu = np.copy(data)
            for io, interp in enumerate(self.namesRef):
                zlims = self.lim_z.interpGriddata(io, data,m5_str=m5_str,cadence_str=cadence_str)
                zlims[np.isnan(zlims)] = -1
                resu = rf.append_fields(data, 'zlim_'+self.namesRef[io], zlims)
                if blue_zcut >0:
                    io = resu['zlim_'+self.namesRef[io]]<=blue_zcut
                    resu = resu[io]

                if restot is None:
                    restot = resu
                else:
                    restot = np.concatenate((restot, resu))

        return restot


class Lims_z_m5:

    def __init__(self, x1,color,namesRef=['SNCosmo'],
                 SNR= dict(zip('rizy', [25., 25., 30., 35.])),
                 blue_zcut=dict(zip('gri',[0.3,0.701,1.0]))):
        

        self.x1 = x1
        self.color = color
        self.SNR = SNR
        self.bands = SNR.keys()
        self.refDir ='../reference_files'
        self.namesRef = namesRef
        self.blue_zcut=blue_zcut

    def process(self,cadences=[3.,4.]):
                 
        zlims = None
        for band in self.bands:

            print('processing',band)
            myclass = Lims_z_m5_band(self.x1,self.color,band,refDir=self.refDir,SNR=self.SNR[band],namesRef=self.namesRef)
    
            mag_min = myclass.mag_range[0]
            mag_max = myclass.mag_range[1]

            r = []
            for mag in np.arange(mag_min,mag_max,0.001):
                for cad in cadences:
                    r.append((cad,mag,band))

            data = np.rec.fromrecords(r,names=['cadence_mean','m5_mean','band'])

            #myclass.plot(data,'unknown',False)
            blue_zcut = -1
            if band in self.blue_zcut.keys():
                blue_zcut = self.blue_zcut[band]
            zl = myclass.getLims(data,'m5_mean','cadence_mean',blue_zcut)
            

            #apply the cut in z below - blue cutoff for SN

            if zlims is None:
                zlims = zl
            else:
                zlims = np.concatenate((zlims,zl))

        return zlims
    

    def plot(self,zlims,cadence,ystr='m5_mean',yleg='m$_5$',yscale='linear',locx=0.45,locy=500.):

        fig, ax = plt.subplots()

        #bands = np.unique(zlims['band'])
        bands = 'rizy'
        print('plotting',bands)
        filtercolors = dict(zip(bands,['g','y','r','m']))
        if 'band' in zlims.dtype.names:
            for band in bands:
                idx = zlims['band'] == band
                sel = zlims[idx]
                self.plotIndiv(ax,sel,ystr,yleg,cadence,band,color=filtercolors[band],locx=locx,locy=locy)
        else:
           self.plotIndiv(ax,zlims,ystr,yleg,cadence,'all',color='k',locx=locx,locy=locy)

        fontsize = 12
        ax.legend(loc = 'upper left',fontsize=fontsize)

        ax.set_xlabel('z$_{faint}$',fontsize=fontsize)
        ax.set_ylabel(yleg,fontsize=fontsize)
        ax.set_yscale(yscale)
        ax.grid()

    def plotIndiv(self,ax,sel,ystr,yleg,cadence,band,color,locx,locy):
        
        
        #for cad in np.unique(sel['cadence_mean']):
        ls = ['-','--']
        for i,cad in enumerate(cadence):
            idb = np.abs(sel['cadence_mean']-cad)<1.e-5
            selb = sel[idb]
            for name in self.namesRef:
                zlim_name = 'zlim_{}'.format(name)
                idc = selb[zlim_name]>0.
                selc = selb[idc]
                if i == 0:
                    ax.plot(selc[zlim_name],selc[ystr],marker=None,
                            color=color,
                            label='{}'.format(band),
                            ls=ls[i])
                else:
                   ax.plot(selc[zlim_name],selc[ystr],marker=None,
                            color=color,
                            ls=ls[i]) 
            if band =='r' or band =='all':
                limsy = ax.get_ylim()
                yscale = limsy[1]-limsy[0]
                locyv = locy-0.06*yscale*i
                ax.plot([locx,locx+0.05],[locyv,locyv],ls=ls[i],color='k')
                ax.text(locx+0.06,locyv,'cadence: {} days'.format(int(cad)))

def nvisits(zlims,med_m5):

    def nvisits_deltam5(m5,m5_median):

        diff = m5-m5_median
        
        nv = 10**(0.8*diff)

        #return nv.astype(int)
        return nv

    zlic = np.copy(zlims)
    res_nv = None
    for band in np.unique(zlic['band']):
        m5med = med_m5[band]
        idx = (zlic['band']==band)&(zlic['m5_mean']>=m5med)
        sel = zlic[idx]
        nv = nvisits_deltam5(sel['m5_mean'],m5med)
        #print(nv)
        sel = rf.append_fields(sel,'nvisits',nv)
        if res_nv is None:
            res_nv = sel
        else:
            res_nv = np.concatenate((res_nv,sel))
        

    return res_nv


class AnaMedValues:

    def __init__(self,fname,plot=False):

        
        medValues = np.load(fname)

        print(medValues.dtype)
        df = pd.DataFrame(np.copy(medValues))
    
        df = df[(df['filter']!='u')&(df['filter']!='g')]

        dfgrp = df.groupby(['fieldname','season','filter']).median().reset_index()

        dfgrp_season = df.groupby(['fieldname','filter']).median().reset_index()

        print(dfgrp[['fieldname','season','filter','fiveSigmaDepth']])
   
        if plot:
            self.plot(dfgrp,dfgrp_season)

        self.medValues =  df.groupby(['filter']).median().reset_index()

    def plot(self,dfgrp,dfgrp_season):

        fontsize = 12
        figres, axres = plt.subplots()
        for band in dfgrp['filter'].unique():
            dffilt = dfgrp[dfgrp['filter']==band]
            dffilt_season = dfgrp_season[dfgrp_season['filter']==band]
            """ this is to plot per band and display all seasons - check dispersion
            fig, ax = plt.subplots()
            fig.suptitle('{} band'.format(band))
            ax.plot(dffilt['fieldname'],dffilt['fiveSigmaDepth'],'ko',mfc='None')
            ax.plot(dffilt_season['fieldname'],dffilt_season['fiveSigmaDepth'],'rs')
            """
            axres.plot(dffilt_season['fieldname'],dffilt_season['fiveSigmaDepth'],marker='s',color=filtercolors[band],label='{} band'.format(band))

        axres.legend(loc='upper left', bbox_to_anchor=(0.01, 1.15),
                     ncol=4, fancybox=True, shadow=True, fontsize=fontsize)
        axres.set_ylabel('median m$_{5}^{single visit}$ [ten seasons]',fontsize=fontsize)
        axres.grid()

    




"""
for fieldname in np.unique(medValues['fieldname']):
    idf = medValues['fieldname']==fieldname
    print(fieldname)
    sel = metricValues[idf]
    for band in np.unique(sel['filter']):
        idfb = sel['filter'] == band
        selb = sel[idfb]
        for season in
"""

def restframeBands():

    fontsize = 12
    telescope = Telescope(airmass=1.2)

    zvals = np.arange(0.01,1.2,0.01)

    bands = 'ugrizy'
    
    wave_frame = None
    for band in bands:
        mean_restframe_wavelength = telescope.mean_wavelength[band] /(1. + zvals)
        arr = np.array(mean_restframe_wavelength,dtype=[('lambda_rf','f8')])
        arr = rf.append_fields(arr,'band',[band]*len(arr)) 
        arr = rf.append_fields(arr,'z',zvals)
        print(band,mean_restframe_wavelength)
        if wave_frame is None:
            wave_frame=arr
        else:
            wave_frame = np.concatenate((wave_frame,arr))

    #print(wave_frame)

    figw, axw = plt.subplots()
    for band in bands:
        ig = wave_frame['band']==band
        selig = wave_frame[ig]
        axw.plot(selig['lambda_rf'],selig['z'],color=filtercolors[band],label=band)

    axw.plot([350.,350.],[0.,1.2],color='r',ls='--')
    axw.plot([380.,380.],[0.,1.2],color='r',ls='--')
    axw.plot([800.,800.],[0.,1.2],color='r',ls='--')
    axw.set_xlabel(r'$\lambda^{LSST\ mean\ band}_{rf}$ [nm]', fontsize=fontsize)
    axw.set_ylabel('z', fontsize=fontsize)
    axw.set_ylim([0.,1.2])
    axw.legend(loc = 'upper right')
    axw.grid()



class Visits_z:

    def __init__(self,x1,color,namesRef=['SNCosmo'],cadences=[3.,4.],plot=False):

        self.cadences = cadences

        # estimate m5 vs zlim
        myclass = Lims_z_m5(x1,color,namesRef=namesRef)
        zlims = myclass.process(cadences)
       
        #plot the results
        if plot:
            myclass.plot(zlims,cadences,locx=0.7,locy=24.)

        #to convert m5 values to nvisits: need m5 one visit

        # load (and plot) med values

        finalMed = AnaMedValues('medValues.npy',plot=plot).medValues

        m5_median = dict(zip(finalMed['filter'],finalMed['fiveSigmaDepth']))

        # now convert m5 to a number of visits
        nvisits_z = nvisits(zlims,m5_median)

        #remove outside range results
        idx = (nvisits_z['nvisits']>0)&(nvisits_z['zlim_SNCosmo']>0.)
        nvisits_z = nvisits_z[idx]

        # plot the results
        if plot:
            myclass.plot(nvisits_z,cadences,'nvisits','Nvisits',locy=500.)

        self.nvisits_tot_z = self.getNvisits_all_z(nvisits_z)
        if plot:
            myclass.plot(self.nvisits_tot_z,cadences,'nvisits','Nvisits',yscale='linear',locy=500)


    def getNvisits_all_z(self,nvisits_z):

        # now estimate the total number of visits vs z
        # two redshift range
        # [..,0.7]: riz bands
        # [0.7,...]: izy bands
        
        ida = nvisits_z['band'] != 'y'
        idb = (nvisits_z['band'] == 'y')&(nvisits_z['zlim_SNCosmo'] >0.7)
        
        nvisits_all_z = np.concatenate((nvisits_z[ida],nvisits_z[idb]))
        
        z = np.arange(0.3,0.9,0.001)

        ssum = None

        arr_tot = None
        for cadence in self.cadences:
            ssum = None
    
            for band in 'rizy':
   
                ik = nvisits_all_z['band'] == band
                ik &= np.abs(nvisits_all_z['cadence_mean']-cadence)<1.e-5
                selb = nvisits_all_z[ik]
        
                f = interpolate.interp1d(selb['zlim_SNCosmo'],
                                         selb['nvisits'],
                                         bounds_error=False,
                                         fill_value=0.)
                if ssum is None:
                    ssum = np.array(f(z))
                else:
                    ssum += np.array(f(z))
            arr_cad = np.array(z,dtype=[('zlim_SNCosmo','f8')])
            arr_cad = rf.append_fields(arr_cad,'nvisits',ssum)
            arr_cad = rf.append_fields(arr_cad,'cadence_mean',[cadence]*len(arr_cad))
            if arr_tot is None:
                arr_tot = arr_cad
            else:
                arr_tot=np.concatenate((arr_tot,arr_cad))

        return arr_tot


class DDbudget_zlim:

    def __init__(self,x1=-2.0,color=0.2):
    
        x1=-2.0
        color=0.2
        
        # get the total number of visits per obs night
        
        visits_per_night = Visits_z(x1,color,plot=False).nvisits_tot_z
        
        #first thing to be done: interplinear of nvisits vs z
        
        self.interp = {}
        self.reverse_interp = {}
        
        for cad in np.unique(visits_per_night['cadence_mean']):
            idx = visits_per_night['cadence_mean']==cad
            sel = visits_per_night[idx]
            self.interp['{}'.format(cad)] = interpolate.interp1d(sel['zlim_SNCosmo'],
                                                            sel['nvisits'],
                                                            bounds_error=False,
                                                            fill_value=0.)
            self.reverse_interp['{}'.format(cad)] = interpolate.interp1d(sel['nvisits'],
                                                                sel['zlim_SNCosmo'],
                                                                bounds_error=False,
                                                                fill_value=0.)

        # define scenarios
        dict_scen = self.scenarios()

        #estimate budget vs zlim
        res = self.calc_budget_zlim(dict_scen)

        # plot the results
        Nvisits=2774123
        Nvisits = 2388477
        self.plot(res,Nvisits=Nvisits)
   

    def scenarios(self):
        
        dict_scen = {}
        names = ['fieldname','Nfields','cadence','season_length','Nseasons','weight_visit']

        r = []
        r.append(('LSSTDDF',4,4.,6.0,10,1))
        r.append(('ADFS',2,4.,6.0,10,1))

        dict_scen['scen1'] = np.rec.fromrecords(r,names=names)

        r = []
        r.append(('LSSTDDF',4,3.,6.0,10,1))
        r.append(('ADFS',2,3.,6.0,10,1))

        dict_scen['scen2'] = np.rec.fromrecords(r,names=names)

        r = []
        r.append(('LSSTDDF',4,4.,6.0,2,1))
        r.append(('ADFS',2,4.,6.0,2,1))

        dict_scen['scen3'] = np.rec.fromrecords(r,names=names)

        r = []
        r.append(('LSSTDDF',4,3.,6.0,2,1))
        r.append(('ADFS',2,3.,6.0,2,1))

        dict_scen['scen4'] = np.rec.fromrecords(r,names=names)

        r = []
        r.append(('LSSTDDF',4,3.,6.0,2,1))
        r.append(('ADFS',2,4.,6.0,2,2))

        dict_scen['scen5'] = np.rec.fromrecords(r,names=names)

        return dict_scen


    def calc_budget_zlim(self,dict_scen):
    
        z = np.arange(0.3,0.89,0.01)
        res_tot = None
        for key, scen in dict_scen.items():
            arr_visits = None
            arr_frac = None
            res_scen = None
            for field in scen:
                cad = np.unique(field['cadence'])[0]
                fieldname = field['fieldname']
                print('hello',z,cad,self.interp.keys())

                zvals = np.copy(z)
                if field['weight_visit']>1:
                    #here we have to find the z range corresponding to this situation
                    # grab the number of visits from the other field
                    """
                    ik = scen['fieldname'] == 'LSSTDDF'
                    scensel = scen[ik]
                    cadb = scensel['cadence'][0]
                    """
                    nvisits = self.interp['{}'.format(cad)](z)
                    zvals = self.reverse_interp['{}'.format(cad)](2.*nvisits)

                nvisits = self.interp['{}'.format(cad)](zvals)
                nvisits*=field['season_length']*field['Nfields']*field['Nseasons']*30./field['cadence']
                #frac = nvisits/Nvisits
                print(nvisits)
                if arr_visits is None:
                    arr_visits = np.array(nvisits)
                    #arr_frac = np.array(frac)
                else:
                    arr_visits += nvisits
                    #arr_frac +=frac
                if res_scen is None:
                    res_scen = np.array(zvals,dtype=[('zlim_{}'.format(fieldname),'f8')])
                    res_scen = rf.append_fields(res_scen,'nvisits_{}'.format(fieldname),nvisits)
                else:
                    res_scen = rf.append_fields(res_scen,'zlim_{}'.format(fieldname),zvals)
                    res_scen = rf.append_fields(res_scen,'nvisits_{}'.format(fieldname),nvisits)
    
            res_scen = rf.append_fields(res_scen,'nvisits',arr_visits)
            #res_scen = rf.append_fields(res_scen,'frac_DDF',arr_frac)
            res_scen = rf.append_fields(res_scen,'scenario',[key]*len(res_scen))

            if res_tot is None:
                res_tot = res_scen
            else:
                res_tot = np.concatenate((res_tot,res_scen))
    
        return res_tot

    def plot(self,res_tot,Nvisits):


        fig, ax = plt.subplots()

        for scen in np.unique(res_tot['scenario']):
            iu = res_tot['scenario']==scen
            sel = res_tot[iu]
            sel.sort(order='nvisits')
            ia = (sel['zlim_ADFS']>0)&(sel['nvisits_ADFS']>0)
            sela = sel[ia]
            sela.sort(order='zlim_ADFS')
            label ='{}-{}'.format(scen,'any field')
            if scen == 'scen5':
                label = '{}-{}'.format(scen,'ADFS')
            ax.plot(sela['zlim_ADFS'],100.*sela['nvisits']/Nvisits,label=label)
            if scen == 'scen5':
                label = '{}-{}'.format(scen,'LSSTDDF')
                ib = (sel['zlim_LSSTDDF']>0)&(sel['nvisits_LSSTDDF']>0)
                selb = sel[ib]
                selb.sort(order='zlim_LSSTDDF')
                ax.plot(selb['zlim_LSSTDDF'],100.*selb['nvisits']/Nvisits,label=label)
                print(selb[['zlim_LSSTDDF','nvisits_LSSTDDF','nvisits_ADFS','nvisits']])

        ax.set_ylim([0.,6.0])
        ax.set_xlim([0.3,0.85])
        ax.legend()
        ax.grid()
        ax.plot([0.3,0.85],[4.5,4.5],color='b',ls='--')
        ax.text(0.4,4.6,'AGN White paper (Nvisits)',color='b')
        ax.set_xlabel(r'$z_{lim}$')
        ax.set_ylabel(r'DD budget [%]')
        



#restframeBands()

#DDbudget_zlim()

#plt.show()

#dbDir = '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.3'
#dbExtens = 'db'

#dbName = 'baseline_v1.3_10yrs'

#n_clusters = 5

#ana = AnaOS(dbDir, dbName,dbExtens,n_clusters)

#print('Total number of visits',ana.nvisits_DD+ana.nvisits_WFD,1./(1.+ana.nvisits_WFD/ana.nvisits_DD))

#plt.show()

