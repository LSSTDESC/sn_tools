import numpy as np
import time
from astropy.table import Table, Column, vstack
import pprint
import multiprocessing
from sn_tools.sn_io import geth5Data, getLC, getFile
import os

def sigma_x0_x1_color_multi(lc,params=['x0','x1','color'],nproc=1):
    
    # stack all lc and start filling output table
    
    index = np.lexsort((lc['daymax'],lc['z']))
    lc = lc[index]
    #print(lc.dtype)

    restab = Table()
    """
    zu = np.unique(lc['z'])
    zmin = np.min(zu)
    zmax = np.max(zu)
   
    if nproc > len(zu):
        nproc = len(zu)

    print('alors',len(zu),nproc)
    batch = list(np.linspace(zmin,zmax,nproc))
    print('hello batch',zmin,zmax,nproc)
    """

    nbatch = len(lc)
    delta = nproc
    if nproc > 1:
        delta = int(delta/(nproc))

    batch = range(0, nbatch, delta)
    if nbatch not in batch:
        batch = np.append(batch, nbatch)

    
    #print('hello batch',zmin,zmax,nproc)
    #if nz not in batch:
    #    np.append(batch,nz)

    print(batch,len(batch))

    result_queue = multiprocessing.Queue()

    for j in range(nproc-1):
        print(j,j+1)
        #idx = (lc['z']>=batch[j])&(lc['z']<batch[j+1])
        #p = multiprocessing.Process(name='Subprocess-'+str(
        #            j), target=sigma_x0_x1_color, args=(lc[idx],params, j, result_queue))
        ida = batch[j]
        idb = batch[j+1]
        p = multiprocessing.Process(name='Subprocess-'+str(
            j), target=sigma_x0_x1_color, args=(lc[ida:idb],params, j, result_queue))
        p.start()

    resultdict = {}
    for j in range(nproc-1):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    for j in range(nproc-1):
        restab = vstack([restab,resultdict[j]])
    
    return restab
def sigma_x0_x1_color(resu,restab,params=['x0','x1','color']):
#def sigma_x0_x1_color(lc,params=['x0','x1','color'], j=-1, output_q=None):

    """
    restab = Table(np.unique(lc[['season','pixRA','pixDec','z','daymax']]))
          
    valu = np.unique(lc['z','daymax'])
    diff = lc['daymax']-valu['daymax'][:, np.newaxis]
    flag = np.abs(diff) < 1.e-5
    diffb = lc['z']-valu['z'][:, np.newaxis]
    flag &= np.abs(diffb) < 1.e-5
    resu = np.ma.array(np.tile(lc, (len(valu), 1)), mask=~flag)
    """
    parts = {}
    time_refa = time.time()
    for ia, vala in enumerate(params):
        for jb, valb in enumerate(params):
            if jb >= ia:
                #print('F_'+vala+valb,np.sum(resu['F_'+vala+valb],axis = 1))
                parts[ia, jb] = np.sum(
                    resu['F_'+vala+valb]/(resu['fluxerr']**2.), axis=1)
                
    #print('there one', time.time()-time_refa)

    #print(parts)
    size = len(resu)
    Fisher_Big = np.zeros((3*size, 3*size))
    Big_Diag = np.zeros((3*size, 3*size))
    Big_Diag = []
    
    #print('there two', time.time()-time_refa)
    time_refa = time.time()
    for iv in range(size):
        Fisher_Matrix = np.zeros((3, 3))
        for ia, vala in enumerate(params):
            for jb, valb in enumerate(params):
                if jb >= ia:
                    #Fisher_Matrix[ia,jb] = parts[ia,jb][iv]
                    Fisher_Big[ia+3*iv][jb+3 *
                                        iv] = parts[ia, jb][iv]

    #pprint.pprint(Fisher_Big)

    Fisher_Big = Fisher_Big + np.triu(Fisher_Big,1).T
    Big_Diag = np.diag(np.linalg.inv(Fisher_Big))

    for ia, vala in enumerate(params):
        indices = range(ia, len(Big_Diag), 3)
        #print('test', ia, indices, vala,
        #      np.take(Big_Diag, indices))
        restab.add_column(
            Column(np.take(Big_Diag, indices), name='Cov_{}{}'.format(vala,vala)))
    #print('resultat',restab)
    #return restab
    """
    if output_q is not None:
        output_q.put({j: restab})
    else:
        return restab
    """
def sigma_x0_x1_color_loop(lcList,params=['x0','x1','color']):
    
    # stack all lc and start filling output table
   
    dictres = {}
    for ilc,lct in enumerate(lcList): 
       
        
        idsnr = lct['snr_m5']>=0.
        lc = lct[idsnr]
        print('hello',len(lc),len(lct))
        if len(lc) <5:
            continue

        for key, val in lc.meta.items():
            if key not in dictres.keys():
                dictres[key] = []
            dictres[key].append(val)

        print(lc[['band','phase','flux','fluxerr','F_x0x0','F_x1x1','F_colorcolor','snr_m5']])
        parts = np.zeros((3, 3))
        for ia, vala in enumerate(params):
            for jb, valb in enumerate(params):
                if jb >= ia:
                #print('F_'+vala+valb,np.sum(resu['F_'+vala+valb],axis = 1))
                    parts[ia, jb] = np.sum(
                        lc['F_'+vala+valb]/(lc['fluxerr']**2.))
                    #parts[ia, jb] = np.sum(
                    #    lc['F_'+vala+valb])
        
        parts = parts
        pprint.pprint(parts)
        parts = parts + np.triu(parts,1).T
        print('after')
        pprint.pprint(parts)

        mat = np.diag(np.linalg.inv(parts))

        for ia, vala in enumerate(params):
            indices = range(ia, len(mat), 3)
            name = 'Cov_{}{}'.format(vala,vala)
            if name not in dictres.keys():
                dictres[name] = []
            #print('alors',np.sqrt(np.take(mat, indices)))
            dictres[name].append(np.take(mat, indices)[0])

        #break

    print('go pal')
    print(dictres)
    restab = Table()

    for key,val in dictres.items():
        restab.add_column(
            Column(val,key))

    return restab

def calc_info(lc,params=['x0','x1','color'], j=-1, output_q=None):
    
    restab = Table(np.unique(lc[['season','pixRA','pixDec','z','daymax']]))
          
    valu = np.unique(lc['z','daymax'])
    diff = lc['daymax']-valu['daymax'][:, np.newaxis]
    flag = np.abs(diff) < 1.e-5
    diffb = lc['z']-valu['z'][:, np.newaxis]
    flag &= np.abs(diffb) < 1.e-5

    tile_band = np.tile(lc['band'], (len(valu), 1))
    tile_snr = np.tile(lc['snr_m5'], (len(valu), 1))
    tile_flux = np.tile(lc['flux_e_sec'], (len(valu), 1))
    tile_flux_5 = np.tile(lc['flux_5'], (len(valu), 1))
    difftime = lc['time']-lc['daymax']
    tile_diff = np.tile(difftime, (len(valu), 1))

    flagp = tile_diff>=0.
    flagn = tile_diff<0.
   
    for key,fl in dict(zip(['aft','bef'],[flagp, flagn])).items():
        fflag = np.copy(flag)
        fflag &= fl
        ma_diff = np.ma.array(tile_diff, mask=~fflag)
        count = ma_diff.count(axis=1)
        restab.add_column(Column(count,name='N_{}'.format(key)))

    for band in 'ugrizy':
        maskb = np.copy(flag)
        maskb &= tile_band=='LSST::'+band
        ma_band = np.ma.array(tile_snr, mask=~maskb,fill_value=0.).filled()
        snr_band = np.sqrt(np.sum(ma_band**2, axis=1))
        ratflux = tile_flux/tile_flux_5
        ma_flux = np.ma.array(ratflux, mask=~maskb,fill_value=0.).filled()
        snr_band_5 = 5.*np.sqrt(np.sum(ma_flux**2, axis=1))
        restab.add_column(Column(snr_band,name='snr_{}'.format(band)))
        restab.add_column(Column(snr_band_5,name='snr_5_{}'.format(band)))
        for key,fl in dict(zip(['aft','bef'],[flagp, flagn])).items():
            fflag = np.copy(maskb)
            fflag &= fl
            ma_diff = np.ma.array(tile_diff, mask=~fflag)
            count = ma_diff.count(axis=1)
            restab.add_column(Column(count,name='N_{}_{}'.format(key,band)))


    #Select LCs with a sufficient number of LC points before and after max

    idx = (restab['N_bef']>=2)&(restab['N_aft']>=5)
    restab_good = Table(restab[idx])
    restab_bad = Table(restab[~idx])
    for par in params:
        restab_bad.add_column(Column([100.]*len(restab_bad),name='Cov_{}{}'.format(par,par)))

    valu = np.unique(restab_good['z','daymax'])
    diff = lc['daymax']-valu['daymax'][:, np.newaxis]
    flag = np.abs(diff) < 1.e-5
    diffb = lc['z']-valu['z'][:, np.newaxis]
    flag &= np.abs(diffb) < 1.e-5
    resu = np.ma.array(np.tile(lc, (len(valu), 1)), mask=~flag)

    restab = restab_bad
    if len(restab_good) > 0:

        sigma_x0_x1_color(resu,restab_good,params=['x0','x1','color'])
        
        restab = vstack([restab,restab_good])


    if output_q is not None:
        output_q.put({j: restab})
    else:
        return restab

class LCtoSN:
    def __init__(self, inputDir,outputDir,procId,tempDir='temp',nproc=1):

        self.inputDir = inputDir
        self.outputDir = outputDir
        self.procId = procId
        self.nproc = nproc
        self.tempDir = tempDir
        
        self.summary, self.lcName, self.keyfile = geth5Data(procId,inputDir)

        print(np.unique(self.summary[['season','pixRA','pixDec']]))

        list_sn = list(np.unique(self.summary[['season','pixRA','pixDec']]))

        print('Number of supernovae to process',len(list_sn))
        self.process()

    def process(self):

        groups = self.summary.group_by(['season','z','pixRA','pixDec'])
        indices = groups.groups.indices
        ngroups = len(indices)-1
        delta = ngroups
        if self.nproc > 1:
            delta = int(delta/(nproc))

        batch = range(0, ngroups, delta)

        if ngroups not in batch:
            batch = np.append(batch, ngroups)

        batch = batch.tolist()
        if batch[-1]-batch[-2]<= 2:
            batch.remove(batch[-2])

        result_queue = multiprocessing.Queue()

        for j in range(len(batch)-1):
            #for j in range(9,10):
            ida = batch[j]
            idb = batch[j+1]
            print('go', ida, idb)
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.lcLoop, args=(
                groups,batch[j],batch[j+1],self.lcName, self.inputDir,self.tempDir,j, result_queue))
            p.start()


        resultdict = {}
        for i in range(len(batch)-1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()


        for key, vals in resultdict.items():
            restot = vstack([restot,vals])


        # save output in npy file
        np.save('{}/SN_{}.npy'.format(self.outDir,self.procId),restot)

    def lcLoop(self,group,ida, idb,lcName,dirFile,tempDir,j=0, output_q=None):

        newname = lcName.replace('.hdf5','_{}.hdf5'.format(j))
   
        cmd = 'scp {}/{} {}/{}'.format(dirFile,lcName,tempDir,newname)

        print('cmd',cmd)
        os.system(cmd)
        lcFile = getFile(tempDir,newname)

        resfi = Table()

        for ind in range(ida,idb,1):
            grp = group.groups[ind]
            res = self.calcSN(grp,lcFile)
            resfi = vstack([resfi,res])
        
        cmd = 'rm {}/{}'.format(tempDir,newname)
        os.system(cmd)   

        if output_q is not None:
            return output_q.put({j: resfi})
        else:
            return resfi    


    def calcSN(self,grp,lcFile):

        all_lc = Table()
        x1 = np.unique(grp['x1'])[0]
        color = np.unique(grp['color'])[0]
        pixRA = np.unique(grp['pixRA'])[0]
        pixDec = np.unique(grp['pixDec'])[0]
        z = np.unique(grp['z'])[0]
   
   
        for index in grp['id_hdf5']:
            all_lc = vstack([all_lc,getLC(lcFile,index)],metadata_conflicts='silent')
        res = calc_info(all_lc)
    
        return res
