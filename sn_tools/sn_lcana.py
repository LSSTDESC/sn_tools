import numpy as np
import time
from astropy.table import Table, Column, vstack
import pprint
import multiprocessing

def sigma_x0_x1_color_multi(lc,params=['x0','x1','color'],nproc=1):
    
    # stack all lc and start filling output table
    
    index = np.lexsort((lc['daymax'],lc['z']))
    lc = lc[index]
    #print(lc.dtype)

    restab = Table()
    zu = np.unique(lc['z'])
    zmin = np.min(zu)
    zmax = np.max(zu)
   
    batch = list(np.linspace(zmin,zmax,nproc))
    #if nz not in batch:
    #    np.append(batch,nz)

    print(batch,len(batch))

    result_queue = multiprocessing.Queue()

    for j in range(nproc-1):
        print(j,j+1)
        idx = (lc['z']>=batch[j])&(lc['z']<batch[j+1])
        p = multiprocessing.Process(name='Subprocess-'+str(
                    j), target=fromFisher, args=(lc[idx],params, j, result_queue))
        p.start()

    resultdict = {}
    for j in range(nproc-1):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    for j in range(nproc-1):
        restab = vstack([restab,resultdict[j]])
    
    return restab

def fromFisher(lc,params, j=-1, output_q=None):

    restab = Table(np.unique(lc[['x1','color','season','pixRa','pixDec','z','daymax']]))
          
    valu = np.unique(lc['z','daymax'])
    diff = lc['daymax']-valu['daymax'][:, np.newaxis]
    flag = np.abs(diff) < 1.e-5
    diffb = lc['z']-valu['z'][:, np.newaxis]
    flag &= np.abs(diffb) < 1.e-5
    resu = np.ma.array(np.tile(lc, (len(valu), 1)), mask=~flag)
    parts = {}
    time_refa = time.time()
    for ia, vala in enumerate(params):
        for jb, valb in enumerate(params):
            if jb >= ia:
                #print('F_'+vala+valb,np.sum(resu['F_'+vala+valb],axis = 1))
                parts[ia, jb] = np.sum(
                    resu['F_'+vala+valb]/(resu['fluxerr']**2.), axis=1)
                
    print('there one', time.time()-time_refa)

    size = len(resu)
    Fisher_Big = np.zeros((3*size, 3*size))
    Big_Diag = np.zeros((3*size, 3*size))
    Big_Diag = []
    
    print('there two', time.time()-time_refa)
    time_refa = time.time()
    for iv in range(size):
        Fisher_Matrix = np.zeros((3, 3))
        for ia, vala in enumerate(params):
            for jb, valb in enumerate(params):
                if jb >= ia:
                    #Fisher_Matrix[ia,jb] = parts[ia,jb][iv]
                    Fisher_Big[ia+3*iv][jb+3 *
                                        iv] = parts[ia, jb][iv]

    Fisher_Big = Fisher_Big + np.triu(Fisher_Big,1).T
    Big_Diag = np.diag(np.linalg.inv(Fisher_Big))

    for ia, vala in enumerate(params):
        indices = range(ia, len(Big_Diag), 3)
        #print('test', ia, indices, vala,
        #      np.take(Big_Diag, indices))
        restab.add_column(
            Column(np.take(Big_Diag, indices), name='Cov_{}{}'.format(vala,vala)))
    print('resultat',restab)
    #return restab
    if output_q is not None:
        output_q.put({j: restab})
    else:
        return restab

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
