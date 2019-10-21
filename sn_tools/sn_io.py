import h5py
from astropy.table import Table,vstack
import numpy as np
import pandas as pd
import os
import glob
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import interp1d

def geth5Data(name,thedir):
    
    sumName = 'Simu_{}.hdf5'.format(name)
    sumFile = h5py.File('{}/{}'.format(thedir,sumName), 'r')
    lcName = 'LC_{}.hdf5'.format(name)
    #lcFile = h5py.File('{}/{}'.format(thedir,lcName), 'r')
    key = list(sumFile.keys())[0]
    summary = Table.read(sumFile, path=key)
    return summary, lcName, key

def getLC(lcFile,id_h5):
    lc = Table.read(lcFile, path='lc_{}'.format(id_h5))
    return lc

def getFile(theDir, theName):

    theFile = h5py.File('{}/{}'.format(theDir,theName), 'r')

    return theFile

def selectIndiv(tab, field, refval):

    idx = np.abs(tab[field]-refval)<1.e-5

    return idx

def select(tab, names, val):

    idx = True
    for name in names:
        idx &= selectIndiv(tab,name,val[name])
    
    return tab[idx]

def loadFile(filename, objtype='pandasDataFrame'):

    name, ext = os.path.splitext(filename)

    if ext == '.npy':
        # numpy array in an npy file
        return np.load(filename)

    else:
        if ext == '.hdf5':
            # open the file
            f = h5py.File(filename, 'r')
            # get the keys
            keys = f.keys()
            if objtype == 'pandasDataFrame':
                res = pd.DataFrame()
            if objtype == 'astropyTable':
                res = Table()

            for kk in keys:
                # loop on the keys and concat objects
                # two possibilities: astropy table or pandas df
                if objtype == 'pandasDataFrame':
                    df = pd.read_hdf(filename, key=kk, mode='r')
                    res = pd.concat([res,df],sort=False)
                if objtype == 'astropyTable':
                    df = Table.read(filename, path=kk)
                    res = vstack([res,df])
            return res
        else:
            print(filename)
            print('unknown format: file will not be downloaded')
            return None

def loopStack(namelist,objtype='pandasDataFrame'):
    
    res = pd.DataFrame()
    if objtype == 'astropyTable':
        res = Table()
    if objtype == 'numpyArray':
        res = None

    for fname in namelist:
        tab = loadFile(fname,objtype)

        if objtype == 'pandasDataFrame':
            res = pd.concat([res,tab],sort=False)
        if objtype == 'astropyTable':
            res = vstack([res,tab])
        if objtype == 'numpyArray':
            if res is None:
                res = tab
            else:
                res = np.concatenate((res,tab))    
                
    return res

def convert_to_npy(namelist,objtype='pandasDataFrame'):

    return loopStack(namelist).to_records(index=False)


def convert_save(inputDir,dbName,metricName,outDir,fieldType='WFD',objtype='pandasDataFrame',unique=False,path=None,remove_galactic=False):

    search_path = path
    if path is None:
        search_path = '{}/*{}Metric_{}*'.format(inputDir,metricName,fieldType)
    print('search path',search_path)
    fileNames = glob.glob(search_path)

    print('files',fileNames)
    tab = convert_to_npy(fileNames,objtype=objtype)
    
    if unique:
        u, indices = np.unique(tab['healpixID'], return_index=True)
        tab = np.copy(tab[indices])

    if remove_galactic:
       tab = remove_galactic_area(tab)


    np.save('{}/{}_{}.npy'.format(outDir,dbName,metricName),np.copy(tab))


def remove_galactic_area(tab):

    rg = []

    for val in np.arange(0., 360., 0.5):
        c_gal = SkyCoord(val*u.degree, 0.*u.degree, frame='galactic')
        
        #c_gal = SkyCoord(val*u.degree, 0.*u.degree,
        #                frame='geocentrictrueecliptic')
        c = c_gal.transform_to('icrs')
        rg.append(('Galactic plane', c.ra.degree, c.dec.degree))
            
    resrg = np.rec.fromrecords(rg, names=['name','Ra','Dec'])
    idx = (resrg['Ra']>=200)&(resrg['Ra']<290.)      
    
    resrg = resrg[idx]

    interp_m = interp1d(resrg['Ra']-11.,resrg['Dec'],bounds_error=False,fill_value=0.)
    interp_p = interp1d(resrg['Ra']+11.,resrg['Dec'],bounds_error=False,fill_value=0.)
    
    ida = (tab['pixDec']<0.)&(tab['pixDec']>=-60.)
    sel = tab[ida]
    

    print('cut1',len(sel))
    idx = (sel['pixRa']>190)&(sel['pixRa']<295.)
    sela = sel[~idx]
    selb = sel[idx]
    
    selb.sort(order='pixRa')
    
    valDec_m = interp_m(np.copy(selb['pixRa']))
    
    idt_m = (selb['pixDec']-valDec_m)>0.
    
    selm = selb[idt_m]
    
    valDec_p = interp_p(np.copy(selb['pixRa']))
    
    idt_p = (selb['pixDec']-valDec_p)<0.

    selp = selb[idt_p]
    
    
    sela = np.concatenate((sela,selm))
    sela = np.concatenate((sela,selp))
    
    return sela
