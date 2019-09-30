import h5py
from astropy.table import Table,vstack
import numpy as np
import pandas as pd
import os

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

def convert_DF_npy(namelist):

    return loopStack(namelist).to_records(index=False)
