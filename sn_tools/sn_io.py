import h5py
from astropy.table import Table
import numpy as np

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
