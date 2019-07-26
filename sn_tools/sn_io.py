import h5py
from astropy.table import Table

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
