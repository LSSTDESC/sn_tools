import h5py
from astropy.table import Table,vstack
import numpy as np
import pandas as pd
import os
import glob
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import interp1d
import numpy.lib.recfunctions as rf
import sqlite3
import logging

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
    
    #Be carefull here...
    # healpixID is sometimes healpixId
    # so add healpixID if not present

    healpixID = 'healpixID'
    if 'healpixId' in tab.dtype.names:
        healpixID = 'healpixId'


    if unique:
        u, indices = np.unique(tab[healpixID], return_index=True)
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
    
    #ida = (tab['pixDec']<0.)&(tab['pixDec']>=-60.)
    #sel = tab[ida]
    sel = tab

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


class Read_Sqlite:
    
    def __init__(self, dbfile, **sel):
        """
        """
        self.dbfile = dbfile
        conn = sqlite3.connect(dbfile)
        self.cur = conn.cursor()
        # get the table list
        self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.tables = self.cur.fetchall()
        self.sql = self.sql_selection(**sel)
        #        self.data = self.groom_data(self.get_data(sql))
        
    def sql_selection(self, **sel):
        sql = ''
        if 'year' in sel and sel['year'] is not None:
            y = sel['year']
            sql += 'night > %i and night < %i' % ((y-1)*365.25, y*365.25)
        if 'mjd_min' in sel and sel['mjd_min'] is not None:
            if len(sql) > 0: sql += ' and '
            sql += 'observationStartMJD > %f' % sel['mjd_min']
        if 'mjd_max' in sel and sel['mjd_max'] is not None:
            if len(sql) > 0: sql += ' and '
            sql += 'observationStartMJD < %f' % sel['mjd_max']            
        if 'proposalId' in sel and sel['proposalId'] is not None:
            if len(sql) > 0: sql += ' and '
            sql += 'proposalId=%d' % sel['proposalId']
        return sql
        
    def get_data(self, cols=None, sql=None, to_degrees=False, new_col_names=None):
        """
        Get the contents of the SQL database dumped into a numpy rec array
        """
        sql_request = 'SELECT '
        if cols is None:
            sql_request += ' * '            
            self.cur.execute('PRAGMA TABLE_INFO(SummaryAllProps)')
            r = self.cur.fetchall()
            cols = [c[1] for c in r]
        else:
            sql_request += ','.join(cols)
            
        sql_request += 'FROM SummaryAllProps'
        if sql is not None and len(sql) > 0:
            sql_request += ' WHERE ' + sql
        sql_request += ';'
        
        logging.info('fetching data from db')
        logging.info('request: %s' % sql_request)
        self.cur.execute(sql_request)
        rows = self.cur.fetchall()
        logging.info('done. %d entries' % len(rows))

        logging.info('converting dump into numpy array')
        colnames = [str(c) for c in cols]
        d = np.rec.fromrecords(rows, names=colnames)
        """
        logging.info('update kAtm values')
        katm = np.zeros(len(d), dtype='float64')        
        logging.info('extending output array')
        d = croaks.rec_append_fields(d, data=[katm], names=['kAtm'])
        for b in kAtm.keys():
            d['kAtm'][d['filter'] == b] = kAtm[b]
        """
        logging.info('done.')

        if to_degrees:
            d['fieldRA'] *= (180. / np.pi)
            d['fieldDec'] *= (180. / np.pi)

        if new_col_names is not None:
            self.update_col_names(d, new_col_names)
            
        return d

    def update_col_names(self, d, new_col_names):
        names = list(d.dtype.names)
        d.dtype.names = [new_col_names[n] if n in new_col_names else n for n in d.dtype.names]
        return d

def getObservations(dbDir, dbName,dbExtens):

    dbFullName = '{}/{}.{}'.format(dbDir, dbName,dbExtens)
    # if extension is npy -> load
    if dbExtens == 'npy':
        observations = np.load(dbFullName)
    else:
        #db as input-> need to transform as npy
        print('looking for',dbFullName)
        keymap = {'observationStartMJD': 'mjd',
                  'filter': 'band',
                  'visitExposureTime': 'exptime',
                  'skyBrightness': 'sky',
                  'fieldRA': 'Ra',
                  'fieldDec': 'Dec',}

        reader = Read_Sqlite(dbFullName)
        #sql = reader.sql_selection(None)
        observations = reader.get_data(cols=None, sql='',
                                       to_degrees=False,
                                       new_col_names=keymap)
    
    return observations
