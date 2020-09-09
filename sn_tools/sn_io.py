import h5py
from astropy.table import Table, vstack
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


def append(metricTot, sel):
    """
    Function to concatenate a numpy array to another numpy array
    with the same structure

    Parameters
    ----------
    metricTot: numpy array
    sel: numpy array

    Returns
    -------
    concatenated numpy array

    """

    if metricTot is None:
        metricTot = np.copy(sel)
    else:
        metricTot = np.concatenate((metricTot, np.copy(sel)))

    return metricTot


def getMetricValues(dirFile, dbName, metricName, fieldType, nside):
    """
    Function to read and analyze files from the metrics

    Parameters
    ----------
    dirFile: str
     location directory of the files
    dbName: str
     name of the simulation to process
    metricName: str
     name of the metric
    nside: int
     nside paramater value (healpix)
    fields_DD: record array
     array of DD fields with the following columns:
     - name: name of the field
     - fieldId: Id of the field
     - RA: RA of the field
     - Dec: Dec of the field
     - fieldnum: field number
    """

    metricTot = None

    search_path = '{}/{}/{}/*{}Metric_{}*_nside_{}_*'.format(
        dirFile, dbName, metricName, metricName, fieldType, nside)

    fileNames = glob.glob(search_path)

    if fileNames:
        # plt.plot(metricValues['pixRA'],metricValues['pixDec'],'ko')
        # plt.show()
        # get the values from the metrics
        metricValues = np.array(loopStack(fileNames, 'astropyTable'))

        # analyze these values
        # tab = getVals(fields_DD, metricValues, dbName.ljust(adjl), nside)

        # plt.plot(sel['pixRA'],sel['pixDec'],'ko')
        # plt.show()

        metricTot = append(metricTot, metricValues)

        return metricTot


def geth5Data(name, thedir):
    """
    Function to load the content of hdf5 files

    Parameters
    ---------------
    name: str
      name of the file
    thedir: str
       directory where the file is located

    Returns
    -----------
    summary: astropy table
      summary of the production (LC)
    lcName: str
       name of the LC file (hdf5 format)
    key: list(str)
       list of keys to access LCs

    """

    sumName = 'Simu_{}.hdf5'.format(name)
    sumFile = h5py.File('{}/{}'.format(thedir, sumName), 'r')
    lcName = 'LC_{}.hdf5'.format(name)
    # lcFile = h5py.File('{}/{}'.format(thedir,lcName), 'r')
    key = list(sumFile.keys())[0]
    summary = Table.read(sumFile, path=key)
    return summary, lcName, key


def getLC(lcFile, id_h5):
    """
    Function to access a table in hdf5 file from a key

    Parameters
    ---------------
    lcFile: str
      name of the hdf5 file to access
    id_hdf5: str
      key to access

    Returns
    ----------
    astropy table

    """

    lc = Table.read(lcFile, path='lc_{}'.format(id_h5))
    return lc


def getFile(theDir, theName):
    """
    Function returning a pointer to a hdf5 file

    Parameters
    ---------------
    theDir: str
       location directory of the file
    theName: str
       name of the file

    Returns
    -----------
    pointer to the file
    """

    theFile = h5py.File('{}/{}'.format(theDir, theName), 'r')

    return theFile


def selectIndiv(tab, field, refval):
    """
    Method to perform a selection on a array of data

    Parameters
    ---------------
    tab: array
       data to select
    field: str
        name of the field (column) to select
    refval: float
        selection value

    Returns
    ----------
    index corresponding to the selection

    """
    idx = np.abs(tab[field]-refval) < 1.e-5

    return idx


def select(tab, names, val):
    """
    Method to perform a selection on a array of data

    Parameters
    ---------------
    tab: array
       data to select
    field: str
        name of the field (column) to select
    refval: float
        selection value

    Returns
    ----------
    selected tab

    """

    idx = True
    for name in names:
        idx &= selectIndiv(tab, name, val[name])

    return tab[idx]


def loadFile(filename, objtype='pandasDataFrame'):
    """
    Function to load a file according to the type of data it contains

    Parameters
    ---------------
    filename: str
       name of the file to consider
    objtype: str
       type of the data the file contains
       possible values: pndasDataFrame, astropyTable, numpy array

    Returns
    -----------
    object of the file (stacked)

    """

    name, ext = os.path.splitext(filename)

    if ext == '.npy':
        # numpy array in an npy file
        return np.load(filename, allow_pickle=True)

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
                    res = pd.concat([res, df], sort=False)
                if objtype == 'astropyTable':
                    df = Table.read(filename, path=kk)
                    res = vstack([res, df])
            return res
        else:
            print(filename)
            print('unknown format: file will not be downloaded')
            return None


def loopStack(namelist, objtype='pandasDataFrame'):
    """
    Function to load a file according to the type of data it contains

    Parameters
    ---------------
    namelist: list(str)
       list of the name of the files to consider
    objtype: str
       type of the data the file contains
       possible values: pndasDataFrame, astropyTable, numpy array

    Returns
    -----------
    object of the file (stacked)

    """
    res = pd.DataFrame()
    if objtype == 'astropyTable':
        res = Table()
    if objtype == 'numpyArray':
        res = None

    for fname in namelist:
        tab = loadFile(fname, objtype)

        if objtype == 'pandasDataFrame':
            res = pd.concat([res, tab], sort=False)
        if objtype == 'astropyTable':
            res = vstack([res, tab])
        if objtype == 'numpyArray':
            if res is None:
                res = tab
            else:
                res = np.concatenate((res, tab))

    return res


def convert_to_npy(namelist, objtype='pandasDataFrame'):
    """
    Function to stacked pandas df to a numpy array

    Parameters
    ---------------
    namelist: list(str)
      list of the filenames to process
    objtype: str, opt
      type of object


    Returns
    ----------
    numpy array 

    """
    res = loopStack(namelist, objtype=objtype)

    if objtype == 'pandasDatFrame':
        return res.to_records(index=False)

    return None


def convert_save(inputDir, dbName, metricName, outDir, fieldType='WFD', objtype='pandasDataFrame', unique=False, path=None, remove_galactic=False):
    """
    Function to convert (*hdf5) and save metric output as numpy array file

    Parameters
    ---------------
    inputDir: str
      location directory of the input file
    dbName: str
       database name file (from the scheduler)
    metricName: str
       name of the metric
    outDir: str
       output directory
    fieldType: str, opt
       type of field to consider (default: WFD)
    objtype: str, opt
       type of the metric output (default: pandas df)
    unique: bool, opt
       to remove potential duplicate (default: False)
    path: str, opt
       path where original hdf5 files are located (default: None)
    remove_galactic: bool, opt
      to remove the galactic plane (default: False)

    """
    search_path = path
    if path is None:
        search_path = '{}/*{}Metric_{}*.hdf5'.format(
            inputDir, metricName, fieldType)
    print('search path', search_path)
    fileNames = glob.glob(search_path)

    print('files', fileNames)
    tab = convert_to_npy(fileNames, objtype=objtype)

    # Be carefull here...
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

    np.save('{}/{}_{}.npy'.format(outDir, dbName, metricName), np.copy(tab))


def remove_galactic_area(tab):
    """
    Function to exclude the galactic plane from data

    Parameters
    ---------------
    tab: array
      array of data

    Returns
    -----------
    Data without the galactic plane

    """

    rg = []

    for val in np.arange(0., 360., 0.5):
        c_gal = SkyCoord(val*u.degree, 0.*u.degree, frame='galactic')

        # c_gal = SkyCoord(val*u.degree, 0.*u.degree,
        #                frame='geocentrictrueecliptic')
        c = c_gal.transform_to('icrs')
        rg.append(('Galactic plane', c.ra.degree, c.dec.degree))

    resrg = np.rec.fromrecords(rg, names=['name', 'RA', 'Dec'])
    idx = (resrg['RA'] >= 200) & (resrg['RA'] < 290.)

    resrg = resrg[idx]

    interp_m = interp1d(resrg['RA']-11., resrg['Dec'],
                        bounds_error=False, fill_value=0.)
    interp_p = interp1d(resrg['RA']+11., resrg['Dec'],
                        bounds_error=False, fill_value=0.)

    # ida = (tab['pixDec']<0.)&(tab['pixDec']>=-60.)
    # sel = tab[ida]
    sel = tab

    idx = (sel['pixRA'] > 190) & (sel['pixRA'] < 295.)
    sela = sel[~idx]
    selb = sel[idx]

    selb.sort(order='pixRA')

    valDec_m = interp_m(np.copy(selb['pixRA']))

    idt_m = (selb['pixDec']-valDec_m) > 0.

    selm = selb[idt_m]

    valDec_p = interp_p(np.copy(selb['pixRA']))

    idt_p = (selb['pixDec']-valDec_p) < 0.

    selp = selb[idt_p]

    sela = np.concatenate((sela, selm))
    sela = np.concatenate((sela, selp))

    return sela


class Read_Sqlite:
    """
    Class to read sqlite file from scheduler (*.db) and convert it to numpy array

    Parameters
    ---------------
    dbFile: str
       name of the file to convert
    sel: selection applied

    """

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
        """
        Method to perform selection on the database

        Parameters
        ---------------
        sel: selection

        Returns
        -----------
        selected database

        """

        sql = ''
        if 'year' in sel and sel['year'] is not None:
            y = sel['year']
            sql += 'night > %i and night < %i' % ((y-1)*365.25, y*365.25)
        if 'mjd_min' in sel and sel['mjd_min'] is not None:
            if len(sql) > 0:
                sql += ' and '
            sql += 'observationStartMJD > %f' % sel['mjd_min']
        if 'mjd_max' in sel and sel['mjd_max'] is not None:
            if len(sql) > 0:
                sql += ' and '
            sql += 'observationStartMJD < %f' % sel['mjd_max']
        if 'proposalId' in sel and sel['proposalId'] is not None:
            if len(sql) > 0:
                sql += ' and '
            sql += 'proposalId=%d' % sel['proposalId']
        return sql

    def get_data(self, cols=None, sql=None, to_degrees=False, new_col_names=None):
        """
        Method to get the contents of the SQL database dumped into a numpy rec array

        Parameters
        ---------------
        cols: list (str), opt
          list of cols to consider (default: None = all cols are considered)
        sql: str, opt
          selection for sql (default: None)
        to_degrees: bool, opt
          to convert rad to degrees (default: False)
        new_col_names: bool, opt
          to rename the columns (default: None)

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
        logging.info('done.')

        if to_degrees:
            d['fieldRA'] *= (180. / np.pi)
            d['fieldDec'] *= (180. / np.pi)

        if new_col_names is not None:
            self.update_col_names(d, new_col_names)

        return d

    def update_col_names(self, d, new_col_names):
        """
        Method to rename columns

        Parameters
        ---------------
        d: numpy array
          data to process
        new_col_names: list(str)
          list of the new column names

        Returns
        -----------
        numpy array with new column names

        """
        names = list(d.dtype.names)
        d.dtype.names = [new_col_names[n]
                         if n in new_col_names else n for n in d.dtype.names]
        return d


def getObservations(dbDir, dbName, dbExtens):
    """
    Function to extract observations: 
    from an initial db from the scheduler, get a numpy array of observations

    Parameters
    ----------------
    dbDir: str
       location directory of the db
    dbName: str
       name of the database
    dbExtens: str
      extension of the db: .db or .npy

    Returns
    -----------
    numpy array of observations

    """

    dbFullName = '{}/{}.{}'.format(dbDir, dbName, dbExtens)
    # if extension is npy -> load
    if dbExtens == 'npy':
        observations = np.load(dbFullName, allow_pickle=True)
    else:
        # db as input-> need to transform as npy
        # print('looking for',dbFullName)
        keymap = {'observationStartMJD': 'mjd',
                  'filter': 'band',
                  'visitExposureTime': 'exptime',
                  'skyBrightness': 'sky',
                  'fieldRA': 'RA',
                  'fieldDec': 'Dec', }

        reader = Read_Sqlite(dbFullName)
        # sql = reader.sql_selection(None)
        observations = reader.get_data(cols=None, sql='',
                                       to_degrees=False,
                                       new_col_names=keymap)

        #save this file on disk if it does not exist
        outDir = dbDir.replace('/db','/npy')
        if not os.path.isdir(outDir):
            os.mkdir(outDir)

        path = '{}/{}.npy'.format(outDir,dbName)
        if not os.path.isfile(path):
            np.save(path,observations)
 
    return observations


def check_get_file(web_server, fDir, fName):
    """
    Function checking if a file is available
    If not, grab it from a web server

    Parameters
    ---------------
    web_server: str
       web server name
    fDir: str
      location dir of the file
    fName: str
      name of the file

    """

    if os.path.isfile('{}/{}'.format(fDir, fName)):
        return

    path = '{}/{}/{}'.format(web_server, fDir, fName)
    cmd = 'wget --no-clobber --no-verbose {} --directory-prefix {}'.format(
        path, fDir)
    os.system(cmd)


def check_get_dir(web_server, fDir, fName):
    """
    Function checking if a dir is available
    If not, grab it from a web server

    Parameters
    ---------------
    web_server: str
       web server name
    fDir: str
      name of the dir on the server
    fName: str
      name of the dir 

    """

    if not os.path.exists(fName):
        fullname = '{}/{}'.format(web_server, fDir)
        print('wget path:', fullname)
        cmd = 'wget --no-verbose --recursive {} --directory-prefix={} --no-clobber --no-parent -nH --cut-dirs=3 -R \'index.html*\''.format(
            fullname+'/', fName)
        os.system(cmd)


def dustmaps(dustDir):
    """
    method to grab dustmaps
    Dust maps will be placed in dustDir

    Parameters
    ---------------
    dustDir: str
       dir where maps will be copied

    """
    from dustmaps.config import config
    config['data_dir'] = dustDir
    import dustmaps.sfd
    dustmaps.sfd.fetch()
