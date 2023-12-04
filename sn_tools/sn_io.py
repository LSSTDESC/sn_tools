import h5py
import astropy
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
from collections import MutableMapping


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
                    res = vstack([res, df], metadata_conflicts='silent')
            return res
        else:
            print(filename)
            print('unknown format: file will not be downloaded')
            return None


def loopStack_params(namelist,
                     params=dict(zip(['objtype'], ['pandasDataFrame'])),
                     j=0, output_q=None):
    """
    Function to load a file according to the type of data it contains

    Parameters
    ---------------
    namelist: list(str)
       list of the name of the files to consider
    objtype: str, opt
       type of the data the file contains (default: pandas DataFrame)
       possible values: pndasDataFrame, astropyTable, numpy array

    Returns
    -----------
    object of the file (stacked)

    """
    objtype = params['objtype']
    res = pd.DataFrame()
    if objtype == 'astropyTable':
        res = Table()
    if objtype == 'numpyArray':
        res = None

    for fname in namelist:
        tab = loadFile(fname, objtype)

        if objtype == 'pandasDataFrame':
            if 'correct_cols' in params.keys():
                ccols = params['correct_cols']
                for key, vals in ccols.items():
                    tab[key] = vals
            res = pd.concat([res, tab], sort=False)
        if objtype == 'astropyTable':
            res = vstack([res, tab])
        if objtype == 'numpyArray':
            if res is None:
                res = tab
            else:
                res = np.concatenate((res, tab))

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


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
        self.conn = sqlite3.connect(dbfile)
        self.cur = self.conn.cursor()
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

    def get_data(self, cols=None, sql=None, to_degrees=False, new_col_names=None, table='observations'):
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
        table: str, opt
          sql table to read (default: observations - used to be SummaryAllProp for fbs < v2.0

        """
        sql_request = 'SELECT '
        if cols is None:
            sql_request += ' * '
            self.cur.execute('PRAGMA TABLE_INFO({})'.format(table))
            r = self.cur.fetchall()
            cols = [c[1] for c in r]
        else:
            sql_request += ','.join(cols)

        sql_request += 'FROM {}'.format(table)
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

    def get_data_df(self, cols=None, sql=None, to_degrees=False, new_col_names=None, table='observations'):
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
        table: str, opt
          sql table to read (default: observations - used to be SummaryAllProp for fbs < v2.0

        """
        sql_request = 'SELECT '
        if cols is None:
            sql_request += ' * '
        else:
            sql_request += ','.join(cols)

        sql_request += 'FROM {}'.format(table)
        if sql is not None and len(sql) > 0:
            sql_request += ' WHERE ' + sql
        sql_request += ';'

        df = pd.read_sql_query(sql_request, self.conn)
        return df

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


def check_get_file(web_server, fDir, fName, fnewDir=None):
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
    if fnewDir is None:
        fnewDir = fDir
    cmd = 'wget --no-clobber --no-verbose {} --directory-prefix {}'.format(
        path, fnewDir)
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


def colName(names, list_search):
    """
    Function to get a name from a list

    Parameters
    ----------------
    names: list(str)
      list of names
    list_search: list(str)
      list of names to find in names

    Returns
    -----------
    the found name (str) or None

    """

    for vv in list_search:
        if vv in names:
            return vv

    return None


def recursive_items(dictionary):
    """
    Method to loop on a nested dictionnary

    Parameters
    --------------
    dictionnary: dict

    Returns
    ----------
    generator (yield) of the 'last' (key,value)

    """
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)


def recursive_keys(keys, dictionary):
    """
    Method to loop on a nested dictionnary

    Parameters
    --------------
    dictionnary: dict

    Returns
    ----------
    generator (yield) the keys

    """
    for key, value in dictionary.items():
        if type(value) is dict:
            keys.append(key)
            return recursive_keys(keys, value)
        else:
            return keys


def recursive_merge(d1, d2):
    """
    Update two dicts of dicts recursively, 
    if either mapping has leaves that are non-dicts, 
    the second s leaf overwrites the first s.

    Parameters
    ---------------
    d1, d2: dicts to merge

    Returns
    -----------
    merged dict
    """
    for k, v in d1.items():  # in Python 2, use .iteritems()!
        if k in d2:
            # this next check is the only difference!
            if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                d2[k] = recursive_merge(v, d2[k])
                # we could further check types and merge as appropriate here.
    d3 = d1.copy()
    d3.update(d2)
    return d3


def make_dict_from_config(path, config_file):
    """
    Function to make a dict from a configuration file

    Parameters
    ---------------
    path: str
      path dir to the config file
    config_file: str
       config file name

    Returns
    ----------
    dict with the config file infos

    """

    # open and load the file here
    ffile = open('{}/{}'.format(path, config_file), 'r')
    line = ffile.read().splitlines()
    ffile.close()

    # process the result
    params = {}
    for i, ll in enumerate(line):
        if ll != '' and ll[0] != '#':
            spla = ll.split('#')
            lspl = spla[0].split(' ')
            lspl = ' '.join(lspl).split()
            n = len(lspl)
            keym = ''
            lim = n-2
            for io, keya in enumerate([lspl[i] for i in range(lim)]):
                keym += keya
                if io != lim-1:
                    keym += '_'
            params[keym] = (lspl[n-1], lspl[n-2], spla[1])
    return params


def make_dict_from_optparse(thedict):
    """
    Function to make a nested dict from a dict
    The idea is to split the original dict key(delimiter: _)  to as many keys

    Parameters
    --------------
    thedict: dict

    Returns
    ----------
    final dict

    """

    params = {}
    for key, vals in thedict.items():
        lspl = key.split('_')
        n = len(lspl)
        mystr = ''
        myclose = ''
        for keya in [lspl[i] for i in range(n)]:
            mystr += '{\''+keya + '\':'
            myclose += ' }'

        if vals[0] != 'str':
            dd = '{} {} {}'.format(mystr, eval(
                '{}({})'.format(vals[0], vals[1])), myclose)
        else:
            dd = '{} \'{}\' {}'.format(mystr, vals[1], myclose)

        thedict = eval(dd)
        params = recursive_merge(params, thedict)

    return params


def decrypt_parser(parser):
    """
    Method to decrypt the parser help

    Parameters
    ---------------
    parser: optparse parser


    Returns
    ----------
    dict with decrypted infos

    """

    file_name = 'help_script.txt'
    file_object = open(file_name, 'w')
    parser.print_help(file_object)
    file_object.close()

    file = open(file_name, 'r')
    line = file.read().splitlines()

    params = {}
    for i, ll in enumerate(line):
        lolo = ' '.join(ll.split(' ')).split()
        if lolo and lolo[0][:2] == '--':
            key = lolo[0].split('--')[1]
            key = key.split('=')
            params[key[0]] = (key[1].split('/')[0], key[1].split('/')[1])

    return params


def make_dict_old(thedict, key, what, val):
    """
    Method to append to a dict infos
    The goal is in fine to create a yaml file

    Parameters
    --------------
    thedict: dict
      the dict to append to
    key: str
     key used to append
    what: (str,str)
      name and type
    val: str
      value

    Returns
    ----------
    thedict: dict
      resulting dict

    """

    keym = key.split('_')[0]
    keys = ''
    if '_' in key:
        keys = '_'.join(key.split('_')[1:])

    names = what[0].split(',')
    dtypes = what[1].split(',')
    val = val.split(',')

    valb = [val[i] if dtypes[i] == 'str' else eval(
        '{}({})'.format(dtypes[i], val[i])) for i in range(len(dtypes))]

    if keys == '':
        if names[0] != 'value':
            thedict[keym] = dict(zip(names, valb))
        else:
            thedict[keym] = valb[0]
    else:
        if keym not in thedict.keys():
            thedict[keym] = {}
        if names[0] != 'value':
            thedict[keym][keys] = dict(zip(names, valb))
        else:
            thedict[keym][keys] = valb[0]

    return thedict


def checkDir(outDir):
    """
    function to check whether a directory exist
    and create it if necessary
    """
    if not os.path.isdir(outDir):
        os.makedirs(outDir)


def add_parser(parser, confDict):
    for key, vals in confDict.items():
        vv = vals[1]
        if vals[0] != 'str':
            vv = eval('{}({})'.format(vals[0], vals[1]))
        parser.add_option('--{}'.format(key), help='{} [%default]'.format(
            vals[2]), default=vv, type=vals[0], metavar='')


class Read_LightCurve:

    def __init__(self, file_name='Data.hdf5', inputDir='dataLC'):
        """
        Parameters
        ----------
        file_name : str
            Name of the hdf5 file that you want to read.
        """
        self.file_name = file_name
        self.file = h5py.File('{}/{}'.format(inputDir, file_name), 'r')

    def get_path(self):
        """
        Method to return the list of keys of the hdf5 file

        Returns
        ----------
        list(str): list of keys (aka paths)

        """

        return list(self.file.keys())

    def get_table(self, path):
        """
        Parameters
        ----------
        path : str
            hdf5 path for light curve.

        Returns
        -------
        AstropyTable
            Returns the reading of an .hdf5 file as an AstropyTable.
        """

        tab = Table()
        try:
            tab = astropy.io.misc.hdf5.read_table_hdf5(
                self.file, path=path, character_as_bytes=False)
        except (OSError, KeyError):
            pass

        return tab

    def get_all_data(self):

        paths = self.get_path()
        paths = list(filter(lambda s: not ('meta_columns' in s), paths))
        metaTot = Table()
        for pp in paths:
            metaTable = self.get_table(path=pp)
            metadata = metaTable.meta
            lcDir = metadata['lc_dir']
            lcName = metadata['lc_fileName']
            metaTable.add_column(lcDir, name='lc_dir')
            metaTable.add_column(lcName, name='lc_fileName')
            metaTot = vstack([metaTot, metaTable])

        return metaTot


def get_meta(prodID, metaDir):
    """
    function to grab metadata from prodID and metaDir

    Parameters
    ----------
    prodID : str
        production ID.
    metaDir : str
        mate data (simu) rep.

    Returns
    -------
    metaTable : astropy table
        table of meta data.

    """

    import glob

    full_path = '{}/Simu_{}*.hdf5'.format(metaDir, prodID)
    fis = glob.glob(full_path)

    metaTable = Table()
    for io, fi in enumerate(fis):
        metaName = fi.split('/')[-1]
        meta = Read_LightCurve(file_name=metaName, inputDir=metaDir)
        metaTot = meta.get_all_data()
        metaTable = vstack([metaTable, metaTot], metadata_conflicts='silent')
    return metaTable


def load_SN(SNDir, SNFile):
    """
    Function to load SN from file

    Parameters
    ----------
    SNDir : str
        SN dir.
    SNFile : str
        SN file.

    Returns
    -------
    SN : astropy table
        table of SN.

    """

    SN = Table()

    from sn_tools.sn_io import loopStack
    path = '{}/{}'.format(SNDir, SNFile)
    SN = loopStack([path], 'astropyTable')

    return SN
