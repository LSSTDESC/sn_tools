import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rf
from shapely import geometry
from shapely import affinity
import shapely.vectorized
from astropy_healpix import HEALPix
from astropy import units as u
from descartes.patch import PolygonPatch
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from matplotlib.patches import Polygon
from shapely.geometry import Point
import pandas as pd
import time
import h5py
from astropy.table import Table
import multiprocessing


def getPix(nside, fieldRA, fieldDec):
    """
    Function returning pixRa, pixDec and healpixId

    Parameters
    ---------------
    nside: int
      nside value for healpix
    fieldRa: float
      field Ra
    fieldDec: float
      field Dec

    Returns
    ----------
    healpixId: Id from healpix
    pixRa: Ra from healpix
    pixDec: Dec from healpix

    """

    healpixId = hp.ang2pix(nside, fieldRA,
                           fieldDec, nest=True, lonlat=True)
    coord = hp.pix2ang(nside,
                       healpixId, nest=True, lonlat=True)

    pixRa = coord[0]
    pixDec = coord[1]

    return healpixId, pixRa, pixDec


def pavingSky(ramin, ramax, decmin, decmax, radius):
    """ Function to perform a paving of the sky

    Parameters
    --------------
    ramin: float
      min ra of the region to pave
    ramax: float
      max ra of the region to pave
    decmin: float
      min dec of the region to pave
    decmax: float
      max dec of the region to pave
    radius: float
      distance reference for the paving; correspond to the radius of a regular hexagon


    Returns
    -----------
    tab: record array of the center position (Ra,Dec) and the radius
    """
    ramin = ramin+radius*np.sqrt(3.)/2.
    decstep = 1.5*radius
    rastep = radius*np.sqrt(3.)
    shift = radius*np.sqrt(3.)/2.
    decrefs = np.arange(decmax, decmin, -decstep)
    r = []
    # this is to pave the sky loopin on Ra, same Dec
    for i in range(len(decrefs)):
        shift_i = shift*(i % 2)
        for ra in list(np.arange(ramin-shift_i, ramax, rastep)):
            r.append((ra, decrefs[i], radius))

    res = np.rec.fromrecords(r, names=['Ra', 'Dec', 'radius'])
    return res


def area(minRa, maxRa, minDec, maxDec):

    return dict(zip(['minRa', 'maxRa', 'minDec', 'maxDec'], [minRa, maxRa, minDec, maxDec]))


def areap(minRa, maxRa, minDec, maxDec):

    poly = [[minRa, minDec], [minRa, maxDec], [maxRa, maxDec], [maxRa, minDec]]

    return geometry.Polygon(poly)


def dataInside(data, Ra, Dec, widthRa, widthDec, RaCol='fieldRa', DecCol='fieldDec', ax=None):

    #time_ref = time.time()
    minRa = Ra-widthRa
    maxRa = Ra+widthRa

    minDec = Dec-widthDec
    maxDec = Dec+widthDec

    # Create area from these
    # Ra is in [0.,360.]
    # special treatement near Ra~0

    areaList = []
    if ax is not None:
        areapList = []

    if maxRa >= 360.:
        # in that case two areas necessary
        #print('hello poly',minRa,360., minDec, maxDec,0.0, maxRa-360., minDec, maxDec)
        areaList.append(area(minRa, 360., minDec, maxDec))
        areaList.append(area(0.0, maxRa-360., minDec, maxDec))
        if ax is not None:
            areapList.append(areap(minRa, 360., minDec, maxDec))
            areapList.append(areap(0.0, maxRa-360., minDec, maxDec))
    else:
        if minRa < 0.:
            # in that case two areas necessary
            areaList.append(area(minRa+360., 360., minDec, maxDec))
            areaList.append(area(-1.e-8, maxRa, minDec, maxDec))
            #areapList.append(areap(minRa+360., 360., minDec, maxDec))
            #areapList.append(areap(-1.e-8, maxRa, minDec, maxDec))
        else:
            areaList.append(area(minRa, maxRa, minDec, maxDec))
            #areapList.append(areap(minRa, maxRa, minDec, maxDec))
    #print('dti a',time.time()-time_ref)
    #time_ref = time.time()
    if ax is not None:
        for poly in areapList:
            pf = PolygonPatch(poly, facecolor=(
                0, 0, 0, 0), edgecolor='red')
            ax.add_patch(pf)
        ax.plot(data[RaCol], data[DecCol], 'ko')

    # select data inside this area
    """
    dataSel = None
    x = data[RaCol]
    y = data[DecCol]
    for poly in areaList:
        idf = shapely.vectorized.contains(poly, x, y)
        if len(data[idf]) > 0.:
            if dataSel is None:
                dataSel = data[idf]
            else:
                dataSel = np.concatenate((dataSel, data[idf]))
    print('dti b',time.time()-time_ref)
    """
    #time_ref = time.time()

    dataSel = None
    for areal in areaList:
        idf = (data[RaCol] >= areal['minRa']) & (data[RaCol] <= areal['maxRa'])
        idf &= (data[DecCol] >= areal['minDec']) & (
            data[DecCol] <= areal['maxDec'])
        if len(data[idf]) > 0.:
            if dataSel is None:
                dataSel = data[idf]
            else:
                dataSel = np.concatenate((dataSel, data[idf]))

    # print('datasel',areaList,dataSel)
    #print('dti c',time.time()-time_ref,dataSel)

    return dataSel


def proj_gnomonic_plane(lamb0, phi1, lamb, phi):

    cosc = np.sin(phi1)*np.sin(phi)
    cosc += np.cos(phi1)*np.cos(phi)*np.cos(lamb-lamb0)

    x = np.cos(phi)*np.sin(lamb-lamb0)
    x /= cosc

    y = np.cos(phi1)*np.sin(phi)
    y -= np.sin(phi1)*np.cos(phi)*np.cos(lamb-lamb0)

    y /= cosc

    return x, y


def proj_gnomonic_sphere(lamb0, phi, x, y):

    rho = (x**2+y**2)**0.5
    c = np.arctan(rho)
    print('c', rho, c, np.rad2deg(c))
    lamb = x*np.sin(c)
    lamb /= (rho*np.cos(phi)*np.cos(c)-y*np.sin(phi)*np.sin(c))
    lamb = lamb0+np.arctan(lamb)

    phi1 = np.cos(c)*np.sin(phi)
    phi1 += (y*np.sin(c)*np.cos(phi))/rho
    phi1 = np.arcsin(phi1)

    return lamb, phi1


def renameFields(tab):

    # print(tab.dtype)
    corresp = {}

    fillCorresp(tab, corresp, 'mjd', 'observationStartMJD')
    fillCorresp(tab, corresp, 'Ra', 'fieldRA')
    fillCorresp(tab, corresp, 'Dec', 'fieldDec')
    fillCorresp(tab, corresp, 'band', 'filter')
    fillCorresp(tab, corresp, 'exptime', 'visitExposureTime')
    fillCorresp(tab, corresp, 'nexp', 'numExposures')

    # print('alors',corresp)
    return rf.rename_fields(tab, corresp)


def fillCorresp(tab, corres, vara, varb):

    if vara in tab.dtype.names and varb not in tab.dtype.names:
        corres[vara] = varb


def pixelate(data, nside, RaCol='Ra', DecCol='Dec'):

    res = data.copy()
    npix = hp.nside2npix(nside)
    table = hp.ang2vec(res[RaCol], res[DecCol], lonlat=True)

    healpixs = hp.vec2pix(
        nside, table[:, 0], table[:, 1], table[:, 2], nest=True)
    coord = hp.pix2ang(nside, healpixs, nest=True, lonlat=True)

    coords = SkyCoord(coord[0], coord[1], unit='deg')
    sfd = SFDQuery()
    ebv = sfd(coords)
    res = rf.append_fields(res, 'healpixID', healpixs)
    res = rf.append_fields(res, 'pixRa', coord[0])
    res = rf.append_fields(res, 'pixDec', coord[1])
    res = rf.append_fields(res, 'ebv', ebv)

    return res


def season(obs, season_gap=80., mjdCol='observationStartMJD'):

    if 'season' in obs.dtype.names:
        return obs

    obs.sort(order=mjdCol)

    if len(obs) == 1:
        obs = np.atleast_1d(obs)
        obs = rf.append_fields([obs], 'season', [1.])
        return obs
    diff = obs[mjdCol][1:]-obs[mjdCol][:-1]

    flag = np.argwhere(diff > season_gap)
    if len(flag) > 0:
        seas = np.zeros((len(obs),))
        flag += 1
        seas[0:flag[0][0]] = 1
        for iflag in range(len(flag)-1):
            seas[flag[iflag][0]:flag[iflag+1][0]] = iflag+2
        seas[flag[-1][0]:] = len(flag)+1
        obs = rf.append_fields(obs, 'season', seas)
    else:
        obs = rf.append_fields(obs, 'season', [1.]*len(obs))

    return obs


def LSSTPointing(xc, yc, angle_rot=0., area=None, maxbound=None):
    """
    arr = [[3, 0], [12, 0], [12, 1], [13, 1], [13, 2], [14, 2], [14, 3], [15, 3],
           [15, 12], [14, 12], [14, 13], [13, 13], [
               13, 14], [12, 14], [12, 15],
           [3, 15], [3, 14], [2, 14], [2, 13], [1, 13], [1, 12], [0, 12],
           [0, 3], [1, 3], [1, 2], [2, 2], [2, 1], [3, 1]]
    """
    # this is a quarter of LSST FP (with corner rafts)
    arr = [[0.0, 7.5], [4.5, 7.5], [4.5, 6.5], [5.5, 6.5], [
        5.5, 5.5], [6.5, 5.5], [6.5, 4.5], [7.5, 4.5], [7.5, 0.0]]

    # this is a quarter of LSST FP (without corner rafts)
    arr = [[0.0, 7.5], [4.5, 7.5], [4.5, 4.5], [7.5, 4.5], [7.5, 0.0]]
    if maxbound is not None:
        arr = [[0.0, maxbound], [maxbound*4.5/7.5, maxbound], [maxbound*4.5 /
                                                               7.5, maxbound*4.5/7.5], [maxbound, maxbound*4.5/7.5], [maxbound, 0.0]]
    # symmetry I: y -> -y
    arrcp = list(arr)
    for val in arr[::-1]:
        if val[1] > 0.:
            arrcp.append([val[0], -val[1]])

    # symmetri II: x -> -x
    arr = list(arrcp)
    for val in arrcp[::-1]:
        if val[0] > 0.:
            arr.append([-val[0], val[1]])

    poly_orig = geometry.Polygon(arr)
    if area is not None:
        poly_orig = affinity.scale(poly_orig, xfact=np.sqrt(
            area/poly_orig.area), yfact=np.sqrt(area/poly_orig.area))

    rotated_poly = affinity.rotate(poly_orig, angle_rot)

    return affinity.translate(rotated_poly,
                              xoff=xc-rotated_poly.centroid.x,
                              yoff=yc-rotated_poly.centroid.y)


class ProcessArea:
    def __init__(self, nside, RaCol, DecCol, num, outDir, dbName, saveData):
        self.nside = nside
        """
        self.Ra = Ra
        self.Dec = Dec
        self.widthRa = widthRa
        self.widthDec = widthDec
        """
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.num = num
        self.outDir = outDir

        self.dbName = dbName
        self.saveData = saveData

        # get the LSST focal plane scale factor
        # corresponding to a sphere radius equal to one
        # (which is the default for gnomonic projections here

        fov = 9.62*(np.pi/180.)**2  # LSST fov in sr
        theta = 2.*np.arcsin(np.sqrt(fov/(4.*np.pi)))

        # if theta >= np.pi/2.:
        #    theta -= np.pi/2.
        #print('theta', theta, np.rad2deg(theta))
        self.fpscale = np.tan(theta)

    def __call__(self, data, metricList, Ra, Dec, widthRa, widthDec, ipoint, nodither, display=False):

        resfi = {}
        for metric in metricList:
            resfi[metric.name] = None
        # select data inside the area

        if display:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots()
            ax.plot(data[self.RaCol], data[self.DecCol], 'ko')
            plt.show()

        # select data inside an area centered in (Ra,Dec) with width (widthRa+1,widthDec+1)

        dataSel = dataInside(data, Ra, Dec, widthRa+1., widthDec+1.,
                             RaCol=self.RaCol, DecCol=self.DecCol)

        if display:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots()
            ax.plot(dataSel[self.RaCol], dataSel[self.DecCol], 'ko')
            plt.show()

        # Possible to remove DD dithering here
        # This is just to test impact of dithering on DDF

        if nodither:
            dataSel[self.RaCol] = np.mean(dataSel[self.RaCol])
            dataSel[self.DecCol] = np.mean(dataSel[self.DecCol])

        if dataSel is not None:

            # mv to panda df
            dataset = pd.DataFrame(np.copy(dataSel))

            # get central pixel ID
            healpixID = hp.ang2pix(self.nside, Ra,
                                   Dec, nest=True, lonlat=True)

            # get nearby pixels
            vec = hp.pix2vec(self.nside, healpixID, nest=True)
            healpixIDs = hp.query_disc(
                self.nside, vec, np.deg2rad(widthRa), inclusive=False, nest=True)

            # get pixel coordinates
            coords = hp.pix2ang(self.nside, healpixIDs, nest=True, lonlat=True)
            pixRa, pixDec = coords[0], coords[1]

            if display:
                print('number of pixels here', len(pixRa))
                import matplotlib.pyplot as plt
                plt.plot(pixRa, pixDec, 'ko')
                plt.show()

            # make groups by (Ra,dec)
            dataset = dataset.round({self.RaCol: 4, self.DecCol: 4})
            groups = dataset.groupby([self.RaCol, self.DecCol])

            if display:
                import matplotlib.pylab as plt
                for name, group in groups:
                    fig, ax = plt.subplots()
                    self.match(group, healpixIDs, pixRa, pixDec, name, ax=ax)
                    # ax.plot(dataset[self.RaCol],dataset[self.DecCol],'bs',mfc='None')
                    plt.show()

            # process pixels with data
            # match pixels to data
            time_ref = time.time()
            matched_pixels = groups.apply(
                lambda x: self.match(x, healpixIDs, pixRa, pixDec)).reset_index()

            print('after matching', time.time()-time_ref,
                  len(matched_pixels['healpixID'].unique()))

            #print('number of pixels',len(matched_pixels['healpixID'].unique()))
            ipix = -1
            isave = -1
            for healpixID in matched_pixels['healpixID'].unique():
                time_ref = time.time()
                ib = matched_pixels['healpixID'] == healpixID
                thematch = matched_pixels.loc[ib]

                if len(thematch) == 0:
                    continue
                ipix += 1

                dataPixel = dataset.iloc[thematch['index'].tolist()].copy()

                pixRa = thematch['pixRa'].unique()
                pixDec = thematch['pixDec'].unique()

                dataPixel.loc[:, 'healpixID'] = healpixID
                dataPixel.loc[:, 'pixRa'] = pixRa[0]
                dataPixel.loc[:, 'pixDec'] = pixDec[0]

                resdict = {}
                time_ref = time.time()

                # run the metrics on those pixels
                for metric in metricList:
                    resdict[metric.name] = metric.run(
                        season(dataPixel.to_records(index=False)))

                # concatenate the results
                for key in resfi.keys():
                    if resdict[key] is not None:
                        if resfi[key] is None:
                            resfi[key] = resdict[key]
                        else:
                            #print('here pal',resdict[key],resfi[key])
                            resfi[key] = np.concatenate(
                                (resfi[key], resdict[key]))

                # from time to time: dump data
                # this is only done for metrics

                if ipix >= 10:
                    if self.saveData:
                        isave += 1
                        for key, vals in resfi.items():
                            if vals is not None:
                                self.dump(vals, nodither, key, ipoint, isave)
                            resfi = {}
                            for metric in metricList:
                                resfi[metric.name] = None
                            ipix = -1

            if ipix != -1:
                if self.saveData:
                    isave += 1
                    for key, vals in resfi.items():
                        if vals is not None:
                            self.dump(vals, nodither, key, ipoint, isave)

    def multi_read(self, groups, names, nproc=4):

        n_names = int(len(names))
        print('hello', n_names)
        delta = n_names
        if nproc > 1:
            delta = int(delta/(nproc))

        tabnames = range(0, n_names, delta)
        if n_names not in tabnames:
            tabnames = np.append(tabnames, n_names)

        tabnames = tabnames.tolist()

        if nproc >= 7:
            if tabnames[-1]-tabnames[-2] <= 10:
                tabnames.remove(tabnames[-2])

        #print(tabnames, len(tabnames))
        result_queue = multiprocessing.Queue()

        for j in range(len(tabnames)-1):

            ida = tabnames[j]
            idb = tabnames[j+1]

            #print('Field', names[ida:idb])
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.concat, args=(groups,
                                                                                             names[ida:idb], j, result_queue))
            p.start()

        resultdict = {}

        for j in range(len(tabnames)-1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        tab_tot = pd.DataFrame()
        for j in range(len(tabnames)-1):
            tab_tot = tab_tot.append(resultdict[j], ignore_index=True)

        return tab_tot

    def concat(self, groups, names, j=-1, output_q=None):

        res = pd.concat([groups.get_group(grname)
                         for grname in names])

        if output_q is not None:
            output_q.put({j: res})
        else:
            return res

    def dump(self, resfi, nodither, key, ipoint, isave):

        outName = '{}/{}_{}_{}.hdf5'.format(self.outDir,
                                            self.dbName, key, self.num)

        df = pd.DataFrame.from_records(resfi)
        tab = Table.from_pandas(df)
        keyhdf = 'metric_{}_{}_{}'.format(self.num, ipoint, isave)
        tab.write(outName, keyhdf, append=True, compression=True)
    """
    def dump_old(self,resfi,nodither,key,ipoint,isave):

        outName = '{}/{}{}_{}_{}.hdf5'.format(self.outDir,
                                              self.dbName, nodither, key, self.num)

        for key, vals in resfi.items():
            if vals is not None:
                    # print(vals)
                    # print(vals.dtype)
                    #df = pd.DataFrame.from_records(vals)

                    keyhdf =  'metric_{}_{}_{}_{}'.format(self.num,ipoint,isave,key)
                    print('rrr',keyhdf)
                    #print('here',df)
                    #with pd.HDFStore(outName) as store:
                    #    store.append('df', df, data_columns= True)
                    #df.to_hdf(outName,key=keyhdf,mode='a',complevel=9)
                    #df.to_hdf(outName,keyhdf,append=True)
                    #store = pd.io.pytables.HDFStore(outName)
                    #store[keyhdf] = df
                    #with pd.get_store(outName) as store:
                    #    store.append('foo',df)
                    # little trick here because of astropy table string format pb
                    # when converted from numpy array
                    df = pd.DataFrame.from_records(vals)
                    tab = Table.from_pandas(df)
                    tab.write(outName,keyhdf,append=True,compression=True)

    def match(self, grp, healpixIDs, pixRa, pixDec,name=None,ax=None):
    =======
                keyhdf = 'metric_{}_{}_{}'.format(self.num, ipoint, isave)
                # print('here',df)
                # with pd.HDFStore(outName) as store:
                #    store.append('df', df, data_columns= True)
                # df.to_hdf(outName,key=keyhdf,mode='a',complevel=9)
                # df.to_hdf(outName,keyhdf,append=True)
                #store = pd.io.pytables.HDFStore(outName)
                #store[keyhdf] = df
                # with pd.get_store(outName) as store:
                #    store.append('foo',df)
                # little trick here because of astropy table string format pb
                # when converted from numpy array
                df = pd.DataFrame.from_records(vals)
                tab = Table.from_pandas(df)
                tab.write(outName, keyhdf, append=True, compression=True)
    """

    def match(self, grp, healpixIDs, pixRa, pixDec, name=None, ax=None):

        # print('hello', grp.columns)
        pixRa_rad = np.deg2rad(pixRa)
        pixDec_rad = np.deg2rad(pixDec)

        # convert data position in rad
        pRa = np.median(grp[self.RaCol])
        pDec = np.median(grp[self.DecCol])
        pRa_rad = np.deg2rad(pRa)
        pDec_rad = np.deg2rad(pDec)

        # gnomonic projection of pixels on the focal plane
        x, y = proj_gnomonic_plane(pRa_rad, pDec_rad, pixRa_rad, pixDec_rad)

        # print(x, y)
        # get LSST FP with the good scale
        fpnew = LSSTPointing(0., 0., maxbound=self.fpscale)

        # print(shapely.vectorized.contains(
        #    fpnew, x, y), self.fpscale, fpnew.area)

        idf = shapely.vectorized.contains(fpnew, x, y)

        """
        if ax is not None:
            ax.plot(x, y, 'ks')
            pf = PolygonPatch(fpnew, facecolor=(0, 0, 0, 0), edgecolor='red')
            ax.add_patch(pf)
        """
        """
        x = np.rad2deg(x)+pRa
        y = np.rad2deg(y)+pDec

        if ax is not None:
            ax.plot(x, y, 'ks')
        # points inside the focal plane
        fp = LSSTPointing(pRa, pDec, area=9.6)
        idf = shapely.vectorized.contains(fp, x, y)
        print(idf)

        print(test)
        if ax is not None:
            pf = PolygonPatch(fp, facecolor=(0, 0, 0, 0), edgecolor='red')
            ax.add_patch(pf)
        """
        """
        if idf:
            grp = group.copy()
            for val in ['healpixID', 'pixRa', 'pixDec']:
                grp.loc[:, val] = np.copy(pixel[val])
        """
        """
        matched_pixels = pd.DataFrame()
        matched_pixels.iloc[:, 'healpixID'] = healpixIDs[idf]
        matched_pixels.iloc[:, 'grname'] = grp.name
        """

        pixID_matched = list(healpixIDs[idf])
        pixRa_matched = list(pixRa[idf])
        pixDec_matched = list(pixDec[idf])

        if ax is not None:
            ax.plot(pixRa, pixDec, 'ko', mfc='None')
            ax.plot(pixRa[idf], pixDec[idf], 'r*')
            fpnew = LSSTPointing(pRa, pDec, area=9.6)
            pf = PolygonPatch(fpnew, facecolor=(0, 0, 0, 0), edgecolor='red')
            ax.add_patch(pf)
            print('matching', grp[self.RaCol, self.DecCol, 'filter'], pixID_matched, len(
                pixID_matched), pixRa_matched, pixDec_matched)

        if name is not None:
            names = [name]*len(pixID_matched)
        else:
            names = [grp.name]*len(pixID_matched)
        # names = ['test']*len(pixID_matched)
        # return pd.Series([matched_pixels], ['healpixIDs'])

        n_index = len(grp.index.values)

        arr_index = grp.index.values

        #arr_index = np.reshape(arr_index,(len(arr_index),1))
        # print('hhh',arr_index)
        df_pix = pd.DataFrame({'healpixID': pixID_matched,
                               'pixRa': pixRa_matched,
                               'pixDec': pixDec_matched,
                               'groupName': names})

        # print(arr_index,df_pix)
        # if n_index > 1:
        #    print('here',n_index,type(grp.index.values))

        n_pix = len(df_pix)
        #n_index = len(df_index)
        # print('indices',n_index,n_pix)
        if n_pix > 1:
            arr_index = arr_index.repeat(n_pix)
        #    if n_index > 1:
        #        print('after repeat',arr_index,arr_index.shape)
        if n_index > 1:
            df_pix = df_pix.append([df_pix]*(n_index-1), ignore_index=True)

        # if n_index > 1:
        #    print(arr_index)
        #    print(n_index,n_pix,len(arr_index),len(df_pix))
        df_pix.loc[:, 'index'] = arr_index

        return df_pix

        grp = pd.concat([grp]*len(pixID_matched), ignore_index=True)
        grp.loc[:, 'healpixID'] = pixID_matched
        grp.loc[:, 'pixRa'] = pixRa_matched
        grp.loc[:, 'pixDec'] = pixDec_matched


class ObsPixel:

    def __init__(self, nside, data, RaCol='Ra', DecCol='Dec'):
        self.nside = nside
        self.data = data
        self.RaCol = RaCol
        self.DecCol = DecCol
        # self.hppix = HEALPix(nside=self.nside, order='nested')

    def matchFast(self, pixel, ax=None):

        time_ref = time.time()
        data = self.pointingsAreaFast(pixel['pixRa'], pixel['pixDec'], 3.)

        if ax is not None:
            val = np.unique(np.unique(data[[self.RaCol, self.DecCol]]))

            ax.plot(val[self.RaCol], val[self.DecCol], 'ko')

        print('pointing', time.time()-time_ref)
        if data is None:
            return None

        dataset = pd.DataFrame(np.copy(data))
        if ax is not None:
            ax.plot(pixel['pixRa'], pixel['pixDec'], 'r*')
            ax.plot(dataset[self.RaCol], dataset[self.DecCol], 'bo')
        # for (pRa,pDec) in np.unique(dataset[[self.RaCol,self.DecCol]]):
        groups = dataset.groupby([self.RaCol, self.DecCol])

        time_ref = time.time()
        print('ngroups', len(groups))
        seldata = None
        for name, group in groups:
            pRa = np.mean(group[self.RaCol])
            pDec = np.mean(group[self.DecCol])

            # convert data position in rad
            pRa_rad = np.deg2rad(pRa)
            pDec_rad = np.deg2rad(pDec)

            # gnomonic projection of pixels on the focal plane
            x, y = proj_gnomonic_plane(pRa_rad, pDec_rad, np.deg2rad(
                pixel['pixRa']), np.deg2rad(pixel['pixDec']))

            x = np.rad2deg(x)+pRa
            y = np.rad2deg(y)+pDec

            if ax is not None:
                ax.plot(x, y, 'ks')
            # points inside the focal planes
            fp = LSSTPointing(pRa, pDec, 0.)
            idf = shapely.vectorized.contains(fp, x, y)
            # print(idf)
            if ax is not None:
                pf = PolygonPatch(fp, facecolor=(
                    0, 0, 0, 0), edgecolor='red')
                ax.add_patch(pf)
            if idf:
                grp = group.copy()
                for val in ['healpixID', 'pixRa', 'pixDec']:
                    grp.loc[:, val] = np.copy(pixel[val])

                if seldata is None:
                    seldata = np.copy(grp.to_records())
                else:
                    seldata = np.concatenate(
                        (seldata, np.copy(grp.to_records())))
        print('datasel', time.time()-time_ref)
        return seldata

    def matchFast_around(self, pixel, ax=None):

        # get pixels around
        healpixID_around = self.hppix.cone_search_lonlat(
            pixel['pixRa'] * u.deg, pixel['pixDec'] * u.deg, radius=3*u.deg)

        # convert to (Ra,Dec)
        coordpix = hp.pix2ang(self.nside, healpixID_around,
                              nest=True, lonlat=True)
        pixelsRa, pixelsDec = coordpix[0], coordpix[1]

        ax.plot(pixelsRa, pixelsDec, 'rs')
        # get the center of the pixels map

        center_Ra = np.mean(pixelsRa)
        center_Dec = np.mean(pixelsDec)

        # get the data around this center
        print('center', center_Ra, center_Dec, pixel['pixRa'], pixel['pixDec'])

        dataset = self.pointingsAreaFast(pixel['pixRa'], pixel['pixDec'], 3.)
        if dataset is None:
            return None

        # for each data set
        # check which pixel is inside the focal plane
        # after gnomonic projection

        for (pRa, pDec) in np.unique(dataset[[self.RaCol, self.DecCol]]):

            # convert data position in rad
            pRa_rad = np.deg2rad(pRa)
            pDec_rad = np.deg2rad(pDec)

            # gnomonic projection of pixels
            x, y = proj_gnomonic_plane(pRa_rad, pDec_rad, np.deg2rad(
                pixelsRa), np.deg2rad(pixelsDec))

            x = np.rad2deg(x)+pRa
            y = np.rad2deg(y)+pDec

            ax.plot(x, y, 'ks')
            # points inside the focal plane
            fp = LSSTPointing(pRa, pDec, 0.)
            idf = shapely.vectorized.contains(fp, x, y)
            print('matching', pixel['healpixID'], healpixID_around[idf])
            if ax is not None:
                pf = PolygonPatch(fp, facecolor=(
                    0, 0, 0, 0), edgecolor='red')
                ax.add_patch(pf)

    def pointingsAreaFast(self, pixRA, pixDec, width):

        # Warning here
        # Ra is in [0,360.]
        # Special care near 0 and 360...

        Ramin = pixRA-width
        Ramax = pixRA+width
        Decmin = pixDec-width
        Decmax = pixDec+width

        areas = []
        # print('there man',Ramin,Ramax,Decmin,Decmax)
        if Ramin < 0:
            areas.append([0., Ramax, Decmin, Decmax])
            areas.append([Ramin+360., 0., Decmin, Decmax])
        else:
            if Ramax > 360.:
                areas.append([0., Ramax-360., Decmin, Decmax])
                areas.append([Ramin, 0., Decmin, Decmax])
            else:
                areas.append([Ramin, Ramax, Decmin, Decmax])

        # print('areas',areas)

        restot = None
        for area in areas:
            res = self.getInside(area, pixRA, pixDec)
            if res is not None:
                if restot is None:
                    restot = res
                else:
                    # print(restot,res)
                    restot = np.concatenate((restot, res))

        # if restot is not None:
        # print('data inside',len(restot))
        return restot

    def getInside(self, area, pixRA, pixDec):

        # print('looking at',pixid,pixRA,pixDec,area)

        Ramin = area[0]
        Ramax = area[1]
        Decmin = area[2]
        Decmax = area[3]

        # print('there man',Ramin,Ramax,Decmin,Decmax)
        # print('mmmm',np.min(self.data[self.RaCol]),
        # np.max(self.data[self.RaCol]),np.min(self.data[self.DecCol]),
        # np.max(self.data[self.DecCol]))

        if Ramax < 1.e-3:
            idx = (self.data[self.RaCol] >= Ramin)

        else:
            # print('booh')
            idx = self.data[self.RaCol] >= Ramin
            idx &= self.data[self.RaCol] <= Ramax

        idx &= self.data[self.DecCol] >= Decmin
        idx &= self.data[self.DecCol] <= Decmax

        res = np.copy(self.data[idx])

        if len(res) == 0:
            return None
        """
        res = rf.append_fields(res, 'healpixID', [pixid]*len(res))
        res = rf.append_fields(res, 'pixRa', [pixRA]*len(res))
        res = rf.append_fields(res, 'pixDec', [pixDec]*len(res))
        """
        return res


class ObsPixel_old:
    def __init__(self, nside, data, scanzone=None, RaCol='Ra', DecCol='Dec'):
        self.nside = nside
        self.data = data
        self.RaCol = RaCol
        self.DecCol = DecCol

        self.hppix = HEALPix(nside=self.nside, order='nested')

        self.scanzone = scanzone

    def matchQuery(self, healpixID):

        step = 1
        lon, lat = self.hppix.boundaries_lonlat(
            healpixID, self.nside, step=step)
        lon = lon.to(u.deg).value
        lat = lat.to(u.deg).value
        coordpix = hp.pix2ang(self.nside, healpixID, nest=True, lonlat=True)
        pixRa, pixDec = coordpix[0], coordpix[1]
        focalplanes = self.pointingsAreaFast(healpixID, pixRa, pixDec, 3.)

        for val in focalplanes:
            lsstpoly = LSSTPointing(val[self.RaCol], val[self.DecCol])
            xp = lsstpoly.exterior.coords.xy[0]
            yp = lsstpoly.exterior.coords.xy[1]
            print(hp.query_polygon(self.nside, [xp, yp, [0.0]*len(xp)]))

        print(test)
        return None

    def matchFast(self, healpixID, ax=None):

        step = 1
        lon, lat = self.hppix.boundaries_lonlat(healpixID, step=step)
        lon = lon.to(u.deg).value
        lat = lat.to(u.deg).value

        coordpix = hp.pix2ang(self.nside, healpixID, nest=True, lonlat=True)
        pixRa, pixDec = coordpix[0], coordpix[1]
        vertices = np.vstack([lon.ravel(), lat.ravel()]).transpose()
        poly = geometry.Polygon(vertices)
        focalplanes = self.pointingsAreaFast(healpixID, pixRa, pixDec, 3.)
        print('pixel area', poly.area)

        # print(self.scanzone.centroid.x,self.scanzone.centroid.y)
        polyscan = affinity.translate(
            self.scanzone, xoff=pixRa-self.scanzone.centroid.x, yoff=pixDec-self.scanzone.centroid.y)
        # check wether this polyscan goes beyond 360. in Ra
        ramax = np.max(polyscan.exterior.coords.xy[0])

        polyscan_b = None
        if ramax >= 360.:
            decmin = np.min(polyscan.exterior.coords.xy[1])
            decmax = np.max(polyscan.exterior.coords.xy[1])
            arrb = [[360., decmin], [380., decmin], [
                380., decmax], [360., decmax], [360., decmin]]
            polyscan_b = geometry.Polygon(arrb)
            polyscan_b = polyscan_b.intersection(polyscan)
            polyscan_b = affinity.translate(polyscan_b, xoff=-360.01)

        finalData = None
        polylist = [polyscan]
        if polyscan_b is not None:
            polylist += [polyscan_b]
        if focalplanes is not None:
            for polyl in polylist:
                idf = shapely.vectorized.contains(
                    polyl, focalplanes[self.RaCol], focalplanes[self.DecCol])
                if len(focalplanes[idf]) > 0:
                    if finalData is None:
                        finalData = np.copy(focalplanes[idf])
                    else:
                        finalData = np.concatenate(
                            (finalData, focalplanes[idf]))

        # print(finalData)
        # This is for display
        if ax is not None:
            po = PolygonPatch(polyscan, facecolor='#fffffe',
                              edgecolor='blue')
            ax.add_patch(po)
            p = PolygonPatch(poly, facecolor=(0, 0, 0, 0), edgecolor='red')
            ax.add_patch(p)
            if polyscan_b is not None:
                p = PolygonPatch(
                    polyscan_b, facecolor='#fffffe', edgecolor='red')
                ax.add_patch(p)
            if focalplanes is not None:
                ax.plot(focalplanes[idf][self.RaCol],
                        focalplanes[idf][self.DecCol], 'gs')
                ax.plot(focalplanes[self.RaCol],
                        focalplanes[self.DecCol], 'r*')
            ax.plot(polyscan.exterior.coords.xy[0],
                    polyscan.exterior.coords.xy[1], 'k.')
            ax.set_xlabel('Ra [deg]')
            ax.set_ylabel('Dec [deg]')

        return finalData

    def pointingsAreaFast(self, pixid, pixRA, pixDec, width):

        # Warning here
        # Ra is in [0,360.]
        # Special care near 0 and 360...

        Ramin = pixRA-width
        Ramax = pixRA+width
        Decmin = pixDec-width
        Decmax = pixDec+width

        areas = []
        # print('there man',Ramin,Ramax,Decmin,Decmax)
        if Ramin < 0:
            areas.append([0., Ramax, Decmin, Decmax])
            areas.append([Ramin+360., 0., Decmin, Decmax])
        else:
            if Ramax > 360.:
                areas.append([0., Ramax-360., Decmin, Decmax])
                areas.append([Ramin, 0., Decmin, Decmax])
            else:
                areas.append([Ramin, Ramax, Decmin, Decmax])

        # print('areas',areas)

        restot = None
        for area in areas:
            res = self.getInside(area, pixid, pixRA, pixDec)
            if res is not None:
                if restot is None:
                    restot = res
                else:
                    # print(restot,res)
                    restot = np.concatenate((restot, res))

        # if restot is not None:
        # print('data inside',len(restot))
        return restot

    def getInside(self, area, pixid, pixRA, pixDec):

        # print('looking at',pixid,pixRA,pixDec,area)

        Ramin = area[0]
        Ramax = area[1]
        Decmin = area[2]
        Decmax = area[3]

        # print('there man',Ramin,Ramax,Decmin,Decmax)
        # print('mmmm',np.min(self.data[self.RaCol]),
        # np.max(self.data[self.RaCol]),np.min(self.data[self.DecCol]),
        # np.max(self.data[self.DecCol]))

        if Ramax < 1.e-3:
            idx = (self.data[self.RaCol] >= Ramin)

        else:
            # print('booh')
            idx = self.data[self.RaCol] >= Ramin
            idx &= self.data[self.RaCol] <= Ramax

        idx &= self.data[self.DecCol] >= Decmin
        idx &= self.data[self.DecCol] <= Decmax

        res = np.copy(self.data[idx])

        if len(res) == 0:
            return None
        res = rf.append_fields(res, 'healpixID', [pixid]*len(res))
        res = rf.append_fields(res, 'pixRa', [pixRA]*len(res))
        res = rf.append_fields(res, 'pixDec', [pixDec]*len(res))

        return res

    def __call__(self, healpixID, ax=None):
        return self.matchFast(healpixID, ax)


class OverlapGnomonic:
    def __init__(self, nside, dRa=0., dDec=0.):

        self.nside = nside
        self.hppix = HEALPix(nside=nside, order='nested', frame='icrs')
        self.dRa = dRa
        self.dDec = dDec

    def overlap_pixlist(self, pixelList, pointing, ax=None):

        pRa = pointing[0]
        pDec = pointing[1]

        pRa_rad = np.deg2rad(pointing[0])
        pDec_rad = np.deg2rad(pointing[1])

        x, y = proj_gnomonic_plane(pRa_rad, pDec_rad, np.deg2rad(
            pixelList['pixRa']), np.deg2rad(pixelList['pixDec']))
        x = np.rad2deg(x)+pRa
        y = np.rad2deg(y)+pDec

        fp = LSSTPointing(pRa, pDec, 0.)
        idf = shapely.vectorized.contains(fp, x, y)
        print(pixelList[idf])

        for pixel in pixelList:
            pixRa, pixDec, poly, polyb = self.polypix(pixel['healpixID'])
            # gnomonic proj
            pixRa_rad = np.deg2rad(pixRa)
            pixDec_rad = np.deg2rad(pixDec)
            x, y = proj_gnomonic_plane(
                pRa_rad, pDec_rad, np.deg2rad(pixRa), np.deg2rad(pixDec))
            # print('pixarea',poly.area,hp.nside2pixarea(self.nside,degrees=True))
            if ax is not None:
                p = PolygonPatch(poly, facecolor='#fffffe', edgecolor='black')
                ax.add_patch(p)
                # p = PolygonPatch(polyb, facecolor='#fffffe', edgecolor='black')
                # ax.add_patch(polyb)
                print(x, y)
                p = Point(x, y)
                ax.plot(np.rad2deg(x)+pRa, np.rad2deg(y)+pDec, 'ks')

        fp = LSSTPointing(pRa, pDec, 0.)
        # print('Pointing area',fp.area)
        if ax is not None:
            pf = PolygonPatch(fp, facecolor=(0, 0, 0, 0), edgecolor='red')
            ax.add_patch(pf)

    def polypix(self, healpixID=10):

        step = 1
        lon, lat = self.hppix.boundaries_lonlat(healpixID, step=step)
        lon = lon.to(u.deg).value
        lat = lat.to(u.deg).value

        coordpix = hp.pix2ang(self.nside, healpixID, nest=True, lonlat=True)
        pixRa, pixDec = coordpix[0], coordpix[1]
        vertices = np.vstack([lon.ravel(), lat.ravel()]).transpose()
        # verticesb = hp.boundaries(self.nside,healpixID,1,True)

        print(vertices, vertices[:, 0])
        minv = np.min(vertices[:, 0])
        maxv = np.max(vertices[:, 0])

        if maxv-minv > 100.:
            for i in range(len(vertices)):
                if vertices[i][0] > 100.:
                    vertices[i][0] -= 360.
        # print(verticesb)
        poly = geometry.Polygon(vertices)
        polyb = Polygon(vertices, closed=True,
                        edgecolor='blue', facecolor='none')
        print(self.nside, self.hppix.pixel_area, healpixID, poly.area)
        return pixRa, pixDec, poly, polyb


class GetOverlap:
    def __init__(self, nside, dRa=0., dDec=0.):

        self.nside = nside
        self.hppix = HEALPix(nside=nside, order='nested', frame='icrs')
        self.dRa = dRa
        self.dDec = dDec

    def polypix(self, healpixID=10):

        step = 1
        lon, lat = self.hppix.boundaries_lonlat(healpixID, step=step)
        lon = lon.to(u.deg).value
        lat = lat.to(u.deg).value

        coordpix = hp.pix2ang(self.nside, healpixID, nest=True, lonlat=True)
        pixRa, pixDec = coordpix[0], coordpix[1]
        vertices = np.vstack([lon.ravel(), lat.ravel()]).transpose()
        # verticesb = hp.boundaries(self.nside,healpixID,1,True)

        print(vertices, vertices[:, 0])
        minv = np.min(vertices[:, 0])
        maxv = np.max(vertices[:, 0])

        if maxv-minv > 100.:
            for i in range(len(vertices)):
                if vertices[i][0] > 100.:
                    vertices[i][0] -= 360.
        # print(verticesb)
        poly = geometry.Polygon(vertices)
        polyb = Polygon(vertices, closed=True,
                        edgecolor='blue', facecolor='none')
        print(self.nside, self.hppix.pixel_area, healpixID, poly.area)
        return pixRa, pixDec, poly, polyb

    def overlap_pixlist(self, pixelList, pointing, ax=None):

        for healpixID in pixelList:
            pixRa, pixDec, poly, polyb = self.polypix(healpixID)
            # print('pixarea',poly.area,hp.nside2pixarea(self.nside,degrees=True))
            if ax is not None:
                p = PolygonPatch(poly, facecolor='#fffffe', edgecolor='black')
                ax.add_patch(p)
                # p = PolygonPatch(polyb, facecolor='#fffffe', edgecolor='black')
                ax.add_patch(polyb)

        fp = LSSTPointing(pointing[0], pointing[1], 0.)
        # print('Pointing area',fp.area)
        if ax is not None:
            pf = PolygonPatch(fp, facecolor=(0, 0, 0, 0), edgecolor='red')
            ax.add_patch(pf)

    def overlap(self, healpixID=100, pointingRa=None, pointingDec=None, ax=None):

        # get initial pixel
        pixRa, pixDec, poly = self.polypix(healpixID)

        if PointingRa is None:
            fpRa = pixRa+self.dRa
            fpDec = pixDec+self.dDec
        else:
            fpRa = pointingRa+self.dRa
            fpDec = pointingDec+self.dDec

        # define a focal plane centered on this pixel
        fp = LSSTPointing(fpRa, fpDec, 0.)

        # get nearby pixels
        healpixID_around = self.hppix.cone_search_lonlat(
            pixRa * u.deg, pixDec * u.deg, radius=3 * u.deg)
        coordpix = hp.pix2ang(self.nside, healpixID_around,
                              nest=True, lonlat=True)
        # coords = SkyCoord(coordpix[0], coordpix[1], unit='deg')
        # print(coordpix[0])
        arr = np.array(healpixID_around, dtype=[('healpixID', 'i8')])
        arr = rf.append_fields(arr, 'pixRa', coordpix[0])
        arr = rf.append_fields(arr, 'pixDec', coordpix[1])

        res = []

        for val in arr:

            pRa, pDec, poly = self.polypix(val['healpixID'])
            pixArea = poly.area
            print(pixArea)
            xpoly = poly.exterior.coords.xy[0]
            ypoly = poly.exterior.coords.xy[1]
            if ax is not None:
                p = PolygonPatch(poly, facecolor='#fffffe', edgecolor='black')
                ax.add_patch(p)
                pf = PolygonPatch(fp, facecolor='#fffffe', edgecolor='red')
                ax.add_patch(pf)
            overlap = poly.intersection(fp).area/pixArea
            res.append((self.nside, val['healpixID'], val['pixRa'], val['pixDec'],
                        overlap, fpRa, fpDec, val['pixRa']-fpRa, val['pixDec']-fpDec, pixArea))

        resrec = np.rec.fromrecords(res, names=[
            'nside', 'healpixID', 'pixRa', 'pixDec', 'overlap', 'fpRa', 'fpDec', 'DRa', 'DDec', 'pixArea'])

        return resrec


class GetShape:
    def __init__(self, nside, overlap):

        self.nside = nside
        self.hppix = HEALPix(nside=nside, order='nested')
        self.overlap = overlap

    def shape(self, healpixID=10, ax=None):

        step = 1
        lon, lat = self.hppix.boundaries_lonlat(healpixID, step=step)
        lon = lon.to(u.deg).value
        lat = lat.to(u.deg).value
        coordpix = hp.pix2ang(self.nside, healpixID, nest=True, lonlat=True)
        pixRa, pixDec = coordpix[0], coordpix[1]
        vertices = np.vstack([lon.ravel(), lat.ravel()]).transpose()
        # print(vertices)
        poly = geometry.Polygon(vertices)
        scanzone = self.followShape(poly, ax)

        # This is for display
        if ax is not None:
            p = PolygonPatch(poly, facecolor='#fffffe', edgecolor='red')
            ax.add_patch(p)
            width = 3.
            ax.set_xlim([pixRa-width, pixRa+width])
            ax.set_ylim([pixDec-width, pixDec+width])
            ax.set_xlabel('Ra [deg]')
            ax.set_ylabel('Dec [deg]')

        return scanzone

    def followShape(self, poly, ax=None):

        xpoly = poly.exterior.coords.xy[0]
        ypoly = poly.exterior.coords.xy[1]
        pixRa = np.mean(xpoly)
        pixDec = np.mean(ypoly)
        pixArea = poly.area

        # print('allo',pixArea)
        dRa = 5.
        dDec = 5.

        RaVals = np.arange(pixRa-dRa, pixRa+dRa, 0.1)
        DecVals = np.arange(pixDec-dDec, pixDec+dDec, 0.1)
        r = []
        for Ra in RaVals:
            for Dec in DecVals:
                fp = LSSTPointing(Ra, Dec, 0.)
                area = poly.intersection(fp).area/fp.area
                area = poly.intersection(fp).area/pixArea
                # print('alors',Ra,Dec,area)
                if area >= self.overlap:
                    r.append((Ra, Dec, area))

        if len(r) > 0.:
            shape = np.rec.fromrecords(r, names=['x', 'y', 'area'])

            idx = shape['area'] >= 0.
            sel = shape[idx]

            r = []
            for x in np.unique(sel['x']):
                id = np.abs(sel['x']-x) < 1.e-5
                r.append((x, np.min(sel[id]['y'])))
                r.append((x, np.max(sel[id]['y'])))

            sel = np.rec.fromrecords(r, names=['x', 'y'])

            ida = sel['y'] > pixDec
            sela = sel[ida]
            sela.sort(order='x')

            idb = sel['y'] <= pixDec
            selb = sel[idb]
            selb[::-1].sort(order='x')

            res = np.concatenate((sela, selb))
            polyshape = geometry.Polygon([[val['x'], val['y']] for val in res])

        if ax is not None:

            p = PolygonPatch(poly, facecolor='#fffffe', edgecolor='red')
            ax.add_patch(p)
            width = 3.
            pixRa = np.mean(xpoly)
            pixDec = np.mean(ypoly)

            p = PolygonPatch(polyshape, edgecolor='blue')
            ax.add_patch(p)
            ax.set_xlim([pixRa-width, pixRa+width])
            ax.set_ylim([pixDec-width, pixDec+width])
            ax.set_xlabel('Ra [deg]')
            ax.set_ylabel('Dec [deg]')

        return polyshape


def getFields_fromId(observations, fieldIds):

    obs = None
    for fieldId in fieldIds:
        idf = observations['fieldId'] == fieldId
        if obs is None:
            obs = observations[idf]
        else:
            obs = np.concatenate((obs, observations[idf]))
    return obs


def getFields(observations, fieldType='WFD', fieldIds=None, nside=64):

    print(observations.dtype)

    obs = None

    # this is for the WFD

    for pName in ['proposalId', 'survey_id']:
        if pName in observations.dtype.names:

            print(np.unique(observations[pName]))
            propId = list(np.unique(observations[pName]))

            # loop on proposal id
            # and take the one with the highest number of evts
            propIds = list(np.unique(observations[pName]))
            r = []
            for propId in propIds:
                idx = observations[pName] == propId
                r.append((propId, len(observations[idx])))

            res = np.rec.fromrecords(r, names=['propId', 'Nobs'])
            if fieldType == 'WFD':
                # Take the propId with the largest number of fields
                propId_WFD = propIds[np.argmax(res['Nobs'])]
                print(res, np.argmax(res['Nobs']), propId_WFD)
                return observations[observations[pName] == propId_WFD]
            if fieldType == 'DD':
                # could be tricky here depending on the database structure
                if 'fieldId' in observations.dtype.names:
                    # print('hello', np.unique(
                    #    observations['fieldId']), len(propIds))
                    fieldIds = np.unique(observations['fieldId'])
                    # easy one: grab DDF from fieldIds
                    # if len(propIds) >= 3:
                    if len(fieldIds) >= 3:
                        obser = getFields_fromId(observations, fieldIds)
                    else:
                        obser = getFields_fromId(observations, [0])
                    return pixelate(obser, nside, RaCol='fieldRA', DecCol='fieldDec')

                else:
                    """
                    Tricky
                    we do not have other ways to identify
                    DD except by selecting pixels with a large number of visits
                    """
                    pixels = pixelate(observations, nside,
                                      RaCol='fieldRA', DecCol='fieldDec')

                    df = pd.DataFrame(np.copy(pixels))

                    groups = df.groupby('healpixID').filter(
                        lambda x: len(x) > 5000)

                    group_DD = groups.groupby(['fieldRA', 'fieldDec']).filter(
                        lambda x: len(x) > 4000)

                    # return np.array(group_DD.to_records().view(type=np.matrix))
                    return group_DD.to_records(index=False)
