import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rf
from shapely import geometry
from shapely import affinity
import shapely.vectorized
from astropy_healpix import HEALPix
from astropy import units as u
from descartes.patch import PolygonPatch
#from astropy.coordinates import SkyCoord
#from dustmaps.sfd import SFDQuery
from matplotlib.patches import Polygon
from shapely.geometry import Point
import pandas as pd
import time
import h5py
from astropy.table import Table
import multiprocessing
import glob
import os
from sn_tools.sn_clusters import ClusterObs
import copy


def DDFields(DDfile=None):
    """
    Function to define DD fields
    The definitions are hardcoded for the moment
    Should move to an input file

    Parameters
    ----------------
    DDfile: str, opt
      csv file with DD infos: name fieldId      RA    Dec fieldnum
    name: name of the field (eg COSMOS, ELAIS, ...)
    fieldID: int
      internal OpSim value
    RA: field RA
    Dec: field Dec
    fieldnum: int, field number (set by the user)

    Returns
    ---------

    fields: pandas DataFrame
      df with the following columns:
     - name: name of the field
     - fieldId: Id of the field
     - RA: RA of the field
     - Dec: Dec of the field
     - fieldnum: field number


    """

    if DDfile is not None:
        fields = pd.read_csv(DDfile)
        return fields
    else:
        fields = pd.DataFrame(
            columns=['name', 'fieldId', 'RA', 'Dec', 'fieldnum'])

        fields.loc[0] = ['ELAIS', 744, 10.0, -45.52, 4]
        fields.loc[1] = ['SPT', 290, 349.39, -63.32, 5]
        fields.loc[2] = ['COSMOS', 2786, 150.36, 2.84, 1]
        fields.loc[3] = ['XMM-LSS', 2412, 34.39, -5.09, 2]
        fields.loc[4] = ['CDFS', 1427, 53.00, -27.44, 3]
        fields.loc[5] = ['ADFS1', 290, 63.59, -47.59, 6]
        fields.loc[6] = ['ADFS2', 290, 58.97, -49.28, 7]

        return fields


def patchObs(observations, fieldType,
             dbName, nside, RAmin, RAmax, Decmin, Decmax,
             RACol, DecCol,
             display=False, nclusters=5, radius=4.):
    """
    Function to grab informations and patches in the sky

    Parameters
    --------------
    observations: numpy array
      array of observations
    fieldType: str
      type of field to consider: DD, WFD or Fake
    dbName: str
      name of observing strategy
    nside: int
       nside parameter for healpix
    RAmin: float
      min RA of the sky area to consider (WFD only)
    RAmax: float
      max RA of the sky area to consider (WFD only)
   Decmin: float
      min Dec of the sky area to consider (WFD only)
    Decmax: float
      max Dec of the sky area to consider (WFD only)
    RACol: str
      RA column name in obs
    DecCol: str
      Dec column name in obs
    display: bool
      to plot patches (WFD only)

    Returns
    ----------
    observations: numpy array
      numpy array with observations
    patches: pandas df
      patches coordinates on the sky

    """

    # radius = 5.

    if fieldType == 'DD':

        # print(np.unique(observations['fieldId']))
        fieldIds = [290, 744, 1427, 2412, 2786]
        observations = getFields(
            observations, fieldType, fieldIds, nside)

        # print('before cluster', len(observations),observations.dtype, nclusters)
        # get clusters out of these obs
        # radius = 4.

        DD = DDFields()
        clusters = ClusterObs(
            observations, nclusters=nclusters, dbName=dbName, fields=DD).clusters

        # clusters = rf.append_fields(clusters, 'radius', [radius]*len(clusters))
        clusters['radius'] = radius
        # areas = rf.rename_fields(clusters, {'RA': 'RA'})
        areas = clusters.rename(columns={'RA': 'RA'})
        patches = pd.DataFrame(areas)
        patches['width_RA'] = radius
        patches['width_Dec'] = radius
        patches = patches.rename(
            columns={"width_RA": "radius_RA", "width_Dec": "radius_Dec"})

    else:
        if fieldType == 'WFD':
            observations = getFields(observations, 'WFD')
            minDec = Decmin
            maxDec = Decmax
            if minDec == -1.0:  # in that case min and max dec are given by obs strategy
                minDec = np.min(observations['fieldDec'])-radius
                minDec = max(minDec, -90.)
            if maxDec == -1.0:
                maxDec = np.max(observations['fieldDec'])+radius
            areas = PavingSky(RAmin, RAmax, minDec, maxDec, radius, radius)
            # print(observations.dtype)
            if display:
                areas.plot()

        if fieldType == 'Fake':
            # in that case: only one (RA,Dec)
            # radius = 0.1
            RA = np.unique(observations[RACol])[0]
            Dec = np.unique(observations[DecCol])[0]
            areas = PavingSky(RA-radius/2., RA+radius/2., Dec -
                              radius/2., Dec+radius/2., radius, radius)

        patches = pd.DataFrame(areas.patches)

    return observations, patches


def getPix(nside, fieldRA, fieldDec):
    """
    Function returning pixRA, pixDec and healpixId

    Parameters
    ---------------
    nside: int
      nside value for healpix
    fieldRA: float
      field RA
    fieldDec: float
      field Dec

    Returns
    ----------
    healpixId: Id from healpix
    pixRA: RA from healpix
    pixDec: Dec from healpix

    """

    healpixId = hp.ang2pix(nside, fieldRA,
                           fieldDec, nest=True, lonlat=True)
    coord = hp.pix2ang(nside,
                       healpixId, nest=True, lonlat=True)

    pixRA = coord[0]
    pixDec = coord[1]

    return healpixId, pixRA, pixDec


class PavingSky:
    def __init__(self, minRA, maxRA, minDec, maxDec, radius_RA, radius_Dec):
        """ class to perform a paving of the sky with rectangles

        Parameters
        --------------
        minRA: float
         min ra of the region to pave
        maxRA: float
         max ra of the region to pave
        minDec: float
         min dec of the region to pave
        maxDec: float
         max dec of the region to pave
         radius_RA: float
          distance reference for the paving in RA; correspond to the length in RA.
         radius_Dec: float
          distance reference for the paving in Dec; correspond to the length in Dec.
        """

        self.RA = np.mean([minRA, maxRA])
        self.Dec = np.mean([minDec, maxDec])
        self.radius_RA = radius_RA
        self.radius_Dec = radius_Dec

        # define the polygon attached to this area
        """
        self.area_poly = areap(self.RA-radius_RA/2.,
                               self.RA+radius_RA/2.,
                               self.Dec-radius_Dec/2.,
                               self.Dec+radius_Dec/2.)
        """
        self.area_poly = areap(minRA, maxRA, minDec, maxDec)
        all_patches = self.getpatches(minRA, maxRA, minDec, maxDec)

        self.patches = self.inside(all_patches)

    def getpatches(self, minRA, maxRA, minDec, maxDec):
        """
        Method to define rectangles patches over the area defined by (minRA,minDec, maxRA, maxDec)

        Parameters
        --------------
        minRA: float
            min RA value of the area
        maxRA: float
            max RA value of the area
        minDec: float
            min Dec value of the area
        maxDec: float
            max Dec value of the area

        Returns
        ----------
        numpy array with the following cols:
        RA, Dec: coordinate of the patch center
        radius_RA, _Dec: diag distance in RA and Dec

        """
        n_ra = int((maxRA-minRA)/self.radius_RA)
        radius_RA = (maxRA-minRA)/n_ra

        n_dec = int((maxDec-minDec)/self.radius_Dec)
        radius_Dec = (maxDec-minDec)/n_dec
        rastep = self.radius_RA
        decstep = self.radius_Dec/2.
        decrefs = np.arange(minDec, maxDec, decstep)
        shift = self.radius_RA/2.

        ras = np.arange(minRA, maxRA+radius_RA, radius_RA)
        decs = np.arange(minDec, maxDec+radius_Dec, radius_Dec)

        r = []
        for ra in ras:
            ramax = ra+radius_RA
            ramean = np.mean([ra, ramax])
            for dec in decs:
                decmax = dec+radius_Dec
                decmean = np.mean([dec, decmax])
                r.append((ramean, decmean, radius_RA,
                          radius_Dec, ra, ramax, dec, decmax))

        return np.rec.fromrecords(r, names=['RA', 'Dec', 'radius_RA', 'radius_Dec', 'minRA', 'maxRA', 'minDec', 'maxDec'])

    def inside(self, areas):
        """
        Method to select patches located (center) inside the area

        Parameters
        --------------
        areas: numpy array with
            RA, Dec: coordinate of the patch center
            radius_RA, _Dec: diag distance in RA and Dec

        Returns
        ----------
        numpy array of patches (same struct as input) matching the area

        """

        poly_orig = geometry.Polygon(self.area_poly)
        poly_origb = affinity.scale(poly_orig, xfact=1.1, yfact=1.1)
        idf = shapely.vectorized.contains(
            poly_origb, areas['RA'], areas['Dec'])

        return areas[idf]

    def plot(self):
        """
        Method to plot/check the result of the class

        On the plot should be visible the initial area (red edges) with diamonds (blue edges)
        whose centers (black point) are located inside the initial area

        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        pf = PolygonPatch(self.area_poly, facecolor=(
            0, 0, 0, 0), edgecolor='red')
        ax.add_patch(pf)

        for val in self.patches:

            # polyb = areap_diamond(val['RA'],val['Dec'],val['radius_RA'],val['radius_Dec'])
            minRA = val['RA']-val['radius_RA']/2.
            maxRA = val['RA']+val['radius_RA']/2.
            minDec = val['Dec']-val['radius_Dec']/2.
            maxDec = val['Dec']+val['radius_Dec']/2.
            polyb = areap(minRA, maxRA, minDec, maxDec)
            pfb = PolygonPatch(polyb, facecolor=(
                0, 0, 0, 0), edgecolor='blue')
            ax.add_patch(pfb)
            ax.plot(val['RA'], val['Dec'], 'k.')

        plt.show()


def pavingSky_old(ramin, ramax, decmin, decmax, radius):
    """ Function to perform a paving of the sky with hexagons

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
    tab: record array of the center position (RA,Dec) and the radius
    """
    ramin = ramin+radius*np.sqrt(3.)/2.
    decstep = 1.5*radius
    rastep = radius*np.sqrt(3.)
    shift = radius*np.sqrt(3.)/2.
    decrefs = np.arange(decmax, decmin, -decstep)
    r = []
    # this is to pave the sky loopin on RA, same Dec
    for i in range(len(decrefs)):
        shift_i = shift*(i % 2)
        for ra in list(np.arange(ramin-shift_i, ramax, rastep)):
            r.append((ra, decrefs[i], radius))

    res = np.rec.fromrecords(r, names=['RA', 'Dec', 'radius'])
    return res


def area(minRA, maxRA, minDec, maxDec):
    """
    Function to make a dict of coordinates

    Parameters
    --------------
    minRA: float
      min RA
    maxRA: float
      maxRA
    minDec: float
     min Dec
    maxDec: float
      max Dec

    Returns
    ----------
    dict with keys corresponding to abovemantioned values

    """

    return dict(zip(['minRA', 'maxRA', 'minDec', 'maxDec'], [minRA, maxRA, minDec, maxDec]))


def areap(minRA, maxRA, minDec, maxDec):
    """
    Function to make a polygon out of coordinates

    Parameters
    --------------
    minRA: float
      min RA
    maxRA: float
      maxRA
    minDec: float
     min Dec
    maxDec: float
      max Dec

    Returns
    ----------
    geometry.Polygon

    """
    poly = [[minRA, minDec], [minRA, maxDec], [maxRA, maxDec], [maxRA, minDec]]

    return geometry.Polygon(poly)


def areap_diamond(RA, Dec, radius_RA, radius_Dec):
    """
    Function defining a diamond

    Parameters
    ---------------
    RA: float
      RA of the center of the diamond
    Dec: float
      Dec of the center of the diamond
    radius_RA: float
      diag value in RA
    radius_Dec: float
      diag value in Dec

    Returns
    ----------
    geometry.Polygon of the diamond

    """

    minRA = RA-radius_RA/2.
    maxRA = RA+radius_RA/2.
    minDec = Dec-radius_Dec/2.
    maxDec = Dec+radius_Dec/2.

    poly = [[minRA, Dec], [RA, maxDec], [maxRA, Dec], [RA, minDec]]

    return geometry.Polygon(poly)


class DataInside:
    def __init__(self, data, RA, Dec, widthRA, widthDec, RACol='fieldRA', DecCol='fieldDec'):
        """
        class to select data points (RA,Dec) inside and area
        defined by the center (RA,Dec) and two width (widthRA, widthDec)

        Parameters
        --------------
        data: array of data
         should contains cols named RACol and DecCol
        RA: float
         RA center of the region to consider
        Dec: float
         Dec center of the region to consider
        widthRA: float
         RA width of the region to consider
        widthDec: float
         Dec width of the region to consider
       RACol: str, opt
        name of the col with RA infos (default:'fieldRA')
       DecCol: str, opt
        name of the col with Dec infos (default:'fieldDec')

        """
        self.RACol = RACol
        self.DecCol = DecCol

        minRA = RA-widthRA
        maxRA = RA+widthRA

        minDec = Dec-widthDec
        maxDec = Dec+widthDec

        self.areas = self.getAreas(RA, Dec, minRA, maxRA, minDec, maxDec)

        self.data = self.selData(data, self.areas)

    def getAreas(self, RA, Dec, minRA, maxRA, minDec, maxDec):
        """
        Method to get areas (geometry.Polygon) corresponding to a region
        centered in (RA,Dec) with min, max RA and Dec

        Parameters
        --------------
         RA: float
         RA center of the region to consider
        Dec: float
         Dec center of the region to consider
         minRA: float
          RA min of the region to consider
        maxRA: float
          RA max of the region to consider
        maxDec: float
         Dec max of the region to consider

        Returns
        ---------
        list of dict with the following keys:
          minRA, maxRA, minDec, maxDec

        """
        # Create area from these
        # RA is in [0.,360.]
        # special treatement near RA~0
        ax = None

        areaList = []
        if ax is not None:
            areapList = []

        if maxRA >= 360.:
            # in that case two areas necessary
            areaList.append(area(minRA, 360., minDec, maxDec))
            areaList.append(area(0.0, maxRA-360., minDec, maxDec))
        else:
            if minRA < 0.:
                # in that case two areas necessary
                areaList.append(area(minRA+360., 360., minDec, maxDec))
                areaList.append(area(-1.e-8, maxRA, minDec, maxDec))
            else:
                areaList.append(area(minRA, maxRA, minDec, maxDec))

        return areaList

    def selData(self, data, areaList):
        """
        Method selecting data located (in (RA,Dec)) inside a list of areas

        Parameters
        --------------
        data: array
          array of data
        areaList: list of areas
           list of areas to consider

        Returns
        ----------
        Data inside the regions defined in areaList

        """

        dataSel = None
        for areal in areaList:
            data_inside = self.inData(data, areal)
            if data_inside is not None:
                if dataSel is None:
                    dataSel = data_inside
                else:
                    dataSel = np.concatenate((dataSel, data_inside))

        return dataSel

    def inData(self, data, area):
        """
        Method selecting data located (in (RA,Dec)) inside a list of areas

        Parameters
        --------------
        data: array
          array of data
        area: area
         dict with the following keys:
           minRA, maxRA, minDec, maxDec

        Returns
        ----------
        Data inside the region defined by area

        """
        diff_add = 0.

        idf = data[self.RACol]-area['minRA'] >= -diff_add
        idf &= data[self.RACol]-area['maxRA'] <= diff_add
        idf &= data[self.DecCol]-area['minDec'] >= -diff_add
        idf &= data[self.DecCol]-area['maxDec'] <= diff_add

        return data[idf]

    def inData_old(self, data, area):
        """
        Method selecting data located (in (RA,Dec)) inside a list of areas

        Parameters
        --------------
        data: array
          array of data
        area: area
         dict with the following keys:
           minRA, maxRA, minDec, maxDec

        Returns
        ----------
        Data inside the region defined by area

        """
        idf = (data[self.RACol] >= area['minRA']) & (
            data[self.RACol] <= area['maxRA'])
        idf &= (data[self.DecCol] >= area['minDec']) & (
            data[self.DecCol] <= area['maxDec'])

        if len(data[idf]) > 0.:
            return data[idf]
        return None

    def plot(self, ax):
        """
        Method to plot the result of the class:

        - area selected (red edges)
        - data selected (black points)

        Parameters
        --------------
        ax: axes.Axes (matplotlib)
          axis to plot the results

        """

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()

        ax.plot(self.data[self.RACol], self.data[self.DecCol], 'ko')

        for val in self.areas:
            polyb = areap(val['minRA'], val['maxRA'],
                          val['minDec'], val['maxDec'])
            pfb = PolygonPatch(polyb, facecolor=(0, 0, 0, 0), edgecolor='red')
            ax.add_patch(pfb)

        # plt.show()


def proj_gnomonic_plane(lamb0, phi1, lamb, phi):
    """
    Function to perform a gnomonic projection
    on a plane of points of the celestial sphere

    The formulas coded here are taken from:
    Map Projections - A working manual
    US Geological Survey - Professional paper 1395
    by John P.Snyder - 1987

    Parameters
    --------------
    lambd0:  float
      longitude of the tangent point of the projection (RA)
    phi1: float
      latitude of the tangent point of the projection (Dec)
    lambd:  float
      longitude of the point to project (RA)
    phi: float
      latitude of the point to project (Dec)

    Returns
    ----------
    x,y: coordinates of the projected point

    """

    cosc = np.sin(phi1)*np.sin(phi)
    cosc += np.cos(phi1)*np.cos(phi)*np.cos(lamb-lamb0)

    x = np.cos(phi)*np.sin(lamb-lamb0)
    x /= cosc

    y = np.cos(phi1)*np.sin(phi)
    y -= np.sin(phi1)*np.cos(phi)*np.cos(lamb-lamb0)

    y /= cosc

    return x, y


def proj_gnomonic_sphere(lamb0, phi, x, y):
    """
    Function to perform a gnomonic projection
    on a sphere of points of a plane.

    The formulas coded here are taken from:
    Map Projections - A working manual
    US Geological Survey - Professional paper 1395
    by John P.Snyder - 1987

    Parameters
    --------------
    lambd0:  float
      longitude  of the tangent point of the projection (RA)
    phi1: float
      latitude of the tangent point of the projection (Dec)
    x:  float
       x position of the point
    y: float
      y position of the point

    Returns
    ----------
    lamb, phi1: coordinates of the projected point on the sphere
     lamb: latitude of the point (Dec)
     phi1:  longitude of the point(RA)

    """
    rho = (x**2+y**2)**0.5
    c = np.arctan(rho)
    # print('c', rho, c, np.rad2deg(c))
    lamb = x*np.sin(c)
    lamb /= (rho*np.cos(phi)*np.cos(c)-y*np.sin(phi)*np.sin(c))
    lamb = lamb0+np.arctan(lamb)

    phi1 = np.cos(c)*np.sin(phi)
    phi1 += (y*np.sin(c)*np.cos(phi))/rho
    phi1 = np.arcsin(phi1)

    return lamb, phi1


def renameFields(tab):
    """
    Function to rename fields

    Parameters
    --------------
    tab: array
      original array of data

    Returns
    ---------
    array of data with modified field names

    """
    corresp = {}

    fillCorresp(tab, corresp, 'mjd', 'observationStartMJD')
    fillCorresp(tab, corresp, 'RA', 'fieldRA')
    fillCorresp(tab, corresp, 'Ra', 'fieldRA')
    fillCorresp(tab, corresp, 'Dec', 'fieldDec')
    fillCorresp(tab, corresp, 'band', 'filter')
    fillCorresp(tab, corresp, 'exptime', 'visitExposureTime')
    fillCorresp(tab, corresp, 'nexp', 'numExposures')

    # print(tab.dtype)

    rb = np.copy(tab)
    for vv, vals in corresp.items():
        rb = rf.drop_fields(rb, vv)
        if vv != 'band':
            rb = rf.append_fields(rb, vals, tab[vv])
        else:
            rb = rf.append_fields(rb, vals, tab[vv], dtypes='<U9')
        # rb = rf.rename_fields(rb, {vv: vals})

    # return rf.rename_fields(tab, corresp)
    return rb


def fillCorresp(tab, corres, vara, varb):
    """
    Function to fill a dict used to change colnams of a nupy array

    Parameters
    --------------
    tab: array
      original array of data
    corresp: dict
      keys: string, items: string
    vara: str
     colname in tab to change
    varb: str
     new colname to replace vara

    Returns
    ---------
    dict with str as keys and items
      correspondence vara<-> varb

    """

    if vara in tab.dtype.names and varb not in tab.dtype.names:
        corres[vara] = varb


def pixelate(data, nside, RACol='RA', DecCol='Dec'):
    """
    Function to pixelate the sky

    Parameters
    --------------
    data: array
      data to use for pixelation
    nside: int
      nside parameter for healpix
    RACol: str, opt
      name of the RA col in data to use (default: RA)
     DecCol: str, opt
      name of the Dec col in data to use (default: Dec)

    Returns
    ----------
    initial array with the following infos added:
     healpixID,pixRA,pixDec: pixel ID, RA and Dec (from healpix)
    ebv: E(B-V)

    """

    res = data.copy()
    npix = hp.nside2npix(nside)
    table = hp.ang2vec(res[RACol], res[DecCol], lonlat=True)

    healpixs = hp.vec2pix(
        nside, table[:, 0], table[:, 1], table[:, 2], nest=True)
    coord = hp.pix2ang(nside, healpixs, nest=True, lonlat=True)

    # This is to get ebvofMW value
    """
    coords = SkyCoord(coord[0], coord[1], unit='deg')
    sfd = SFDQuery()
    ebv = sfd(coords)
    """
    ebv = -1.
    res = rf.append_fields(res, 'healpixID', healpixs)
    res = rf.append_fields(res, 'pixRA', coord[0])
    res = rf.append_fields(res, 'pixDec', coord[1])
    res = rf.append_fields(res, 'ebv', [ebv]*len(healpixs))

    return res


def season(obs, season_gap=80., mjdCol='observationStartMJD'):
    """
    Function to estimate seasons

    Parameters
    --------------
    obs: numpy array
      array of observations
    season_gap: float, opt
       minimal gap required to define a season (default: 80 days)
    mjdCol: str, opt
      col name for MJD infos (default: observationStartMJD)

    Returns
    ----------
    original numpy array with season appended

    """

    # check wether season has already been estimated
    if 'season' in obs.dtype.names:
        return obs

    obs.sort(order=mjdCol)

    """
    if len(obs) == 1:
        obs = np.atleast_1d(obs)
        obs = rf.append_fields([obs], 'season', [1.])
        return obs
    diff = obs[mjdCol][1:]-obs[mjdCol][:-1]

    flag = np.argwhere(diff > season_gap)
    if len(flag) > 0:
        seas = np.zeros((len(obs),), dtype=int)
        flag += 1
        seas[0:flag[0][0]] = 1
        for iflag in range(len(flag)-1):
            seas[flag[iflag][0]:flag[iflag+1][0]] = iflag+2
        seas[flag[-1][0]:] = len(flag)+1
        obs = rf.append_fields(obs, 'season', seas)
    else:
        obs = rf.append_fields(obs, 'season', [1]*len(obs))
    """
    seasoncalc = np.ones(obs.size, dtype=int)

    if len(obs) > 1:
        diff = np.diff(obs[mjdCol])
        flag = np.where(diff > season_gap)[0]

        if len(flag) > 0:
            for i, indx in enumerate(flag):
                seasoncalc[indx+1:] = i+2

    obs = rf.append_fields(obs, 'season', seasoncalc)
    return obs


def LSSTPointing(xc, yc, angle_rot=0., area=None, maxbound=None):
    """
    Function to build a focal plane for LSST

    Parameters
    ---------------

    xc: float
       x-position of the center FP (RA)
    yc: float
       y-position of the center FP (Dec)
    angle_rot: float, opt
      angle of rotation of the FP (default: 0.)
    area: float
      area for the FP (default: None)
    maxbound: float
      to reduce area  (default: None)
    Returns
    ----------
    LSST FP (geometry.Polygon)

    """

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

    # symmetry II: x -> -x
    arr = list(arrcp)
    for val in arrcp[::-1]:
        if val[0] > 0.:
            arr.append([-val[0], val[1]])

    # build polygon
    poly_orig = geometry.Polygon(arr)

    # set area
    if area is not None:
        poly_orig = affinity.scale(poly_orig, xfact=np.sqrt(
            area/poly_orig.area), yfact=np.sqrt(area/poly_orig.area))

    # set rotation angle
    rotated_poly = affinity.rotate(poly_orig, angle_rot)

    return affinity.translate(rotated_poly,
                              xoff=xc-rotated_poly.centroid.x,
                              yoff=yc-rotated_poly.centroid.y)


def LSSTPointing_circular(xc, yc, angle_rot=0., area=None, maxbound=None):
    """
    Function to build a focal plane for LSST

    Parameters
    ---------------

    xc: float
       x-position of the center FP (RA)
    yc: float
       y-position of the center FP (Dec)
    angle_rot: float, opt
      angle of rotation of the FP (default: 0.)
    area: float
      area for the FP (default: None)
    maxbound: float
      to reduce area (default: None)
    Returns
    ----------
    LSST FP (geometry.Polygon)

    """

    #
    arr = []

    radius = 1.
    if maxbound is not None:
        radius = maxbound

    step = radius/100.
    for x in np.arange(0., radius, step):
        y = np.sqrt(radius*radius-x*x)
        arr.append([x, y])

    # symmetry I: y -> -y
    arrcp = list(arr)
    for val in arr[::-1]:
        if val[1] >= 0.:
            arrcp.append([val[0], -val[1]])

    # symmetry II: x -> -x
    arr = list(arrcp)
    for val in arrcp[::-1]:
        if val[0] > 0.:
            arr.append([-val[0], val[1]])

    # build polygon
    poly_orig = geometry.Polygon(arr)

    # set area
    if area is not None:
        poly_orig = affinity.scale(poly_orig, xfact=np.sqrt(
            area/poly_orig.area), yfact=np.sqrt(area/poly_orig.area))

    # set rotation angle
    if angle_rot > 0.1:
        rotated_poly = affinity.rotate(poly_orig, angle_rot)
    else:
        rotated_poly = poly_orig

    return affinity.translate(rotated_poly,
                              xoff=xc-rotated_poly.centroid.x,
                              yoff=yc-rotated_poly.centroid.y)


class DataToPixels:
    """
    class to match observations to sky pixels

    Parameters
    ---------------
    nside: int
      nside parameter for healpix tessalation
    RACol: str
     name of the RA field
    DecCol: str
      name of the Dec field      
    num: int
      index (related to multiprocessing)
    outDir: str
      output dir path
    dbName: str
      observing strategy name
    saveData: bool, opt
       to save (True) the data or not (False) (default: False)
    """

    def __init__(self, nside, RACol, DecCol, outDir, dbName):

        # load parameters
        self.nside = nside
        self.RACol = RACol
        self.DecCol = DecCol
        #self.obsIdCol = obsIdCol
        #self.num = num
        #self.outDir = outDir

        self.dbName = dbName

        # get the LSST focal plane scale factor
        # corresponding to a sphere radius equal to one
        # (which is the default for gnomonic projections here)

        fov = 9.62*(np.pi/180.)**2  # LSST fov in sr
        theta = 2.*np.arcsin(np.sqrt(fov/(4.*np.pi)))

        # if theta >= np.pi/2.:
        #    theta -= np.pi/2.
        # print('theta', theta, np.rad2deg(theta))
        self.fpscale = np.tan(theta)

    def __call__(self, data, RA, Dec, widthRA, widthDec, nodither=False, display=False):
        """
        call method: this is where the processing is.

        Parameters
        --------------
        data: numpy array
          data to process
        RA: float
           RA position (center of the area to process)
        Dec: float
           Dec position (center of the area to process)
        widthRA: float
          width in RA of the area to process
        widthDec: float
          width in Dec of the area to process
        nodither: bool,opt
          to remove dithering (default: False)
        display: bool, opt
          to display matching FP/observations in "real time" (default: False)

        Returns:
        -----------
        matched_pixels: pandas df
          array with matched healpix infos and obs infos

        """

        # display: (RA,Dec) distribution of the data

        if display:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(data[self.RACol], data[self.DecCol], 'ko')
            plt.show()

        # select data inside an area centered in (RA,Dec) with width (widthRA+1,widthDec+1)

        # print('searching data inside',RA,Dec,widthRA,widthDec)
        dataSel = DataInside(data, RA, Dec, widthRA+1., widthDec+1.,
                             RACol=self.RACol, DecCol=self.DecCol)

        # display: (RA,Dec) distribution of the selected data (ie inside the area)
        # if display:
        #    dataSel.plot()

        self.observations = None

        # print('there man',len(dataSel.data))
        if len(dataSel.data) == 0:
            return None

        # mv to panda df
        dataset = pd.DataFrame(np.copy(dataSel.data))

        # Possible to remove DD dithering here
        # This is usually to test impact of dithering on DDF
        if nodither:
            dataset[self.RACol] = np.mean(dataset[self.RACol])
            dataset[self.DecCol] = np.mean(dataset[self.DecCol])

        self.observations = dataset

        # get central pixel ID
        healpixID = hp.ang2pix(self.nside, RA,
                               Dec, nest=True, lonlat=True)

        # get nearby pixels
        vec = hp.pix2vec(self.nside, healpixID, nest=True)
        self.healpixIDs = hp.query_disc(
            self.nside, vec, 3.*np.deg2rad(widthRA), inclusive=False, nest=True)

        # get pixel coordinates
        coords = hp.pix2ang(self.nside, self.healpixIDs,
                            nest=True, lonlat=True)
        self.pixRA, self.pixDec = coords[0], coords[1]

        # display (RA,Dec) of pixels
        if display:
            print('number of pixels here', len(self.pixRA))
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(self.pixRA, self.pixDec, 'r*')
            dataSel.plot(ax)
            plt.show()

        # make groups by (RA,dec)
        dataset = dataset.round({self.RACol: 4, self.DecCol: 4})
        groups = dataset.groupby([self.RACol, self.DecCol])

        # match pixels to data
        time_ref = time.time()
        matched_pixels = groups.apply(
            lambda x: self.match(x, self.healpixIDs, self.pixRA, self.pixDec)).reset_index()

        """
        print('after matching', time.time()-time_ref,
              len(matched_pixels['healpixID'].unique()))
        """
        return matched_pixels

    def match(self, grp, healpixIDs, pixRA, pixDec):
        """
        Method to match a set of pixels to a grp of observations

        Parameters
        ---------------
        grp: pandas grp
           observations
        healpixIDs: list(int)
          pixels IDs
        pixRA: list(float)
          pixel RAs
        pixDec: list(float)
          pixel Decs

        Returns:
        ----------
        pandas df with the following cols:
        fieldRA, fieldDec: RA and Dec of observations (FP center)
        healpixID,pixRA,pixDec: pixels lying inside LSST FP centered in (fieldRA, fieldDec)

        """

        # print('hello', grp.columns)
        pixRA_rad = np.deg2rad(pixRA)
        pixDec_rad = np.deg2rad(pixDec)

        # convert data position in rad
        pRA = np.median(grp[self.RACol])
        pDec = np.median(grp[self.DecCol])
        pRA_rad = np.deg2rad(pRA)
        pDec_rad = np.deg2rad(pDec)

        # gnomonic projection of pixels on the focal plane
        x, y = proj_gnomonic_plane(pRA_rad, pDec_rad, pixRA_rad, pixDec_rad)
        # x, y = proj_gnomonic_plane(np.deg2rad(self.LSST_RA-pRA),np.deg2rad(self.LSST_Dec-pDec), pixRA_rad, pixDec_rad)

        # get LSST FP with the good scale
        # pnew = LSSTPointing(0., 0., area=np.pi*self.fpscale**2)
        fpnew = LSSTPointing_circular(0., 0., maxbound=self.fpscale)
        # fpnew = LSSTPointing(np.deg2rad(self.LSST_RA-pRA),np.deg2rad(self.LSST_Dec-pDec),area=np.pi*self.fpscale**2)
        # maxbound=self.fpscale)

        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(x, y, 'ko')
        pf = PolygonPatch(fpnew, facecolor=(0, 0, 0, 0), edgecolor='red')
        ax.add_patch(pf)
        plt.show()
        """

        # print(shapely.vectorized.contains(
        #    fpnew, x, y), self.fpscale, fpnew.area)

        idf = shapely.vectorized.contains(fpnew, x, y)

        pixID_matched = list(healpixIDs[idf])
        pixRA_matched = list(pixRA[idf])
        pixDec_matched = list(pixDec[idf])

        # names = [grp.name]*len(pixID_matched)
        df_pix = pd.DataFrame({'healpixID': pixID_matched,
                               'pixRA': pixRA_matched,
                               'pixDec': pixDec_matched, })
        # 'groupName': names})

        return df_pix
        """
        n_index = len(grp.index.values)

        arr_index = grp.index.values

        n_pix = len(df_pix)
        if n_pix > 1:
            arr_index = arr_index.repeat(n_pix)
        if n_index > 1:
            df_pix = df_pix.append([df_pix]*(n_index-1), ignore_index=True)

        df_pix.loc[:, 'index'] = arr_index
        
        return df_pix
        """

    def plot(self, pixels, plt):
        """
         Method to plot matching results
         For each observation, the LSST FP is drawn as well as the center of matched pixels

         """

        print(np.unique(pixels[[self.RACol, self.DecCol]], axis=0))
        for vv in np.unique(pixels[[self.RACol, self.DecCol]], axis=0):
            fig, ax = plt.subplots()
            # plot all the pixels candidate for matching
            ax.plot(self.pixRA, self.pixDec, 'ko', mfc='None')
            fpnew = LSSTPointing(vv[0], vv[1], area=9.6)
            pf = PolygonPatch(fpnew, facecolor=(0, 0, 0, 0), edgecolor='red')
            ax.add_patch(pf)
            # compare with a circular FP
            fpnew_c = LSSTPointing_circular(vv[0], vv[1], area=9.6)
            pfc = PolygonPatch(fpnew_c, facecolor=(
                0, 0, 0, 0), edgecolor='red')
            ax.add_patch(pfc)
            idf = np.abs(pixels[self.RACol]-vv[0]) < 1.e-5
            idf &= np.abs(pixels[self.DecCol]-vv[1]) < 1.e-5
            ax.plot(pixels[idf]['pixRA'], pixels[idf]['pixDec'], 'r*')
            plt.show()


class ProcessPixels:
    def __init__(self, metricList, ipoint, outDir='', dbName='', RACol='fieldRA', DecCol='fieldDec', saveData=False):
        """
        class to process metrics on a set of data corresponding to pixels

        Parameters
        --------------
        metricList: list(metrics)
          list of sn_metrics to process
        ipoint: int
         internal parameter 
        outDir: str, opt
          output directory (default: '')
        dbName: str,
          observing strategy name (default: '')
        RACol: str, opt
          RA name (default: ='fieldRA')
       DecCol: str, opt
          Dec name (default: ='fieldDec')
       saveData: bool,opt
         to save the data (or not) (default: False)

        """

        self.metricList = metricList
        self.RACol = RACol
        self.DecCol = DecCol
        self.saveData = saveData
        self.outDir = outDir
        self.dbName = dbName
        self.num = ipoint

        # data will be save so clean the output directory first
        if self.saveData:
            self.clean()

    def clean(self):
        """
        Method to clean potential existing output files

        """

        for metric in self.metricList:
            listf = glob.glob(
                '{}/*_{}_{}*'.format(self.outDir, metric.name, self.num))
            if len(listf) > 0:
                for val in listf:
                    os.system('rm {}'.format(val))

    def __call__(self, pixels, observations, ip):
        """
        Main processing here

        Parameters
        --------------
        pixels: pandas df
          containing list of pixels (healpixID, pixRA, pixDec) with corresponding observations (self.RACol, self.DecCol)
        observations: array
           array of observations (from the scheduler)
        ip: int,
          internal parameter


        """
        # metric results are stored in a dict
        self.resfi = {}
        for metric in self.metricList:
            self.resfi[metric.name] = pd.DataFrame()

        data = pd.DataFrame(observations)

        # run the metrics on those pixels
        ipix = -1  # counter to estimate when to dump
        isave = -1  # counter to estimate how many dumps
        """
        print('number of pixels', len(pixels),
              len(pixels['healpixID'].unique()))
        """
        for ipixel, vv in enumerate(pixels['healpixID'].unique()):
            #print('processing pixel', ipixel, vv)
            time_ref = time.time()
            ipix += 1
            idf = pixels['healpixID'] == vv
            selpix = pixels[idf]
            dataPixels = self.getData(data, selpix)
            #print('got datapixels', time.time()-time_ref, selpix)
            # dataPixels = data.iloc[selpix['index'].tolist()].copy()

            for val in ['healpixID', 'pixRA', 'pixDec']:
                dataPixels[val] = selpix[val].unique().tolist()*len(dataPixels)
            # time_ref = time.time()

            dataPixels['iproc'] = [self.num]*len(dataPixels)

            self.runMetrics(dataPixels)
            # print('pixel processed',ipixel,time.time()-time_ref)
            if self.saveData and ipix >= 20:
                isave += 1
                self.dump(ip, isave)
                ipix = -1

        if ipix >= 0 and self.saveData:
            isave += 1
            self.dump(ip, isave)
            ipix = -1

        self.finish()

    def finish(self):
        """
        Method to save metadata for simulation


        """
        for metric in self.metricList:
            if metric.name == 'simulation':
                metric.finish()

    def getData(self, data, selpix):
        """
        Method to select data from a list

        Parameters
        ---------------
        data: pandas df
          observations to select; Should contain at least self.RACol and self.DecCol cols
        selpix: pandas df
          data used for selection. Should contain at least self.RACol and self.DecCol cols

        Returns
        ----------
        dataPixel: pandas df of selected observations

        """
        # idfb = [((data[self.RACol] - lat)**2 + (data[self.DecCol] - lon)**2).idxmin() for index,lat, lon in selpix[[self.RACol,self.DecCol]].itertuples()]
        dataPixel = pd.DataFrame()
        dataset = pd.DataFrame(data)

        dataset = dataset.round({self.RACol: 4, self.DecCol: 4})
        ido = dataset[self.RACol].isin(selpix[self.RACol])
        ido &= dataset[self.DecCol].isin(selpix[self.DecCol])

        return dataset[ido]
        """
        for index, row in selpix.iterrows():
            idfb = np.abs(data[self.RACol] -row[self.RACol])<1.e-4
            idfb &= np.abs(data[self.DecCol] -row[self.DecCol])<1.e-4
            dataPixel = pd.concat((dataPixel,data[idfb]),sort=False)
        return dataPixel
        """

    def runMetrics(self, dataPixel):
        """
        Method to run the metrics on the data

        Parameters
        --------------
        dataPixel: array
          set of data used as input to the metric

        """

        resdict = {}
        # run the metrics on these data
        if len(dataPixel) <= 5:
            return
        for metric in self.metricList:
            resdict[metric.name] = metric.run(
                season(dataPixel.to_records(index=False)))
            # print('running',len(resdict[metric.name]))

        # concatenate the results
        for key in self.resfi.keys():
            if resdict[key] is not None:
                self.resfi[key] = pd.concat((self.resfi[key], resdict[key]))

    def dump(self, ipoint, isave):
        """
        Method to dump results in hdf5 file

        Parameters
        --------------
        ipoint: int
         internal parameter
        isave: int
          number of dumps for this file

        """

        for key, vals in self.resfi.items():
            outName = '{}/{}_{}_{}.hdf5'.format(self.outDir,
                                                self.dbName, key, self.num)
            if vals is not None:
                # transform to astropy table to dump in hdf5 file
                tab = Table.from_pandas(vals)
                keyhdf = 'metric_{}_{}_{}'.format(self.num, ipoint, isave)
                tab.write(outName, keyhdf, append=True, compression=True)

        # reset the metric after dumping
        for metric in self.metricList:
            self.resfi[metric.name] = pd.DataFrame()


class ProcessArea:
    def __init__(self, nside, RACol, DecCol, num, outDir, dbName, saveData=False):
        """
        class to process (ie apply a metric) a given part of the sky

        Parameters
        ---------------
        nside: int
          nside parameter for healpix tessalation
        RACol: str
          name of the RA field
        DecCol: str
          name of the Dec field      
        num: int
          index (related to multiprocessing)
        outDir: str
           output dir path
        dbName: str
           observing strategy name
        saveData: bool, opt
           to save (True) the data or not (False) (default: False)


        """

        # load parameters
        self.nside = nside
        self.RACol = RACol
        self.DecCol = DecCol
        self.num = num
        self.outDir = outDir

        self.dbName = dbName
        self.saveData = saveData

        # get the LSST focal plane scale factor
        # corresponding to a sphere radius equal to one
        # (which is the default for gnomonic projections here)

        fov = 9.62*(np.pi/180.)**2  # LSST fov in sr
        theta = 2.*np.arcsin(np.sqrt(fov/(4.*np.pi)))

        # if theta >= np.pi/2.:
        #    theta -= np.pi/2.
        # print('theta', theta, np.rad2deg(theta))
        self.fpscale = np.tan(theta)

    def __call__(self, data, metricList, RA, Dec, widthRA, widthDec, ipoint, nodither=False, display=False):
        """
        call method: this is where the processing is.

        Parameters
        --------------
        data: numpy array
          data to process
        metricList: list(metric)
           list of metric to process
        RA: float
           RA position (center of the area to process)
        Dec: float
           Dec position (center of the area to process)
        widthRA: float
          width in RA of the area to process
        widthDec: float
          width in Dec of the area to process
        ipoint: int
        nodither: bool,opt
          to remove dithering (default: False)
        display: bool, opt
          to display matching FP/observations in "real time" (default: False)

        """

        # metric results are stored in a dic
        resfi = {}
        for metric in metricList:
            resfi[metric.name] = None
        # select data inside the area

        # display: (RA,Dec) distribution of the data
        if display:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(data[self.RACol], data[self.DecCol], 'ko')
            plt.show()

        # select data inside an area centered in (RA,Dec) with width (widthRA+1,widthDec+1)
        dataSel = DataInside(data, RA, Dec, widthRA+1., widthDec+1.,
                             RACol=self.RACol, DecCol=self.DecCol)

        # display: (RA,Dec) distribution of the selected data (ie inside the area)
        # if display:
        #    dataSel.plot()
        # Possible to remove DD dithering here
        # This is usually to test impact of dithering on DDF
        if nodither:
            dataSel[self.RACol] = np.mean(dataSel[self.RACol])
            dataSel[self.DecCol] = np.mean(dataSel[self.DecCol])

        if dataSel is not None:

            # mv to panda df
            dataset = pd.DataFrame(np.copy(dataSel.data))

            # get central pixel ID
            healpixID = hp.ang2pix(self.nside, RA,
                                   Dec, nest=True, lonlat=True)

            # get nearby pixels
            vec = hp.pix2vec(self.nside, healpixID, nest=True)
            healpixIDs = hp.query_disc(
                self.nside, vec, 3.*np.deg2rad(widthRA), inclusive=False, nest=True)

            # get pixel coordinates
            coords = hp.pix2ang(self.nside, healpixIDs, nest=True, lonlat=True)
            pixRA, pixDec = coords[0], coords[1]

            # display (RA,Dec) of pixels
            if display:
                print('number of pixels here', len(pixRA))
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(pixRA, pixDec, 'r*')
                dataSel.plot(ax)
                plt.show()

            # make groups by (RA,dec)
            dataset = dataset.round({self.RACol: 4, self.DecCol: 4})
            groups = dataset.groupby([self.RACol, self.DecCol])

            # display matching pixels/observations
            if display:
                import matplotlib.pylab as plt
                for name, group in groups:
                    fig, ax = plt.subplots()
                    print('matching pixels')
                    self.match(group, healpixIDs, pixRA, pixDec, name, ax=ax)
                    # ax.plot(dataset[self.RACol],dataset[self.DecCol],'bs',mfc='None')
                    plt.show()

            # process pixels with data
            # match pixels to data
            time_ref = time.time()
            matched_pixels = groups.apply(
                lambda x: self.match(x, healpixIDs, pixRA, pixDec)).reset_index()

            """
            print('after matching', time.time()-time_ref,
                  len(matched_pixels['healpixID'].unique()))
            """
            # print('number of pixels',len(matched_pixels['healpixID'].unique()))
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

                pixRA = thematch['pixRA'].unique()
                pixDec = thematch['pixDec'].unique()

                dataPixel.loc[:, 'healpixID'] = healpixID
                dataPixel.loc[:, 'pixRA'] = pixRA[0]
                dataPixel.loc[:, 'pixDec'] = pixDec[0]

                resdict = {}
                time_ref = time.time()

                # run the metrics on those pixels
                for metric in metricList:
                    resdict[metric.name] = metric.run(
                        season(dataPixel.to_records(index=False)))

                # concatenate the results
                for key in resfi.keys():
                    if resdict[key] is not None and resdict[key].size > 0:
                        if resfi[key] is None:
                            resfi[key] = resdict[key]
                        else:
                            # print('here pal',type(resdict[key]),type(resfi[key]))
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

        # print(tabnames, len(tabnames))
        result_queue = multiprocessing.Queue()

        for j in range(len(tabnames)-1):

            ida = tabnames[j]
            idb = tabnames[j+1]

            # print('Field', names[ida:idb])
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

    def match(self, grp, healpixIDs, pixRA, pixDec, name=None, ax=None):

        # print('hello', grp.columns)
        pixRA_rad = np.deg2rad(pixRA)
        pixDec_rad = np.deg2rad(pixDec)

        # convert data position in rad
        pRA = np.median(grp[self.RACol])
        pDec = np.median(grp[self.DecCol])
        pRA_rad = np.deg2rad(pRA)
        pDec_rad = np.deg2rad(pDec)

        # gnomonic projection of pixels on the focal plane
        # x, y = proj_gnomonic_plane(pRA_rad, pDec_rad, pixRA_rad, pixDec_rad)
        x, y = proj_gnomonic_plane(pDec_rad, pRA_rad, pixDec_rad, pixRA_rad)

        # print('after gnomonic')
        # print(x, y)
        # get LSST FP with the good scale
        fpnew = LSSTPointing_circular(0., 0., maxbound=self.fpscale)

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
        x = np.rad2deg(x)+pRA
        y = np.rad2deg(y)+pDec

        if ax is not None:
            ax.plot(x, y, 'ks')
        # points inside the focal plane
        fp = LSSTPointing(pRA, pDec, area=9.6)
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
            for val in ['healpixID', 'pixRA', 'pixDec']:
                grp.loc[:, val] = np.copy(pixel[val])
        """
        """
        matched_pixels = pd.DataFrame()
        matched_pixels.iloc[:, 'healpixID'] = healpixIDs[idf]
        matched_pixels.iloc[:, 'grname'] = grp.name
        """

        pixID_matched = list(healpixIDs[idf])
        pixRA_matched = list(pixRA[idf])
        pixDec_matched = list(pixDec[idf])

        if ax is not None:
            ax.plot(pixRA, pixDec, 'ko', mfc='None')
            ax.plot(pixRA[idf], pixDec[idf], 'r*')
            fpnew = LSSTPointing(pRA, pDec, area=9.6)
            pf = PolygonPatch(fpnew, facecolor=(0, 0, 0, 0), edgecolor='red')
            ax.add_patch(pf)
            print('printing here')
            print('matching', grp[[self.RACol, self.DecCol, 'filter']], pixID_matched, len(
                pixID_matched), pixRA_matched, pixDec_matched)

        if name is not None:
            names = [name]*len(pixID_matched)
        else:
            names = [grp.name]*len(pixID_matched)
        # names = ['test']*len(pixID_matched)
        # return pd.Series([matched_pixels], ['healpixIDs'])

        n_index = len(grp.index.values)

        arr_index = grp.index.values

        # arr_index = np.reshape(arr_index,(len(arr_index),1))
        # print('hhh',arr_index)
        df_pix = pd.DataFrame({'healpixID': pixID_matched,
                               'pixRA': pixRA_matched,
                               'pixDec': pixDec_matched,
                               'groupName': names})

        # print(arr_index,df_pix)
        # if n_index > 1:
        #    print('here',n_index,type(grp.index.values))

        n_pix = len(df_pix)
        # n_index = len(df_index)
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
        grp.loc[:, 'pixRA'] = pixRA_matched
        grp.loc[:, 'pixDec'] = pixDec_matched


class ObsPixel:
    """
    This class is deprecated
    """

    def __init__(self, nside, data, RACol='RA', DecCol='Dec'):
        self.nside = nside
        self.data = data
        self.RACol = RACol
        self.DecCol = DecCol
        # self.hppix = HEALPix(nside=self.nside, order='nested')

    def matchFast(self, pixel, ax=None):

        time_ref = time.time()
        data = self.pointingsAreaFast(pixel['pixRA'], pixel['pixDec'], 3.)

        if ax is not None:
            val = np.unique(np.unique(data[[self.RACol, self.DecCol]]))

            ax.plot(val[self.RACol], val[self.DecCol], 'ko')

        print('pointing', time.time()-time_ref)
        if data is None:
            return None

        dataset = pd.DataFrame(np.copy(data))
        if ax is not None:
            ax.plot(pixel['pixRA'], pixel['pixDec'], 'r*')
            ax.plot(dataset[self.RACol], dataset[self.DecCol], 'bo')
        # for (pRA,pDec) in np.unique(dataset[[self.RACol,self.DecCol]]):
        groups = dataset.groupby([self.RACol, self.DecCol])

        time_ref = time.time()
        print('ngroups', len(groups))
        seldata = None
        for name, group in groups:
            pRA = np.mean(group[self.RACol])
            pDec = np.mean(group[self.DecCol])

            # convert data position in rad
            pRA_rad = np.deg2rad(pRA)
            pDec_rad = np.deg2rad(pDec)

            # gnomonic projection of pixels on the focal plane
            x, y = proj_gnomonic_plane(pRA_rad, pDec_rad, np.deg2rad(
                pixel['pixRA']), np.deg2rad(pixel['pixDec']))

            x = np.rad2deg(x)+pRA
            y = np.rad2deg(y)+pDec

            if ax is not None:
                ax.plot(x, y, 'ks')
            # points inside the focal planes
            fp = LSSTPointing(pRA, pDec, 0.)
            idf = shapely.vectorized.contains(fp, x, y)
            # print(idf)
            if ax is not None:
                pf = PolygonPatch(fp, facecolor=(
                    0, 0, 0, 0), edgecolor='red')
                ax.add_patch(pf)
            if idf:
                grp = group.copy()
                for val in ['healpixID', 'pixRA', 'pixDec']:
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
            pixel['pixRA'] * u.deg, pixel['pixDec'] * u.deg, radius=3*u.deg)

        # convert to (RA,Dec)
        coordpix = hp.pix2ang(self.nside, healpixID_around,
                              nest=True, lonlat=True)
        pixelsRA, pixelsDec = coordpix[0], coordpix[1]

        ax.plot(pixelsRA, pixelsDec, 'rs')
        # get the center of the pixels map

        center_RA = np.mean(pixelsRA)
        center_Dec = np.mean(pixelsDec)

        # get the data around this center
        print('center', center_RA, center_Dec, pixel['pixRA'], pixel['pixDec'])

        dataset = self.pointingsAreaFast(pixel['pixRA'], pixel['pixDec'], 3.)
        if dataset is None:
            return None

        # for each data set
        # check which pixel is inside the focal plane
        # after gnomonic projection

        for (pRA, pDec) in np.unique(dataset[[self.RACol, self.DecCol]]):

            # convert data position in rad
            pRA_rad = np.deg2rad(pRA)
            pDec_rad = np.deg2rad(pDec)

            # gnomonic projection of pixels
            x, y = proj_gnomonic_plane(pRA_rad, pDec_rad, np.deg2rad(
                pixelsRA), np.deg2rad(pixelsDec))

            x = np.rad2deg(x)+pRA
            y = np.rad2deg(y)+pDec

            ax.plot(x, y, 'ks')
            # points inside the focal plane
            fp = LSSTPointing(pRA, pDec, 0.)
            idf = shapely.vectorized.contains(fp, x, y)
            print('matching', pixel['healpixID'], healpixID_around[idf])
            if ax is not None:
                pf = PolygonPatch(fp, facecolor=(
                    0, 0, 0, 0), edgecolor='red')
                ax.add_patch(pf)

    def pointingsAreaFast(self, pixRA, pixDec, width):

        # Warning here
        # RA is in [0,360.]
        # Special care near 0 and 360...

        RAmin = pixRA-width
        RAmax = pixRA+width
        Decmin = pixDec-width
        Decmax = pixDec+width

        areas = []
        # print('there man',RAmin,RAmax,Decmin,Decmax)
        if RAmin < 0:
            areas.append([0., RAmax, Decmin, Decmax])
            areas.append([RAmin+360., 0., Decmin, Decmax])
        else:
            if RAmax > 360.:
                areas.append([0., RAmax-360., Decmin, Decmax])
                areas.append([RAmin, 0., Decmin, Decmax])
            else:
                areas.append([RAmin, RAmax, Decmin, Decmax])

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

        RAmin = area[0]
        RAmax = area[1]
        Decmin = area[2]
        Decmax = area[3]

        # print('there man',RAmin,RAmax,Decmin,Decmax)
        # print('mmmm',np.min(self.data[self.RACol]),
        # np.max(self.data[self.RACol]),np.min(self.data[self.DecCol]),
        # np.max(self.data[self.DecCol]))

        if RAmax < 1.e-3:
            idx = (self.data[self.RACol] >= RAmin)

        else:
            # print('booh')
            idx = self.data[self.RACol] >= RAmin
            idx &= self.data[self.RACol] <= RAmax

        idx &= self.data[self.DecCol] >= Decmin
        idx &= self.data[self.DecCol] <= Decmax

        res = np.copy(self.data[idx])

        if len(res) == 0:
            return None
        """
        res = rf.append_fields(res, 'healpixID', [pixid]*len(res))
        res = rf.append_fields(res, 'pixRA', [pixRA]*len(res))
        res = rf.append_fields(res, 'pixDec', [pixDec]*len(res))
        """
        return res


class ObsPixel_old:
    """
    This class is deprecated
    """

    def __init__(self, nside, data, scanzone=None, RACol='RA', DecCol='Dec'):
        self.nside = nside
        self.data = data
        self.RACol = RACol
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
        pixRA, pixDec = coordpix[0], coordpix[1]
        focalplanes = self.pointingsAreaFast(healpixID, pixRA, pixDec, 3.)

        for val in focalplanes:
            lsstpoly = LSSTPointing(val[self.RACol], val[self.DecCol])
            xp = lsstpoly.exterior.coords.xy[0]
            yp = lsstpoly.exterior.coords.xy[1]
            print(hp.query_polygon(self.nside, [xp, yp, [0.0]*len(xp)]))

        return None

    def matchFast(self, healpixID, ax=None):

        step = 1
        lon, lat = self.hppix.boundaries_lonlat(healpixID, step=step)
        lon = lon.to(u.deg).value
        lat = lat.to(u.deg).value

        coordpix = hp.pix2ang(self.nside, healpixID, nest=True, lonlat=True)
        pixRA, pixDec = coordpix[0], coordpix[1]
        vertices = np.vstack([lon.ravel(), lat.ravel()]).transpose()
        poly = geometry.Polygon(vertices)
        focalplanes = self.pointingsAreaFast(healpixID, pixRA, pixDec, 3.)
        print('pixel area', poly.area)

        # print(self.scanzone.centroid.x,self.scanzone.centroid.y)
        polyscan = affinity.translate(
            self.scanzone, xoff=pixRA-self.scanzone.centroid.x, yoff=pixDec-self.scanzone.centroid.y)
        # check wether this polyscan goes beyond 360. in RA
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
                    polyl, focalplanes[self.RACol], focalplanes[self.DecCol])
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
                ax.plot(focalplanes[idf][self.RACol],
                        focalplanes[idf][self.DecCol], 'gs')
                ax.plot(focalplanes[self.RACol],
                        focalplanes[self.DecCol], 'r*')
            ax.plot(polyscan.exterior.coords.xy[0],
                    polyscan.exterior.coords.xy[1], 'k.')
            ax.set_xlabel('RA [deg]')
            ax.set_ylabel('Dec [deg]')

        return finalData

    def pointingsAreaFast(self, pixid, pixRA, pixDec, width):

        # Warning here
        # RA is in [0,360.]
        # Special care near 0 and 360...

        RAmin = pixRA-width
        RAmax = pixRA+width
        Decmin = pixDec-width
        Decmax = pixDec+width

        areas = []
        # print('there man',RAmin,RAmax,Decmin,Decmax)
        if RAmin < 0:
            areas.append([0., RAmax, Decmin, Decmax])
            areas.append([RAmin+360., 0., Decmin, Decmax])
        else:
            if RAmax > 360.:
                areas.append([0., RAmax-360., Decmin, Decmax])
                areas.append([RAmin, 0., Decmin, Decmax])
            else:
                areas.append([RAmin, RAmax, Decmin, Decmax])

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

        RAmin = area[0]
        RAmax = area[1]
        Decmin = area[2]
        Decmax = area[3]

        # print('there man',RAmin,RAmax,Decmin,Decmax)
        # print('mmmm',np.min(self.data[self.RACol]),
        # np.max(self.data[self.RACol]),np.min(self.data[self.DecCol]),
        # np.max(self.data[self.DecCol]))

        if RAmax < 1.e-3:
            idx = (self.data[self.RACol] >= RAmin)

        else:
            # print('booh')
            idx = self.data[self.RACol] >= RAmin
            idx &= self.data[self.RACol] <= RAmax

        idx &= self.data[self.DecCol] >= Decmin
        idx &= self.data[self.DecCol] <= Decmax

        res = np.copy(self.data[idx])

        if len(res) == 0:
            return None
        res = rf.append_fields(res, 'healpixID', [pixid]*len(res))
        res = rf.append_fields(res, 'pixRA', [pixRA]*len(res))
        res = rf.append_fields(res, 'pixDec', [pixDec]*len(res))

        return res

    def __call__(self, healpixID, ax=None):
        return self.matchFast(healpixID, ax)


class OverlapGnomonic:
    """
    This class is deprecated
    """

    def __init__(self, nside, dRA=0., dDec=0.):

        self.nside = nside
        self.hppix = HEALPix(nside=nside, order='nested', frame='icrs')
        self.dRA = dRA
        self.dDec = dDec

    def overlap_pixlist(self, pixelList, pointing, ax=None):

        pRA = pointing[0]
        pDec = pointing[1]

        pRA_rad = np.deg2rad(pointing[0])
        pDec_rad = np.deg2rad(pointing[1])

        x, y = proj_gnomonic_plane(pRA_rad, pDec_rad, np.deg2rad(
            pixelList['pixRA']), np.deg2rad(pixelList['pixDec']))
        x = np.rad2deg(x)+pRA
        y = np.rad2deg(y)+pDec

        fp = LSSTPointing(pRA, pDec, 0.)
        idf = shapely.vectorized.contains(fp, x, y)
        print(pixelList[idf])

        for pixel in pixelList:
            pixRA, pixDec, poly, polyb = self.polypix(pixel['healpixID'])
            # gnomonic proj
            pixRA_rad = np.deg2rad(pixRA)
            pixDec_rad = np.deg2rad(pixDec)
            x, y = proj_gnomonic_plane(
                pRA_rad, pDec_rad, np.deg2rad(pixRA), np.deg2rad(pixDec))
            # print('pixarea',poly.area,hp.nside2pixarea(self.nside,degrees=True))
            if ax is not None:
                p = PolygonPatch(poly, facecolor='#fffffe', edgecolor='black')
                ax.add_patch(p)
                # p = PolygonPatch(polyb, facecolor='#fffffe', edgecolor='black')
                # ax.add_patch(polyb)
                print(x, y)
                p = Point(x, y)
                ax.plot(np.rad2deg(x)+pRA, np.rad2deg(y)+pDec, 'ks')

        fp = LSSTPointing(pRA, pDec, 0.)
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
        pixRA, pixDec = coordpix[0], coordpix[1]
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
        return pixRA, pixDec, poly, polyb


class GetOverlap:
    """
    This class is deprecated
    """

    def __init__(self, nside, dRA=0., dDec=0.):

        self.nside = nside
        self.hppix = HEALPix(nside=nside, order='nested', frame='icrs')
        self.dRA = dRA
        self.dDec = dDec

    def polypix(self, healpixID=10):

        step = 1
        lon, lat = self.hppix.boundaries_lonlat(healpixID, step=step)
        lon = lon.to(u.deg).value
        lat = lat.to(u.deg).value

        coordpix = hp.pix2ang(self.nside, healpixID, nest=True, lonlat=True)
        pixRA, pixDec = coordpix[0], coordpix[1]
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
        return pixRA, pixDec, poly, polyb

    def overlap_pixlist(self, pixelList, pointing, ax=None):

        for healpixID in pixelList:
            pixRA, pixDec, poly, polyb = self.polypix(healpixID)
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

    def overlap(self, healpixID=100, pointingRA=None, pointingDec=None, ax=None):

        # get initial pixel
        pixRA, pixDec, poly = self.polypix(healpixID)

        if PointingRA is None:
            fpRA = pixRA+self.dRA
            fpDec = pixDec+self.dDec
        else:
            fpRA = pointingRA+self.dRA
            fpDec = pointingDec+self.dDec

        # define a focal plane centered on this pixel
        fp = LSSTPointing(fpRA, fpDec, 0.)

        # get nearby pixels
        healpixID_around = self.hppix.cone_search_lonlat(
            pixRA * u.deg, pixDec * u.deg, radius=3 * u.deg)
        coordpix = hp.pix2ang(self.nside, healpixID_around,
                              nest=True, lonlat=True)
        # coords = SkyCoord(coordpix[0], coordpix[1], unit='deg')
        # print(coordpix[0])
        arr = np.array(healpixID_around, dtype=[('healpixID', 'i8')])
        arr = rf.append_fields(arr, 'pixRA', coordpix[0])
        arr = rf.append_fields(arr, 'pixDec', coordpix[1])

        res = []

        for val in arr:

            pRA, pDec, poly = self.polypix(val['healpixID'])
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
            res.append((self.nside, val['healpixID'], val['pixRA'], val['pixDec'],
                        overlap, fpRA, fpDec, val['pixRA']-fpRA, val['pixDec']-fpDec, pixArea))

        resrec = np.rec.fromrecords(res, names=[
            'nside', 'healpixID', 'pixRA', 'pixDec', 'overlap', 'fpRA', 'fpDec', 'DRA', 'DDec', 'pixArea'])

        return resrec


class GetShape:
    """
    This class is deprecated
    """

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
        pixRA, pixDec = coordpix[0], coordpix[1]
        vertices = np.vstack([lon.ravel(), lat.ravel()]).transpose()
        # print(vertices)
        poly = geometry.Polygon(vertices)
        scanzone = self.followShape(poly, ax)

        # This is for display
        if ax is not None:
            p = PolygonPatch(poly, facecolor='#fffffe', edgecolor='red')
            ax.add_patch(p)
            width = 3.
            ax.set_xlim([pixRA-width, pixRA+width])
            ax.set_ylim([pixDec-width, pixDec+width])
            ax.set_xlabel('RA [deg]')
            ax.set_ylabel('Dec [deg]')

        return scanzone

    def followShape(self, poly, ax=None):

        xpoly = poly.exterior.coords.xy[0]
        ypoly = poly.exterior.coords.xy[1]
        pixRA = np.mean(xpoly)
        pixDec = np.mean(ypoly)
        pixArea = poly.area

        # print('allo',pixArea)
        dRA = 5.
        dDec = 5.

        RAVals = np.arange(pixRA-dRA, pixRA+dRA, 0.1)
        DecVals = np.arange(pixDec-dDec, pixDec+dDec, 0.1)
        r = []
        for RA in RAVals:
            for Dec in DecVals:
                fp = LSSTPointing(RA, Dec, 0.)
                area = poly.intersection(fp).area/fp.area
                area = poly.intersection(fp).area/pixArea
                # print('alors',RA,Dec,area)
                if area >= self.overlap:
                    r.append((RA, Dec, area))

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
            pixRA = np.mean(xpoly)
            pixDec = np.mean(ypoly)

            p = PolygonPatch(polyshape, edgecolor='blue')
            ax.add_patch(p)
            ax.set_xlim([pixRA-width, pixRA+width])
            ax.set_ylim([pixDec-width, pixDec+width])
            ax.set_xlabel('RA [deg]')
            ax.set_ylabel('Dec [deg]')

        return polyshape


def getFields_fromId(observations, fieldIds):
    """
    Function to get a set of fields
    from a set of observations using fieldIds (internal scheduler parameter)

    Parameters
    ---------------
    observations: array
       data to process
    fieldIds: list(int), opt
       list of fieldIds to consider (dafault: None)

    Returns
    ----------
    array of observations corresponding to fieldIds

    """

    obs = None
    for fieldId in fieldIds:
        idf = observations['fieldId'] == fieldId
        if obs is None:
            obs = observations[idf]
        else:
            obs = np.concatenate((obs, observations[idf]))
    return obs


def getFields(observations, fieldType='WFD', fieldIds=None,
              nside=64, RACol='fieldRA', DecCol='fieldDec'):
    """
    Function to get a type of field (DD, WFD)
    from a set of observations

    Parameters
    ---------------
    observations: array
       data to process
    fieldType: str, opt
       type of field (DD or WFD) to consider
    fieldIds: list(int), opt
       list of fieldIds to consider (dafault: None)
    nside: int, opt
       healpix nside parameter (default: 64)
    RACol: str, opt
       RA column name (default: fieldRA)
    DecCol: str, opt
       Dec column name (default: fieldDef)

    Returns
    ----------
    array

    """

    obs = None

    # this is for the WFD

    for pName in ['proposalId', 'survey_id']:
        if pName in observations.dtype.names:

            # print(np.unique(observations[pName]))
            propId = list(np.unique(observations[pName]))

            # loop on proposal id
            # and take the one with the highest number of evts
            propIds = list(np.unique(observations[pName]))
            """
            r = []
            for propId in propIds:
                idx = observations[pName] == propId
                r.append((propId, len(observations[idx])))

            res = np.rec.fromrecords(r, names=['propId', 'Nobs'])
            """
            if fieldType == 'WFD':
                # Take the propId with the largest number of fields
                #print('hello', res)
                #propId_WFD = propIds[np.argmax(res['Nobs'])]
                #print(res, np.argmax(res['Nobs']), propId_WFD)
                #a = observations['note']
                df = pd.DataFrame(np.copy(observations))
                if 'note' in df.columns:
                    idx = df['note'].str.contains('DD')
                    return df[~idx].to_records(index=False)
                else:
                    return df.to_records(index=False)
                # return observations[observations[pName] == propId_WFD]
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
                    return pixelate(obser, nside, RACol=RACol, DecCol=DecCol)

                else:
                    """
                    Tricky
                    we do not have other ways to identify
                    DD except by selecting pixels with a large number of visits
                    """
                    pixels = pixelate(observations, nside,
                                      RACol=RACol, DecCol=DecCol)

                    df = pd.DataFrame(np.copy(pixels))

                    groups = df.groupby('healpixID').filter(
                        lambda x: len(x) > 5000)

                    group_DD = groups.groupby([RACol, DecCol]).filter(
                        lambda x: len(x) > 4000)

                    # return np.array(group_DD.to_records().view(type=np.matrix))
                    return group_DD.to_records(index=False)
