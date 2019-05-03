import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rf
from shapely import geometry
from shapely import affinity
import shapely.vectorized
from astropy_healpix import HEALPix
from astropy import units as u
from descartes.patch import PolygonPatch

def renameFields(tab):

    #print(tab.dtype)
    corresp={}

    fillCorresp(tab,corresp,'mjd','observationStartMJD')
    fillCorresp(tab,corresp,'Ra','fieldRA')
    fillCorresp(tab,corresp,'Dec','fieldDec')
    fillCorresp(tab,corresp,'band','filter')
    fillCorresp(tab,corresp,'exptime','visitExposureTime')
    fillCorresp(tab,corresp,'nexp','numExposures')
    
    #print('alors',corresp)
    return rf.rename_fields(tab,corresp)

def fillCorresp(tab, corres, vara, varb):

    if vara in tab.dtype.names and varb not in tab.dtype.names:
        corres[vara] = varb

def pixelate(data,nside,RaCol='Ra',DecCol='Dec'):

        res = data.copy()
        npix = hp.nside2npix(nside)
        table = hp.ang2vec(res[RaCol], res[DecCol], lonlat=True)

        healpixs = hp.vec2pix(nside, table[:, 0], table[:, 1], table[:, 2], nest=True)
        coord = hp.pix2ang(nside, healpixs, nest=True, lonlat=True)

        res = rf.append_fields(res, 'healpixID', healpixs)
        res = rf.append_fields(res, 'pixRa', coord[0])
        res = rf.append_fields(res, 'pixDec', coord[1])

        return res

def season(obs, mjdCol='observationStartMJD'):
    
    obs.sort(order=mjdCol)
    
    if len(obs) == 1:
        obs = np.atleast_1d(obs)
        obs = rf.append_fields([obs],'season',[1.])
        return obs
    diff = obs[mjdCol][1:]-obs[mjdCol][:-1]
    flag = np.argwhere(diff>100.)
    if len(flag) > 0:
        seas = np.zeros((len(obs),))
        flag = flag+1
        seas[0:flag[0][0]] = 1
        for iflag in range(len(flag)-1):
            seas[flag[iflag][0]:flag[iflag+1][0]]= iflag+2
        seas[flag[-1][0]:] = len(flag)+1
        obs = rf.append_fields(obs,'season',seas)
    else:
        obs = rf.append_fields(obs,'season',[1.]*len(obs))
    return obs

def LSSTPointing(xc, yc, angle_rot = 0., fov=9.6):

    arr = [[3, 0], [12, 0], [12, 1], [13, 1], [13, 2], [14, 2], [14, 3], [15, 3],
           [15, 12], [14, 12], [14, 13], [13, 13], [
               13, 14], [12, 14], [12, 15],
           [3, 15], [3, 14], [2, 14], [2, 13], [1, 13], [1, 12], [0, 12],
           [0, 3], [1, 3], [1, 2], [2, 2], [2, 1], [3, 1]]

    poly_orig = geometry.Polygon(arr)
    reduced_poly = affinity.scale(poly_orig, xfact=np.sqrt(
        fov/poly_orig.area), yfact=np.sqrt(fov/poly_orig.area))

    rotated_poly = affinity.rotate(reduced_poly,angle_rot)

    return affinity.translate(rotated_poly, 
                              xoff=xc-reduced_poly.centroid.x, 
                              yoff=yc-reduced_poly.centroid.y)


class ObsPixel:
    def __init__(self, nside, data, scanzone=None, RaCol='Ra', DecCol='Dec'):
        self.nside = nside
        self.data = data
        self.RaCol = RaCol
        self.DecCol = DecCol

        self.hppix = HEALPix(nside=self.nside, order='nested')
        
        self.scanzone = scanzone

    def matchFast(self,healpixID,ax=None):
        
        step = 1
        lon, lat = self.hppix.boundaries_lonlat(healpixID, step=step)
        lon = lon.to(u.deg).value
        lat = lat.to(u.deg).value
        coordpix = hp.pix2ang(self.nside, healpixID, nest=True, lonlat=True)
        pixRa, pixDec = coordpix[0], coordpix[1]
        vertices = np.vstack([lon.ravel(), lat.ravel()]).transpose()
        poly = geometry.Polygon(vertices)
        focalplanes = self.pointingsAreaFast(healpixID,pixRa,pixDec,7.)

        #print(self.scanzone.centroid.x,self.scanzone.centroid.y)
        polyscan = affinity.translate(self.scanzone, xoff=pixRa-self.scanzone.centroid.x, yoff=pixDec-self.scanzone.centroid.y)
        idf = shapely.vectorized.contains(polyscan,focalplanes[self.RaCol],focalplanes[self.DecCol])

        # This is for display
        if ax is not None:
            po = PolygonPatch(polyscan, facecolor='#fffffe', edgecolor='blue')
            ax.add_patch(po)
            p = PolygonPatch(poly, facecolor='#fffffe', edgecolor='red')
            ax.add_patch(p)
            ax.plot(focalplanes[idf][self.RaCol],focalplanes[idf][self.DecCol],'g.')
            ax.plot(polyscan.exterior.coords.xy[0],polyscan.exterior.coords.xy[1],'k.')
            ax.set_xlabel('Ra [deg]')
            ax.set_ylabel('Dec [deg]')

        return focalplanes[idf]

    def pointingsAreaFast(self, pixid, pixRA, pixDec, width):

        idx = np.abs(self.data[self.RaCol]-pixRA) <= width
        idx &= np.abs(self.data[self.DecCol]-pixDec) <= width

        res = self.data[idx]
        res = rf.append_fields(res,'healpixID',[pixid]*len(res))
        res = rf.append_fields(res,'pixRa',[pixRA]*len(res))
        res = rf.append_fields(res,'pixDec',[pixDec]*len(res))

        return res

    def __call__(self,healpixID,ax=None):
        return self.matchFast(healpixID,ax)


class GetShape:
    def __init__(self, nside):

        self.nside=nside
        self.hppix = HEALPix(nside=nside, order='nested')

    def shape(self,ax=None):
        healpixID = 10
        step = 1
        lon, lat = self.hppix.boundaries_lonlat(healpixID, step=step)
        lon = lon.to(u.deg).value
        lat = lat.to(u.deg).value
        coordpix = hp.pix2ang(self.nside, healpixID, nest=True, lonlat=True)
        pixRa, pixDec = coordpix[0], coordpix[1]
        vertices = np.vstack([lon.ravel(), lat.ravel()]).transpose()
        #print(vertices)
        poly = geometry.Polygon(vertices)
        scanzone = self.followShape(poly,ax)
       

        # This is for display
        if ax is not None:
            p = PolygonPatch(poly, facecolor='#fffffe', edgecolor='red')
            ax.add_patch(p)
            width = 3.
            ax.set_xlim([pixRa-width,pixRa+width])
            ax.set_ylim([pixDec-width,pixDec+width])
            ax.set_xlabel('Ra [deg]')
            ax.set_ylabel('Dec [deg]')
            
        return scanzone

    def followShape(self,poly,ax=None):

        xpoly = poly.exterior.coords.xy[0]
        ypoly = poly.exterior.coords.xy[1]
        pixRa =np.mean(xpoly)
        pixDec = np.mean(ypoly)

        dRa = 3.
        dDec = 3.

        RaVals = np.arange(pixRa-dRa,pixRa+dRa,0.1)
        DecVals = np.arange(pixDec-dDec,pixDec+dDec,0.1)
        r=[]
        for Ra in RaVals:
            for Dec in DecVals:
                fp = LSSTPointing(Ra,Dec,0.)
                area = poly.intersection(fp).area/fp.area
                if area > 0.:
                    r.append((Ra,Dec,area))

        if len(r)>0.:
            shape = np.rec.fromrecords(r, names= ['x','y','area'])
            idx = shape['area']>= 0.01
            sel = shape[idx]

            r = []
            for x in np.unique(sel['x']):
                id = np.abs(sel['x']-x)<1.e-5
                r.append((x,np.min(sel[id]['y'])))
                r.append((x,np.max(sel[id]['y'])))
                         
            sel = np.rec.fromrecords(r, names=['x','y'])


            ida = sel['y']>pixDec
            sela = sel[ida]
            sela.sort(order='x')
            
            idb = sel['y']<=pixDec
            selb = sel[idb]
            selb[::-1].sort(order='x')

            res = np.concatenate((sela,selb))
            polyshape= geometry.Polygon([[val['x'], val['y']] for val in res])
            
            
        if ax is not None:
            
            p = PolygonPatch(poly, facecolor='#fffffe', edgecolor='red')
            ax.add_patch(p)
            width = 3.
            pixRa =np.mean(xpoly)
            pixDec = np.mean(ypoly)
            
            p = PolygonPatch(polyshape,edgecolor='blue')
            ax.add_patch(p)
            ax.set_xlim([pixRa-width,pixRa+width])
            ax.set_ylim([pixDec-width,pixDec+width])
            ax.set_xlabel('Ra [deg]')
            ax.set_ylabel('Dec [deg]')
            
       
        
        return polyshape
