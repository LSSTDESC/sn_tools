import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import time
from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table

def plotPixels(fluxdist,seeing):
    """
    Method to plot pixel-map of flux distribution

    Parameters
    ---------------
    fluxdist: astropy table
      with the following clos: xpixel,ypixel,fluxdist
    seeing: float
       seeing value

    """

    fig,ax = plt.subplots()
    fig.suptitle('seeing: {}\'\''.format(seeing))
    xpixelmin, xpixelmax, xpixelstep, nxpixel = limVals(fluxdist, 'xpixel')
    ypixelmin, ypixelmax, ypixelstep, nypixel = limVals(fluxdist, 'ypixel')
    
    xpixelstep = np.round(xpixelstep, 3)
    ypixelstep = np.round(ypixelstep, 3)
    
    xpixelv = np.linspace(xpixelmin, xpixelmax, nxpixel)
    ypixelv = np.linspace(ypixelmin, ypixelmax, nypixel)
    
    index = np.lexsort((fluxdist['xpixel'], fluxdist['ypixel']))
    flux = np.reshape(fluxdist[index]['fluxdist'], (nxpixel, nypixel))
    
    grid=RegularGridInterpolator((xpixelv,ypixelv), flux, method='linear', bounds_error=False, fill_value=0.)
    
    X,Y = np.meshgrid(xpixelv,ypixelv)
    
    xmin = np.min(xpixelv)
    xmax = np.max(xpixelv)
    ymin = np.min(ypixelv)
    ymax = np.max(ypixelv)

    fluxpixels = grid((X,Y))
    im = ax.imshow(fluxpixels,extent=[xmin,xmax,ymin,ymax],vmin=np.min(fluxpixels),vmax=np.max(fluxpixels))

    fig.colorbar(im)
    
def limVals(lc, field):
    """ Get unique values of a field in  a table
    Parameters
    ----------
    lc: Table
        astropy Table (here probably a LC)
    field: str
        name of the field of interest
    
    Returns
    -------
    vmin: float
        min value of the field
    vmax: float
        max value of the field
    vstep: float
        step value for this field (median)
    nvals: int
        number of unique values
    """

    lc.sort(field)
    vals = np.unique(lc[field].data.round(decimals=4))
    # print(vals)
    vmin = np.min(vals)
    vmax = np.max(vals)
    vstep = np.median(vals[1:]-vals[:-1])

    return vmin, vmax, vstep, len(vals)


class PSF_Flux:
    """
    class to estimate the flux distribution over pixels
    as a function of the seeing

    Parameters
    ---------------
    seeing: float
      seeing value
    dx: float
      dx grid value for fluxes estimation
    dy: float
       dy grid value for flux estimation
    nsigma: float
       number of sigma to define the pixel grid where the flux estimation has to be done
    xs: float
       x position of the source center
    ys: float
       y position of the source center

    """

    def __init__(self, seeing, dx=0.01, dy=0.01, nsigma=3.0, xs=0.0, ys=0.0):

        self.dx = dx
        self.dy = dy
    
        pixel_LSST = 0.2  # LSSt pixel: 0.2"
        self.pixel_size_x = 1.
        self.pixel_size_y = 1.
        seeing_pixel = seeing/pixel_LSST
        self.sigma = seeing_pixel/2.355
        self.xs = xs
        self.ys = ys
        self.nsigma = nsigma
        nsize = int(nsigma*self.sigma)

        self.xmin = int(-nsigma*self.sigma)-self.pixel_size_x/2.
        self.xmax = int(nsigma*self.sigma)+self.pixel_size_x/2.
        self.ymin = int(-nsigma*self.sigma)-self.pixel_size_y/2.
        self.ymax = int(nsigma*self.sigma)+self.pixel_size_y/2.

        self.npixels_x = int(self.xmax-self.xmin)
        self.npixels_y = int(self.ymax-self.ymin)

        print('Total number of pixels', self.xmin, self.xmax, self.ymin,
              self.ymax, self.npixels_x, self.npixels_y, self.npixels_x*self.npixels_y)

    def fluxDist(self,nproc=1):
        """
        Method to estimate the flux distribution over the pixel.
        This method splits the total pixel area to improve speed processing.
      
        Parameters
        --------------
        nproc: int, opt
          number of procs for multiprocessing

        Returns
        ----------
        pandas df with the following columns: xpixel, ypixel, fluxdist


        """

        npixels_area_x = np.min([self.npixels_x,15])
        npixels_area_y = np.min([self.npixels_y,15])
        deltax_area = (self.xmax-self.xmin)/npixels_area_x 
        deltay_area = (self.ymax-self.ymin)/npixels_area_y

        deltax_area = np.round(deltax_area,0)
        deltay_area = np.round(deltay_area,0)
        
        print(deltax_area,self.xmin,self.xmax)

        # define areas here
        areas = []
        for i in range(npixels_area_x):
            xmin_a = self.xmin+i*deltax_area
            for i in range(npixels_area_y):
                ymin_a = self.ymin+i*deltay_area
                areas.append(dict(zip(['xmin','xmax','ymin','ymax'],[xmin_a,xmin_a+deltax_area,ymin_a, ymin_a+deltay_area])))
                

        # estimate the flux distribution using multiprocessing
        df = self.fluxDist_multiproc(areas,nproc)
     
        return self.format(df)

    def fluxDist_multiproc(self,areas,nproc=1):
        """
        Method to estimate the flux distribution over the pixel.
        This method uses multiprocessing to improve speed processing.

        Parameters
        --------------
        areas : list(dict)
          list of areas to process with parameters given by dict with the following keys: xmin, xmax, ymin, ymax
        nproc: int, opt
          number of procs for multiprocessing

        Returns
        ----------
        pandas df with the following columns: xpixel, ypixel, fluxdist

        """
        
        nareas= len(areas)
        batch = np.linspace(0, nareas, nproc+1, dtype='int')

        #print('batch',batch)
        result_queue = multiprocessing.Queue()
        for i in range(nproc):

            ida = batch[i]
            idb = batch[i+1]
            
            p = multiprocessing.Process(name='Subprocess', target=self.fluxDist_loop, args=(areas[ida:idb], i, result_queue))
            p.start()

        resultdict = {}
        
        for j in range(nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        df = pd.DataFrame()
        for j in range(nproc):
            df = pd.concat((df,resultdict[j]))

        return df

    def fluxDist_loop(self, areas,j=0, output_q=None):
        """
        Method to estimate the flux distribution over the pixel.
        This method loops on areas and estimate fluxes using fluxDist_area
    
        Parameters
        --------------
        areas : list(dict)
          list of areas to process with parameters given by dict with the following keys: xmin, xmax, ymin, ymax
        j: int, opt
           process number for multiptocessing (default: 0)
        output_q: multiprocessing.Queue(), opt
           queue for the multiprocessing (default: None)


        Returns
        ----------
        pandas df with the following columns: xpixel, ypixel, fluxdist


        """

        dfres = pd.DataFrame()
        for val in areas:
            dfres= pd.concat((dfres,self.fluxDist_area(val['xmin'], val['xmax'], val['ymin'],val['ymax'])))

        if output_q is not None:
            output_q.put({j: dfres})
        else:
            return dfres

    def fluxDist_all_at_once(self):
        """
        Old Method to estimate the flux distribution over the pixel.
        This method processes the complete pixel map at once
        
        It is not recommanded to use it when the seeing or/and the number of sigma are large.
        In that case, the required memory for the processing is high and may result to a crash of the program.
        It is advised to use the fluxDist() method instead


        Returns
        ----------
        pandas df with the following columns: xpixel, ypixel, fluxdist

        """

        df = self.fluxDist_area(self.xmin, self.xmax, self.ymin, self.ymax)
      
        return self.format(df)
     
    def format(self, df):
        """
        Method to reformat the df


        Parameters
        --------------
        df: pandas df to reformat

        Returns
        ----------
        reformatted df

        """
        
        df = df.round({'xpixel': 0, 'ypixel': 0})
        df = df.sort_values(by=['xpixel','ypixel'])

        # select only pixels in the initial region
        idx = df['xpixel']>=self.xmin
        idx &= df['xpixel']<=self.xmax
        idx &= df['ypixel']>=self.ymin
        idx &= df['ypixel']<=self.ymax
        return df[idx]
        
    def fluxDist_area(self, xmin, xmax, ymin, ymax):
        """
        Method to estimate the flux distribution on a set of pixels
        on an area defined by (xmin,xmax,ymin, ymax)

        Parameters
        --------------
        xmin: float
          min x value of the area
        xmax: float
          max x value of the area  
        ymin: float
          min y value of the area
        ymax: float
          max y value of the area

        Returns
        ----------
        pandas df with the following columns: xpixel, ypixel, fluxdist

        """
        pixel_size_x = self.pixel_size_x
        pixel_size_y = self.pixel_size_y

        x = np.arange(xmin, xmax+self.dx, self.dx)
        y = np.arange(ymin, ymax+self.dy, self.dy)
        xtile, ytile = np.meshgrid(x, y)

        """
        plt.plot(xtile,ytile,'ko')
        plt.show()
        """
        # get the fluxes over the pixels (per point)
        fluxes = self.PSF(xtile, ytile,
                          self.xs, self.ys, self.sigma)*self.dx*self.dy

        #print('fluxes',fluxes)
        # get the flux distribution per pixel using broadcasting
        xpixel = np.arange(xmin+pixel_size_x/2., xmax +
                           pixel_size_x/2., pixel_size_x)
        ypixel = np.arange(ymin+pixel_size_y/2., ymax +
                           pixel_size_y/2., pixel_size_y)[::-1]

        xpixel_border_inf = np.arange(xmin, xmax, pixel_size_x)
        xpixel_border_sup = np.arange(
            xmin+pixel_size_x, xmax+pixel_size_x, pixel_size_x)

        ypixel_border_inf = np.arange(ymin, ymax, pixel_size_y)[::-1]
        ypixel_border_sup = np.arange(
            ymin+pixel_size_y, ymax+pixel_size_y, pixel_size_y)[::-1]

        """
        print('xpixels', xpixel,
              xpixel_border_inf, xpixel_border_sup)
        print('ypixels', ypixel,
              ypixel_border_inf, ypixel_border_sup)
        """
        xpixel = xpixel.reshape((len(xpixel), 1))
        ypixel = ypixel.reshape((len(ypixel), 1))

        xpixel_border_inf = xpixel_border_inf.reshape(
            (len(xpixel_border_inf), 1))
        xpixel_border_sup = xpixel_border_sup.reshape(
            (len(xpixel_border_sup), 1))
        ypixel_border_inf = ypixel_border_inf.reshape(
            (len(ypixel_border_inf), 1))
        ypixel_border_sup = ypixel_border_sup.reshape(
            (len(ypixel_border_sup), 1))

        #print('hello', xpixel.shape, xpixel_border_inf.shape)
        xtileb = np.tile(xtile, (len(xpixel), len(ypixel), 1, 1))
        ytileb = np.tile(ytile, (len(xpixel), len(ypixel), 1, 1))
        fluxes_b = np.tile(fluxes, (len(xpixel), len(ypixel), 1, 1))

        xdiff = xtileb-xpixel[:, np.newaxis, np.newaxis]
        ydiff = ytileb-ypixel[:, np.newaxis]

        xdiff_inf = xtileb-xpixel_border_inf[:, np.newaxis, np.newaxis]
        xdiff_sup = xtileb-xpixel_border_sup[:, np.newaxis, np.newaxis]
        ydiff_inf = ytileb-ypixel_border_inf[:, np.newaxis]
        ydiff_sup = ytileb-ypixel_border_sup[:, np.newaxis]

        flagx = np.abs(xdiff) <= 0.5
        flagy = np.abs(ydiff) <= 0.5

        flagx_inf = xdiff_inf > 0.
        flagx_sup = xdiff_sup < 0.

        flagy_inf = ydiff_inf >0.
        flagy_sup = ydiff_sup < 0.

        """
        xtilem = np.ma.array(xtileb, mask=~(flagx & flagy))
        ytilem = np.ma.array(ytileb, mask=~(flagx & flagy))
        fluxm = np.ma.array(fluxes_b, mask=~(flagx & flagy))
        """

        flag = flagx_inf & flagx_sup & flagy_inf & flagy_sup
        xtilem = np.ma.array(xtileb, mask=~flag)
        ytilem = np.ma.array(ytileb, mask=~flag)
        fluxm = np.ma.array(fluxes_b, mask=~flag)

        #print('xtilem', xtilem)
        xp = np.mean(np.ma.mean(xtilem, axis=3), axis=2).flatten()
        yp = np.mean(np.ma.mean(ytilem, axis=3), axis=2).flatten()
        fluxdist = np.sum(np.sum(fluxm, axis=3), axis=2).flatten()

        """
        print('mean x', np.mean(np.ma.mean(xtilem, axis=3), axis=2).flatten())
        print('mean y', np.mean(np.ma.mean(ytilem, axis=3), axis=2).flatten())

        print('fluxes', np.sum(np.sum(fluxm, axis=3), axis=2).flatten())
        """
        return pd.DataFrame({'xpixel': xp, 'ypixel': yp, 'fluxdist': fluxdist})

    def PSF(self, x, y, xs, ys, sigma):
        """
        Method to estimate a single gaussian PSF

        Parameters
        --------------
        x: float
          x-coordinate where the PSF has to be estimated
        y: float
          y-coordinate where the PSF has to be estimated
        xs: float
          x-center of the source
        ys: float
          y-center of the source
        sigma: float
          sigma of the gaussian

        Returns
        ---------
        flux in the pixel (float)
        """
        # sigma = self.seeing_pixel/2.355
        val = (x-xs)**2+(y-ys)**2

        func = np.exp(-val/2./sigma**2)
        func /= (2.*np.pi*sigma**2)
        # func /= (2.*np.pi)
        # func /= (sigma*np.sqrt(2.*np.pi))
        return func
