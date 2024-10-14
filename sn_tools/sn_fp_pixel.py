#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:25:51 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import numpy as np
import pandas as pd
from sn_tools.sn_obs import proj_gnomonic_plane
import healpy as hp


class FocalPlane:
    def __init__(self, nx=dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 8*15])),
                 ny=dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 2*15])),
                 FoV=9.62,
                 level='raft',
                 raft_sub=dict(
                     zip(['to_remove'], [['1_1', '1_5', '5_1', '5_5']])),
                 ccd_sub=dict(zip(['to_remove', 'guide', 'sensor'],
                                  [['1_1', '1_2', '1_3', '2_1', '2_2', '3_1',
                                    '1_13', '1_14', '1_15',
                                    '2_14', '2_15', '3_15',
                                    '15_1', '15_2', '15_3', '14_1',
                                    '14_2', '13_1',
                                    '15_13', '15_14', '15_15', '14_14',
                                    '14_15', '13_15'],
                              ['2_3', '3_2', '2_13', '3_14',
                                      '14_3', '13_2', '14_13', '13_14'],
                              ['3_3', '3_13', '13_3', '13_13']]))):
        """
        Focal Plane class

        Parameters
        ----------
        nx : dict, optional
            x-axis segmentation (level dep.).
            The default is dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 8*15])).
        ny : dict, optional
            y-axis segmentation (level dependent).
            The default is dict(zip(['raft', 'ccd', 'sensor'], [5, 15, 2*15])).
        FoV : float, optional
            Field of view. The default is 9.62.
        level : str, optional
            segmentation level (raft,ccd,sensor). The default is 'raft'.
        raft_sub: dict, optional
            list of rafts to remove
            the default is dict(zip(['to_remove'],['1_1','1_4','5,1','5_5']))
        ccd_sub :dict, optional
            list of ccds to remove.
            The default is dict(zip(['to_remove', 'guide', 'sensor'],
                            [['1_1', '1_2', '1_3', '2_1', '2_2', '3_1',
                            '1_13', '1_14', '1_15', '2_14', '2_15', '3_15',
                            '15_1', '15_2', '15_3', '14_1', '14_2', '13_1',
                            '15_13', '15_14', '15_15', '14_14', '14_15', 
                            '13_15'],
                            ['2_3', '3_2', '2_13', '3_14',
                             '14_3', '13_2', '14_13', '13_14'],
                            ['3_3', '3_13', '13_3', '13_13']])).

        Returns
        -------
        None.

        """

        fov_str = FoV*(np.pi/180.)**2  # LSST fov in sr
        theta = 2.*np.arcsin(np.sqrt(fov_str/(4.*np.pi)))
        fpscale = np.tan(theta)

        self.xmin, self.xmax = -fpscale, fpscale
        self.ymin, self.ymax = -fpscale, fpscale

        self.dx = self.xmax-self.xmin
        self.dy = self.ymax-self.ymin

        self.level = level
        self.nx = nx
        self.ny = ny
        self.index_level = ['raft', 'ccd', 'sensor']
        self.ccd_sub = ccd_sub

        self.fp = self.buildIt()

        if self.level == 'raft':
            self.remove_cells('raft', raft_sub)

        if self.level == 'ccd' or self.level == 'sensor':
            self.remove_cells('ccd', ccd_sub)

        self.ccols = ['healpixID', 'pixRA', 'pixDec',
                      'observationId', 'raft']
        if level == 'ccd':
            self.ccols.append('ccd')
        if level == 'sensor':
            self.ccols.append('ccd')
            self.ccols.append('sensor')

    def remove_cells(self, colName, cell_dict):
        """
        Method to remove cells

        Parameters
        ----------
        cell_dict : dict
            List of cells to remove.

        Returns
        -------
        None.

        """

        for key, vals in cell_dict.items():
            self.remove_cell(colName, vals)

    def set_display_mode(self):
        """
        add cols required in display_mode

        Returns
        -------
        None.

        """

        self.ccols += ['xpixel', 'ypixel', 'xmin', 'xmax', 'ymin', 'ymax']

    def buildIt(self):
        """
        Build the FP here

        Returns
        -------
        pandas df
            resulting FP.

        """

        d_elem_x = self.dx/self.nx[self.level]
        d_elem_y = self.dy/self.ny[self.level]

        xvalues = np.arange(self.xmin, self.xmax, d_elem_x)
        yvalues = np.arange(self.ymin, self.ymax, d_elem_y)
        xv, yv = np.meshgrid(xvalues, yvalues)

        df_fp = pd.DataFrame(xv.flatten(), columns=['x'])
        df_fp['y'] = yv.flatten()
        df_fp['xc'] = df_fp['x']+0.5*d_elem_x
        df_fp['yc'] = df_fp['y']+0.5*d_elem_y
        df_fp['xmin'] = df_fp['xc']-0.5*d_elem_x
        df_fp['xmax'] = df_fp['xc']+0.5*d_elem_x
        df_fp['ymin'] = df_fp['yc']-0.5*d_elem_y
        df_fp['ymax'] = df_fp['yc']+0.5*d_elem_y

        # get index here

        idx_level = self.index_level.index(self.level)
        for i in range(idx_level+1):
            vv = self.index_level[i]
            d_elem_xx = self.dx/self.nx[vv]
            d_elem_yy = self.dy/self.ny[vv]
            df_fp = self.get_index(df_fp, d_elem_xx, d_elem_yy, vv)

        # remove extra cells
        idx = df_fp['xc'] > self.xmin
        idx &= df_fp['xc'] < self.xmax
        idx &= df_fp['yc'] > self.ymin
        idx &= df_fp['yc'] < self.ymax

        return pd.DataFrame(df_fp[idx])

    def get_index(self, df_fp, d_elem_x, d_elem_y, level):
        """
        Method to estimate cells index from position

        Parameters
        ----------
        df_fp : pandas df
            Data to process.
        d_elem_x : float
            x-axis cell size.
        d_elem_y : float
            y-axis cell size.
        level : str
            segmentation level.

        Returns
        -------
        df_fp : pandas df
            original df+index.

        """

        ipos = (df_fp['xc']-self.xmin)/d_elem_x+1
        jpos = (self.ymax-df_fp['yc'])/d_elem_y+1
        lpj = '{}_j'.format(level)
        lpi = '{}_i'.format(level)
        df_fp[lpj] = ipos
        df_fp[lpi] = jpos
        df_fp[lpj] = df_fp[lpj].astype(int)
        df_fp[lpi] = df_fp[lpi].astype(int)
        df_fp[level] = df_fp[lpi].astype(str)+'_'+df_fp[lpj].astype(str)
        df_fp = df_fp.drop(columns=[lpi, lpj])

        return df_fp

    def check_fp(self, top_level='raft', low_level='ccd'):
        """
        Method to estimate some infos on the focal plane

        Parameters
        ----------
        top_level : str, optional
            top-level to get infos. The default is 'raft'.
        low_level : str, optional
            lower level for infos. The default is 'ccd'.

        Returns
        -------
        None.

        """

        # posi = '{}_i'.format(top_level)
        # posj = '{}_j'.format(top_level)

        ijpos = self.fp[top_level].unique()

        print('Number of {}s'.format(top_level), len(ijpos))
        if low_level != top_level:
            for vv in ijpos:
                idx = self.fp[top_level] == vv
                sel = self.fp[idx]
                idxb = sel[low_level].unique()
                print('{}'.format(top_level), vv,
                      'N {}s'.format(low_level), len(idxb))

    def remove_cell(self, colName, celllist):
        """
        Method to remove cells from fp

        Parameters
        ----------
        colName : str
            cell type (raft, ccd).
        celllist : list(str)
            List of cells to remove.

        Returns
        -------
        None.

        """

        idx = self.fp[colName].isin(celllist)

        self.fp = self.fp[~idx]

    def plot_fp(self):
        """
        Method to plot the focal plane

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.fp['x'], self.fp['y'], 'k.')
        ax.plot(self.fp['xc'], self.fp['yc'], 'r*')

        ax.plot(self.fp['xmin'], self.fp['ymin'], color='b',
                marker='s', mfc='None', linestyle='None')
        ax.plot(self.fp['xmin'], self.fp['ymax'], color='b',
                marker='s', mfc='None', linestyle='None')
        ax.plot(self.fp['xmax'], self.fp['ymin'], color='b',
                marker='s', mfc='None', linestyle='None')
        ax.plot(self.fp['xmax'], self.fp['ymax'], color='b',
                marker='s', mfc='None', linestyle='None')

        plt.show()

    def plot_fp_pixels(self, pixels=None, signal=None):
        """
        Method to plot the FP pixels and pixels inside

        Parameters
        ----------
        pixels : pandas df, optional
            Pixel coordinates. The default is None.
        signal : pandas df, optional
            FP pixels with pixels (healpiX) inside. The default is None.

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))

        # draw the focal plane

        for i, row in self.fp.iterrows():

            rect = self.get_rect(row)
            ax.add_patch(rect)

        xmin = self.fp['xmin'].min()
        xmax = self.fp['xmax'].max()
        ymin = self.fp['ymin'].min()
        ymax = self.fp['ymax'].max()

        k = 1.5
        ax.set_xlim([k*xmin, k*xmax])
        ax.set_ylim([k*ymin, k*ymax])

        if pixels is not None:
            ax.plot(pixels['xpixel'], pixels['ypixel'], 'r.')

        if signal is not None:
            for i, row in signal.iterrows():
                rect = self.get_rect(row, fill=True)
                ax.add_patch(rect)

        plt.show()

    def get_rect(self, row, fill=False):
        """
        get matplotlib rectangle

        Parameters
        ----------
        row : array
            loc array.
        fill : bool, optional
            To fill the rectangle or not. The default is False.

        Returns
        -------
        rect : Rectangle from matplotlib.patches
            Rectangle result.

        """

        from matplotlib.patches import Rectangle
        xy = (row['xmin'], row['ymin'])
        height = row['ymax']-row['ymin']
        width = row['xmax']-row['xmin']
        rect = Rectangle(xy, width, height, fill=fill, alpha=0.5)

        return rect

    def pix_to_obs(self, df_pix):
        """
        Method to select pixels inside the LSST FP

        Parameters
        ----------
        df_pix : pandas df
            List of pixels to consider.

        Returns
        -------
        pandas df
            Selected pixels/obs.

        """

        # make super df

        df_super = self.fp.merge(df_pix, how='cross')
        # select pixels inside FP
        idx = df_super['xpixel'] >= df_super['xmin']
        idx &= df_super['xpixel'] <= df_super['xmax']
        idx &= df_super['ypixel'] >= df_super['ymin']
        idx &= df_super['ypixel'] <= df_super['ymax']

        res = pd.DataFrame(df_super[idx])
        res['healpixID'] = res['healpixID'].astype(int)
        # delete df_super
        return res[self.ccols]


def get_pixels_in_window(nside, RA_min, RA_max, Dec_min, Dec_max):
    """
    Method to grab pixels in window defined by 
    (RA_min, RA_max, Dec_min, Dec_max)

    Parameters
    ----------
    nside : int
        healpix nside parameter.
    RA_min : float
        Min RA.
    RA_max : float
        Max RA.
    Dec_min : float
        Min Dec.
    Dec_max : float
        Max Dec.

    Returns
    -------
    df : pandas df
        Output data (healpixID, pixRA,pixDec).

    """

    import astropy
    import healpy as hp
    ra_poly = np.array([RA_min, RA_max, RA_max, RA_min])
    dec_poly = np.array([Dec_min, Dec_min, Dec_max, Dec_max])
    xyzpoly = astropy.coordinates.spherical_to_cartesian(
        1, np.deg2rad(dec_poly), np.deg2rad(ra_poly))

    healpixIDs = hp.query_polygon(
        nside, np.array(xyzpoly).T, nest=True, inclusive=True).tolist()

    # get pixel coordinates
    coords = hp.pix2ang(nside, healpixIDs, nest=True, lonlat=True)
    pixRA, pixDec = coords[0], coords[1]

    df = pd.DataFrame(healpixIDs, columns=['healpixID'])
    df['pixRA'] = pixRA
    df['pixDec'] = pixDec

    idx = df['pixRA'] >= RA_min
    idx = df['pixRA'] < RA_max

    return df[idx]


def get_window(data, RACol='fieldRA', DecCol='fieldDec',
               radius=np.sqrt(12./3.14)):
    """
    get (RA, Dec) window from a central FP 

    Parameters
    ----------
    data : pandas df
        data (observations) to process.
    RACol : str, optional
        RA colname. The default is 'fieldRA'.
    DecCol : str, optional
        Dec colname. The default is 'fieldDec'.
    radius : float, optional
        Radius of the window. The default is np.sqrt(12./3.14).

    Returns
    -------
    RA_min : float
        RA min.
    RA_max : float
        RA max.
    Dec_min : float
        Dec min.
    Dec_max : float
        Dec max.

    """

    RA_mean = data[RACol].mean()
    Dec_mean = data[DecCol].mean()

    RA_min = RA_mean-radius
    RA_max = RA_mean+radius
    Dec_min = Dec_mean-radius
    Dec_max = Dec_mean+radius

    return RA_min, RA_max, Dec_min, Dec_max


def get_xy_pixels(pointings, healpixID, pixRA, pixDec, nside=64,
                  RACol='fieldRA', DecCol='fieldDec', filterCol='filter'):
    """
    Grab gnomonic projection of pixels around(RA,Dec)

    Parameters
    ----------
    RA : float
        RA value.
    Dec : float
        Dec value.
    nside: int, opt.
        nside healpix value. The default is 64.
    RACol : str, optional
        RA colname. The default is 'fieldRA'.
    DecCol : str, optional
        Dec colname. The default is 'DecCol'.

    Returns
    -------
    x : float
        x-axis values.
    y : float
        y-axis values.

    """

    # print(pixRA, pixDec)
    pixRA_rad = np.deg2rad(pixRA)
    pixDec_rad = np.deg2rad(pixDec)
    # convert data position in rad
    # pRA = np.median(sel_data['RA'])
    # pDec = np.median(sel_data['Dec'])
    RA = pointings[RACol].tolist()
    Dec = pointings[DecCol].tolist()

    pRA_rad = np.deg2rad(RA)
    pDec_rad = np.deg2rad(Dec)

    # gnomonic projection of pixels on the focal plane
    x, y = proj_gnomonic_plane(pRA_rad, pDec_rad, pixRA_rad, pixDec_rad)

    df = pd.DataFrame(x, columns=['xpixel_norot'])
    # df['xpixel_norot'] = x
    df['ypixel_norot'] = y
    df['healpixID'] = healpixID
    df['pixRA'] = pixRA
    df['pixDec'] = pixDec
    ccols = ['observationId', filterCol, 'rotSkyPos']
    ccols += [RACol, DecCol]
    for var in ccols:
        df[var] = pointings[var]

    # pixel rotation here
    df['rotSkyPixel'] = -np.deg2rad(df['rotSkyPos'])
    # df['rotSkyPixel'] = 0.
    df['xpixel'] = np.cos(df['rotSkyPixel'])*df['xpixel_norot']
    df['xpixel'] -= np.sin(df['rotSkyPixel'])*df['ypixel_norot']
    df['ypixel'] = np.sin(df['rotSkyPixel'])*df['xpixel_norot']
    df['ypixel'] += np.cos(df['rotSkyPixel'])*df['ypixel_norot']

    return df


def get_pixels(RA, Dec, nside=64, widthRA=5.):
    """
    grab pixels around (RA,Dec) (window width: widthRA)

    Parameters
    ----------
    RA : float
        RA value.
    Dec : float
        Dec value.
    nside : int, optional
        nside healpix parameter. The default is 64.
    widthRA : float, optional
        window width. The default is 5..

    Returns
    -------
    healpixIDs : int
        healpixIDs.
    float
        pixRA.
    float
        pixDec.

    """

    healpixID = hp.ang2pix(nside, RA, Dec, nest=True, lonlat=True)
    vec = hp.pix2vec(nside, healpixID, nest=True)
    healpixIDs = hp.query_disc(nside, vec, np.deg2rad(widthRA),
                               inclusive=True, nest=True)

    # get pixel coordinates
    coords = hp.pix2ang(nside, healpixIDs, nest=True, lonlat=True)

    return healpixIDs, coords[0], coords[1]


def get_xy(RA, Dec, nside=64):
    """
    Grab gnomonic projection of pixels around(RA,Dec)

    Parameters
    ----------
    RA : float
        RA value.
    Dec : float
        Dec value.
    nside: int, opt.
        nside healpix value. The default is 64.

    Returns
    -------
    x : float
        x-axis values.
    y : float
        y-axis values.

    """
    healpixID, pixRA, pixDec = get_pixels(RA, Dec, nside=nside)

    # print(pixRA, pixDec)
    pixRA_rad = np.deg2rad(pixRA)
    pixDec_rad = np.deg2rad(pixDec)
    # convert data position in rad
    # pRA = np.median(sel_data['RA'])
    # pDec = np.median(sel_data['Dec'])
    pRA_rad = np.deg2rad(RA)
    pDec_rad = np.deg2rad(Dec)

    # gnomonic projection of pixels on the focal plane
    x, y = proj_gnomonic_plane(pRA_rad, pDec_rad, pixRA_rad, pixDec_rad)

    df = pd.DataFrame(x, columns=['xpixel_norot'])
    # df['xpixel_norot'] = x
    df['ypixel_norot'] = y
    df['healpixID'] = healpixID
    df['pixRA'] = pixRA
    df['pixDec'] = pixDec

    return df


def get_proj_data(sel_data, nside=64, RACol='fieldRA',
                  DecCol='fieldDec', filterCol='filter'):
    """
    Function to get gnomonic projection of a pixel corresponding
    to a set of pointings.

    Parameters
    ----------
    sel_data : pandas df
        Data to process.
    nside: int, opt.
        healpix nside parameter. The default is 64.

    Returns
    -------
    df_pix : pandas df
        projected pixels.

    """

    df_pix = pd.DataFrame()
    for i, vv in sel_data.iterrows():
        dd = get_xy(vv[RACol], vv[DecCol], nside=nside)
        """
        dd = pd.DataFrame(x, columns=['xpixel_norot'])
        dd['ypixel_norot'] = y
        """
        ccols = ['observationId', filterCol, 'rotSkyPos']
        ccols += [RACol, DecCol]
        for var in ccols:
            dd[var] = vv[var]
        df_pix = pd.concat((df_pix, dd))

    # pixel rotation here
    df_pix['rotSkyPixel'] = -np.deg2rad(df_pix['rotSkyPos'])
    # df_pix['rotSkyPixel'] = 0.
    df_pix['xpixel'] = np.cos(df_pix['rotSkyPixel'])*df_pix['xpixel_norot']
    df_pix['xpixel'] -= np.sin(df_pix['rotSkyPixel'])*df_pix['ypixel_norot']
    df_pix['ypixel'] = np.sin(df_pix['rotSkyPixel'])*df_pix['xpixel_norot']
    df_pix['ypixel'] += np.cos(df_pix['rotSkyPixel'])*df_pix['ypixel_norot']

    return df_pix


def get_data_window(pixRA, pixDec,
                    data,
                    RACol='fieldRA', DecCol='fieldDec',
                    radius=np.sqrt(12/3.14156)):
    """
    Method to get data inside a window

    Parameters
    ----------
    pixRA : float
        RA mean window.
    pixDec : float
        Dec mean window.
    data : array
        data to process.
    RACol : str, optional
        RA col name. The default is 'fieldRA'.
    DecCol : str, optional
        Dec colname. The default is 'fieldDec'.
    radius : float, optional
        width of the window. The default is np.sqrt(12/3.14156).

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    RA_min = pixRA-radius
    RA_max = pixRA+radius
    Dec_min = pixDec-radius
    Dec_max = pixDec+radius

    idx = data[RACol] >= RA_min
    idx &= data[RACol] <= RA_max
    idx &= data[DecCol] >= Dec_min
    idx &= data[DecCol] <= Dec_max

    return data[idx]
