#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:04:16 2026

@author: philippe.gris@clarmont.in2p3.fr
"""
from astropy.table import Table
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import numpy as np

__all__=['RegularGrid_interp']

class RegularGrid_interp:
    def __init__(self,df_tot,ccols,zcol='distmod'):
        """
        class to build a regulargrid interpolator

        Parameters
        ----------
        df_tot : pandas df
            Data to use to build the interpolator.
        ccols : list(str)
            List of columns for the interpolator.
        zcol : str, optional
            interpolator resuls. The default is 'distmod'.

        Returns
        -------
        interp : RegularGridInterpolator
            the interpolator.

        """
    
        self.data = Table.from_pandas(df_tot)
        self.ccols = ccols
        self.zcol = zcol
    
    def __call__(self):
        """
        Main method: build the interpolator

        Returns
        -------
        interp : RegularGrid Interpolator
            the result.

        """   
        ccols = self.ccols
       
        
        tab = self.data
        dlims = {}
        for vv in ccols:
            # Fluxes and errors
            xmin, xmax, xstep, nx = self.limVals(tab, vv)
            xstep = np.round(xstep,2)
            dlims[vv] = (xmin, xmax, xstep, nx)
    
        dlinsp = {}
        
        for vv in ccols:
            dlinsp[vv]=np.linspace(dlims[vv][0], dlims[vv][1], dlims[vv][3])
            
        tup = []
        ntup = []
        vtup = []
        for vv in ccols:
            tup += [tab[ccols]]
            ntup += [dlims[vv][3]]
            vtup += [dlinsp[vv]]
            
        tup = tuple(tup)
        ntup = tuple(ntup)
        vtup = tuple(vtup)
        
        index = np.lexsort(tup)
        distmod = np.reshape(tab[index][self.zcol], ntup)
        
        interp = RegularGridInterpolator(vtup,distmod,method='nearest', 
                                         bounds_error=False, fill_value=-1.0)
            
        return interp


    def limVals(self,lc, field):
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
        # vals = np.unique(lc[field].data.round(decimals=4))
        vals = np.unique(lc[field].data)
        vmin = np.min(vals)
        vmax = np.max(vals)
        vstep = np.median(vals[1:]-vals[:-1])
    
        # make a check here
        test = list(np.round(np.arange(vmin, vmax+vstep, vstep), 2))
        if len(test) != len(vals):
            print('problem here with ', field)
            print('missing value', set(test).difference(set(vals)))
            print('Interpolation results may not be accurate!!!!!')
    
        return vmin, vmax, vstep, len(vals)