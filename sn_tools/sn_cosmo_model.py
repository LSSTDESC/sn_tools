#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:50:59 2026

@author: jade.maisonneuve@clermont.in2p3.fr
         philippe.gris@clermont.in2p3.fr
"""
import numpy as np
import astropy.units as u
from astropy.cosmology import FLRW
from scipy.integrate import quad

class DDE_FLRW(FLRW):
    def __init__(self, H0, Om0, Ode0=None, 
                 model="CPL", 
                 de_params=dict(zip(["w0","wa"],[-1,0])), 
                 de_eos=None):
       
        Ode0 = Ode0 if Ode0 is not None else 1.0 - Om0
        # Call parent FLRW constructor
        super().__init__(H0=H0, Om0=Om0, Ode0=Ode0)
        # Store model info
        self.model = model
        self.de_params = de_params
        self.de_eos = de_eos
        # Check parameters depending on the model
        if model == "CPL":
            if "w0" not in self.params or "wa" not in self.params:
                raise ValueError("CPL requires w0 and wa")
        elif model == "free":
            if self.de_eos is None:
                raise ValueError("Free model requires DE equation-of-state")
        else:
            raise ValueError("Model must be CPL or free")

    # Dark energy equation of state w(z)
    def w(self, z):
        """
        Method to estimate w(z)

        Parameters
        ----------
        z : float array
            Redshifts.

        Returns
        -------
        array(float)
            w(z).

        """
        a = 1.0 / (1.0 + z) # Scale factor
        if self.model == "CPL":
            # CPL formula: w(z) = w0 + wa*(1-a)
            return self.de_params["w0"] + self.de_params["wa"] * (1.0 - a)
        elif self.model == "free":
            # Evaluate a custom expression for w(z)
            return eval(self.de_eos, {"np": np, "z": z, "a": a}, self.de_params)

    # Dark energy density evolution
    def de_density_scale(self, z):
        """
        Method to estimate DE density scale.

        Parameters
        ----------
        z : float array
            Redshifts.

        Returns
        -------
        array(float)
            DE energy scale

        """

        if self.model == "CPL":
            # Analytic expression for CPL
            w0 = self.params["w0"]
            wa = self.params["wa"]
            return (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))

        elif self.model == "free":
            # Numerical integration for free model
            def integrand(x):
                return (1+self.w(x))/(1+x)

            if np.isscalar(z):
                return np.exp(3 * quad(integrand, 0, z)[0])
            else:
                return np.array([np.exp(3 * quad(integrand, 0, zi)[0]) for zi in z])

    # E(z) = H(z)/H0
    def efunc(self, z):
        """
        Returns E(z) = H(z)/H0
        Astropy uses this function for all distance calculations.
        
        Parameters
        ----------
        z : float or array-like
            Redshift
            
        Returns
        -------
        float or np.ndarray
            E(z) = H(z)/H0
        """
        #print("custom  efunc called") 
        return np.sqrt(
            self.Om0*(1+z)**3                    # Matter contribution
            + self.Ode0*self.de_density_scale(z) # Dark energy contribution
            + self.Ok0*(1+z)**2    # Curvature contribution
        )
    # Inverse of E(z)
    def inv_efunc(self, z):
        """
        Inverse of efunc 

        Parameters
        ----------
        z : float array
            Redshifts.

        Returns
        -------
        float array
            1/efunc(z)

        """
        return 1.0 / self.efunc(z)
    
    def q_parameter(self, z):
        """
        Method to estimate the decelaration parameter

        Parameters
        ----------
        z : float array
            Redshifts.

        Returns
        -------
        float array
            Deceleration parameter.

        """
        dz = 1e-5   # Step for numerical derivative
        if np.isscalar(z):
            Hz = self.H(z)
            dHdz = (self.H(z+dz)-Hz)/dz
            return (1+z)/Hz*dHdz - 1
        else:
            # Handle array of z
            return np.array([(1+zi)/self.H(zi)*(self.H(zi+dz)-self.H(zi))/dz - 1 for zi in z])
