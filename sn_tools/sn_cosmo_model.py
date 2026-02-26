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
from astropy.cosmology.parameter import Parameter
from astropy.cosmology._src.utils import aszarr
from astropy.cosmology._src.flrw import scalar_inv_efuncs

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

class w0waDDE(FLRW):
    r"""FLRW cosmology with a CPL dark energy equation of state and curvature.

    The equation for the dark energy equation of state uses the
    CPL form as described in Chevallier & Polarski [1]_ and Linder [2]_:
    :math:`w(z) = w_0 + w_a (1-a) = w_0 + w_a z / (1+z)`.

    Parameters
    ----------
    H0 : float or scalar quantity-like ['frequency']
        Hubble constant at z = 0. If a float, must be in [km/sec/Mpc].

    Om0 : float
        Omega matter: density of non-relativistic matter in units of the
        critical density at z=0.

    Ode0 : float
        Omega dark energy: density of dark energy in units of the critical
        density at z=0.

    w0 : float, optional
        Dark energy equation of state at z=0 (a=1). This is pressure/density
        for dark energy in units where c=1.

    wa : float, optional
        Negative derivative of the dark energy equation of state with respect
        to the scale factor. A cosmological constant has w0=-1.0 and wa=0.0.

    Tcmb0 : float or scalar quantity-like ['temperature'], optional
        Temperature of the CMB z=0. If a float, must be in [K]. Default: 0 [K].
        Setting this to zero will turn off both photons and neutrinos
        (even massive ones).

    Neff : float, optional
        Effective number of Neutrino species. Default 3.04.

    m_nu : quantity-like ['energy', 'mass'] or array-like, optional
        Mass of each neutrino species in [eV] (mass-energy equivalency enabled).
        If this is a scalar Quantity, then all neutrino species are assumed to
        have that mass. Otherwise, the mass of each species. The actual number
        of neutrino species (and hence the number of elements of m_nu if it is
        not scalar) must be the floor of Neff. Typically this means you should
        provide three neutrino masses unless you are considering something like
        a sterile neutrino.

    Ob0 : float or None, optional
        Omega baryons: density of baryonic matter in units of the critical
        density at z=0.  If this is set to None (the default), any computation
        that requires its value will raise an exception.

    name : str or None (optional, keyword-only)
        Name for this cosmological object.

    meta : mapping or None (optional, keyword-only)
        Metadata for the cosmology, e.g., a reference.
        
    model: int
           DE model (1=w0wa, 2=other). The default is 1

    Examples
    --------
    >>> from astropy.cosmology import w0waCDM
    >>> cosmo = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-0.9, wa=0.2)

    The comoving distance in Mpc at redshift z:

    >>> z = 0.5
    >>> dc = cosmo.comoving_distance(z)

    References
    ----------
    .. [1] Chevallier, M., & Polarski, D. (2001). Accelerating Universes with
           Scaling Dark Matter. International Journal of Modern Physics D,
           10(2), 213-223.
    .. [2] Linder, E. (2003). Exploring the Expansion History of the
           Universe. Phys. Rev. Lett., 90, 091301.
    """

    w0 = Parameter(doc="Dark energy equation of state at z=0.",
                   fvalidate="float")
    wa = Parameter(
        doc="Negative derivative of dark energy equation of state w.r.t. a.",
        fvalidate="float",
    )
    model = Parameter(doc="DDE model", fvalidate="float")

    def __init__(
        self,
        H0,
        Om0,
        Ode0,
        w0=-1.0,
        wa=0.0,
        model=1,
        Tcmb0=0.0 * u.K,
        Neff=3.04,
        m_nu=0.0 * u.eV,
        Ob0=None,
        *,
        name=None,
        meta=None
    ):
        super().__init__(
            H0=H0,
            Om0=Om0,
            Ode0=Ode0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu=m_nu,
            Ob0=Ob0,
            name=name,
            meta=meta,
        )
        self.w0 = w0
        self.wa = wa
        self.model = model

        # Please see :ref:`astropy-cosmology-fast-integrals` for discussion
        # about what is being done here.
        if self.Tcmb0.value == 0:
            self._inv_efunc_scalar = scalar_inv_efuncs.w0wacdm_inv_efunc_norel
            self._inv_efunc_scalar_args = (
                self.Om0,
                self.Ode0,
                self.Ok0,
                self.w0,
                self.wa,
            )
        elif not self._massivenu:
            self._inv_efunc_scalar = scalar_inv_efuncs.w0wacdm_inv_efunc_nomnu
            self._inv_efunc_scalar_args = (
                self.Om0,
                self.Ode0,
                self.Ok0,
                self.Ogamma0 + self._Onu0,
                self.w0,
                self.wa,
            )
        else:
            self._inv_efunc_scalar = scalar_inv_efuncs.w0wacdm_inv_efunc
            self._inv_efunc_scalar_args = (
                self.Om0,
                self.Ode0,
                self.Ok0,
                self.Ogamma0,
                self.neff_per_nu,
                self.nmasslessnu,
                self.nu_y_list,
                self.w0,
                self.wa,
            )

    def w(self, z):
        r"""Returns dark energy equation of state at redshift ``z``.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like, or `~numbers.Number`
            Input redshift.

        Returns
        -------
        w : ndarray or float
            The dark energy equation of state
            Returns `float` if the input is scalar.

        Notes
        -----
        The dark energy equation of state is defined as
        :math:`w(z) = P(z)/\rho(z)`, where :math:`P(z)` is the pressure at
        redshift z and :math:`\rho(z)` is the density at redshift z, both in
        units where c=1. Here this is
        :math:`w(z) = w_0 + w_a (1 - a) = w_0 + w_a \frac{z}{1+z}`.
        """
        z = aszarr(z)

        if self.model == 1:
            res = self.w0 + self.wa * z / (z + 1.0)

        if self.model == 2:
            res = -1. + self.w0*np.sin(self.wa*z)/(1.+z**2)

        return res

    def de_density_scale(self, z):
        r"""Evaluates the redshift dependence of the dark energy density.

        Parameters
        ----------
        z : Quantity-like ['redshift'], array-like, or `~numbers.Number`
            Input redshift.

        Returns
        -------
        I : ndarray or float
            The scaling of the energy density of dark energy with redshift.
            Returns `float` if the input is scalar.

        Notes
        -----
        The scaling factor, I, is defined by :math:`\rho(z) = \rho_0 I`,
        and in this case is given by

        .. math::

           I = \left(1 + z\right)^{3 \left(1 + w_0 + w_a\right)}
                     \exp \left(-3 w_a \frac{z}{1+z}\right)
        """
        z = aszarr(z)
        zp1 = z + 1.0  # (converts z [unit] -> z [dimensionless])

        if self.model == 1:
            res = zp1 ** (3 * (1 + self._w0 + self._wa)) * \
                np.exp(-3 * self._wa * z / zp1)
            return res
        else:
            return self.de_density_scale_int(z)

    def de_density_scale_int(self, z):
        """
        Method to estimate DE density from integral

        Parameters
        ----------
        z : numpy array
            redshift values.

        Returns
        -------
        array
            density scale vs z.

        """

        a = aszarr(z)

        r = []

        for zz in a:
            rr = self.de_density_scale_z(zz)
            r.append(rr)

        return np.asarray(r)

    def de_density_scale_z(self, z):
        """
        Method to estimate de density scale using integ

        Parameters
        ----------
        z : float
            redshift value.

        Returns
        -------
        float
            DE density.

        """

        res = quad(self.integrand, 0, z)[0]

        return np.exp(3.*res)

    def integrand(self, x):
        """
        Integrand for DE

        Parameters
        ----------
        x : float
            var to integrate.

        Returns
        -------
        float
            integrant dor DE density.

        """

        return (1.+self.w(x))/(1.+x)
