import numpy as np
from sn_tools.sn_rate import SN_Rate
import os
import numpy.lib.recfunctions as rf
from astropy.table import Table


class GenerateSample:
    """ Generates a sample of parameters for simulation

    Parameters
    -----------
    sn_parameters : dict
      supernovae parameters: x1, color, z, daymax, ...
    cosmo_parameters: dict
      cosmology parameters: H0, Om0
    mjdCol : str,opt
       name of the column corresponding to MJD
       Default : 'mjd'
    seasonCol : str, opt
      name of the column corresponding to season
      Default : 'season'
    filterCol : str, opt
      name of the column corresponding to filter
      Default : 'filter'
    min_rf_phase : float, opt
       min rest-frame phase for supernovae
       Default : -15.
    max_rf_phase : float, opt
       max rest-frame phase for supernovae
       Default : 30.
    area : float, opt
       area of the survey (in deg\^2)
       Default : 9.6 deg\^2

    """

    def __init__(self, sn_parameters, cosmo_parameters, mjdCol='mjd', seasonCol='season', filterCol='filter', min_rf_phase=-15., max_rf_phase=30., area=9.6, dirFiles='reference_files'):
        self.dirFiles = dirFiles
        self.params = sn_parameters
        self.sn_rate = SN_Rate(rate=self.params['z']['rate'],
                               H0=cosmo_parameters['H0'],
                               Om0=cosmo_parameters['Omega_m'])

        self.x1_color = self.getDist(self.params['x1_color']['rate'])
        self.mjdCol = mjdCol
        self.seasonCol = seasonCol
        self.filterCol = filterCol
        self.area = area
        self.min_rf_phase = min_rf_phase
        self.max_rf_phase = max_rf_phase

    def __call__(self, obs):
        """
        Compute set of parameters for simulation

        Parameters
        ---------
        array of observations

        Returns
        ---------
        array of sn parameters for simulation:
        z, float
          redshift
        x1, float
           x1 parameter
        color, float,
          color parameter
        daymax, float
          T0 parameter
        epsilon_x0,float
          epsilon for x0 parameter
        epsilon_x1, float
          epsilon for x1 parameter
        epsilon_color, float
          epsilon for color parameter
        min_rf_phase, float
          min rest-frame phase for LC points
        max_rf_phase, float
          max rest-frame phase for LC points
        """
        epsilon = 1.e-08

        r = []
        for season in np.unique(obs[self.seasonCol]):
            idx = (obs[self.seasonCol] == season) & (
                obs[self.filterCol] != 'u')
            sel_obs = obs[idx]
            # get duration of obs
            daymin = np.min(sel_obs[self.mjdCol])
            daymax = np.max(sel_obs[self.mjdCol])
            duration = daymax-daymin

            rp = self.getParameters(daymin, daymax, duration)
            if len(rp) > 0:
                r += rp
        print('Number of SN to simulate:', len(r))
        if len(r) > 0:
            names = ['z', 'x1', 'color', 'daymax',
                     'epsilon_x0', 'epsilon_x1', 'epsilon_color',
                     'min_rf_phase', 'max_rf_phase']
            types = ['f8']*len(names)
            #params = np.zeros(len(r), dtype=list(zip(names, types)))
            params = np.asarray(r, dtype=list(zip(names, types)))
            return params
        else:
            return None

    def getParameters(self, daymin, daymax, duration):
        """ Get parameters

        Parameters
        --------------
        daymin : float
           min MJD
        daymax, float
           max MJD
        duration : float
           duration of observations

        Returns
        ----------
        list of parameters
        z, float
          redshift
        x1, float
           x1 parameter
        color, float,
          color parameter
        daymax, float
          T0 parameter
        epsilon_x0,float
          epsilon for x0 parameter
        epsilon_x1, float
          epsilon for x1 parameter
        epsilon_color, float
          epsilon for color parameter
        min_rf_phase, float
          min rest-frame phase for LC points
        max_rf_phase, float
          max rest-frame phase for LC points

        """

        # get z range
        zmin = self.params['z']['min']
        zmax = self.params['z']['max']
        r = []
        epsilon = 1.e-8
        if self.params['z']['type'] == 'random':
            # get sn rate for this z range
            #print(zmin, zmax, duration, self.area)
            zz, rate, err_rate, nsn, err_nsn = self.sn_rate(
                zmin=zmin, zmax=zmax,
                duration=duration,
                survey_area=self.area,
                account_for_edges=True,dz=0.001)
            # get number of supernovae
            N_SN = int(np.cumsum(nsn)[-1])
            if np.cumsum(nsn)[-1] <0.5:
                return r
            weight_z = np.cumsum(nsn)/np.sum(np.cumsum(nsn))
            
            if N_SN < 1:
                N_SN = 1
                #weight_z = 1
            print('nsn', N_SN)
            for j in range(N_SN):
                z = self.getVal(self.params['z']['type'], zmin, zz, weight_z)
                zrange = 'low_z'
                if z >= 0.1:
                    zrange = 'high_z'
                x1_color = self.getVal(self.params['x1_color']['type'],
                                       self.params['x1_color']['min'],
                                       self.x1_color[zrange][['x1', 'color']],
                                       self.x1_color[zrange]['weight'])
                T0_values = []
                if self.params['daymax']['type'] == 'unique':
                    T0_values = [daymin+20.*(1.+z)]
                if self.params['daymax']['type'] == 'random':
                    T0_values = np.arange(
                        daymin-(1.+z)*self.min_rf_phase, daymax-(1.+z)*self.max_rf_phase, 0.1)
                dist_daymax = T0_values
                #print('daymax',dist_daymax,type(dist_daymax))
                if dist_daymax.size == 0:
                    continue
                T0 = self.getVal(self.params['daymax']['type'],
                                 -1., dist_daymax,
                                 [1./len(dist_daymax)]*len(dist_daymax))
                r.append((z, x1_color[0], x1_color[1], T0, 0.,
                          0., 0., self.min_rf_phase, self.max_rf_phase))

        if self.params['z']['type'] == 'uniform':
            zstep = self.params['z']['step']
            daystep = self.params['daymax']['step']
            x1_color = self.params['x1_color']['min']

            if zmin == 0.01:
                zmin = 0.
            for z in np.arange(zmin, zmax+zstep, zstep):
                if z == 0.:
                    z = 0.01
                if self.params['daymax']['type'] == 'uniform':
                    T0_values = np.arange(
                        daymin-(1.+z)*self.min_rf_phase, daymax-(1.+z)*self.max_rf_phase, daystep)
                if self.params['daymax']['type'] == 'unique':
                    T0_values = [daymin+20.*(1.+z)]
                # print('phases', z, daymin, daymax, (daymax-daymin)/(1.+z))
                # print('T0s', T0_values)
                for T0 in T0_values:
                    r.append((z, x1_color[0], x1_color[1], T0, 0.,
                              0., 0., self.min_rf_phase, self.max_rf_phase))
                    if self.params['differential_flux']:
                        rstart = [z, x1_color[0], x1_color[1], T0, 0.,
                                  0., 0., self.min_rf_phase, self.max_rf_phase]
                        for kdiff in [4, -4, 5, -5, 6, -6]:
                            rstartc = list(rstart)
                            rstartc[np.abs(kdiff)] = epsilon*np.sign(kdiff)
                            r.append(tuple(rstartc))

        if self.params['z']['type'] == 'unique':
            daystep = self.params['daymax']['step']
            x1_color = self.params['x1_color']['min']
            z = self.params['z']['min']
            if self.params['daymax']['type'] == 'uniform':
                T0_values = np.arange(daymin, daymax, daystep)
            if self.params['daymax']['type'] == 'unique':
                T0_values = [daymin+20.*(1.+z)]
            for T0 in T0_values:
                r.append((z, x1_color[0], x1_color[1], T0, 0.,
                          0., 0., self.min_rf_phase, self.max_rf_phase))
                if self.params['differential_flux']:
                    if self.params['differential_flux']:
                        rstart = [z, x1_color[0], x1_color[1], T0, 0.,
                                  0., 0., self.min_rf_phase, self.max_rf_phase]
                        for kdiff in [4, -4, 5, -5, 6, -6]:
                            rstartc = list(rstart)
                            rstartc[np.abs(kdiff)] = epsilon*np.sign(kdiff)
                            r.append(tuple(rstartc))

        return r

    def getVal(self, type, val, distrib, weight):
        """ Get values of a given parameter

        Parameters
        ------------
        type: str
           random or not
        val : float
            value to retuen (if not random)
        distrib : float
           distribution where to get the parameters
        weight : float
           weight corresponding to the distribution

        Returns
        ---------
        parameter value : float
        """

        if type == 'random':
            return np.random.choice(distrib, 1, p=weight)[0]
        else:
            return val

    def getDist(self, rate):
        """ get (x1,color) distributions

        Parameters
        ---------
        rate: str
            name of the x1_color distrib (JLA, ...)

        Returns
        ---------
        dict of (x1,color) rates
        keys : 'low_z' and 'high_z'
        values (float) : recarray with X1,Color,weight_X1,weight_Color,weight 
        """

        #prefix = os.getenv('SN_UTILS_DIR')+'/input/Dist_X1_Color_'+rate+'_'
        prefix = '{}/Dist_X1_Color_{}'.format(self.dirFiles, rate)
        suffix = '.txt'
        # names=['x1','c','weight_x1','weight_c','weight_tot']
        dtype = np.dtype([('x1', np.float), ('color', np.float),
                          ('weight_x1', np.float), ('weight_color', np.float),
                          ('weight', np.float)])
        x1_color = {}
        for val in ['low_z', 'high_z']:
            x1_color[val] = np.loadtxt('{}_{}{}'.format(
                prefix, val, suffix), dtype=dtype)

        return x1_color

    def plotParameters(self, gen_params):
        """ Plot the generated parameters
        (z,x1,color,daymax)

        Parameters
        --------------
        gen_params : array
          array of parameters
          should at least contain ['z', 'x1', 'color', 'daymax'] fields

        """
        import pylab as plt

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 9))

        todraw = ['z', 'x1', 'color', 'daymax']
        idx = dict(zip(todraw, [(0, 0), (0, 1), (1, 1), (1, 0)]))

        for name in todraw:
            i = idx[name][0]
            j = idx[name][1]
            ax[i][j].hist(gen_params[name], histtype='step')
            ax[i][j].set_xlabel(name)
            ax[i][j].set_ylabel('Number of entries')

        plt.show()


class Make_Files_for_Cadence_Metric:

    def __init__(self, file_name, telescope, simulator_name):
        """ Class to generate two files that will be used as input for the Cadence metric

        Parameters
        ---------
        LC file (filename)

        Returns
        ---------
        recordarray of LC:
        MJD, Ra, Dec, band,m5,Nexp, ExpTime, Season
        recordarray of mag_to_flux values
        """

        self.telescope = telescope
        self.simulator_name = simulator_name

        self.prod_mag_to_flux()
        if file_name != '':
            self.prod_(file_name)

    def prod_mag_to_flux(self):
        """
        Mag to flux estimation
        Output file created: Mag_to_Flux_simulator.npy
        """
        mag_range = (12., 28.0)
        m5 = np.linspace(mag_range[0], mag_range[1], 50)
        mag_to_flux_tot = None
        for band in 'grizy':
            mag_to_flux = np.array(m5, dtype=[('m5', 'f8')])
            exptime = [30.] * len(m5)
            b = [band] * len(m5)
            f5 = self.telescope.mag_to_flux_e_sec(m5, b, exptime)
            mag_to_flux = rf.append_fields(mag_to_flux, ['band', 'flux_e'], [
                b, f5[:, [1]]], dtypes=['U256', 'f8'])
            if mag_to_flux_tot is None:
                mag_to_flux_tot = mag_to_flux
            else:
                mag_to_flux_tot = np.concatenate(
                    (mag_to_flux_tot, mag_to_flux))
        # print(mag_to_flux_tot)
        # print('done')
        # np.save('Mag_to_Flux_'+self.simulator_name +
        # '.npy', np.copy(mag_to_flux_tot))
        np.save('Mag_to_Flux_LSST_sim' +
                '.npy', np.copy(mag_to_flux_tot))

    def prod_(self, filename):
        """
        Reformat LC input files to numpy array

        Parameters
        --------------
        filename : str
          name of the file with LC points (format: hdf5 ! )

        Returns
        ----------
        npy file produced 
        name of the file: Li_simulator_x1_color.npy
        numpy array with the following fields:
        time : float
           time (MJD)
         band : str
           band
         flux_e : float
           flux (in e/sec)
        flux : float
          flux (Jky?)
        z : float
          redshift
        daymax : float
          T0
        """
        import h5py
        f = h5py.File(filename, 'r')
        # print(f.keys())
        simu = {}
        for i, key in enumerate(f.keys()):
            # print(i)
            simu[i] = Table.read(filename, path=key)

        restot = None
        for key, val in simu.items():
            # print(val.meta)
            z = val.meta['z']
            x1 = val.meta['x1']
            color = val.meta['color']
            daymax = val.meta['daymax']
            idx = val['flux_e'] > 0.
            sel = val[idx]

            res = np.array(np.copy(sel[['time', 'band', 'flux_e', 'flux']]), dtype=[
                ('time', '<f8'), ('band', 'U8'), ('flux_e', '<f8'), ('flux', '<f8')])

            res = rf.append_fields(res, 'z', [z]*len(res))
            res = rf.append_fields(res, 'DayMax', [DayMax]*len(res))
            if restot is None:
                restot = res
            else:
                restot = np.concatenate((restot, res))

        # print(restot)
        np.save('Li_'+self.simulator_name+'_'+str(x1) +
                '_'+str(color)+'.npy', np.copy(restot))
