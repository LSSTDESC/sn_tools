import numpy as np
from sn_tools.sn_rate import SN_Rate
import os
import numpy.lib.recfunctions as rf
from astropy.table import Table


class Generate_Sample:
    """ Generates a sample of parameters for simulation
    Input
    ---------
    sn_parameters: X1, Color, z, DayMax, ...
    (see yaml file input)
    cosmology parameters: H0, Om0
    Returns (call)
    ---------
    recordarray of parameters to simulate:
    X1,Color,z,DayMax
    """

    def __init__(self, sn_parameters, cosmo_parameters, mjdCol='mjd', seasonCol='season', filterCol='filter', min_rf_phase=-15., max_rf_phase=30., area=9.6):

        self.params = sn_parameters
        self.sn_rate = SN_Rate(rate=self.params['z']['rate'],
                               H0=cosmo_parameters['H0'],
                               Om0=cosmo_parameters['Omega_m'])

        self.x1_color = self.Get_Dist(self.params['X1_Color']['rate'])
        self.mjdCol = mjdCol
        self.seasonCol = seasonCol
        self.filterCol = filterCol
        self.area = area
        self.min_rf_phase = min_rf_phase
        self.max_rf_phase = max_rf_phase

    def __call__(self, obs):
        """
        Input
        ---------
        recarray of observations

        Returns
        ---------
        recarray of sn parameters for simulation:
        z, X1 , Color ,  DayMax
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

            rp = self.Get_parameters(daymin, daymax, duration)
            if len(rp) > 0:
                r += rp
        print('Number of SN to simulate:', len(r))
        if len(r) > 0:
            return np.rec.fromrecords(r, names=['z', 'X1', 'Color', 'DayMax', 'epsilon_X0', 'epsilon_X1', 'epsilon_Color', 'min_rf_phase', 'max_rf_phase'])
        else:
            return None

    def Get_parameters(self, daymin, daymax, duration):
        # get z range
        zmin = self.params['z']['min']
        zmax = self.params['z']['max']
        r = []
        epsilon = 1.e-8
        if self.params['z']['type'] == 'random':
            # get sn rate for this z range
            zz, rate, err_rate, nsn, err_nsn = self.sn_rate(
                zmin=zmin, zmax=zmax,
                duration=duration,
                survey_area=self.area,
                account_for_edges=True)
            # get number of supernovae
            N_SN = int(np.cumsum(nsn)[-1])
            weight_z = np.cumsum(nsn)/np.sum(np.cumsum(nsn))

            for j in range(N_SN):
                z = self.Get_Val(self.params['z']['type'], zmin, zz, weight_z)
                zrange = 'low_z'
                if z >= 0.1:
                    zrange = 'high_z'
                x1_color = self.Get_Val(self.params['X1_Color']['type'],
                                        self.params['X1_Color']['min'],
                                        self.x1_color[zrange][['X1', 'Color']],
                                        self.x1_color[zrange]['weight'])
                T0_values = []
                if self.params['DayMax']['type'] == 'unique':
                    T0_values = [daymin+20.*(1.+z)]
                if self.params['DayMax']['type'] == 'random':
                    T0_values = np.arange(
                        daymin-(1.+z)*self.min_rf_phase, daymax-(1.+z)*self.max_rf_phase, 0.1)
                dist_daymax = T0_values
                T0 = self.Get_Val(self.params['DayMax']['type'],
                                  -1., dist_daymax,
                                  [1./len(dist_daymax)]*len(dist_daymax))
                r.append((z, x1_color[0], x1_color[1], T0, 0.,
                          0., 0., self.min_rf_phase, self.max_rf_phase))

        if self.params['z']['type'] == 'uniform':
            zstep = self.params['z']['step']
            daystep = self.params['DayMax']['step']
            x1_color = self.params['X1_Color']['min']
            # if self.params['DayMax']['type'] == 'uniform':
            #T0_values = np.arange(daymin, daymax, daystep)
            #T0_values = np.arange(daymin+(1.+z)*self.min_rf_phase, daymax-(1.+z)*self.max_rf_phase, daystep)
            if zmin == 0.01:
                zmin = 0.
            for z in np.arange(zmin, zmax+zstep, zstep):
                if z == 0.:
                    z = 0.01
                if self.params['DayMax']['type'] == 'uniform':
                    T0_values = np.arange(
                        daymin-(1.+z)*self.min_rf_phase, daymax-(1.+z)*self.max_rf_phase, daystep)
                if self.params['DayMax']['type'] == 'unique':
                    T0_values = [daymin+20.*(1.+z)]
                #print('phases', z, daymin, daymax, (daymax-daymin)/(1.+z))
                #print('T0s', T0_values)
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
            daystep = self.params['DayMax']['step']
            x1_color = self.params['X1_Color']['min']
            z = self.params['z']['min']
            if self.params['DayMax']['type'] == 'uniform':
                T0_values = np.arange(daymin, daymax, daystep)
            if self.params['DayMax']['type'] == 'unique':
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

    def Get_Val(self, type, val, distrib, weight):
        """ Get values of a given parameter
        Input
        ---------
        type: random or not
        val : return value (if not random)
        distrib, weight: distrib and weight to get the parameter

        Returns
        ---------
        parameter value
        """

        if type == 'random':
            return np.random.choice(distrib, 1, p=weight)[0]
        else:
            return val

    def Get_Dist(self, rate):
        """ get (X1,C) distributions
        Input
        ---------
        rate: name of the X1_C distrib (JLA, ...)

        Returns
        ---------
        dict of (X1,C) rates
        keys: 'low_z' and 'high_z'
        val: recarray with X1,Color,weight_X1,weight_Color,weight
        """

        prefix = os.getenv('SN_UTILS_DIR')+'/input/Dist_X1_Color_'+rate+'_'
        suffix = '.txt'
        # names=['x1','c','weight_x1','weight_c','weight_tot']
        dtype = np.dtype([('X1', np.float), ('Color', np.float),
                          ('weight_X1', np.float), ('weight_Color', np.float),
                          ('weight', np.float)])
        x1_color = {}
        for val in ['low_z', 'high_z']:
            x1_color[val] = np.loadtxt(prefix+val+suffix, dtype=dtype)

        return x1_color

    def Plot_Parameters(self, gen_params):
        """ Plot the generated parameters
        (z,X1,Color,DayMax)
        """
        import pylab as plt

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 9))

        todraw = ['z', 'X1', 'Color', 'DayMax']
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
        Input
        ---------
        LC file (filename)
        Output: 
        ---------
        recordarray of LC:
        MJD, Ra, Dec, band,m5,Nexp, ExpTime, Season
        recordarray of mag_to_flux values
        """

        self.telescope = telescope
        self.simulator_name = simulator_name

        self.Prod_mag_to_flux()
        # self.Prod_(file_name)

    def Prod_mag_to_flux(self):
        mag_range = (20., 28.0)
        m5 = np.linspace(mag_range[0], mag_range[1], 50)
        mag_to_flux_tot = None
        for band in 'grizy':
            mag_to_flux = np.array(m5, dtype=[('m5', 'f8')])
            exptime = [30.] * len(m5)
            b = [band] * len(m5)
            f5 = self.telescope.mag_to_flux_e_sec(m5, b)
            #print(b, f5[:], f5[:, [0]])
            mag_to_flux = rf.append_fields(mag_to_flux, ['band', 'flux_e'], [
                                           b, f5[:, [1]]], dtypes=['U256', 'f8'])
            if mag_to_flux_tot is None:
                mag_to_flux_tot = mag_to_flux
            else:
                mag_to_flux_tot = np.concatenate(
                    (mag_to_flux_tot, mag_to_flux))
        # print(mag_to_flux_tot)
        # print('done')
        np.save('Mag_to_Flux_'+self.simulator_name +
                '.npy', np.copy(mag_to_flux_tot))

    def Prod_(self, filename):
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
            X1 = val.meta['X1']
            Color = val.meta['Color']
            DayMax = val.meta['DayMax']
            idx = val['flux_e'] > 0.
            sel = val[idx]
            #print(z, len(val), len(val[idx]))
            res = np.array(np.copy(sel[['time', 'band', 'flux_e', 'flux']]), dtype=[
                           ('time', '<f8'), ('band', 'U8'), ('flux_e', '<f8'), ('flux', '<f8')])
            #print(res.dtype, z)
            res = rf.append_fields(res, 'z', [z]*len(res))
            res = rf.append_fields(res, 'DayMax', [DayMax]*len(res))
            if restot is None:
                restot = res
            else:
                restot = np.concatenate((restot, res))

        # print(restot)
        np.save('Li_'+self.simulator_name+'_'+str(X1) +
                '_'+str(Color)+'.npy', np.copy(restot))
