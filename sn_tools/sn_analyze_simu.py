import numpy as np
import h5py
import argparse
from astropy.table import Table
import pylab as plt
from sn_tools.sn_rate import SN_Rate


def plotParameters(tab, season=None):
    """ 
    Simple function to show how to access results from simulation

    Parameters
    ---------------
    tab: recarray of parameters
    season: season

    Returns
    ------------
    Plot (x1,color,daymax,SN_Rate)
    """

    sel = tab
    if season is not None:
        idx = tab['season'] == season
        sel = tab[idx]
    thesize = 15
    toplot = ['x1', 'color', 'daymax', 'z']
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 9))

    for i, var in enumerate(toplot):
        ix = int(i/2)
        iy = i % 2
        axis = ax[ix][iy]
        if var != 'z':
            axis.hist(sel[var])
        axis.set_xlabel(var, fontsize=20)
        axis.set_ylabel('Number of entries', fontsize=thesize)
        axis.tick_params(axis='x', labelsize=thesize)
        axis.tick_params(axis='y', labelsize=thesize)
        if var == 'z':
            n, bins, patches = axis.hist(sel[var])
            bin_center = (bins[:-1] + bins[1:]) / 2
            dz = bins[1]-bins[0]
            sn_rate = SN_Rate(rate='Perrett', H0=72, Om0=0.3)
            zmin = np.min(sel['z'])
            zmax = np.max(sel['z'])
            duration = np.max(sel['daymax'])-np.min(sel['daymax'])
            zz, rate, err_rate, nsn, err_nsn = sn_rate(
                zmin=zmin-dz/2., zmax=zmax, dz=dz, duration=duration, survey_area=9.6)
            axis.plot(zz, np.cumsum(nsn))
            axis.plot(bin_center, np.cumsum(n))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simu file analysis')
    parser.add_argument('--filename',
                        default='',
                        help='filename to analyze')

    args = parser.parse_args()

    # load the hdf5 file
    filename = args.filename

    f = h5py.File(filename, 'r')
    print(f.keys())
    for i, key in enumerate(f.keys()):
        summary = Table.read(filename, path=key)
        print(len(summary), summary.dtype)
        plotParameters(summary)
