import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

import pycis
from defunct.bbo import bbo_slo


def plot_bbo_dispersion(wavelength_range=(400e-9, 700e-9)):
    """ Plot just about everything you would want to know about BBO dispersion over specified wavelength range. """

    wavelength = np.linspace(wavelength_range[0], wavelength_range[1], 1001)
    sources = pycis.model.bbo_sources
    no_sources = len(sources)
    source_idx = np.linspace(0, no_sources - 1, no_sources, dtype=np.int)

    figsize = [8, 5]
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rc('axes', titlesize = 20, labelsize=20)
    color_idx = np.linspace(0, 1, no_sources)
    colormap = plt.get_cmap('nipy_spectral')

    # birefringence
    fig_biref = plt.figure(figsize=figsize)
    ax_biref = fig_biref.add_subplot(111)
    ax_biref.set_xlabel(r'$\lambda$ [m]')
    ax_biref.set_ylabel(r'$B(\lambda)$')

    # refractive indices
    fig_indices = plt.figure(figsize=figsize)
    ax_indices = fig_indices.add_subplot(111)
    ax_indices.set_xlabel(r'$\lambda$ [m]')
    ax_indices.set_ylabel(r'$n_{e, o}(\lambda)$')

    # Dispersion parameter kappa
    fig_kappa = plt.figure(figsize=figsize)
    ax_kappa = fig_kappa.add_subplot(111)
    ax_kappa.set_xlabel(r'$\lambda$ [m]')
    ax_kappa.set_ylabel(r'$\kappa(\lambda)$')

    # birefringence first derivative wrt. wavelength
    fig_dBdlambda = plt.figure(figsize=figsize)
    ax_dBdlambda = fig_dBdlambda.add_subplot(111)
    ax_dBdlambda.set_xlabel(r'$\lambda$ [m]')
    ax_dBdlambda.set_ylabel(r'$\frac{dB}{d\lambda}$')

    # birefringence second derivative wrt. wavelength
    fig_d2Bdlambda2 = plt.figure(figsize=figsize)
    ax_d2Bdlambda2 = fig_d2Bdlambda2.add_subplot(111)
    ax_d2Bdlambda2.set_xlabel(r'$\lambda$ [m]')
    ax_d2Bdlambda2.set_ylabel(r'$\frac{d^2B}{d\lambda^2}$')

    for i, j in zip(source_idx, color_idx):

        # retrieve dispersion info
        biref, n_e, n_o, kappa, dBdlambda, d2Bdlambda2,  d3Bdlambda3  = bbo_slo(wavelength, source=sources[i])

        # add to plot
        ax_biref.plot_raw(wavelength, biref, color=colormap(j), label=sources[i])

        ax_indices.plot_raw(wavelength, n_e, color=colormap(j), label=sources[i])
        ax_indices.plot_raw(wavelength, n_o, color=colormap(j))

        ax_kappa.plot_raw(wavelength, kappa, color=colormap(j), label=sources[i])

        ax_dBdlambda.plot_raw(wavelength, dBdlambda, color=colormap(j), label=sources[i])
        ax_d2Bdlambda2.plot_raw(wavelength, d2Bdlambda2, color=colormap(j), label=sources[i])

    leg_biref = ax_biref.legend(prop={'size': 15}, loc=0)
    leg_indices = ax_indices.legend(prop={'size': 15}, loc=0)
    leg_kappa = ax_kappa.legend(prop={'size': 15}, loc=0)
    leg_dBdlambda = ax_dBdlambda.legend(prop={'size': 15}, loc=0)
    leg_d2Bdlambda2 = ax_d2Bdlambda2.legend(prop={'size': 15}, loc=0)

    plt.show()
    return



if __name__ == '__main__':
    plot_bbo_dispersion()