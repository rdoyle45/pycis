import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.constants import c
import scipy.interpolate
import os

import pycis
import pystark


class ContrastCurve(object):
    """ Look-up table  contrast. Key analysis tool for contrast measurements. """

    def __init__(self, l_wp, n_upper, line_model, overwrite=False):
        """
        
        :param l_wp:  waveplate thickness [m] 
        :param n_upper: upper principal quantum number
        :param line_model: lineshape model for Stark-broadened line.
        """

        self.l_wp = l_wp
        self.n_upper = n_upper
        self.line_model = line_model

        self.n_lower = 2  # only Balmer series supported
        self.dens_axis = np.logspace(19., 21.99, 50)  # [m^-3]
        self.temp_axis = np.logspace(-0.3, 1.45, 40)  # [eV]
        self.dens_mesh, self.temp_mesh = np.meshgrid(self.dens_axis, self.temp_axis)

        # check valid input
        assert line_model in pystark.line_models

        # get line name
        assert n_upper > 2
        line_names = ['', '', '', 'Ba_alpha', 'Ba_beta', 'Ba_gamma', 'Ba_delta', 'Ba_epsilon']
        self.line_name = line_names[n_upper]

        save_dir = os.path.join(pycis.paths.tools_path, 'saved_contrast_curves')
        self.fname = os.path.join(save_dir, self.line_name + '_' + line_model + '_' + pycis.tools.to_precision(l_wp * 1e3, 3) + 'mm.npy')

        # load or calculate the contrast curve
        if os.path.isfile(self.fname) and overwrite is False:
            self.curve_grid = np.load(self.fname)
        else:
            self.curve_grid = self.make()
            self.save()

        # smooth curve
        self.curve_interp = scipy.interpolate.RectBivariateSpline(self.temp_axis, self.dens_axis, self.curve_grid, s=0.03)
        self.curve_grid_smooth = self.curve_interp(self.temp_axis, self.dens_axis)

        # extract the density dependence of the contrast curve at the lowest available temperature
        self.curve_0 = self.curve_grid_smooth[0, :]

    def make(self):

        LEN_WL_AXIS = 501

        contrast_curve = np.zeros_like(self.dens_mesh)

        print('--pycis: making contrast curve')

        # populate ls_mesh at all densities
        for idx_dens, dens in enumerate(self.dens_axis):
            for idx_temp, temp in enumerate(self.temp_axis):

                # print(idx_dens, idx_temp)

                wl_axis = pystark.get_wl_axis(self.n_upper, dens, temp, 0., npts=LEN_WL_AXIS)

                bls = pystark.BalmerLineshape(self.n_upper, dens, temp, 0., line_model=self.line_model, wl_axis=wl_axis)
                ls_m = bls.ls_szd

                # calculate interferometer delay time
                degree_coherence, abbo_axis = pycis.model.degree_coherence_general(ls_m, wl_axis, display=False)
                contrast_delay_curve = abs(degree_coherence)

                # interpolate contrast at the fixed delay -- this bit can be sped up.
                contrast_curve[idx_temp, idx_dens] = np.interp(self.l_wp, abbo_axis, contrast_delay_curve)

        return contrast_curve

    def save(self):
        np.save(self.fname, self.curve_grid)

    def measure_contrast(self, dens, temp):
        """ Given a plasma density, look up the measured contrast value. """
        return self.curve_interp.ev(temp, dens)

    def infer_density(self, measured_contrast):
        """ Given a measured contrast, infer the plasma density."""

        inferred_density = np.interp(measured_contrast, self.curve_0[::-1], self.dens_axis[::-1])
        return inferred_density

    def systematic_error_curve(self, real_temp, display=False):
        """ Given """

        systematic_error_curve = np.zeros_like(self.dens_axis)

        for idx_dens, dens in enumerate(self.dens_axis):

            measured_contrast = self.measure_contrast(dens, real_temp)
            inferred_density = self.infer_density(measured_contrast)

            systematic_error_curve[idx_dens] = inferred_density - dens

        systematic_error_curve[np.argmax(systematic_error_curve < 0.):] = 0.
        systematic_error_curve = scipy.signal.savgol_filter(systematic_error_curve, 5, 1, mode='interp')

        if display:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot_raw(self.dens_axis, (systematic_error_curve / self.dens_axis))

            ax.semilogx()
            plt.show()


        return systematic_error_curve

    # Plotting methods

    def add_to_plot(self, ax, **kwargs):
        """ Add contrast curve to existing matplotlib plot by passing the axis as an argument."""
        ax.plot(self.dens_axis, self.curve_0, **kwargs)

    def display(self):
        """ Generate a new matplotlibfigure window, plotting the contrast density curve."""

        fsize = 14

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        im = ax.pcolor(self.dens_mesh, self.temp_mesh, self.curve_grid_smooth)
        contour_contrast = 0.8
        CS = ax.contour(self.dens_mesh, self.temp_mesh, self.curve_grid_smooth, levels=[contour_contrast], colors='red', linestyles='solid', linewidths=[3])

        fmt = {}
        strs = [str(contour_contrast)]
        for l, s in zip(CS.levels, strs):
            fmt[l] = s

        # manual_locations = [(3e19, 4e0)]
        # plt.clabel(CS, CS.levels, inline=False, fmt=fmt, colors='red', fontsize=fsize,
        #            manual=manual_locations)  # contour line labels

        ax.semilogx()
        ax.semilogy()
        ax.set_xlim([1e19, 2e21])

        ax.set_xlabel('density (m$^{-3}$)', size=fsize)
        ax.set_ylabel('temperature (eV)', size=fsize)

        cbar = plt.colorbar(im, ax=ax, ticks=[0., np.max(self.curve_grid_smooth)])
        cbar.set_label(label='contrast', size=fsize)
        cbar.ax.set_yticklabels(['0', '1'])

        # ax.annotate('0.8', )


        plt.show()


if __name__ == '__main__':
    cdc = ContrastCurve(4.48e-3, 5, 'lomanowski', overwrite=False)
    print(cdc.measure_contrast(3, 1.2e19))

    # cdc.display()

    systematic_error_curve = cdc.systematic_error_curve(1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot_raw(cdc.dens_axis, systematic_error_curve / cdc.dens_axis)
    ax.plot_raw(cdc.dens_axis, cdc.curve_0)
    ax.plot_raw(cdc.dens_axis, cdc.curve_grid_smooth[-5, :])

    print(cdc.temp_axis[-5])

    ax.semilogx()
    ax.set_ylim([0, 1])
    plt.show()

