import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.constants import c
import scipy.interpolate
import os

import pycis
import pystark


class ContrastCurve(object):
    """ Look-up table between plasma properties and inferred contrast. Key analysis tool for contrast measurements. """

    def __init__(self, l_wp, n_upper, line_model, overwrite=False):
        """
        
        :param l_wp:  waveplate thickness [m] 
        :param n_upper: upper principal quantum number
        :param line_model: lineshape model for Stark-broadened line.
        """

        self.l_wp = l_wp
        self.n_upper = n_upper
        self.ls_model = line_model

        self.n_lower = 2  # only Balmer series supported
        self.dens_axis = np.logspace(19., 21.99, 50)  # [m^-3]
        self.temp_axis = np.logspace(-0.3, 1.45, 40)  # [eV]
        self.dens_mesh, self.temp_mesh = np.meshgrid(self.dens_axis, self.temp_axis)

        # check valid input
        ls_models = {'lomanowski': pystark.simple_profile, 'stehle': pystark.stehle_profile,
                     'rosato': pystark.rosato_profile}
        assert line_model in ls_models

        # get line name
        assert n_upper > 2
        line_names = ['', '', '', 'Ba_alpha', 'Ba_beta', 'Ba_gamma', 'Ba_delta', 'Ba_epsilon']
        self.line_name = line_names[n_upper]

        # save_dir = '/Users/jsallcock/Documents/physics/phd/code/CIS/pycis_stark/saved_contrast_curves/'
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

        # extract the density dependence of the contrast curve at the lowest avaiolabel temperature
        self.curve_0 = self.curve_grid_smooth[0, :]

    def make(self):

        LEN_WL_AXIS = 501

        contrast_curve = np.zeros_like(self.dens_mesh)

        # populate ls_mesh at all densities
        for idx_dens, dens in enumerate(self.dens_axis):
            for idx_temp, temp in enumerate(self.temp_axis):

                print(idx_dens, idx_temp)

                wl_axis = pystark.generate_wavelength_axis(self.n_upper, temp, dens, npts=LEN_WL_AXIS)

                if self.ls_model == 'lomanowski':
                    ls_m, _, _ = pystark.simple_profile(self.n_upper, self.n_lower, temp, dens, wl_axis, model='lomanowski', display=False)

                elif self.ls_model == 'stehle':
                    ls_m = pystark.stehle_profile(self.n_upper, self.n_lower, temp, dens, wl_axis, display=False)

                elif self.ls_model == 'rosato':
                    ls_m = pystark.rosato_profile(self.n_upper, dens, temp, 0., 0., wl_axis, display=False)

                else:
                    raise Exception('enter valid model.')

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
        ax.plot_raw(self.dens_axis, self.curve_grid, **kwargs)

    def display(self):
        """ Generate a new matplotlibfigure window, plotting the contrast density curve."""

        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.pcolor(self.dens_mesh, self.temp_mesh, self.curve_grid_smooth)

        ax.semilogx()
        ax.semilogy()

        cbar = plt.colorbar(im, ax=ax)

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

