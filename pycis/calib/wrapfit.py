import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import math

import pycis
import inference


def wrapfit(wls, phase, phase_std, l_wp_guess, idx_0=0, num_steps=120000, polynomial_order=2, display=True):
    """ Fit to wrapped phase measurements.

    :param wls: wavelength values (m)
    :param phase: wrapped phase values (waves)
    :param phase_std: (waves)
    :param idx_0: int.
    :param l_wp_guess: (m)
    :param polynomial_order: linear (1), quadratic (2) and cubic (3) are supported. 

    :return: 
    """

    post = WrapFitPosterior(wls, phase, phase_std, idx_0, l_wp_guess, polynomial_order=polynomial_order, display=False)

    # post.plot_exact_fractions()

    # MCMC Sampling
    chain = inference_tools.inference.mcmc.GibbsChain(posterior=post, start=post.parameter_starts, widths=post.parameter_widths)

    print(post.wl_0, post.parameter_starts)
    chain.advance(num_steps)

    # we can check the status of the chain using the plot_diagnostics method
    if display:
        post.plot()
        chain.plot_diagnostics()
    # chain.matrix_plot()

    gd0_posterior = chain.marginalise(0, burn=10000, thin=2, unimodal=False)
    gd0_mode = gd0_posterior.mode
    mu, var, skw, kur = gd0_posterior.moments()
    gd0_std = np.sqrt(var)

    parameter_modes = [gd0_mode]
    parameter_stds = [gd0_std]

    if polynomial_order > 1:
        for i in range(2, polynomial_order + 1):
            marginalised_posterior = chain.marginalise(i - 1, burn=10000, thin=2, unimodal=False)
            parameter_modes.append(marginalised_posterior.mode)
            parameter_stds.append(np.sqrt(marginalised_posterior.moments()[1]))

    print(parameter_modes, parameter_stds)

    # output marginalised group delay pdf on an appropriate axis:

    gd0_axis = np.linspace(parameter_modes[0] - 5 * parameter_stds[0], parameter_modes[0] + 5 * parameter_stds[0], 300)
    gd0_pdf = gd0_posterior(gd0_axis)

    if display:

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        post.plot_phase_ambiguities(ax1)

        wl_axis = np.linspace(np.min(post.wls), np.max(post.wls), 100)
        wl_shift_axis = wl_axis - post.wl_0

        phase_shift_mode = pycis.tools.phase_shift_wl_poly(wl_shift_axis, parameter_modes) / wl_axis
        phase_shift_std = pycis.tools.phase_shift_wl_poly_std(wl_shift_axis, parameter_stds) / wl_axis

        ax1.plot_raw(wl_shift_axis, phase_shift_mode, 'r', lw=1, label='mode')
        ax1.fill_between(wl_shift_axis, phase_shift_mode - phase_shift_std, phase_shift_mode + phase_shift_std, color='peachpuff',
                        label='1std')

        plt.show()


    return gd0_axis, gd0_pdf


class WrapFitPosterior(object):
    def __init__(self, wls, phase, phase_std, idx_0, l_wp_guess, polynomial_order=2, display=False):

        """
        
        :param wls:  [m] 
        :param phase: [fringes]
        :param phase_std: [fringes]
        :param idx_0: 
        :param l_wp_guess: 
        :param polynomial_order: 
        :param display: 
        """

        # posterior object created by wrapfit()

        self.wls = wls
        self.phase = phase
        self.phase_std = phase_std

        self.fsize = 13

        # ensure valid idx_0
        assert idx_0 > -len(wls) and idx_0 < len(wls)
        if idx_0 < 0:
            idx_0 = len(wls) + idx_0

        self.idx_0 = idx_0
        self.l_wp_guess = l_wp_guess
        self.polynomial_order = polynomial_order
        self.NUM_SIGMA = 5

        # extract wavelength, phase, phase uncertainty for chosen 'centre' wavelength, at which the group delay will be
        # found.
        self.wl_0 = wls[idx_0]
        self.phase_0 = phase[idx_0]
        self.phase_std_0 = phase_std[idx_0]

        # now remove this point from the shifted coordinates
        omit_zero_idx = self.wls != self.wl_0

        self.wls_reduced = self.wls[omit_zero_idx]  # tabulated_data wavelengths, excluding the targeted wavelength wl_0.
        self.wl_shift = (self.wls[omit_zero_idx] - self.wl_0)
        self.phase_shift = pycis.demod.wrap(phase[omit_zero_idx] - self.phase_0, units='fringes')
        self.phase_shift_std = np.sqrt(phase_std[omit_zero_idx] ** 2 + self.phase_std_0 ** 2)

        self.phase_shift_wl = self.phase_shift * self.wls_reduced
        self.phase_shift_wl_std = self.phase_shift_std * self.wls_reduced

        self.NUM_PHASE_POINTS = len(wls)
        self.NUM_PHASE_SHIFT_POINTS = np.sum(omit_zero_idx)

        # calculate the number and index of the phase ambiguity points that must be evaluated at each wavelength location:
        self.phase_shift_ambig_idx = []
        self.ambig_no = []  # number of ambiguities accounted for at each wavelength
        self.phase_shift_prior_mean = []  # mean of the prior PDF expressed in phase shift

        phase_shift_nom, phase_shift_bounds = self._get_phase_shift_bounds(self.wls_reduced)  # NUM_SIGMA?

        for i, wli in enumerate(self.wls_reduced):

            ambiguity_idx = self._get_ambiguities(self.phase_shift[i], phase_shift_bounds[:, i])
            self.phase_shift_ambig_idx.append(ambiguity_idx)
            self.ambig_no.append(len(ambiguity_idx))

        # find max number of phase ambiguity points across all raw_data points - this informs size of the padded
        # likelihood exponent array.
        self.max_ambig_no = len(max(self.phase_shift_ambig_idx, key=len))

        # create arrays for quantities used in vectorised likelihood calculation
        self.ambig_idx_mesh = np.zeros([self.NUM_PHASE_SHIFT_POINTS, self.max_ambig_no])
        self.data_mesh = np.zeros([self.NUM_PHASE_SHIFT_POINTS, self.max_ambig_no])
        self.phase_shift_std_mesh = np.zeros([self.NUM_PHASE_SHIFT_POINTS, self.max_ambig_no])

        self.wavelength_shift_mesh = np.zeros([self.NUM_PHASE_SHIFT_POINTS, self.max_ambig_no])
        self.wavelength_mesh = np.zeros([self.NUM_PHASE_SHIFT_POINTS, self.max_ambig_no])

        for k in range(0, self.NUM_PHASE_SHIFT_POINTS):

            self.data_mesh[k, :] = self.phase_shift[k]
            self.wavelength_mesh[k, :] = self.wls_reduced[k]
            self.wavelength_shift_mesh[k, :] = self.wl_shift[k]
            self.phase_shift_std_mesh[k, :] = self.phase_shift_std[k]

            # pad ambig_idx with negative infinity
            while len(self.phase_shift_ambig_idx[k]) < self.max_ambig_no:
                self.phase_shift_ambig_idx[k] = np.append(self.phase_shift_ambig_idx[k], np.inf)

            self.ambig_idx_mesh[k, :] = self.phase_shift_ambig_idx[k]

        if display:
            self.plot()
            plt.show()

    def __call__(self, theta):
        return self.likelihood(theta) + self.prior(theta)

    def _get_phase_shift_bounds(self, wl_axis, num_sigma=5):
        """ Given a range of wavelengths, use prior info to set hard bounds on the corresponding phase shifts."""

        wl_shift_axis = wl_axis - self.wl_0

        self.parameter_starts, self.parameter_widths = self._get_starts_and_widths(self.wl_0)

        phase_shift_prior_mean = pycis.tools.phase_shift_wl_poly(wl_shift_axis, self.parameter_starts) / wl_axis
        phase_shift_prior_std = pycis.tools.phase_shift_wl_poly_std(wl_shift_axis, self.parameter_widths) / wl_axis

        phase_shift_lb = phase_shift_prior_mean - (num_sigma * phase_shift_prior_std)
        phase_shift_ub = phase_shift_prior_mean + (num_sigma * phase_shift_prior_std)

        phase_shift_bounds = np.array([phase_shift_lb, phase_shift_ub])

        return phase_shift_prior_mean, phase_shift_bounds

    def _get_starts_and_widths(self, wl):
        """ Get the starting points and approximate distribution widths, in fit parameter space, for the MCMC sampler.
         This probably needs a better name. """

        # take mean of BBO dispersive quantities for best initial guess of true values
        bbo_source_idxs = [0, 1, 2]
        NUM_BBO_SELLMEIER_SOURCES = len(bbo_source_idxs)
        biref_sources = np.zeros(NUM_BBO_SELLMEIER_SOURCES)
        kappa_sources = np.zeros(NUM_BBO_SELLMEIER_SOURCES)
        dbdlambda_sources = np.zeros(NUM_BBO_SELLMEIER_SOURCES)
        d2bdlambda2_sources = np.zeros(NUM_BBO_SELLMEIER_SOURCES)
        d3bdlambda3_sources = np.zeros(NUM_BBO_SELLMEIER_SOURCES)

        for i in range(NUM_BBO_SELLMEIER_SOURCES):
            biref_sources[i], _, _, kappa_sources[i], dbdlambda_sources[i], d2bdlambda2_sources[i], d3bdlambda3_sources[
                i] = pycis.model.bbo_dispersion(wl, i)

        biref_mean = np.mean(biref_sources)
        kappa_mean = np.mean(kappa_sources)

        dbdlambda_mean = np.mean(dbdlambda_sources)

        # print(dbdlambda_mean, (biref_mean / wl) * (1 - kappa_mean))
        d2bdlambda2_mean = np.mean(d2bdlambda2_sources)
        d3bdlambda3_mean = np.mean(d3bdlambda3_sources)

        group_delay = - kappa_mean * biref_mean * self.l_wp_guess / wl

        # print(biref_sources, kappa_sources, self.l_wp, wl, group_delay)

        derivatives = [biref_mean, dbdlambda_mean, d2bdlambda2_mean, d3bdlambda3_mean]
        poly_coeffs = [group_delay]

        for i, derivative in enumerate(derivatives[:self.polynomial_order + 1]):
            if i < 2:
                pass
            else:
                poly_coeffs.append(((self.l_wp_guess / math.factorial(i)) * derivative))

        cutoff_fraction = 0.05

        poly_coeffs = np.array(poly_coeffs)
        # poly_coeffs[2] = 1e18
        poly_coeff_stds = cutoff_fraction * poly_coeffs  # used to inform the prior, a measure of the uncertainty in the
        # fitting parameters prior to measurement.

        return poly_coeffs, poly_coeff_stds

    def _get_ambiguities(self, wrapped_phase_shift, phase_shift_bounds):
        # phase ambiguities for display (only true values)

        phase_shift_lb, phase_shift_ub = phase_shift_bounds

        # look for ambiguities above wrapped point, between the upper and lower bounds
        idx_throw_int, idx_throw_mod = divmod(phase_shift_lb - wrapped_phase_shift, 1)
        idx_keep_int, idx_keep_mod = divmod(phase_shift_ub - phase_shift_lb + idx_throw_mod, 1)

        phase_shift_ambiguity_idx = np.arange(idx_throw_int + 1, idx_throw_int + idx_keep_int + 1)

        # phase_shift_ambiguity_indices.append(phase_change_ambiguity_index)

        return phase_shift_ambiguity_idx


    def prior(self, theta):
        """ uniform prior distribution in log-space.
        """

        return 0.

    def likelihood(self, theta):
        """
        In this example we assume that the errors on our raw_data
        are Gaussian, such that the log-likelihood takes the
        form given below:
        """

        exponent_mesh = - 0.5 * ((self.data_mesh + self.ambig_idx_mesh - (self.forward_model(self.wavelength_shift_mesh, theta) / self.wavelength_mesh)) / self.phase_shift_std_mesh) ** 2
        point_log_likelihood = scipy.misc.logsumexp(exponent_mesh, axis=1)

        return sum(point_log_likelihood)

    def forward_model(self, x, theta):
        """
        Makes a prediction of the experimental tabulated_data we would expect to measure given a specific state of the
        system, which is specified by the model parameters theta.
        """

        return pycis.tools.phase_shift_wl_poly(x, theta)

    def plot_phase_ambiguities(self, ax):

        phase_shift_excess_label = 'Measured $\Delta\phi$ (wrapped)'
        phase_shift_ambiguous_label = 'Measured $\Delta\phi$ (unwrapped, ambiguous)'

        ax.plot_raw(self.wl_shift, self.phase_shift, '.', color='lightblue', label=phase_shift_excess_label)
        ax.plot_raw(0, 0, '.', color='lightblue')

        # loop over raw_data points
        for i, wl_shift_i in enumerate(self.wl_shift):

            phase_shift_i = self.phase_shift[i]

            # loop over phase ambiguities
            for k, idx in enumerate(self.phase_shift_ambig_idx[i]):
                phase_shift_ambiguity = phase_shift_i + idx

                if i == 1 and k == 0:
                    ax.plot_raw(wl_shift_i, phase_shift_ambiguity, '.', fillstyle='none', color='#1f77b4',
                                label=phase_shift_ambiguous_label)
                else:
                    ax.plot_raw(wl_shift_i, phase_shift_ambiguity, '.', fillstyle='none', color='#1f77b4')

        return

    def plot_exact_fractions(self, ax, **kwargs):

        sellmeier_source_idx = 1

        gd0_axis = np.linspace(1300, 1500, 8000)
        phase_0 = self.phase_0
        biref_0, ne_0, no_0, kappa_0, _, _, _ = pycis.model.bbo_dispersion(self.wl_0, sellmeier_source_idx)

        phase_shift_model = np.zeros([len(self.wls), len(gd0_axis)])
        phase_shift_residual = np.zeros([len(self.wls), len(gd0_axis)])

        color_idx = np.linspace(0, 1, len(self.wls_reduced))

        for i in range(len(self.wls_reduced)):

            wl_i = self.wls_reduced[i]
            phase_shift_i = self.phase_shift[i]

            biref, n_e, n_o, kappa, _, _, _ = pycis.model.bbo_dispersion(wl_i, sellmeier_source_idx)

            phase_shift_model[i, :] = (gd0_axis / kappa_0) * (biref * self.wl_0 / (biref_0 * wl_i) - 1)
            phase_shift_residual[i, :] = abs(pycis.demod.wrap(phase_shift_i - phase_shift_model[i, :], units='fringes'))

            ax.plot_raw(gd0_axis, phase_shift_residual[i, :], color = plt.cm.cool(color_idx[i]), **kwargs)


        # ax_exacfrac.set_xlabel('Group Delay (waves)', fontsize=self.fsize)
        # ax_exacfrac.set_ylabel('Residual (fringes)', fontsize=self.fsize)
        # ax_exacfrac.set_title('Exact Fractions', fontsize=self.fsize)
        # leg = ax_exacfrac.legend(fontsize=self.fsize)
        # leg.draggable()



        return


    def plot(self):

        fsize = 15
        fig = plt.figure()
        ax = fig.add_subplot(111)

        phase_shift_excess_label = 'Measured $\Delta\phi$ (wrapped)'
        phase_shift_ambiguous_label = 'Measured $\Delta\phi$ (unwrapped, ambiguous)'
        phase_shift_bounds_label = 'Range of consideration (prior)'


        wl_axis_plot = np.linspace(self.wls[0], self.wls[-1], 50)
        wl_shift_axis_plot = wl_axis_plot - self.wl_0

        self.plot_phase_ambiguities(ax)
        # evaluate quantities on an axis for plotting

        phase_shift_axis, phase_shift_bounds = self._get_phase_shift_bounds(wl_axis_plot)

        biref, n_e, n_o, kappa, _, _, _ = pycis.model.bbo_array(wl_axis_plot, 0)

        ax.fill_between(wl_shift_axis_plot, phase_shift_bounds[0, :], phase_shift_bounds[1, :], color='peachpuff',
                             label=phase_shift_bounds_label)

        ax.set_ylabel(r'$\Delta\phi$ (fringes)', fontsize=fsize)
        ax.set_xlabel(r'$\Delta\lambda$ (m)', fontsize=fsize)
        leg = ax.legend(loc='lower left')
        leg.draggable()

        return
