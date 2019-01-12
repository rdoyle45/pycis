import numpy as np
import pickle
import pandas as pd

import os.path
import pycis
from pycis.tools.to_precision import to_precision
from scipy.constants import e, c, k, atomic_mass
from matplotlib import pyplot as plt


def get_lines(line_name):
    """Retrieves transition raw_data outputted from NIST database, converts from list of dictionaries to pandas Dataframe.
    
    line_name inputs supported:
    'CIII'
    'CII'
    'HeII'
    'Cd'
    'D_alpha'
    'D_beta'
    'D_gamma'
    'D_delta'
    """

    lines = pickle.load(open(os.path.join(pycis.paths.lines_path, line_name + '.p'), 'rb'))
    lines = pd.DataFrame(lines)  # [list of dicts] --> [pandas DataFrame]

    # lines = lines[~np.isnan(lines.wave_obs)]  # Delete all transitions not observed
    # lines = lines[~np.isnan(lines.wave_ritz)]

    # Account for any deletions in row indices
    line_no = len(lines)
    lines.index = range(0, line_no)

    lines.loc[:, 'zeeman_component'] = None

    # Normalise relative intensity to 1:
    # tot_int = sum(lines.loc[:, 'rel_int'])
    # lines.loc[:, 'rel_int'] /= tot_int

    return lines


def create_arbitrary_lines(name, lambda_0, rel_int, mass_number):
    """
    Creates and saves an arbitrary set of lines. Want extra functionality, to stitch different spectra together:
    
    :param line_name: str
    :param lambda_0: [m]
    :return: 
    """

    lambda_0 = np.array(lambda_0, dtype=np.float)
    rel_int = np.array(rel_int, dtype=np.float)
    mass_number = np.array(mass_number, dtype=np.float)

    no_lines = np.size(lambda_0)

    if no_lines != np.size(rel_int):
        raise Exception('length of lambda_0 and rel_int inputs must be the same.')

    # Normalise relative intensity to 1:
    if no_lines > 1:
        tot_int = sum(rel_int)
        rel_int /= tot_int

    lines = []
    blank_singlet = pickle.load(open(os.path.join(pycis.paths.lines_path,'blank_singlet.p'), 'rb'))

    if no_lines == 1:
        lines.append(dict(blank_singlet[0]))
        lines[0]['wave'] = lambda_0
        lines[0]['wave_obs'] = lambda_0
        lines[0]['wave_ritz'] = lambda_0
        lines[0]['rel_int'] = rel_int
        lines[0]['mass_number'] = mass_number
    else:
        for i in range(0, no_lines):
            # Manually create new lines for testing based on the Cd singlet list of dicts
            lines.append(dict(blank_singlet[0]))
            lines[i]['wave'] = lambda_0[i]
            lines[i]['wave_obs'] = lambda_0[i]
            lines[i]['wave_ritz'] = lambda_0[i]
            lines[i]['rel_int'] = rel_int[i]
            lines[i]['mass_number'] = mass_number[i]

    pickle.dump(lines, open(os.path.join(pycis.paths.lines_path, name + '.p'), 'wb'))
    return


def correct_dictionaries(line_name, mass_number):
    """
    single use to add mass number and to corect the wavlength units to metres for saved lists of dictionaries for 
    spectral lines
    :param line_name: 
    :param mass_number: 
    :return: 
    """

    lines = pickle.load(open(pycis.paths.lines_path + line_name + '.p', 'rb'))

    # now loop through list of dicts, adding an 'm_i' key for mass number and populating the value
    no_lines = len(lines)
    for i in range(0, no_lines):
        # convert wavelengths [A] --> [m]
        lines[i]['wave'] *= 1e-10
        lines[i]['wave_ritz'] *= 1e-10
        lines[i]['wave_obs'] *= 1e-10

        # add dictionary key for mass number
        lines[i]['mass_number'] = mass_number

    # save the corrected dictionary:
    pickle.dump(lines, open(os.path.join(pycis.paths.lines_path, line_name + '.p'), 'wb'))

    return


def create_blank_lineshape():
    """
    used once to create empty 'lines' dictionary which will be used to create arbitrary lineshapes.
    :param line_name: str
    :param lambda_0: [m]
    :return: 
    """

    # Hack to manually create new lines for testing based on the Cd singlet list of dicts
    singlet = pickle.load(open(os.path.join(pycis.paths.lines_path, 'Cd.p'), 'rb'))

    singlet[0] = dict.fromkeys(singlet[0], None)
    pickle.dump(singlet, open(os.path.join(pycis.paths.lines_path, 'blank_singlet' + '.p'), 'wb'))
    return


class Lineshape(object):
    """
    # pyCIS

    # Lineshape
    
    # By retrieving spectral line synth_data from the lines database, and then convolving a gaussian function at each
    # delta-function in wavelength, an idealised spectral line shape is produced. Lineshape output has n wavelength bins and
    # is normalised such that the integral over all wavelengths =I0. Relative line intensities are calculated assuming that
    # the levels are populated according to their statistical weights. To save on space, lineshape stored as delta functions
    # actual lineshape is created to specified wavelength resolution upon calling the make() method.
    #
    # Details of the lines synth_data fields can be found at:
    #
    # http://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html
    #
    # Input parameters are:
    #
    # line_name: string name of the line in question eg. 'CIII'
    # I0: Line intensity (integral over intensity spectrum I(nu)) [ photons / pixel area / camera integration step ]
    # vi: line-of-sight ion velocity [m/s]
    # Ti: ion temperature [eV]
    # B: magnetic field strength [T]
    # theta: angle between B-field and line-of-sight [rad]
    
    # Currently assumes weak-field, anomalous Zeeman splitting of line config. This is only valid when the energy level
    # perturbation is small compared to the multiplet structure separation of the line.
    
    # jsallcock
    # created: 16/02/2017

    
    """
    def __init__(self, name, I0, vi, Ti, b_field=0, theta=0):

        #TODO - need 'stationary lines' rather than raw lines (ie. move after accounting for zeeman splitting)

        self.name = name
        self.raw_lines = get_lines(name)  # retrieve line raw_data
        self.m_i = self.raw_lines.loc[0, 'mass_number'] * atomic_mass

        self.I0 = I0
        self.vi = vi  # m/s
        self.Ti = Ti  # ti
        self.b_field = b_field
        self.theta = theta

        # If relevant atomic state information is available in 'raw_lines' dictionary, calculate relative line intensities by
        # statistical weight:
        have_atomic_state_info = None not in (self.raw_lines['Aki'].tolist() + self.raw_lines['gi'].tolist() + self.raw_lines['gk'].tolist())
        if have_atomic_state_info:
            self._relative_intensity_from_statistical_weights()

        # Normalise sum of relative intensities to 1:
        norm_factor = self.raw_lines.loc[:, 'rel_int'].sum()
        self.raw_lines.loc[:, 'rel_int'] /= norm_factor

        # create copy of lines DataFrame to create lineshape, which will then be discarded.
        lines = self.raw_lines.copy(deep=True)
        line_no = len(lines)

        # determine Doppler shifting and Zeeman splitting:
        if (b_field > 0) and have_atomic_state_info:
            lines = self._zeeman_split(lines)

        doppler_shift = lambda wl: wl / (1 - (vi / c))  # positive velocity away from observer
        lines.loc[:, 'wave'] = doppler_shift(lines.loc[:, 'wave'])  # [m]
        lines.loc[:, 'wave_obs'] = doppler_shift(lines.loc[:, 'wave_obs'])  # [m]
        lines.loc[:, 'wave_ritz'] = doppler_shift(lines.loc[:, 'wave_ritz'])  # [m]

        # calculate sigma + FWHM for each line component:
        sigma_nu = np.zeros([line_no])
        sigma_lambda = np.zeros([line_no])
        fwhm_nu = np.zeros([line_no])
        fwhm_lambda = np.zeros([line_no])
        for i in range(0, line_no):
            nu_i = c / lines.loc[i, 'wave']  # [Hz]
            sigma_nu[i] = nu_i * (((e * Ti) / ((c ** 2) * self.m_i)) ** 0.5)  # [Hz]  (k_B * Ti * (e / k_B) is [eV] --> [K])
            sigma_lambda[i] = abs((c / nu_i) - (c / (nu_i + sigma_nu[i])))  # [m]
            fwhm_nu[i] = 2 * ((2 * np.log(2)) ** 0.5) * sigma_nu[i]
            fwhm_lambda[i] = 2 * ((2 * np.log(2)) ** 0.5) * sigma_lambda[i]

        # now add to lines Dataframe:
        lines['sigma_nu'] = pd.Series(sigma_nu, index=lines.index)
        lines['sigma_lambda'] = pd.Series(sigma_lambda, index=lines.index)
        lines['fwhm_nu'] = pd.Series(fwhm_nu, index=lines.index)
        lines['fwhm_lambda'] = pd.Series(fwhm_lambda, index=lines.index)

        # assign relevant attributes to Lineshape class instance (perhaps a cumbersome way of structuring this?)
        self.lines = lines
        self.line_no = line_no



    # methods:
    def make(self, n):
        """ Construct the lineshape, with a resolution of n points per line. """

        # method for constructing the lineshape

        # Dynamic sampling for wavelength axis:

        # loop over the lines
        # for each set of two lines, does the sum of the standard deviations divided by 2 exceed the absolute wavelength difference
        # between them? If it does then they are sampled independently of one another. If not then they are sampled together.

        wlaxis = np.array([])
        n_sigma = 5  # number of standard deviations from the mean over which Gaussian is sampled
        i = 0  # i is the number of lines sampled

        while i < self.line_no:
            if i < self.line_no - 1:
                broadening_factor_i = (self.lines['sigma_lambda'].iloc[i] + self.lines['sigma_lambda'].iloc[i + 1])
                wavelength_difference_i = self.lines['wave'].iloc[i + 1] - self.lines['wave'].iloc[i]

                if broadening_factor_i > wavelength_difference_i:
                    sigma_max_i = np.max(np.array([self.lines['sigma_lambda'].iloc[i], self.lines['sigma_lambda'].iloc[i+1]]))
                    wlaxis_lbound_i = self.lines['wave'].iloc[i] - (n_sigma * sigma_max_i)
                    wlaxis_ubound_i = self.lines['wave'].iloc[i+1] + (n_sigma * sigma_max_i)
                    wlaxis_i = np.linspace(wlaxis_lbound_i, wlaxis_ubound_i, n)
                    wlaxis = np.concatenate([wlaxis, wlaxis_i])


                    i += 2
                else:
                    wlaxis_lbound_i = self.lines['wave'].iloc[i] - (n_sigma * self.lines['sigma_lambda'].iloc[i])
                    wlaxis_ubound_i = self.lines['wave'].iloc[i] + (n_sigma * self.lines['sigma_lambda'].iloc[i])
                    wlaxis_i = np.linspace(wlaxis_lbound_i, wlaxis_ubound_i, n)
                    wlaxis = np.concatenate([wlaxis, wlaxis_i])
                    i += 1
            else:
                wlaxis_lbound_i = self.lines['wave'].iloc[i] - (n_sigma * self.lines['sigma_lambda'].iloc[i])
                wlaxis_ubound_i = self.lines['wave'].iloc[i] + (n_sigma * self.lines['sigma_lambda'].iloc[i])
                wlaxis_i = np.linspace(wlaxis_lbound_i, wlaxis_ubound_i, n)
                wlaxis = np.concatenate([wlaxis, wlaxis_i])
                i += 1

        wlaxis = np.sort(wlaxis)
        nuaxis = c / wlaxis[::-1]


        # Generate normalised lineshape:
        lineshape_nu = np.zeros_like(wlaxis)
        lineshape_lambda = np.zeros_like(wlaxis)
        line_no = len(self.lines)
        v_th = ((2 * e * self.Ti) / (self.m_i)) ** 0.5  # [m/s]

        # loop through lines, adding a weighted gaussian function at each line.
        for i in range(0, line_no):
            lambda_i = self.lines['wave'].iloc[i]
            nu_i = c / lambda_i
            lineshape_nu_i = self.lines.rel_int[i] * (nu_i ** -1) * ((np.pi * (v_th ** 2 / (c ** 2))) ** -0.5) * np.exp(-((((nuaxis - nu_i) ** 2) * (c ** 2)) / (v_th ** 2 * (nu_i ** 2))))
            # Converting between spectral intensities [photons / Hz] --> [photons / m] requires some care to conserve number of photons:
            lineshape_lambda_i = (c / (wlaxis ** 2)) * lineshape_nu_i[::-1]

            lineshape_nu += lineshape_nu_i
            lineshape_lambda += lineshape_lambda_i

        # Calculate moments of the spectral distribution:
        norm_factor_nu = np.trapz(lineshape_nu, nuaxis)
        nu_0 = (np.trapz(lineshape_nu * nuaxis, nuaxis)) / (norm_factor_nu)

        norm_factor_lambda = np.trapz(lineshape_lambda, wlaxis)
        lambda_0 = (np.trapz(lineshape_lambda * wlaxis, wlaxis)) / (norm_factor_lambda)

        # Normalise such that integral over spectrum = I0:
        lineshape_nu = (lineshape_nu * self.I0) / (norm_factor_nu)
        lineshape_lambda = (lineshape_lambda * self.I0) / (norm_factor_lambda)

        return lineshape_lambda, wlaxis, self.I0, lambda_0

    def plot(self, n):
        plt.close()
        self._plot(n)
        plt.show()
        return

    def print(self):
        line_no = len(self.lines)
        print('plotted lines at:')
        for i in range(0, line_no):
            ps_lambda = to_precision(str(1e9 * (self.lines['wave'].iloc[i])), 6)
            ps_rel_int = to_precision(str(self.lines['rel_int'].iloc[i]), 5)
            print("{}. {}nm,  rel_int: {}, zeeman component: {}".format(i + 1, ps_lambda, ps_rel_int, self.lines['zeeman_component'].iloc[i]))
        return

    def print_raw(self):
        line_no = len(self.raw_lines)
        print('plotted lines at:')
        for i in range(0, line_no):
            ps_lambda = to_precision(str(1e9 * (self.raw_lines.wave[i])), 6)
            ps_rel_int = to_precision(str(self.raw_lines.rel_int[i]), 5)
            print("{}. {}nm,  rel_int: {}, zeeman component: {}".format(i + 1, ps_lambda, ps_rel_int, self.raw_lines.zeeman_component[i]))
        return
    
    def save_fig(self, savepath, savename, saveformat='png', norm=False):
        self._plot(norm)
        plt.savefig(savepath + savename + '.' + saveformat, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()
        return

    def save(self):
        pickle.dump(self, open(os.path.join(pycis.paths.lines_path, self.name + '.p'), 'wb'))
        return

    # Internal Methods:

    def _relative_intensity_from_statistical_weights(self):
        """
        Calculate relative intensities of the multiplet, assuming states are populated by statistical weight
        
        :return: 
        """

        line_no_raw = len(self.raw_lines)

        # Extract info from 0th line in dictionary:
        self.raw_lines.loc[0, 'rel_int'] = 1
        Aki_0 = self.raw_lines.loc[0, 'Aki']
        gk_0 = self.raw_lines.loc[0, 'gk']
        lambda_0 = self.raw_lines.loc[0, 'wave']

        # Loop over lines, calculating intensity relative to 0th line:
        for ind in range(1, line_no_raw):
            Aki_ind = self.raw_lines.loc[ind, 'Aki']
            gk_ind = self.raw_lines.loc[ind, 'gk']
            lambda_ind = self.raw_lines.loc[ind, 'wave']
            self.raw_lines.loc[ind, 'rel_int'] = self.raw_lines.loc[0, 'rel_int'] * (
                (Aki_ind * gk_ind * lambda_0) / (Aki_0 * gk_0 * lambda_ind))

        return

    def _zeeman_split(self, lines):
        """
        
        Calculates wavelengths & intensities of zeeman split line config for
        ions described by L-S coupling (weak field anomolous zeeman effect).
        
        :param lines: 
        :return: 
        """

        # Empty dataframe for output:
        zeeman_lines = pd.DataFrame()
        line_no = len(lines)

        # Loop over each input transition:
        for i in range(0, line_no):
            # total angular momentum QN j of upper and lower states:
            print(lines.loc[i, 'upper_J'])
            J_us = float(eval(lines.loc[i, 'upper_J']))  # eval() necessary e.g. in the J = '3/2' case
            J_ls = float(eval(lines.loc[i, 'lower_J']))

            # empty dataframe output for the split lines of raw transition i:
            zeeman_lines_i = pd.DataFrame()

            # Loop over total angular momentum projections mJ of upper and lower transition levels:
            for mJ_us in np.linspace(-1 * J_us, J_us,
                                     (2 * J_us) + 1):  # (2*J_us) + 1 is multiplicity of energy level
                for mJ_ls in np.linspace(-1 * J_ls, J_ls, (2 * J_ls) + 1):
                    delta_J = J_us - J_ls
                    delta_mJ = mJ_us - mJ_ls
                    # make a line if we satisfy the selection rule:
                    if abs(delta_mJ) <= 1:

                        E_us = lines.loc[i, 'Ek']  # [eV]
                        E_ls = lines.loc[i, 'Ei']  # [eV]
                        g_us = lines.loc[i, 'gk']
                        g_ls = lines.loc[i, 'gi']

                        # zeeman perturbed energy states:
                        E_us = (E_us + (mJ_us * muB * self.b_field * g_us))  # [eV]
                        E_ls = (E_ls + (mJ_ls * muB * self.b_field * g_ls))  # [eV]
                        wave = (h * c / ((E_us - E_ls) * e))  # [m]

                        # Now calculate relative line intensities for the zeeman fine structure:
                        # Delta J = 1 transition:
                        if delta_J == 1:
                            # If Sigma +
                            if delta_mJ == 1:
                                zeeman_component = 'sigma+'
                                rel_int = 0.25 * (J_us + mJ_us) * (J_us - 1 + mJ_us)
                                delta_mJ = 1
                            # If Sigma -
                            elif delta_mJ == -1:
                                zeeman_component = 'sigma-'
                                rel_int = 0.25 * (J_us - mJ_us) * (J_us - 1 - mJ_us)
                                delta_mJ = -1
                            # If Pi
                            elif delta_mJ == 0:
                                zeeman_component = 'pi'
                                rel_int = J_us ** 2 - mJ_us ** 2
                                delta_mJ = 0

                        # Delta J = -1 transition:
                        elif delta_J == -1:
                            # If Sigma +
                            if delta_mJ == 1:
                                zeeman_component = 'sigma+'
                                rel_int = 0.25 * (J_us + 1 - mJ_us) * (J_us + 2 - mJ_us)
                                delta_mJ = 1
                            # If Sigma -
                            elif delta_mJ == -1:
                                zeeman_component = 'sigma-'
                                rel_int = 0.25 * (J_us + mJ_us + 1) * (J_us + 2 + mJ_us)
                                delta_mJ = -1
                            # If Pi
                            elif delta_mJ == 0:
                                zeeman_component = 'pi'
                                rel_int = (J_us + 1) ** 2 - mJ_us ** 2
                                delta_mJ = -1

                        # Delta J = 0 transition:
                        elif delta_J == 0:
                            # If Sigma +
                            if delta_mJ == 1:
                                zeeman_component = 'sigma+'
                                rel_int = 0.25 * (J_us + mJ_us) * (J_us + 1 - mJ_us)
                                delta_mJ = 1
                            # If Sigma -
                            elif delta_mJ == -1:
                                zeeman_component = 'sigma-'
                                rel_int = 0.25 * (J_us - mJ_us) * (J_us + 1 + mJ_us)
                                delta_mJ = -1
                            # If Pi
                            elif delta_mJ == 0:
                                zeeman_component = 'pi'
                                rel_int = mJ_us ** 2
                                delta_mJ = 0

                        # Now append this to the dataframe:
                        # Take a copy of the transition in lines, and modify the relevant fields.
                        this_line = lines.loc[i, :].copy()
                        this_line.loc['Ei'] = E_us
                        this_line.loc['Ek'] = E_ls
                        this_line.loc['wave'] = wave
                        this_line.loc['rel_int'] = rel_int
                        # Add some new columns with zeeman info: component (pi+, pi-, sigma), delta_mJ:
                        this_line['delta_mJ'] = delta_mJ
                        this_line['zeeman_component'] = zeeman_component

                        # Add this pd.Series to the output DataFrame:
                        zeeman_lines_i = zeeman_lines_i.append(this_line)

            # Normalise total rel_int to 1 for this transition:
            zeeman_line_i_no = len(zeeman_lines_i)
            zeeman_lines_i.index = range(0, zeeman_line_i_no)  # ensure dataframe indices in order
            if zeeman_line_i_no != 0:
                norm_factor = zeeman_lines_i.rel_int.sum()
                if norm_factor > 0:
                    zeeman_lines_i.loc[:, 'rel_int'] /= norm_factor

                    zeeman_lines_i.loc[:, 'rel_int'] *= lines.loc[i, 'rel_int']

            # Remove any components with zero intensity
            for j in range(0, zeeman_line_i_no):
                if zeeman_lines_i.loc[j, 'rel_int'] < 0.001:
                    zeeman_lines_i = zeeman_lines_i.drop(j)
            zeeman_line_i_no = len(zeeman_lines_i)
            zeeman_lines_i.index = range(0, zeeman_line_i_no)  # ensure dataframe indices in order

            # append the split lines of this transition to the output dataframe:
            zeeman_lines = zeeman_lines.append(zeeman_lines_i)

        # reorder indices by ascending wavelength:
        zeeman_lines = zeeman_lines.sort('wave', ascending=True)
        zeeman_line_no = len(zeeman_lines)
        zeeman_lines.index = range(zeeman_line_no)

        # adjust relative intensities of pi / sigma components due to viewing angle:
        if self.b_field > 0:
            for k in range(0, zeeman_line_no):
                if zeeman_lines.loc[k, 'zeeman_component'] is 'pi':
                    zeeman_lines.loc[k, 'rel_int'] *= np.sin(self.theta) ** 2
                elif zeeman_lines.loc[k, 'zeeman_component'] is ('sigma-' or 'sigma+'):
                    zeeman_lines.loc[k, 'rel_int'] *= (1 + np.cos(self.theta) ** 2)

        # Normalise total rel_int to 1 across all transitions:
        if zeeman_line_no != 0:
            norm_factor = zeeman_lines.rel_int.sum()
            if norm_factor > 0:
                zeeman_lines.loc[:, 'rel_int'] /= norm_factor

        return zeeman_lines

    def _plot(self, n):
        """
        tools plotting method:
        
        :param n: 
        :return: 
        """
        lineshape, wlaxis, I0, lambda_0 = self.make(n)

        # plot complete lineshape:
        plt.figure(figsize=(15, 8))
        plt.plot((wlaxis * 1e9), lineshape, 'b', linewidth=1.25)
        # plt.ylabel('Spectral Intensity [photons / m]')
        plt.ylabel('Intensity')
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        plt.xlabel('Wavelength [nm]')
        # plt.grid()

        line_no = len(self.lines)
        for i in range(0, line_no):
            alpha = self.lines.loc[i, 'rel_int'] / np.max(self.lines.loc[:, 'rel_int'])
            if i == 0:
                # plot vertical lines to indicate shifted and unshifted line centres:
                plt.axvline(x=(self.lines.wave[i] * 1e9), ls='-', color='r', linewidth=1.5, alpha=alpha, label='Doppler-shifted Line-centres')
                plt.axvline(x=(self.raw_lines.wave[i] * 1e9), ls=':', color='r', linewidth=1.5, alpha=alpha, label='Unshifted Line-centres')

                # plot constituent gaussian lineshapes:
                plt.plot(wlaxis * 1e9, lineshape)

            # else:
                # plt.axvline(x=(self.lines.wave[i] * 1e9), ls='-', color='r', linewidth=1.5, alpha=alpha)
                # plt.axvline(x=(self.raw_lines.wave[i] * 1e9), ls=':', color='r', linewidth=1.5, alpha=alpha)

        # Vertical line to indicate C.O.M. wavelength:
        # plt.axvline(x=(lambda_0 * 1e9), ls='-', color='k', linewidth=2, label='Doppler-shifted: C.O.M. = ' + to_precision(lambda_0 * 1e9, 6) +'nm')

        # plt.legend(prop={'size': 15}, loc=0)
        return


if __name__ == '__main__':
    # ls = Lineshape(line_name='HeII', I0=1e19, vi=10000, Ti=1, B=0, theta=0)
    # ls.plot(1000)

    create_arbitrary_lines('Cd_508nm', 508.58e-9, 1, 112)

    # wlaxis = np.linspace(464e-9, 465e-9, 20)
    # I = 1e5 * np.ones_like(wlaxis)
    # inc_angle = 0.
    # azim_angle_wp = 0.
    # azim_angle_sp = 0.
    # L_wp = 5e-3
    # L_sp = 5e-3
    # m_i = 12.
    # T_i = 5.
    # nu_0 = 3e8 / 464.5e-9
    # lines_wl = np.array([465.5])
    # lines_rel_int = np.array([1.])
    #
    # param = pycis.model.synth_pixel_calib(wlaxis, I, inc_angle, azim_angle_wp, azim_angle_sp, L_wp, L_sp, m_i, T_i, nu_0, lines_wl, lines_rel_int)
    # print(param)
