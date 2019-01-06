import numpy as np
import pycis


class InterferometerComponent:
    """ base class for CIS interferometer components """

    def __init__(self, orientation):
        """
        :param orientation: [ rad ]
        """

        self.orientation = orientation

    @staticmethod
    def calculate_rot_mat(angle):
        """
        generate matrix for frame rotation
        
        :param angle: [ rad ]
        :return: 
        """
        return np.array([[1, 0, 0, 0],
                         [0, np.cos(angle), np.sin(angle), 0],
                         [0, -np.sin(angle), np.cos(angle), 0],
                         [0, 0, 0, 1]])

    def orient(self, mat):
        """
        account for orientation of interferometer component with given vertical Mueller matrix
        :param mueller_mat: 
        :return: 
        """

        fmt = 'ij...,jl...->il...'
        mat_rot = np.einsum(fmt, self.calculate_rot_mat(-self.orientation), mat)

        return np.einsum(fmt, mat_rot, self.calculate_rot_mat(self.orientation))


class LinearPolariser(InterferometerComponent):
    """ linear polariser (vertical) """

    def __init__(self, orientation):
        super().__init__(orientation)


    def calculate_mueller_mat(self):

        m = 0.5 * np.array([[1, -1, 0, 0],
                            [-1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])

        return self.orient(m)


class BirefringentComponent(InterferometerComponent):
    """ base class for CIS crystal components """

    def __init__(self, orientation, thickness, material='a-BBO'):
        super().__init__(orientation)

        self.thickness = thickness
        self.material = material

    def calculate_mueller_mat(self, wl, inc_angle, azim_angle):

        phase = self.calculate_phase_delay(wl, inc_angle, azim_angle)

        # TODO CHECK if this expression is applicable in the case of Savart plates, A. Thorman thesis suggests not.

        # padding <- can i get rid of this using broadcasting?
        a1 = np.ones_like(phase)
        a0 = np.zeros_like(phase)

        m = np.array([[a1, a0, a0, a0],
                      [a0, a1, a0, a0],
                      [a0, a0, np.cos(phase), np.sin(phase)],
                      [a0, a0, -np.sin(phase), np.cos(phase)]])

        return self.orient(m)

    def calculate_phase_delay(self, wl, inc_angle, azim_angle, n_e=None, n_o=None):
        raise NotImplementedError()


class UniaxialCrystal(BirefringentComponent):
    """ general uniaxial crystal """

    def __init__(self, orientation, thickness, cut_angle, material='a-BBO'):
        """
        :param thickness: [ m ]
        :type thickness: float
        
        :param cut_angle: [ rad ] angle between optic axis and crystal front face
        :type cut_angle: float
        
        :param material:
        :type material: string
        
        :param name:
        :type name: string 
        """

        super().__init__(orientation, thickness, material)
        self.cut_angle = cut_angle

    def calculate_phase_delay(self, wl, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        calculate phase delay due to uniaxial crystal.

        Vectorised. If inc_angle and azim_angle are arrays, they must have the same dimensions.  
        source: Francisco E Veiras, Liliana I Perez, and María T Garea. “Phase shift formulas in uniaxial media: an 
        application to waveplates”

        :param wl: wavelength [ m ]
        :type wl: float or array-like

        :param inc_angle: incidence angle [ rad ]
        :type inc_angle: float or array-like

        :param azim_angle: azimuthal angle [ rad ]
        :type azim_angle: float or array-like

        :param n_e: manually set extraordinary refractive index (for fitting)
        :type n_e: float

        :param n_o: manually set ordinary refractive index (for fitting)
        :type n_o: float

        :return: phase [ rad ]
        """

        # if refractive indices have not been manually set, calculate them using Sellmeier eqn.
        if n_e is None and n_o is None:
            biref, n_e, n_o = pycis.model.dispersion(wl, self.material)
        else:
            assert pycis.tools.safe_len(n_e) == pycis.tools.safe_len(n_o) == pycis.tools.safe_len(wl)

        # if wl, theta and omega are arrays, vectorise
        if not pycis.tools.is_scalar(wl) and not pycis.tools.is_scalar(inc_angle) and not pycis.tools.is_scalar(
                azim_angle):
            img_dim = inc_angle.shape
            assert img_dim == azim_angle.shape

            wl_len = len(wl)

            reps_img = [wl_len, 1, 1]
            reps_axis = [img_dim[0], img_dim[1], 1]

            wl = np.tile(wl, reps_axis)
            n_e = np.tile(n_e, reps_axis)
            n_o = np.tile(n_o, reps_axis)

            inc_angle = np.moveaxis(np.tile(inc_angle, reps_img), 0, -1)
            azim_angle = np.moveaxis(np.tile(azim_angle, reps_img), 0, -1)

        # calculation
        term_1 = np.sqrt(n_o ** 2 - np.sin(inc_angle) ** 2)

        term_2 = (n_o ** 2 - n_e ** 2) * \
                 (np.sin(self.cut_angle) * np.cos(self.cut_angle) * np.cos(azim_angle) * np.sin(inc_angle)) / \
                 (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)

        term_3 = - n_o * np.sqrt(
            (n_e ** 2 * (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)) -
            ((n_e ** 2 - (n_e ** 2 - n_o ** 2) * np.cos(self.cut_angle) ** 2 * np.sin(
                azim_angle) ** 2) * np.sin(inc_angle) ** 2)) / \
                 (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)

        return 2 * np.pi * (self.thickness / wl) * (term_1 + term_2 + term_3)


class SavartPlate(BirefringentComponent):
    """ Savart plate """

    def __init__(self, orientation, thickness, material='a-BBO', mode='francon'):
        """
        :param thickness: [ m ]
        :type thickness: float
        
        :param material:
        :type material: string
        
        :param mode: source for the equation for phase delay: 'francon' or 'veiras'
        :type mode: string
        
        :param name: 
        """
        super().__init__(orientation, thickness, material)
        self.mode = mode

    def calculate_phase_delay(self, wl, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        calculate phase delay due to Savart plate.

        Vectorised. If inc_angle and azim_angle are arrays, they must have the same dimensions.  
        source: Lei Wu, Chunmin Zhang, and Baochang Zhao. “Analysis of the lateral displacement and optical path difference
        in wide-field-of-view polarization interference imaging spectrometer”. In: Optics Communications 273.1 (2007), 
        pp. 67–73. issn: 00304018. doi: 10.1016/j.optcom.2006.12.034.

        :param wl: wavelength [ m ]
        :type wl: float or array-like

        :param inc_angle: incidence angle [ rad ]
        :type inc_angle: float or array-like

        :param azim_angle: azimuthal angle [ rad ]
        :type azim_angle: float or array-like

        :param n_e: manually set extraordinary refractive index (for fitting)
        :type n_e: float

        :param n_o: manually set ordinary refractive index (for fitting)
        :type n_o: float

        :return: phase [ rad ]
        """

        if self.mode == 'francon':

            # if refractive indices have not been manually set, calculate them using Sellmeier eqn.
            if n_e is None and n_o is None:
                biref, n_e, n_o = pycis.model.dispersion(wl, self.material)
            else:
                assert pycis.tools.safe_len(n_e) == pycis.tools.safe_len(n_o) == pycis.tools.safe_len(wl)

            a = 1 / n_e
            b = 1 / n_o

            # if wl, theta and omega are arrays, vectorise
            if not pycis.tools.is_scalar(wl) and not pycis.tools.is_scalar(inc_angle) and not pycis.tools.is_scalar(
                    azim_angle):
                img_dim = inc_angle.shape
                assert img_dim == azim_angle.shape

                wl_length = len(wl)

                reps_img = [wl_length, 1, 1]
                reps_axis = [img_dim[0], img_dim[1], 1]

                wl = np.tile(wl, reps_axis)
                a = np.tile(a, reps_axis)
                b = np.tile(b, reps_axis)

                inc_angle = np.moveaxis(np.tile(inc_angle, reps_img), 0, -1)
                azim_angle = np.moveaxis(np.tile(azim_angle, reps_img), 0, -1)

            # calculation
            term_1 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2)) * (np.cos(azim_angle) + np.sin(azim_angle)) * np.sin(
                inc_angle)

            term_2 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2) ** (3 / 2)) * ((a ** 2) / np.sqrt(2)) * \
                     (np.cos(azim_angle) ** 2 - np.sin(azim_angle) ** 2) * np.sin(inc_angle) ** 2

            # minus sign here makes the OPD calculation consistent with veiras
            phase = 2 * np.pi * - (self.thickness / (2 * wl)) * (term_1 + term_2)

        elif self.mode == 'veiras':
            # explicitly model Savart plate as the combination of two uniaxial crystals (1 & 2)

            azim_angle_1 = azim_angle
            azim_angle_2 = azim_angle - (np.pi / 2)
            t = self.thickness / 2

            crystal_1 = UniaxialCrystal(t, cut_angle=-np.pi / 4, material=self.material)
            crystal_2 = UniaxialCrystal(t, cut_angle=np.pi / 4, material=self.material)

            phase = crystal_1.calculate_phase_delay(wl, inc_angle, azim_angle_1, n_e=n_e, n_o=n_o) - \
                    crystal_2.calculate_phase_delay(wl, inc_angle, azim_angle_2, n_e=n_e, n_o=n_o)

        else:
            raise Exception('invalid mode')

        return phase






