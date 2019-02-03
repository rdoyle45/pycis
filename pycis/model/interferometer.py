import numpy as np
import pycis
from pycis.tools import is_scalar


def calculate_rot_mat(angle):
    """
    general Mueller matrix for frame rotation

    :param angle: rotation angle [ rad ]
    :return:

    """

    angle2 = 2 * angle
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(angle2), np.sin(angle2), 0],
                     [0, -np.sin(angle2), np.cos(angle2), 0],
                     [0, 0, 0, 1]])


class InterferometerComponent:
    """
    base class for CIS interferometer components

    """

    def __init__(self, orientation, clear_aperture=None):
        """
        :param orientation: orientation angle [ rad ]

        :param clear_aperture: clear aperture of component [ m ]. If None, infinite aperture assumed.
        """

        self.orientation = orientation
        self.clear_aperture = clear_aperture

    def orient(self, mat):
        """
        account for orientation of interferometer component with given vertical Mueller matrix
        
        :param mat: 4 x 4 Mueller matrix

        :return:

        """

        # matrix multiplication
        subscripts = 'ij...,jl...->il...'
        mat_rot = np.einsum(subscripts, calculate_rot_mat(-self.orientation), mat)
        return np.einsum(subscripts, mat_rot, calculate_rot_mat(self.orientation))

    def calculate_matrix(self, wl, inc_angle, azim_angle):
        """
        abstract method, calculate component's Mueller matrix

        """

        raise NotImplementedError


class LinearPolariser(InterferometerComponent):
    """
    linear polariser

    """

    def __init__(self, orientation, clear_aperture=None, tx_1=1, tx_2=0):
        """
        :param orientation: [ rad ] 0 aligns vertical polariser axis to vertical interferometer axis
        :param tx_1: transmission primary component. [0, 1] - defaults to 1
        :param tx_2: transmission secondary (orthogonal) component. [0, 1] - defaults to 0

        """
        super().__init__(orientation, clear_aperture=clear_aperture)

        assert 0 <= tx_1 <= 1
        assert 0 <= tx_2 <= 1
        self.tx_1 = tx_1
        self.tx_2 = tx_2

    def calculate_matrix(self, wl, inc_angle, azim_angle):
        """
        general Mueller matrix for a linear polariser. No dependence on wavelength / ray angles assumed.

        :return:

        """

        m = 0.5 * np.array([[self.tx_2 ** 2 + self.tx_1 ** 2, self.tx_2 ** 2 - self.tx_1 ** 2, 0, 0],
                            [self.tx_2 ** 2 - self.tx_1 ** 2, self.tx_2 ** 2 + self.tx_1 ** 2, 0, 0],
                            [0, 0, 2 * self.tx_2 * self.tx_1, 0],
                            [0, 0, 0, 2 * self.tx_2 * self.tx_1]])

        return self.orient(m)


class BirefringentComponent(InterferometerComponent):
    """
    base class for CIS crystal components

    """

    def __init__(self, orientation, thickness, clear_aperture=None, material='a-BBO', contrast=1):
        """
        :param thickness: [ m ]
        :type thickness: float

        :param material: string denoting crystal material
        :type material: str

        :param contrast: arbitrary contrast degradation factor for crystal, uniform contrast only for now.
        :type contrast: float
        """
        super().__init__(orientation, clear_aperture=clear_aperture)

        self.thickness = thickness
        self.material = material
        self.contrast = contrast

    def calculate_matrix(self, wl, inc_angle, azim_angle):
        """
        general Mueller matrix for a phase retarder
        
        :param wl: [ m ]
        :param inc_angle: [ rad ] 
        :param azim_angle: [ rad ]
        :return: 
        """

        phase = self.calculate_phase_delay(wl, inc_angle, azim_angle)

        # TODO can i get rid of this using broadcasting?

        a1 = np.ones_like(phase)
        a0 = np.zeros_like(phase)

        # precalculate trig fns
        c_cphase = self.contrast * np.cos(phase)
        c_sphase = self.contrast * np.sin(phase)

        m = np.array([[a1, a0, a0, a0],
                      [a0, a1, a0, a0],
                      [a0, a0, c_cphase, c_sphase],
                      [a0, a0, -c_sphase, c_cphase]])

        return self.orient(m)

    def calculate_phase_delay(self, wl, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        abstract method

        """

        raise NotImplementedError()


class UniaxialCrystal(BirefringentComponent):
    """
    Uniaxial crystal

    """

    def __init__(self, orientation, thickness, cut_angle, material='a-BBO', contrast=1):
        """
        
        :param cut_angle: [ rad ] angle between optic axis and crystal front face
        :type cut_angle: float

        """

        super().__init__(orientation, thickness, material, contrast)
        self.cut_angle = cut_angle

    def calculate_phase_delay(self, wl, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        calculate phase delay due to uniaxial crystal.

        Vectorised. If inc_angle and azim_angle are arrays, they must have the same dimensions.  
        source: Francisco E Veiras, Liliana I Perez, and María T Garea. “Phase shift formulas in uniaxial media: an 
        application to waveplates
        
        Veiras defines optical path difference as OPL_o - OPL_e ie. +ve phase indicates a delayed extraordinary 
        ray

        :param wl: wavelength [ m ]
        :type wl: float or array-like (1-D)

        :param inc_angle: ray incidence angle [ rad ]
        :type inc_angle: float or array-like (up to 2-D)

        :param azim_angle: ray azimuthal angle [ rad ]
        :type azim_angle: float or array-like (up to 2-D)

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
        if not is_scalar(wl) and not is_scalar(inc_angle) and not is_scalar(azim_angle):

            assert inc_angle.shape == azim_angle.shape

            if inc_angle.ndim == 1:
                # pad 1-D ray angle arrays

                inc_angle = inc_angle[:, np.newaxis]
                azim_angle = azim_angle[:, np.newaxis]

            # tile wl arrays to image dimensions for vectorisation
            reps = [1, inc_angle.shape[0], inc_angle.shape[1]]

            wl = np.tile(wl[:, np.newaxis, np.newaxis], reps)
            n_e = np.tile(n_e[:, np.newaxis, np.newaxis], reps)
            n_o = np.tile(n_o[:, np.newaxis, np.newaxis], reps)

        s_inc_angle = np.sin(inc_angle)

        term_1 = np.sqrt(n_o ** 2 - s_inc_angle ** 2)

        term_2 = (n_o ** 2 - n_e ** 2) * \
                 (np.sin(self.cut_angle) * np.cos(self.cut_angle) * np.cos(azim_angle) * s_inc_angle) / \
                 (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)

        term_3 = - n_o * np.sqrt(
            (n_e ** 2 * (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)) -
            ((n_e ** 2 - (n_e ** 2 - n_o ** 2) * np.cos(self.cut_angle) ** 2 * np.sin(
                azim_angle) ** 2) * s_inc_angle ** 2)) / \
                 (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)

        return 2 * np.pi * (self.thickness / wl) * (term_1 + term_2 + term_3)


class SavartPlate(BirefringentComponent):
    """
    Savart plate

    """

    def __init__(self, orientation, thickness, material='a-BBO', mode='francon', contrast=1):
        """
        :param mode: source for the equation for phase delay: 'francon' (approx.) or 'veiras' (exact)
        :type mode: string

        """
        super().__init__(orientation, thickness, material, contrast)
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

        :param inc_angle: ray incidence angle [ rad ]
        :type inc_angle: float or array-like

        :param azim_angle: ray azimuthal angle [ rad ]
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
            if not is_scalar(wl) and not is_scalar(inc_angle) and not is_scalar(azim_angle):

                assert inc_angle.shape == azim_angle.shape

                if inc_angle.ndim == 1:
                    # pad 1-D ray angle arrays

                    inc_angle = inc_angle[:, np.newaxis]
                    azim_angle = azim_angle[:, np.newaxis]

                # tile wl arrays to image dimensions for vectorisation
                reps = [1, inc_angle.shape[0], inc_angle.shape[1]]

                # tile wl arrays to image dimensions for vectorisation
                wl = np.tile(wl[:, np.newaxis, np.newaxis], reps)
                a = np.tile(a[:, np.newaxis, np.newaxis], reps)
                b = np.tile(b[:, np.newaxis, np.newaxis], reps)

            # precalculate trig fns
            c_azim_angle = np.cos(azim_angle)
            s_azim_angle = np.sin(azim_angle)
            s_inc_angle = np.sin(inc_angle)

            # calculation
            term_1 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2)) * (c_azim_angle + s_azim_angle) * s_inc_angle

            term_2 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2) ** (3 / 2)) * ((a ** 2) / np.sqrt(2)) * \
                     (c_azim_angle ** 2 - s_azim_angle ** 2) * s_inc_angle ** 2

            # minus sign here makes the OPD calculation consistent with Veiras' definition
            phase = 2 * np.pi * - (self.thickness / (2 * wl)) * (term_1 + term_2)

        elif self.mode == 'veiras':
            # explicitly model plate as the combination of two uniaxial crystals

            or1 = self.orientation
            or2 = self.orientation - np.pi / 2

            azim_angle1 = azim_angle
            azim_angle2 = azim_angle - np.pi / 2
            t = self.thickness / 2

            crystal_1 = UniaxialCrystal(or1, t, cut_angle=-np.pi / 4, material=self.material)
            crystal_2 = UniaxialCrystal(or2, t, cut_angle=np.pi / 4, material=self.material)

            phase = crystal_1.calculate_phase_delay(wl, inc_angle, azim_angle1, n_e=n_e, n_o=n_o) - \
                    crystal_2.calculate_phase_delay(wl, inc_angle, azim_angle2, n_e=n_e, n_o=n_o)

        else:
            raise Exception('invalid SavartPlate.mode')

        return phase


class QuarterWaveplate(BirefringentComponent):
    """
    Idealised quarter waveplate

    """

    def __init__(self, orientation):
        """

        :param orientation:
        """

        thickness = 1.  # this value is arbitrary

        super().__init__(orientation, thickness)


    def calculate_phase_delay(self, wl, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        calculate phase delay due to ideal quarter waveplate

        :param wl:
        :param inc_angle:
        :param azim_angle:
        :param n_e:
        :param n_o:
        :return: phase [ rad ]
        """

        if is_scalar(wl):
            if is_scalar(inc_angle) and is_scalar(azim_angle):
                ones_shape = 1

            else:
                assert isinstance(inc_angle, np.ndarray) and isinstance(azim_angle, np.ndarray)
                assert inc_angle.shape == azim_angle.shape

                ones_shape = np.ones_like(inc_angle)

        elif isinstance(wl, np.ndarray) and isinstance(inc_angle, np.ndarray) and isinstance(azim_angle, np.ndarray):

            assert wl.ndim == 1
            assert inc_angle.shape == azim_angle.shape

            ones_shape = np.ones(wl.shape[0], inc_angle.shape[0], inc_angle.shape[1])

        else:
            raise Exception('unable to interpret inputs')

        return np.pi / 2 * ones_shape


# TODO class FieldWidenedSavartPlate(BirefringentComponent):




