import numpy as np
import xarray as xr

import pycis

def calculate_rot_mat(angle):
    """
    general Mueller matrix for frame rotation

    :param angle: rotation angle [ rad ]
    :return:

    """

    angle2 = 2 * angle

    rot_mat = np.array([[1, 0, 0, 0],
                     [0, np.cos(angle2), np.sin(angle2), 0],
                     [0, -np.sin(angle2), np.cos(angle2), 0],
                     [0, 0, 0, 1]])

    return xr.DataArray(rot_mat, dims=('mueller_v', 'mueller_h'))


class InterferometerComponent:
    """
    abstract base class for CIS interferometer components

    """

    def __init__(self, orientation, clr_aperture=None):
        """
        :param orientation: orientation angle [ rad ]

        :param clr_aperture: clear aperture of component [ m ]. If None, infinite aperture assumed.
        """

        self.orientation = orientation
        self.clear_aperture = clr_aperture

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
        calculate component's Mueller matrix

        """

        raise NotImplementedError


class LinearPolariser(InterferometerComponent):
    """
    linear polariser

    """

    def __init__(self, orientation, tx_1=1, tx_2=0, clr_aperture=None):
        """
        :param orientation: [ rad ] 0 aligns vertical polariser axis to vertical interferometer axis
        :param tx_1: transmission primary component. [0, 1] - defaults to 1
        :param tx_2: transmission secondary (orthogonal) component. [0, 1] - defaults to 0

        """
        super().__init__(orientation, clr_aperture=clr_aperture)

        assert 0 <= tx_1 <= 1
        assert 0 <= tx_2 <= 1
        self.tx_1 = tx_1
        self.tx_2 = tx_2

    def calculate_matrix(self, wl, inc_angle, azim_angle):
        """
        general Mueller matrix for a linear polariser. No dependence on wavelength / ray angles assumed.

        :return:

        """

        mat = 0.5 * np.array([[self.tx_2 ** 2 + self.tx_1 ** 2, self.tx_2 ** 2 - self.tx_1 ** 2, 0, 0],
                            [self.tx_2 ** 2 - self.tx_1 ** 2, self.tx_2 ** 2 + self.tx_1 ** 2, 0, 0],
                            [0, 0, 2 * self.tx_2 * self.tx_1, 0],
                            [0, 0, 0, 2 * self.tx_2 * self.tx_1]])

        mat = xr.DataArray(mat, dims=('mueller_v', 'mueller_h'))

        return self.orient(mat)


class BirefringentComponent(InterferometerComponent):
    """
    base class for CIS crystal components

    """

    def __init__(self, orientation, thickness, material='a-BBO', contrast=1, clr_aperture=None):
        """
        :param thickness: [ m ]
        :type thickness: float

        :param material: string denoting crystal material
        :type material: str

        :param contrast: arbitrary contrast degradation factor for crystal, uniform contrast only for now.
        :type contrast: float
        """
        super().__init__(orientation, clr_aperture=clr_aperture)

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

        phase = self.calculate_phase(wl, inc_angle, azim_angle)

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

    def calculate_phase(self, wavelength, ray_inc_angle, ray_azim_angle):
        """
        abstract method

        """
        raise NotImplementedError()


class UniaxialCrystal(BirefringentComponent):
    """
    Uniaxial crystal

    """

    def __init__(self, orientation, thickness, cut_angle, material='a-BBO', contrast=1, clr_aperture=None):
        """
        
        :param cut_angle: [ rad ] angle between optic axis and crystal front face
        :type cut_angle: float

        """

        super().__init__(orientation, thickness, material=material, contrast=contrast, clr_aperture=clr_aperture)
        self.cut_angle = cut_angle

    def calculate_phase(self, wavelength, ray_inc_angle, ray_azim_angle):
        """
        calculate phase delay due to uniaxial crystal.

        source: Francisco E Veiras, Liliana I Perez, and María T Garea. “Phase shift formulas in uniaxial media: an
        application to waveplates

        Veiras defines optical path difference as OPL_o - OPL_e ie. +ve phase indicates a delayed extraordinary
        ray

        :param wavelength:
        :type wavelength: xr.DataArray

        :param ray_inc_angle:
        :type ray_inc_angle: xr.DataArray

        :param ray_azim_angle:
        :type ray_azim_angle: xr.DataArray

        :return: phase [ rad ]
        """

        biref, n_e, n_o = pycis.dispersion(wavelength, self.material)
        s_inc_angle = np.sin(ray_inc_angle)

        term_1 = np.sqrt(n_o ** 2 - s_inc_angle ** 2)

        term_2 = (n_o ** 2 - n_e ** 2) * \
                 (np.sin(self.cut_angle) * np.cos(self.cut_angle) * np.cos(ray_azim_angle) * s_inc_angle) / \
                 (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)

        term_3 = - n_o * np.sqrt(
            (n_e ** 2 * (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)) -
            ((n_e ** 2 - (n_e ** 2 - n_o ** 2) * np.cos(self.cut_angle) ** 2 * np.sin(
                ray_azim_angle) ** 2) * s_inc_angle ** 2)) / \
                 (n_e ** 2 * np.sin(self.cut_angle) ** 2 + n_o ** 2 * np.cos(self.cut_angle) ** 2)

        phase = 2 * np.pi * (self.thickness / wavelength) * (term_1 + term_2 + term_3)

        return phase


class SavartPlate(BirefringentComponent):
    """
    Savart plate

    """

    def __init__(self, orientation, thickness, material='a-BBO', mode='francon', contrast=1, clr_aperture=None):
        """
        :param mode: source for the equation for phase delay: 'francon' (approx.) or 'veiras' (exact)
        :type mode: string

        """
        super().__init__(orientation, thickness, material=material, contrast=contrast, clr_aperture=clr_aperture)
        self.mode = mode

    def calculate_phase(self, wavelength, ray_inc_angle, ray_azim_angle):
        """
        calculate phase delay due to Savart plate.

        source: Lei Wu, Chunmin Zhang, and Baochang Zhao. “Analysis of the lateral displacement and optical path difference
        in wide-field-of-view polarization interference imaging spectrometer”. In: Optics Communications 273.1 (2007), 
        pp. 67–73. issn: 00304018. doi: 10.1016/j.optcom.2006.12.034.

        :param wavelength: wavelength [ m ]
        :type wavelength: xr.DataArray

        :param ray_inc_angle: ray incidence angle [ rad ]
        :type ray_inc_angle: xr.DataArray

        :param ray_azim_angle: ray azimuthal angle [ rad ]
        :type ray_azim_angle: xr.DataArray

        :return: phase [ rad ]

        """

        if self.mode == 'francon':

            biref, n_e, n_o = pycis.model.dispersion(wavelength, self.material)

            a = 1 / n_e
            b = 1 / n_o

            # precalculate trig fns
            c_azim_angle = np.cos(ray_azim_angle)
            s_azim_angle = np.sin(ray_azim_angle)
            s_inc_angle = np.sin(ray_inc_angle)

            # calculation
            term_1 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2)) * (c_azim_angle + s_azim_angle) * s_inc_angle

            term_2 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2) ** (3 / 2)) * ((a ** 2) / np.sqrt(2)) * \
                     (c_azim_angle ** 2 - s_azim_angle ** 2) * s_inc_angle ** 2

            # minus sign here makes the OPD calculation consistent with Veiras' definition
            phase = 2 * np.pi * - (self.thickness / (2 * wavelength)) * (term_1 + term_2)

        elif self.mode == 'veiras':
            # explicitly model plate as the combination of two uniaxial crystals

            or1 = self.orientation
            or2 = self.orientation - np.pi / 2

            azim_angle1 = ray_azim_angle
            azim_angle2 = ray_azim_angle - np.pi / 2
            t = self.thickness / 2

            crystal_1 = UniaxialCrystal(or1, t, cut_angle=-np.pi / 4, material=self.material)
            crystal_2 = UniaxialCrystal(or2, t, cut_angle=np.pi / 4, material=self.material)

            phase = crystal_1.calculate_phase(wavelength, ray_inc_angle, azim_angle1) - \
                    crystal_2.calculate_phase(wavelength, ray_inc_angle, azim_angle2)

        else:
            raise Exception('invalid SavartPlate.mode')

        return phase


class QuarterWaveplate(BirefringentComponent):
    """
    Idealised quarter waveplate

    """

    def __init__(self, orientation, clr_aperture=None):
        """

        :param orientation:
        """

        thickness = 1.  # this value is arbitrary

        super().__init__(orientation, thickness, clr_aperture=clr_aperture)

    def calculate_phase(self, wavelength, ray_inc_angle, ray_azim_angle):
        """
        calculate phase delay due to ideal quarter waveplate

        :param wl:
        :param inc_angle:
        :param azim_angle:
        :param n_e:
        :param n_o:
        :return: phase [ rad ]
        """

        phase = wavelength + ray_inc_angle + ray_azim_angle
        phase[:] = np.pi / 2

        return phase


# TODO class FieldWidenedSavartPlate(BirefringentComponent):




