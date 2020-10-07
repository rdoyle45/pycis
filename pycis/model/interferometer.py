import numpy as np
import xarray as xr
from numba import vectorize, f8
from pycis.tools import is_scalar
from pycis.model import calculate_dispersion

"""
Misc. conventions:
- Mueller matrices are xr.DataArrays with dimensions that include 'mueller_v' and 'mueller_h' (each with length = 4) 
- Stokes vectors are xr.DataArrays with dimensions that include 'stokes' (with length = 4)

"""


def mueller_product(mat1, mat2):
    """
    Mueller matrix product

    :param mat1: (xr.DataArray) Mueller matrix
    :param mat2: (xr.DataArray) can be Mueller matrix or Stokes vector
    :return: either a Mueller matrix or Stokes vector, depending on the dimensions of mat2

    """

    if 'mueller_v' in mat2.dims and 'mueller_h' in mat2.dims:
        mat2_i = mat2.rename({'mueller_h': 'mueller_i', 'mueller_v': 'mueller_h'})
        return mat1.dot(mat2_i, dims=('mueller_h', ), ).rename({'mueller_i': 'mueller_h'})

    elif 'stokes' in mat2.dims:
        mat2_i = mat2.rename({'stokes': 'mueller_h'})
        return mat1.dot(mat2_i, dims=('mueller_h', ), ).rename({'mueller_v': 'stokes'})

    else:
        raise Exception('input not understood')


def calculate_rot_matrix(angle):
    """
    general Mueller matrix for frame rotation (anti-clockwise from x-axis)

    :param angle: rotation angle [ rad ]
    :return:

    """

    angle2 = 2 * angle
    rot_mat = np.array([[1, 0, 0, 0],
                        [0, np.cos(angle2), np.sin(angle2), 0],
                        [0, -np.sin(angle2), np.cos(angle2), 0],
                        [0, 0, 0, 1]])
    return xr.DataArray(rot_mat, dims=('mueller_v', 'mueller_h'), )


class InterferometerComponent:
    """
    base class for CI interferometer component

    """

    def __init__(self, orientation, ):
        """
        :param orientation: orientation angle [ rad ]

        """

        self.orientation = orientation

    def orient(self, mat):
        """
        orient component
        
        :param mat: (xr.DataArray) Mueller matrix
        :return:

        """

        mat_i = mueller_product(mat, calculate_rot_matrix(self.orientation))
        return mueller_product(calculate_rot_matrix(-self.orientation), mat_i)

    def calculate_matrix(self, wl, inc_angle, azim_angle):
        raise NotImplementedError


class LinearPolariser(InterferometerComponent):
    """
    linear polariser

    """

    def __init__(self, orientation, tx_1=1, tx_2=0, ):
        """
        :param orientation: [ rad ] 0 aligns transmission axis to x-axis
        :param tx_1: transmission primary component. [0, 1] - defaults to 1
        :param tx_2: transmission secondary (orthogonal) component. [0, 1] - defaults to 0

        """
        super().__init__(orientation, )

        assert 0 <= tx_1 <= 1
        assert 0 <= tx_2 <= 1
        self.tx_1 = tx_1
        self.tx_2 = tx_2

    def calculate_matrix(self, wl, inc_angle, azim_angle):
        """
        Mueller matrix for ideal linear polariser

        :param wl: pass
        :param inc_angle: pass
        :param azim_angle: pass
        :return:
        """

        mat = 0.5 * np.array([[self.tx_2 ** 2 + self.tx_1 ** 2, self.tx_1 ** 2 - self.tx_2 ** 2, 0, 0],
                              [self.tx_1 ** 2 - self.tx_2 ** 2, self.tx_2 ** 2 + self.tx_1 ** 2, 0, 0],
                              [0, 0, 2 * self.tx_2 * self.tx_1, 0],
                              [0, 0, 0, 2 * self.tx_2 * self.tx_1]])
        return self.orient(xr.DataArray(mat, dims=('mueller_v', 'mueller_h'), ))


class BirefringentComponent(InterferometerComponent):
    """
    base class for CIS crystal components

    """

    def __init__(self, orientation, thickness, material='a-BBO', source=None, contrast=1, ):
        """
        :param thickness: [ m ]
        :type thickness: float

        :param material: string denoting crystal material
        :type material: str

        :param source: string denoting source of Sellmeier coefficients describing dispersion in the crystal. If
        blank, the default material source specified in pycis.model.dispersion
        :type material: str

        :param contrast: arbitrary contrast degradation factor for crystal, uniform contrast only for now.
        :type contrast: float
        """
        super().__init__(orientation, )

        self.thickness = thickness
        self.material = material
        self.source = source
        self.contrast = contrast

    def calculate_matrix(self, wl, inc_angle, azim_angle):
        """
        general Mueller matrix for a linear retarder
        
        :param wl: [ m ]
        :param inc_angle: [ rad ] 
        :param azim_angle: [ rad ]
        :return: 
        """

        phase = self.calculate_delay(wl, inc_angle, azim_angle)

        a1 = xr.ones_like(phase)
        a0 = xr.zeros_like(phase)
        c_cphase = self.contrast * np.cos(phase)
        c_sphase = self.contrast * np.sin(phase)

        mat = [[a1, a0, a0, a0],
               [a0, a1, a0, a0],
               [a0, a0, c_cphase, c_sphase],
               [a0, a0, -c_sphase, c_cphase]]
        mat = xr.combine_nested(mat, concat_dim=('mueller_v', 'mueller_h', ), )

        return self.orient(mat)

    def calculate_delay(self, wl, inc_angle, azim_angle, n_e=None, n_o=None):
        raise NotImplementedError


class UniaxialCrystal(BirefringentComponent):
    """
    Uniaxial crystal

    """

    def __init__(self, orientation, thickness, cut_angle, material='a-BBO', source=None, contrast=1, ):
        """
        
        :param cut_angle: [ rad ] angle between optic axis and crystal front face
        :type cut_angle: float

        """

        super().__init__(orientation, thickness, material=material, contrast=contrast, )
        self.cut_angle = cut_angle

    def calculate_delay(self, wavelength, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        calculate phase delay (in rad) due to uniaxial crystal.

        Veiras defines optical path difference as OPL_o - OPL_e ie. +ve phase indicates a delayed extraordinary
        ray

        :param wavelength: wavelength [ m ]
        :type wavelength: float or array-like (1-D)

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

        # if refractive indices have not been manually set, calculate
        if n_e is None and n_o is None:
            biref, n_e, n_o = calculate_dispersion(wavelength, self.material, source=self.source)

        args = [wavelength, inc_angle, azim_angle, n_e, n_o, self.cut_angle, self.thickness, ]
        return xr.apply_ufunc(_calculate_delay_uniaxial_crystal, *args, dask='allowed', )


class SavartPlate(BirefringentComponent):
    """
    Savart plate

    """

    def __init__(self, orientation, thickness, material='a-BBO', source=None, mode='francon', contrast=1,
                 clr_aperture=None):
        """
        :param mode: source for the equation for phase delay: 'francon' (approx.) or 'veiras' (exact)
        :type mode: string

        """
        super().__init__(orientation, thickness, material=material, source=source, contrast=contrast, )
        self.mode = mode

    def calculate_delay(self, wl, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        calculate phase delay (in rad) due to Savart plate.

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
                biref, n_e, n_o = calculate_dispersion(wl, self.material, source=self.source)

            a = 1 / n_e
            b = 1 / n_o

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

            phase = crystal_1.calculate_delay(wl, inc_angle, azim_angle1, n_e=n_e, n_o=n_o) - \
                    crystal_2.calculate_delay(wl, inc_angle, azim_angle2, n_e=n_e, n_o=n_o)

        else:
            raise Exception('invalid SavartPlate.mode')

        return phase


class QuarterWaveplate(BirefringentComponent):
    """
    Ideal quarter waveplate

    """

    def __init__(self, orientation, clr_aperture=None):
        """

        :param orientation:
        """

        thickness = 1.  # this value is arbitrary

        super().__init__(orientation, thickness, clr_aperture=clr_aperture)

    def calculate_delay(self, wl, inc_angle, azim_angle, n_e=None, n_o=None):
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

            ones_shape = np.ones([wl.shape[0], inc_angle.shape[0], inc_angle.shape[1]])

        else:
            raise Exception('unable to interpret inputs')

        return np.pi / 2 * ones_shape


class HalfWaveplate(BirefringentComponent):
    """
    Ideal half waveplate

    """

    def __init__(self, orientation, clr_aperture=None):
        """

        :param orientation:
        """

        thickness = 1.  # this value is arbitrary

        super().__init__(orientation, thickness, clr_aperture=clr_aperture)

    def calculate_delay(self, wl, inc_angle, azim_angle, n_e=None, n_o=None):
        """
        calculate phase delay due to ideal half waveplate

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

        return np.pi * ones_shape


@vectorize([f8(f8, f8, f8, f8, f8, f8, f8), ], nopython=True, fastmath=True, cache=True, )
def _calculate_delay_uniaxial_crystal(wavelength, inc_angle, azim_angle, n_e, n_o, cut_angle, thickness, ):
    s_inc_angle = np.sin(inc_angle)
    s_inc_angle_2 = s_inc_angle ** 2
    s_cut_angle_2 = np.sin(cut_angle) ** 2
    c_cut_angle_2 = np.cos(cut_angle) ** 2

    term_1 = np.sqrt(n_o ** 2 - s_inc_angle_2)

    term_2 = (n_o ** 2 - n_e ** 2) * \
             (np.sin(cut_angle) * np.cos(cut_angle) * np.cos(azim_angle) * s_inc_angle) / \
             (n_e ** 2 * s_cut_angle_2 + n_o ** 2 * c_cut_angle_2)

    term_3 = - n_o * np.sqrt(
        (n_e ** 2 * (n_e ** 2 * s_cut_angle_2 + n_o ** 2 * c_cut_angle_2)) -
        ((n_e ** 2 - (n_e ** 2 - n_o ** 2) * c_cut_angle_2 * np.sin(
            azim_angle) ** 2) * s_inc_angle_2)) / \
             (n_e ** 2 * s_cut_angle_2 + n_o ** 2 * c_cut_angle_2)

    return 2 * np.pi * (thickness / wavelength) * (term_1 + term_2 + term_3)

