from __future__ import division
import numpy as np
cimport numpy as np
from cpython cimport array

from libc.math cimport sqrt
cimport cython

DTYPE = np.float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

cdef double d_lambda = 1.e-8
cdef double d_lambda_micron = d_lambda * 1.e6

cdef double *sell_coefs_Ae = [2.31197, 2.3753, 2.37153, 2.3753, 2.3730]
cdef double *sell_coefs_Ao = [2.67579, 2.7471, 2.7471, 2.7359, 2.7405]

cdef double *sell_coefs_Be = [0.01184, 0.01224, 0.01224, 0.01224, 0.0128]
cdef double *sell_coefs_Bo = [0.02099, 0.01878, 0.01878, 0.01878, 0.0184]

cdef double *sell_coefs_Ce = [-0.01607, -0.01667,  -0.01667, -0.01667, -0.0156]
cdef double *sell_coefs_Co = [-0.00470, -0.01822, -0.01822, -0.01822, -0.0179]

cdef double *sell_coefs_De = [-0.00400, -0.01516, -0.01516, -0.01516, -0.0044]
cdef double *sell_coefs_Do = [-0.00528, -0.01354, -0.01354, -0.01354, -0.0155]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef bbo_array(np.ndarray[DTYPE_t, ndim=1] wl, int source):
    """
    :param wl: 
    :param source: 
    :return: 
    """

    cdef int wl_length = wl.shape[0]
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] biref = np.zeros(wl_length, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] n_e = np.zeros(wl_length, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] n_o = np.zeros(wl_length, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] kappa = np.zeros(wl_length, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] dBdlambda = np.zeros(wl_length, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] d2Bdlambda2 = np.zeros(wl_length, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] d3Bdlambda3 = np.zeros(wl_length, dtype=DTYPE)

    for i in range(wl_length):
        biref[i], n_e[i], n_o[i], kappa[i], dBdlambda[i], d2Bdlambda2[i], d3Bdlambda3[i] = bbo_dispersion(wl[i], source)

    return biref, n_e, n_o, kappa, dBdlambda, d2Bdlambda2, d3Bdlambda3



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef bbo_dispersion(double wl, int source):
    """
    Barium borate (BBO) optical properties over a given wavelength range.
    
    The Sellmeier model is used to calculate the BBO refractive indices, with multiple coefficient sources available:  
    
    - alpha-BBO coefficients:
       - ** 0. 'newlightphotonics.com':** measured in-house at http://www.newlightphotonics.com/Birefringent-Crystals/alpha-BBO-Crystals
       - **'1. agoptics.com':** (supposedly!) from the literature, found at http://www.agoptics.com/Alpha-BBO.html
       - **'2. kim':** from Kim, Seong-Yeol, et al. "LONGITUDINAL ELECTRON BUNCH SHAPING EXPERIMENTS AT THE PAL-ITF."
    
    - beta-BBO coefficients:
       - **'3. kato_1986':** from Kato, K. "Second-harmonic generation to 2048 Å in Β-Ba2O4." IEEE journal of quantum electronics 22.7 (1986): 1013-1014.
       - **'kato_2010':** from Kato, K., N. Umemura, and T. Mikami. "Sellmeier and thermo-optic dispersion formulas for β-BaB2O4 (revisited)." Proc. SPIE. Vol. 7582. 2010.
       - **'4. eimerl':** from Eimerl, David, et al. "Optical, mechanical, and thermal properties of barium borate." Journal of applied physics 62.5 (1987): 1968-1983.
       - **'chen':** unpublished, but mentioned in eimerl 1987 paper.
    
    :param wl: wavelength number or array in [m]
    :type wl: array_like
    :param source: Specify Sellmeier coefficient source.
    :type source: int.
    
    :return: birefringence (B), extraordinary refractive index (n_e), ordinary refractive index (n_o), first-order dispersion parameter (kappa), 1st derivative of B wrt. wavelength (dBdlambda) and 2nd derivative of B wrt. wavelength (d2Bdlambda2) for the wavelength (range) specified.
    """

    cdef double *sell_coefs_e = [sell_coefs_Ae[source], sell_coefs_Be[source], sell_coefs_Ce[source], sell_coefs_De[source]]
    cdef double *sell_coefs_o = [sell_coefs_Ao[source], sell_coefs_Bo[source], sell_coefs_Co[source], sell_coefs_Do[source]]

    cdef double n_e, n_o, biref, dBdlambda, d2Bdlambda2, d3Bdlambda3, kappa
    cdef double wl_mic = wl * 1.e6

    # Sellmeier equation for calculation of refractive indices, birefringence

    biref, n_e, n_o = bbo(wl, source)

    # birefringence derivatives wrt. wavelength

    # first symmetric derivative of birefringence wrt. wavelength
    dBdlambda = (biref_dif(wl_mic, 1., sell_coefs_e, sell_coefs_o) - biref_dif(wl_mic, -1., sell_coefs_e, sell_coefs_o)) / (2 * d_lambda)

    # second symmetric derivative of birefringence wrt. wavelength
    d2Bdlambda2 = (biref_dif(wl_mic, 1., sell_coefs_e, sell_coefs_o) - (2 * biref) + biref_dif(wl_mic, -1., sell_coefs_e, sell_coefs_o)) / (d_lambda ** 2)

    # third symmetric derivative of birefringence wrt. wavelength
    d3Bdlambda3 = (biref_dif(wl_mic, 3./2., sell_coefs_e, sell_coefs_o) - (3. * biref_dif(wl_mic, 1./2., sell_coefs_e, sell_coefs_o)) + (3. * biref_dif(wl_mic, -1./2., sell_coefs_e, sell_coefs_o)) - biref_dif(wl_mic, -3./2., sell_coefs_e, sell_coefs_o)) / (d_lambda ** 3)

    # first order dispersion factor kappa:
    kappa = 1 - (wl / biref) * dBdlambda

    return biref, n_e, n_o, kappa, dBdlambda, d2Bdlambda2, d3Bdlambda3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef bbo(double wl, int source):
    """
    Barium borate (BBO) optical properties over a given wavelength range.
    
    The Sellmeier model is used to calculate the BBO refractive indices, with multiple coefficient sources available:  
    
    - alpha-BBO coefficients:
       - ** 0. 'newlightphotonics.com':** measured in-house at http://www.newlightphotonics.com/Birefringent-Crystals/alpha-BBO-Crystals
       - **'1. agoptics.com':** (supposedly!) from the literature, found at http://www.agoptics.com/Alpha-BBO.html
       - **'2. kim':** from Kim, Seong-Yeol, et al. "LONGITUDINAL ELECTRON BUNCH SHAPING EXPERIMENTS AT THE PAL-ITF."
    
    - beta-BBO coefficients:
       - **'3. kato_1986':** from Kato, K. "Second-harmonic generation to 2048 Å in Β-Ba2O4." IEEE journal of quantum electronics 22.7 (1986): 1013-1014.
       - **'kato_2010':** from Kato, K., N. Umemura, and T. Mikami. "Sellmeier and thermo-optic dispersion formulas for β-BaB2O4 (revisited)." Proc. SPIE. Vol. 7582. 2010.
       - **'4. eimerl':** from Eimerl, David, et al. "Optical, mechanical, and thermal properties of barium borate." Journal of applied physics 62.5 (1987): 1968-1983.
       - **'chen':** unpublished, but mentioned in eimerl 1987 paper.
    
    :param wl: wavelength number or array in [m]
    :type wl: array_like
    :param source: Specify Sellmeier coefficient source.
    :type source: int.
    
    :return: birefringence (B), extraordinary refractive index (n_e), ordinary refractive index (n_o), first-order dispersion parameter (kappa), 1st derivative of B wrt. wavelength (dBdlambda) and 2nd derivative of B wrt. wavelength (d2Bdlambda2) for the wavelength (range) specified.
    """

    cdef double wl_micron = wl * 1.e6

    cdef double *sell_coefs_e = [sell_coefs_Ae[source], sell_coefs_Be[source], sell_coefs_Ce[source], sell_coefs_De[source]]
    cdef double n_e = sellmeier_equation(wl_micron, sell_coefs_e)

    cdef double *sell_coefs_o = [sell_coefs_Ao[source], sell_coefs_Bo[source], sell_coefs_Co[source], sell_coefs_Do[source]]
    cdef double n_o = sellmeier_equation(wl_micron, sell_coefs_o)

    cdef double biref = n_e - n_o

    return biref, n_e, n_o



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double sellmeier_equation(double wl_mic, double *sell_coefs):
    return (sell_coefs[0] + (sell_coefs[1] / ((wl_mic ** 2) + sell_coefs[2])) + (sell_coefs[3] * (wl_mic ** 2))) ** 0.5



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double biref_dif(double wavelength_mic, double dif_coef, double *sell_coefs_e, double *sell_coefs_o):
    """ 
    birefringence at some product of d_lambda distance from wavelength. 
    """

    cdef double wavelength_dif = wavelength_mic + (dif_coef * d_lambda_micron)
    return sellmeier_equation(wavelength_dif, sell_coefs_e) - sellmeier_equation(wavelength_dif, sell_coefs_o)



