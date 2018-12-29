import numpy as np
import pycis


def uniaxial_crystal(wl, thickness, inc_angle, azim_angle, cut_angle=0., material='a-BBO', n_e=None, n_o=None):
    """
    calculate phase delay due to uniaxial crystal.
    
    Vectorised. If inc_angle and azim_angle are arrays, they must have the same dimensions.  
    source: Francisco E Veiras, Liliana I Perez, and María T Garea. “Phase shift formulas in uniaxial media: an 
    application to waveplates”

    :param wl: wavelength [ m ]
    :type wl: float or array-like
    
    :param thickness: [ m ]
    :type thickness: float
    
    :param inc_angle: incidence angle [ rad ]
    :type inc_angle: float or array-like
    
    :param azim_angle: azimuthal angle [ rad ]
    :type azim_angle: float or array-like
    
    :param cut_angle: angle between crystal optic axis and crystal front face [ rad ]
    :type cut_angle: float
    
    :param material:
    :type material: string
    
    :param n_e: manually set extraordinary refractive index (for fitting)
    :type n_e: float
    
    :param n_o: manually set ordinary refractive index (for fitting)
    :type n_o: float

    :return: phase [ rad ]
    """

    # if refractive indices have not been manually set, calculate them using Sellmeier eqn.
    if n_e and n_o == None:
        biref, n_e, n_o = pycis.model.dispersion(wl, material)
    else:
        assert pycis.tools.safe_len(n_e) == pycis.tools.safe_len(n_o) == pycis.tools.safe_len(wl)

    # if wl, theta and omega are arrays, vectorise
    if not pycis.tools.is_scalar(wl) and not pycis.tools.is_scalar(inc_angle) and not pycis.tools.is_scalar(azim_angle):

        img_dim = inc_angle.shape
        assert img_dim == azim_angle.shape

        wl_length = len(wl)

        reps_img = [wl_length, 1, 1]
        reps_axis = [img_dim[0], img_dim[1], 1]

        wl = np.tile(wl, reps_axis)
        n_e = np.tile(n_e, reps_axis)
        n_o = np.tile(n_o, reps_axis)

        inc_angle = np.moveaxis(np.tile(inc_angle, reps_img), 0, -1)
        azim_angle = np.moveaxis(np.tile(azim_angle, reps_img), 0, -1)

    # calculation
    term_1 = np.sqrt(n_o ** 2 - np.sin(inc_angle) ** 2)

    term_2 = (n_o ** 2 - n_e ** 2) * \
             (np.sin(cut_angle) * np.cos(cut_angle) * np.cos(azim_angle) * np.sin(inc_angle)) / \
             (n_e ** 2 * np.sin(cut_angle) ** 2 + n_o ** 2 * np.cos(cut_angle) ** 2)

    term_3 = - n_o * np.sqrt(
        (n_e ** 2 * (n_e ** 2 * np.sin(cut_angle) ** 2 + n_o ** 2 * np.cos(cut_angle) ** 2)) -
        ((n_e ** 2 - (n_e ** 2 - n_o ** 2) * np.cos(cut_angle) ** 2 * np.sin(
            azim_angle) ** 2) * np.sin(inc_angle) ** 2)) / \
             (n_e ** 2 * np.sin(cut_angle) ** 2 + n_o ** 2 * np.cos(cut_angle) ** 2)

    return 2 * np.pi * (thickness / wl) * (term_1 + term_2 + term_3)


def savart_plate(wl, thickness, inc_angle, azim_angle, material='a-BBO', mode='francon', n_e=None, n_o=None):
    """
    calculate phase delay due to Savart plate.
    
    Vectorised. If inc_angle and azim_angle are arrays, they must have the same dimensions.  
    source: Lei Wu, Chunmin Zhang, and Baochang Zhao. “Analysis of the lateral displacement and optical path difference
    in wide-field-of-view polarization interference imaging spectrometer”. In: Optics Communications 273.1 (2007), 
    pp. 67–73. issn: 00304018. doi: 10.1016/j.optcom.2006.12.034.
    
    :param wl: wavelength [ m ]
    :type wl: float or array-like
    
    :param thickness: [ m ]
    :type thickness: float
    
    :param inc_angle: incidence angle [ rad ]
    :type inc_angle: float or array-like
    
    :param azim_angle: azimuthal angle [ rad ]
    :type azim_angle: float or array-like
    
    :param material: 
    :type material: string
    
    :param mode: source for the equation for phase delay: 'wu' or 'veiras'
    :type mode: string
    
    :param n_e: manually set extraordinary refractive index (for fitting)
    :type n_e: float
    
    :param n_o: manually set ordinary refractive index (for fitting)
    :type n_o: float
    
    :return: phase [ rad ]
    """

    if mode == 'francon':

        # if refractive indices have not been manually set, calculate them using Sellmeier eqn.
        if n_e and n_o == None:
            biref, n_e, n_o = pycis.model.dispersion(wl, material)
        else:
            assert pycis.tools.safe_len(n_e) == pycis.tools.safe_len(n_o) == pycis.tools.safe_len(wl)

        a = 1 / n_e
        b = 1 / n_o

        # if wl, theta and omega are arrays, vectorise
        if not pycis.tools.is_scalar(wl) and not pycis.tools.is_scalar(inc_angle) and not pycis.tools.is_scalar(azim_angle):

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
        term_1 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2)) * (np.cos(azim_angle) + np.sin(azim_angle)) * np.sin(inc_angle)

        term_2 = ((a ** 2 - b ** 2) / (a ** 2 + b ** 2) ** (3 / 2)) * ((a ** 2) / np.sqrt(2)) * \
                 (np.cos(azim_angle) ** 2 - np.sin(azim_angle) ** 2) * np.sin(inc_angle) ** 2

        phase = 2 * np.pi * (thickness / (2 * wl)) * (term_1 + term_2)

    elif mode == 'veiras':

        omega_1 = azim_angle
        omega_2 = azim_angle - (np.pi / 2)
        t = thickness / 2

        phase = uniaxial_crystal(wl, t, inc_angle, omega_1, cut_angle=-np.pi / 4, n_e=n_e, n_o=n_o) - \
                uniaxial_crystal(wl, t, inc_angle, omega_2, cut_angle=np.pi / 4, n_e=n_e, n_o=n_o)

    else:
        raise Exception('invalid mode')

    return phase

