import numpy as np
import pycis


def apply_phase_correction(wl_plasma, wl_calib, phase_calib_img):
    """ Apply wavelength corrections to the measured calibration phase image. 
    
    :param wl_plasma: 
    :param wl_calib: 
    :param phase_calib_img: 
    :return: 
    """

    roi_dim = (100, 100)

    phase_calib_img = pycis.demod.unwrap(phase_calib_img)

    # split into shape and offset components
    phase_calib_offs = np.mean(pycis.tools.get_roi(phase_calib_img, roi_dim=roi_dim))
    phase_calib_shape = phase_calib_img - phase_calib_offs

    biref_plasma, n_e_plasma, n_o_plasma, _, _, _, _ = pycis.model.bbo_dispersion(wl_plasma, 1)
    biref_calib, n_e_calib, n_o_calib, kappa_calib, _, _, _ = pycis.model.bbo_dispersion(wl_calib, 1)

    # phase shape extrapolation
    coeff_a = (n_o_calib ** 2 + n_e_calib ** 2) / (n_o_calib ** 2 - n_e_calib ** 2)
    coeff_b = (n_o_plasma ** 2 - n_e_plasma ** 2) / (n_o_plasma ** 2 + n_e_plasma ** 2)

    coeff = coeff_a * coeff_b

    phase_plasma_shape = (wl_calib / wl_plasma) * coeff * phase_calib_shape

    # phase offset extrapolation
    group_delay_calib = 2 * np.pi * 1381
    phase_shift = -(group_delay_calib / kappa_calib) * (((biref_plasma * wl_calib) / (biref_calib * wl_plasma)) - 1)


    # print(phase_shift / (2 * np.pi), 'fringes')
    # print(phase_calib_offs, 'rad')

    phase_plasma = pycis.demod.wrap(phase_plasma_shape + phase_calib_offs + phase_shift, units='rad')

    return phase_plasma


