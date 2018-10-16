import numpy as np


def lithium_niobate(wl):
    """ Get information on the dispersion of lithium niobate.
    
    Source: 
    
    Infrared corrected Sellmeier coefficients for congruently grown lithium niobate 
    and 5 mol. % magnesium oxide–doped lithium niobate
    
    D.E. Zelmon, D. L. Small
    J. Opt. Soc. Am. B/Vol. 14, No. 12/December 1997
    
    """

    wl_micron = wl * 1.e6

    sell_coefs_e = [2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08]
    sell_coefs_o = [2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.6]

    n_e = sellmeier_eqn(wl_micron, sell_coefs_e)
    n_o = sellmeier_eqn(wl_micron, sell_coefs_o)

    biref = n_e - n_o

    return biref, n_e, n_o


def sellmeier_eqn(wl, sell_coeffs):
    a, b, c, d, e, f = sell_coeffs
    return np.sqrt((a * wl ** 2 / (wl ** 2 - b)) + (c * wl ** 2 / (wl ** 2 - d)) + (e * wl ** 2 / (wl ** 2 - f)) + 1)