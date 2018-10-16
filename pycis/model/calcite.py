import numpy as np


def calcite(wl):
    """ Get information on the dispersion of calcite.

    Source: 

    G. Ghosh. 
    
    Dispersion-equation coefficients for the refractive index and birefringence of calcite and quartz crystals
    
    Opt. Commun. 163, 95-102 (1999)

    """

    wl_micron = wl * 1.e6

    sell_coefs_e = [0.35859695, 0.82427830, 1.06689543e-2, 0.14429128, 120]
    sell_coefs_o = [0.73358749, 0.96464345, 1.94325203e-2, 1.82831454, 120]

    n_e = sellmeier_eqn(wl_micron, sell_coefs_e)
    n_o = sellmeier_eqn(wl_micron, sell_coefs_o)

    biref = n_e - n_o

    print(biref)

    return biref, n_e, n_o


def sellmeier_eqn(wl, sell_coeffs):
    a, b, c, d, e = sell_coeffs
    return np.sqrt(a + (b * wl ** 2 / (wl ** 2 - c)) + (d * wl ** 2 / (wl ** 2 - e)) + 1)