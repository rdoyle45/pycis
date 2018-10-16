import numpy as np


def yttrium_ortho_vanadate(wl):
    """ Get information on the dispersion of yttrium ortho vanadate.

    Source: 

    H. S. Shi, G. Zhang and H. Y. Shen. Measurement of principal refractive indices and the thermal refractive index coefficients of yttrium vanadate, J. Synthetic Cryst. 30, 85-88 (2001)
    
    and
    
    https://refractiveindex.info/?shelf=main&book=YVO4&page=Shi-o-20C
    
    """

    wl_micron = wl * 1.e6

    sell_coefs_e = [4.607200, 0.108087, 0.052495, 0.014305]
    sell_coefs_o = [3.778790, 0.070479, 0.045731, 0.009701]

    n_e = sellmeier_eqn(wl_micron, sell_coefs_e)
    n_o = sellmeier_eqn(wl_micron, sell_coefs_o)

    biref = n_e - n_o

    return biref, n_e, n_o


def sellmeier_eqn(wl, sell_coeffs):
    a, b, c, d = sell_coeffs
    return np.sqrt(a + (b / (wl ** 2 + c)) + d * wl ** 2)