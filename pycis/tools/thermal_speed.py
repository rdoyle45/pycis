import numpy as np
# from pycis.tools.phys_constants import k_B, amu_kg, e
from scipy.constants import e, atomic_mass, k

mass_units = {'amu': atomic_mass, 'kg': 1}


def thermal_speed(t_ev, mass, mass_unit='amu'):
    """ calculate thermal speed.

    :param t_ev: species temperature in eV
    :param mass_no: species mass number
    :return:
    """

    print(e, k, atomic_mass)

    assert mass_unit in mass_units
    mass_unit = mass_units[mass_unit]

    mass_kg = mass * mass_unit
    t_kelvin = t_ev * (e / k)

    return np.sqrt(2 * k * t_kelvin / mass_kg)