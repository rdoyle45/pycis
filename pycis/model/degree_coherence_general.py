import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import pycis


def degree_coherence_general(lineshape, wavelengths, display=False):
    """
    Given a spectral lineshape, calculate the degree of coherence. It is important here to account for crystal
    dispersion, which reduces the contrast envelope.

    Currently assumes that delay is a-BBO waveplate at normal incidence -- may need to be more general.

    :param lineshape:
    :param wavelengths:
    :param display:
    :return: degree_coherence
    """

    # TODO this needs updating badly

    npts = len(wavelengths)
    assert len(lineshape) == npts

    # interpolate onto uniform frequency axis
    wl_min, wl_max = np.min(wavelengths), np.max(wavelengths)
    freq_min, freq_max = c / wl_max, c / wl_min

    freq_axis = np.linspace(freq_min, freq_max, npts)
    ls_freq = np.interp(freq_axis, c / wavelengths[::-1], lineshape[::-1] * wavelengths ** 2 / c)

    # estimate line fwhm
    # fwhm_ls = pycis.tools.get_fwhm(freq_axis, ls_freq)
    # fwhm_ls = (freq_max - freq_min) / 5
    #
    # wl_avg = np.mean(wavelengths)
    # biref_avg, _, _ = pycis.model.bbo(wl_avg, 1)
    #
    # fwhm_doc = abs((1 / fwhm_ls) * c / biref_avg)
    #
    # abbo_axis_lim = fwhm_doc * 5
    # abbo_axis_lim_lower_bound = 20e-3
    # abbo_axis_lim_upper_bound = 100e-3
    #
    # if abbo_axis_lim < abbo_axis_lim_lower_bound:
    #     abbo_axis_lim = abbo_axis_lim_lower_bound
    # elif abbo_axis_lim > abbo_axis_lim_upper_bound:
    #     abbo_axis_lim = abbo_axis_lim_upper_bound
    #
    abbo_axis_lim = 20e-3

    abbo_axis = np.linspace(0, abbo_axis_lim, npts)
    biref_axis, _, _, _, _, _, _ = pycis.model.bbo_array(c / freq_axis, 1)

    abbo_mesh, freq_mesh = np.meshgrid(abbo_axis, freq_axis)
    abbo_mesh, biref_mesh = np.meshgrid(abbo_axis, biref_axis)
    abbo_mesh, ls_freq_mesh = np.meshgrid(abbo_axis, ls_freq)

    integrand = ls_freq_mesh * np.exp(-2 * np.pi * 1j * abbo_mesh * biref_mesh * freq_mesh / c)
    degree_coherence = np.trapz(integrand, freq_mesh, axis=0)

    if display:
        fig = plt.figure(figsize=(9, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        wavelengths_nm = wavelengths * 1e9
        wl_min_nm, wl_max_nm = wl_min * 1e9, wl_max * 1e9
        abbo_axis_mm = abbo_axis * 1e3
        abbo_axis_lim_mm = abbo_axis_lim * 1e3

        ax1.plot(wavelengths_nm, lineshape)

        ax2.plot(abbo_axis_mm, np.real(degree_coherence), label='real()')
        ax2.plot(abbo_axis_mm, np.imag(degree_coherence), label='imag()')
        ax2.plot(abbo_axis_mm, abs(degree_coherence), color='r', ls='--', label='abs()')

        ax1.set_xlabel('wavelength (m)')
        ax1.set_title('spectrum')
        ax2.set_xlabel('delay (mm aBBO)')
        ax2.set_title('coherence')

        ax1.set_xlim([wl_min_nm, wl_max_nm])
        ax2.set_xlim([0, abbo_axis_lim_mm])

        leg = ax2.legend()
        plt.show()

    return degree_coherence, abbo_axis