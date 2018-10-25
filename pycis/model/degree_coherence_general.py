import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import pycis



def degree_coherence_general(lineshape, wavelengths, display=False):
    """ Given a spectral lineshape, calculate the degree of coherence. It is important here to account for crystal 
    dispersion, which acts to reduce the contrast envelope. """

    npts = len(wavelengths)

    # interpolate onto uniform frequency axis
    freq_axis = np.linspace(c / np.max(wavelengths), c / np.min(wavelengths), len(wavelengths))
    ls_freq = np.interp(freq_axis, c / wavelengths[::-1], lineshape[::-1] * wavelengths ** 2 / c)

    # estimate line fwhm
    fwhm_ls = pycis.tools.get_fwhm(freq_axis, ls_freq)

    wl_avg = np.mean(wavelengths)
    biref_avg, _, _ = pycis.model.bbo(wl_avg, 1)

    fwhm_doc = abs((1/ fwhm_ls) * c / biref_avg)

    abbo_axis = np.linspace(-5 * fwhm_doc, 5 * fwhm_doc, npts)
    biref_axis, _, _, _, _, _, _ = pycis.model.bbo_array(c / freq_axis, 1)

    abbo_mesh, freq_mesh = np.meshgrid(abbo_axis, freq_axis)
    abbo_mesh, biref_mesh = np.meshgrid(abbo_axis, biref_axis)
    abbo_mesh, ls_freq_mesh = np.meshgrid(abbo_axis, ls_freq)

    integrand = ls_freq_mesh * np.exp(-2 * np.pi * 1j * abbo_mesh * biref_mesh * freq_mesh / c)
    degree_coherence = np.trapz(integrand, freq_mesh, axis=0)


    if display:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(wavelengths, lineshape)
        ax2.plot(abbo_axis, np.real(degree_coherence))
        ax2.plot(abbo_axis, np.imag(degree_coherence))

        ax2.plot(abbo_axis, abs(degree_coherence), color='r', ls='--')

        plt.show()

    return degree_coherence, abbo_axis