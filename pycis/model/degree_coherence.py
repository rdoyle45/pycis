import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import pycis


def calculate_degree_coherence(spectrum, wl_axis, material='a-BBO', delay_max=10e-3, npts=None, display=False):
    """
    calculate degree of coherence (DOC), given a wavelength spectrum.

    In general, DOC is a complex quantity. In the absence of dispersion, DOC is the Fourier transform of the (
    normalised) frequency spectrum. Since the spectrum is real, DOC will be an even function of interferometer delay
    -- so only positive delays are returned.

    In Fourier transform spectroscopy, measured DOC depends on dispersion present in the two-beam interferometer. In
    the case of CIS, the dispersion is determined by the choice of birefringent crystal material.

    :param spectrum: [ arb. ]
    :type spectrum: np.ndarray

    :param wl_axis: wavelengths [ m ]
    :type wl_axis: np.ndarray

    :param material:
    :type material: str

    :param delay_max: [ m ]
    :type delay_max: float

    :param display:
    :type display: bool

    :return: degree_coherence
    """

    assert len(spectrum) == len(wl_axis)

    if npts is None:
        npts = len(wl_axis)

    # interpolate spectrum onto uniform frequency axis
    wl_min, wl_max = np.min(wl_axis), np.max(wl_axis)
    freq_min, freq_max = c / wl_max, c / wl_min
    freq_axis = np.linspace(freq_min, freq_max, npts)
    biref_axis, _, _ = pycis.dispersion(c / freq_axis, material=material)
    spectrum_freq = np.interp(freq_axis, c / wl_axis[::-1], spectrum[::-1] * wl_axis ** 2 / c)

    # define interferometer delay axis and calculate DOC
    delay_axis = np.linspace(0, delay_max, npts)
    delay_mesh, freq_mesh = np.meshgrid(delay_axis, freq_axis)
    _, spectrum_freq_mesh = np.meshgrid(delay_axis, spectrum_freq)
    _, biref_mesh = np.meshgrid(delay_axis, biref_axis)
    integrand = spectrum_freq_mesh * np.exp(-2 * np.pi * 1j * delay_mesh * biref_mesh * freq_mesh / c)
    degree_coherence = np.trapz(integrand, freq_mesh, axis=0)

    if display:
        # calculate dispersion-free DOC (dfree) for visual comparison
        # spectrum's centre-of-mass freq denoted 0
        freq_0 = 1 / np.sum(spectrum_freq) * np.sum(spectrum_freq * freq_axis)
        biref_0, _, _ = pycis.dispersion(c / freq_0, material=material)
        integrand_dfree = spectrum_freq_mesh * np.exp(-2 * np.pi * 1j * delay_mesh * biref_0 * freq_mesh / c)
        degree_coherence_dfree = np.trapz(integrand_dfree, freq_axis, axis=0)

        # plot
        fig = plt.figure(figsize=(9, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        wl_axis_nm = wl_axis * 1e9
        wl_min_nm, wl_max_nm = wl_min * 1e9, wl_max * 1e9
        delay_axis_mm = delay_axis * 1e3
        delay_max_mm = delay_max * 1e3

        ax1.plot(wl_axis_nm, spectrum)

        ax2.plot(delay_axis_mm, np.real(degree_coherence), label='Re($\gamma$)')
        ax2.plot(delay_axis_mm, np.imag(degree_coherence), label='Im($\gamma$)')
        ax2.plot(delay_axis_mm, abs(degree_coherence_dfree), color='r', ls='--', label='$|\gamma|$ (dfree)')
        ax2.plot(delay_axis_mm, abs(degree_coherence), color='r', ls='--', label='$|\gamma|$')

        ax1.set_xlabel('wavelength (m)')
        ax1.set_title('spectrum')
        ax2.set_xlabel('delay (mm aBBO)')
        ax2.set_title('coherence')

        ax1.set_xlim([wl_min_nm, wl_max_nm])
        ax2.set_xlim([0, delay_max_mm])

        leg = ax2.legend()
        plt.show()

    return degree_coherence, delay_axis

#
# if __name__ == '__main__':
#
#     temp = 2  # [eV]
#     dens = 1e20  # [m-3]
#     bfield = 0.  # [T]
#     viewangle = 90 * np.pi / 180  # [rad]
#
#     # specify line
#     isotope = 'D'
#     n_upper = 5
#
#     bls = pystark.BalmerLineshape(n_upper, dens, temp, bfield, viewangle=viewangle, line_model='voigt')
#
#     spectrum = bls.ls_szd
#     wl_axis = bls.wl_axis
#
#     calculate_degree_coherence(spectrum, wl_axis, display=True, delay_max=30e-3)



