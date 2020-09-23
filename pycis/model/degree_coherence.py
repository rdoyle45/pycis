import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.constants import c
import pycis


def calculate_degree_coherence(spectrum, delay, material=None, freq_com=None):
    """
    calculate the degree of (temporal) coherence (DOC) of a given intensity spectrum, at a given interferometer delay.

    In general, DOC is a complex quantity. In the absence of dispersion, DOC is the Fourier transform of the
    (area-normalised) frequency spectrum. Since the spectrum is real, DOC is an even function of interferometer delay.

    The presence of instrument dispersion breaks the Fourier transform relationship of the spectrum and the DOC, but
    the `group delay approximation' is a first-order

    :param spectrum: area-normalised spectrum, arbitrary (spectral) units
    :type spectrum: xr.DataArray with dim 'wavelength' in ( m ) or else dim 'frequency' in ( Hz )

    :param delay: interferometer delay [ rad ]. If delay does not have a 'wavelength' dim or a 'frequency' dim then it
    is assumed that the delay value(s) correspond to the centre-of-mass (COM) frequency of spectrum and the DOC is
    calculated for each delay using the group delay approximation. If, however, delay has either a 'wavelength' dim or a
    'frequency' dim, then the full dispersive integral is evaluated instead.
    :type delay: xr.DataArray

    :param material: if material is specified, dispersion will be accounted for to first order about the weighted mean
    frequency
    :type material: str

    :param freq_com: centre of mass frequency of spectrum, if it has already been calculated
    :type freq_com: xr.DataArray

    :return: degree_coherence
    """

    # if necessary, convert spectrum's wavelength (m) dim + coordinate to frequency (Hz)
    if 'wavelength' in spectrum.dims:
        spectrum = spectrum.copy(deep=True)
        spectrum = spectrum.rename({'wavelength': 'frequency'})
        spectrum['frequency'] = c / spectrum['frequency']
        spectrum = spectrum * c / spectrum['frequency'] ** 2

    # calculate centre of mass (c.o.m.) frequency if not supplied
    if freq_com is None:
        freq_com = (spectrum * spectrum['frequency']).integrate(dim='frequency') / \
                   spectrum.integrate(dim='frequency')

    if 'frequency' in delay.dims or 'wavelength' in delay.dims:
        # do the full dispersive calculation

        # if necessary, convert spectrum's wavelength (m) dim + coordinate to frequency (Hz)
        if 'wavelength' in delay.dims:
            delay = delay.copy(deep=True)
            delay = delay.rename({'wavelength': 'frequency'})
            delay['frequency'] = c / delay['frequency']

        delay_time = delay / (2 * np.pi * delay.frequency)
        delay_com = delay.interp({'frequency': freq_com, }, ).drop('frequency')
        integrand = spectrum * np.exp(2 * np.pi * 1j * delay_time * delay['frequency'])

    else:
        # assume that the delay values given correspond to the c.o.m. frequency
        if material is not None:
            kappa_0 = pycis.dispersion(c / freq_com, material=material, output_derivatives=True)[3]
        else:
            kappa_0 = 1

        delay_time = delay / (2 * np.pi * freq_com)
        freq_shift_norm = (spectrum['frequency'] - freq_com) / freq_com
        integrand = spectrum * np.exp(2 * np.pi * 1j * delay_time * freq_com * (1 + kappa_0 * freq_shift_norm))

    degree_coherence = integrand.integrate(dim='frequency')

    return degree_coherence


def test():
    """
    numerical / analytical test of calculate_degree_coherence() using a Gaussian lineshape

    :return:
    """
    import time

    # ANALYTICAL
    wl_0 = 464.8e-9
    material = 'a-BBO'
    kappa_0 = pycis.dispersion(wl_0, material, output_derivatives=True)[3]
    wl_sigma = 0.05e-9
    n_sigma = 5
    n_bins = 5000

    # generate spectrum in frequency-space
    freq_0 = c / wl_0
    freq_sigma = c / wl_0 ** 2 * wl_sigma
    freq = np.linspace(freq_0 - n_sigma * freq_sigma, freq_0 + n_sigma * freq_sigma, n_bins)
    freq = xr.DataArray(freq, dims=('frequency', ), coords=(freq, ), )
    wl = c / freq.values
    wl = xr.DataArray(wl, dims=('wavelength', ), coords=(wl, ), )

    spec_freq = 1 / (freq_sigma * np.sqrt(2 * np.pi)) * np.exp(- 1 / 2 * ((freq - freq_0) / freq_sigma) ** 2)

    delay_time_analytical = np.linspace(0, n_sigma / 5 * 1 / freq_sigma, n_bins)
    doc_analytical = np.exp(-2 * (np.pi * freq_sigma * delay_time_analytical * kappa_0) ** 2) * np.exp(2 * np.pi * 1j * freq_0 * delay_time_analytical)

    # NUMERICAL 1 -- tests group delay approximation
    # spec_wl = spec_freq * freq ** 2 / c
    # spec_wl = spec_wl.rename({'frequency': 'wavelength'})
    # spec_wl['wavelength'] = wl
    # spec_wl = spec_wl * c / spectrum['frequency'] ** 2

    lwps = np.array([4.48e-3,
                     6.35e-3,
                     9.79e-3, ])
    biref_0 = pycis.dispersion(wl_0, material, )[0]

    delay = 2 * np.pi * lwps * np.abs(biref_0) / wl_0  # (rad), arbitrary values

    delay_time_numerical = delay / (2 * np.pi * freq_0)
    delay = xr.DataArray(delay, dims=('delay', ), coords=(delay, ))
    s = time.time()
    doc_numerical_1 = calculate_degree_coherence(spec_freq, delay, material=material)
    print(doc_numerical_1)
    e = time.time()
    print('numerical_1:', e - s, 'seconds')

    doc_analytical_fine = np.exp(-2 * (np.pi * freq_sigma * delay_time_numerical * kappa_0) ** 2) * \
                          np.exp(2 * np.pi * 1j * freq_0 * delay_time_numerical)

    # NUMERICAL 2 -- tests full dispersion integral
    doc_numerical_2 = np.zeros(len(lwps), dtype=complex)
    s = time.time()
    for i, lwp in enumerate(lwps):
        birefs = pycis.dispersion(c / freq, material)[0]
        delays_n2 = 2 * np.pi * np.abs(birefs) * lwp / (c / freq)
        delays_n2 = xr.DataArray(delays_n2, dims=('wavelength', ), coords=(c / freq, ), )
        doc_numerical_2[i] = complex(calculate_degree_coherence(spec_freq, delays_n2, material=material))
    print(doc_numerical_2)
    e = time.time()
    print('numerical_2:', e - s, 'seconds')

    # # PLOT
    # fig = plt.figure(figsize=(10, 5,))
    # axes = [fig.add_subplot(1, 3, i) for i in [1, 2, 3, ]]
    # titles = ['Spectrum', 'Contrast', 'Phase', ]
    # for ax, title in zip(axes, titles):
    #     ax.set_title(title)
    #
    # ax1, ax2, ax3 = axes
    #
    # # spectrum
    # ax1.plot(freq, spec_freq_arr)
    #
    # # contrast
    # ax2.plot(delay_time_analytical, np.abs(doc_analytical), label='Analytical\n(Group delay approx.)')
    # ax2.plot(delay_time_numerical, np.abs(doc_numerical_1).values, lw=0, marker = '.', markeredgewidth=0.5,
    #          markeredgecolor='k', label='Numerical\n(Group delay approx.)', markersize=8)
    # ax2.plot(delay_time_numerical, np.abs(doc_numerical_2), lw=0, marker='d', markersize=12, fillstyle='none', label='Numerical (full)')
    # leg = ax2.legend(frameon=False, fontsize=7)
    #
    # #phase
    # ax3.plot(delay_time_numerical, np.angle(doc_analytical_fine), lw=0, marker='x', markersize=12, label='Analytical\n(Group delay approx.)')
    # ax3.plot(delay_time_numerical, np.angle(doc_numerical_1.values), lw=0, marker='.', markeredgewidth=0.5,
    #          markeredgecolor='k', markersize=8, label='Numerical\n(Group delay approx.)')
    # ax3.plot(delay_time_numerical, np.angle(doc_numerical_2), lw=0, marker='d', markersize=12, fillstyle='none',
    #          label='Numerical (full)')
    # leg = ax3.legend(frameon=False, fontsize=7)
    #
    # plt.show()


# LEGACY CODE #
def calculate_degree_coherence_full(spectrum, wl_axis, l_wp, material='a-BBO'):
    """

    :param spectrum:
    :param wl_axis:
    :return:
    """

    assert len(spectrum) == len(wl_axis)

    freq_axis = c / wl_axis
    spectrum_freq = spectrum * wl_axis ** 2 / c
    freq_com = np.trapz(freq_axis * spectrum_freq, freq_axis) / np.trapz(spectrum_freq, freq_axis)  # weighted mean
    biref = pycis.dispersion(wl_axis, material=material)[0]
    biref_com = pycis.dispersion(c / freq_com, material=material)[0]

    integrand = spectrum_freq * np.exp(-2 * np.pi * 1j * l_wp * biref * freq_axis / c)
    degree_coherence = np.trapz(integrand, freq_axis, axis=0)
    delay_com = 2 * np.pi * l_wp * biref_com * freq_com / c

    return degree_coherence, delay_com


def plot_degree_coherence(spectrum, wl_axis, delay_max=5e4, material=None, npts=None, axes=None, display=True):
    """
    basically the same as above but evaluated over many interferometer delays to make a nice plot

    :param spectrum: [ arb. ]
    :type spectrum: np.ndarray

    :param wl_axis: wavelengths [ m ]
    :type wl_axis: np.ndarray

    :param delay_max: [ rad ]
    :type delay_max: float

    :param material:
    :type material: str

    :param npts: number of points for the delay axis
    :type npts: int

    :return: degree_coherence
    """

    #TODO merge this with the above function, use xarray for max flexibility

    if npts is None:
        npts = len(wl_axis)

    freq_axis = c / wl_axis
    spectrum_freq = spectrum * wl_axis ** 2 / c
    freq_com = np.trapz(freq_axis * spectrum_freq, freq_axis) / np.trapz(spectrum_freq, freq_axis)  # weighted mean

    # account for interferometer dispersion if birefringent material is specified
    if material is not None:
        kappa = pycis.dispersion(c / freq_com, material=material, output_derivatives=True)[3]
    else:
        kappa = 1

    # define interferometer delay axis and calculate DOC
    delay_axis = np.linspace(0, delay_max, npts)
    delay_time_axis = delay_axis * kappa / (2 * np.pi * freq_com)  # [ s ]

    delay_time_mesh, freq_mesh = np.meshgrid(delay_time_axis, freq_axis)
    _, spectrum_freq_mesh = np.meshgrid(delay_axis, spectrum_freq)
    integrand = spectrum_freq_mesh * np.exp(-2 * np.pi * 1j * delay_time_mesh * freq_mesh)
    degree_coherence = np.trapz(integrand, freq_mesh, axis=0)

    # fwhm_stark = pystark.get_fwhm_stark('H', 5, 2, 5e20)
    # nu0 = c / pystark.get_wl_centre('H', 5, 2)
    # contrast_lz = np.exp(-(fwhm_stark / nu0) / 2 * delay_axis * kappa)

    # plot
    if display:
        if axes is None:
            fig = plt.figure(figsize=(9, 4))
            axes = (fig.add_subplot(121), fig.add_subplot(122))

        ax1, ax2 = axes
        wl_axis_nm = wl_axis * 1e9
        ax1.plot(wl_axis_nm, spectrum)

        # ax2.plot(delay_axis, np.real(degree_coherence), color='C0', label='Re($\gamma$)', lw=0.7)
        # ax2.plot(delay_axis, np.imag(degree_coherence), color='C1', label='Im($\gamma$)', lw=0.7)
        ax2.plot(delay_axis, abs(degree_coherence), ls='--', label='$|\gamma|$', lw=2)
        # ax2.plot(delay_axis, contrast_lz, ls=':', lw=2)

        # fwhm = pystark.get_fwhm_stark(n_upper, e_dens)
        # contrast_lorentzian = np.exp(- np.pi * fwhm * np.abs(delay_time_axis))
        # ax2.plot(delay_axis, contrast_lorentzian, color='C3', ls='--', label='$|\gamma|$', lw=2)

        ax1.set_xlabel('wavelength (nm)')
        ax1.set_title('spectrum')
        ax2.set_xlabel('delay (rad)')
        ax2.set_title('coherence')

        ax1.set_xlim([wl_axis.min() * 1e9, wl_axis.max() * 1e9])
        ax2.set_xlim([0, delay_max])
        ax2.set_yticks([0., 1.])
        leg = ax2.legend()

    return degree_coherence, delay_axis


if __name__ == '__main__':
    test()
