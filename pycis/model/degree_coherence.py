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

        integrand = spectrum * np.exp(1j * delay)

    else:
        # assume that the delay values given correspond to the c.o.m. frequency
        if material is not None:
            kappa_0 = pycis.dispersion(c / freq_com, material=material, output_derivatives=True)[3]
        else:
            kappa_0 = 1

        freq_shift_norm = (spectrum['frequency'] - freq_com) / freq_com
        integrand = spectrum * np.exp(1j * delay * (1 + kappa_0 * freq_shift_norm))

    degree_coherence = integrand.integrate(dim='frequency')

    return degree_coherence


def test_plot():
    """
    numerical / analytical test of calculate_degree_coherence() using a modelled Gaussian spectral lineshape

    :return:
    """
    import time

    # ANALYTICAL
    wl_0 = 464.8e-9
    material = 'a-BBO'
    kappa_0 = pycis.dispersion(wl_0, material, output_derivatives=True)[3]
    wl_sigma = 0.05e-9
    n_sigma = 10
    n_bins = 20000
    thickness = np.array([4.48e-3,
                          6.35e-3,
                          9.79e-3, ])  # waveplate thicknesses to test, in m

    # calculate delays at wl_0 for the given waveplate thicknesses
    biref_0 = pycis.dispersion(wl_0, material, )[0]
    delay_0 = 2 * np.pi * thickness * np.abs(biref_0) / wl_0  # (rad)
    delay_0 = xr.DataArray(delay_0, dims=('delay_0',), coords=(delay_0,), attrs={'units': 'rad'})

    # generate spectrum in frequency-space
    freq_0 = c / wl_0
    freq_sigma = c / wl_0 ** 2 * wl_sigma
    freq = np.linspace(freq_0 - n_sigma * freq_sigma, freq_0 + n_sigma * freq_sigma, n_bins)
    freq = xr.DataArray(freq, dims=('frequency', ), coords=(freq, ), attrs={'units': 'Hz'})
    spectrum = 1 / (freq_sigma * np.sqrt(2 * np.pi)) * np.exp(- 1 / 2 * ((freq - freq_0) / freq_sigma) ** 2)

    delay_time_axis_0 = np.linspace(0, n_sigma / 20 * 1 / freq_sigma, n_bins)
    delay_axis_0 = 2 * np.pi * delay_time_axis_0 * freq_0
    delay_axis_0 = xr.DataArray(delay_axis_0, coords=(delay_axis_0, ), dims=('delay_0', ), attrs={'units': 'rad'}, )

    def doc_analytical(delay_0):
        doc_analytical = np.exp(-2 * (np.pi * freq_sigma * (delay_0 / (2 * np.pi * freq_0)) * kappa_0) ** 2) * \
                         np.exp(1j * delay_0)
        return doc_analytical

    # NUMERICAL 1 (n1) -- tests group delay approximation
    s = time.time()
    doc_n1 = calculate_degree_coherence(spectrum, delay_0, material=material)
    e = time.time()
    print('numerical_1:', e - s, 'seconds')

    # NUMERICAL 2 (n2) -- tests full dispersion integral
    thickness = xr.DataArray(thickness, dims=('delay_0', ), coords=(delay_0, ))
    biref = pycis.dispersion(c / freq, material)[0]
    delay = 2 * np.pi * np.abs(biref) * thickness / (c / freq)

    s = time.time()
    doc_n2 = calculate_degree_coherence(spectrum, delay, material=material)
    e = time.time()
    print('numerical_2:', e - s, 'seconds')

    # PLOT
    fig = plt.figure(figsize=(10, 5,))
    axes = [fig.add_subplot(1, 3, i) for i in [1, 2, 3, ]]
    titles = ['Spectrum', 'Contrast', 'Phase', ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)

    # plot spectrum
    spectrum.plot(ax=axes[0])

    # plot contrast and phase
    funcs = [np.abs, xr.ufuncs.angle]
    for idx, (func, ax, ) in enumerate(zip(funcs, axes[1:])):
        if idx == 0:
            func(doc_analytical(delay_axis_0)).plot(ax=ax, lw=0.5, color='C0')
        func(doc_analytical(delay_0)).plot(ax=ax, lw=0, marker='x', markersize=12,
                                             label='Analytical\n(Group delay approx.)', color='C0')
        func(doc_n1).plot(ax=ax, lw=0, marker = '.', markeredgewidth=0.5, markeredgecolor='k',
                              label='Numerical\n(Group delay approx.)', markersize=8, color='C1')
        func(doc_n2).plot(ax=ax, lw=0, marker='d', markersize=12, fillstyle='none', label='Numerical (full)', color='C2')
        leg = ax.legend(fontsize=7)

    plt.show()


if __name__ == '__main__':
    test_plot()
