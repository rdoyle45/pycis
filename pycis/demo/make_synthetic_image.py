import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pycis
import time
import os

bit_depth = 12
sensor_format = (1000, 1000)
pixel_size = 6.5e-6
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
cam = pycis.Camera(bit_depth, sensor_format, pixel_size, qe, epercount, cam_noise)

optics = [17e-3, 105e-3, 150e-3, ]

pol_1 = pycis.LinearPolariser(0)
wp_1 = pycis.UniaxialCrystal(np.pi / 4, 10e-3, 0, )
pol_2 = pycis.LinearPolariser(0)
interferometer = [pol_1, wp_1, pol_2]
inst = pycis.Instrument(cam, optics, interferometer)

wavelength = np.linspace(460e-9, 460.05e-9, 30)
wavelength = xr.DataArray(wavelength, dims=('wavelength', ), coords=(wavelength, ), )
x, y = inst.calculate_pixel_position()

spec = xr.ones_like(x * y * wavelength, )
spec /= spec.integrate(dim='wavelength')
spec *= 5e3

fpath_spec = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'spec.nc')

spec.to_netcdf(fpath_spec)
spec = xr.open_dataarray(fpath_spec, chunks={'x': 200, 'y': 200, })

s = time.time()
igram = inst.capture_image(spec, )
igram.load()
e = time.time()
print(e - s, 'sec')


def demo_multi_delay_cm():
    """
    multi-delay CIS instrument

    :return:
    """

    # define camera
    # pco.edge 5.5 camera
    bit_depth = 16
    # sensor_dim = (2560, 2160)
    sensor_dim = (1000, 1000)
    pix_size = 6.5e-6
    qe = 0.35
    epercount = 0.46  # [e / count]
    cam_noise = 2.5
    cam = pycis.Camera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

    # define imaging lens
    flength = 50e-3
    backlens = pycis.Lens(flength)

    # list interferometer components
    pol_1 = pycis.LinearPolariser(0)
    hwp = pycis.HalfWaveplate(-np.pi / 8)
    wp1 = pycis.UniaxialCrystal(-np.pi / 2, 6e-3, np.pi / 4, contrast=0.9)
    wp2 = pycis.UniaxialCrystal(0, 3e-3, np.pi / 4, contrast=0.9)
    wp3 = pycis.UniaxialCrystal(-np.pi / 4, 4e-3, np.pi / 4, contrast=0.9)
    pol_2 = pycis.LinearPolariser(-np.pi / 8)

    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol_1, hwp, wp1, wp2, wp3, pol_2]

    # bringing it together into an instrument
    inst = pycis.Instrument(cam, backlens, interferometer)

    wl = 466e-9
    spec = 1e5

    # wl0 = 464.9e-9
    # std = 0.090e-9
    # wl = np.linspace(wl0 - 3 * std, wl0 + 3 * std, 21)
    #
    # generate spectrum
    # spec = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(-1 / 2 * ((wl - wl0) / std) ** 2) * 1e5

    # pad speectrum to sensor array dimensions
    # spec = np.tile(spec[:, np.newaxis, np.newaxis], [1, sensor_dim[0], sensor_dim[1]])

    # stokes parameters
    # a0 = np.zeros_like(spec)
    # spec = np.array([spec, a0, a0, a0])

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    si.img_igram()
    si.img_fft()
    plt.show()


def demo_multi_delay():
    """
    multi-delay CIS instrument

    :return:
    """

    # define camera
    # pco.edge 5.5 camera
    bit_depth = 16
    # sensor_dim = (2560, 2160)
    sensor_dim = (1000, 1000)
    pix_size = 6.5e-6
    qe = 0.35
    epercount = 0.46  # [e / count]
    cam_noise = 2.5
    cam = pycis.Camera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

    # define imaging lens
    flength = 50e-3
    backlens = pycis.Lens(flength)

    # list interferometer components
    pol_1 = pycis.LinearPolariser(np.pi / 4)
    sp_1 = pycis.SavartPlate(np.pi / 4, 15e-3)
    dp_1 = pycis.UniaxialCrystal(np.pi / 2, 2e-3, np.pi / 4, contrast=0.9)
    pol_2 = pycis.LinearPolariser(np.pi / 4)
    # wp_2 = pycis.UniaxialCrystal(-np.pi / 4, 9e-3, np.pi / 8, contrast=0.9)
    # pol_3 = pycis.LinearPolariser(0)

    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol_1, sp_1, dp_1, pol_2]  # , wp_2, pol_3]

    # bringing it together into an instrument
    inst = pycis.Instrument(cam, backlens, interferometer)

    wl = 466e-9
    spec = 1e5

    # wl0 = 464.9e-9
    # std = 0.090e-9
    # wl = np.linspace(wl0 - 3 * std, wl0 + 3 * std, 21)
    #
    # generate spectrum
    # spec = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(-1 / 2 * ((wl - wl0) / std) ** 2) * 1e5

    # pad speectrum to sensor array dimensions
    # spec = np.tile(spec[:, np.newaxis, np.newaxis], [1, sensor_dim[0], sensor_dim[1]])

    # stokes parameters
    # a0 = np.zeros_like(spec)
    # spec = np.array([spec, a0, a0, a0])

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    si.img_igram()
    si.img_fft()
    plt.show()

