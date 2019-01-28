import numpy as np
import matplotlib.pyplot as plt
import pycis
import time


def demo_1():
    """
    Conventional CIS instrument

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
    flength = 85e-3
    backlens = pycis.Lens(flength)

    # list interferometer components
    pol_1 = pycis.LinearPolariser(0)
    sp_1 = pycis.SavartPlate(np.pi / 4, 4.0e-3)
    wp_1 = pycis.UniaxialCrystal(np.pi / 4.1, 4.48e-3, 0)
    pol_2 = pycis.LinearPolariser(0)

    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol_1, wp_1, sp_1, pol_2]

    # bringing it together into an instrument

    inst = pycis.Instrument(cam, backlens, interferometer)

    # wl = 466e-9
    # spec = 1e5

    wl0 = 464.9e-9
    std = 0.090e-9
    wl = np.linspace(wl0 - 3 * std, wl0 + 3 * std, 21)

    # generate spectrum
    spec = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(-1 / 2 * ((wl - wl0) / std) ** 2) * 1e5

    # pad speectrum to sensor array dimensions
    spec = np.tile(spec[:, np.newaxis, np.newaxis], [1, sensor_dim[0], sensor_dim[1]])

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


def demo_2():
    """
    Polarisation camera CIS instrument

    """

    # define camera
    bit_depth = 16
    # sensor_dim = (2560, 2160)
    sensor_dim = (1024, 1024)
    pix_size = 6.5e-6
    qe = 0.35
    epercount = 0.46  # [e / count]
    cam_noise = 2.5
    cam = pycis.PolCamera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

    # define imaging lens
    flength = 85e-3
    backlens = pycis.Lens(flength)

    # define interferometer components
    wp = pycis.UniaxialCrystal(np.pi / 4, 4.48e-3, 0)
    qwp = pycis.QuarterWaveplate(0)
    pol = pycis.LinearPolariser(0)
    # first component in interferometer list is the first component that the light passes through
    interferometer = [pol, wp, qwp]

    # bringing it together into an instrument
    inst = pycis.Instrument(cam, backlens, interferometer)

    wl = 466e-9  # [ m ]
    spec = 1e5  # [ photons / pixel / time step ]

    s = time.time()
    si = pycis.SynthImage(inst, wl, spec)
    e = time.time()
    print(e - s, ' seconds')

    si.img_igram()
    si.img_fft()
    plt.show()

if __name__ == '__main__':
    demo_2()
