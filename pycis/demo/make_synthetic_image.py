import numpy as np
import matplotlib.pyplot as plt
import pycis

# quick synthetic image

# pco.edge 5.5 camera
bit_depth = 16
sensor_dim = (2560, 2160)
pix_size = 6.5e-6
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
cam = pycis.Camera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

flength = 85e-3
backlens = pycis.Lens(flength)

# interferometer components
pol_1 = pycis.LinearPolariser(0.5)
sp_1 = pycis.SavartPlate(np.pi / 4, 4.0e-3)
wp_1 = pycis.UniaxialCrystal(np.pi / 4, 4.48e-3, 0)
pol_2 = pycis.LinearPolariser(0.5)

inst = pycis.Instrument(cam, backlens, [pol_1, wp_1, sp_1, pol_2])
# inst.calculate_sensor_coords(downsample=None, crop=(100, 250, 100, 250), display=True)
wl = 466e-9

spectra = {'wl': wl,
           'spec': 50000,
           'spec units': 'cnts',
           }

si = pycis.model.SynthImage(inst, spectra)

si.img_igram()
si.img_fft()
plt.show()
