import numpy as np
import matplotlib.pyplot as plt

import pycis

# quick synthetic image
cam_name = 'pco.edge 5.5'
bit_depth = 16
sensor_dim = (2560, 2160)
pix_size = 6.5e-6
qe = 0.35
epercount = 0.46  # [e / count]
cam_noise = 2.5
cam = pycis.Camera(cam_name, bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)

lens_name = 'lens_name'
flength = 85e-3
backlens = pycis.Lens(lens_name, flength)

sp_1 = pycis.SavartPlate(np.pi / 4, 4.0e-3)
wp_1 = pycis.UniaxialCrystal(np.pi / 8, 4.48e-3, 0)

inst_name = 'ga'
inst_contrast = 1.0
inst = pycis.model.Instrument(inst_name, cam, backlens, [sp_1, wp_1],
                              instrument_contrast=inst_contrast)
wl = 466e-9

spectra = {'wl': wl,
           'spec': 50000,
           'spec units': 'cnts',
           }

si = pycis.model.SynthImagePhaseCalib(inst, spectra, 'example_name')

si.img_igram()
si.img_fft()
plt.show()
