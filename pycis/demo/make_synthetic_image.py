import numpy as np
import matplotlib.pyplot as plt
import pycis

# define camera
# pco.edge 5.5 camera
bit_depth = 16
sensor_dim = (2560, 2160)
# sensor_dim = (250, 210)
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
wp_1 = pycis.UniaxialCrystal(np.pi / 4, 4.48e-3, 0)
pol_2 = pycis.LinearPolariser(0)
interferometer = [pol_1, wp_1, sp_1, pol_2]

# bringing it together into an instrument
inst = pycis.Instrument(cam, backlens, interferometer)


# inst.calculate_sensor_coords(downsample=None, crop=(100, 250, 100, 250), display=True)
wl = 466e-9

si = pycis.model.SynthImage(inst, wl, 1e5)

si.img_igram()
si.img_fft()
plt.show()
