import numpy as np

# wavelength info ( NIST ) for the Cd / Zn spectral lines used for Doppler CIS calibration [ m ]
wl_cd1 = 467.81493e-9
wl_zn1 = 468.014e-9
wl_zn2 = 472.215e-9
wl_cd2 = 479.99123e-9
wl_zn3 = 481.053206e-9

lamp_wls = np.array([wl_cd1, wl_zn1, wl_zn2, wl_cd2, wl_zn3])

