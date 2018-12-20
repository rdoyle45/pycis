import pycis
import os
import matplotlib.pyplot as plt

##### Oct 2017: masking not working yet due to dependency on opencv

# DEMOD EXAMPLE 1: Demodulate experimental CIS tabulated_data (HL-2A #29657, viewing high field side CIII)

# load raw image
raw_img_path = os.path.join(pycis.paths.demos_path, 'raw_image.tif')
raw_img = plt.imread(raw_img_path)
raw_col = raw_img[:, 676]

# demodulate single column:
# dc_col, phase_col, contrast_col = pycis.uncertainty.fd_column(raw_col, display=True)

# demodulate entire image:
# dc_img, phase_img, contrast_img = pycis.uncertainty.fd_image_1d(raw_img, despeckle=False, display=True)
dc_img, phase_img, contrast_img = pycis.demod.fourier_demod_2d(raw_img, despeckle=True, display=True)


# DEMOD EXAMPLE 2: Generate and then demodulate synthetic CIS tabulated_data - with comparison to input to assess demodulation
# accuracy




