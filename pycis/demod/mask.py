from numpy.lib.stride_tricks import as_strided as ast
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d
# import cv2
import os
import scipy.signal
import scipy.ndimage.filters
import pycis


class Mask(object):

    def __init__(self, raw_image):
        """ Mask off low brightness areas in the image with NaNs.
        
        :param raw_image: 
        """

        self.mask = self.get(raw_image)

    def get(self, raw_image):
        """ Calculate the mask for a raw CIS image. """

        blur_std, brightness_threshold = (20, 0.03)

        filtered_image = scipy.ndimage.filters.gaussian_filter(raw_image, blur_std)
        mask = np.zeros_like(raw_image)

        brightness_lim = np.max(raw_image) * brightness_threshold

        mask[filtered_image > brightness_lim] = 1

        return mask

    def apply(self, image_to_mask):
        """ Apply the mask to an inputted image. """

        image_to_mask[self.mask == 0] = np.nan

        return image_to_mask





def get_mask_old(phase, intensity, display=False):
    """ Input phi and I0 outputs from fftdemod, function will create a validity mask, setting areas with bad phase / contrast
    signal to 0. As faithful as possible a python port of the masking within Scott Silburn's 'IFdemod.m'.
    
    
    Not yet up and running until opencv becomes available on Freia (requested Oct 17)

    """

    y_pixels, x_pixels = np.shape(phase)

    block_dimension = 4  # [pix] Scott has 4
    block_height = block_dimension
    block_width = block_dimension
    block_array = block_view(phase, block=(block_height, block_width))

    y_block, x_block, _, _ = np.shape(block_array)

    phase_stdev_block = np.zeros([y_block, x_block])

    # loop through the blocks, taking the standard deviation of the 16 pixels in each.
    for i in range(0, y_block):
        for j in range(0, x_block):
            phase_stdev_block[i, j] = np.std(block_array[i, j, :, :])
            #phi_stdev_block[i, j] = np.mean(block_array[i, j, :, :])

    devim = cv2.resize(phase_stdev_block, (y_pixels, x_pixels), interpolation=cv2.INTER_CUBIC)

    tolim = np.zeros_like(devim)
    tolim[intensity > 0] = np.log10(intensity[intensity > 0]) * 1 # Scott has 0.2
    tolim[intensity == 0] = 0

    #tolim = scipy.signal.medfilt(tolim, 3) # Scott has no med filt

    tolim = scipy.ndimage.filters.gaussian_filter(input=tolim, sigma=5) # Scott has no Gaussian
    devim = scipy.ndimage.filters.gaussian_filter(input=devim, sigma=5) # Scott has no Gaussian

    mask = np.ones_like(phase, dtype=bool)
    mask[devim > tolim] = 0

    if display:
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(phase, interpolation='none')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.imshow(phase_stdev_block, interpolation='none')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.imshow(tolim, interpolation='none')
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.imshow(mask, interpolation='none')
        plt.colorbar()

        plt.show()


    return mask, devim, tolim


def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""

    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)

    shape= (int(A.shape[0] / block[0]), int(A.shape[1] / block[1])) + block
    strides= (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)

if __name__ == '__main__':
    # load CIS raw image (HL-2A #29657, viewing high field side CIII)
    raw_image_path = os.path.join(pycis.paths.demos_path, 'raw_image.tif')
    raw_image = plt.imread(raw_image_path)

    intensity, phase, contrast = pycis.demod.fd_image_1d(raw_image, mask=False, despeckle=True, display=False)
    phase = pycis.demod.unwrap(phase)

    mask, devim, tolim = get_mask_old(phase, intensity, display=True)


