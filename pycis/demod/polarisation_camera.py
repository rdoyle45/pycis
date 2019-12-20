import numpy as np
from . import despeckle as apply_despeckle
from scipy.constants import pi
import pycis

def polcam_demod(im, pixel_order=np.array([[0,45],[135,90]]), despeckle=False, components_out=False):
    '''
    Demodulation of polarisation camera CIS data.

    !!! WARNING !!! THIS MAY WELL BE WRONG IN SOME UNKNOWN WAY AT THE MOMENT, YET TO BE 
    FULLY TESTED & VALIDATED (seems to do strange things on my test data --Scott.)

    Parameters:

        im (np.ndarray)         : Array containing the image to demodulate.

        pixel_order (2x2 array) : Array specifying the layout of the polarisers on the \
                                  pixel array. This should be a 2x2 array representing the \
                                  repeating unit on the detector, containing the numbers 0,45, \
                                  90 and 135 to represent the polariser angles. I *think* changing \
                                  the definition of angle = 0 or direction will change the definition \
                                  of phi = 0 in the output, but the demodulation will still work.

        despeckle (bool)        : Whether to apply despeckle filter

        components_out (bool)   : Whether to also return the 4 individual polarised images.

    Returns:
    
        NuMPy arrays containing:
        Intensity image
        Phase image
        Contrast image

        and if requested:

        Dictionary containing integer keys 0,45,90 and 135 containing the \
        individual polarised images. The keys correspond to the polariser angles.

    '''

    # Check input array sizes
    if pixel_order.shape != (2,2):
        raise ValueError('Pixel order should be a 2x2 array!')

    if im.shape[0] % 2 or im.shape[1] % 2:
        raise ValueError('Image diemsnions must be even in both directions! Provided image is {:d}x{:d}.'.format(im.shape[1],im.shape[0]))

    # Separate out the 4 polarisation images
    signal = np.zeros( tuple(np.array(im.shape)//2) + (4,))
    for a,angle in enumerate([0,45,90,135]):
        i,j = np.where(pixel_order == angle)
        signal[:,:,a] = im[i[0]::2,j[0]::2]


    # pre-processing (pp): remove neutron speckles
    if despeckle:
        for channel in range(4):
            signal[:,:,channel] = apply_despeckle(signal[:,:,channel])


    # Isolate and remove the DC offset (mean of quadrature samples)
    I0 = signal.mean(axis=2)
    signal = signal - np.tile(I0[:,:,np.newaxis],(1,1,4))
    I0 = I0 * 2

    # Contrast (calculate based on 2 different pixel combinations and average to help noise robustness)
    contrast = (np.sqrt(np.sum(signal[:,:,:2]**2,axis=2)) + np.sqrt(np.sum(signal[:,:,2:]**2,axis=2)) )  / I0

    # Phase (calculate based on 2 different pixel combinations and average to help noise robustness)
    phi = np.arctan2(signal[:,:,2] - signal[:,:,0],signal[:,:,3] - signal[:,:,0])

    output = (I0,phi,contrast)

    if components_out:
        signal = signal + np.tile(I0[:,:,np.newaxis],(1,1,4)) / 2
        components = {0:signal[:,:,0],45:signal[:,:,1],90:signal[:,:,2],135:signal[:,:,3]}
        output = output + (components,)

    return output
