import numpy as np
from . import despeckle as apply_despeckle
from scipy.constants import pi
import pycis

def polcam_demod(im, pixel_order=np.array([[0,45],[135,90]]), despeckle=False, uncertainty_out=False, components_out=False):
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

        uncertainty_out (bool)  : Whether to also return estimates of the phase & contrast uncertainties

        components_out (bool)   : Whether to also return the 4 individual polarised images.

    Returns:
    
        NuMPy arrays containing:
        Intensity image
        Phase image
        Contrast image

        and if requested:

        Dictionary containing keys 'delta_phi' and 'delta_contrast' which are \
        sort of estimates of the uncertainties.

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


    # Contrast (calculate based on 2 different pixel combinations and average to help noise robustness)
    contrast = np.sqrt( signal[:,:,0]**2 + signal[:,:,1]**2 )   / I0
    contrast2 = np.sqrt( signal[:,:,2]**2 + signal[:,:,3]**2 ) / I0
    contrast = (contrast + contrast2) / 2

    # Phase (calculate based on 2 different pixel combinations and average to help noise robustness)
    phi = np.arctan2(signal[:,:,1],signal[:,:,0])
    phi2 = np.arctan2(signal[:,:,3],signal[:,:,2])

    # Since we're averaging 2 phase measurements, make sure we don't end up with problems if the noise makes the phase wrap.
    phi2 = phi2 + -np.sign(phi2)*pi
    phi2[phi2 - phi > pi] = phi2[phi2 - phi > pi] - 2*pi
    phi2[phi2 - phi < -pi] = phi2[phi2 - phi < -pi] + 2*pi

    phi = (phi2 + phi)/2
    phi[phi < -pi] = phi[phi < -pi] + 2*pi
    phi[phi > pi] = phi[phi > pi] - 2*pi

    phi = pycis.wrap(phi)

    # Scale the intensity correctly.
    I0 = I0 * 2

    output = (I0,phi,contrast)

    if uncertainty_out:
        # Here we give the uncertainty based on the differences of the phase and contrast calculated from the two 
        # sets of pixel combinations.
        uncertainty = {'delta_phi':np.abs(phi2 - phi), 'delta_contrast':np.abs(contrast2 - contrast)}
        output = output + (uncertainty,)

    if components_out:
        signal = signal + np.tile(I0[:,:,np.newaxis],(1,1,4)) / 2
        components = {0:signal[:,:,0],45:signal[:,:,1],90:signal[:,:,2],135:signal[:,:,3]}
        output = output + (components,)

    return output


def polcam_demod2(im, pixel_order=np.array([[0, 45], [135, 90]]), despeckle=False, uncertainty_out=False,
                 components_out=False):
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

        uncertainty_out (bool)  : Whether to also return estimates of the phase & contrast uncertainties

        components_out (bool)   : Whether to also return the 4 individual polarised images.

    Returns:

        NuMPy arrays containing:
        Intensity image
        Phase image
        Contrast image

        and if requested:

        Dictionary containing keys 'delta_phi' and 'delta_contrast' which are \
        sort of estimates of the uncertainties.

        Dictionary containing integer keys 0,45,90 and 135 containing the \
        individual polarised images. The keys correspond to the polariser angles.

    '''

    # Check input array sizes
    if pixel_order.shape != (2, 2):
        raise ValueError('Pixel order should be a 2x2 array!')

    if im.shape[0] % 2 or im.shape[1] % 2:
        raise ValueError(
            'Image diemsnions must be even in both directions! Provided image is {:d}x{:d}.'.format(im.shape[1],
                                                                                                    im.shape[0]))

    # Separate out the 4 polarisation images
    signal = np.zeros(tuple(np.array(im.shape) // 2) + (4,))
    for a, angle in enumerate([0, 45, 90, 135]):
        i, j = np.where(pixel_order == angle)
        signal[:, :, a] = im[i[0]::2, j[0]::2]

    # pre-processing (pp): remove neutron speckles
    if despeckle:
        for channel in range(4):
            signal[:, :, channel] = apply_despeckle(signal[:, :, channel])

    # Isolate and remove the DC offset (mean of quadrature samples)
    i0 = signal.mean(axis=2)
    signal = signal - np.tile(i0[:, :, np.newaxis], (1, 1, 4))

    # Contrast (calculate based on 2 different pixel combinations and average to help noise robustness)
    # contrast = np.sqrt(signal[:, :, 0] ** 2 + signal[:, :, 1] ** 2) / i0
    # contrast2 = np.sqrt(signal[:, :, 2] ** 2 + signal[:, :, 3] ** 2) / i0
    # contrast = (contrast + contrast2) / 2

    contrast = 1 / np.sqrt(2) * np.sqrt(signal[:, :, 0] ** 2 + signal[:, :, 1] ** 2 + signal[:, :, 2] ** 2 +
                                    signal[:, :, 3] ** 2) / i0

    # Phase (calculate based on 2 different pixel combinations and average to help noise robustness)
    phi = np.arctan2(signal[..., 2] - signal[..., 0], signal[..., 3] - signal[..., 1])

    # Scale the intensity correctly.
    i0 = i0 * 2

    output = (i0, phi, contrast)

    # if uncertainty_out:
    #     # Here we give the uncertainty based on the differences of the phase and contrast calculated from the two
    #     # sets of pixel combinations.
    #     uncertainty = {'delta_phi': np.abs(phi2 - phi), 'delta_contrast': np.abs(contrast2 - contrast)}
    #     output = output + (uncertainty,)
    #
    # if components_out:
    #     signal = signal + np.tile(i0[:, :, np.newaxis], (1, 1, 4)) / 2
    #     components = {0: signal[:, :, 0], 45: signal[:, :, 1], 90: signal[:, :, 2], 135: signal[:, :, 3]}
    #     output = output + (components,)

    return output