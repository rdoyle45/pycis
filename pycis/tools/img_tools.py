import copy
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.ndimage
# from PIL import Image
import pycis


default_roi_dim = (50, 50)

# manually set demodulation function
def demod_function(img, display=False):
    return pycis.demod.fourier_demod_2d(img, display=display)


def downsample(img, f):
    """
    reduce image sample rate by factor f in both x and y

    :param img: input image
    :param f: downsampling factor
    :return: 
    """

    assert isinstance(img, np.ndarray)
    assert img.ndim == 2

    return img[::f, ::f]


def crop(img, f):
    """
    crop image to rectangle with vertices at (x1, y1), (x1, y2), (x2, y1), (x2, y2).

    :param img:
    :param f: tuple (y1, y2, x1, x2)
    :return:
    """

    assert isinstance(img, np.ndarray)
    assert img.ndim == 2

    return img[f[0]:f[1], f[2]:f[3]]


def letterbox(img, f):
    """
    
    :param img: 
    :param f: 
    :return: 
    """

    assert isinstance(img, np.ndarray)
    assert img.ndim == 2

    return img[f:-f, :]


def left_crop(img, f):
    """
    
    :param img: 
    :param f: 
    :return: 
    """
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2

    return img[:, f:]

# def linearity_correction():
#
#
#     return


def get_lamp_data(units='m'):
    """ Returns wavelength in [nm]."""

    root_path = '/Users/jsallcock/Documents/physics/phd/collaborations/SWIP_2017/spectrometer_data/analysed/'

    file_names = ['swip_Cd', 'swip_Zn']
    suffixes = ['_wavelength.npy', '_intensity.npy']

    wavelength = []
    intensity = []

    for i, name in enumerate(file_names):
        load_path_wavelength = os.path.join(root_path, file_names[i] + suffixes[0])
        load_path_intensity = os.path.join(root_path, file_names[i] + suffixes[1])

        wavelength.append(np.load(load_path_wavelength))
        intensity.append(np.load(load_path_intensity))

    zn_wavelength = wavelength[1] + 0.095
    zn_intensity = intensity[1]
    cd_wavelength = wavelength[0] + 0.06
    cd_intensity = intensity[0]


    # background subtraction:
    zn_background_signal = get_background_signal(zn_intensity)
    cd_background_signal = get_background_signal(cd_intensity)

    zn_intensity -= zn_background_signal
    cd_intensity -= cd_background_signal

    # normalise to height of tallest peak:
    zn_intensity /= np.max(zn_intensity)
    cd_intensity /= np.max(cd_intensity)

    if units == 'm':
        zn_wavelength *= 1e-9
        cd_wavelength *= 1e-9
    else:
        pass


    return zn_wavelength, zn_intensity, cd_wavelength, cd_intensity


def plot_lamps():

    zn_wavelength, zn_intensity, cd_wavelength, cd_intensity = get_lamp_data()

    plt.figure()

    plt.plot(zn_wavelength, zn_intensity, label='Zn')
    plt.plot(cd_wavelength, cd_intensity, label='Cd')
    plt.legend(loc=0)
    plt.semilogy()
    plt.show()


    return


def get_background_signal(intensity):
    """ crude background subtract. """

    intensity = copy.deepcopy(intensity)

    length = len(intensity)
    subtract_factor = 0.1
    idx_lim = int(length * subtract_factor)

    return np.mean(np.append(intensity[0: idx_lim], intensity[-idx_lim: -1]))


def convert_predicted_cis_vignetting_from_matlab_to_npy():
    """ ... """

    dist = scipy.io.loadmat('/Users/jsallcock/Documents/physics/phd/scott_files/calibration_data/vignetting/distance.mat')
    etendue = scipy.io.loadmat('/Users/jsallcock/Documents/physics/phd/scott_files/calibration_data/vignetting/etendue.mat')

    etendue = np.squeeze(etendue['E'])
    dist = np.squeeze(dist['dist'])

    norm_etendue = etendue / etendue[0]

    dist = np.array(dist)
    etendue = np.array(etendue)


    plt.figure()
    plt.plot(dist, norm_etendue)
    plt.show()


    np.save('/Users/jsallcock/Documents/physics/phd/code/CIS/pycis/model/config/instrument/sensor_distance.npy', dist)
    np.save('/Users/jsallcock/Documents/physics/phd/code/CIS/pycis/model/config/instrument/etendue.npy', etendue)

    return


def get_img_flist(path, fmt='tif'):
    """

    :param path:
    :param fmt:
    :return:
    """

    assert os.path.isdir(path)

    return sorted(glob.glob(os.path.join(path, '*.' + fmt)))


def get_img(path, display=False):

    assert os.path.isfile(path)

    # img = np.array(Image.open(path), dtype=np.float64)
    img = np.array(plt.imread(path), dtype=np.float64)

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow_raw(img, 'gray')
        plt.colorbar(im)

        plt.show()

    return img


def get_img_stack(path, rot90=0, fmt='tif', display=False, overwrite=False, delete=False):
    """
    stack all images in a directory
    
    :param path: 
    :param rot90: number of 90 deg rotations (direction is from first towards the second axis, as per np.rot90
    convention)
    :param fmt: 
    :param overwrite: bool, if False, look for an existing image stack from a previous calculation.
    :param delete: delete original image files after stacking -- use with caution.
    :return: 
    """

    img_stack_path = os.path.join(path, 'img_stack.npy')
    flist = get_img_flist(path, fmt=fmt)
    fnum = len(flist)

    if os.path.isfile(img_stack_path) and overwrite is False:
        img_stack = np.load(img_stack_path)

    else:

        if fnum == 0:
            raise Exception('No files of this type found in the given directory.')

        img_stack = 0

        for f in flist:
            img_stack += get_img(f)

        if rot90 != 0:
            img_stack = np.rot90(img_stack, k=rot90)

        np.save(img_stack_path, img_stack)

    if delete:
        print('You are about to delete the contents of: ')
        print(str(path))
        print('-----')
        choice = input('Proceed? [y/n]: ')
        if choice == 'y':
            for f in flist:
                os.remove(f)
            print('deletion completed')
        else:
            pass

    if display:

        plt.figure()
        plt.imshow(img_stack, 'gray')
        plt.colorbar()
        plt.show(block=True)

    return img_stack


def get_phase_img_stack(path, rot90=0, fmt='tif', overwrite=False, **kwargs):
    """
    Calculate and save the phase of the stacked images of the specified format in the given directory.
    """

    assert os.path.isdir(path)

    phase_img_stack_path = os.path.join(path, 'phase_img_stack.npy')

    if os.path.isfile(phase_img_stack_path) and overwrite is False:
        phase_img_stack = np.load(phase_img_stack_path)

    else:

        img_stack = get_img_stack(path, rot90=rot90, fmt=fmt, overwrite=overwrite)

        intensity, phase, contrast = pycis.fourier_demod_2d(img_stack, mask=True, despeckle=True)
        phase_img_stack = phase
        np.save(phase_img_stack_path, phase_img_stack)

    return phase_img_stack


def get_contrast_img_stack(path, rot90=0, fmt='tif', overwrite=False):

    assert os.path.isdir(path)

    contrast_img_stack_path = os.path.join(path, 'contrast_img_stack.npy')

    if os.path.isfile(contrast_img_stack_path) and overwrite is False:
        contrast_img_stack = np.load(contrast_img_stack_path)

    else:

        img_stack = get_img_stack(path, rot90=rot90, fmt=fmt, overwrite=overwrite)

        intensity, phase, contrast = demod_function(img_stack, display=False)

        contrast_img_stack = contrast

        np.save(contrast_img_stack_path, contrast_img_stack)

    return contrast_img_stack


def get_dc_img_stack(path, rot90=0, fmt='tif', overwrite=False):

    assert os.path.isdir(path)

    dc_img_stack_path = os.path.join(path, 'dc_img_stack.npy')

    if os.path.isfile(dc_img_stack_path) and overwrite is False:
        dc_img_stack = np.load(dc_img_stack_path)

    else:

        img_stack = get_img_stack(path, rot90=rot90, fmt=fmt, overwrite=overwrite)

        intensity, phase, contrast = demod_function(img_stack, display=False)

        dc_img_stack = intensity

        np.save(dc_img_stack_path, dc_img_stack)

    return dc_img_stack


def get_contrast_roi_mean(path, roi_dim=default_roi_dim, overwrite=False):
    """ Calculate the mean roi contrast and estimate the std. roi contrast

        :param path: 
        :param fmt: 
        :param roi_dim:
        :param overwrite: (bool.)
        :return: phase_roi_mean, phase_roi_std (both in radians)
        """

    assert os.path.isdir(path)

    contrast_roi_stack_mean_path = os.path.join(path, 'contrast_roi_stack_mean.npy')

    # contrast ROI mean
    if os.path.isfile(contrast_roi_stack_mean_path) and overwrite is False:
        contrast_roi_stack_mean = np.load(contrast_roi_stack_mean_path)

    else:
        contrast_img_stack = get_contrast_img_stack(path, overwrite=overwrite)

        contrast_roi_stack_mean = np.mean(pycis.tools.get_roi(contrast_img_stack, roi_dim=roi_dim))

        # save
        np.save(contrast_roi_stack_mean_path, contrast_roi_stack_mean)

    return contrast_roi_stack_mean


def get_phase_roi_mean(path, rot90=0, roi_dim=default_roi_dim, overwrite=False):
    """ Calculate the mean roi phase and estimate the std. roi phase

    :param path: 
    :param fmt: 
    :param roi_dim:
    :param overwrite: (bool.)
    :return: phase_roi_mean, phase_roi_std (both in radians)
    """

    assert os.path.isdir(path)

    phase_roi_stack_mean_path = os.path.join(path, 'phase_roi_stack_mean.npy')

    # phase ROI mean (rad)
    if os.path.isfile(phase_roi_stack_mean_path) and overwrite is False:
        phase_roi_stack_mean = np.load(phase_roi_stack_mean_path)

    else:
        phase_img_stack = get_phase_img_stack(path, rot90=rot90, overwrite=overwrite)

        phase_img_stack = pycis.demod.unwrap(phase_img_stack)

        phase_roi_stack_mean = pycis.demod.wrap(np.mean(pycis.tools.get_roi(phase_img_stack, roi_dim=roi_dim)), units='rad')

        # save
        np.save(phase_roi_stack_mean_path, phase_roi_stack_mean)

    return phase_roi_stack_mean

#
# def get_phase_pixel_stack_std(path, rot90=0, fmt='tif', img_lim=30, overwrite=False):
#     """ Calculate standard deviation in phase for a single pixel in a series of images (assumes all images in the
#     given directory are the same dimensions).
#
#
#         :param path:
#         :param fmt:
#         :param roi_dim:
#         :param overwrite: (bool.)
#         :return: phase_roi_mean, phase_roi_std (both in radians)
#         """
#
#     assert os.path.isdir(path)
#
#     phase_pixel_stack_std_path = os.path.join(path, 'phase_pixel_stack_std.npy')
#
#
#     if os.path.isfile(phase_pixel_stack_std_path) and overwrite is False:
#         phase_pixel_stack_std = np.load(phase_pixel_stack_std_path)
#
#     else:
#         flist = get_img_flist(path, fmt=fmt)
#         fnum = len(flist)
#
#         # use all the images available, up to a set limit (img_lim)
#         if fnum > img_lim:
#             fnum_trunc = img_lim
#         else:
#             fnum_trunc = fnum
#
#         # get img dimensions and indices for a central pixel
#         img_h, img_w = np.array(Image.open(flist[0]), dtype=np.float64).shape
#         pix_y = round(img_h / 2)
#         pix_x = round(img_w / 2)
#
#         phase_pixel = []
#
#         for imgpath in flist[:fnum_trunc]:
#             print(imgpath)
#             img = np.array(Image.open(imgpath), dtype=np.float64)
#
#             if rot90 != 0:
#                 img = np.rot90(img, k=rot90)
#
#             intensity, phase, contrast = pycis.demod.fourier_demod_2d(img)
#
#             phase_pixel.append(phase[pix_y, pix_x])
#
#         phase_pixel_std = np.std(np.array(phase_pixel))
#
#         # scale to obtain phase std for the stacked pixel.
#         phase_pixel_stack_std = phase_pixel_std * fnum ** - 0.5
#
#         # save
#         np.save(phase_pixel_stack_std_path, phase_pixel_stack_std)
#
#     return phase_pixel_stack_std

def get_contrast_pixel_stack_std(path, fmt='tif', img_lim=30, overwrite=False):
    """ Calculate standard deviation in contrast for a single pixel in a series of images (assumes all images in the 
    given directory are the same dimensions).


        :param path: 
        :param fmt: 
        :param roi_dim:
        :param overwrite: (bool.)
        :return: phase_roi_mean, phase_roi_std (both in radians)
        """

    assert os.path.isdir(path)

    contrast_pixel_stack_std_path = os.path.join(path, 'contrast_pixel_stack_std.npy')

    if os.path.isfile(contrast_pixel_stack_std_path) and overwrite is False:
        contrast_pixel_stack_std = np.load(contrast_pixel_stack_std_path)

    else:
        flist = get_img_flist(path, fmt=fmt)
        fnum = len(flist)

        if fnum > img_lim:
            fnum_trunc = img_lim
        else:
            fnum_trunc = fnum

        # get img dimensions
        img_h, img_w = np.array(Image.open(flist[0]), dtype=np.float64).shape
        pix_y = round(img_h / 2)
        pix_x = round(img_w / 2)

        contrast_pixel = []

        for imgpath in flist[:fnum_trunc]:
            print(imgpath)
            img = np.array(Image.open(imgpath), dtype=np.float64)

            intensity, phase, contrast = pycis.demod.fourier_demod_2d(img)

            contrast_pixel.append(contrast[pix_y, pix_x])

        contrast_pixel_std = np.std(np.array(contrast_pixel))

        # scale to obtain phase STD for stacked pixel.
        contrast_pixel_stack_std = contrast_pixel_std * fnum ** - 0.5

        # save
        np.save(contrast_pixel_stack_std_path, contrast_pixel_stack_std)

    return contrast_pixel_stack_std


def get_phase_roi_std_err(path, rot90=0, fmt='tif', roi_dim=default_roi_dim, img_lim=25, overwrite=False):
    """ Calculate the mean roi phase and estimate the std. roi phase

    :param path: 
    :param fmt: 
    :param roi_dim:
    :param overwrite: (bool.)
    :return: phase_roi_mean, phase_roi_std (both in radians)
    """

    assert os.path.isdir(path)

    phase_roi_stack_std_path = os.path.join(path, 'phase_roi_stack_std.npy')
    phase_roi_mean_path = os.path.join(path, 'phase_roi_mean.npy')


    if os.path.isfile(phase_roi_stack_std_path) and overwrite is False:
        phase_roi_stack_std_err = np.load(phase_roi_stack_std_path)
        phase_roi_mean = np.load(phase_roi_mean_path)

    else:
        flist = get_img_flist(path, fmt=fmt)
        fnum = len(flist)

        if fnum > img_lim:
            fnum_trunc = img_lim
        else:
            fnum_trunc = fnum

        phase_roi_mean = []

        for imgpath in flist[:fnum_trunc]:
            print(imgpath)
            img = np.array(Image.open(imgpath), dtype=np.float64)

            if rot90 != 0:
                img = np.rot90(img, k=rot90)

            intensity, phase, contrast = pycis.demod.fourier_demod_2d(img)
            phase = pycis.demod.unwrap(phase)

            phase_roi_mean.append(pycis.demod.wrap(np.mean(pycis.tools.get_roi(phase, roi_dim=roi_dim))))

        phase_roi_mean = np.array(phase_roi_mean)

        # calculate standard error
        phase_roi_stack_std_err = np.std(phase_roi_mean) / fnum ** 0.5

        # save
        np.save(phase_roi_stack_std_path, phase_roi_stack_std_err)
        np.save(phase_roi_mean_path, phase_roi_mean)

    return phase_roi_stack_std_err, phase_roi_mean


def get_contrast_roi_std(path, fmt='tif', roi_dim=default_roi_dim, img_lim=25, overwrite=False):
    """ Calculate the mean roi contrast and estimate the std. roi contrast

    :param path: 
    :param fmt: 
    :param roi_dim:
    :param overwrite: (bool.)
    :return: phase_roi_mean, phase_roi_std (both in radians)
    """

    assert os.path.isdir(path)

    contrast_roi_stack_std_path = os.path.join(path, 'contrast_roi_stack_std.npy')
    contrast_roi_mean_path = os.path.join(path, 'contrast_roi_mean.npy')


    if os.path.isfile(contrast_roi_stack_std_path) and overwrite is False:
        contrast_roi_stack_std = np.load(contrast_roi_stack_std_path)
        contrast_roi_mean = np.load(contrast_roi_mean_path)

    else:
        flist = get_img_flist(path, fmt=fmt)
        fnum = len(flist)

        if fnum > img_lim:
            fnum_trunc = img_lim
        else:
            fnum_trunc = fnum

        contrast_roi_mean = []

        for imgpath in flist[:fnum_trunc]:
            print(imgpath)
            img = np.array(Image.open(imgpath), dtype=np.float64)

            intensity, phase, contrast = pycis.demod.fourier_demod_2d(img)
            # phase = pycis.uncertainty.unwrap(phase)

            contrast_roi_mean.append(pycis.demod.wrap(np.mean(pycis.tools.get_roi(phase, roi_dim=roi_dim))))

        contrast_roi_mean = np.array(contrast_roi_mean)

        contrast_roi_stack_std = np.std(contrast_roi_mean) / fnum ** (0.25)  # TODO: double check this scaling.

        # save
        np.save(contrast_roi_stack_std_path, contrast_roi_stack_std)
        np.save(contrast_roi_mean_path, contrast_roi_mean)


    return contrast_roi_stack_std, contrast_roi_mean


def offset_shape(phase, roi_dim=default_roi_dim):
    """ phase must be wrapped and in radians (ie. raw output of fd_image).
    
    :param phase: 
    :param roi_dim: 
    :return: 
    """

    phase = pycis.demod.unwrap(phase)

    offset = np.mean(pycis.tools.get_roi(phase, roi_dim=roi_dim))
    shape = phase - offset

    return offset, shape


def get_quick_mask(img, frac=0.2, display=False):
    """ Make crude mask for low signal areas using a single image. """

    img = scipy.ndimage.filters.gaussian_filter(img, 10)

    mask = np.zeros_like(img)

    intensity_lim = frac * np.max(img)

    mask[img > intensity_lim] = 1

    if display:
        plt.figure()
        plt.imshow(mask)
        plt.colorbar()
        plt.show()

    return mask



