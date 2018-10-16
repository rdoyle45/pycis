import numpy as np
import pickle
import pycis
import os.path
import csv
import matplotlib.pyplot as plt


import scipy.ndimage


class Filter(object):

    def __init__(self, name, wavelength, transmission, ref_index):
        """ Diescribes properties and application of optical bandpass filter. """

        self.name = name
        self.wavelength = wavelength
        self.transmission = transmission
        self.ref_index = ref_index

        self.centre_wavelength = self.wavelength[np.argmax(self.transmission)]  # roughly

    def get_transmission(self, target_wavelength, inc_angle=0, units='rad'):
        """ interpolation filter transmission window at given wavelength and for given incidence angle."""

        if inc_angle != 0:
            effective_wavelength = self.tilt_wavelength_shift(inc_angle, units=units)
        else:
            effective_wavelength = self.wavelength

        # interpolate filter transmission profile
        transmission_interp = np.interp(target_wavelength, effective_wavelength, self.transmission)

        return transmission_interp

    def apply(self, target_wavelength, target_spectrum, inc_angle=0, units='rad', display=False):
        """ Apply filter to user specified spectrum.
        
        :param target_wavelength: in metres
        :param target_spectrum: 
        :param inc_angle: (defaults to degrees)
        :param display: Boolean, plots interpolated filter profile and spectrum before and after filtering - handy!
        :return: 
        """


        if inc_angle != 0:
            effective_wavelength = self.tilt_wavelength_shift(inc_angle, units=units)
        else:
            effective_wavelength = self.wavelength

        transmission_interp = self.get_transmission(target_wavelength, inc_angle=inc_angle, units=units)

        # apply filter
        filtered_spectrum = target_spectrum * transmission_interp

        if display:
            display_norm_factor = np.max(target_spectrum)
            filter_profile_label = 'filter profile ' + pycis.tools.to_precision(inc_angle, 2) + ' deg tilt'

            plt.figure()
            plt.plot(effective_wavelength, self.transmission, 'k', label=filter_profile_label)
            plt.plot(target_wavelength, target_spectrum / display_norm_factor, label='target spectrum')
            plt.plot(target_wavelength, filtered_spectrum / display_norm_factor, label='filtered spectrum')

            plt.xlabel('wavelength (m)')
            plt.ylabel('norm.')

            plt.legend(loc=0)
            plt.show()

        return filtered_spectrum

    def tilt_wavelength_shift(self, inc_angle, units='rad'):
        """ Accounts for blue-shift introduced by non-zero incidence angle to first order. """

        if units == 'deg':
            inc_angle *= np.pi / 180
        elif units != 'rad':
            raise Exception('units kwarg must be either "rad" or "deg"')

        wavelength_shift = self.centre_wavelength * (np.sqrt(1 - (np.sin(inc_angle) / self.ref_index) ** 2) - 1)
        effective_wavelength = self.wavelength + wavelength_shift

        return effective_wavelength

    def plot_transmission(self, inc_angle=0, units='rad'):
        """ Plot filter transmission profile, if inc_angle specified will plot a comparison between tilted and untiled."""

        plt.figure()
        self.add_transmission_to_figure(inc_angle=0, units=units, color='b', label='untilted')

        if inc_angle != 0:
            if units == 'rad':
                inc_angle_label = 'tilted ' + pycis.tools.to_precision(pycis.tools.rad2deg(inc_angle), 2) + ' deg'
            elif units == 'deg':
                inc_angle_label = 'tilted ' + pycis.tools.to_precision(inc_angle, 2) + ' deg'
            else:
                raise Exception('units kwarg must be either "rad" or "deg"')

            self.add_transmission_to_figure(inc_angle=inc_angle, units=units, color='k', ls=':', label=inc_angle_label)

        plt.xlabel('wavelength (m)')
        plt.ylabel('transmission')
        plt.legend(loc=0)

        plt.show()

        return

    def add_transmission_to_figure(self, inc_angle=0, units='rad', **kwargs):
        """ Add transmission profile to an existing plot (for a given incidence angle). """

        effective_wavelength = self.tilt_wavelength_shift(inc_angle, units=units)
        plt.plot(effective_wavelength, self.transmission, **kwargs)
        return

    def save(self):
        pickle.dump(self, open(os.path.join(pycis.paths.filter_path, self.name + '.p'), 'wb'))
        return

def save_ccfe_filters():
    # load the from manufacturer .csv ccfe filter transmission profiles and save as instances of Filter class

    filter_names = ['ccfe_CIII', 'ccfe_HeII', 'ccfe_CII']
    filter_ref_indices = [1.45, 1.45, 1.45]  # nominal ref index for CII filter is 2.05 but 1.45 matches intensity rings
    # more closely
    num_filters = len(filter_names)

    for name, ref_index in zip(filter_names, filter_ref_indices):

        # read csv filter raw_data (from Scott Silburn)
        fm_load_path = os.path.join(pycis.paths.filter_path, 'raw_data', name + '_fm.csv')
        m_load_path_t = os.path.join(pycis.paths.filter_path, 'raw_data', name + '_transmission_m.npy')
        m_load_path_wl = os.path.join(pycis.paths.filter_path, 'raw_data', name + '_wavelength_m.npy')

        fm_wavelength = []
        fm_transmission = []

        with open(fm_load_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                fm_wavelength.append(row['Lambda (nm)'])
                fm_transmission.append(row['T'])

        fm_wavelength = np.array(fm_wavelength, dtype=np.float)
        fm_transmission = np.array(fm_transmission, dtype=np.float)

        fm_wavelength *= 1e-9  # convert [nm] --> [m]

        m_wavelength = np.load(m_load_path_wl)
        m_transmission = np.load(m_load_path_t)

        fm_optical_filter = pycis.model.Filter(name + '_fm', fm_wavelength, fm_transmission, ref_index)
        m_optical_filter = pycis.model.Filter(name + '_m', m_wavelength, m_transmission, ref_index)

        # fm_optical_filter.plot_transmission()
        # m_optical_filter.plot_transmission()
        fm_optical_filter.save()
        m_optical_filter.save()

    return

def save_calib_filters():
    # load the from manufacturer .csv ccfe filter transmission profiles and save as instances of Filter class

    filter_names = ['calib_filter_1', 'calib_filter_2']
    filter_ref_indices = [1.45, 1.45]  # not sure on this one right

    for name, ref_index in zip(filter_names, filter_ref_indices):

        # read .npy raw_data
        load_path_t = os.path.join(pycis.paths.filter_path, 'raw_data', name + '_transmission.npy')
        load_path_wl = os.path.join(pycis.paths.filter_path, 'raw_data', name + '_wavelength.npy')

        wl = np.load(load_path_wl)
        t = np.load(load_path_t)

        optical_filter = pycis.model.Filter(name, wl, t, ref_index)

        # optical_filter.plot_transmission()
        optical_filter.save()

# def save_swip_he_filter():
#     # load the from manufacturer .csv ccfe filter transmission profiles and save as instances of Filter class
#
#     name = 'swip_HeII_filter'
#     ref_idx = 1.45  # not sure on this one right
#
#     # read .npy raw_data
#     wl, tx = swip18.load_swip_heII_filter()
#
#     tx[wl > 470] = scipy.ndimage.filters.gaussian_filter(tx, 20)[wl > 470]
#
#     wl *= 1e-9
#
#     optical_filter = pycis.model.Filter(name, wl[::-1], tx[::-1], ref_idx)
#
#     # optical_filter.plot_transmission()
#     optical_filter.save()



if __name__ == '__main__':
    save_swip_he_filter()








