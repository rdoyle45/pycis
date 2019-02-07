import numpy as np
import scipy.interpolate
import os, csv, glob
import matplotlib.pyplot as plt
import pycis


class BandpassFilter:
    """ 
    base class for optical bandpass filters
    """

    def __init__(self, wl, tx, ref_index, name=None):
        """
        :param wl: wavelength [ m ]
        :param tx: transmission [ fraction ]
        :param ref_index: refractive index
        :param name: 
        """

        self.wl = wl
        self.tx = tx
        self.ref_index = ref_index
        self.name = name

        # rough central wavelength of filters
        self.wl_centre = self.wl[np.argmax(self.tx)]

    # core methods inherited by all filters
    def tilt(self, inc_angle):
        """
        Account for the blue-shift in the transmission profile caused by off-incidence rays (filters tilt), returning a
        shifted wavelength axis.
        
        :param inc_angle: [ rad ]
        :return: 
        """

        wavelength_shift = self.wl_centre * (np.sqrt(1 - (np.sin(inc_angle) / self.ref_index) ** 2) - 1)
        wl_effective = self.wl + wavelength_shift

        return wl_effective

    def interp_tx(self, wl_target, inc_angle=0.):
        """
        interpolation filters transmission window at given wavelength and for a given incidence angle.
        
        :param wl_target: 
        :param inc_angle: [ rad ]
        :return: 
        """

        if inc_angle != 0:
            wl_tilt = self.tilt(inc_angle)
        else:
            wl_tilt = self.wl

        # interpolate filters transmission profile
        tx_interp = np.interp(wl_target, wl_tilt, self.tx)
        # TODO alternative interpolation schemes

        return tx_interp

    def apply(self, wl_target, spec_target, inc_angle=0., display=False):
        """ 
        Apply filters to target spectrum.

        :param wl_target: [ m ]
        :param spec_target: light spectrum to be filtered [ arb. ]
        :param inc_angle: [ rad ]
        :param display: Boolean, plots interpolated filters profile and spectrum before and after filtering

        :return: 
        """

        # interpolate transmission onto target wavelength axis
        tx_interp = self.interp_tx(wl_target, inc_angle=inc_angle)

        # apply filters
        filtered_spectrum = spec_target * tx_interp

        if display:
            norm_factor = np.max(spec_target)

            fig, ax = plt.subplots()
            self.plot_tx(ax, inc_angle=0., color='grey', ls=':', label='filters profile - untilted')
            self.plot_tx(ax, inc_angle=inc_angle, color='k', label='filters profile - tilted')
            ax.plot(wl_target, spec_target / norm_factor, label='target spectrum')
            ax.plot(wl_target, filtered_spectrum / norm_factor, label='filtered spectrum')

            ax.set_xlabel('wavelength (m)')
            ax.set_ylabel('tx')

            leg = ax.legend(loc=0)
            plt.show()

        return filtered_spectrum

    def plot_tx(self, ax, inc_angle=0., wl_units='m', **kwargs):
        """
        Add transmission profile to an existing plot (for a given incidence angle).

        """

        wl_effective = self.tilt(inc_angle)

        if wl_units == 'nm':
            wl_plot = wl_effective * 1e9
        elif wl_units == 'm':
            wl_plot = wl_effective
        else:
            raise Exception('invalid wl_units')
        ax.plot(wl_plot, self.tx, **kwargs)
        return

    def save_csv(self, name=None):
        """
        save the filters properties to csv file in the filters repo directory, in the format that can be read using 
        'FilterFromFile' and 'FilterFromName'
        :return: 
        """

        if name is None:
            if self.name is None:
                raise Exception('please enter a name for the filters in order to save it')
            else:
                name = self.name

        fpath = os.path.join(pycis.paths.filters_path, name + '.csv')

        with open(fpath, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            csv_writer.writerow(['wl (m)', 'tx (fraction)', 'refractive index'])

            for idx, (wl_idx, tx_idx) in enumerate(zip(self.wl, self.tx)):
                if idx == 0:
                    csv_writer.writerow([wl_idx, tx_idx, self.ref_index])
                else:
                    csv_writer.writerow([wl_idx, tx_idx])


class AndoverSemiCustomFilter(BandpassFilter):

    def __init__(self, wl_centre, fwhm, peak_tx, type, ref_index):
        """ set the normalised transmission values, and the corresponding fwhm coefficients based on info from:
        https://www.andovercorp.com/technical/bandpass-filters-fundamentals/filters-types/

        :param wl_centre: centre wavelength [m]
        :param fwhm: full width half max. wavelength [m]
        :param peak_tx: transmission at centre, as a fraction. 
        :param type: type of semi-custom filters, according to Andover, basically the number of cavities.
        :param ref_index
        """

        self.wl_centre = wl_centre
        self.fwhm = fwhm
        self.peak_tx = peak_tx
        self.type = type

        self._tx_values = np.array([1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 5.e-1, 9.e-1]) * self.peak_tx

        self.fwhm_coeffs = {2: np.array([45, 15, 6.3, 3.5, 2, 1, 0.5]),
                            3: np.array([15, 5.4, 3.2, 2.2, 1.5, 1, 0.65]),
                            4: np.array([12, 4.25, 2.25, 1.8, 1.3, 1, 0.8])
                            }

        wl, tx = self._get_semicustom_tx()
        # wl, tx = self.get_interp_profile()

        super().__init__(wl, tx, ref_index)

    def _get_semicustom_tx(self):
        """
        use the info on the filters profiles from 
        https://www.andovercorp.com/technical/bandpass-filters-fundamentals/filters-types/ to calculate the rough tx 
        profile.
        
        :return:  tx
        """

        tx = np.concatenate([self._tx_values, [1 * self.peak_tx], self._tx_values[::-1]])
        wl = np.zeros_like(tx)

        for idx in range(len(self._tx_values)):
            half_width = (self.fwhm_coeffs[self.type][idx] * self.fwhm) / 2
            wl[idx] = self.wl_centre - half_width
            wl[-(idx + 1)] = self.wl_centre + half_width

        wl[len(self._tx_values)] = self.wl_centre

        return wl, tx

    def get_interp_profile(self):
        """ interpolate the few points provided by the manufacturer onto a finer wavelength grid."""

        no_fwhm = 10
        wl_lo = self.wl_centre - (no_fwhm * self.fwhm)
        wl_hi = self.wl_centre + (no_fwhm * self.fwhm)

        interp_wavelength_axis = np.linspace(wl_lo, wl_hi, 100)

        f = scipy.interpolate.InterpolatedUnivariateSpline(self.wls, self.tx, k=1, ext='zeros')
        interp_transmission = f(interp_wavelength_axis)

        return interp_wavelength_axis, interp_transmission


class FilterFromFile(BandpassFilter):

    def __init__(self, fname, ref_index=None):
        """
        
        load filters data saved as a .csv file
        
        .csv format: 
        the wavelengths as the first column [ m ] and transmission as the 
        second [ fraction ]. If not given as an input, effective refractive index of the filters can be specified as
        first entry in third column of csv file. 
        
        :param fname: 'path/to/filters/file.csv'
        :param ref_index: 
        """

        wl = []
        tx = []

        with open(fname, newline='') as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if row_idx != 0:
                    wl.append(np.float(row[0]))
                    tx.append(np.float(row[1]))
                if ref_index is None and row_idx == 1:
                    try:
                        ref_index = np.float(row[2])
                    except:
                        # no refractive index value found
                        ref_index = None

        wl = np.array(wl)
        tx = np.array(tx)

        super().__init__(wl, tx, ref_index)


class FilterFromName(FilterFromFile):
    def __init__(self, name):
        """
        Looks in filters repo directory for a named file
        :param name: 
        """

        if name[-4:] != '.csv':
            name += '.csv'

        valid_names = self.get_valid_names()
        assert name in valid_names

        fname = os.path.join(pycis.paths.filters_path, name)

        super().__init__(fname)

    @staticmethod
    def get_valid_names():

        valid_paths = glob.glob(os.path.join(pycis.paths.filters_path, '*.csv'))

        valid_names = []

        for path in valid_paths:
            valid_names.append(os.path.split(path)[1])

        return valid_names


if __name__ == '__main__':

    filt1 = FilterFromName('flow_plasma_HeII_andover_measured')
    filt2 = FilterFromName('flow_calib_1_andover_measured')
    filt3 = FilterFromName('flow_calib_2_andover_measured')

    fig, ax = plt.subplots()
    filt1.plot_tx(ax)
    filt2.plot_tx(ax)
    filt3.plot_tx(ax)
    plt.show()









