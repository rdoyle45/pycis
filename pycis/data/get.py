import sys
sys.path.append('/home/rdoyle/CIS')

from pyIpx.movieReader import ipxReader
import calcam
import pycis
from scipy.io import loadmat
import os
import pandas
import numpy as np
from pyEquilibrium import equilibrium
from scipy.interpolate import interp2d, griddata, interp1d
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.colors

import pyuda
client = pyuda.Client()

# Where the MAST calibrations are stored.
cal_lut_file = '/home/rdoyle/ssilburn/CIS_MATLAB_calibrations/LUT.xlsx'
cal_dir = '/home/rdoyle/ssilburn/CIS_MATLAB_calibrations'

try:
    # Load the ol' custom flow colour map
    cmd = np.loadtxt('/home/rdoyle/ssilburn/CIS/flow_cmap.txt')
    flow = matplotlib.colors.LinearSegmentedColormap.from_list('flow', cmd, N=128)
except:
    flow=0
    print('--pycis: problem loading custom colormap')


# This will be for later use; get the EFIT vector B field
def get_Bfield(pulse, time):
    eq = equilibrium.equilibrium(device='MAST', shot=pulse, time=time, with_bfield=True)
    bt = interp2d(eq.R, eq.Z, eq.Bt)
    br = interp2d(eq.R, eq.Z, eq.BR)
    bz = interp2d(eq.R, eq.Z, eq.BZ)
    return (bt, br, bz)


# Class for representing a frame of coherence imaging raw_data.
class CISImage():
    def __init__(self, shot, frame):
        """ Accessing the MAST CIS raw_data. Code written by Scott Silburn. """
        # Get raw raw_data
        self.shot = shot
        if shot > 28630 and shot < 30472:
            cam = 'rbc'
        else:
            raise ValueError('There was no CIS diagnostic for this shot!')
        # Get the raw_data using ipxReader and set the timestamp to the actual frame time stamp.
        #ipxreader = ipxReader(shot=shot, camera=cam)
        #ipxreader.set_frame_time(time)
        #_, self.raw_data, header = ipxreader.read()
        
        #self.raw_data = self.raw_data / 64
        #self.time = header['time_stamp']

        try:
            ipx_data = client.get('NEWIPX::read(filename=/net/fuslsa/data/MAST_Data/{0}/LATEST/rbc0{0}.ipx, frame={1})'.format(shot,frame), '')
        except ValueError:
            raise Exception("Frame {} does not exist. Please choose a different frame.".format(frame))

        frame = ipx_data.frames[0]
        self.raw_data = frame.k

        self.time = frame.time

        # Get calibrations
        self._get_calibrations()

        # Do the demodulation

        self._demodulate()

        # Assuming we have sight-line info, set the flow offset by looking at (geometrically) radial sight lines.
        # Not the most rigorous way of doing it but not too bad for now.
        if self.cal_dict['tangency_R'] is not None:
            self._apply_geom_calib()
        else:
            print('WARNING: No calcam calib for this shot! No radial sight-line calib applied.')

    # Fancy plotting!
    # type can be 'flow', 'I0', 'raw' or 'I0_flow'
    def show(self, type='I0_flow', ax=None, vmax=None, cmap=flow, show_sep=False):
        """ edited by jallcock to allow for passage of ax kwarg, a matplotlib axis"""

        if show_sep:
            xsep = []
            ysep = []
            eq = equilibrium.equilibrium(device='MAST', shot=self.shot, time=self.time)
            Rsep, Zsep = eq.get_fluxsurface(1.)
            xpoint_x = np.argmin(np.abs(self.cal_dict['tangency_R'][0, :]))
            Rtan = self.cal_dict['tangency_R'].copy()
            Rtan[:, xpoint_x:] = 0
            for i in range(Rsep.size):
                dist = np.sqrt((Rtan - Rsep[i]) ** 2 + (self.cal_dict['tangency_Z'] - Zsep[i]) ** 2)
                if dist.min() < 2e-3:
                    mpos = np.argmin(dist)
                    xsep.append(np.unravel_index(mpos, dist.shape)[1])
                    ysep.append(np.unravel_index(mpos, dist.shape)[0])
                else:
                    xsep.append(np.nan)
                    ysep.append(np.nan)

            Rtan = self.cal_dict['tangency_R'].copy()
            Rtan[:, :xpoint_x] = 0
            for i in range(Rsep.size):
                dist = np.sqrt((Rtan - Rsep[i]) ** 2 + (self.cal_dict['tangency_Z'] - Zsep[i]) ** 2)
                if dist.min() < 2e-3:
                    mpos = np.argmin(dist)
                    xsep.append(np.unravel_index(mpos, dist.shape)[1])
                    ysep.append(np.unravel_index(mpos, dist.shape)[0])
                else:
                    xsep.append(np.nan)
                    ysep.append(np.nan)

                    # Set the flow colour limits to +- vmax or +- 25km/s
        if vmax is None:
            clim = [-25, 25]
        else:
            clim = [-vmax, vmax]

        if type == 'flow':
            if ax is not None:
                im = ax.imshow(self.v_los / 1e3, cmap=cmap, clim=clim)
                cbar = plt.colorbar(im, ax=ax, label='Line-of-Sight flow (km/s)')
            else:
                plt.imshow(self.v_los / 1e3, cmap=cmap, clim=clim)
                plt.colorbar(label='Line-of-Sight flow (km/s)')
                plt.title('{:s} Flow image: #{:d} @ {:.0f} ms'.format(self.specline, self.shot, self.time * 1e3))
                plt.xlabel('X Pixel')
                plt.ylabel('Y Pixel')
                plt.show()

        elif type == 'I0':
            if ax is not None:
                im = ax.imshow(self.I0, cmap='gray', clim=clim)
                cbar = plt.colorbar(im, ax=ax, label='I0 (DL)')
            else:
                plt.imshow(self.I0, cmap='gray')
                plt.colorbar(label='I0 (DL)')
                plt.title('{:s} Intensity image: #{:d} @ {:.0f} ms'.format(self.specline, self.shot, self.time * 1e3))
                plt.xlabel('X Pixel')
                plt.ylabel('Y Pixel')
                plt.show()

        elif type == 'raw':
            if ax is not None:
                im = ax.imshow(pycis.demod.despeckle(self.raw_data), cmap='gray')
                # cbar = plt.colorbar(im, ax=ax, label='Raw (DL)')
            else:
                plt.imshow(self.raw_data, cmap='gray')
                plt.title('Raw image: #{:d} @ {:.0f} ms'.format(self.shot, self.time * 1e3))
                plt.colorbar(label='Raw raw_data (DL)')
                plt.xlabel('X Pixel')
                plt.ylabel('Y Pixel')
                plt.show()

        elif type == 'I0_flow':
            cm = matplotlib.cm.get_cmap(cmap)
            print(clim)
            vmap = self.v_los - clim[0] * 1e3
            vmap[vmap > (clim[1] - clim[0]) * 1e3] = (clim[1] - clim[0]) * 1e3
            vmap[vmap < 0] = 0
            vmap = vmap / ((clim[1] - clim[0]) * 1e3)
            cmapped = cm(vmap)
            Inorm = self.I0 / self.I0.max()
            Inorm = np.minimum(Inorm * 2, 1)
            for ax_idx in range(3):
                cmapped[:, :, ax_idx] = cmapped[:, :, ax_idx] * Inorm
            if ax is not None:
                print(ax)
                im0 = ax.imshow(self.v_los / 1e3, cmap=cmap, clim=clim)
                cbar = plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(label='flow (km/s)', size=14)
                ax.imshow(cmapped)
                ax.format_coord = self._format_coord_data
                if show_sep:
                    ax.plot_raw(xsep, ysep), 'w:'
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                im0 = ax.imshow(self.v_los / 1e3, cmap=cmap, clim=clim)
                plt.colorbar(im0, label='Line-of-Sight flow (km/s)')
                ax.imshow(cmapped)
                plt.title(
                    '{:s} Intensity & flow image: #{:d} @ {:.0f} ms'.format(self.specline, self.shot, self.time * 1e3))
                plt.xlabel('X Pixel')
                plt.ylabel('Y Pixel')
                ax.format_coord = self._format_coord_data
                if show_sep:
                    ax.plot_raw(xsep, ysep), 'w:'
                plt.show()
        else:
            raise ValueError('Unknown type of plot "{:s}"; can be "flow", "I0", "raw" or "I0_flow"'.format(type))

    def _demodulate(self):

        # Do the demodulation!
        self.I0, self.phi, self.xi = pycis.demod.fourier_demod_2d(self.raw_data, despeckle=True)
                                                          #tilt_angle=0)  # self.fringe_tilt)

        # Subtract calib phase and wrap in to [-pi,pi]
        self.deltaphi = self.phi - self.cal_dict['phi0']
        while self.deltaphi.max() > np.pi:
            self.deltaphi[self.deltaphi > np.pi] = self.deltaphi[self.deltaphi > np.pi] - 2 * np.pi
        while self.deltaphi.min() < np.pi:
            self.deltaphi[self.deltaphi < np.pi] = self.deltaphi[self.deltaphi < np.pi] + 2 * np.pi

        # Calibrate contrast (note: probably a load of rubbish; MAST contrast calibrations were not good).
        self.contrast = self.xi / self.cal_dict['xi0']

        # Convert demodulated phase to a flow!
        self.v_los = (3e8 * self.deltaphi / (2 * np.pi * self.cal_dict['N']))

        # Apply intensity flat field
        # self.I0 = self.I0 / self.cal_dict['flatfield']

    # Apply viewing geometry based calib offset correction.
    def _apply_geom_calib(self):

        if self.cal_dict['tangency_R'] is None:
            raise ValueError('No line-of-sight raw_data; cannot do radial calib!')

        # Just take sight-lines with I0 > 10 and tangency R < 5cm and use those as a zero flow reference
        self.v_cal_offset = self.v_los[np.logical_and(self.cal_dict['tangency_R'] < 0.05, self.I0 > 10)].mean()
        self.v_los = self.v_los - self.v_cal_offset

    # Get CIS calib raw_data based on MAST calib log and
    # loading old MATLAB files
    def _get_calibrations(self):

        # This dictionary will store the calobration
        caldict = {}

        # Check which row of the calib lookup table we need
        cal_ref_shot = None
        calib_log = pandas.read_excel(cal_lut_file, index_col=0)
        for first_shot in calib_log.index.values:
            if self.shot > first_shot:
                if self.shot < calib_log['last_shot'][first_shot]:
                    cal_ref_shot = first_shot
                    break

        if cal_ref_shot is None:
            raise ValueError('No calib found for this pulse!')
        
        # Load the MATLAB calib raw_data!
        matfile_path = os.path.join(cal_dir, '{:s}.mat'.format(calib_log['matlab_file'][cal_ref_shot]))
        matfile_data = loadmat(matfile_path)

        # Put the main quantities of interest in to our calib dictionary

        # inspecting the matfile_data dict:
        # print('----------')
        # for key, value in matfile_data.items():
        #     print(key, value)
        # print('----------')

        caldict['phi0'] = matfile_data['calibration'][0][0][0].copy()
        caldict['phi0'] = matfile_data['calibration'][0][0][0].copy()
        caldict['xi0'] = matfile_data['calibration'][0][0][1].copy()
        caldict['N'] = matfile_data['calibration'][0][0][4][0][0]
        # caldict['flatfield'] = matfile_data['calib'][0][0][9].copy()

        del matfile_data

        # Get the view geometry frmo Calcam - not currently used
        ''' 
        calcam_name = 'CIS/{:s}'.format( '-'.join( calib_log['extrinsics'][cal_ref_shot].split('-')[1:]) )
        try:
            raydata = calcam.RayData(calcam_name)
            caldict['los_directions'] = raydata.get_ray_directions()
            caldict['los_length'] = raydata.get_ray_lengths()
        except:
            print('WARNING: No Calcam raydata found for this pulse.')
            caldict['los_directions'] = None
            caldict['los_length'] = None
        '''

        # Load the sight-line calib also from the old MATLAB / IDL calcam raw_data
        raydata_path = os.path.join(cal_dir, 'Raydata-{:s}.mat'.format(
            '-'.join(calib_log['extrinsics'][cal_ref_shot].split('-')[1:])))
        raydata = loadmat(raydata_path)

        # Store each sight line's tangency R,Z
        # Most of the old raydata is calculated with 4x4 binning, so interpolate up to full resolution.
        x, y = np.meshgrid(np.linspace(0, 1023, raydata['tangency_R'].shape[1]),
                           np.linspace(0, 1023, raydata['tangency_R'].shape[0]))
        xn, yn = np.meshgrid(np.arange(1024), np.arange(1024))
        R = griddata((x.flatten(), y.flatten()), raydata['tangency_R'].flatten(), (xn, yn))

        x, y = np.meshgrid(np.linspace(0, 1023, raydata['tangency_Z'].shape[1]),
                           np.linspace(0, 1023, raydata['tangency_R'].shape[0]))
        Z = griddata((x.flatten(), y.flatten()), raydata['tangency_Z'].flatten(), (xn, yn))

        caldict['tangency_R'] = R
        caldict['tangency_Z'] = Z

        self.cal_dict = caldict

        # Store some miscellaneous metadata
        self.view_name = calib_log['view_name'][cal_ref_shot]
        self.plate_name = calib_log['delayplate'][cal_ref_shot]
        self.fringe_tilt = calib_log['fringe_tilt'][cal_ref_shot]
        self.specline = calib_log['spec_line'][cal_ref_shot]
        self.f1 = calib_log['f1'][cal_ref_shot]

    # Used for producing mouse-over text on plots.
    def _format_coord_data(self, x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and row >= 0 and col < self.v_los.shape[1] - 1 and row < self.v_los.shape[0] - 1:

            return 'Flow: {:.1f} km/s, I0: {:.0f} DL, Contrast: {:.2f}\nTangency R,Z: {:.3f} m, {:.3f}m\n'.format(
                self.v_los[row, col] / 1e3, self.I0[row, col], self.contrast[row, col],
                self.cal_dict['tangency_R'][row, col], self.cal_dict['tangency_Z'][row, col])
        else:
            return ''

    # Make print() do something useful
    def __str__(self):

        outstr = '\nMAST CIS Frame: #{:d} @ {:.0f} ms\n-------------------------------\n'.format(self.shot,
                                                                                                 self.time * 1e3)
        outstr = outstr + 'View:          {:s}\nf1:            {:s}\nSpectral line: {:s}\nDelay plate:   {:s}\n'.format(
            self.view_name, self.f1, self.specline, self.plate_name)
        return outstr


# If run as a script with a pulse number and time, e.g. "python CISImage.py 28751 0.13"
# Plot and print metadata for the given pulse and time.
if __name__ == '__main__':
    pulse = int(sys.argv[1])
    time = float(sys.argv[2])

    im = CISImage(pulse, time)
    print(im)
    im.show()
