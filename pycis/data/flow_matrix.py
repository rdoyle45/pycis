import copy
import random
import time
import numpy as np
import calcam
from calcam.gm import misc, config, _bin_image, CoordTransformer, PoloidalVolumeGrid, ZipSaveFile
import matplotlib.pyplot as plt
import multiprocessing
from pyEquilibrium.equilibrium import equilibrium
from pycis.solvers import sart
from .get import CISImage, get_Bfield
from functools import partial
from scipy import sparse, io
from progress.bar import Bar
import os
import json


class FlowGeoMatrix:
    """
    Class to represent a flow geometry matrix and associated metadata.

    The matrix itself can be accessed in the `data` attribute, where it is
    stored as a sparse matrix using the scipy.sprase.csr_matrix class.

    The matrix, denoted by Nu_ij, and is constructed as per eq. 4.6.6 [1], where
    epsilon_0,j is inv_emis[0], Epsilon_ij' is geom_mat.data, B_k djfd and l_k is l_k_vectors

    References:
        [1] SILBURN,SCOTT,ALAN(2014)A Doppler Coherence Imaging Diagnostic for
        the Mega-Amp Spherical Tokamak, Durham theses, Durham University.

    Parameters:
        shot (int)				: MAST shot being studied

        frame (int)				: Frame required from CIS camera

        raydata (calcam.RayData or file)
                                : RayData object or Saved Ray data file for the camera to be inverted

        geom_mat (calcam.gm.GeometryMatrix or file)
                                : GeometryMatrix object or saved Geometry Matrix (NumPy or Matlab format)

        grid (calcam.gm.PoloidalVolumeGrid)
                                : PoloidalVolumeGrid object to use to generate geom_mat if geom_mat = None

        raw_cis (file)     : File containing I0, v_los and time data as output from CISImage.save().

        inv_emis (file)		: File containing inverted emissivity as a numpy array
                              Calculated by solving y = Ax + b.

        pixel_order (str)		: What pixel order to use when flattening the 2D image
                                    array in to the 1D data vector. Default 'C' goes
                                    left-to-right, then row-by-row (NumPy default),
                                    alternatively 'F' goes top-to-bottom, then
                                    column-by-column (MATLAB default).

        calc_status_callback (callable)
                                : Callable which takes a single argument, which will be
                                    called with status updates about the calculation.
                                    It will be called with either a string for textual
                                    status updates or a float from 0 to 1 specifying
                                    the progress of the calculation. By default, status
                                    updates are printed to stdout.  If set to None, no
                                    status updates are issued.

        verbose (Boolean) : Print outputs as the program runs

        trim (Boolean)  : Trim unused grid cells from the matrix

    """

    def __init__(self, shot, frame, raydata=None, geom_mat=None, grid=None, raw_cis=None, inv_emis=None, pixel_order='C',
                 calc_status_callback=misc.LoopProgPrinter().update, verbose=False, trim=False):

        if shot is not None:
            self.shot = shot
            self.frame = frame

            if raydata:
                if isinstance(raydata, str):
                    self.raydata = calcam.RayData(raydata)  # Open RayData File
            else:
                raise Exception('No RayData object or saved file provided. Please provide a file or run '
                                'calcam.raycast_sightlines()')

            # Open GeometryMatrix file, if no file is provided one will be constructed from the
            # grid and ray data provided
            if geom_mat:
                if isinstance(geom_mat, str):
                    self.geom_mat = calcam.gm.GeometryMatrix.fromfile(geom_mat)
                    self.grid = copy.copy(self.geom_mat.grid)
                    self.pixel_mask = self.geom_mat.pixel_mask
            else:
                if not grid:
                    print('Generating default square grid for MAST.')
                    grid = calcam.gm.squaregrid('MAST', cell_size=1e-2, zmax=-0.6)

                self.grid = copy.copy(grid)
                self.geom_mat = calcam.gm.GeometryMatrix(self.grid, self.raydata)
                self.grid = self.geom_mat.grid  # Redefine grid as cells will have been removed for lack of data
                self.geom_mat.save("{0}/geom_{1}_{2}.npz".format(os.getcwd(), str(shot), str(frame)))

            geom_data = self.geom_mat.data

            if self.raydata.fullchip:
                if self.raydata.fullchip is True:
                    raise Exception('Raydata object does not contain information on the image orientation used for ray '
                                    'casting.\n Please set the "fullchip" attribute of the raydata object to either '
                                    '"Display" or "Original".')
                else:
                    self.image_coords = self.raydata.fullchip
                    self.binning = self.raydata.binning
                    self.pixel_order = pixel_order
                    if self.image_coords.lower() == 'original':
                        imdims = (np.array(self.raydata.transform.get_original_shape()[::-1]) / self.binning).astype(int)
                    elif self.image_coords.lower() == 'display':
                        imdims = (np.array(self.raydata.transform.get_display_shape()[::-1]) / self.binning).astype(int)
                    else:
                        raise Exception("Raydata object does not contain information on the image orientation used for ray "
                                        "casting.\n Please set the \"fullchip\" attribute of the raydata object to either "
                                        "\"Display\" or \"Original\".")
                    self.pixel_mask = np.ones(imdims, dtype=bool)
            else:

                self.image_coords = None
                self.binning = None
                self.pixel_order = None
                self.pixel_mask = None

            self.image_geometry = self.raydata.transform
            self.history = {'los': self.raydata.history, 'grid': self.grid.history, 'matrix': 'Created by {:s} on {:s} at {:s}'.format(misc.username, misc.hostname,
                            misc.get_formatted_time())}

            # Solve y = Ax + b for x, the inverted emissivity matrix
            if inv_emis:
                self.inv_emis = np.load(inv_emis, allow_pickle=True)['data']
                #self.time = np.load(inv_emis, allow_pickle=True)['time']
                self.time = 0.31
            else:

                emis_vector, self.time = self._data_vector(raw_emis=raw_cis)
                self.inv_emis = sart.solve(geom_data, emis_vector, tol=1e-4, max_iter=5000)[0]

            b_field_funcs = get_Bfield(self.shot, self.time)  # Functions to calculate B-field components at a given point

            ray_start_coords = self.raydata.ray_start_coords.reshape(-1, 3, order=self.pixel_order)
            ray_end_coords = self.raydata.ray_end_coords.reshape(-1, 3, order=self.pixel_order)

            # Number of grid cells and sight lines
            n_cells = self.grid.n_cells
            n_los = self.raydata.x.size

            # Shuffling indices results in a better time remaining estimation
            inds = list(range(n_los))
            random.shuffle(inds)

            weight_rowinds, weight_colinds, weight_values = _weighting_matrix(geom_data, self.inv_emis, verbose)
            weighting_matrix = sparse.csr_matrix((weight_values, (weight_rowinds, weight_colinds)),
                                                 shape=(n_los, n_cells))
            # coo_weight = weighting_matrix.tocoo()
            # np.savez_compressed("{0}/weight_matrix_{1}_{2}.npz".format(os.getcwd(), str(shot), str(frame)),
            #                     mat_row_inds=coo_weight.row,
            #                     mat_col_inds=coo_weight.col,
            #                     mat_data=coo_weight.data,
            #                     mat_shape=coo_weight.shape)

            # Multi-threadedly loop over each sight-line in raydata and calculate the positions at which
            # each interacts with a cell wall
            if calc_status_callback is not None and verbose:
                calc_status_callback('Calculating sight-line cell interactions using {:d} '
                                     'CPUs...'.format(config.n_cpus))

            last_status_update = 0.

            rays = np.hstack((ray_start_coords[inds, :], ray_end_coords[inds, :]))  # Combine coords for imap
            b_l_matrix_data = []

            if verbose:
                # Progress bar indicating how much of the weighting matrix has been completed
                progressbar = Bar('B_l Matrix', max=len(inds), suffix='%(percent)d%%')

            with multiprocessing.Pool(config.n_cpus) as cpupool:
                if verbose:
                    calc_status_callback(0.)
                for (i, data), pixel in zip(enumerate(cpupool.imap(partial(calculate_geom_mat_elements, self.grid,
                                                                           b_field_funcs), rays, 10)), inds):
                    if data is not None:
                        b_l = data[0]
                        cells = data[1]
                        for cell_no, b_l_data in zip(cells, b_l):
                            if weighting_matrix[pixel, int(cell_no)] != 0:
                                b_l_matrix_data.append([pixel, int(cell_no), b_l_data[0]])

                    if verbose:
                        if time.time() - last_status_update > 1. and calc_status_callback is not None:
                            calc_status_callback(float(i) / n_los)
                            last_status_update = time.time()
                        progressbar.next()

            if verbose:
                progressbar.finish()

                if calc_status_callback is not None:
                    calc_status_callback(1.)

            b_l_matrix_data = np.asarray(b_l_matrix_data)
            b_l_sparse_matrix = sparse.csr_matrix((b_l_matrix_data[:, 2], (b_l_matrix_data[:, 0], b_l_matrix_data[:, 1])),
                                                  shape=(n_los, n_cells))
            # coo_bl = b_l_sparse_matrix.tocoo()
            # np.savez_compressed("{0}/bl_matrix_{1}_{2}.npz".format(os.getcwd(), str(shot), str(frame)),
            #                     mat_row_inds=coo_bl.row,
            #                     mat_col_inds=coo_bl.col,
            #                     mat_data=coo_bl.data,
            #                     mat_shape=coo_bl.shape)

            self.data = weighting_matrix.multiply(b_l_sparse_matrix)

            if trim:
                # Remove any grid cells + matrix rows which have no sight-line coverage.
                unused_cells = np.where(np.abs(self.data.sum(axis=0)) == 0)[1]
                self.grid.remove_cells(unused_cells)

                used_cols = np.where(np.abs(self.data.sum(axis=0)) > 0)[1]
                self.data = self.data[:, used_cols]

                self.set_included_pixels(pixel_mask=self.pixel_mask)

    def set_binning(self, binning):
        """
        Set the level of image binning. Can be used to
        decrease the size of the matrix to reduce memory or
        computation requirements for inversions.

        Parameters:

            binning (float) : Desired image binning. Must be larger than the \
                              existing binning value.

        """
        if binning < self.binning:
            raise ValueError('Specified binning is lower than existing binning! The binning can only be increased.')
        elif binning == self.binning:
            return
        else:

            if self.image_coords is not None:
                if self.image_coords.lower() == 'display':
                    image_dims = self.image_geometry.get_display_shape()
                elif self.image_coords.lower() == 'original':
                    image_dims = self.image_geometry.get_original_shape()
                else:
                    raise Exception('Raydata object does not contain information on the image orientation used for ray '
                                    'casting.\n Please set the "fullchip" attribute of the raydata object to either '
                                    '"Display" or "Original".')
            else:
                raise Exception('Nope, no worky.')

            bin_factor = int(binning / self.binning)

            init_shape = (np.array(image_dims) / self.binning).astype(np.uint32)
            row_inds = np.arange(np.prod(init_shape), dtype=np.uint32)
            row_inds = np.reshape(row_inds, init_shape, order=self.pixel_order)

            px_mask = self.pixel_mask.reshape(self.pixel_mask.size, order=self.pixel_order)
            index_map = np.zeros(np.prod(image_dims) // int(self.binning ** 2), dtype=np.uint32)
            index_map[px_mask == True] = np.arange(np.count_nonzero(px_mask), dtype=np.uint32)

            self.pixel_mask = _bin_image(self.pixel_mask, bin_factor, bin_func=np.min).astype(bool)

            px_mask = self.pixel_mask.reshape(self.pixel_mask.size, order=self.pixel_order)

            ind_arrays = []
            for colshift in range(bin_factor):
                for rowshift in range(bin_factor):
                    ind_arrays.append(row_inds[colshift::bin_factor, rowshift::bin_factor].reshape(
                        int(np.prod(init_shape) / bin_factor ** 2), order=self.pixel_order))
                    ind_arrays[-1] = index_map[ind_arrays[-1]][px_mask == True]

            new_data = self.data[ind_arrays[0], :]
            for indarr in ind_arrays[1:]:
                new_data = new_data + self.data[indarr, :]

            norm_factor = sparse.diags(np.ones(self.grid.n_cells) / bin_factor ** 2, format='csr')

            self.data = new_data * norm_factor

            self.binning = binning

    def set_included_pixels(self, pixel_mask, coords=None):
        """
        Set which image pixels should be included, or not. Can be
        used to exclude image pixels which are known to have bad data or
        otherwise do not conform to the assumptions of the inversion.

        Note: excluding pixels is a non-reversible process since their
        matrix rows will be removed. It is therefore recommended to keep a
        master copy of the matrix with all pixels included and then
        use this function on a transient copy of the matrix.

        Parameters:

            pixel_mask (numpy.ndarray) : Boolean array the same shape as the un-binned \
                                         camera image, where True or 1 indicates a \
                                         pixel to be included and False or 0 represents \
                                         a pixel to be excluded.

            coords (str)               : Either 'Display' or 'Original', \
                                         specifies what orientation the input \
                                         pixel mask is in. If not givwn, it will be \
                                         auto-detected if possible.
        """
        if self.pixel_mask is None:
            raise Exception('Cannot set pixel mask for a geometry matrix which does not include full sensor.')

        if coords is None:

            # If there are no transform actions, we have no problem.
            if len(self.image_geometry.transform_actions) == 0 and self.image_geometry.pixel_aspectratio == 1:
                coords = self.image_coords

            elif self.image_geometry.get_display_shape() != self.image_geometry.get_original_shape():

                if pixel_mask.shape == self.image_geometry.get_display_shape()[::-1]:
                    coords = 'Display'
                elif pixel_mask.shape == self.image_geometry.get_original_shape()[::-1]:
                    coords = 'Original'
                else:
                    raise ValueError(
                        'Input pixel mask has an unexpected shape! Got {:d}x{:d}; expected {:d}x{:d} or {:d}x{:d}'.format(
                            image.shape[1], image.shape[0], self.image_geometry.get_display_shape[1],
                            self.image_geometry.get_display_shape[0], self.image_geometry.get_original_shape[1],
                            self.image_geometry.get_original_shape[0]))
            else:
                raise Exception(
                    'Cannot determine mask orientation automatically; please provide the "coords" input argument.')

        if coords.lower() == 'display' and self.image_coords.lower() == 'original':
            pixel_mask = self.image_geometry.display_to_original_image(pixel_mask.astype(int), interpolation='nearest')

        elif coords.lower() == 'original' and self.image_coords.lower() == 'display':
            pixel_mask = self.image_geometry.original_to_display_image(pixel_mask.astype(int), interpolation='nearest')

        if self.binning > 1:
            pixel_mask = _bin_image(pixel_mask, self.binning, bin_func=np.min)

        mask_delta = self.pixel_mask.astype(int) - pixel_mask.astype(int)

        if mask_delta.min() == -1:
            raise ValueError('Provided pixel mask includes previously excluded pixels; pixels can only be '
                             'disabled, not re-enabled using this function.')

        old_px_mask = self.pixel_mask.reshape(self.pixel_mask.size, order=self.pixel_order)
        mask_delta = mask_delta.reshape(mask_delta.size, order=self.pixel_order)

        mask_delta = mask_delta[old_px_mask == True]

        self.data = self.data[mask_delta == 0, :]

        self.pixel_mask = pixel_mask.astype(bool)

    def get_included_pixels(self):
        """
        Get a mask showing which image pixels are included in the
        geometry matrix.

        Returns:

            numpy.ndarray : Boolean array the same shape as the camera image after binning, \
                            where True indicates the corresponding pixel is included \
                            and False indicates the pixel is excluded.
        """
        return self.pixel_mask

    def save(self, filename):
        """
        Save the geometry matrix to a file.

        Parameters:

            filename (str) : File name to save to, including file extension. \
                             The file extension determines the format to be saved: \
                             '.npz' for compressed NumPy binary format or \
                             '.mat' for MATLAB format

        Note: .npz is the recommended format; .mat is provided if compatibility
        with MATLAB is required but produces larger file sizes.
        """
        try:
            fmt = filename.split('.')[1:][-1]
        except IndexError:
            raise ValueError(
                'Given file name does not include file extension; extension .npz, .mat or .zip must be '
                'included to determine file type!')

        if fmt == 'npz':
            self._save_npz(filename)
        elif fmt == 'mat':
            self._save_matlab(filename)
        else:
            raise ValueError('File extension "{:s}" not understood; options are "npz", "mat" or "zip".'.format(fmt))

    def format_image(self, image, coords=None):
        """
        Format a given 2D camera image in to a 1D data vector
        (i.e. :math:`b` in :math:`Ax = b`) appropriate for use with this
        geometry matrix. This will bin the image, remove any excluded
        pixels and reshape it to a 1D vector.

        Parameters:

            image (numpy.ndarray) : Input image.

            coords (str)          : Either 'Display' or 'Original', \
                                    specifies what orientation the input \
                                    image is in. If not givwn, it will be \
                                    auto-detected if possible.

        Returns:

            scipy.sparse.csr_matrix : 1xN_pixels image data vector. Note that this is \
                                      returned as a sparse matrix object despite its \
                                      density being 100%; this is for consistency with the \
                                      matrix itself.

        """
        if coords is None:

            # If there are no transform actions, we have no problem.
            if len(self.image_geometry.transform_actions) == 0 and self.image_geometry.pixel_aspectratio == 1:
                coords = self.image_coords

            elif self.image_geometry.get_display_shape() != self.image_geometry.get_original_shape():

                if image.shape == self.image_geometry.get_display_shape()[::-1]:
                    coords = 'Display'
                elif image.shape == self.image_geometry.get_original_shape()[::-1]:
                    coords = 'Original'
                else:
                    raise ValueError(
                        'Input image has an unexpected shape! Got {:d}x{:d}; expected {:d}x{:d} or {:d}x{:d}'.format(
                            image.shape[1], image.shape[0], self.image_geometry.get_display_shape[1],
                            self.image_geometry.get_display_shape[0], self.image_geometry.get_original_shape[1],
                            self.image_geometry.get_original_shape[0]))
            else:
                raise Exception(
                    'Cannot determine image orientation automatically; please provide the "coords" input argument.')

        if coords.lower() == 'display' and self.image_coords.lower() == 'original':
            im_out = self.image_geometry.display_to_original_image(image)

        elif coords.lower() == 'original' and self.image_coords.lower() == 'display':
            im_out = self.image_geometry.original_to_display_image(image)

        else:
            im_out = image.copy()

        if self.binning > 1:
            im_out = _bin_image(im_out, self.binning, bin_func=np.mean)

        elif self.binning < 1:
            raise Exception('This matrix has binning < 1 which is not really meaningful. '
                            'Set binning =>1 before trying to use this matrix.')

        im_out = im_out.reshape(im_out.size, order=self.pixel_order)

        return sparse.csr_matrix(im_out[self.pixel_mask.reshape(self.pixel_mask.size, order=self.pixel_order) == True])

    def _save_npz(self, filename):
        """
        Save the geometry matrix in compressed NumPy binary format.
        """
        coo_data = self.data.tocoo()

        np.savez_compressed(filename,
                            mat_row_inds=coo_data.row,
                            mat_col_inds=coo_data.col,
                            mat_data=coo_data.data,
                            mat_shape=self.data.shape,
                            grid_verts=self.grid.vertices,
                            grid_cells=self.grid.cells,
                            grid_wall=self.grid.wall_contour,
                            binning=self.binning,
                            pixel_order=self.pixel_order,
                            pixel_mask=self.pixel_mask,
                            history=self.history,
                            grid_type=self.grid.__class__.__name__,
                            im_transforms=self.image_geometry.transform_actions,
                            im_px_aspect=self.image_geometry.pixel_aspectratio,
                            im_shape=self.pixel_mask.shape[::-1],
                            im_coords=self.image_coords,
                            inv_emis=self.inv_emis
                            )

    def _load_npz(self, filename, flow_matrix=True):
        """
        Load a geometry matrix from a compressed NumPy binary file
        """
        f = np.load(filename, allow_pickle=True)

        if flow_matrix:
            self.binning = float(f['binning'])
            self.pixel_order = str(f['pixel_order'])
            self.history = f['history'].item()
            self.pixel_mask = f['pixel_mask']
            self.image_coords = str(f['im_coords'])
            self.image_geometry = CoordTransformer()
            self.image_geometry.set_transform_actions(f['im_transforms'])
            self.image_geometry.set_pixel_aspect(f['im_px_aspect'], relative_to='Original')
            self.image_geometry.set_image_shape(*self.binning * np.array(self.pixel_mask.shape[::-1]),
                                                coords=self.image_coords)
            self.inv_emis = f['inv_emis']

            self.grid = PoloidalVolumeGrid(f['grid_verts'], f['grid_cells'], f['grid_wall'], src=self.history['grid'])

        self.data = sparse.csr_matrix((f['mat_data'], (f['mat_row_inds'], f['mat_col_inds'])), shape=f['mat_shape'])

    def _save_matlab(self, filename):
        """
        Save the geometry matrix in MATLAB format.
        """
        io.savemat(filename,
                         {'geom_mat': self.data,
                          'grid_verts': self.grid.vertices,
                          'grid_cells': self.grid.cells,
                          'grid_wall': self.grid.wall_contour,
                          'binning': self.binning,
                          'pixel_order': self.pixel_order,
                          'sightline_history': self.history['los'],
                          'matrix_history': self.history['matrix'],
                          'grid_history': self.history['grid'],
                          'pixel_mask': self.pixel_mask,
                          'grid_type': self.grid.__class__.__name__,
                          'im_transforms': np.array(self.image_geometry.transform_actions, dtype=np.object),
                          'im_px_aspect': self.image_geometry.pixel_aspectratio,
                          'im_coords': self.image_coords
                          }
                         )

    def _load_matlab(self, filename):
        """
        Load geometry matrix from a MATLAB file.
        """
        f = io.loadmat(filename)

        self.binning = float(f['binning'][0, 0])
        self.pixel_order = str(f['pixel_order'][0])
        self.pixel_mask = f['pixel_mask']
        self.history = {'los': str(f['sightline_history'][0]), 'grid': str(f['grid_history'][0]),
                        'matrix': str(f['matrix_history'][0])}
        self.image_coords = str(f['im_coords'][0])

        self.grid = PoloidalVolumeGrid(f['grid_verts'], f['grid_cells'], f['grid_wall'], src=self.history['grid'])
        self.data = f['geom_mat'].tocsr()

        self.image_geometry = CoordTransformer()
        self.image_geometry.set_transform_actions(f['im_transforms'])
        self.image_geometry.set_pixel_aspect(f['im_px_aspect'], relative_to='Original')
        self.image_geometry.set_image_shape(*self.binning * np.array(self.pixel_mask.shape[::-1]), coords=self.image_coords)

    @classmethod
    def fromfile(cls, filename, flow_matrix=True):
        """
        Load a saved geometry matrix from disk.

        Parameters:

            filename (str)  : File name to load from. Can be a NumPy (.npz), MATLAB (.mat) or \
                              zipped ASCII (.zip) file.

        Returns:

            calcam.GeometryMatrix : Loaded geometry matrix.

        """
        geommat = cls(None, None)

        try:
            fmt = filename.split('.')[1:][-1]
        except IndexError:
            raise ValueError(
                'Given file name does not include file extension; extension must be specified to determine file type!')

        if fmt == 'npz':
            geommat._load_npz(filename, flow_matrix=flow_matrix)
        elif fmt == 'mat':
            geommat._load_matlab(filename)
        else:
            raise ValueError(
                'File extension "{:s}" not understood; should be an "npz", "mat" or "zip" file.'.format(fmt))

        return geommat

    def _data_vector(self, raw_emis):

        if raw_emis is None:
            # Load in CIS intensity data
            cis_image = CISImage(self.shot, self.frame)
            cis_data = cis_image.I0
            cis_time = cis_image.time
        else:
            cis_image = np.load(raw_emis)
            cis_data = cis_image['I0']
            cis_time = cis_image['time']

        emis_vector = self.geom_mat.format_image(cis_data)  # Construct data vector for CIS data

        return emis_vector, cis_time

    # Plot velocities along defined flux surface in 1D
    def flow_1D(self, psiN, flow_profile, gfile=None, shot=None, time=None, plot=False):

        """
        Extract flow velocities for specific points along a line of constant \
        Normalised Flux (psiN)

        Parameters:

            psiN (int)            : Normalised Flux value - this determines what data points \
                                    are considered.

            gfile (str)           : geqdsk file containing magnetic equilibrium data

            flow_profile (str)    : File containing SART solved flow profile data

        Returns:

            numpy.ndarray : Nx3 array containing the flow velocity, and R,Z coords of the point \
                            that velocity is from, where N is the number of usable points \
                            within the defined grid

        """

        if gfile:
            eq = equilibrium(gfile=gfile)  # Load in geqdsk file containing equilibrium data
        else:
            eq = equilibrium(device='MAST', shot=shot, time=time, with_bfield=True)

        flow_data = np.load(flow_profile)  # Load in SART solved flow profile

        # R,Z coordinates along a given flux surface
        flux_surface_coords = np.stack((eq.get_fluxsurface(psiN)[0], eq.get_fluxsurface(psiN)[1]), axis=-1)

        grid_bounds_min = self.grid.vertices.min(axis=0)
        grid_bounds_max = self.grid.vertices.max(axis=0)

        # Trim useless coords
        usable_coords = []
        for coords in flux_surface_coords:
            if grid_bounds_min[0] <= coords[0] <= grid_bounds_max[0] and \
                    grid_bounds_min[1] <= coords[1] <= grid_bounds_max[1]:
                usable_coords.append(coords)
        usable_coords = np.asarray(usable_coords)

        surface_data = np.zeros((usable_coords.shape[0],3))  # Array to hold velocity and coord data

        for i, point in enumerate(usable_coords):
            for j, cell in enumerate(self.grid.cells):

                R = point[0]
                Z = point[1]

                # Cell Boundaries
                cell_rmin = self.grid.vertices[cell[0]][0]
                cell_rmax = self.grid.vertices[cell[1]][0]
                cell_zmin = self.grid.vertices[cell[0]][1]
                cell_zmax = self.grid.vertices[cell[3]][1]

                # Check if coordinate falls within the cell boundaries
                if cell_rmin <= R <= cell_rmax and cell_zmin <= Z <= cell_zmax:
                    surface_data[i] = [flow_data['data'][j], R, Z]
                    break
                else:
                    surface_data[i] = [np.nan, R, Z]

        return surface_data


def calculate_geom_mat_elements(grid, b_field_funcs, rays):

    ray_start_coords = np.array(rays[:3])
    ray_end_coords = np.array(rays[3:])

    # Calculate the positions and cells that a sight line intersects
    positions, intersected_cells = grid.get_cell_intersections(ray_start_coords, ray_end_coords)

    if positions.shape[0] <= 1:
        return None
       
    b_field_coords, l_k_vectors = _get_b_field_coords(positions, ray_start_coords, ray_end_coords)

    # Ray may not interact with any cells, in which case no data is to be output
    if not isinstance(b_field_coords, np.ndarray):
        return None

    #Convert b_field coordinates to RZ for use in function later
    coords_in_RZ, theta = convert_xy_r(b_field_coords)

    # Using b-field functions to get the b-field at given points
    b_field_rtz = get_b_field_comp(b_field_funcs, coords_in_RZ)
    # Convert these b-field values back to xyz coordinate system
    b_field_xyz = convert_rt_xy(b_field_rtz, theta, coords_in_RZ)

    # As each b_field_xyz segment has the same l_k_vector these are just once
    # So b_field_xyz will be seg_factor times larger than l_k_vectors and therefore when
    # iterating later it is important to know where to take the b_field_xyz values from
    seg_factor = int(len(b_field_xyz)/len(l_k_vectors))
    b_dot_l = np.ndarray(shape=(len(l_k_vectors), 1))

    for i in range(len(l_k_vectors)):
        total_seg_val = 0
        for j in range(seg_factor):

            # Every seg_factor (i.e. 10) b_field_xyz values are associated with one l_k_vectors
            # value (l_k_vectors[i]), index ensures the right 10 values are being read
            index = seg_factor*i + j
            b_l = np.dot(b_field_xyz[index], l_k_vectors[i]) # Dot product of B-field and line vector

            total_seg_val += b_l

        b_dot_l[i] = total_seg_val

    #  Array for the cell numbers related to each b_dot_l value
    cell_no = np.zeros(positions.size-1)    
    in_cell = set()

    # Convert the lists of intersected cells in to sets for later convenience.
    intersected_cells = [set(cells) for cells in intersected_cells]

    for i in range(positions.size):

        if len(in_cell) == 0:
            # Entering the grid
            in_cell = intersected_cells[i]

        else:
            # Going from one cell to another         
            leaving_cell = list(intersected_cells[i] & in_cell)

            if len(leaving_cell) == 1:

                cell_no[i-1] = leaving_cell[0]
                in_cell = intersected_cells[i]
                in_cell.remove(leaving_cell[0])

    return b_dot_l, cell_no


def _get_b_field_coords(pos, ray_start, ray_end):

    ray_vector = ray_end - ray_start  # Vector pointing along the sightline
    ray_length = np.sqrt(np.sum(ray_vector**2))

    relative_position = pos/ray_length

    if len(relative_position):
        n_segs = len(relative_position) - 1
    #if len(pos):
     #   n_segs = len(pos) - 1
    else:
        return None, None

    b_field_coords = np.ndarray(shape=(n_segs, 10, 3))

    # Segment midpoints at which to take B-field values
    b_field_points = np.arange(0.05, 1, 0.1)

    l_k = []  # This variable represents the l_k vector in the formula for flow geometry matrix

    for i in range(n_segs):

        #  Start and end points of the the parts of each sight line that lie in each cell
        seg_start = ray_start + relative_position[i] * ray_vector
        seg_end = ray_start + relative_position[i + 1] * ray_vector

        #seg_start = ray_start + pos[i] * ray_vector
        #seg_end = ray_start + pos[i + 1] * ray_vector

        # Vector describing those parts
        seg_vector = seg_end - seg_start

        # Division by 10 to decrease the magnitude of the vector as the segment is only a tenth the size
        l_k.append(seg_vector*0.1)

        for j in range(10):

            b_field_coords[i][j] = np.asarray(seg_start + b_field_points[j]*seg_vector)

    l_k_vectors = np.asarray(l_k)

    return b_field_coords, l_k_vectors


def convert_xy_r(coords):

    b_field_theta = []
    b_field_coords = coords.reshape(-1, 3, order='C')

    b_field_RZ = np.ndarray(shape=(len(b_field_coords), 2))

    for i, row in enumerate(b_field_coords):

        b_field_RZ[i][0] = np.sqrt(row[0]**2 + row[1]**2)
        b_field_RZ[i][1] = row[2]

        theta = np.arctan2(row[1], row[0])

        b_field_theta.append(theta)

    return b_field_RZ, b_field_theta


def get_b_field_comp(b_funcs, coords):

    # Functions to calculate the
    bt = b_funcs[0]
    br = b_funcs[1]
    bz = b_funcs[2]

    b_field_comp = np.ndarray(shape=(len(coords), 3))

    for i, row in enumerate(coords):

        b_field_comp[i][0] = br(row[0], row[1])
        b_field_comp[i][1] = bt(row[0], row[1])
        b_field_comp[i][2] = bz(row[0], row[1])

    return b_field_comp


def convert_rt_xy(comps, theta, coords):

    b_field_xyz = np.ndarray(shape=comps.shape)

    for i, (b_field, theta, rz_coords) in enumerate(zip(comps, theta, coords)):

        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array(((c, -rz_coords[0]*s, 0), (s, rz_coords[0]*c, 0), (0, 0, 1)))  # Rotation Matrix

        b_field_xyz[i] = np.matmul(rot_mat, b_field.transpose()).transpose()

    return b_field_xyz


def _weighting_matrix(data, inv_emis, verbose):

    print("Calculating Weighting Matrix...")
    # # Generate the weighting matrix data
    weight_rowinds = []
    weight_colinds = []
    weight_values = []

    denom = data @ inv_emis # Matrix multiply to calculate the denominator of the weighting fraction

    if verbose:
        # Progress bar indicating how much of the weighting matrix has been completed
        progressbar = Bar('Weighting Matrix', max=data.shape[0], suffix='%(percent)d%%')

    # Loop over each row, extracting the non-zero columns and calculating the weighting value at
    # that row and cell value
    for i, row in enumerate(data):

        row_data = sparse.find(row)
        cols = row_data[1]
        values = row_data[2]

        if cols.shape[0] == 0:
            continue

        for index, data in zip(cols, values):
            if inv_emis[index] != 0:
                weight_rowinds.append(i)
                weight_colinds.append(index)
                weight_values.append(inv_emis[index] / denom[i])
        if verbose:
            progressbar.next()
    if verbose:
        progressbar.finish()

    # Reshape array and combine to a sparse matrix
    weight_rowinds = np.asarray(weight_rowinds).reshape(-1, )
    weight_colinds = np.asarray(weight_colinds).reshape(-1, )
    weight_values = np.asarray(weight_values).reshape(-1, )

    return weight_rowinds, weight_colinds, weight_values


