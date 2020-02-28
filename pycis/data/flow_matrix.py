import copy
import random
import time
import numpy as np
import calcam
from calcam.gm import misc, config
import matplotlib.pyplot as plt
import multiprocessing
from pyEquilibrium.equilibrium import equilibrium
from pycis.solvers import sart
from .get import CISImage, get_Bfield


class FlowGeoMatrix:
    """
    Class to represent a flow geometry matrix and associated metadata.

    The matrix itself can be accessed in the `data` attribute, where it is
    stored as a sparse matrix using the scipy.sprase.csr_matrix class.

    The matrix, denoted by Nu_ij, and is constructed as per eq. 4.6.6 [1], where
    epsilon_0,j is inv_emis[0], Epsilon_ij' is geom_mat.data, B_k djfd and l_k is ksjdksj

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

        inv_emis (tuple)		: Tuple containing inverted emissivity as a matrix, x
                                    and Array of length n_iterations indicating convergence
                                    behaviour. Calculated by solving y = Ax + b.

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
    """

    def __init__(self, shot, frame, raydata=None, geom_mat=None, grid=None, inv_emis=None, pixel_order='C',
                 calc_status_callback=misc.LoopProgPrinter().update):

        self.shot = shot
        self.frame = frame
        self.order = pixel_order

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
        else:
            if grid:
                self.geom_mat = calcam.gm.GeometryMatrix(grid, raydata)
            else:
                raise Exception('PoloidalVolumeGrid object not provided for grid.')

        self.grid = copy.copy(self.geom_mat.grid)
        self.geom_data = self.geom_mat.data

        # Solve y = Ax + b for x, the inverted emissivity matrix
        if inv_emis is None:
            self.inv_emis = sart.solve(self.geom_data, self._data_vector())[0]

        ray_start_coords = self.raydata.ray_start_coords.reshape(-1, 3, order=self.order)
        ray_end_coords = self.raydata.ray_end_coords.reshape(-1, 3, order=self.order)

        # Number of grid cells and sight lines
        n_cells = self.grid.n_cells
        n_los = self.raydata.x.size

        inds = list(range(n_los))
        random.shuffle(inds)
        last_status_update = 0.

        self.ray_cell_data = []

        with multiprocessing.Pool(config.n_cpus) as cpupool:
            calc_status_callback(0.)
            for i, data in enumerate(cpupool.imap(self._get_ray_cell_interactions, np.hstack((ray_start_coords[inds,:], ray_end_coords[inds,:])), 10)):

                self.ray_cell_data.append(data)

                if time.time() - last_status_update > 1. and calc_status_callback is not None:
                    calc_status_callback(float(i) / n_los)
                    last_status_update = time.time()

    def _data_vector(self):

        # Load in CIS intensity data
        cis_image = CISImage(self.shot, self.frame)
        cis_data = cis_image.I0

        emis_vector = self.geom_mat.format_image(cis_data)  # Construct data vector for CIS data

        return emis_vector

    def _get_ray_cell_interactions(self, rays):

        ray_start_coords = np.array(rays[:3])
        ray_end_coords = np.array(rays[3:])

        positions, interacted_cells = self.grid.get_cell_intersections(ray_start_coords, ray_end_coords)

        return positions, interacted_cells


