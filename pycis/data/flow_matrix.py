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
from functools import partial


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

        b_field_funcs = get_Bfield(self.shot, self.frame)  # Functions to calculate B-field components at a given point

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
        else:
            if grid:
                self.grid = copy.copy(grid)
            else:
                print('Generating default square grid for MAST.')
                self.grid = calcam.gm.squaregrid('MAST', cell_size=1e-2, zmax=-0.6)

            self.geom_mat = calcam.gm.GeometryMatrix(self.grid, self.raydata)

        self.geom_data = self.geom_mat.data

        # Solve y = Ax + b for x, the inverted emissivity matrix
        if inv_emis is None:
            self.inv_emis = sart.solve(self.geom_data, self._data_vector())[0]

        ray_start_coords = self.raydata.ray_start_coords.reshape(-1, 3, order=self.order)
        ray_end_coords = self.raydata.ray_end_coords.reshape(-1, 3, order=self.order)

        # Number of grid cells and sight lines
        n_cells = self.grid.n_cells
        n_los = self.raydata.x.size

        # Shuffling indices results in a better time remaining estimation
        inds = list(range(n_los))
        random.shuffle(inds)

        # Multi-threadedly loop over each sight-line in raydata and calculate the positions at which
        # each interacts with a cell wall
        if calc_status_callback is not None:
            calc_status_callback('Calculating sight-line cell interactions using {:d} '
                                 'CPUs...'.format(config.n_cpus))

        last_status_update = 0.

        rays = np.hstack((ray_start_coords[inds, :], ray_end_coords[inds, :]))  # Combine coords for imap
        self.mag_length = []

        with multiprocessing.Pool(config.n_cpus) as cpupool:
            calc_status_callback(0.)
            for i, data in enumerate(cpupool.imap(partial(calculate_geom_mat_elements, self.grid, b_field_funcs), rays, 10)):

                self.mag_length.append(data)  # Store ray interaction data

                if time.time() - last_status_update > 1. and calc_status_callback is not None:
                    calc_status_callback(float(i) / n_los)
                    last_status_update = time.time()

        if calc_status_callback is not None:
            calc_status_callback(1.)

    def _data_vector(self):

        # Load in CIS intensity data
        cis_image = CISImage(self.shot, self.frame)
        cis_data = cis_image.I0

        emis_vector = self.geom_mat.format_image(cis_data)  # Construct data vector for CIS data

        return emis_vector


def calculate_geom_mat_elements(grid, b_field_funcs, rays):

    ray_start_coords = np.array(rays[:3])
    ray_end_coords = np.array(rays[3:])

    # Calculate the positions and cells that a sight line intersects
    positions, interacted_cells = grid.get_cell_intersections(ray_start_coords, ray_end_coords)

    b_field_coords, l_k_vectors = _get_b_field_coords(positions, ray_start_coords, ray_end_coords)
    coords_in_RZ, theta = _convert_xy_r(b_field_coords)

    b_field_rtz = _get_b_field_comp(b_field_funcs, coords_in_RZ)
    b_field_xyz = _convert_rt_xy(b_field_rtz, theta)

    seg_factor = len(b_field_xyz)/len(l_k_vectors)
    b_dot_l = np.ndarray(shape=(len(l_k_vectors),1))

    for i in range(len(l_k_vectors)):
        total_seg_val = 0
        for j in range(len(b_field_xyz)):

            index = seg_factor*i + j
            b_l = np.dot(b_field_xyz[index], l_k_vectors[i])

            total_seg_val += b_l

        b_dot_l[i] = total_seg_val

    return b_dot_l, interacted_cells


def _get_b_field_coords(pos, ray_start, ray_end):

    ray_vector = ray_end - ray_start  # Vector pointing along the sightline
    ray_length = np.sqrt(np.sum((ray_end - ray_start)**2 ))

    relative_position = pos/ray_length
    n_interactions = len(relative_position)
    b_field_coords = np.ndarray(shape=(n_interactions, 10))

    # Segment midpoints at which to take B-field values
    b_field_points = np.arange(0.05, 1, 0.1)

    l_k = []  # This variable represents the l_k vector in the formula for flow geometry matrix

    for i in range(n_interactions-1):

        #  Start and end points of the the parts of each sight line that lie in each cell
        seg_start = ray_start + relative_position[i] * ray_vector
        seg_end = ray_start + relative_position[i + 1] * ray_vector

        # Vector describing those parts
        seg_vector = seg_end - seg_start

        l_k.append(seg_vector*0.1)

        for j in range(10):

            b_field_coords[i][j] = ray_start + b_field_points[j]*seg_vector

    l_k_vectors = np.asarray(l_k)

    return b_field_coords, l_k_vectors


def _convert_xy_r(coords):

    b_field_theta = []
    b_field_coords = coords.reshape(-1, 3, order='C')

    b_field_RZ = np.ndarray(shape=(len(b_field_coords), 2))

    for i, row in enumerate(b_field_coords):

        b_field_RZ[i][0] = np.sqrt(row[0]**2 + row[1]**2)
        b_field_RZ[i][1] = row[2]

        theta = np.arctan2(row[1], row[0])

        if row[1] < 0:
            theta = 2*np.pi + theta

        b_field_theta.append(theta)

    return b_field_RZ, b_field_theta


def _get_b_field_comp(b_funcs, coords):

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


def _convert_rt_xy(comps, theta):

    b_field_xyz = np.ndarray(shape=comps.shape)

    for i, (b_field, theta) in enumerate(zip(comps, theta)):

        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))  # Rotation Matrix

        b_field_xyz[i] = np.matmul(rot_mat, b_field_xyz.transpose()).transpose()

    return b_field_xyz


