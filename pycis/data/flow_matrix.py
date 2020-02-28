import numpy as np
import calcam
from calcam.gm import misc
import matplotlib.pyplot as plt
import multiprocessing
from pyEquilibrium.equilibrium import equilibrium
from pycis.solvers import sart
from .get import CISImage

class FlowGeoMatrix():

	'''
	Class to represent a flow geometry matrix and associated metadata.
    
	The matrix itself can be accessed in the `data` attribute, where it is
	stored as a sparse matrix using the scipy.sprase.csr_matrix class.

	Parameters:

	shot (int)			     : MAST shot being studied

	frame (int)			     : Frame required from CIS camera

        raydata (calcam.RayData or file)     : RayData object or Saved Ray data file for the 
					       camera to be inverted

	geom_mat (calcam.gm.GeometryMatrix or file)  : 
					       GeometryMatrix object or saved Geometry Matrix 
 					       (NumPy or Matlab format)

        grid (calcam.gm.PoloidalVolumeGrid)  : PoloidalVolumeGrid object to use to generate 
					       geom_mat if geom_mat = None

	inv_emis (tuple)		     : Tuple containing inverted emissivity as a matrix, x 
					       and Array of length n_iterations indicating convergence 
					       behaviour. Calculated by solving y = Ax + b.

        pixel_order (str)                    : What pixel order to use when flattening \
                                               the 2D image array in to the 1D data vector. \
                                               Default 'C' goes left-to-right, then row-by-row (NumPy
					       default), alternatively 'F' goes top-to-bottom, 
					       then column-by-column (MATLAB default).
        
        calc_status_callback (callable)      : Callable which takes a single argument, which will be
					       called with status updates about the calculation. 
					       It will be called with either a string for textual 
					       status updates or a float from 0 to 1 specifying 
					       the progress of the calculation. By default, status 
					       updates are printed to stdout.  If set to None, no 
					       status updates are issued.

	'''

	def __init__(self, shot, frame, raydata=None, geom_mat=None, grid=None, inv_emis=None, pixel_order='C',calc_status_callback = misc.LoopProgPrinter().update):

		if raydata:
			if isinstance(raydata, str):
				raydata = calcam.RayData(raydata)
		else:
			raise Exception('No RayData object or saved file provided. Please provide a file or run calcam.raycast_sightlines()')

		if geom_mat:
			if isinstance(geom_mat, str):
                                geom_mat = calcam.gm.GeometryMatrix.fromfile(geom_mat)
		else:
			if grid:
				geom_mat = calcam.gm.GeometryMatrix(grid,raydata)
			else:
				raise Exception('PoloidalVolumeGrid object not provided for grid.')
		
		if inv_emis is None:
			self.inv_emis = sart.solve(geom_mat.data, self._data_vector(shot,frame, geom_mat))
	
	def _data_vector(self, shot, frame, geom_mat):

		cis_image = CISImage(shot, frame)
		cis_data = cis_image.I0

		emis_vector = geom_mat.format_image(cis_data)

		return emis_vector

		
