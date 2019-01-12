import numpy as np
import pickle
import pycis
import os.path


class Lens(object):
    """ base class for (thin) lens """

    def __init__(self, focal_length):

        self.focal_length = focal_length



