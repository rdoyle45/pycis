import calcam
import numpy as np
from pycis import data
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """
    Add an 3d arrow to an `Axes3D` instance.
    """

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
    setattr(Axes3D, 'arrow3D', _arrow3D)


def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(-height_z, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid


def check_Bfield_coords(rays, shot, time, plot=False):

    """
    Function to check the direction of the magnetic field vector to ensure the correct sign conventions are
    being used.

    Parameters:

        rays (calcam.Raydata or file)   :   File output from calcam containing the sightlines for a specific shot

        shot (int)                      :   Shot number

        time (float)                    :   Time of the frame being studied

    """

    raydata = calcam.RayData(rays)  # Load raydata for shot

    bt, br, bz = data.get.get_Bfield(shot, time)  # Get B-field data functions from pyEquilibrium

    test_set = random.sample(range(0, raydata.x.size), 10)  # 10 random integers corresponding to 10 test sightlines

    # Load test sightlines from raydata
    ray_start_coords = raydata.ray_start_coords.reshape(-1, 3)[test_set]
    ray_end_coords = raydata.ray_end_coords.reshape(-1, 3)[test_set]

    midpoints = (ray_start_coords[:]+ray_end_coords[:])/2  # Find the midpoint coords for each sightline

    # Remove points outside vessel
    del_rows = []
    for i, row in enumerate(midpoints):
        if any(row[:] < -2) or any(row[:] > 2):
            del_rows.append(i)
    midpoints = np.delete(midpoints, del_rows, axis=0)

    midpoints_cyl = np.ndarray((len(midpoints), 3))  # Array to hold the midpoint components in cylindrical coords

    # convert cartesian midpoints to cylindrical
    midpoints_cyl[:, 0] = np.sqrt(midpoints[:, 0]**2 + midpoints[:, 1]**2)
    midpoints_cyl[:, 1] = np.arctan2(midpoints[:, 1], midpoints[:, 0])
    midpoints_cyl[:, 2] = midpoints[:, 2]

    b_field_cyl = np.ndarray((len(midpoints), 3))  # Array to hold the b-field components in cylindrical coords

    # Get B-field components from pyEquilbrium (cylindrical coords)
    for i, coords in enumerate(midpoints_cyl):

        b_field_cyl[i, 0] = br(coords[0], coords[2])
        b_field_cyl[i, 1] = bt(coords[0], coords[2])
        b_field_cyl[i, 2] = bz(coords[0], coords[2])

    b_field_cart = np.ndarray((len(midpoints), 3))  # Array to hod B-field components in cartesian coords

    for i, (point_coords, b_field_comp) in enumerate(zip(midpoints_cyl, b_field_cyl)):

        c, s = np.cos(point_coords[1]), np.sin(point_coords[1])
        rot_mat = np.array(((c, -point_coords[0]*s, 0), (s, point_coords[0]*c, 0), (0, 0, 1)))  # Rotation Matrix

        b_field_cart[i] = np.matmul(rot_mat, b_field_comp.T).T

    if plot:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)

        # Plot the midpoints and B-field vectors at those points
        for row, b_field in zip(midpoints, b_field_cart):
            ax.arrow3D(row[0], row[1], row[2],
                       b_field[0], b_field[1], b_field[2],
                       mutation_scale=20,
                       arrowstyle="-|>",
                       linestyle='dashed')

        # Plot a cylinder to replicate MAST
        Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, 2, 2)
        ax.plot_surface(Xc, Yc, Zc, alpha=0.1)
        ax.set_title('3D Arrows Demo')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig.tight_layout()





