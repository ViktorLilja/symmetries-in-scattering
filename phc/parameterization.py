import numpy as np
import skimage as ski

_HEXAGON_LATTICE_PARAM = 1000 # Hexagon side length in nm

def smooth_polygon_from_radii(radii, n_fold=4, n_interp=5, simplification_tol=0, theta0=0):
    """
        Returns a closed polygon with C_{n_fold} symmetry parametrized
        by its radius equally spaced angles given by radii.
        
        Result is interpolated by first doing a linear radius interpolation,
        then applying smoothing with a gaussian kernel with standard deviation 
        equal to the angle between given radii. The final number of points in 
        the polygon is len(radii) * n_fold * n_interp + 1 where the
        first and last point of polygon are the same.

        The returned polygon will not pass through the given radii unless
        n_interp = 1, but the result will be smooth.

        If simplification_tol is speficied, the final polygon will be simplified
        by removing some points and shifting others. The simplified polygon
        will match the original polygon within a tolerance set by this value.
        In this case, the number of points may be less than stated above.
    """

    n_point = len(radii) # Number of points used to generate polygon

    # Linear interpolation    
    theta         = theta0 + np.linspace(0, 2 * np.pi, n_point * n_fold + 1)                 # before interpolation, with last point
    theta_interp  = theta0 + np.linspace(0, 2 * np.pi, n_point * n_fold * n_interp + 1)[:-1] # after interpolation, without last point

    r_interp = np.tile(radii, reps=n_fold)              # Repeat around circle
    r_interp = np.concatenate([r_interp, r_interp[:1]]) # Close polygon
    r_interp = np.interp(theta_interp, theta, r_interp) # Interpolate

    # Apply smoothing
    r_repeat = np.tile(r_interp, reps=3) # Periodic padding
    kernel_x = np.linspace(-2, 2, 2*n_interp-1)
    kernel_y = np.exp(-0.5 * kernel_x**2)
    kernel_y /= np.sum(kernel_y)
    r_repeat = np.convolve(r_repeat, kernel_y, mode='same') # Smooth
    r_interp = r_repeat[len(r_interp):2*len(r_interp)]      # Remove padding

    # Close polygon
    r_interp     = np.concatenate([r_interp, r_interp[:1]])
    theta_interp = np.concatenate([theta_interp, theta_interp[:1]])

    # Create numpy array representing polygon
    polygon = np.zeros(shape=(len(theta_interp), 2))
    polygon[:,0] = r_interp * np.cos(theta_interp)
    polygon[:,1] = r_interp * np.sin(theta_interp)

    # Simplyfi polygon by removing points
    polygon = ski.measure.approximate_polygon(polygon, simplification_tol)

    return polygon


def calculate_radii(parameters):
    """
    Converts normalized design paramters to radius of control points in nm.
    """

    r_min = 0.2 * _HEXAGON_LATTICE_PARAM / 2 # Inner circle radius
    r_max = 0.9 * _HEXAGON_LATTICE_PARAM / 2 # Outer circle radius
    
    r_mid   = (r_max + r_min) / 2
    r_range = (r_max - r_min) / 2 
    radii   = r_mid + r_range * parameters

    return radii


def polygon_from_parameters(parameters):
    """
        Converts normalized design paramters to the hole shape in the form of a 
        where the unit of the vertices are in nm.
    """

    radii = calculate_radii(parameters)

    polygon = smooth_polygon_from_radii(radii, n_fold=1, n_interp=4, simplification_tol=2, theta0=2*np.pi/24)

    return polygon


def visualize(ax, parameters, plot_control_points=False):
    """
    Given the axis of a matplotlib figure and an array of design parameters,
    plots an illustration of the PhC unit cell corresponding to the design 
    parameters on the axis.
    """

    polygon = polygon_from_parameters(parameters)

    d = _HEXAGON_LATTICE_PARAM / np.sqrt(3)
    hexagon = np.array([
            [d, 0], 
            [d/2, np.sqrt(3)/2*d], 
            [-d/2, np.sqrt(3)/2*d], 
            [-d, 0], 
            [-d/2, -np.sqrt(3)/2*d], 
            [d/2, -np.sqrt(3)/2*d],
            [d, 0]
    ])

    ax.fill(hexagon[:,0], hexagon[:,1], "lightgray")
    ax.plot(hexagon[:,0], hexagon[:,1], "k")
    ax.fill(polygon[:,0], polygon[:,1], "w")
    ax.plot(polygon[:,0], polygon[:,1], "k")

    if plot_control_points:
        radii = calculate_radii(parameters)
        theta = np.linspace(0, 2*np.pi, 13)[:-1] + 2 * np.pi / 24
        points = np.array([radii * np.cos(theta), radii*np.sin(theta)]).T

        points_inner = np.array([calculate_radii(-np.ones(12)) * np.cos(theta), calculate_radii(-np.ones(12))*np.sin(theta)]).T
        points_outer = np.array([calculate_radii(np.ones(12)) * np.cos(theta), calculate_radii(np.ones(12))*np.sin(theta)]).T

        ax.plot(points[:,0], points[:,1], "o", color="k", markersize=4)
        ax.plot(np.stack((points_inner, points_outer))[:,:,0], np.stack((points_inner, points_outer))[:,:,1], "-", color="k")

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)