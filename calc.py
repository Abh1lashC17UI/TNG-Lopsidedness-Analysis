# Import standard libraries
import numpy as np
import numba as nb
from scipy.special import sph_harm
from astropy.coordinates import spherical_to_cartesian
from scipy.interpolate import interp1d

def list_dict(keys: list, type: type = list):

    """ Make new dictionary with keys bound to empty lists or dicts.
    Parameters:
        keys: list of keys to use in dictionary
        type: type of empty container to initialise with
    Returns:
        list_dict: dictionary initialised with empty lists or dicts
    """

    list_dict = {}
    for key in keys:
        if type is list:
            list_dict[key] = []
        elif type is dict:
            list_dict[key] = {}

    return list_dict


# === ISOTROPIC METRIC GENERATION =================================================================

seed=0

@nb.njit(nb.float64[:,:](nb.int32))#, nb.int64))
def isovecs_3d(num):

    """ Generates isotropically distributed 3D unit vectors.
    Parameters:
        num: number of vectors to generate
        seed: seed for random generation (integer)
    Returns:
        vectors: (N,3) array of isotropically distributed vectors
            * vectors.reshape(n_sys, n_sys, n_dim)
    """

    np.random.seed(seed)

    theta = np.random.uniform(0, 2*np.pi, num)
    phi = np.arccos(2 * np.random.uniform(0, 1, num) - 1)

    vectors = np.empty((num, 3))

    vectors[:,0] = np.sin(phi) * np.cos(theta)
    vectors[:,1] = np.sin(phi) * np.sin(theta)
    vectors[:,2] = np.cos(phi)

    return vectors

@nb.njit(nb.float64[:,:](nb.int32))#, nb.int64))
def isovecs_2d(num):

    """ Generates isotropically distributed 3D unit vectors.
    Parameters:
        num: number of vectors to generate
        seed: seed for random generation (integer)
    Returns:
        vectors: (N,2) array of isotropically distributed vectors
            * vectors.reshape(n_sys, n_sys, n_dim)
    """

    np.random.seed(seed)

    theta = np.random.uniform(0, 2*np.pi, num)

    vectors = np.empty((num, 2))

    vectors[:,0] = np.sin(theta)
    vectors[:,1] = np.cos(theta)

    return vectors

@nb.njit(nb.float64[:,:](nb.int32, nb.int64))
def isovecs_sph(num, seed):

    """ Generates isotropic 3D spherical coordinates.
    Parameters:
        num: number of vectors to generate
        seed: seed for random generation (integer)
    Returns:
        vectors: (N,2) array of longitude [0, 2pi] and latitude [0, pi]
    """

    np.random.seed(seed)

    vectors = np.empty((num, 2))

    vectors[:,0] = np.random.uniform(0, 2*np.pi, num)
    vectors[:,1] = np.arccos(1 - 2 * np.random.uniform(0, 1, num))

    return vectors



# === ANGLE CALCULATION ===========================================================================

@nb.njit(nb.float64(nb.float64[:], nb.float64[:]))
def angle_single_2d(vec, ref):

    """ Calculates the smallest angle between a vector and a reference vector.
    Parameters:
        vec: (2,) array of vector of interest
        ref: reference (2,) vector to calculate angle from
    Returns:
        angle: angle between vectors in [0, pi] radians
    """

    # Calculate magnitude of two vectors
    mag_vec = np.sqrt(vec[0]**2 + vec[1]**2)
    mag_ref = np.sqrt(ref[0]**2 + ref[1]**2)

    # Derive angle from dot product
    dot_product = vec[0] * ref[0] + vec[1] * ref[1]
    angle = np.arccos(dot_product / (mag_vec * mag_ref))

    return angle

@nb.njit(nb.float64[:](nb.float64[:,:], nb.float64[:]))
def angle_multi_2d(vecs, ref):

    """ Calculates the smallest angle between multiple vectors and a reference vector.
    Parameters:
        vecs: (N,2) array of vectors of interest
        ref: reference (2,) vector to calculate angle from
    Returns:
        angles: array of angles between vectors in [0, pi] radians
    """

    # Calculate magnitude of reference vector
    mag_ref = np.sqrt(ref[0]**2 + ref[1]**2)

    # Generate containers for vector magnitudes
    num_vecs = len(vecs[:,0])
    mag_vecs = np.empty(num_vecs)

    # Find magnitudes of vectors of interest
    for i in nb.prange(num_vecs):
        mag_vecs[i] = np.sqrt(vecs[i,0]**2 + vecs[i,1]**2)

    # Generate container for angles
    angles = np.empty(num_vecs)

    # Derive angles from dot product
    for i in nb.prange(num_vecs):
        dot_product = vecs[i,0] * ref[0] + vecs[i,1] * ref[1]
        angles[i] = np.arccos(dot_product / (mag_vecs[i] * mag_ref))

    return angles

@nb.njit(nb.float64(nb.float64[:], nb.float64[:]))
def angle_single_3d(vec, ref):

    """ Calculates the smallest angle between a vector and a reference vector.
    Arguments:
        vec: (3,) array of vector of interest
        ref: reference (3,) vector to calculate angle from
    Returns:
        angles: array of angles between vectors in [0, pi] radians
    """

    # Calculate magnitudes of two vectors
    mag_vec = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    mag_ref = np.sqrt(ref[0]**2 + ref[1]**2 + ref[2]**2)

    # Derive angle from dot product
    dot_product = vec[0]*ref[0] + vec[1]*ref[1] + vec[2]*ref[2]
    angle = np.arccos(dot_product / (mag_vec * mag_ref))

    return angle

@nb.njit(nb.float64[:](nb.float64[:,:], nb.float64[:]))
def angle_multi_3d(vecs, ref):

    """ Calculates the smallest angle between multiple vectors and a reference vector.
    Arguments:
        vecs: (N,3) array of vector of interest
        ref: reference (3,) vector to calculate angle from
    Returns:
        angles: array of angles between vectors in [0, pi] radians
    """

    # Calculate magnitude of reference vector
    mag_ref = np.sqrt(ref[0]**2 + ref[1]**2 + ref[2]**2)

    # Generate containers for vector magnitudes
    num_vecs = len(vecs[:,0])
    mag_vecs = np.empty(num_vecs)

    # Find magnitudes of vectors of interest
    for i in range(num_vecs):
        mag_vecs[i] = np.sqrt(vecs[i,0]**2 + vecs[i,1]**2 + vecs[i,2]**2)

    # Generate container for angles
    angles = np.empty(num_vecs)

    # Derive angles from dot product
    for i in range(num_vecs):
        dot_product = vecs[i,0]*ref[0] + vecs[i,1]*ref[1] + vecs[i,2]*ref[2]
        angles[i] = np.arccos(dot_product / (mag_vecs[i] * mag_ref))

    return angles


# === RADIAL DISTRIBUTION MODELS ==================================================================

def radial_model(r_min, r_max, alpha):

    """ A very simple toy model of satellite radial distributions.
    Parameters:
        r_min: minimum distance a satellite should be located
        r_max: maximum distance a satellite should be located
        alpha: shape parameter (a<1 is steep near r_min)
    Returns:
        inv_cdf: interpolation function for inverse CDF
    """

    # Generate radial distribution
    r = np.linspace(r_min, r_max, 1000)
    cdf = ((r - r_min) / (r_max - r_min)) ** alpha

    # Create interpolation function for inverse CDF
    inv_cdf = interp1d(cdf, r, bounds_error=False, fill_value=(r_min, r_max))

    return inv_cdf

def sample_radial(pos, inv_cdf, seed):

    """ Samples given radial distribution and generates positions.
    Parameters:
        pos: (Nsys,N,3), (Nsys,N,2), (N,3) or (N,2) unit vector positions.
        inv_cdf: interpolation function for inverse CDF
        seed: seed for randomness (integer)
    Returns:
        pos: (N,3) or (N,2) satellite positions after applying the distribution
    """

    np.random.seed(seed)

    # Detect whether system or set of systems
    if pos.ndim == 2: pos = np.array([pos])

    # For each system
    for i in range(len(pos)):

        # Sample and apply distances
        radii = inv_cdf(np.random.uniform(0, 1, len(pos[i])))
        pos[i] = pos[i] * radii[:,None]

    # Collapse if only one system
    if len(pos) == 1: pos = pos[0]

    return pos
