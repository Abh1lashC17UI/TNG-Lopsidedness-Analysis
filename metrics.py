# Import standard libraries
import numpy as np
import numba as nb
import multiprocessing as mp
from tqdm import tqdm

# Import user libraries
#from .
import calc

class Centroid:

    name = "centroid"
    radial = True
    true_bounds = (1, 0)  # Max, min anisotropy
    grid_bounds = (0.95, 0)  # Values possible in isotropy (extrapolate at flat value beyond)

    @staticmethod
    @nb.njit(nb.float64(nb.float64[:,:]))
    def eval(pos):

        """ Routine to find centroid metric of given system (both 2D and 3D).
        Parameters:
            pos: (N,3) or (N,2) position array (with host at origin)
        Returns:
            shift: magnitude of centroid shift normalised by RMS distance
            trueshift: absolute centroid shift
            vector: direction of centroid shift
        """

        # Find centroid of satellite distribution
        num_sat, num_dim = pos.shape
        centroid = np.empty(num_dim)

        for i in range(num_dim):
            centroid[i] = np.mean(pos[:,i])

        # Break down shift into magnitude and direction
        trueshift = np.sqrt(np.sum(centroid**2))
        vector = centroid / trueshift

        # Find root-mean-square distance of satellites
        dists_sq = np.empty(num_sat)
        for i in range(num_sat):
            dists_sq[i] = np.sum(pos[i,:]**2)
        d_rms = np.sqrt(np.mean(dists_sq))

        # Normalise centroid shift
        shift = trueshift / d_rms

        # Set return datatypes
        shift = np.float64(shift)
        trueshift = np.float64(trueshift)
        vector = vector.astype(np.float64)

        return shift #, trueshift, vector

    @staticmethod
    def batch(pos, cpu=None, pbar=False):

        """ Parallelizes the centroid metric for a large number of systems.
        Parameters:
            pos: (Nsys,N,3) or (Nsys,N,2) position array (with host at origin)
            cpu: number of CPU cores to use for multiprocessing (default: max-1)
        """

        if cpu is None: cpu = mp.cpu_count() - 1

        with mp.Pool(cpu) as pool:
            shift = pool.map(Centroid.eval,
                tqdm(pos, total=len(pos), desc="Finding centroid",disable=not pbar), chunksize=100)

        return np.array(shift)


class MRL:

    name = "mrl"
    radial = False
    true_bounds = (1, 0)
    grid_bounds = (0.95, 0)

    @staticmethod
    @nb.njit(nb.float64(nb.float64[:,:]))
    def eval(pos):

        """ Routine to find Mean Resultant Length (MRL) of given system.
        Parameters:
            pos: (N,3) or (N,2) position array (with host at origin)
        Returns:
            length: mean resultant length (between 0: isotropic, 1: directional)
            vector: (3,) or (2,) array of resultant vector unit direction
        """

        # Create empty data arrays
        num_sat, num_dim = pos.shape
        norm_pos = np.empty((num_sat, num_dim))
        vector_sum = np.empty(num_dim)

        # Normalise positions to unity
        for i in range(num_sat):
            mag = np.sqrt(np.sum(pos[i,:] ** 2))
            norm_pos[i,:] = pos[i,:] / mag

        # Find sum of all position vectors
        for i in range(num_dim):
            vector_sum[i] = np.sum(norm_pos[:,i])

        # Find mean resultant length of system
        sum_mag = np.sqrt(np.sum(vector_sum ** 2))
        length = sum_mag / num_sat
        vector = vector_sum / sum_mag

        # Set return datatypes
        length = np.float64(length)
        vector = vector.astype(np.float64)

        return length #, vector

    @staticmethod
    def batch(pos, cpu=None, pbar=False):

        """ Parallelizes the MRL metric for a large number of systems.
        Parameters:
            pos: (Nsys,N,3) or (Nsys,N,2) position array (with host at origin)
            cpu: number of CPU cores to use for multiprocessing (default: max-1)
        """

        if cpu is None: cpu = mp.cpu_count() - 1

        with mp.Pool(cpu) as pool:
            length = pool.map(MRL.eval,
                tqdm(pos, total=len(pos), desc="Finding MRL", disable=not pbar), chunksize=100)

        return np.array(length)


class Pairwise:

    name = "pairwise"
    radial = False
    true_bounds = (0, 100)
    grid_bounds = (40, 100)

    @staticmethod
    @nb.njit(nb.float64(nb.float64[:,:]))
    def eval(pos):

        """ Routine to find pairwise metric of given system.
        Parameters:
            pos: (N,3) or (N,2) position array (with host at origin)
        Returns:
            mean_angle: mean of angles between every satellite pair in degrees
        """

        # Find satellite count and dimensions
        num_sat, num_dim = pos.shape

        # Find start and increment indices
        start_index = np.zeros(num_sat)
        start_index[1:] = np.cumsum(np.arange(num_sat - 1, 0, -1))
        start_index = start_index.astype(np.int32)

        # Define container for pairwise angles
        angles = np.empty(int(num_sat * (num_sat - 1) / 2))

        # For each triangular cycle, calculate angles in cycle
        for i in nb.prange(num_sat - 1):

            if num_dim == 3:
                angles[start_index[i]:start_index[i+1]] = calc.angle_multi_3d(pos[i+1:], pos[i])

            if num_dim == 2:
                angles[start_index[i]:start_index[i+1]] = calc.angle_multi_2d(pos[i+1:], pos[i])

        # Set return datatype
        angles = np.rad2deg(angles).astype(np.float64)
        mean_angle = np.mean(angles)

        return mean_angle

    @staticmethod
    def batch(pos, cpu=None, pbar=False):

        """ Parallelizes the pairwise metric for a large number of systems.
        Parameters:
            pos: (Nsys,N,3) or (Nsys,N,2) position array (with host at origin)
            cpu: number of CPU cores to use for multiprocessing (default: max-1)
        """

        if cpu is None: cpu = mp.cpu_count() - 1

        with mp.Pool(cpu) as pool:
            mean_angle = pool.map(Pairwise.eval,
                tqdm(pos, total=len(pos), desc="Finding pairwise", disable=not pbar), chunksize=100)

        return np.array(mean_angle)


class TOI:

    name = "toi"
    radial = True
    true_bounds = (0, 1)
    grid_bounds = (0.05, 1)

    @staticmethod
    @nb.njit(nb.float64(nb.float64[:,:]))
    def eval3d(pos):

        """ Performs unweighted tensor-of-inertia fit to calculate 3D plane properties.
        Parameters:
            pos: (N,3) array of satellite positions
        Returns:
            ca: minor-to-major axis ratio
            ba: intermediate-to-major axis ratio
        """

        inertia_tensor = np.zeros((3, 3))
        num_sat, num_dim = pos.shape

        for vec in pos:

            vec = np.ascontiguousarray(vec)
            vecT = np.ascontiguousarray(vec.T)

            left = float(np.dot(vec, vecT)) * np.eye(3, 3)
            right = np.outer(vec, vec)
            inertia_tensor = inertia_tensor + left - right

        eig_vals, eig_vecs = np.linalg.eig(inertia_tensor)

        indices = np.argsort(eig_vals)

        norm_a = eig_vecs.T[indices[0]]
        norm_b = eig_vecs.T[indices[1]]
        norm_c = eig_vecs.T[indices[2]]

        mag_squared_a = np.sum(np.array([np.dot(np.ascontiguousarray(vec), norm_a) ** 2 for vec in pos]))
        mag_squared_b = np.sum(np.array([np.dot(np.ascontiguousarray(vec), norm_b) ** 2 for vec in pos]))
        mag_squared_c = np.sum(np.array([np.dot(np.ascontiguousarray(vec), norm_c) ** 2 for vec in pos]))

        mag_a = np.sqrt(mag_squared_a / num_sat)
        mag_b = np.sqrt(mag_squared_b / num_sat)
        mag_c = np.sqrt(mag_squared_c / num_sat)

        ca = np.float64(mag_c / mag_a)
        ba = np.float64(mag_b / mag_a)

        return ca #, ba

    @staticmethod
    @nb.njit(nb.float64(nb.float64[:,:]))
    def eval2d(pos):

        """ Performs unweighted tensor-of-inertia fit to calculate 2D plane properties.
        Parameters:
            pos: (N,2) array of satellite positions
        Returns:
            ba: minor-to-major axis ratio
        """

        num_sat, num_dim = pos.shape
        inertia_tensor = np.zeros((num_dim, num_dim))

        for vec in pos:

            vec = np.ascontiguousarray(vec)
            vecT = np.ascontiguousarray(vec.T)

            left = float(np.dot(vec, vecT)) * np.eye(num_dim, num_dim)
            right = np.outer(vec, vec)
            inertia_tensor = inertia_tensor + left - right

        eig_vals, eig_vecs = np.linalg.eig(inertia_tensor)

        indices = np.argsort(eig_vals)

        norm_a = eig_vecs.T[indices[0]]
        norm_b = eig_vecs.T[indices[1]]

        mag_squared_a = np.sum(np.array([np.dot(np.ascontiguousarray(vec), norm_a) ** 2 for vec in pos]))
        mag_squared_b = np.sum(np.array([np.dot(np.ascontiguousarray(vec), norm_b) ** 2 for vec in pos]))

        mag_a = np.sqrt(mag_squared_a / num_sat)
        mag_b = np.sqrt(mag_squared_b / num_sat)

        ba = np.float64(mag_b / mag_a)

        return ba

    @staticmethod
    def batch(pos, cpu=None, pbar=False):

        """ Parallelizes the ToI approach for a large number of systems.
        Parameters:
            pos: (Nsys,N,3) or (Nsys,N,2) position array (with host at origin)
            cpu: number of CPU cores to use for multiprocessing (default: max-1)
        """

        if cpu is None: cpu = mp.cpu_count() - 1
        ndim = pos.shape[-1]

        with mp.Pool(cpu) as pool:

            if ndim == 3:

                ca = pool.map(TOI.eval3d,
                    tqdm(pos, total=len(pos), desc="Applying ToI", disable=not pbar), chunksize=100)

                return np.array(ca)

            elif ndim == 2:

                ba = pool.map(TOI.eval2d,
                    tqdm(pos, total=len(pos), desc="Applying ToI", disable=not pbar), chunksize=100)

                return np.array(ba)


class Hemi:

    true_bounds = (1, 0)
    grid_bounds = (1, 0)

    @staticmethod
    #@nb.njit#(nb.types.Tuple((nb.float64, nb.int32, nb.float64[:]))(nb.float64[:,:], nb.int32))
    def eval(pos, num_grid):

        """
        Routine to find hemisphere metric of given system.
        Parameters:
            pos: (N,3) or (N,2) position array (with host at origin)
            num_grid: number of isotropic grid vectors to generate
        Returns:
            contrast: contrast metric max(2 - N/Nlop)
            num: satellite population of most populated hemisphere
            vector: unit vector of most populated hemisphere
        """

        num_sat, num_dim = pos.shape

        # Generate istropically generated unit vectors
        if num_dim == 2:
            grid = calc.isovecs_2d(num_grid)
        elif num_dim == 3:
            grid = calc.isovecs_3d(num_grid)

        # For each vector in grid
        hemisphere = np.empty(num_grid)
        for i, vec in enumerate(grid):

            # For each satellite
            dots = np.empty(num_sat)
            for j in range(num_sat):

                # Calculate dot vector
                dots[j] = np.sum(vec[:] * pos[j,:])

            # Calculate hemisphere population
            hemisphere[i] = np.count_nonzero(dots >= 0)

        # Find max hemisphere population
        max_index = np.argmax(hemisphere)
        num = hemisphere[max_index]
        contrast = 2 - num_sat / num
        vector = grid[max_index, :]

        # Set return datatypes
        num = np.int32(num)
        contrast = np.float64(contrast)
        vector = vector.astype(np.float64)

        return contrast, num, vector

    @staticmethod
    def batch(pos, num_grid, cpu=None):

        """ Parallelizes the hemisphere metric for a large number of systems.
        Parameters:
            pos: (Nsys,N,3) or (Nsys,N,2) position array (with host at origin)
            num_grid: number of isotropic grid vectors to generate
            cpu: number of CPU cores to use for multiprocessing (default: max-1)
        """

        if cpu is None: cpu = mp.cpu_count() - 1

        with mp.Pool(cpu) as pool:
            result = pool.starmap(Hemi.eval, tqdm(zip(pos, repeat(num_grid, len(pos))),
                    total=len(pos), desc="Finding hemisphere"), chunksize=100)

        num = np.array([item[0] for item in result])
        contrast = np.array([item[1] for item in result])
        vector = np.array([item[2] for item in result])

        return contrast, num, vector


class Conical:

    true_bounds = (1, 0)
    grid_bounds = (1, 0)

    @staticmethod
    #@nb.njit#(nb.types.Tuple((nb.float64[:], nb.int32[:], nb.float64[:,:]))(nb.float64[:,:], nb.float64[:], nb.int32))
    def eval(pos, cone_angles, num_grid):

        """ Routine to find cone metric of given system.
        Parameters:
            pos: (N,3) or (N,2) position array (with host at origin)
            cone_angles: float array of half cone opening angles in degrees
            num_grid: number of isotropic grid vectors to generate
        Returns:
            frac: num as fraction of total satellites
            num: array of maximum satellite populations per cone opening angle
            vector: array of corresponding unit vectors
        """

        num_angle = len(cone_angles)
        num_sat, num_dim = pos.shape

        # Generate isotropic grid
        if num_dim == 2:
            grid = calc.isovecs_2d(num_grid)
        elif num_dim == 3:
            grid = calc.isovecs_3d(num_grid)

        # Convert cone angles to radians in [0, pi]
        cone_angles = np.deg2rad(cone_angles)

        # Create empty data arrays
        in_cone = np.empty((num_angle, num_grid))
        num = np.empty(num_angle)
        vector = np.empty((num_angle, num_dim))

        # For each grid vector
        for i, vec in enumerate(grid):

            # Calculate angle to all satellites
            if num_dim == 2:
                sat_angles = calc.angle_multi_2d(pos, vec)
            elif num_dim == 3:
                sat_angles = calc.angle_multi_3d(pos, vec)

            # For each opening angle
            for j, cone_angle in enumerate(cone_angles):

                # Find number of satellites in opening angle
                in_cone[j,i] = np.count_nonzero(sat_angles <= cone_angle)

        # Find maximum population in each opening cone
        for i, cone_angle in enumerate(cone_angles):
            max_index = np.argmax(in_cone[i,:])
            num[i] = in_cone[i,max_index]
            vector[i,:] = grid[max_index,:]

        # Set return datatypes
        num = num.astype(np.int32)
        frac = (num / num_sat).astype(np.float64)
        vector = vector.astype(np.float64)

        return frac, num, vector

    @staticmethod
    def batch(pos, cone_angles, num_grid, cpu=None):

        """ Parallelizes the cone metric for a large number of systems.
        Parameters:
            pos: (Nsys,N,3) or (Nsys,N,2) position array (with host at origin)
            cone_angles: float array of half cone opening angles (degrees)
            num_grid: number of isotropic grid vectors to generate
            cpu: number of CPU cores to use for multiprocessing (default: max-1)
        """

        if cpu is None: cpu = mp.cpu_count() - 1

        with mp.Pool(cpu) as pool:
            result = pool.starmap(Conical.eval, tqdm(zip(pos, cone_angles, repeat(num_grid, len(pos))),
                    total=len(pos), desc="Finding cone"), chunksize=100)

        frac = np.array([item[0] for item in result])
        num = np.array([item[1] for item in result])
        vector = np.array([item[2] for item in result])

        return frac, num, vector


all_metric_names = ["centroid", "mrl", "pairwise", "toi", "hemi", "cone"]
metric_class = {"centroid": Centroid, "mrl": MRL, "pairwise": Pairwise, "toi": TOI, "hemi": Hemi, "cone": Conical}
