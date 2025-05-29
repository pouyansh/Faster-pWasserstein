# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import cython
import math
from constants import *
import numpy as np
cimport numpy as np
from DataStructures.point cimport Point
from libc.math cimport sqrt, ceil


@cython.profile(True)
def compute_euclidean_distance(np.ndarray[np.float64_t, ndim=1] coords1,
                               np.ndarray[np.float64_t, ndim=1] coords2):
    """
    Compute the Euclidean distance between two lists of coordinates.

    Parameters
    ----------
    coords1 : ndarray
        First list of coordinates (1D array).
    coords2 : ndarray
        Second list of coordinates (1D array).

    Returns
    -------
    float
        The Euclidean distance between the two lists of coordinates.
    """
    cdef int i
    cdef double diff, distance = 0.0

    # Check that the input lists have the same length
    if coords1.shape[0] != coords2.shape[0]:
        raise ValueError("Both coordinate lists must have the same length")

    # Calculate the squared differences
    for i in range(coords1.shape[0]):
        diff = coords1[i] - coords2[i]
        distance += diff * diff

    # Return the square root of the sum of squared differences
    return sqrt(distance) / sqrt(coords1.shape[0])


def compute_euclidean_distances(list[Point] A, list[Point] B, int p):
    cdef int n1 = len(A), n2 = len(B)
    cdef np.ndarray[np.float64_t, ndim=2] C = np.zeros((n1, n2), dtype=np.float64)

    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = math.pow(compute_euclidean_distance(A[i].coordinates, B[j].coordinates), p)

    return C 


@cython.profile(True)
def compute_rounded_euclidean_distance(np.ndarray[np.float64_t, ndim=1] coords1, np.ndarray[np.float64_t, ndim=1] coords2):
    """
    Compute the Euclidean distance between two lists of coordinates.

    Parameters
    ----------
    coords1 : ndarray
        First list of coordinates (1D array).
    coords2 : ndarray
        Second list of coordinates (1D array).

    Returns
    -------
    float
        The Euclidean distance between the two lists of coordinates.
    """
    cdef int i
    cdef double diff, distance = 0.0
    cdef double delta = 1e-8

    # Check that the input lists have the same length
    if coords1.shape[0] != coords2.shape[0]:
        raise ValueError("Both coordinate lists must have the same length")

    # Calculate the squared differences
    for i in range(coords1.shape[0]):
        diff = coords1[i] - coords2[i]
        distance += diff * diff

    new_distance = ceil((sqrt(distance) / sqrt(coords1.shape[0])) / delta) * delta

    # Return the square root of the sum of squared differences
    return new_distance



@cython.profile(True)
def compute_l1_distance(np.ndarray[np.float64_t, ndim=1] coords1,
                        np.ndarray[np.float64_t, ndim=1] coords2):
    """
    Compute the L1 distance between two lists of coordinates.

    Parameters
    ----------
    coords1 : ndarray
        First list of coordinates (1D array).
    coords2 : ndarray
        Second list of coordinates (1D array).

    Returns
    -------
    float
        The L1 distance between the two lists of coordinates.
    """
    cdef int i
    cdef double diff, distance = 0.0

    # Check that the input lists have the same length
    if coords1.shape[0] != coords2.shape[0]:
        raise ValueError("Both coordinate lists must have the same length")

    # Calculate the squared differences
    for i in range(coords1.shape[0]):
        diff = coords1[i] - coords2[i]
        distance += abs(diff)

    # Return the square root of the sum of squared differences
    return distance / coords1.shape[0]


def compute_l1_distances(list[Point] A, list[Point] B, int p):
    cdef int n1 = len(A), n2 = len(B)
    cdef np.ndarray[np.float64_t, ndim=2] C = np.zeros((n1, n2), dtype=np.float64)

    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = math.pow(compute_l1_distance(A[i].coordinates, B[j].coordinates), p)

    return C 



@cython.profile(True)
def compute_rounded_l1_distance(np.ndarray[np.float64_t, ndim=1] coords1, np.ndarray[np.float64_t, ndim=1] coords2):
    """
    Compute the L1 distance between two lists of coordinates.

    Parameters
    ----------
    coords1 : ndarray
        First list of coordinates (1D array).
    coords2 : ndarray
        Second list of coordinates (1D array).

    Returns
    -------
    float
        The L1 distance between the two lists of coordinates.
    """
    cdef int i
    cdef double diff, distance = 0.0
    cdef double delta = 1e-5

    # Check that the input lists have the same length
    if coords1.shape[0] != coords2.shape[0]:
        raise ValueError("Both coordinate lists must have the same length")

    # Calculate the squared differences
    for i in range(coords1.shape[0]):
        diff = coords1[i] - coords2[i]
        distance += abs(diff)

    # Return the square root of the sum of squared differences
    return ceil(distance / coords1.shape[0] / delta) * delta


def compute_approximation_factors(real_matrix, approx_matrix, p, d, date, m):
    """
    Compute the min, max, average, median, and standard deviation of the approximation factors
    between the real cost matrix and the approximate cost matrix.

    :param real_matrix: Numpy array, the real cost matrix
    :param approx_matrix: Numpy array, the approximate cost matrix
    :return: Dictionary with min, max, average, median, and std of the approximation factors
    """
    # Ensure the input matrices are numpy arrays
    real_matrix = np.array(real_matrix, dtype=float)
    approx_matrix = np.array(approx_matrix, dtype=float)

    # Avoid division by zero by masking entries where real_matrix is 0
    nonzero_mask = real_matrix != 0
    real_values = real_matrix[nonzero_mask]
    approx_values = approx_matrix[nonzero_mask]

    # Calculate the approximation factors
    approximation_factors = approx_values / real_values

    # Compute statistics
    min_val = np.min(approximation_factors)
    max_val = np.max(approximation_factors)
    mean_val = np.mean(approximation_factors)
    median_val = np.median(approximation_factors)
    std_val = np.std(approximation_factors)

    with open("results/distance_date" + date + ".txt", "a") as f:
        f.write(
            "\t".join(
                [
                    "n:",
                    str(len(real_matrix)),
                    "p:",
                    str(p),
                    "d:",
                    str(d),
                    "m:",
                    str(m),
                    "min:",
                    f"{min_val:.3f}",
                    "max:",
                    f"{max_val:.3f}",
                    "mean:",
                    f"{mean_val:.3f}",
                    "median:",
                    f"{median_val:.3f}",
                    "std:",
                    f"{std_val:.3f}",
                    "\n",
                ]
            )
        )

    # Return results as a dictionary
    results = {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
    }
    return results
