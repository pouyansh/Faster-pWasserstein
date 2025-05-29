import math
import pyximport
cimport cython
import time

pyximport.install()

from constants import *
from DataStructures.point cimport Point, Edge
from DataStructures.decomposition cimport Decomposition
from Algorithms.hungarian cimport Hungarian
from utils import (
    compute_rounded_euclidean_distance,
    compute_euclidean_distance,
    compute_rounded_l1_distance,
    compute_l1_distance,
)

@cython.profile(True)
cdef round_masses(list[Point] A, list[Point] B, float delta, bint is_matching):
    cdef Point a, b
    cdef double prev_adj_mass
    cdef Edge edge
    cdef double total_removed_mass = 0, total_mass = 0, diff
    cdef int id
    for a in A:
        prev_adj_mass = a.adj_mass
        a.adj_mass = math.ceil(a.mass / delta) * delta
        diff = prev_adj_mass - a.adj_mass
        while a.transported_mass > 0 and diff > 0 and len(list(a.map.keys())) > 0:
            id = list(a.map.keys())[0]
            edge = a.map[id]
            if edge.mass < diff:
                a.map.pop(id)
                a.transported_mass -= edge.mass
                edge.point_b.transported_mass -= edge.mass
                diff -= edge.mass
                total_removed_mass += edge.mass
            else:
                edge.mass -= diff
                a.transported_mass -= diff
                edge.point_b.transported_mass -= diff
                total_removed_mass += diff
                diff = 0
    for b in B:
        b.adj_mass = math.floor(b.mass / delta) * delta
        total_mass += b.adj_mass
    return total_removed_mass, total_mass

@cython.profile(True)
cdef correct_masses(list[Point] A, list[Point] B):
    cdef Point a, b
    cdef double prev_adj_mass
    cdef Edge edge
    cdef double total_removed_mass = 0, total_mass = 0, diff
    for a in A:
        prev_adj_mass = a.adj_mass
        a.adj_mass = a.mass
        diff = prev_adj_mass - a.mass
        if a.transported_mass > EPS and diff > 0:
            while diff > 0 and len(list(a.map.keys())):
                id = list(a.map.keys())[0]
                edge = a.map[id]
                if edge.mass < diff:
                    a.map.pop(id)
                    a.transported_mass -= edge.mass
                    edge.point_b.transported_mass -= edge.mass
                    diff -= edge.mass
                    total_removed_mass += edge.mass
                else:
                    edge.mass -= diff
                    a.transported_mass -= diff
                    edge.point_b.transported_mass -= diff
                    total_removed_mass += diff
                    diff = 0
    for b in B:
        b.adj_mass = b.mass
        total_mass += b.adj_mass
    return total_removed_mass, total_mass

@cython.profile(True)
cdef match_remaining(list[Point] A, list[Point] B):
    cdef int ptr_b = 0
    cdef int ptr_a = 0
    cdef Point a, b
    cdef double remaining_mass_b, mass, remaining_mass_a, total_mass = 0
    while ptr_b < len(B):
        b = B[ptr_b]
        remaining_mass_b = b.mass - b.transported_mass
        if remaining_mass_b > 0:
            while ptr_a < len(A):
                a = A[ptr_a]
                remaining_mass_a = a.mass - a.transported_mass
                if remaining_mass_a > 0:
                    mass = min(remaining_mass_a, remaining_mass_b)
                    remaining_mass_b -= mass
                    a.transported_mass += mass
                    b.transported_mass += mass
                    total_mass += mass
                    if b.id in a.map.keys():
                        a.map[b.id].mass += mass
                        b.map[a.id].mass += mass
                    else:
                        a.map[b.id] = Edge(a, b, mass, 0)
                        b.map[a.id] = a.map[b.id]
                    if remaining_mass_b == 0:
                        break
                ptr_a += 1
        ptr_b += 1

    ptr_b = 0
    while ptr_a < len(A):
        a = A[ptr_a]
        remaining_mass_a = a.mass - a.transported_mass
        if remaining_mass_a > 0:
            while ptr_b < len(B):
                b = B[ptr_b]
                remaining_mass_b = b.mass - b.transported_mass
                if remaining_mass_b > 0:
                    mass = min(remaining_mass_b, remaining_mass_a)
                    remaining_mass_a -= mass
                    b.transported_mass += mass
                    a.transported_mass += mass
                    total_mass += mass
                    if a.id in b.map.keys():
                        b.map[a.id].mass += mass
                        a.map[b.id].mass += mass
                    else:
                        b.map[a.id] = Edge(a, b, mass, 0)
                        a.map[b.id] = b.map[a.id]
                    if remaining_mass_a == 0:
                        break
                ptr_b += 1
        ptr_a += 1

@cython.profile(True)
def compute_p_wasserstein(list[Point] A, list[Point] B, int n, int p, int d, int m, list[list[float]] C, bint is_matching):
    cdef list[Decomposition] decompositions = []
    cdef int i
    cdef Decomposition decomp

    if IS_L1:
        distance_function = compute_rounded_l1_distance
        exact_distance_function = compute_l1_distance
    else:
        distance_function = compute_rounded_euclidean_distance
        exact_distance_function = compute_euclidean_distance

    construction_time = time.time()
    for i in range(m):
        decompositions.append(
            Decomposition(
                A, B, distance_function, i, p, d
            )
        )
    construction_time = time.time() - construction_time

    hungarian_time = time.time()
        
    cdef Hungarian hungarian = Hungarian(A, B, decompositions)

    cdef float delta = 1 / n, delta_cost = 0, fake_cost = 0
    cdef float total_count = 0
    cdef int counter = 0, same_counter = 0
    cdef bint check = True
    cdef float augmenting_paths_num, removed_mass, total_mass = n
    
    while delta >= 1 / (10 * n):
        if not is_matching:
            removed_mass, total_mass = round_masses(A, B, delta, is_matching)
            total_count -= removed_mass
        check = True
        while check:
            counter += 1
            augmenting_paths_num, delta_cost = hungarian.hungarian_search(is_matching)
            fake_cost += delta_cost
            if augmenting_paths_num == 0:
                same_counter += 1
            else:
                same_counter = 0
            total_count += augmenting_paths_num
            if total_count > total_mass - EPS or same_counter >= 5:
                break
        if is_matching:
            break
        delta = delta / 2
    if not is_matching:
        removed_mass, total_mass = correct_masses(A, B)
        match_remaining(A, B)
    
    hungarian_time = time.time() - hungarian_time

    if n <= 20000 and COST_MATRIX:
        for decomp in decompositions:
                decomp.compute_cost_matrix(C)

    cdef Point a, b
    cdef float matching_cost = 0, remaining_mass = 0
    if not is_matching:
        matching_cost = 1 - total_mass
    for a in A:
        if is_matching:
            b = a.match
            if b is None:
                pass
            else:
                matching_cost += (
                    math.pow(
                        exact_distance_function(
                            a.coordinates, b.coordinates
                        ),
                        p,
                    )
                    / n
                )
        else:
            for edge in a.map.values():
                matching_cost += (
                    math.pow(
                        exact_distance_function(
                            a.coordinates, edge.point_b.coordinates
                        ),
                        p,
                    )
                    * edge.mass
                )
            if a.mass - a.transported_mass > EPS:
                remaining_mass += a.mass - a.transported_mass
    matching_cost = math.pow(matching_cost, 1 / float(p))
    fake_cost = math.pow(fake_cost / n, 1 / float(p))

    del A
    del B
    del decompositions

    return matching_cost, total_count, counter, hungarian, fake_cost, construction_time, hungarian_time, C
