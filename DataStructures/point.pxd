cimport numpy as cnp

cdef class Point:
    cdef public cnp.ndarray coordinates
    cdef public int id
    cdef public double y, mass, transported_mass, adj_mass
    cdef public bint visited, DFS_visited
    cdef public Edge matching_edge
    cdef public Point match
    cdef public dict[int, Edge] map


cdef class Edge:
    cdef public Point point_a, point_b
    cdef public float mass, slack