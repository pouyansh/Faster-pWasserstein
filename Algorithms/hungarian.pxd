# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

cimport cython
from DataStructures.decomposition cimport Decomposition
from DataStructures.point cimport Point
from DataStructures.myheap cimport MinHeap
from constants import *

cdef class Hungarian:
    cdef list[Point] A, B
    cdef list[Decomposition] decompositions
    cdef MinHeap heap
    cdef int visited_nodes, visited_edges, dfs_visited_nodes, dfs_visited_edges

    cpdef (float, bint) partial_dfs(self, int n, Point curr_b, float path_capacity, bint is_matching)