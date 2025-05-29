from DataStructures.point cimport Point, Edge
from DataStructures.myheap cimport MinHeap


cdef class Decomposition:
    cdef public list[Point] A, B, U
    cdef public float gamma
    cdef public int id
    cdef public Node root
    cdef public Edge smallest_slack_edge
    cdef public object cost_function, point_list

    cpdef create_node_list(self)


cdef class Node:
    cdef public list[Point] A, B
    cdef public int level
    cdef public float gamma, diameter, distance
    cdef public MinHeap heap_a, heap_b, smallest_children_slack
    cdef public Point min_a, max_b
    cdef public bint is_leaf
    cdef public Edge smallest_slack_edge
    cdef dict[int, list] children
    cdef dict[int, Node] children_nodes
    cdef public object point_list
    cdef float base
    cdef list[float] min_point, max_point, mid_point

    cpdef decompose(self, list[Point] A, list[Point] B)

    cpdef initialize_heaps(self)

    cpdef Edge compute_smallest_slack_edge(self)

    cpdef visit_a(self, Point point_a)

    cpdef visit_b(self, Point point_b)
            
    cpdef Point min_slack_b(self, Point point_b)

    cpdef object find_center(self, Point point)