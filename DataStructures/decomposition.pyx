# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import random
import math
from constants import *
cimport cython
cimport numpy as cnp

from DataStructures.point cimport Point, Edge
from DataStructures.myheap cimport MinHeap


cdef class Decomposition:

    def __init__(
        self, list[Point] A, list[Point] B, cost_function, int id, int p, int d
    ) -> None:
        """
        Initialize the Hierarchical Decomposition.
        :param A: points A
        :param B: points B
        :param C: the cost matrix
        """
        self.A = A
        self.B = B
        self.cost_function = cost_function
        self.id = id

        self.gamma = random.random() + 1

        self.U = []
        self.point_list = []

        if not USE_QUADTREE:
            # Creating a random permutation
            self.U = A.copy()
            self.point_list = [[] for _ in range(len(self.A) + len(self.B))]
            random.shuffle(self.U)
            self.create_node_list()
            self.root = Node(A, B, self.point_list, 0, self.gamma, p, 2)

        else:
            left = [-random.random() for _ in range(d)]
            right = [left[i] + 2 for i in range(d)]
            self.root = Node(A, B, self.point_list, 0, 0, p, 2 * math.pow(d, 1/float(p)), left, right)

        self.smallest_slack_edge = self.root.smallest_slack_edge

    @cython.profile(True)
    cpdef create_node_list(self):
        """
        Creating a list consisting of possible center points for each point
        """
        cdef Point a, b, u
        cdef float dist
        cdef list closest_a, closest_b

        for a in self.A:
            u = self.U[0]
            closest_a = [(self.cost_function(a.coordinates, u.coordinates), u.id)]
            for i in range(1, len(self.U)):
                u = self.U[i]
                dist = self.cost_function(a.coordinates, u.coordinates)
                if dist < closest_a[-1][0]:
                    closest_a.append((dist, u.id))
                    if dist == 0:
                        break
            self.point_list[a.id] = closest_a

        for b in self.B:
            u = self.U[0]
            closest_b = [(self.cost_function(b.coordinates, u.coordinates), u.id)]
            for i in range(1, len(self.U)):
                u = self.U[i]
                dist = self.cost_function(b.coordinates, u.coordinates)
                if dist < closest_b[-1][0]:
                    closest_b.append((dist, u.id))
                    if dist == 0:
                        break
            closest_b.append((0, b.id))
            self.point_list[b.id] = closest_b

    @cython.profile(True)
    def compute_cost_matrix(self, cost_matrix: list[list[float]]) -> None:
        self.root.compute_cost_matrix(cost_matrix)

    @cython.profile(True)
    def visit_a(self, Point point_a) -> None:
        self.root.visit_a(point_a)
        self.smallest_slack_edge = self.root.smallest_slack_edge

    @cython.profile(True)
    def visit_b(self, Point point_b) -> None:
        self.root.visit_b(point_b)
        self.smallest_slack_edge = self.root.smallest_slack_edge

    @cython.profile(True)
    def initialize_heap(self):
        self.root.initialize_heaps()
        self.smallest_slack_edge = self.root.smallest_slack_edge

    @cython.profile(True)
    def check_feasibility(self):
        self.root.check_feasibility()

    @cython.profile(True)    
    def min_slack_b(self, Point point_b):
        return self.root.min_slack_b(point_b)


cdef class Node:

    def __init__(
        self,
        A: list[Point],
        B: list[Point],
        point_list: list[list[tuple]],
        level: int,
        diameter: float,
        p: int,
        base: float,
        min_point=[],
        max_point=[]
    ):
        """
        Initialize the Decomposition Node.
        :param A: points A inside node
        :param B: points B inside node
        :point_list: list of possible center nodes
        :level: level of node
        :gamma: random between (1, 2)
        """
        self.point_list = point_list
        self.level = level

        self.base = base
        self.diameter = diameter
        self.distance = math.pow(2 * base, p)
        self.children = {}
        self.children_nodes = {}

        self.min_point = min_point
        self.max_point = max_point
        self.mid_point = [(max_point[i] - min_point[i]) / 2 for i in range(len(max_point))]

        cdef list child
        self.is_leaf = False
        self.A = A
        self.B = B
        if len(A) == 0 or len(B) == 0 or self.distance < math.pow(DELTA, p):  # Leaf node
            self.is_leaf = True
            self.A = A
            self.B = B
        else:
            # Partitioniong points into clusters
            self.decompose(A, B)

            # Creating children nodes
            for key, child in self.children.items():
                child_min_point = []
                child_max_point = []
                if USE_QUADTREE:
                    child_min_point = [self.min_point[i] if key[i] == "0" else self.mid_point[i] for i in range(len(self.min_point))]
                    child_max_point = [self.max_point[i] if key[i] == "1" else self.mid_point[i] for i in range(len(self.max_point))]
                self.children_nodes[key] = Node(
                    child[0],
                    child[1],
                    point_list,
                    level + 1,
                    self.diameter / 2,
                    p,
                    base / 2,
                    child_min_point,
                    child_max_point
                )
        self.heap_a = MinHeap(min_heap=True)
        self.heap_b = MinHeap(min_heap=False)
        self.smallest_children_slack = MinHeap(min_heap=True)

        
        if len(A) > 0:
            self.min_a = A[0]
        else:
            self.min_a = None
        if len(B) > 0:
            self.max_b = B[0]
        else:
            self.max_b = None
        self.smallest_slack_edge = Edge(None, None, 0, -1)

    @cython.profile(True)
    cpdef decompose(self, list[Point] A, list[Point] B):
        """
        Decompose the points A and B into clusters
        """
        cdef Point a, b
        for a in A:
            center = self.find_center(a)
            if center not in self.children.keys():
                self.children[center] = [[], []]
            self.children[center][0].append(a)

        for b in B:
            center = self.find_center(b)
            if center not in self.children.keys():
                self.children[center] = [[], []]
            self.children[center][1].append(b)

    @cython.profile(True)
    cpdef initialize_heaps(self):
        """
        initialize the heaps
        """
        cdef Point a, a_min
        cdef Node child
        for c in self.children_nodes.keys():
            child = self.children_nodes[c]
            child.initialize_heaps()
        self.heap_a.clear()
        self.heap_b.clear()
        self.smallest_children_slack.clear()
        self.smallest_slack_edge = Edge(None, None, 0, -1)
        self.max_b = None
        if self.is_leaf:
            for a in self.A:
                self.heap_a.insert(a.y, a, a.id)
            self.min_a = self.heap_a.peek()
        else:
            for c in self.children_nodes.keys():
                child = self.children_nodes[c]
                if child.min_a is not None:
                    a_min = child.min_a
                    self.heap_a.insert(a_min.y, child, c)
            child = self.heap_a.peek()
            self.min_a = None if child is None else child.min_a

    @cython.profile(True)
    cpdef Edge compute_smallest_slack_edge(self):
        cdef Edge result = Edge(None, None, 0, -1)

        cdef double slack

        if self.min_a is not None and self.max_b is not None:
            slack = self.distance - self.max_b.y + self.min_a.y + DELTA
            result = Edge(self.min_a, self.max_b, 0, slack)
        if self.is_leaf:
            return result
        cdef Node min_child = self.smallest_children_slack.peek()
        cdef Edge edge
        if min_child is not None:
            edge = min_child.smallest_slack_edge
            if result.slack == -1 or edge.slack < result.slack and edge.point_a is not None:
                result = edge
        return result

    @cython.profile(True)
    cpdef visit_a(self, Point point_a):
        cdef Node child, min_child
        cdef Point a_min
        cdef Edge min_edge
        if not self.is_leaf:
            center = self.find_center(point_a)
            child = self.children_nodes[center]
            child.visit_a(point_a)

            a_min = child.min_a
            if a_min is not None:
                self.heap_a.insert(a_min.y, child, center)
            else:
                self.heap_a.remove(center)

            min_edge = child.smallest_slack_edge
            if min_edge.slack != -1:
                self.smallest_children_slack.insert(
                    min_edge.slack, child, center
                )
            else:
                self.smallest_children_slack.remove(center)

            min_child = self.heap_a.peek()
            self.min_a = None if min_child is None else min_child.min_a

            self.smallest_slack_edge = self.compute_smallest_slack_edge()
        else:
            self.heap_a.remove(point_a.id)
            self.min_a = self.heap_a.peek()
            self.smallest_slack_edge = self.compute_smallest_slack_edge()

    @cython.profile(True)
    cpdef visit_b(self, Point point_b):
        cdef Node child, max_child
        cdef Point b_max, b
        cdef Edge min_edge
        if not self.is_leaf:
            center = self.find_center(point_b)
            child = self.children_nodes[center]
            child.visit_b(point_b)
            b_max = child.max_b
            self.heap_b.insert(b_max.y, child, center)

            min_edge = child.smallest_slack_edge
            if min_edge.slack != -1:
                self.smallest_children_slack.insert(
                    min_edge.slack, child, center
                )
            else:
                self.smallest_children_slack.remove(center)

            max_child = self.heap_b.peek()
            self.max_b = None if max_child is None else max_child.max_b
            self.smallest_slack_edge = self.compute_smallest_slack_edge()
        else:
            self.heap_b.insert(point_b.y, point_b, point_b.id)
            self.max_b = self.heap_b.peek()
            self.smallest_slack_edge = self.compute_smallest_slack_edge()

    @cython.profile(True)      
    cpdef Point min_slack_b(self, Point point_b):
        if self.min_a is not None and self.distance - point_b.y + self.min_a.y + DELTA < EPS:
            return self.min_a
        if self.is_leaf: return None
        center = self.find_center(point_b)
        cdef Node child = self.children_nodes[center]
        return child.min_slack_b(point_b)

    @cython.profile(True)
    cpdef object find_center(self, Point point):
        cdef double distance = self.diameter / 2
        cdef str center = ""
        cdef int ptr = 0
        cdef cnp.ndarray coords = point.coordinates
        # finding the closest center to point
        if USE_QUADTREE:
            for i in range(len(coords)):
                center += "0" if coords[i] < self.mid_point[i] else "1"
            return center 
        else:
            while self.point_list[point.id][ptr][0] > distance:
                ptr += 1
            return self.point_list[point.id][ptr][1]

    def check_feasibility(self):
        cdef Point a, b
        for a in self.A:
            for b in self.B:
                if b.y - a.y > self.distance + DELTA + EPS:
                    print(
                        "Infeasibility (",
                        a.id,
                        b.id,
                        ")",
                        self.distance - b.y + a.y + DELTA,
                        a.visited,
                        b.visited,
                        self.smallest_slack_edge.slack,
                        self.heap_b.to_string(),
                        self.heap_a.to_string(),
                        self.smallest_children_slack.to_string()
                    )
        cdef Node child 
        for c in self.children_nodes.keys():
            child = self.children_nodes[c]
            child.check_feasibility()

    def compute_cost_matrix(self, cost_matrix: list[list[float]]) -> None:
        cdef double distance = self.distance
        for a1 in self.A:
            for b in self.B:
                if (
                    cost_matrix[a1.id][b.id - len(cost_matrix)] == -1
                    or cost_matrix[a1.id][b.id - len(cost_matrix)] > distance
                ):
                    cost_matrix[a1.id][b.id - len(cost_matrix)] = distance
                    cost_matrix[b.id - len(cost_matrix)][a1.id] = distance
        for c in self.children_nodes.keys():
            self.children_nodes[c].compute_cost_matrix(cost_matrix)