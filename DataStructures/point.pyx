from constants import *
import numpy as np

cdef class Point:

    def __init__(self, list coordinates, int id, float mass):
        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.id = id
        self.y = 0
        self.mass = mass
        self.visited = False
        self.DFS_visited = False
        self.map = {}
        self.transported_mass = 0
        self.adj_mass = mass
        self.match = None
        self.matching_edge = None


cdef class Edge:
    
    def __init__(self, Point point_a, Point point_b, float mass, float slack):
        self.point_a = point_a
        self.point_b = point_b
        self.mass = mass
        self.slack = slack
