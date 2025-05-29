# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

cimport cython
from DataStructures.decomposition cimport Decomposition
from DataStructures.point cimport Point, Edge
from DataStructures.myheap cimport MinHeap
from constants import *

cdef class Hungarian:

    def __init__(self, list[Point] A, list[Point] B, list[Decomposition] decompositions) -> None:
        self.A = A
        self.B = B
        self.decompositions = decompositions
        self.heap = MinHeap(True)
        self.visited_nodes, self.visited_edges, self.dfs_visited_nodes, self.dfs_visited_edges = 0, 0, 0, 0

    @cython.profile(True)
    def hungarian_search(self, is_matching=True):
        """
        Implements the Hungarian search to optimize the assignment between points in sets A and B.

        Args:
            A (list of Point): List of points in set A.
            B (list of Point): List of points in set B.

        Returns:
            number of augmenting paths
        """

        cdef int n = len(self.A) + len(self.B)
        cdef Decomposition d
        cdef Point a, b

        for d in self.decompositions:
            d.initialize_heap()
        for a in self.A:
            a.visited = False
            a.DFS_visited = False
        for b in self.B:
            b.visited = False
            b.DFS_visited = False
        for b in self.B:
            if b.transported_mass + EPS < b.adj_mass:
                b.visited = True
                for d in self.decompositions:
                    d.visit_b(b)
                self.visited_nodes += 1

        self.heap.clear()

        cdef int index, match_id
        cdef float distance
        cdef list queue_b
        cdef Edge edge, matching_edge
        cdef Point match_b

        for index, d in enumerate(self.decompositions):
            edge = d.smallest_slack_edge
            self.heap.insert(edge.slack, d, index)

        distance = 0

        queue_b = []
        # Process pairs in increasing cost
        while self.heap:
            d = self.heap.peek()
            if d is None:
                break
            edge = d.smallest_slack_edge
            if edge.slack == -1:
                for a in self.A:
                    if a.visited:
                        a.y += distance
                for b in self.B:
                    if b.visited:
                        b.y += distance
                break
            assert edge.point_b.visited
            assert not edge.point_a.visited

            if edge.slack > distance + EPS and len(queue_b) > 0:
                distance += DELTA
                for a, b in queue_b:
                    b.visited = True
                    b.y -= distance
                    if is_matching:
                        a.matching_edge.slack = 0
                    else:
                        a.map[b.id].slack = 0
                    for d in self.decompositions:
                        d.visit_b(b)
                    self.visited_nodes += 1
                queue_b = []
                for d in self.decompositions:
                    self.heap.update(d.id, d.smallest_slack_edge.slack)
                continue

            distance = edge.slack

            edge.point_a.visited = True
            edge.point_a.y -= distance

            for d in self.decompositions:
                d.visit_a(edge.point_a)
            self.visited_nodes += 1
            self.visited_edges += 1

            if edge.point_a.transported_mass + EPS < edge.point_a.adj_mass:
                for a in self.A:
                    if a.visited:
                        a.y += distance
                for b in self.B:
                    if b.visited:
                        b.y += distance
                break

            if is_matching:
                if edge.point_a.match is not None:
                    matching_edge = edge.point_a.matching_edge
                    match_b = matching_edge.point_b
                    if matching_edge.slack == 0:
                        match_b.y -= distance
                        match_b.visited = True
                        for d in self.decompositions:
                            d.visit_b(match_b)
                        self.visited_nodes += 1
                    else:
                        queue_b.append((edge.point_a, match_b))
                    self.visited_edges += 1
            else:
                for match_id in edge.point_a.map.keys():
                    matching_edge = edge.point_a.map[match_id]
                    match_b = matching_edge.point_b
                    if match_b.visited:
                        matching_edge.slack = DELTA
                    else:
                        if matching_edge.slack == 0:
                            match_b.y -= distance
                            match_b.visited = True
                            for d in self.decompositions:
                                d.visit_b(match_b)
                            self.visited_nodes += 1
                        else:
                            queue_b.append((edge.point_a, match_b))
                    self.visited_edges += 1

            for d in self.decompositions:
                self.heap.update(d.id, d.smallest_slack_edge.slack)

        cdef int i
        cdef float counter = 0
        cdef bint check
        cdef float delta_cost = 0
        for d in self.decompositions:
            d.initialize_heap()
        for i in range(len(self.B)):
            b = self.B[i]
            if b.transported_mass + EPS < b.adj_mass:
                b.DFS_visited = True
                capacity = b.adj_mass - b.transported_mass
                self.dfs_visited_nodes += 1
            
                transported, check = self.partial_dfs(n, b, capacity, is_matching)
                if check:
                    b.transported_mass += transported
                counter += transported
                delta_cost += transported * b.y
                if not is_matching and transported > 0:
                    i -= 1
        return counter, delta_cost


    @cython.profile(True)
    cpdef (float, bint) partial_dfs(self, int n, Point curr_b, float path_capacity, bint is_matching):
        cdef int index = 0, i = 0, match_id = 0
        cdef Point next_a, temp_a, temp_b, temp_a_next, match_b
        cdef list temp_queue
        cdef Decomposition d
        cdef float capacity = 0, count = 0, added_count = 0
        cdef Edge edge, reducing_edge
        cdef bint check = False

        while index < len(self.decompositions):
            next_a = self.decompositions[index].min_slack_b(curr_b)
            self.dfs_visited_edges += 1
            if next_a is None:
                if index == len(self.decompositions) - 1:
                    return count, False
                else:
                    index += 1
                continue

            for d in self.decompositions:
                d.visit_a(next_a)
            self.dfs_visited_nodes += 1

            next_a.DFS_visited = True

            if next_a.transported_mass + EPS < next_a.adj_mass:
                # found an augmenting path
                if is_matching:
                    capacity = next_a.adj_mass
                else:
                    capacity = min(
                        path_capacity,
                        next_a.adj_mass - next_a.transported_mass
                    )
                if not is_matching and curr_b.id in next_a.map.keys():
                    next_a.map[curr_b.id].mass += capacity
                else:
                    if is_matching:
                        next_a.match = curr_b
                        curr_b.match = next_a
                        curr_b.y -= DELTA
                        next_a.matching_edge = Edge(next_a, curr_b, capacity, 0)
                        curr_b.matching_edge = next_a.matching_edge
                    else:
                        next_a.map[curr_b.id] = Edge(next_a, curr_b, capacity, DELTA)
                        curr_b.map[next_a.id] = next_a.map[curr_b.id]

                next_a.transported_mass += capacity
                count += capacity
                return count, True

            if not is_matching:
                for match_id in next_a.map.keys():
                    edge = next_a.map[match_id]
                    match_b = edge.point_b
                    if not match_b.DFS_visited and edge.slack == 0:
                        match_b.DFS_visited = True
                        added_count, check = self.partial_dfs(n, match_b, min(path_capacity, edge.mass), is_matching)
                        count += added_count
                        self.dfs_visited_nodes += 1
                        self.dfs_visited_edges += 1
                        if check:
                            reducing_edge = next_a.map[match_b.id]
                            reducing_edge.mass -= count
                            if reducing_edge.mass < EPS:
                                next_a.map.pop(match_b.id)
                                match_b.map.pop(next_a.id)

                            if curr_b.id in next_a.map.keys():
                                next_a.map[curr_b.id].mass += count
                            else:
                                next_a.map[curr_b.id] = Edge(next_a, curr_b, capacity, DELTA)
                                curr_b.map[next_a.id] = next_a.map[curr_b.id]
                            return count, check
            else:
                edge = next_a.matching_edge
                match_b = edge.point_b
                if not match_b.DFS_visited and edge.slack == 0:
                    match_b.DFS_visited = True
                    added_count, check = self.partial_dfs(n, match_b, path_capacity, is_matching)
                    count += added_count
                    self.dfs_visited_nodes += 1
                    self.dfs_visited_edges += 1
                    if check > 0:
                        next_a.match = curr_b
                        curr_b.match = next_a
                        curr_b.y -= DELTA
                        next_a.matching_edge = Edge(next_a, curr_b, capacity, 0)
                        curr_b.matching_edge = next_a.matching_edge
                        return count, check
                    
        return count, False

    def get_visited_nodes(self): return self.visited_nodes
    def get_visited_edges(self): return self.visited_edges
    def get_dfs_visited_nodes(self): return self.dfs_visited_nodes
    def get_dfs_visited_edges(self): return self.dfs_visited_edges
