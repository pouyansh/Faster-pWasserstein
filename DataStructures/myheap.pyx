# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

cimport cython

cdef class MinHeapNode:

    def __init__(self, float key, value, id):
        self.key = key
        self.value = value
        self.index = -1
        self.id = id
        self.parent_index = -1
        self.left_index = -1
        self.right_index = -1


cdef class MinHeap:

    def __init__(self, min_heap=True):
        self.heap = []  # List to store heap elements
        self.min_heap = min_heap
        self.entry_finder = {}  # Dictionary to map tasks to their entries
        self.ptr = 0

    @cython.profile(True)
    cpdef int get_parent_index(self, MinHeapNode node):
        return (node.index - 1) // 2

    @cython.profile(True)
    cpdef int get_left_child_index(self, MinHeapNode node):
        return node.index * 2 + 1

    @cython.profile(True)
    cpdef int get_right_child_index(self, MinHeapNode node):
        return node.index * 2 + 2

    cpdef bint has_parent(self, MinHeapNode node):
        return node.parent_index >= 0

    cpdef bint has_left_child(self, MinHeapNode node):
        return 0 <= node.left_index < self.ptr

    cpdef bint has_right_child(self, MinHeapNode node):
        return 0 <= node.right_index < self.ptr

    cpdef MinHeapNode parent(self, MinHeapNode node):
        return self.heap[node.parent_index]

    cpdef MinHeapNode left_child(self, MinHeapNode node):
        return self.heap[node.left_index]

    cpdef MinHeapNode right_child(self, MinHeapNode node):
        return self.heap[node.right_index]

    @cython.profile(True)
    cpdef swap(self, MinHeapNode node1,  MinHeapNode node2):
        cdef int index1= node1.index, index2 = node2.index
        self.heap[index1], self.heap[index2] = (
            self.heap[index2],
            self.heap[index1],
        )
        node1.index, node2.index = index2, index1
        node1.parent_index, node2.parent_index = node2.parent_index, node1.parent_index
        node1.left_index, node2.left_index = node2.left_index, node1.left_index
        node1.right_index, node2.right_index = node2.right_index, node1.right_index

    @cython.profile(True)
    def insert(self, float key, value, id):
        if id in self.entry_finder.keys():
            return self.update(id, key if self.min_heap else -key)
        node = MinHeapNode(key if self.min_heap else -key, value, id)
        self.entry_finder[id] = node
        if len(self.heap) == self.ptr:
            self.heap.append(node)
        else:
            self.heap[self.ptr] = node
        node.index = self.ptr
        self.ptr += 1
        node.parent_index = self.get_parent_index(node)
        node.left_index = self.get_left_child_index(node)
        node.right_index = self.get_right_child_index(node)
        self.heapify_up(node)

    @cython.profile(True)
    cpdef heapify_up(self, MinHeapNode node):
        while (
            node.parent_index >= 0 and self.parent(node).key > node.key
        ):
            self.swap(node, self.parent(node))

    @cython.profile(True)
    cpdef extract_min(self):
        if self.ptr == 0:
            raise IndexError("Heap is empty")

        cdef MinHeapNode min_node = self.heap[0]
        cdef MinHeapNode last_node = self.heap[self.ptr - 1]
        self.swap(min_node, last_node)
        self.ptr -= 1
        self.entry_finder.pop(min_node.id)
        if self.ptr > 0:
            self.heapify_down(self.heap[0])
        return min_node

    @cython.profile(True)
    def remove(self, id):
        cdef MinHeapNode node, last_node
        if id in self.entry_finder.keys():
            node = self.entry_finder[id]
            last_node = self.heap[self.ptr - 1]
            self.swap(node, last_node)
            self.ptr -= 1
            self.entry_finder.pop(id)
            if node != last_node:
                self.heapify_down(last_node)
                self.heapify_up(last_node)

    @cython.profile(True)
    cpdef heapify_down(self, MinHeapNode node):
        cdef MinHeapNode smaller_child
        while self.has_left_child(node):
            smaller_child = self.left_child(node)
            if (
                self.has_right_child(node)
                and self.right_child(node).key < self.left_child(node).key
            ):
                smaller_child = self.right_child(node)

            if node.key < smaller_child.key:
                break
            else:
                self.swap(node, smaller_child)

    @cython.profile(True)
    def update(self, id, float new_key):
        cdef MinHeapNode node = self.entry_finder[id]
        node.key = new_key
        self.heapify_up(node)
        self.heapify_down(node)

    @cython.profile(True)
    def peek(self):
        if self.ptr == 0:
            return None
        return self.heap[0].value

    @cython.profile(True)
    cpdef to_string(self):
        result = ""
        cdef int i
        cdef MinHeapNode element
        for i in range(self.ptr):
            element = self.heap[i]
            result += f"({round(element.key, 6)}, {element.id})"
        return result

    def __len__(self):
        return self.ptr

    @cython.profile(True)
    cpdef check_heap(self):
        cdef int index = 1
        while index < len(self.heap):
            node = self.heap[index]
            assert node.parent_index == self.get_parent_index(node)
            assert node.left_index == self.get_left_child_index(
                node
            ), f"{node.index}, {node.left_index}, {self.get_left_child_index(node)}"
            assert node.right_index == self.get_right_child_index(node)
            assert node.key >= self.parent(node).key, (
                str(self.min_heap) + " " + self.to_string()
            )
            index += 1

    @cython.profile(True)
    cpdef clear(self):
        self.ptr = 0
        self.entry_finder = {}
        self.heap = []
