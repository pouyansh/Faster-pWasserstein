cdef class MinHeapNode:
    cdef public float key
    cdef public int parent_index, left_index, right_index, index
    cdef public object value
    cdef public object id 


cdef class MinHeap:
    cdef list heap
    cdef int ptr
    cdef bint min_heap
    cdef dict[int, MinHeapNode] entry_finder

    cpdef int get_parent_index(self, MinHeapNode node)

    cpdef int get_left_child_index(self, MinHeapNode node)

    cpdef int get_right_child_index(self, MinHeapNode node)

    cpdef bint has_parent(self, MinHeapNode node)

    cpdef bint has_left_child(self, MinHeapNode node)

    cpdef bint has_right_child(self, MinHeapNode node)

    cpdef MinHeapNode parent(self, MinHeapNode node)

    cpdef MinHeapNode left_child(self, MinHeapNode node)

    cpdef MinHeapNode right_child(self, MinHeapNode node)

    cpdef swap(self, MinHeapNode node1,  MinHeapNode node2)

    cpdef heapify_up(self, MinHeapNode node)

    cpdef extract_min(self)

    cpdef heapify_down(self, MinHeapNode node)

    cpdef to_string(self)

    cpdef check_heap(self)

    cpdef clear(self)