from commonnn._primitive_types cimport AINDEX, AVALUE, ABOOL
from commonnn._types cimport Labels  #, ReferenceIndices

from libcpp.deque cimport deque as stddeque
from libcpp.vector cimport vector as stdvector

cdef class Bundle:
    cdef public:
        object _input_data
        object _graph
        object _labels
        object _reference_indices
        dict _children
        str alias
        object _parent
        dict meta
        object _summary
        AINDEX _hierarchy_level
        AVALUE _lambda

    cdef object __weakref__
    cpdef void isolate(
        self,
        bint purge=*,
        bint isolate_input_data=*)
