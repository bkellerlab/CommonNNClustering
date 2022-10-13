from commonnn._primitive_types cimport AINDEX, AVALUE, ABOOL, UINT_MAX
from commonnn._primitive_types cimport maxint
from commonnn._types cimport Labels, ReferenceIndices

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
        dict _meta
        object _summary
        AINDEX _hierarchy_level
        AVALUE _lambda

    cdef object __weakref__

    cpdef void isolate(self, bint purge=*, bint isolate_input_data=*)
    cpdef void reel(self, AINDEX depth=*)


cpdef void isolate(Bundle bundle, bint purge=*, bint isolate_input_data=*)
cpdef void reel(Bundle bundle, AINDEX depth=*)
cdef void _reel(Bundle parent, AINDEX depth)

cpdef void check_children(
    Bundle bundle, AINDEX member_cutoff, bint needs_folding=*)
