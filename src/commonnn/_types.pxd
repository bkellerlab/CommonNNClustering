from commonnn._primitive_types cimport AINDEX, AVALUE, ABOOL
from commonnn._primitive_types cimport _allocate_and_fill_aindex_array, _allocate_and_fill_avalue_array

from libc.stdlib cimport malloc, free
from libcpp.unordered_set cimport unordered_set as stduset


cdef class ClusterParameters:
    cdef:
        AVALUE *fparams
        AINDEX *iparams

cdef class CommonNNParameters(ClusterParameters): pass


cdef class Labels:
    cdef public:
        dict meta

    cdef:
        AINDEX[::1] _labels
        ABOOL[::1] _consider
        stduset[AINDEX] _consider_set
