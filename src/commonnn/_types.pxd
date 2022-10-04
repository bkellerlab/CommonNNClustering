from commonnn._primitive_types cimport AINDEX, AVALUE, ABOOL
from commonnn._primitive_types cimport _allocate_and_fill_aindex_array, _allocate_and_fill_avalue_array

from libc.stdlib cimport malloc, free
from libcpp.unordered_set cimport unordered_set as stduset
from libcpp.vector cimport vector as stdvector


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


cdef class ReferenceIndices:
    cdef:
        AINDEX[::1] _root
        AINDEX[::1] _parent


cdef class InputDataExtInterface:
    cdef public:
        AINDEX n_points
        AINDEX n_dim
        dict meta

    cdef AVALUE _get_component(
            self, const AINDEX point, const AINDEX dimension) nogil
    cdef AINDEX _get_n_neighbours(self, const AINDEX point) nogil
    cdef AINDEX _get_neighbour(self, const AINDEX point, const AINDEX member) nogil
    cdef AVALUE _get_distance(self, const AINDEX point_a, const AINDEX point_b) nogil
    cdef void _compute_distances(self, InputDataExtInterface input_data) nogil
    cdef void _compute_neighbourhoods(
            self,
            InputDataExtInterface input_data, AVALUE r,
            ABOOL is_sorted, ABOOL is_selfcounting) nogil

cdef class InputDataExtComponentsMemoryview(InputDataExtInterface):
    cdef AVALUE[:, ::1] _data

cdef class InputDataExtDistancesLinearMemoryview(InputDataExtInterface):
    cdef AVALUE[::1] _data

cdef class InputDataExtNeighbourhoodsMemoryview(InputDataExtInterface):
    cdef:
        AINDEX[:, ::1] _data
        AINDEX[::1] _n_neighbours

cdef class InputDataExtNeighbourhoodsVector(InputDataExtInterface):
    cdef:
        stdvector[stdvector[AINDEX]] _data
        stdvector[AINDEX] _n_neighbours
