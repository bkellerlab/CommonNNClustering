from commonnn._primitive_types cimport AINDEX, AVALUE, ABOOL
from commonnn._primitive_types cimport _allocate_and_fill_aindex_array, _allocate_and_fill_avalue_array

from libc.stdlib cimport malloc, free
from libcpp.unordered_set cimport unordered_set as stduset
from libcpp.set cimport set as stdset
from libcpp.vector cimport vector as stdvector
from libcpp.queue cimport queue as stdqueue, priority_queue as stdprioqueue


cdef extern from "<algorithm>" namespace "std":
    Iter find[Iter, T](Iter first, Iter last, const T& value) nogil


cdef class ClusterParameters:
    cdef:
        AVALUE *fparams
        AINDEX *iparams

cdef class CommonNNParameters(ClusterParameters): pass


cdef class Labels:
    cdef public:
        dict _meta

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
        AINDEX _n_points
        AINDEX _n_dim
        dict _meta

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


cdef class NeighboursExtInterface:
    cdef public:
        AINDEX _n_points

    cdef void _assign(self, const AINDEX member) nogil
    cdef void _reset(self) nogil
    cdef bint _enough(self, const AINDEX member_cutoff) nogil
    cdef AINDEX _get_member(self, const AINDEX index) nogil
    cdef bint _contains(self, const AINDEX member) nogil


cdef class NeighboursGetterExtInterface:
    cdef public:
        bint is_selfcounting
        bint is_sorted

    cdef void _get(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil

    cdef void _get_other(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil


cdef class DistanceGetterExtInterface:
    cdef AVALUE _get_single(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _get_single_other(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil


cdef class MetricExtInterface:
    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil


cdef class SimilarityCheckerExtInterface:

    cdef bint _check(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil

    cdef AINDEX _get(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil

cdef class QueueExtInterface:

    cdef void _push(self, const AINDEX value) nogil
    cdef AINDEX _pop(self) nogil
    cdef bint _is_empty(self) nogil


cdef class PriorityQueueExtInterface:

    cdef void _reset(self) nogil
    cdef void _push(self, const AINDEX a, const AINDEX b, const AVALUE weight) nogil
    cdef (AINDEX, AINDEX, AVALUE) _pop(self) nogil
    cdef bint _is_empty(self) nogil
    cdef AINDEX _size(self) nogil


cdef class InputDataExtComponentsMemoryview(InputDataExtInterface):
    cdef AVALUE[:, ::1] _data

cdef class InputDataExtDistancesMemoryview(InputDataExtInterface):
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


cdef class NeighboursExtVector(NeighboursExtInterface):
    cdef:
        AINDEX _initial_size
        stdvector[AINDEX] _neighbours

cdef class NeighboursExtSet(NeighboursExtInterface):
    cdef stdset[AINDEX] _neighbours

cdef class NeighboursExtUnorderedSet(NeighboursExtInterface):
    cdef stduset[AINDEX] _neighbours

cdef class NeighboursExtVectorUnorderedSet(NeighboursExtInterface):
    cdef:
        AINDEX _initial_size
        stdvector[AINDEX] _neighbours
        stduset[AINDEX] _neighbours_view


cdef class SimilarityCheckerExtContains(SimilarityCheckerExtInterface): pass
cdef class SimilarityCheckerExtSwitchContains(SimilarityCheckerExtInterface): pass
cdef class SimilarityCheckerExtScreensorted(SimilarityCheckerExtInterface): pass

cdef class QueueExtLIFOVector(QueueExtInterface):
    cdef stdvector[AINDEX] _queue

cdef class QueueExtFIFOQueue(QueueExtInterface):
    cdef stdqueue[AINDEX] _queue
