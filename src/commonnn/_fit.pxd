cimport numpy as np

from libcpp.vector cimport vector as stdvector
from libcpp.unordered_map cimport unordered_map as stdumap
from libcpp.unordered_set cimport unordered_set as stduset

from commonnn._primitive_types cimport AVALUE, AINDEX, ABOOL
from commonnn._types cimport (
    ClusterParameters, CommonNNParameters, RadiusParameters,
    Labels, ReferenceIndices
)
from commonnn._bundle cimport Bundle
from commonnn._types cimport (
    InputDataExtInterface,
    NeighboursGetterExtInterface,
    NeighboursExtInterface,
    SimilarityCheckerExtInterface,
    QueueExtInterface,
    PriorityQueueExtInterface
)

cdef class FitterExtInterface:
    cdef void _fit_inner(
            self,
            InputDataExtInterface input_data,
            Labels labels,
            ClusterParameters cluster_params) nogil

cdef class FitterExtCommonNNInterface(FitterExtInterface): pass

cdef class FitterExtCommonNNBFS(FitterExtCommonNNInterface):

    cdef public:
        NeighboursGetterExtInterface _neighbours_getter
        NeighboursExtInterface _neighbours
        NeighboursExtInterface _neighbour_neighbours
        SimilarityCheckerExtInterface _similarity_checker
        QueueExtInterface _queue


cdef AINDEX get_root(AINDEX p, AINDEX[::1] parent_indicator) nogil