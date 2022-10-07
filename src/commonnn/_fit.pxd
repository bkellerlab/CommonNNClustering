cimport numpy as np

from libcpp.vector cimport vector as stdvector
from libcpp.unordered_map cimport unordered_map as stdumap
from libcpp.unordered_set cimport unordered_set as stduset

from commonnn._primitive_types cimport AVALUE, AINDEX, ABOOL
from commonnn._types cimport ClusterParameters, CommonNNParameters, Labels, ReferenceIndices
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
    cdef void _fit(
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


cdef class FitterExtCommonNNBFSDebug(FitterExtCommonNNInterface):
    cdef:
        bint _verbose
        bint _yielding

    cdef public:
        NeighboursGetterExtInterface _neighbours_getter
        NeighboursExtInterface _neighbours
        NeighboursExtInterface _neighbour_neighbours
        SimilarityCheckerExtInterface _similarity_checker
        QueueExtInterface _queue
