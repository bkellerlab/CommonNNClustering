from commonnn._primitive_types cimport AINDEX, AVALUE, ABOOL


cdef class ClusterParameters:
    cdef public:
        AVALUE radius_cutoff
        AINDEX similarity_cutoff
        AVALUE similarity_cutoff_continuous
        AINDEX n_member_cutoff
        AINDEX current_start


cdef class Labels:
    cdef public:
        dict meta

    cdef:
        AINDEX[::1] _labels
        ABOOL[::1] _consider
        cppunordered_set[AINDEX] _consider_set