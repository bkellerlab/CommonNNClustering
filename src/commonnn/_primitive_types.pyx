import numpy as np


P_AINDEX = np.intp
P_AVALUE = np.float64
P_AVALUE32 = np.float32
P_ABOOL = np.uint8


cdef AVALUE* _allocate_and_fill_avalue_array(AINDEX n, list values):

    cdef AVALUE *ptr
    cdef AINDEX i

    ptr = <AVALUE*>malloc(n * sizeof(AVALUE))

    if ptr == NULL:
        raise MemoryError()

    for i in range(n):
        ptr[i] = values[i]

    return ptr

cdef AINDEX* _allocate_and_fill_aindex_array(AINDEX n, list values):

    cdef AINDEX *ptr
    cdef AINDEX i

    ptr = <AINDEX*>malloc(n * sizeof(AINDEX))

    if ptr == NULL:
        raise MemoryError()

    for i in range(n):
        ptr[i] = values[i]

    return ptr
