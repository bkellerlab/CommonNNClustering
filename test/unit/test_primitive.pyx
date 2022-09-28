from commonnn._primitive_types import P_AVALUE, P_AVALUE32, P_AINDEX, P_ABOOL
from commonnn._primitive_types cimport AVALUE, AVALUE32, AINDEX, ABOOL
from commonnn._primitive_types cimport _allocate_and_fill_aindex_array, _allocate_and_fill_avalue_array
from cython_helper import cytest

from libc.stdlib cimport free

@cytest
def test_primitive_consistency():
    isinstance(P_AVALUE, float)
    isinstance(P_AVALUE32, float)
    isinstance(P_AINDEX, int)
    isinstance(P_ABOOL, bool)


@cytest
def test_allocate():
    cdef AINDEX i
    cdef AINDEX *iptr
    cdef AVALUE *vptr

    iptr = _allocate_and_fill_aindex_array(0, [])
    assert iptr != NULL
    free(iptr)

    list_ = [1, 2, 3]
    iptr = _allocate_and_fill_aindex_array(len(list_), list_)

    for i in range(len(list_)):
        assert iptr[i] == list_[i]
    free(iptr)

    vptr = _allocate_and_fill_avalue_array(0, [])
    assert vptr != NULL
    free(vptr)

    vptr = _allocate_and_fill_avalue_array(len(list_), list_)

    for i in range(len(list_)):
        assert vptr[i] == list_[i]
    free(vptr)
