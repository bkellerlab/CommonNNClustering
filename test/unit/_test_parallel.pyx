import numpy as np
cimport numpy as np
import pytest

from commonnn._primitive_types import P_AVALUE, P_AVALUE32, P_AINDEX, P_ABOOL
from commonnn._primitive_types cimport AVALUE, AVALUE32, AINDEX, ABOOL
from cython.parallel import prange, threadid
from cython_helper import cytest


@pytest.mark.parametrize("n_threads", [1, 2, 4])
@cytest
def test_prange_threadcount(AINDEX n_threads):
    cdef AINDEX i
    cdef AINDEX n = 1000
    cdef AINDEX[:] arr = np.zeros(n, dtype=P_AINDEX)
    cdef AINDEX[:] used_thread = np.zeros(n, dtype=P_AINDEX)
    expected = np.arange(n, dtype=P_AINDEX)

    with nogil:
        for i in prange(n, num_threads=n_threads):
            arr[i] = i
            used_thread[i] = threadid()

    np.testing.assert_array_equal(np.asarray(arr), expected)
    np.testing.assert_array_equal(np.unique(used_thread), np.arange(n_threads))
