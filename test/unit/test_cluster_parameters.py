import pytest

from commonnn._types import ClusterParameters


@pytest.mark.parametrize("args", [(0.1,), (0.2, 1, 1, 5, 2)])
def test_paramters_init(args, file_regression):
    cluster_params = ClusterParameters(*args)
    repr_ = f"{cluster_params!r}"
    str_ = f"{cluster_params!s}"
    file_regression.check(f"{repr_}\n{str_}")
