import pytest

from commonnn._types import ClusterParameters, CommonNNParameters


def test_abstract_init():
    with pytest.raises(RuntimeError):
        ClusterParameters([], [])


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "radius_cutoff": 0.1,
            "similarity_cutoff": 1, "_support_cutoff": 1, "start_label": 1
        },
        pytest.param(
            {
                "radius_cutoff": 0.1,
            },
            marks=[pytest.mark.raises(exception=KeyError)]
        )
    ]
)
def test_commonnn_paramters_from_mapping(parameters, file_regression):
    cluster_params = CommonNNParameters.from_mapping(parameters)
    repr_ = f"{cluster_params!r}"
    str_ = f"{cluster_params!s}"
    file_regression.check(f"{repr_}\n{str_}")
