from asyncio import base_futures
import numpy as np
import pytest

from commonnn import recipes
from commonnn._primitive_types import P_AVALUE
from commonnn import _fit, _types


@pytest.mark.parametrize(
    "fitter_type,args",
    [
        (_fit.FitterCommonNNBFS, (None, None, None, None, None)),
        (_fit.FitterExtCommonNNBFS, (None, None, None, None, None)),
    ]
)
def test_init_fitter(fitter_type, args, file_regression):
    fitter = fitter_type(*args)
    if hasattr(fitter, "get_fit_signature"):
        signature = "\n" + fitter.get_fit_signature()
    else:
        signature = ""

    file_regression.check(f"{fitter!r}\n{fitter!s}{signature}")


@pytest.mark.parametrize(
    "fitter_type,args,expected",
    [
        (
            _fit.FitterCommonNNBFS, (None, None, None, None, None), {
                "radius_cutoff": 2.,
                "similarity_cutoff": 2,
                "_support_cutoff": 2,
                "start_label": 1
            }
        ),
        (
            _fit.FitterExtCommonNNBFS, (None, None, None, None, None), {
                "radius_cutoff": 2.,
                "similarity_cutoff": 2,
                "_support_cutoff": 2,
                "start_label": 1
            }
        ),
        (
            _fit.FitterExtCommonNNBFS, (
                _types.NeighboursGetterExtBruteForce(
                    distance_getter=_types.DistanceGetterExtMetric(
                        metric=_types.MetricExtEuclideanReduced()
                    )
                ), None, None, None, None
            ), {
                "radius_cutoff": 4.,
                "similarity_cutoff": 4,
                "_support_cutoff": 3,
                "start_label": 1
            }
        )
    ]
)
def test_make_parameters(fitter_type, args, expected):
    parameters = {
        "radius_cutoff": 2.,
        "similarity_cutoff": 2,
    }

    fitter = fitter_type(*args)
    cluster_params = fitter.make_parameters(**parameters)

    assert isinstance(cluster_params, fitter._parameter_type)
    assert expected == cluster_params.to_dict()


def test_fit(basic_components):
    data = np.array(basic_components, order="c", dtype=P_AVALUE)
    builder = recipes.Builder()
    input_data = builder.make_input_data(data)
    fitter = builder.make_component("fitter")
    assert fitter is not None
    cluster_params = fitter.make_parameters(radius_cutoff=1.5, similarity_cutoff=1)
    labels = _types.Labels.from_length(input_data.n_points)
    expected = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2])
    fitter.fit(input_data, labels, cluster_params)
    print(fitter)
    print(cluster_params)
    print(labels)
    np.testing.assert_array_equal(labels.labels, expected)
