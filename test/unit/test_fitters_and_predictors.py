import numpy as np
import pytest

from commonnn import recipes
from commonnn._primitive_types import P_AVALUE
from commonnn import _fit, _types


@pytest.mark.parametrize(
    "fitter_type,args",
    [
        (_fit.FitterCommonNNBFS, (None, None, None, None, None)),
        (_fit.FitterCommonNNBFSDebug, (None, None, None, None, None)),
        (_fit.FitterExtCommonNNBFS, (None, None, None, None, None)),
        (_fit.PredictorCommonNNFirstmatch, (None, None, None, None, None)),
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
        ),
        (
            _fit.PredictorCommonNNFirstmatch, (None, None, None, None, None), {
                "radius_cutoff": 2.,
                "similarity_cutoff": 2,
                "_support_cutoff": 2,
                "start_label": 1
            }
        ),
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


@pytest.mark.parametrize(
    "fitter",
    [
        (_fit.FitterCommonNNBFSDebug, (), {"verbose": False, "yielding": False}),
        (_fit.FitterCommonNNBFS, (), {}),
        (_fit.FitterExtCommonNNBFS, (), {}),
    ]
)
def test_fit(fitter, basic_components):
    data = np.array(basic_components, order="c", dtype=P_AVALUE)
    builder = recipes.Builder()
    builder.recipe["fitter"] = fitter
    input_data = builder.make_input_data(data)
    fitter = builder.make_component("fitter")
    assert fitter is not None
    cluster_params = fitter.make_parameters(radius_cutoff=1.5, similarity_cutoff=1)
    labels = _types.Labels.from_length(input_data.n_points)
    expected = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2])
    fitter._fit(input_data, labels, cluster_params)
    np.testing.assert_array_equal(labels.labels, expected)


@pytest.mark.parametrize(
    "fitter",
    [
        (_fit.FitterCommonNNBFSDebug, (), {"verbose": True, "yielding": True}),
    ]
)
def test_fit_debug(fitter, basic_components, file_regression, capsys):
    data = np.array(basic_components, order="c", dtype=P_AVALUE)
    builder = recipes.Builder()
    builder.recipe["fitter"] = fitter
    input_data = builder.make_input_data(data)
    fitter = builder.make_component("fitter")
    assert fitter is not None
    cluster_params = fitter.make_parameters(radius_cutoff=1.5, similarity_cutoff=1)
    labels = _types.Labels.from_length(input_data.n_points)
    yielded = "\n".join(
        str(x) for x in fitter._fit_debug(input_data, labels, cluster_params)
    )
    captured = capsys.readouterr()
    file_regression.check(yielded + "\n\n" + captured.out)
